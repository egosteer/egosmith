"""Batch run state: per-clip / per-stage progress tracking and status.json persistence."""

import os
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from lib.pipeline.batch.config import BatchRunConfig
from lib.pipeline.proc.stage_api import (
    PipelineVideoTask,
    get_stage_done_marker,
)
from lib.pipeline.datasets.descriptors import ClipDescriptor
from lib.pipeline.io.workspace import resolve_seq_folder


@dataclass
class VideoTaskState:
    video_path: str
    video_name: str
    run_id: str
    log_dir: Path
    descriptor: Optional[ClipDescriptor] = None
    stage_status: Dict[str, str] = field(default_factory=dict)
    retry_count: Dict[str, int] = field(default_factory=dict)
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    @classmethod
    def create(cls, video_path: str, stages, run_id: str, log_dir: Path, descriptor: Optional[ClipDescriptor] = None):
        video_name = Path(video_path).stem if descriptor is None else descriptor.clip_id
        return cls(
            video_path=video_path,
            video_name=video_name,
            run_id=run_id,
            log_dir=log_dir,
            descriptor=descriptor,
            stage_status={stage: "pending" for stage in stages},
            retry_count={stage: 0 for stage in stages},
        )

    def to_dict(self):
        return {
            "video_path": self.video_path,
            "video_name": self.video_name,
            "stage_status": self.stage_status,
            "retry_count": self.retry_count,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


def status_backup_path(status_path: Path) -> Path:
    return status_path.with_name(f"{status_path.name}.bak")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    backup_path = status_backup_path(path)
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            try:
                os.replace(path, backup_path)
            except OSError:
                pass
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _load_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _init_recovered_task(video_path: str, stages: Iterable[str] | None) -> dict[str, Any]:
    stage_list = list(stages or [])
    return {
        "video_path": video_path,
        "video_name": Path(video_path).stem,
        "stage_status": {stage: "pending" for stage in stage_list},
        "retry_count": {stage: 0 for stage in stage_list},
        "start_time": None,
        "end_time": None,
    }


def build_status_payload_from_events(
    events_path: Path,
    *,
    video_paths: Iterable[str] | None = None,
    stages: Iterable[str] | None = None,
    run_dir: str | None = None,
) -> tuple[dict, dict]:
    tasks: dict[str, dict] = {}
    stage_order = list(stages or [])
    stage_seen = set(stage_order)
    malformed_lines = 0
    parsed_lines = 0
    recovered_results = 0
    last_error = None

    for video_path in video_paths or []:
        tasks[video_path] = _init_recovered_task(video_path, stage_order)

    with events_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                malformed_lines += 1
                last_error = f"line {line_no}: {error}"
                continue
            parsed_lines += 1
            event = payload.get("event")
            if event not in {"stage_success", "stage_failure"}:
                continue
            video_path = payload.get("video")
            stage = payload.get("stage")
            if not video_path or not stage:
                continue
            if stage not in stage_seen:
                stage_order.append(stage)
                stage_seen.add(stage)
                for task in tasks.values():
                    task["stage_status"].setdefault(stage, "pending")
                    task["retry_count"].setdefault(stage, 0)
            task = tasks.setdefault(video_path, _init_recovered_task(video_path, stage_order))
            task["stage_status"][stage] = "completed" if event == "stage_success" else "failed"
            task["retry_count"].setdefault(stage, 0)
            recovered_results += 1

    recovered = {
        "run_dir": str(run_dir or events_path.parent),
        "gpus": [],
        "stages": stage_order,
        "tasks": tasks,
    }
    meta = {
        "source": "events",
        "events_path": str(events_path),
        "parsed_lines": parsed_lines,
        "malformed_lines": malformed_lines,
        "recovered_results": recovered_results,
        "last_malformed_error": last_error,
    }
    return recovered, meta


def load_status_payload_with_fallback(
    status_path: Path,
    *,
    events_path: Path | None = None,
    video_paths: Iterable[str] | None = None,
    stages: Iterable[str] | None = None,
) -> tuple[dict | None, dict]:
    backup_path = status_backup_path(status_path)
    load_errors = []

    for candidate, source in ((status_path, "status"), (backup_path, "backup")):
        if not candidate.exists():
            continue
        try:
            return _load_json_file(candidate), {"source": source, "path": str(candidate), "load_errors": load_errors}
        except (OSError, json.JSONDecodeError, ValueError) as error:
            load_errors.append(f"{source}:{candidate}: {error}")

    if events_path is not None and events_path.exists():
        payload, meta = build_status_payload_from_events(
            events_path,
            video_paths=video_paths,
            stages=stages,
            run_dir=str(status_path.parent),
        )
        meta["load_errors"] = load_errors
        return payload, meta

    if not status_path.exists() and not backup_path.exists():
        return None, {"source": "missing", "path": str(status_path), "load_errors": load_errors}

    raise RuntimeError(
        f"Failed to load batch status from {status_path}"
        + (f" (backup: {backup_path})" if backup_path != status_path else "")
        + (f". Errors: {' | '.join(load_errors)}" if load_errors else "")
    )


class BatchRunState:
    def __init__(self, config: BatchRunConfig):
        self.config = config
        self.log_dir = config.run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.status_file = config.run_dir / "status.json"
        self.events_file = config.run_dir / "events.jsonl"
        descriptor_map = config.descriptor_map
        self.tasks = {
            video_path: VideoTaskState.create(
                video_path=video_path,
                stages=config.stages,
                run_id=config.run_dir.name,
                log_dir=self.log_dir,
                descriptor=descriptor_map.get(video_path),
            )
            for video_path in config.video_paths
        }
        self._done_marker_cache: Dict[str, Dict[str, bool]] = defaultdict(dict)

    def _task_seq_folder(self, task: VideoTaskState) -> Path:
        return resolve_seq_folder(descriptor=task.descriptor, video_path=task.video_path)

    def _first_incomplete_stage(self, task: VideoTaskState) -> str:
        return next(
            (stage for stage in self.config.stages if task.stage_status.get(stage) != "completed"),
            "unknown",
        )

    def build_pipeline_task(self, video_path: str) -> PipelineVideoTask:
        task = self.tasks[video_path]
        return PipelineVideoTask.from_inputs(video_path=video_path, descriptor=task.descriptor)

    def get_seq_folder(self, video_path: str) -> Path:
        return self._task_seq_folder(self.tasks[video_path])

    def save(self):
        status_data = {
            "run_dir": str(self.config.run_dir),
            "gpus": self.config.gpus,
            "stages": self.config.stages,
            "tasks": {video_path: task.to_dict() for video_path, task in self.tasks.items()},
        }
        _atomic_write_json(self.status_file, status_data)

    def load(self):
        payload, meta = load_status_payload_with_fallback(
            self.status_file,
            events_path=self.events_file,
            video_paths=self.config.video_paths,
            stages=self.config.stages,
        )
        if payload is None:
            return
        if meta.get("source") != "status":
            extra = ""
            if meta.get("source") == "events":
                extra = (
                    f" parsed_lines={meta.get('parsed_lines', 0)}"
                    f" malformed_lines={meta.get('malformed_lines', 0)}"
                    f" recovered_results={meta.get('recovered_results', 0)}"
                )
            print(
                f"[Resume] Recovered batch status from {meta.get('source')} "
                f"for run_dir={self.config.run_dir}.{extra}",
                flush=True,
            )
            for error in meta.get("load_errors", []):
                print(f"[Resume] Status load warning: {error}", flush=True)
        for video_path, task_data in payload.get("tasks", {}).items():
            if video_path not in self.tasks:
                continue
            task = self.tasks[video_path]
            task.stage_status = task_data.get("stage_status", task.stage_status)
            task.retry_count = task_data.get("retry_count", task.retry_count)
            task.start_time = task_data.get("start_time")
            task.end_time = task_data.get("end_time")

    def prepare_for_resume(self):
        if not self.config.resume:
            return
        self.load()
        self._normalize_resume_states()
        self._print_resume_distribution()

    def _stage_done_marker_exists(self, task: VideoTaskState, stage: str) -> bool:
        cached = self._done_marker_cache[task.video_path]
        if stage not in cached:
            cached[stage] = get_stage_done_marker(self._task_seq_folder(task), stage).exists()
        return cached[stage]

    def _mark_stage_completed_from_done_marker(self, task: VideoTaskState, stage: str) -> bool:
        if not self._stage_done_marker_exists(task, stage):
            return False
        task.stage_status[stage] = "completed"
        return True

    def _normalize_resume_states(self):
        reconciled_done = defaultdict(int)
        normalized_running = defaultdict(int)

        for task in self.tasks.values():
            for stage in self.config.stages:
                current_status = task.stage_status.get(stage, "pending")

                if self._mark_stage_completed_from_done_marker(task, stage):
                    if current_status != "completed":
                        reconciled_done[stage] += 1
                elif current_status == "running":
                    normalized_running[stage] += 1
                    task.stage_status[stage] = "pending"

        if reconciled_done or normalized_running:
            print("\n[Resume Normalization]")
            for stage in self.config.stages:
                if reconciled_done[stage] > 0:
                    print(f"  {stage}: reconciled {reconciled_done[stage]} from .done markers")
                if normalized_running[stage] > 0:
                    print(f"  {stage}: normalized {normalized_running[stage]} stale 'running' -> 'pending'")
            print()

    def _print_resume_distribution(self):
        print("\n[Resume Status Distribution]")
        for stage in self.config.stages:
            status_counts = defaultdict(int)
            for task in self.tasks.values():
                status = task.stage_status.get(stage, "pending")
                status_counts[status] += 1
            print(f"  {stage}: " + ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items())))
        print()

    def mark_batch_started(self):
        timestamp = datetime.now(timezone.utc).isoformat()
        for task in self.tasks.values():
            if task.start_time is None:
                task.start_time = timestamp

    def mark_stage_running(self, video_path: str, stage: str):
        self.tasks[video_path].stage_status[stage] = "running"

    def mark_stage_result(self, video_path: str, stage: str, success: bool):
        self.tasks[video_path].stage_status[stage] = "completed" if success else "failed"
        if success:
            self._done_marker_cache[video_path][stage] = True

    def record_retry(self, video_path: str, stage: str, attempt: int):
        self.tasks[video_path].retry_count[stage] = attempt

    def finalize_videos(self):
        success_count = 0
        fail_count = 0
        completed = []
        failed = []

        for video_path, task in self.tasks.items():
            if all(task.stage_status.get(stage) == "completed" for stage in self.config.stages):
                success_count += 1
                if task.end_time is None:
                    task.end_time = datetime.now(timezone.utc).isoformat()
                completed.append(video_path)
            else:
                fail_count += 1
                failed.append((video_path, self._first_incomplete_stage(task)))

        return success_count, fail_count, completed, failed

    def get_stage_pending_videos(self, stage: str):
        stage_idx = self.config.stages.index(stage)
        prev_stage = self.config.stages[stage_idx - 1] if stage_idx > 0 else None
        excluded_completed = 0
        excluded_running = 0
        excluded_other = 0
        excluded_prev_stage = 0
        excluded_done_marker = 0

        candidates = []
        for video_path in self.config.video_paths:
            task = self.tasks[video_path]

            if prev_stage is not None:
                prev_status = task.stage_status.get(prev_stage, "pending")
                if self.config.resume and prev_status != "completed":
                    self._mark_stage_completed_from_done_marker(task, prev_stage)
                    prev_status = task.stage_status.get(prev_stage, "pending")
                if prev_status != "completed":
                    excluded_prev_stage += 1
                    continue

            current_status = task.stage_status.get(stage, "pending")
            if self.config.resume and current_status != "completed":
                if self._mark_stage_completed_from_done_marker(task, stage):
                    excluded_done_marker += 1
                    continue
                current_status = task.stage_status.get(stage, "pending")

            if current_status not in ("pending", "failed"):
                if current_status == "completed":
                    excluded_completed += 1
                elif current_status == "running":
                    excluded_running += 1
                else:
                    excluded_other += 1
                continue

            candidates.append(video_path)

        if not self.config.resume:
            self._print_stage_eligibility(
                stage,
                scheduled=len(candidates),
                excluded_completed=excluded_completed,
                excluded_running=excluded_running,
                excluded_prev_stage=excluded_prev_stage,
                excluded_done_marker=0,
                excluded_other=excluded_other,
            )
            return candidates

        self._print_stage_eligibility(
            stage,
            scheduled=len(candidates),
            excluded_completed=excluded_completed,
            excluded_running=excluded_running,
            excluded_prev_stage=excluded_prev_stage,
            excluded_done_marker=excluded_done_marker,
            excluded_other=excluded_other,
        )
        return candidates

    def _print_stage_eligibility(
        self,
        stage: str,
        *,
        scheduled: int,
        excluded_completed: int,
        excluded_running: int,
        excluded_prev_stage: int,
        excluded_done_marker: int,
        excluded_other: int,
    ):
        total = len(self.config.video_paths)
        print(
            f"  [{stage}] Eligibility: total={total} scheduled={scheduled} "
            f"excluded_completed={excluded_completed} excluded_running={excluded_running} "
            f"excluded_prev_stage={excluded_prev_stage} excluded_done={excluded_done_marker} "
            f"excluded_other={excluded_other}"
        )
