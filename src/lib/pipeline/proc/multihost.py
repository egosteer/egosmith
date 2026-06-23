from __future__ import annotations

import json
import shlex
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from lib.pipeline.clips.clip_manifest import load_clip_manifest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_cli_list(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item).strip())
    raw = str(value).strip()
    return (raw,) if raw else ()


def _normalize_env_map(value) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized = {}
    for key, item in value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        normalized[key_text] = str(item)
    return normalized


def _parse_gpu_ids(gpus: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in str(gpus).split(",") if part.strip())
    if not parsed:
        raise ValueError("Host `gpus` must contain at least one GPU id")
    return parsed


def _mapping_without_keys(mapping: dict | None, excluded: set[str]) -> dict:
    return {key: value for key, value in (mapping or {}).items() if key not in excluded}


@dataclass(frozen=True)
class MultihostWorkerHost:
    name: str
    ssh_target: str
    gpus: str
    project_root: str
    hawor_python: str
    slam_python: str
    ssh_options: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    @property
    def gpu_ids(self) -> tuple[int, ...]:
        return _parse_gpu_ids(self.gpus)

    def python_for_stage(self, runtime_key: str) -> str:
        if runtime_key == "slam":
            return self.slam_python
        return self.hawor_python


@dataclass(frozen=True)
class MultihostConfig:
    enabled: bool
    mode: str = "stage_queue"
    shard_factor: int = 2
    hosts: tuple[MultihostWorkerHost, ...] = ()
    ssh_options: tuple[str, ...] = ()

    def to_summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "shard_factor": self.shard_factor,
            "ssh_options": list(self.ssh_options),
            "hosts": [
                {
                    "name": host.name,
                    "ssh_target": host.ssh_target,
                    "gpus": host.gpus,
                    "project_root": host.project_root,
                    "hawor_python": host.hawor_python,
                    "slam_python": host.slam_python,
                    "ssh_options": list(host.ssh_options),
                    "env": dict(host.env),
                }
                for host in self.hosts
            ],
        }


def parse_multihost_config(
    raw_config: dict | None,
    *,
    default_project_root: str | Path,
    default_hawor_python: str,
    default_slam_python: str,
) -> MultihostConfig:
    raw = dict(raw_config or {})
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return MultihostConfig(enabled=False)

    mode = str(raw.get("mode") or "stage_queue").strip() or "stage_queue"
    if mode != "stage_queue":
        raise ValueError(f"Unsupported infer.multihost.mode: {mode!r}. Expected 'stage_queue'.")

    shard_factor = int(raw.get("shard_factor", 2))
    if shard_factor < 1:
        raise ValueError("infer.multihost.shard_factor must be >= 1")

    global_ssh_options = _normalize_cli_list(raw.get("ssh_options"))
    global_env = _normalize_env_map(raw.get("env"))
    project_root = str(raw.get("project_root") or default_project_root)
    hawor_python = str(raw.get("hawor_python") or default_hawor_python)
    slam_python = str(raw.get("slam_python") or default_slam_python)

    host_entries = raw.get("hosts")
    if not isinstance(host_entries, list) or not host_entries:
        raise ValueError("infer.multihost.hosts must be a non-empty list when multihost is enabled")

    hosts = []
    seen_names = set()
    for index, item in enumerate(host_entries):
        if not isinstance(item, dict):
            raise ValueError(f"infer.multihost.hosts[{index}] must be a mapping")
        name = str(item.get("name") or f"host-{index + 1:02d}").strip()
        if not name:
            raise ValueError(f"infer.multihost.hosts[{index}] has an empty name")
        if name in seen_names:
            raise ValueError(f"Duplicate infer.multihost host name: {name}")
        seen_names.add(name)

        host_gpus = str(item.get("gpus", "")).strip()
        _parse_gpu_ids(host_gpus)

        host_env = dict(global_env)
        host_env.update(_normalize_env_map(item.get("env")))
        hosts.append(
            MultihostWorkerHost(
                name=name,
                ssh_target=str(item.get("ssh_target") or name),
                gpus=host_gpus,
                project_root=str(item.get("project_root") or project_root),
                hawor_python=str(item.get("hawor_python") or hawor_python),
                slam_python=str(item.get("slam_python") or slam_python),
                ssh_options=tuple(global_ssh_options) + _normalize_cli_list(item.get("ssh_options")),
                env=host_env,
            )
        )

    return MultihostConfig(
        enabled=True,
        mode=mode,
        shard_factor=shard_factor,
        hosts=tuple(hosts),
        ssh_options=global_ssh_options,
    )


@dataclass(frozen=True)
class MultihostStageSpec:
    pipeline_stage: str
    batch_stages: str
    runtime_key: str
    extra_args: tuple[str, ...]
    worker_count_per_gpu: int


@dataclass(frozen=True)
class StageShard:
    shard_id: int
    start: int
    end: int
    weight: int


class MultihostStageQueueRunner:
    def __init__(
        self,
        *,
        config: MultihostConfig,
        manifest_path: str | Path,
        run_dir: str | Path,
        infer_resume: bool,
    ):
        self.config = config
        self.manifest_path = Path(manifest_path).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.infer_resume = bool(infer_resume)
        self.records = load_clip_manifest(self.manifest_path)
        if not self.records:
            raise ValueError(f"Multihost infer manifest is empty: {self.manifest_path}")

    def _stage_root(self, stage_name: str) -> Path:
        return self.run_dir / "multihost" / stage_name

    def _dispatch_path(self, stage_name: str) -> Path:
        return self._stage_root(stage_name) / "dispatch.json"

    def _weights(self) -> list[int]:
        return [max(1, int(record.descriptor.frame_count)) for record in self.records]

    def _plan_stage_shards(self, spec: MultihostStageSpec) -> list[StageShard]:
        total_records = len(self.records)
        total_slots = sum(len(host.gpu_ids) * spec.worker_count_per_gpu for host in self.config.hosts)
        desired_shards = min(total_records, max(len(self.config.hosts), total_slots * self.config.shard_factor))
        if desired_shards <= 1:
            return [StageShard(shard_id=0, start=0, end=total_records, weight=sum(self._weights()))]

        weights = self._weights()
        total_weight = sum(weights)
        target_weight = max(1.0, float(total_weight) / float(desired_shards))

        shards = []
        shard_start = 0
        shard_weight = 0
        for index, weight in enumerate(weights):
            shard_weight += weight
            records_remaining = total_records - (index + 1)
            shards_remaining = desired_shards - len(shards) - 1
            should_close = records_remaining >= shards_remaining and (
                shard_weight >= target_weight or records_remaining == shards_remaining
            )
            if not should_close:
                continue
            shards.append(StageShard(shard_id=len(shards), start=shard_start, end=index + 1, weight=shard_weight))
            shard_start = index + 1
            shard_weight = 0

        if shard_start < total_records:
            shards.append(
                StageShard(
                    shard_id=len(shards),
                    start=shard_start,
                    end=total_records,
                    weight=sum(weights[shard_start:total_records]),
                )
            )
        return shards

    def _initial_dispatch_state(self, spec: MultihostStageSpec, shards: Iterable[StageShard]) -> dict:
        return {
            "version": 1,
            "mode": self.config.mode,
            "stage": spec.pipeline_stage,
            "batch_stages": spec.batch_stages,
            "manifest_path": str(self.manifest_path),
            "infer_resume": self.infer_resume,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "shards": [
                {
                    "shard_id": shard.shard_id,
                    "start": shard.start,
                    "end": shard.end,
                    "weight": shard.weight,
                    "status": "pending",
                    "attempts": 0,
                    "host": None,
                    "exit_code": None,
                    "run_dir": str((self._stage_root(spec.pipeline_stage) / "shards" / f"shard-{shard.shard_id:04d}").resolve()),
                    "log_path": None,
                    "remote_command": None,
                    "started_at": None,
                    "ended_at": None,
                }
                for shard in shards
            ],
        }

    def _load_dispatch_state(self, spec: MultihostStageSpec, shards: list[StageShard]) -> dict:
        dispatch_path = self._dispatch_path(spec.pipeline_stage)
        stage_root = self._stage_root(spec.pipeline_stage)
        stage_root.mkdir(parents=True, exist_ok=True)
        (stage_root / "logs").mkdir(parents=True, exist_ok=True)
        (stage_root / "shards").mkdir(parents=True, exist_ok=True)

        expected = self._initial_dispatch_state(spec, shards)
        if not dispatch_path.exists():
            self._write_dispatch_state(dispatch_path, expected)
            return expected
        if not self.infer_resume:
            self._write_dispatch_state(dispatch_path, expected)
            return expected

        with dispatch_path.open("r", encoding="utf-8") as handle:
            current = json.load(handle)

        current_shards = current.get("shards") or []
        expected_shards = expected["shards"]
        if len(current_shards) != len(expected_shards):
            raise ValueError(
                f"Existing multihost dispatch layout for {spec.pipeline_stage} does not match current manifest/shard plan: "
                f"{dispatch_path}"
            )

        for current_entry, expected_entry in zip(current_shards, expected_shards):
            if (
                current_entry.get("shard_id") != expected_entry["shard_id"]
                or current_entry.get("start") != expected_entry["start"]
                or current_entry.get("end") != expected_entry["end"]
            ):
                raise ValueError(
                    f"Existing multihost dispatch layout for {spec.pipeline_stage} does not match current manifest/shard plan: "
                    f"{dispatch_path}"
                )
            if current_entry.get("status") == "running":
                current_entry["status"] = "pending"
        current["updated_at"] = _utc_now()
        self._write_dispatch_state(dispatch_path, current)
        return current

    @staticmethod
    def _write_dispatch_state(path: Path, payload: dict) -> None:
        payload["updated_at"] = _utc_now()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _build_remote_command(
        self,
        *,
        host: MultihostWorkerHost,
        spec: MultihostStageSpec,
        shard_entry: dict,
    ) -> tuple[list[str], str]:
        shard_run_dir = shard_entry["run_dir"]
        remote_batch_cmd = [
            host.python_for_stage(spec.runtime_key),
            str(Path(host.project_root) / "scripts" / "batch_infer.py"),
            "--descriptor_manifest",
            str(self.manifest_path),
            "--run_dir",
            shard_run_dir,
            "--stages",
            spec.batch_stages,
            "--start",
            str(shard_entry["start"]),
            "--end",
            str(shard_entry["end"]),
            "--gpus",
            host.gpus,
            *spec.extra_args,
        ]
        env_prefix = ""
        if host.env:
            env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(host.env.items())) + " "
        remote_command = (
            "set -euo pipefail; "
            f"cd {shlex.quote(host.project_root)}; "
            f"{env_prefix}{shlex.join(remote_batch_cmd)}"
        )
        local_cmd = [
            "ssh",
            *host.ssh_options,
            host.ssh_target,
            "/bin/bash",
            "-lc",
            remote_command,
        ]
        return local_cmd, remote_command

    def _run_single_shard(
        self,
        *,
        host: MultihostWorkerHost,
        spec: MultihostStageSpec,
        shard_entry: dict,
        dispatch_state: dict,
        dispatch_path: Path,
        dispatch_lock: threading.Lock,
    ) -> bool:
        log_dir = self._stage_root(spec.pipeline_stage) / "logs"
        attempt = int(shard_entry.get("attempts", 0)) + 1
        log_path = log_dir / f"{host.name}.shard-{int(shard_entry['shard_id']):04d}.attempt-{attempt:02d}.log"
        local_cmd, remote_command = self._build_remote_command(host=host, spec=spec, shard_entry=shard_entry)

        with dispatch_lock:
            shard_entry["attempts"] = attempt
            shard_entry["status"] = "running"
            shard_entry["host"] = host.name
            shard_entry["log_path"] = str(log_path.resolve())
            shard_entry["remote_command"] = remote_command
            shard_entry["started_at"] = _utc_now()
            shard_entry["ended_at"] = None
            shard_entry["exit_code"] = None
            self._write_dispatch_state(dispatch_path, dispatch_state)

        print(
            f"[{spec.pipeline_stage}] start shard-{int(shard_entry['shard_id']):04d} "
            f"{shard_entry['start']}:{shard_entry['end']} on {host.name}"
        )
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("$ " + shlex.join(local_cmd) + "\n\n")
            handle.flush()
            process = subprocess.Popen(
                local_cmd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            exit_code = process.wait()

        success = exit_code == 0
        with dispatch_lock:
            shard_entry["status"] = "success" if success else "failed"
            shard_entry["exit_code"] = exit_code
            shard_entry["ended_at"] = _utc_now()
            self._write_dispatch_state(dispatch_path, dispatch_state)

        outcome = "ok" if success else f"failed(exit={exit_code})"
        print(
            f"[{spec.pipeline_stage}] finish shard-{int(shard_entry['shard_id']):04d} "
            f"on {host.name}: {outcome}"
        )
        return success

    def run_stage(self, spec: MultihostStageSpec) -> dict:
        shards = self._plan_stage_shards(spec)
        dispatch_path = self._dispatch_path(spec.pipeline_stage)
        dispatch_state = self._load_dispatch_state(spec, shards)
        dispatch_lock = threading.Lock()
        stop_event = threading.Event()

        pending_ids = deque(
            entry["shard_id"]
            for entry in dispatch_state["shards"]
            if entry.get("status") != "success"
        )

        print(
            f"[{spec.pipeline_stage}] multihost stage-queue: "
            f"{len(pending_ids)}/{len(dispatch_state['shards'])} shard(s) pending across {len(self.config.hosts)} host(s)"
        )
        if not pending_ids:
            return {
                "dispatch_path": str(dispatch_path.resolve()),
                "success": True,
                "failed_shards": [],
                "completed_shards": len(dispatch_state["shards"]),
            }

        shard_lookup = {entry["shard_id"]: entry for entry in dispatch_state["shards"]}

        def host_loop(host: MultihostWorkerHost):
            while True:
                with dispatch_lock:
                    if stop_event.is_set() or not pending_ids:
                        return
                    shard_id = pending_ids.popleft()
                    shard_entry = shard_lookup[shard_id]

                if stop_event.is_set():
                    with dispatch_lock:
                        pending_ids.appendleft(shard_id)
                    return

                if not self._run_single_shard(
                    host=host,
                    spec=spec,
                    shard_entry=shard_entry,
                    dispatch_state=dispatch_state,
                    dispatch_path=dispatch_path,
                    dispatch_lock=dispatch_lock,
                ):
                    stop_event.set()
                    return

        threads = [
            threading.Thread(target=host_loop, name=f"multihost-{spec.pipeline_stage}-{host.name}", args=(host,))
            for host in self.config.hosts
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        failed_shards = [
            {
                "shard_id": entry["shard_id"],
                "host": entry.get("host"),
                "exit_code": entry.get("exit_code"),
                "log_path": entry.get("log_path"),
            }
            for entry in dispatch_state["shards"]
            if entry.get("status") == "failed"
        ]
        success = not failed_shards and all(entry.get("status") == "success" for entry in dispatch_state["shards"])
        if not success:
            self._write_dispatch_state(dispatch_path, dispatch_state)

        return {
            "dispatch_path": str(dispatch_path.resolve()),
            "success": success,
            "failed_shards": failed_shards,
            "completed_shards": sum(1 for entry in dispatch_state["shards"] if entry.get("status") == "success"),
            "total_shards": len(dispatch_state["shards"]),
        }


def sanitize_infer_args_for_multihost(
    infer_common_cfg: dict | None,
    *,
    reserved_keys: set[str] | None = None,
) -> dict:
    reserved = {"gpus"}
    if reserved_keys:
        reserved.update(reserved_keys)
    return _mapping_without_keys(infer_common_cfg, reserved)
