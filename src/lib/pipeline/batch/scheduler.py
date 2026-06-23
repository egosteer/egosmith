"""Batch scheduler: drives clips through the selected stages across the worker pool."""

import time
from typing import Dict

from tqdm import tqdm

from lib.pipeline.batch.config import BatchRunConfig
from lib.pipeline.batch.events import BatchEventLogger
from lib.pipeline.batch.state import BatchRunState
from lib.pipeline.batch.worker_pool import StageWorkerPool


class BatchScheduler:
    def __init__(self, config: BatchRunConfig):
        self.config = config
        self.state = BatchRunState(config)
        self.events = BatchEventLogger(config.run_dir / "events.jsonl")
        self.worker_pool = StageWorkerPool(config)

    @staticmethod
    def _summarize_hotspot(metrics: Dict[str, object] | None) -> str:
        timing = {}
        if isinstance(metrics, dict):
            maybe_timing = metrics.get("timing")
            if isinstance(maybe_timing, dict):
                timing = maybe_timing
        best_key = None
        best_value = 0.0
        for key, value in timing.items():
            if key == "total" or key.endswith(("_frames", "_chunks", "_batches", "_count", "_hit")):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if numeric > best_value:
                best_key = key
                best_value = numeric
        if best_key is None:
            return "-"
        return f"{best_key}:{best_value:.1f}s"

    @staticmethod
    def _update_progress_bar(pbar, stage_results: Dict[str, bool], perf_state: Dict[str, object]):
        if pbar is None:
            return
        pbar.update(1)
        processed = int(perf_state.get("processed", 0))
        total_wall = float(perf_state.get("total_wall_sec", 0.0))
        avg_wall = total_wall / processed if processed > 0 else 0.0
        pbar.set_postfix(
            {
                "success": sum(1 for ok in stage_results.values() if ok),
                "failed": sum(1 for ok in stage_results.values() if not ok),
                "avg_s": f"{avg_wall:.2f}",
                "last_s": f"{float(perf_state.get('last_wall_sec', 0.0)):.2f}",
                "hot": str(perf_state.get("last_hotspot", "-")),
            }
        )

    def _apply_stage_result(self, stage: str, result: Dict[str, object], stage_results: Dict[str, bool], perf_state: Dict[str, object], pbar=None):
        video_path = result["video"]
        success = result["success"]
        gpu = result.get("gpu")
        wall_sec = float(result.get("wall_sec", 0.0) or 0.0)
        metrics = result.get("metrics")

        self.state.mark_stage_result(video_path, stage, success)
        event_name = "stage_success" if success else "stage_failure"
        event_payload = {"video": video_path, "stage": stage, "gpu": gpu, "wall_sec": wall_sec}
        if metrics is not None:
            event_payload["metrics"] = metrics
        if not success and result.get("error"):
            event_payload["error"] = result["error"]
        self.events.emit(event_name, **event_payload)

        stage_results[video_path] = success
        perf_state["processed"] = int(perf_state.get("processed", 0)) + 1
        perf_state["total_wall_sec"] = float(perf_state.get("total_wall_sec", 0.0)) + wall_sec
        perf_state["last_wall_sec"] = wall_sec
        perf_state["last_hotspot"] = self._summarize_hotspot(metrics)
        self._update_progress_bar(pbar, stage_results, perf_state)

    def _run_stage_with_retries(self, stage: str, pbar=None, initial_pending_videos=None):
        final_results = {}
        for attempt in range(self.config.max_stage_retries + 1):
            if attempt == 0 and initial_pending_videos is not None:
                pending_videos = initial_pending_videos
            else:
                pending_videos = self.state.get_stage_pending_videos(stage)
            if not pending_videos:
                break

            worker_count_per_gpu = self.config.worker_count_for_stage(stage)
            total_workers = len(self.config.gpus) * worker_count_per_gpu
            print(
                f"  [{stage}] Worker slots: {worker_count_per_gpu}/GPU, total={total_workers}"
            )

            for video_path in pending_videos:
                self.state.record_retry(video_path, stage, attempt)
                self.state.mark_stage_running(video_path, stage)
            self.state.save()

            self.events.emit("wave_start", stage=stage, total=len(pending_videos), attempt=attempt)
            stage_results = {}
            completed_since_save = 0
            perf_state = {
                "processed": 0,
                "total_wall_sec": 0.0,
                "last_wall_sec": 0.0,
                "last_hotspot": "-",
            }

            def on_result(result):
                nonlocal completed_since_save
                self._apply_stage_result(stage, result, stage_results, perf_state, pbar=pbar)
                completed_since_save += 1
                if completed_since_save >= 10:
                    self.state.save()
                    completed_since_save = 0

            current_results = self.worker_pool.run_stage(stage, pending_videos, on_result=on_result)
            if completed_since_save > 0:
                self.state.save()

            for video_path, success in current_results.items():
                final_results[video_path] = success

            failed = [video_path for video_path, success in current_results.items() if not success]
            self.events.emit(
                "wave_end",
                stage=stage,
                attempt=attempt,
                success=sum(1 for success in current_results.values() if success),
                failed=len(failed),
            )
            if not failed:
                break
            if attempt < self.config.max_stage_retries:
                self.events.emit("wave_retry", stage=stage, attempt=attempt + 1, failed=len(failed))

        return final_results

    def run(self):
        resume_prep_start = time.monotonic()
        self.state.prepare_for_resume()
        resume_prep_elapsed = time.monotonic() - resume_prep_start
        self.state.mark_batch_started()

        self.events.emit("batch_start", total_videos=len(self.config.video_paths), gpus=self.config.gpus, mode="wave")

        print(f"\n{'=' * 60}")
        print(f"Stage-Wave Scheduling: {len(self.config.video_paths)} videos, {len(self.config.gpus)} GPUs")
        if self.config.resume:
            print(f"Resume preparation: {resume_prep_elapsed:.2f}s")
        print(f"{'=' * 60}\n")

        for stage_idx, stage in enumerate(self.config.stages, 1):
            pending_videos = self.state.get_stage_pending_videos(stage)
            total_for_stage = len(pending_videos)

            print(f"Stage {stage_idx}/{len(self.config.stages)}: {stage} ({total_for_stage} videos)")
            with tqdm(total=total_for_stage, desc=f"  {stage}", unit="video", leave=True) as pbar:
                self._run_stage_with_retries(stage, pbar=pbar, initial_pending_videos=pending_videos)

        success_count, fail_count, completed_videos, failed_videos = self.state.finalize_videos()
        for video_path in completed_videos:
            self.events.emit("video_completed", video=video_path)
        for video_path, failed_stage in failed_videos:
            self.events.emit("video_failed", video=video_path, stage=failed_stage)

        self.state.save()
        self.events.emit("batch_end", total=len(self.config.video_paths), success=success_count, failed=fail_count)

        print(f"\n{'=' * 60}")
        print("Batch Inference Complete")
        print(f"{'=' * 60}")
        print(f"Total videos: {len(self.config.video_paths)}")
        print(f"Success: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Run directory: {self.config.run_dir}")
        print(f"Status file: {self.state.status_file}")
        print(f"Events log: {self.events.events_file}")
        print(f"{'=' * 60}\n")

        return fail_count == 0
