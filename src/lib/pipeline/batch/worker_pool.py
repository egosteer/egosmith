"""Multiprocessing stage worker pool: spawns per-GPU stage workers and manages their lifecycle."""

import contextlib
import multiprocessing as mp
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty
from typing import Dict, List, Optional

import joblib
import numpy as np

from lib.pipeline.batch.config import BatchRunConfig
from lib.pipeline.proc.errors import CorruptStageDataError
from lib.pipeline.io.frame_source import build_frame_source
from lib.pipeline.proc.logging_setup import get_logger
from lib.pipeline.proc.runtime import WorkerRuntime, set_determinism
from lib.pipeline.proc.stage_api import (
    PipelineVideoTask,
    get_track_range,
    get_tracks_dir,
    is_stage_complete,
    run_pipeline_stage,
)

_logger = get_logger("batch.worker_pool")


def _build_runtime(config: BatchRunConfig, gpu: int) -> WorkerRuntime:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    set_determinism(42)
    return WorkerRuntime(
        gpu=str(gpu),
        checkpoint=config.checkpoint,
        infiller_weight=config.infiller_weight,
        img_focal=config.img_focal,
        chunk_batch_size=config.chunk_batch_size,
        num_workers=config.num_workers,
        render_batch_size=config.render_batch_size,
        any4d_batch_size=config.any4d_batch_size,
        detect_batch_size=config.detect_batch_size,
        detect_io_workers=config.detect_io_workers,
        detect_device=config.detect_device,
        detect_half_precision=config.detect_half_precision,
        infiller_window_batch_size=config.infiller_window_batch_size,
        rebuild_cam_space_cache=config.rebuild_cam_space_cache,
        depth_predict_all_frames=config.depth_predict_all_frames,
        any4d_repo_root=config.any4d_repo_root,
        any4d_checkpoint_path=config.any4d_checkpoint_path,
        any4d_resolution_set=config.any4d_resolution_set,
        any4d_use_amp=config.any4d_use_amp,
        stage3_tmp_root=config.stage3_tmp_root,
        keep_intermediates=getattr(config, "keep_intermediates", "all"),
    )


def _build_pipeline_task(video_path: str, descriptor_map) -> PipelineVideoTask:
    return PipelineVideoTask.from_inputs(video_path=video_path, descriptor=descriptor_map.get(video_path))


def _configure_worker_env(config: BatchRunConfig) -> None:
    env_values = config.worker_env_overrides()
    for key, value in env_values.items():
        if value is None or value == "":
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _worker_log_path(config: BatchRunConfig, stage: str, gpu: int, worker_slot: int) -> Path:
    log_dir = config.run_dir / "logs" / "workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{stage}_gpu{gpu}_slot{worker_slot}.log"


def _prefetch_video_data(video_path: str, stage: str, descriptor_map, config: BatchRunConfig):
    if stage not in {"motion", "slam", "infiller"}:
        return None

    try:
        pipeline_task = _build_pipeline_task(video_path, descriptor_map)
        seq_folder = pipeline_task.seq_folder

        if config.resume and is_stage_complete(stage, seq_folder, fast_check=True):
            return None

        if stage == "slam":
            frame_source = pipeline_task.build_frame_source() or build_frame_source(video_path)
            return {
                "frame_source": frame_source,
            }

        if stage == "infiller":
            start_idx, end_idx = get_track_range(seq_folder, fast=True)
            tracks_dir = get_tracks_dir(seq_folder, start_idx, end_idx)
            prefetched = {
                "frame_chunks_all": joblib.load(tracks_dir / "frame_chunks_all.npy"),
            }
            cache_path = Path(seq_folder) / "cam_space_cache.joblib"
            if cache_path.exists() and not config.rebuild_cam_space_cache:
                prefetched["cam_space_cache"] = joblib.load(cache_path)
            return prefetched

        start_idx, end_idx = get_track_range(seq_folder)
        tracks_dir = get_tracks_dir(seq_folder, start_idx, end_idx)

        frame_chunks_file = tracks_dir / "frame_chunks_all.npy"
        model_masks_file = tracks_dir / "model_masks.npy"
        if config.resume and frame_chunks_file.exists() and model_masks_file.exists():
            return None

        frame_source = pipeline_task.build_frame_source() or build_frame_source(video_path)
        tracks = np.load(tracks_dir / "model_tracks.npy", allow_pickle=True).item()

        return {
            "frame_source": frame_source,
            "tracks": tracks,
        }
    except Exception:
        # Prefetch is a best-effort optimization; the stage re-loads on its own.
        # Log so a recurring prefetch failure (corrupt cache, disk error) is visible
        # instead of silently degrading throughput.
        _logger.warning("Prefetch failed for %s (stage=%s); stage will load inline.", video_path, stage, exc_info=True)
        return None


def _run_single_video(video_path: str, stage: str, runtime: WorkerRuntime, descriptor_map, config: BatchRunConfig, prefetched_data=None):
    pipeline_task = _build_pipeline_task(video_path, descriptor_map)
    return run_pipeline_stage(
        stage,
        pipeline_task,
        runtime.stage_config,
        runtime=runtime,
        prefetched_data=prefetched_data,
        resume=config.resume,
        force=not config.resume,
    )


def _stage_worker_main(gpu: int, worker_slot: int, stage: str, video_queue: mp.Queue, result_queue: mp.Queue, descriptor_map, config: BatchRunConfig):
    log_path = _worker_log_path(config, stage, gpu, worker_slot)
    os.environ["HAWOR_QUIET"] = "1"
    _configure_worker_env(config)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            print(
                f"[worker] stage={stage} gpu={gpu} slot={worker_slot} pid={os.getpid()}",
                flush=True,
            )
            try:
                runtime = _build_runtime(config, gpu)
                runtime.ensure_runner(stage)
            except Exception:
                traceback.print_exc()
                return

            with ThreadPoolExecutor(max_workers=1) as prefetcher:
                prefetch_future = None
                video_path = video_queue.get()

                while video_path is not None:
                    prefetched_data = None
                    if prefetch_future is not None:
                        try:
                            prefetched_data = prefetch_future.result()
                        except Exception:
                            prefetched_data = None

                    next_video = video_queue.get()

                    next_prefetch_future = None
                    if next_video is not None:
                        next_prefetch_future = prefetcher.submit(
                            _prefetch_video_data,
                            next_video,
                            stage,
                            descriptor_map,
                            config,
                        )

                    try:
                        stage_result = _run_single_video(
                            video_path,
                            stage,
                            runtime,
                            descriptor_map,
                            config,
                            prefetched_data=prefetched_data,
                        )
                        result_queue.put(
                            {
                                "video": video_path,
                                "success": stage_result.get("status") in ("success", "skipped"),
                                "status": stage_result.get("status"),
                                "gpu": gpu,
                                "wall_sec": float(stage_result.get("wall_sec", 0.0)),
                                "metrics": stage_result.get("metrics"),
                            }
                        )
                    except CorruptStageDataError as error:
                        result_queue.put(
                            {
                                "video": video_path,
                                "success": False,
                                "gpu": gpu,
                                "error": str(error),
                            }
                        )
                    except Exception as error:
                        traceback.print_exc()
                        result_queue.put(
                            {
                                "video": video_path,
                                "success": False,
                                "gpu": gpu,
                                "error": str(error),
                            }
                        )

                    video_path = next_video
                    prefetch_future = next_prefetch_future


class StageWorkerPool:
    def __init__(self, config: BatchRunConfig):
        self.config = config
        self.descriptor_map = config.descriptor_map

    def _estimate_video_work(self, video_path: str, stage: str) -> int:
        descriptor = self.descriptor_map.get(video_path)
        if descriptor is not None:
            return int(descriptor.frame_count)

        if stage != "detect_track":
            try:
                seq_folder = _build_pipeline_task(video_path, self.descriptor_map).seq_folder
                start_idx, end_idx = get_track_range(seq_folder, fast=True)
                return end_idx - start_idx
            except Exception:
                # Fall back to file size for work estimation; log so a broken track
                # range (which skews scheduling priority) is observable.
                _logger.debug("Work estimate via track range failed for %s (stage=%s); using file size.", video_path, stage)

        try:
            return os.path.getsize(video_path)
        except OSError as error:
            _logger.warning("Could not estimate work for %s: %s", video_path, error)
            return 0

    def _prioritize_descriptor_locality_videos(self, video_paths: List[str], stage: str) -> List[str]:
        shard_groups = defaultdict(list)
        shard_work = {}

        for video_path in video_paths:
            descriptor = self.descriptor_map.get(video_path)
            if descriptor is not None:
                shard_key = descriptor.shard_path or descriptor.frame_dir or video_path
            else:
                shard_key = video_path
            estimated_work = self._estimate_video_work(video_path, stage)
            shard_groups[shard_key].append((video_path, estimated_work))
            shard_work[shard_key] = shard_work.get(shard_key, 0) + estimated_work

        prioritized = []
        ordered_shards = sorted(
            shard_groups,
            key=lambda shard_key: (
                -shard_work[shard_key],
                shard_key,
            ),
        )
        for shard_key in ordered_shards:
            shard_videos = sorted(
                shard_groups[shard_key],
                key=lambda item: (-item[1], item[0]),
            )
            prioritized.extend(video_path for video_path, _ in shard_videos)
        return prioritized

    def _prioritize_videos(self, video_paths: List[str], stage: str) -> List[str]:
        if stage in {"detect_track", "motion", "slam"} and self.descriptor_map:
            return self._prioritize_descriptor_locality_videos(video_paths, stage)
        return sorted(video_paths, key=lambda video_path: self._estimate_video_work(video_path, stage), reverse=True)

    @staticmethod
    def _stop_workers(workers: List[mp.Process], *, force: bool):
        for worker in workers:
            if not worker.is_alive():
                continue
            if force:
                worker.kill()
            else:
                worker.terminate()

    @staticmethod
    def _join_workers(workers: List[mp.Process], timeout_sec: float = 5.0):
        for worker in workers:
            worker.join(timeout=timeout_sec)
        lingering = [worker for worker in workers if worker.is_alive()]
        if lingering:
            StageWorkerPool._stop_workers(lingering, force=False)
            for worker in lingering:
                worker.join(timeout=timeout_sec)
        stubborn = [worker for worker in workers if worker.is_alive()]
        if stubborn:
            StageWorkerPool._stop_workers(stubborn, force=True)
            for worker in stubborn:
                worker.join(timeout=timeout_sec)

    def run_stage(self, stage: str, video_paths: List[str], on_result) -> Dict[str, bool]:
        if not video_paths:
            return {}

        prioritized_videos = self._prioritize_videos(video_paths, stage)
        video_queue = mp.Queue()
        result_queue = mp.Queue()

        for video_path in prioritized_videos:
            video_queue.put(video_path)
        worker_count_per_gpu = self.config.worker_count_for_stage(stage)
        total_workers = len(self.config.gpus) * worker_count_per_gpu
        for _ in range(total_workers):
            video_queue.put(None)

        workers = []
        for gpu in self.config.gpus:
            for worker_slot in range(worker_count_per_gpu):
                process = mp.Process(
                    target=_stage_worker_main,
                    args=(gpu, worker_slot, stage, video_queue, result_queue, self.descriptor_map, self.config),
                )
                process.start()
                workers.append(process)

        stage_results = {}
        completed = 0
        total = len(prioritized_videos)
        last_result_time = time.monotonic()
        stall_timeout_sec = self.config.wave_stall_timeout_sec
        stalled = False

        while completed < total:
            try:
                result = result_queue.get(timeout=1)
            except Empty:
                alive_workers = [worker for worker in workers if worker.is_alive()]
                if not alive_workers:
                    break
                if time.monotonic() - last_result_time >= stall_timeout_sec:
                    print(
                        f"[wave:{stage}] No worker results for {stall_timeout_sec}s; "
                        f"terminating {len(alive_workers)} stalled worker(s)."
                    )
                    self._stop_workers(alive_workers, force=False)
                    stalled = True
                    break
                continue

            video_path = result["video"]
            stage_results[video_path] = result["success"]
            completed += 1
            last_result_time = time.monotonic()
            on_result(result)

        self._join_workers(workers)

        missing = [video_path for video_path in prioritized_videos if video_path not in stage_results]
        for video_path in missing:
            error = "worker_stalled_without_result" if stalled else "worker_exited_without_result"
            synthetic_result = {"video": video_path, "success": False, "gpu": None, "error": error}
            stage_results[video_path] = False
            on_result(synthetic_result)

        return stage_results
