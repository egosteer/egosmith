"""Public pipeline stage API: the PipelineVideoTask / StageExecutionConfig types, stage entry helpers, and stage-output validation used by both the single-video and batch paths."""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from lib.pipeline.datasets.descriptors import ClipDescriptor
from lib.pipeline.io.frame_sources import build_frame_source_from_descriptor

# Stage-output filesystem locations + per-stage validation live in dedicated modules;
# re-exported here so the historical lib.pipeline.proc.stage_api import paths keep working.
from .stage_paths import get_seq_folder, get_stage_done_marker, get_tracks_dir
from .stage_validators import (
    _FAST_STAGE_VALIDATORS,
    _STAGE_VALIDATORS,
    validate_stage_output,
    validate_stage_output_fast,
)


STAGES = ["detect_track", "motion", "slam", "infiller"]


@dataclass(frozen=True)
class StageExecutionConfig:
    img_focal: Optional[float] = None
    checkpoint: str = "./weights/hawor/checkpoints/hawor.ckpt"
    infiller_weight: str = "./weights/hawor/checkpoints/infiller.pt"
    chunk_batch_size: int = 64
    num_workers: int = 16
    render_batch_size: int = 8
    any4d_batch_size: int = 32
    detect_batch_size: int = 128
    detect_io_workers: int = 8
    infiller_window_batch_size: int = 64
    rebuild_cam_space_cache: bool = False
    detect_device: str = "cuda:0"
    detect_half_precision: bool = True
    depth_predict_all_frames: Optional[bool] = None
    any4d_repo_root: Optional[str] = None
    any4d_checkpoint_path: Optional[str] = None
    any4d_resolution_set: Optional[int] = None
    any4d_use_amp: Optional[bool] = None
    stage3_tmp_root: Optional[str] = None
    vis_mode: str = "world"
    skip_vis: bool = True
    # Retention after the final (infiller) stage: all|slam|none. Default 'all'
    # (no cleanup) so programmatic callers are never surprised; the batch CLI
    # sets the user-facing default.
    keep_intermediates: str = "all"

    @classmethod
    def from_namespace(cls, ns):
        return cls(
            img_focal=getattr(ns, "img_focal", None),
            checkpoint=getattr(ns, "checkpoint", "./weights/hawor/checkpoints/hawor.ckpt"),
            infiller_weight=getattr(ns, "infiller_weight", "./weights/hawor/checkpoints/infiller.pt"),
            chunk_batch_size=getattr(ns, "chunk_batch_size", 64),
            num_workers=getattr(ns, "num_workers", 16),
            render_batch_size=getattr(ns, "render_batch_size", 8),
            any4d_batch_size=getattr(ns, "any4d_batch_size", 32),
            detect_batch_size=getattr(ns, "detect_batch_size", 128),
            detect_io_workers=getattr(ns, "detect_io_workers", 8),
            infiller_window_batch_size=getattr(ns, "infiller_window_batch_size", 64),
            rebuild_cam_space_cache=bool(getattr(ns, "rebuild_cam_space_cache", False)),
            detect_device=getattr(ns, "detect_device", "cuda:0"),
            detect_half_precision=bool(getattr(ns, "detect_half_precision", True)),
            depth_predict_all_frames=getattr(ns, "depth_predict_all_frames", None),
            any4d_repo_root=getattr(ns, "any4d_repo_root", None),
            any4d_checkpoint_path=getattr(ns, "any4d_checkpoint_path", None),
            any4d_resolution_set=getattr(ns, "any4d_resolution_set", None),
            any4d_use_amp=getattr(ns, "any4d_use_amp", None),
            stage3_tmp_root=getattr(ns, "stage3_tmp_root", None),
            keep_intermediates=getattr(ns, "keep_intermediates", "all"),
        )

    def to_stage_args(self, video_path: str):
        return argparse.Namespace(
            img_focal=self.img_focal,
            video_path=video_path,
            checkpoint=self.checkpoint,
            infiller_weight=self.infiller_weight,
            chunk_batch_size=self.chunk_batch_size,
            num_workers=self.num_workers,
            render_batch_size=self.render_batch_size,
            any4d_batch_size=self.any4d_batch_size,
            detect_batch_size=self.detect_batch_size,
            detect_io_workers=self.detect_io_workers,
            infiller_window_batch_size=self.infiller_window_batch_size,
            rebuild_cam_space_cache=self.rebuild_cam_space_cache,
            detect_device=self.detect_device,
            detect_half_precision=self.detect_half_precision,
            depth_predict_all_frames=self.depth_predict_all_frames,
            any4d_repo_root=self.any4d_repo_root,
            any4d_checkpoint_path=self.any4d_checkpoint_path,
            any4d_resolution_set=self.any4d_resolution_set,
            any4d_use_amp=self.any4d_use_amp,
            stage3_tmp_root=self.stage3_tmp_root,
            vis_mode=self.vis_mode,
            skip_vis=self.skip_vis,
        )


@dataclass(frozen=True)
class PipelineVideoTask:
    video_path: str
    seq_folder: Path
    descriptor: Optional[ClipDescriptor] = None

    @classmethod
    def from_inputs(cls, video_path: Optional[str] = None, descriptor: Optional[ClipDescriptor] = None):
        if descriptor is not None:
            return cls(
                video_path=descriptor.media_path or descriptor.clip_id,
                seq_folder=Path(descriptor.seq_folder),
                descriptor=descriptor,
            )
        if video_path is None:
            raise ValueError("video_path is required when descriptor is not provided")
        return cls(
            video_path=video_path,
            seq_folder=get_seq_folder(video_path=video_path),
            descriptor=None,
        )

    @classmethod
    def from_namespace(cls, ns):
        return cls.from_inputs(
            video_path=getattr(ns, "video_path", None),
            descriptor=getattr(ns, "_descriptor", None),
        )

    def build_frame_source(self):
        if self.descriptor is None:
            return None
        return build_frame_source_from_descriptor(self.descriptor)


@dataclass(frozen=True)
class StageArtifacts:
    start_idx: int
    end_idx: int
    seq_folder: Path
    stage_name: str = ""

    @property
    def tracks_dir(self) -> Path:
        return get_tracks_dir(self.seq_folder, self.start_idx, self.end_idx)

    @property
    def done_marker(self) -> Path:
        return get_stage_done_marker(self.seq_folder, self.stage_name)


def _get_track_range_cache_file(seq_folder: Path) -> Path:
    return seq_folder / ".track_range"


def _read_cached_track_range(seq_folder: Path):
    cache_file = _get_track_range_cache_file(seq_folder)
    if not cache_file.exists():
        return None

    try:
        content = cache_file.read_text().strip()
        start_idx, end_idx = map(int, content.split(","))
        return start_idx, end_idx
    except Exception:
        return None


def _write_cached_track_range(seq_folder: Path, start_idx: int, end_idx: int):
    _get_track_range_cache_file(seq_folder).write_text(f"{start_idx},{end_idx}")


def _discover_fast_track_range(seq_folder: Path):
    for track_dir in seq_folder.iterdir():
        if not track_dir.is_dir() or not track_dir.name.startswith("tracks_0_"):
            continue
        parts = track_dir.name.split("_")
        if len(parts) != 3:
            continue
        try:
            start_idx = int(parts[1])
            end_idx = int(parts[2])
        except ValueError:
            continue
        _write_cached_track_range(seq_folder, start_idx, end_idx)
        return start_idx, end_idx
    return None


def _parse_track_dir(track_dir: Path):
    parts = track_dir.name.split("_")
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _collect_track_dirs(seq_folder: Path):
    track_dirs = []
    for track_dir in seq_folder.glob("tracks_*_*"):
        parsed = _parse_track_dir(track_dir)
        if parsed is None:
            continue
        start_idx, end_idx = parsed

        if track_dir.is_dir():
            contents = list(track_dir.iterdir())
            if len(contents) == 0:
                try:
                    track_dir.rmdir()
                    continue
                except OSError:
                    pass

        track_dirs.append((start_idx, end_idx, track_dir))
    return track_dirs


def get_track_range(seq_folder: Path, fast=False):
    if fast:
        cached_range = _read_cached_track_range(seq_folder)
        if cached_range is not None:
            return cached_range

        discovered_range = _discover_fast_track_range(seq_folder)
        if discovered_range is not None:
            return discovered_range

    track_dirs = _collect_track_dirs(seq_folder)
    if not track_dirs:
        raise FileNotFoundError(f"No tracks_*_* folder found under {seq_folder}")

    track_dirs.sort(key=lambda item: (item[1], item[0]))
    start_idx, end_idx, _ = track_dirs[-1]
    _write_cached_track_range(seq_folder, start_idx, end_idx)
    return start_idx, end_idx


def _mark_stage_done(seq_folder: Path, stage: str):
    get_stage_done_marker(seq_folder, stage).touch()


def _run_fast_stage_check(stage: str, seq_folder: Path):
    start_idx, end_idx = get_track_range(seq_folder, fast=True)
    result = validate_stage_output_fast(stage, seq_folder, start_idx, end_idx)
    if result:
        _mark_stage_done(seq_folder, stage)
    return result


def _run_full_stage_check(stage: str, seq_folder: Path):
    start_idx, end_idx = get_track_range(seq_folder, fast=False)
    validate_stage_output(stage, seq_folder, start_idx, end_idx)
    _mark_stage_done(seq_folder, stage)
    return True


def is_stage_complete(stage: str, seq_folder: Path, fast_check=False):
    if not seq_folder.exists():
        return False

    # If the consolidated final result exists, the whole clip is done -- even if
    # retention cleanup removed upstream intermediates and stage markers. Without
    # this, a resumed run would wrongly re-run earlier stages whose outputs were
    # cleaned away.
    from lib.pipeline.io.result_io import final_artifact_exists

    if final_artifact_exists(seq_folder):
        return True

    if fast_check and get_stage_done_marker(seq_folder, stage).exists():
        return True

    try:
        if fast_check:
            return _run_fast_stage_check(stage, seq_folder)
        return _run_full_stage_check(stage, seq_folder)
    except Exception:
        return False


def _reset_track_range_cache(seq_folder: Path):
    cache_file = _get_track_range_cache_file(seq_folder)
    if cache_file.exists():
        cache_file.unlink()


def _resolve_non_detect_track_artifacts(stage: str, seq_folder: Path) -> StageArtifacts:
    start_idx, end_idx = get_track_range(seq_folder, fast=True)
    tracks_dir = get_tracks_dir(seq_folder, start_idx, end_idx)
    if not tracks_dir.exists():
        _reset_track_range_cache(seq_folder)
        start_idx, end_idx = get_track_range(seq_folder, fast=False)
        tracks_dir = get_tracks_dir(seq_folder, start_idx, end_idx)
        if not tracks_dir.exists():
            raise FileNotFoundError(f"Tracks directory not found: {tracks_dir}")

    return StageArtifacts(start_idx=start_idx, end_idx=end_idx, seq_folder=seq_folder, stage_name=stage)


def resolve_stage_artifacts(stage: str, seq_folder: Path) -> StageArtifacts:
    if stage == "detect_track":
        return StageArtifacts(start_idx=0, end_idx=0, seq_folder=seq_folder, stage_name=stage)
    return _resolve_non_detect_track_artifacts(stage, seq_folder)


def _ensure_runtime_for_stage(runtime, stage: str):
    if runtime is not None and hasattr(runtime, "ensure_runner"):
        runtime.ensure_runner(stage)


def _build_motion_mano_models(runtime):
    if getattr(runtime, "mano_right", None) is None or getattr(runtime, "mano_left", None) is None:
        return None
    return {"right": runtime.mano_right, "left": runtime.mano_left}


def _run_detect_track_stage(task, stage_args, config, runtime, frame_source, force):
    from lib.pipeline.stages.detect_track import detect_track_video

    start_idx, end_idx, _, _ = detect_track_video(
        stage_args,
        detector_runner=getattr(runtime, "detector_runner", None),
        force=force,
        detect_batch_size=config.detect_batch_size,
        num_io_workers=config.detect_io_workers,
        device=config.detect_device,
        half_precision=config.detect_half_precision,
        frame_source=frame_source,
        seq_folder=str(task.seq_folder),
    )
    return start_idx, end_idx


def _run_motion_stage(task, stage_args, config, runtime, frame_source, profiler, prefetched_data, force, start_idx, end_idx):
    from lib.pipeline.stages.hawor_video import run_motion_for_video

    _frame_chunks_all, _img_focal, timing = run_motion_for_video(
        stage_args,
        start_idx,
        end_idx,
        str(task.seq_folder),
        motion_runner=getattr(runtime, "motion_runner", None),
        profiler=profiler,
        mano_models=_build_motion_mano_models(runtime),
        prefetched_data=prefetched_data,
        frame_source=frame_source,
        force=force,
        return_timing=True,
    )
    return {"timing": timing}


def _run_slam_stage(task, stage_args, config, runtime, frame_source, start_idx, end_idx):
    from lib.pipeline.stages.slam import hawor_slam

    metrics = hawor_slam(
        stage_args,
        start_idx,
        end_idx,
        any4d_runner=getattr(runtime, "any4d_runner", None),
        any4d_batch_size=config.any4d_batch_size,
        frame_source=frame_source,
        seq_folder=str(task.seq_folder),
        return_timing=True,
    )
    if isinstance(metrics, dict) and isinstance(metrics.get("timing"), dict):
        return metrics
    return {"timing": metrics}


def _run_infiller_stage(task, stage_args, runtime, frame_source, prefetched_data, start_idx, end_idx):
    from lib.pipeline.stages.hawor_video import run_infiller_for_video

    tracks_dir = get_tracks_dir(task.seq_folder, start_idx, end_idx)
    frame_chunks_all = None
    cam_space_cache = None
    if isinstance(prefetched_data, dict):
        frame_chunks_all = prefetched_data.get("frame_chunks_all")
        cam_space_cache = prefetched_data.get("cam_space_cache")
    if frame_chunks_all is None:
        frame_chunks_all = joblib.load(tracks_dir / "frame_chunks_all.npy")
    num_frames = None
    if task.descriptor is not None:
        num_frames = int(task.descriptor.frame_count)
    return run_infiller_for_video(
        stage_args,
        start_idx,
        end_idx,
        frame_chunks_all,
        infiller_runner=getattr(runtime, "infiller_runner", None),
        frame_source=frame_source,
        seq_folder=str(task.seq_folder),
        num_frames=num_frames,
        cam_space_cache=cam_space_cache,
        return_timing=True,
    )


def _run_non_detect_stage(stage, task, stage_args, config, runtime, frame_source, profiler, prefetched_data, force):
    artifacts = resolve_stage_artifacts(stage, task.seq_folder)
    start_idx = artifacts.start_idx
    end_idx = artifacts.end_idx
    metrics = {}

    if stage == "motion":
        metrics = _run_motion_stage(task, stage_args, config, runtime, frame_source, profiler, prefetched_data, force, start_idx, end_idx)
    elif stage == "slam":
        metrics = _run_slam_stage(task, stage_args, config, runtime, frame_source, start_idx, end_idx)
    elif stage == "infiller":
        metrics = _run_infiller_stage(task, stage_args, runtime, frame_source, prefetched_data, start_idx, end_idx)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return start_idx, end_idx, metrics


def _finalize_stage_run(stage: str, seq_folder: Path, start_idx: int, end_idx: int, *, metrics: dict | None = None, wall_sec: float | None = None):
    validate_stage_output(stage, seq_folder, start_idx, end_idx)
    _mark_stage_done(seq_folder, stage)
    result = {
        "status": "success",
        "start_idx": start_idx,
        "end_idx": end_idx,
    }
    if metrics:
        result["metrics"] = metrics
    if wall_sec is not None:
        result["wall_sec"] = float(wall_sec)
    return result


def run_pipeline_stage(
    stage: str,
    task: PipelineVideoTask,
    config: StageExecutionConfig,
    *,
    runtime=None,
    prefetched_data=None,
    profiler=None,
    resume=True,
    force=False,
):
    if resume and not force and is_stage_complete(stage, task.seq_folder, fast_check=True):
        return {
            "status": "skipped",
            "reason": "existing_valid_output",
            "wall_sec": 0.0,
        }

    _ensure_runtime_for_stage(runtime, stage)

    prefetched_frame_source = None
    if isinstance(prefetched_data, dict):
        prefetched_frame_source = prefetched_data.get("frame_source")
    frame_source = prefetched_frame_source
    if frame_source is None and stage != "infiller":
        frame_source = task.build_frame_source()
    stage_args = config.to_stage_args(task.video_path)
    stage_start_time = time.time()
    metrics = None

    if stage == "detect_track":
        start_idx, end_idx = _run_detect_track_stage(task, stage_args, config, runtime, frame_source, force)
    else:
        start_idx, end_idx, metrics = _run_non_detect_stage(
            stage,
            task,
            stage_args,
            config,
            runtime,
            frame_source,
            profiler,
            prefetched_data,
            force,
        )

    finalize_result = _finalize_stage_run(
        stage,
        task.seq_folder,
        start_idx,
        end_idx,
        metrics=metrics,
        wall_sec=time.time() - stage_start_time,
    )

    # Retention: after the final (infiller) stage succeeds, optionally drop the now
    # redundant intermediates (depth is preserved inside result.npz). Only the
    # batch/infer-only path enables this; the dataset orchestrator passes
    # keep_intermediates='all' so the downstream build still has what it needs.
    level = getattr(config, "keep_intermediates", "all")
    if stage == "infiller" and level != "all":
        _run_retention_cleanup(task.seq_folder, level, start_idx, end_idx, config)

    return finalize_result


def _run_retention_cleanup(seq_folder: Path, level: str, start_idx: int, end_idx: int, config):
    try:
        from lib.pipeline.proc.cleanup import cleanup_seq_folder
        from lib.pipeline.io.workspace import resolve_tmp_root

        tmp_root = resolve_tmp_root(config, required=False)
        cleanup_seq_folder(
            seq_folder,
            level=level,
            tmp_root=tmp_root,
            start_idx=start_idx,
            end_idx=end_idx,
        )
    except Exception as error:  # cleanup must never fail the run
        from lib.pipeline.proc.logging_setup import get_logger

        get_logger("stage_api").warning("Retention cleanup failed for %s: %s", seq_folder, error)
