"""Batch-inference run configuration (BatchRunConfig) and stage-name aliases."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Mapping, Optional

from lib.pipeline.batch.cli import DEFAULT_INFER_PROFILE

if TYPE_CHECKING:
    from lib.pipeline.datasets.descriptors import ClipDescriptor


STAGE_ALIASES = {
    "detect": "detect_track",
}

VALID_BATCH_STAGES = ["detect_track", "motion", "slam", "infiller"]


@dataclass(frozen=True, kw_only=True)
class BatchRunConfig:
    video_paths: List[str]
    gpus: List[int]
    stages: List[str]
    run_dir: Path
    descriptors: Optional[List["ClipDescriptor"]] = None
    resume: bool = True
    checkpoint: str = "./weights/hawor/checkpoints/hawor.ckpt"
    infiller_weight: str = "./weights/hawor/checkpoints/infiller.pt"
    img_focal: Optional[float] = None
    chunk_batch_size: int = 64
    num_workers: int = 16
    any4d_batch_size: int = 32
    any4d_overlap: int = 4
    hand_anchor: bool = False
    hand_anchor_alpha: bool = False
    hand_shape_stabilize: bool = False
    render_batch_size: int = 8
    infiller_window_batch_size: int = 64
    detect_batch_size: int = 128
    detect_device: str = "cuda:0"
    detect_half_precision: bool = False
    detect_io_workers: int = 8
    rebuild_cam_space_cache: bool = False
    depth_predict_all_frames: Optional[bool] = True
    any4d_repo_root: Optional[str] = None
    any4d_checkpoint_path: Optional[str] = None
    any4d_resolution_set: Optional[int] = None
    any4d_use_amp: Optional[bool] = None
    stage3_tmp_root: Optional[str] = None
    keep_intermediates: str = "all"
    infer_profile: str = DEFAULT_INFER_PROFILE
    local_cache_root: Optional[str] = None
    local_cache_quota_gb: Optional[float] = None
    local_cache_mode: str = "off"
    local_cache_min_frames: int = 1
    max_stage_retries: int = 1
    wave_stall_timeout_sec: int = 3600
    workers_per_gpu: int = 1
    detect_track_workers_per_gpu: Optional[int] = None
    motion_workers_per_gpu: Optional[int] = None
    slam_workers_per_gpu: Optional[int] = None
    infiller_workers_per_gpu: Optional[int] = None
    enable_profiler: bool = False

    @classmethod
    def from_args(cls, args, *, video_paths: List[str], descriptors: Optional[List["ClipDescriptor"]], run_dir: Path):
        return cls.from_namespace(args, video_paths=video_paths, descriptors=descriptors, run_dir=run_dir)

    @classmethod
    def from_namespace(
        cls,
        ns,
        *,
        video_paths: List[str],
        descriptors: Optional[List["ClipDescriptor"]],
        run_dir: Path,
    ):
        raw_stages = [part.strip() for part in str(getattr(ns, "stages", "")).split(",") if part.strip()]
        stages = [STAGE_ALIASES.get(stage, stage) for stage in raw_stages]
        invalid = [stage for stage in stages if stage not in VALID_BATCH_STAGES]
        if invalid:
            raise ValueError(f"Unknown stages: {invalid}. Valid stages: {VALID_BATCH_STAGES}")

        gpus = [int(gpu.strip()) for gpu in str(getattr(ns, "gpus", "")).split(",") if gpu.strip()]
        if not gpus:
            raise ValueError("At least one GPU must be specified via --gpus")

        any4d_batch_size = getattr(ns, "any4d_batch_size", None)
        if any4d_batch_size is None:
            any4d_batch_size = 32
        wave_stall_timeout_sec = getattr(ns, "wave_stall_timeout_sec", None)
        if wave_stall_timeout_sec is None:
            wave_stall_timeout_sec = 3600

        worker_counts = {
            "workers_per_gpu": getattr(ns, "workers_per_gpu", 1),
            "detect_track_workers_per_gpu": getattr(ns, "detect_track_workers_per_gpu", None),
            "motion_workers_per_gpu": getattr(ns, "motion_workers_per_gpu", None),
            "slam_workers_per_gpu": getattr(ns, "slam_workers_per_gpu", None),
            "infiller_workers_per_gpu": getattr(ns, "infiller_workers_per_gpu", None),
        }
        invalid_counts = {name: value for name, value in worker_counts.items() if value is not None and value < 1}
        if invalid_counts:
            raise ValueError(f"Worker counts must be >= 1: {invalid_counts}")
        if any4d_batch_size < 1:
            raise ValueError("--any4d_batch_size must be >= 1")
        any4d_overlap = int(getattr(ns, "any4d_overlap", 4) or 0)
        if any4d_overlap < 0:
            raise ValueError("--any4d_overlap must be >= 0")
        if any4d_overlap >= any4d_batch_size:
            raise ValueError(
                f"--any4d_overlap ({any4d_overlap}) must be < --any4d_batch_size ({any4d_batch_size})"
            )
        if wave_stall_timeout_sec < 1:
            raise ValueError("--wave_stall_timeout_sec must be >= 1")

        return cls(
            video_paths=video_paths,
            descriptors=descriptors,
            gpus=gpus,
            stages=stages,
            resume=bool(getattr(ns, "resume", True)),
            run_dir=run_dir,
            checkpoint=getattr(ns, "checkpoint", "./weights/hawor/checkpoints/hawor.ckpt"),
            infiller_weight=getattr(ns, "infiller_weight", "./weights/hawor/checkpoints/infiller.pt"),
            img_focal=getattr(ns, "img_focal", None),
            chunk_batch_size=getattr(ns, "chunk_batch_size", 64),
            num_workers=getattr(ns, "num_workers", 16),
            any4d_batch_size=any4d_batch_size,
            any4d_overlap=any4d_overlap,
            hand_anchor=bool(getattr(ns, "hand_anchor", False)),
            hand_anchor_alpha=bool(getattr(ns, "hand_anchor_alpha", False)),
            hand_shape_stabilize=bool(getattr(ns, "hand_shape_stabilize", False)),
            render_batch_size=getattr(ns, "render_batch_size", 8),
            infiller_window_batch_size=getattr(ns, "infiller_window_batch_size", 64),
            detect_batch_size=getattr(ns, "detect_batch_size", 128),
            detect_device=getattr(ns, "detect_device", "cuda:0"),
            detect_half_precision=bool(getattr(ns, "detect_half_precision", False)),
            detect_io_workers=getattr(ns, "detect_io_workers", 8),
            rebuild_cam_space_cache=bool(getattr(ns, "rebuild_cam_space_cache", False)),
            depth_predict_all_frames=getattr(ns, "depth_predict_all_frames", True),
            any4d_repo_root=getattr(ns, "any4d_repo_root", None),
            any4d_checkpoint_path=getattr(ns, "any4d_checkpoint_path", None),
            any4d_resolution_set=getattr(ns, "any4d_resolution_set", None),
            any4d_use_amp=getattr(ns, "any4d_use_amp", None),
            stage3_tmp_root=getattr(ns, "stage3_tmp_root", None),
            keep_intermediates=getattr(ns, "keep_intermediates", "all"),
            infer_profile=getattr(ns, "infer_profile", DEFAULT_INFER_PROFILE),
            local_cache_root=getattr(ns, "local_cache_root", None),
            local_cache_quota_gb=getattr(ns, "local_cache_quota_gb", None),
            local_cache_mode=getattr(ns, "local_cache_mode", "off"),
            local_cache_min_frames=getattr(ns, "local_cache_min_frames", 1),
            max_stage_retries=getattr(ns, "max_stage_retries", 1),
            wave_stall_timeout_sec=wave_stall_timeout_sec,
            workers_per_gpu=getattr(ns, "workers_per_gpu", 1),
            detect_track_workers_per_gpu=getattr(ns, "detect_track_workers_per_gpu", None),
            motion_workers_per_gpu=getattr(ns, "motion_workers_per_gpu", None),
            slam_workers_per_gpu=getattr(ns, "slam_workers_per_gpu", None),
            infiller_workers_per_gpu=getattr(ns, "infiller_workers_per_gpu", None),
            enable_profiler=bool(getattr(ns, "enable_profiler", False)),
        )

    @property
    def descriptor_map(self):
        if not self.descriptors:
            return {}
        return {descriptor.video_key: descriptor for descriptor in self.descriptors}

    def worker_count_for_stage(self, stage: str) -> int:
        overrides = {
            "detect_track": self.detect_track_workers_per_gpu,
            "motion": self.motion_workers_per_gpu,
            "slam": self.slam_workers_per_gpu,
            "infiller": self.infiller_workers_per_gpu,
        }
        return overrides.get(stage) or self.workers_per_gpu

    def worker_env_overrides(self) -> Mapping[str, str | None]:
        overrides: dict[str, str | None] = {
            "HAWOR_LOCAL_CACHE_ROOT": self.local_cache_root,
            "HAWOR_LOCAL_CACHE_QUOTA_GB": None if self.local_cache_quota_gb is None else str(self.local_cache_quota_gb),
            "HAWOR_LOCAL_CACHE_MODE": self.local_cache_mode,
            "HAWOR_LOCAL_CACHE_MIN_FRAMES": str(self.local_cache_min_frames),
        }
        # Overlap is always propagated so the resolved value (default 4, or an explicit 0 to
        # disable) deterministically reaches the SLAM workers and overrides the runtime default.
        overrides["HAWOR_ANY4D_OVERLAP"] = str(int(self.any4d_overlap))
        # Hand-anchor knobs stay opt-in, so the defaults leave any externally-exported
        # HAWOR_HAND_ANCHOR* untouched.
        if self.hand_anchor:
            overrides["HAWOR_HAND_ANCHOR"] = "1"
        if self.hand_anchor_alpha:
            overrides["HAWOR_HAND_ANCHOR_ALPHA"] = "1"
        if self.hand_shape_stabilize:
            overrides["HAWOR_HAND_SHAPE_STABILIZE"] = "1"
        return overrides
