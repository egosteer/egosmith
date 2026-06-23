from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from lib.pipeline.clips.clip_manifest import load_clip_manifest


DEFAULT_BATCH_STAGES = "detect_track,motion,slam,infiller"
DEFAULT_ANY4D_BATCH_SIZE = 32
DEFAULT_INFILLER_WINDOW_BATCH_SIZE = 64
DEFAULT_WAVE_STALL_TIMEOUT_SEC = 3600
DEFAULT_INFER_PROFILE = "standard"
INFER_PROFILE_CHOICES = ("standard", "throughput_80gb")
LOCAL_CACHE_MODE_CHOICES = ("off", "tar", "image_sequence", "all")
SHARED_PROFILE_CACHE_OPTION_DESTS = (
    "infer_profile",
    "local_cache_root",
    "local_cache_quota_gb",
    "local_cache_mode",
    "local_cache_min_frames",
)


@dataclass(frozen=True)
class BatchInputSelection:
    input_mode: str
    input_path: str
    total_items: int
    start_idx: int
    end_idx: int
    video_paths: list[str]
    descriptors: list | None


def collect_videos(video_dir: Path, extensions=(".mp4", ".avi", ".mov")) -> list[str]:
    videos = []
    for ext in extensions:
        videos.extend(str(path) for path in video_dir.rglob(f"*{ext}"))
    return sorted(videos)


def add_infer_profile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--infer_profile",
        type=str,
        default=DEFAULT_INFER_PROFILE,
        choices=list(INFER_PROFILE_CHOICES),
        help="Optional throughput tuning profile that only adjusts scheduling/cache defaults.",
    )


def add_any4d_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    include_depth_predict_all_frames: bool = True,
    include_stage3_tmp_root: bool = True,
) -> None:
    if include_depth_predict_all_frames:
        parser.add_argument(
            "--depth_predict_all_frames",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Predict dense depth for all frames in the SLAM stage. Default: enabled.",
        )
    parser.add_argument("--any4d_repo_root", type=str, default=None, help="Optional Any4D repository root.")
    parser.add_argument("--any4d_checkpoint_path", type=str, default=None, help="Optional Any4D checkpoint path.")
    parser.add_argument("--any4d_resolution_set", type=int, default=None, help="Optional Any4D resolution set.")
    parser.add_argument(
        "--any4d_use_amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override AMP usage for Any4D inference.",
    )
    if include_stage3_tmp_root:
        parser.add_argument("--stage3_tmp_root", type=str, default=None, help="Temporary workspace root for Any4D SLAM.")


def add_local_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--local_cache_root", type=str, default=None, help="Optional local scratch root for shared per-clip frame cache.")
    parser.add_argument("--local_cache_quota_gb", type=float, default=None, help="Optional quota for the shared local clip cache.")
    parser.add_argument(
        "--local_cache_mode",
        type=str,
        default="off",
        choices=list(LOCAL_CACHE_MODE_CHOICES),
        help="Which descriptor types should use the shared local clip cache.",
    )
    parser.add_argument(
        "--local_cache_min_frames",
        type=int,
        default=1,
        help="Only materialize clips with at least this many frames into the local clip cache.",
    )


def build_batch_infer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-GPU batch inference scheduler for HaWoR")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--descriptor_manifest",
        type=str,
        help="Frozen clip manifest JSONL path. Preferred input mode.",
    )
    input_group.add_argument(
        "--video_list",
        type=str,
        help="Compatibility input mode: path to text file with one video path per line.",
    )
    input_group.add_argument(
        "--video_dir",
        type=str,
        help="Compatibility input mode: directory to recursively search for video files.",
    )

    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs (e.g. '0,1,2,3').")
    parser.add_argument("--stages", type=str, default=DEFAULT_BATCH_STAGES, help="Comma-separated stage names.")
    add_infer_profile_arg(parser)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing outputs. Default: enabled.",
    )
    parser.add_argument("--run_dir", type=str, help="Custom run directory. Default: batch_runs/<timestamp>.")
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Root for per-video stage intermediates (sets $HAWOR_OUTPUT_ROOT). "
            "Intermediates go to <output_root>/<stem>.hawor_pipeline/stage_outputs/<stem>. "
            "Default: a sibling <stem>.hawor_pipeline next to each video."
        ),
    )
    parser.add_argument(
        "--legacy-seq-folder",
        dest="legacy_seq_folder",
        action="store_true",
        default=False,
        help=(
            "Opt in to the legacy layout that writes intermediates directly next to "
            "the input video (parent/<stem>/). Off by default."
        ),
    )
    parser.add_argument(
        "--keep_intermediates",
        type=str,
        default="none",
        choices=["none", "slam", "all"],
        help=(
            "Retention after the final (infiller) stage. 'none' (default): keep only "
            "the consolidated result.npz (+ SLAM scale), removing tracks/masks/cam-space/"
            "stage-3 frames/markers and the redundant depth+pose files. 'slam': keep "
            "intermediates but drop heavy redundant SLAM caches + stage-3 frames. 'all': "
            "keep everything."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./weights/hawor/checkpoints/hawor.ckpt",
        help="Path to HaWoR checkpoint.",
    )
    parser.add_argument(
        "--infiller_weight",
        type=str,
        default="./weights/hawor/checkpoints/infiller.pt",
        help="Path to infiller weights.",
    )
    parser.add_argument("--img_focal", type=float, help="Image focal length.")
    parser.add_argument(
        "--chunk_batch_size",
        type=int,
        default=64,
        help="Number of 16-frame chunks processed per forward in the motion stage.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of DataLoader workers for frame loading in the motion stage.",
    )
    parser.add_argument(
        "--any4d_batch_size",
        type=int,
        default=None,
        help="Batch size for Any4D depth in the SLAM stage.",
    )
    parser.add_argument(
        "--any4d_overlap",
        type=int,
        default=4,
        help="Frames of overlap between consecutive Any4D chunks (default 4, matches the EgoSmith "
        "paper's cross-batch metric-scale stitch; 0 = off). >0 runs the rigorous overlap "
        "scale-stitch and sets HAWOR_ANY4D_OVERLAP for SLAM workers. Must be < --any4d_batch_size.",
    )
    parser.add_argument(
        "--hand_anchor",
        action="store_true",
        default=False,
        help="Anchor the dense depth metric to the trusted HaWoR hand with one global factor k "
        "(before est_scale, so camera+hand inherit the hand metric). Sets HAWOR_HAND_ANCHOR.",
    )
    parser.add_argument(
        "--hand_anchor_alpha",
        action="store_true",
        default=False,
        help="On top of --hand_anchor, apply a temporally-smooth per-frame scale alpha(t) to the "
        "SAVED depth map only (hand sits on the depth surface per-frame; camera/hand trajectory "
        "untouched). Sets HAWOR_HAND_ANCHOR_ALPHA.",
    )
    parser.add_argument(
        "--hand_shape_stabilize",
        action="store_true",
        default=False,
        help="Per-clip hand-shape stabilization in the motion stage: replace per-frame MANO betas "
        "with one median shape and depth-compensate (trans×f), preserving the 2D overlay. Sets "
        "HAWOR_HAND_SHAPE_STABILIZE.",
    )
    parser.add_argument(
        "--render_batch_size",
        type=int,
        default=8,
        help="Batch size for the rendering phase in the motion stage.",
    )
    parser.add_argument(
        "--infiller_window_batch_size",
        type=int,
        default=DEFAULT_INFILLER_WINDOW_BATCH_SIZE,
        help="Number of infiller windows processed per batch.",
    )
    parser.add_argument(
        "--detect_batch_size",
        type=int,
        default=128,
        help="Batch size for YOLO detection in the detect_track stage.",
    )
    parser.add_argument(
        "--detect_io_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers for frame loading in the detect_track stage.",
    )
    parser.add_argument(
        "--detect_device",
        type=str,
        default="cuda:0",
        help="Device for YOLO detection in the detect_track stage.",
    )
    parser.add_argument(
        "--detect_half_precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use FP16 for YOLO detection. Default: disabled.",
    )
    parser.add_argument(
        "--enable_profiler",
        action="store_true",
        help="Enable torch profiler to diagnose performance bottlenecks.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index of the selected input list (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="End index of the selected input list (exclusive).")
    parser.add_argument(
        "--scheduler_mode",
        type=str,
        default="legacy",
        choices=["legacy", "wave"],
        help="Compatibility flag. Unified scheduler always uses stage-wave execution.",
    )
    parser.add_argument(
        "--persistent_worker",
        action="store_true",
        help="Compatibility flag retained for older launch scripts.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Deprecated alias for --max_stage_retries.",
    )
    parser.add_argument(
        "--max_stage_retries",
        type=int,
        default=1,
        help="Max retries per stage wave.",
    )
    parser.add_argument(
        "--wave_stall_timeout_sec",
        type=int,
        default=DEFAULT_WAVE_STALL_TIMEOUT_SEC,
        help="Terminate and retry a stage wave if workers emit no results for this many seconds.",
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="Default worker slots per GPU for stage-wave execution.",
    )
    parser.add_argument(
        "--detect_track_workers_per_gpu",
        type=int,
        default=None,
        help="Optional per-stage override for detect_track worker slots per GPU.",
    )
    parser.add_argument(
        "--motion_workers_per_gpu",
        type=int,
        default=None,
        help="Optional per-stage override for motion worker slots per GPU.",
    )
    parser.add_argument(
        "--slam_workers_per_gpu",
        type=int,
        default=None,
        help="Optional per-stage override for slam worker slots per GPU.",
    )
    parser.add_argument(
        "--infiller_workers_per_gpu",
        type=int,
        default=None,
        help="Optional per-stage override for infiller worker slots per GPU.",
    )
    parser.add_argument(
        "--slam_backend",
        type=str,
        default="dpvo",
        choices=["dpvo"],
        help="SLAM backend for the slam stage (only 'dpvo' is supported).",
    )
    parser.add_argument(
        "--depth_backend",
        type=str,
        default=None,
        choices=["any4d"],
        help="Depth backend for SLAM scale estimation (only 'any4d' is supported).",
    )
    parser.add_argument("--any4d", action="store_true", help="No-op; Any4D is the only depth backend. Kept for compatibility.")
    parser.add_argument(
        "--rebuild_cam_space_cache",
        action="store_true",
        help="Rebuild cached camera-space infiller inputs before running infiller.",
    )
    add_any4d_runtime_args(parser)
    add_local_cache_args(parser)
    return parser


def normalize_batch_infer_args(args, *, raw_argv: list[str] | None = None) -> list[str]:
    raw_argv = list(sys.argv[1:] if raw_argv is None else raw_argv)
    notes = []

    if getattr(args, "any4d", False):
        args.depth_backend = "any4d"

    if getattr(args, "any4d_batch_size", None) is None:
        args.any4d_batch_size = DEFAULT_ANY4D_BATCH_SIZE

    retries = getattr(args, "retries", None)
    if retries is not None and "--max_stage_retries" not in raw_argv:
        args.max_stage_retries = int(retries)
        notes.append("`--retries` is deprecated; treating it as `--max_stage_retries`.")

    scheduler_mode = getattr(args, "scheduler_mode", "wave")
    if scheduler_mode != "wave":
        notes.append(
            f"`--scheduler_mode {scheduler_mode}` is retained for compatibility; the unified scheduler uses wave mode."
        )
    if getattr(args, "persistent_worker", False):
        notes.append("`--persistent_worker` is retained for compatibility and has no effect in the unified scheduler.")

    infer_profile = getattr(args, "infer_profile", DEFAULT_INFER_PROFILE)
    if infer_profile == "throughput_80gb":
        if getattr(args, "detect_track_workers_per_gpu", None) is None:
            args.detect_track_workers_per_gpu = 2
        if getattr(args, "motion_workers_per_gpu", None) is None:
            args.motion_workers_per_gpu = 1
        if getattr(args, "slam_workers_per_gpu", None) is None:
            args.slam_workers_per_gpu = 1
        if getattr(args, "infiller_workers_per_gpu", None) is None:
            args.infiller_workers_per_gpu = 2
        if getattr(args, "local_cache_mode", "off") == "off":
            args.local_cache_mode = "all"
        if getattr(args, "local_cache_quota_gb", None) is None:
            args.local_cache_quota_gb = 2000.0
        if getattr(args, "local_cache_root", None) is None:
            args.local_cache_root = getattr(args, "stage3_tmp_root", None)
        if getattr(args, "local_cache_min_frames", 1) < 96:
            args.local_cache_min_frames = 96
        if "--num_workers" not in raw_argv:
            args.num_workers = max(int(getattr(args, "num_workers", 16)), 32)
        notes.append(
            "`--infer_profile throughput_80gb` enabled shared local clip cache and more aggressive motion loading defaults."
        )

    return notes


def load_batch_inputs(args) -> BatchInputSelection:
    descriptors = None
    if args.descriptor_manifest:
        if not Path(args.descriptor_manifest).is_file():
            raise FileNotFoundError(f"--descriptor_manifest not found: {args.descriptor_manifest}")
        records = load_clip_manifest(args.descriptor_manifest)
        descriptors = [record.descriptor for record in records]
        video_paths = [descriptor.video_key for descriptor in descriptors]
        input_mode = "descriptor_manifest"
        input_path = args.descriptor_manifest
    elif args.video_list:
        if not Path(args.video_list).is_file():
            raise FileNotFoundError(f"--video_list not found: {args.video_list}")
        with open(args.video_list, "r", encoding="utf-8") as handle:
            video_paths = [line.strip() for line in handle if line.strip()]
        input_mode = "video_list"
        input_path = args.video_list
    else:
        if not args.video_dir or not Path(args.video_dir).is_dir():
            raise NotADirectoryError(
                f"--video_dir not found or not a directory: {args.video_dir!r} "
                "(pass --video_list, --video_dir, or --descriptor_manifest)"
            )
        video_paths = collect_videos(Path(args.video_dir))
        input_mode = "video_dir"
        input_path = args.video_dir

    total_items = len(video_paths)
    if total_items <= 0:
        raise ValueError("No videos found for batch inference")

    start_idx = int(args.start)
    end_idx = total_items if args.end is None else int(args.end)
    if start_idx < 0 or start_idx >= total_items:
        raise ValueError(f"--start {start_idx} is out of range [0, {total_items})")
    if end_idx < start_idx or end_idx > total_items:
        raise ValueError(f"--end {end_idx} is out of range [{start_idx}, {total_items}]")

    selected_video_paths = video_paths[start_idx:end_idx]
    selected_descriptors = descriptors[start_idx:end_idx] if descriptors is not None else None
    if not selected_video_paths:
        raise ValueError(f"No videos in range [{start_idx}, {end_idx})")

    return BatchInputSelection(
        input_mode=input_mode,
        input_path=input_path,
        total_items=total_items,
        start_idx=start_idx,
        end_idx=end_idx,
        video_paths=selected_video_paths,
        descriptors=selected_descriptors,
    )
