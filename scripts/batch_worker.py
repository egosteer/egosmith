import argparse
import json
import os
import random
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch

# Suppress common warnings to reduce output noise
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*timm.models.layers.*')
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# src-layout: first-party packages live under src/; scripts/ stays importable from root.
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Temp dir: redirect process-level temp files only when HAWOR_BATCH_TMPDIR is set.
# Never fall back to the repo dir; an unset value leaves the system default in place.
# IMPORTANT: Set this AFTER importing torch to avoid library loading issues
_env_tmp = os.environ.get("HAWOR_BATCH_TMPDIR")
if _env_tmp:
    SHARED_TMP_DIR = Path(_env_tmp).expanduser().resolve()
    SHARED_TMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(SHARED_TMP_DIR)
    os.environ["TEMP"] = str(SHARED_TMP_DIR)
    os.environ["TMP"] = str(SHARED_TMP_DIR)
    tempfile.tempdir = str(SHARED_TMP_DIR)

# Quiet routine stage chatter by default, but let the user override with
# HAWOR_QUIET=0. Genuine warnings/errors go through lib.pipeline.proc.logging_setup,
# which stays visible regardless of this flag.
os.environ.setdefault("HAWOR_QUIET", "1")


STAGES = ["detect_track", "motion", "slam", "infiller"]


def set_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True: safe because input sizes are fixed (256x256 crops),
    # gives 5-15% speedup on convolutions via cuDNN auto-tuning
    torch.backends.cudnn.benchmark = True


def get_seq_folder(video_path: str) -> Path:
    # NOTE: This is the legacy demo/standalone fork (also used by the
    # webdataset_features infiller fallback, which passes a clip *directory* as
    # video_path and expects world_space_res.pth written in place). It must stay
    # on the next-to-video layout and self-consistent with the scripts_test_video
    # stage copies. The centralized layout (lib.pipeline.io.workspace) is for the
    # canonical pipeline only.
    video_path = Path(video_path)
    return video_path.parent / video_path.stem


def get_track_range(seq_folder: Path, fast=False):
    """
    Get track range from seq_folder.

    Args:
        seq_folder: Sequence folder path
        fast: If True, use ultra-fast method (read from cache file)
    """
    if fast:
        # Ultra-fast mode: read from .track_range cache file
        cache_file = seq_folder / ".track_range"
        if cache_file.exists():
            try:
                content = cache_file.read_text().strip()
                start_idx, end_idx = map(int, content.split(","))
                return start_idx, end_idx
            except (OSError, ValueError):
                # Corrupt/unreadable cache -> fall back to directory scan below.
                pass

        # Fast mode: assume standard naming tracks_0_N
        # Try to find it without full iteration
        for p in seq_folder.iterdir():
            if p.is_dir() and p.name.startswith("tracks_0_"):
                parts = p.name.split("_")
                if len(parts) == 3:
                    try:
                        start_idx = int(parts[1])
                        end_idx = int(parts[2])
                        # Cache for next time
                        cache_file.write_text(f"{start_idx},{end_idx}")
                        return start_idx, end_idx
                    except ValueError:
                        pass
        # Fallback to slow method if fast fails

    # Slow method: glob all tracks_*_* directories
    track_dirs = []
    for p in seq_folder.glob("tracks_*_*"):
        parts = p.name.split("_")
        if len(parts) != 3:
            continue
        try:
            start_idx = int(parts[1])
            end_idx = int(parts[2])
        except ValueError:
            continue

        # Check if directory is empty (from previous failures)
        # Empty tracks directories should be cleaned up
        if p.is_dir():
            contents = list(p.iterdir())
            if len(contents) == 0:
                # Empty directory - remove it
                try:
                    p.rmdir()
                    continue  # Skip this directory
                except OSError:
                    pass  # If removal fails (race / non-empty), keep it in the list

        track_dirs.append((start_idx, end_idx, p))

    if not track_dirs:
        raise FileNotFoundError(f"No tracks_*_* folder found under {seq_folder}")

    track_dirs.sort(key=lambda x: (x[1], x[0]))
    start_idx, end_idx, _ = track_dirs[-1]

    # Cache the result
    cache_file = seq_folder / ".track_range"
    cache_file.write_text(f"{start_idx},{end_idx}")

    return start_idx, end_idx


def validate_stage_output(stage: str, seq_folder: Path, start_idx: int, end_idx: int):
    tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"

    if stage == "detect_track":
        assert (tracks_dir / "model_boxes.npy").exists(), "model_boxes.npy missing"
        assert (tracks_dir / "model_tracks.npy").exists(), "model_tracks.npy missing"
        return

    if stage == "motion":
        # Check for incomplete outputs and auto-fix
        frame_chunks_file = tracks_dir / "frame_chunks_all.npy"
        model_masks_file = tracks_dir / "model_masks.npy"

        if frame_chunks_file.exists() and not model_masks_file.exists():
            # Incomplete output detected - remove to force re-run
            import sys
            print(f"Warning: Incomplete motion output detected for {seq_folder}", file=sys.stderr)
            print(f"  - frame_chunks_all.npy exists but model_masks.npy missing", file=sys.stderr)
            print(f"  - Removing incomplete output to force re-run", file=sys.stderr)
            frame_chunks_file.unlink()
            raise AssertionError("Incomplete motion output - removed and will retry")

        assert frame_chunks_file.exists(), "frame_chunks_all.npy missing"
        assert model_masks_file.exists(), "model_masks.npy missing"
        # Read only the npy header (~200 bytes) instead of loading 264-622 MB
        with open(model_masks_file, 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        assert len(shape) == 3, f"model_masks should be (T,H,W), got shape {shape}"
        return

    if stage == "slam":
        slam_file = seq_folder / "SLAM" / f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
        assert slam_file.exists(), "SLAM npz missing"
        data = np.load(slam_file, allow_pickle=True)
        assert "traj" in data and "scale" in data, "invalid SLAM npz keys"
        return

    if stage == "infiller":
        world_file = seq_folder / "world_space_res.pth"
        assert world_file.exists(), "world_space_res.pth missing"
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_file)
        assert pred_trans.shape[0] == 2 and pred_trans.shape[-1] == 3, "pred_trans shape invalid"
        assert pred_rot.shape[0] == 2 and pred_rot.shape[-1] == 3, "pred_rot shape invalid"
        assert pred_hand_pose.shape[0] == 2 and pred_hand_pose.shape[-1] == 45, "pred_hand_pose shape invalid"
        assert pred_betas.shape[0] == 2 and pred_betas.shape[-1] == 10, "pred_betas shape invalid"
        assert pred_valid.shape[0] == 2, "pred_valid shape invalid"
        return

    raise ValueError(f"Unknown stage: {stage}")


def is_stage_complete(stage: str, seq_folder: Path, fast_check=False):
    """
    Check if a stage is complete.

    Args:
        stage: Stage name
        seq_folder: Sequence folder path
        fast_check: If True, use ultra-fast check (only check .done marker file)
    """
    # Check if seq_folder exists first
    if not seq_folder.exists():
        return False

    if fast_check:
        # Ultra-fast check: only check .done marker file
        done_marker = seq_folder / f".{stage}.done"
        if done_marker.exists():
            return True
        # If no marker, fall through to file existence check

    try:
        start_idx, end_idx = get_track_range(seq_folder, fast=fast_check)

        if fast_check:
            # Fast check: only verify files exist, don't load them
            result = validate_stage_output_fast(stage, seq_folder, start_idx, end_idx)
            # Create .done marker for next time
            if result:
                done_marker = seq_folder / f".{stage}.done"
                done_marker.touch()
            return result
        else:
            # Full validation: load and check content
            validate_stage_output(stage, seq_folder, start_idx, end_idx)
            # Create .done marker
            done_marker = seq_folder / f".{stage}.done"
            done_marker.touch()
            return True
    except Exception:
        return False


def validate_stage_output_fast(stage: str, seq_folder: Path, start_idx: int, end_idx: int):
    """Fast validation: only check if required files exist."""
    tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"

    if stage == "detect_track":
        return (
            (tracks_dir / "model_boxes.npy").exists() and
            (tracks_dir / "model_tracks.npy").exists()
        )

    if stage == "motion":
        return (
            (tracks_dir / "frame_chunks_all.npy").exists() and
            (tracks_dir / "model_masks.npy").exists()
        )

    if stage == "slam":
        slam_file = seq_folder / "SLAM" / f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
        return slam_file.exists()

    if stage == "infiller":
        world_res = seq_folder / "world_space_res.pth"
        return world_res.exists()

    return False


class WorkerRuntime:
    def __init__(
        self,
        gpu: str,
        checkpoint: str,
        infiller_weight: str,
        img_focal: float = None,
        input_type: str = "file",
        chunk_batch_size: int = 4,
        num_workers: int = 16,
        render_batch_size: int = 8,
        any4d_batch_size: int = 32,
        detect_batch_size: int = 128,
        detect_io_workers: int = 8,
        slam_backend: str = "dpvo",
    ):
        self.gpu = gpu
        self.checkpoint = checkpoint
        self.infiller_weight = infiller_weight
        self.img_focal = img_focal
        self.input_type = input_type
        self.chunk_batch_size = chunk_batch_size
        self.num_workers = num_workers
        self.render_batch_size = render_batch_size
        self.any4d_batch_size = any4d_batch_size
        self.detect_batch_size = detect_batch_size
        self.detect_io_workers = detect_io_workers
        self.slam_backend = slam_backend

        self.detector_runner = None
        self.motion_runner = None
        self.infiller_runner = None
        self.mano_right = None
        self.mano_left = None

        if self.gpu is not None and self.gpu != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

    def build_stage_args(self, video_path: str):
        class StageArgs:
            pass

        args = StageArgs()
        args.img_focal = self.img_focal
        args.video_path = video_path
        args.input_type = self.input_type
        args.checkpoint = self.checkpoint
        args.infiller_weight = self.infiller_weight
        args.chunk_batch_size = self.chunk_batch_size
        args.num_workers = self.num_workers
        args.render_batch_size = self.render_batch_size
        args.any4d_batch_size = self.any4d_batch_size
        args.detect_batch_size = self.detect_batch_size
        args.detect_io_workers = self.detect_io_workers
        args.vis_mode = "world"
        args.skip_vis = True
        return args

    def ensure_runner(self, stage: str):
        if stage == "detect_track" and self.detector_runner is None:
            from ultralytics import YOLO

            self.detector_runner = YOLO('./weights/external/detector.pt')

        if stage == "motion" and self.motion_runner is None:
            from lib.stage_runners.hawor_video import build_motion_runner

            self.motion_runner = build_motion_runner(self.checkpoint)

            # Cache MANO models (created once per worker, reused across videos)
            from lib.models.mano_wrapper import MANO
            device = self.motion_runner['device']

            self.mano_right = MANO(
                data_dir='_DATA/data/',
                model_path='_DATA/data/mano',
                gender='neutral',
                num_hand_joints=15,
                create_body_pose=False,
            ).to(device)

            self.mano_left = MANO(
                data_dir='_DATA/data_left/',
                model_path='_DATA/data_left/mano_left',
                gender='neutral',
                num_hand_joints=15,
                create_body_pose=False,
                is_rhand=False,
            ).to(device)
            # Fix MANO shapedirs of the left hand bug
            self.mano_left.shapedirs[:, 0, :] *= -1

        if stage == "infiller" and self.infiller_runner is None:
            from lib.stage_runners.hawor_video import build_infiller_runner

            self.infiller_runner = build_infiller_runner(self.infiller_weight)



def build_stage_args(ns):
    class StageArgs:
        pass

    args = StageArgs()
    args.img_focal = ns.img_focal
    args.video_path = ns.video_path
    args.input_type = ns.input_type
    args.checkpoint = ns.checkpoint
    args.infiller_weight = ns.infiller_weight
    args.chunk_batch_size = ns.chunk_batch_size
    args.vis_mode = "world"
    args.skip_vis = True
    return args


def run_stage_with_runtime(runtime: WorkerRuntime, ns, prefetched_data=None):
    stage_args = runtime.build_stage_args(ns.video_path)
    seq_folder = get_seq_folder(ns.video_path)

    if ns.resume and not ns.force and is_stage_complete(ns.stage, seq_folder, fast_check=True):
        return {
            "status": "skipped",
            "reason": "existing_valid_output",
        }

    from lib.stage_runners.detect_track_video import detect_track_video
    from lib.stage_runners.hawor_slam import hawor_slam
    from lib.stage_runners.hawor_video import run_infiller_for_video, run_motion_for_video

    runtime.ensure_runner(ns.stage)

    # For SLAM stage, aggressively release runners that are no longer needed
    # (detector, motion, infiller) to free GPU memory before DPVO.
    if ns.stage == "slam":
        try:
            runtime.detector_runner = None
            runtime.motion_runner = None
            runtime.infiller_runner = None
            torch.cuda.empty_cache()
        except Exception:
            pass

    if ns.stage == "detect_track":
        start_idx, end_idx, _, _ = detect_track_video(
            stage_args,
            detector_runner=runtime.detector_runner,
            force=ns.force,
            detect_batch_size=ns.detect_batch_size,
            num_io_workers=ns.detect_io_workers,
            device=ns.detect_device,
            half_precision=ns.detect_half_precision,
        )
    else:
        start_idx, end_idx = get_track_range(seq_folder, fast=True)
        # Verify the tracks directory actually exists
        tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"
        if not tracks_dir.exists():
            # Cache was stale, invalidate and retry
            cache_file = seq_folder / ".track_range"
            if cache_file.exists():
                cache_file.unlink()
            start_idx, end_idx = get_track_range(seq_folder, fast=False)
            tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"
            if not tracks_dir.exists():
                raise FileNotFoundError(f"Tracks directory not found: {tracks_dir}")

    if ns.stage == "motion":
        mano_models = None
        if runtime.mano_right is not None and runtime.mano_left is not None:
            mano_models = {'right': runtime.mano_right, 'left': runtime.mano_left}
        run_motion_for_video(
            stage_args,
            start_idx,
            end_idx,
            str(seq_folder),
            motion_runner=runtime.motion_runner,
            mano_models=mano_models,
            prefetched_data=prefetched_data,
        )
    elif ns.stage == "slam":
        hawor_slam(
            stage_args,
            start_idx,
            end_idx,
            any4d_batch_size=ns.any4d_batch_size,
            slam_backend=getattr(ns, "slam_backend", "dpvo"),
            depth_backend=getattr(ns, "depth_backend", None),
            depth_predict_all_frames=getattr(ns, "depth_predict_all_frames", True),
        )
    elif ns.stage == "infiller":
        tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"
        frame_chunks_all = joblib.load(tracks_dir / "frame_chunks_all.npy")
        run_infiller_for_video(
            stage_args,
            start_idx,
            end_idx,
            frame_chunks_all,
            infiller_runner=runtime.infiller_runner,
        )
    elif ns.stage == "detect_track":
        pass
    else:
        raise ValueError(f"Unknown stage: {ns.stage}")

    validate_stage_output(ns.stage, seq_folder, start_idx, end_idx)

    # Create .done marker after successful validation
    done_marker = seq_folder / f".{ns.stage}.done"
    done_marker.touch()

    return {
        "status": "success",
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def worker_runtime_loop(ns):
    set_determinism(ns.seed)
    runtime = WorkerRuntime(
        gpu=ns.gpu,
        checkpoint=ns.checkpoint,
        infiller_weight=ns.infiller_weight,
        img_focal=ns.img_focal,
        input_type=ns.input_type,
        chunk_batch_size=ns.chunk_batch_size,
        num_workers=getattr(ns, 'num_workers', 16),
        render_batch_size=getattr(ns, 'render_batch_size', 8),
        detect_io_workers=getattr(ns, 'detect_io_workers', 8),
        slam_backend=getattr(ns, 'slam_backend', "dpvo"),
    )

    with open(ns.video_list) as f:
        video_paths = [line.strip() for line in f if line.strip()]

    overall_success = True
    for video_path in video_paths:
        task_ns = argparse.Namespace(**vars(ns))
        task_ns.video_path = video_path

        common_fields = {
            "video": task_ns.video_path,
            "stage": task_ns.stage,
            "gpu": task_ns.gpu,
        }
        started_at = time.time()
        emit_event("stage_start", **common_fields)
        try:
            result = run_stage_with_runtime(runtime, task_ns)
            emit_event(
                "stage_end",
                **common_fields,
                status=result.get("status", "success"),
                elapsed_sec=round(time.time() - started_at, 3),
                reason=result.get("reason"),
                start_idx=result.get("start_idx"),
                end_idx=result.get("end_idx"),
            )
        except Exception as err:
            overall_success = False
            emit_event(
                "stage_end",
                **common_fields,
                status="failed",
                elapsed_sec=round(time.time() - started_at, 3),
                error=str(err),
            )
            traceback.print_exc()

        # Free GPU memory between videos to prevent fragmentation
        torch.cuda.empty_cache()

    return overall_success


def emit_event(event: str, **kwargs):
    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **kwargs,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def run_stage(ns):
    if ns.gpu is not None and ns.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ns.gpu)

    set_determinism(ns.seed)

    stage_args = build_stage_args(ns)
    seq_folder = get_seq_folder(ns.video_path)

    if ns.resume and not ns.force and is_stage_complete(ns.stage, seq_folder, fast_check=True):
        return {
            "status": "skipped",
            "reason": "existing_valid_output",
        }

    from lib.stage_runners.detect_track_video import detect_track_video
    from lib.stage_runners.hawor_slam import hawor_slam
    from lib.stage_runners.hawor_video import hawor_infiller, hawor_motion_estimation

    if ns.stage == "detect_track":
        start_idx, end_idx, _, _ = detect_track_video(
            stage_args,
            detect_batch_size=ns.detect_batch_size,
            num_io_workers=ns.detect_io_workers,
            device=ns.detect_device,
            half_precision=ns.detect_half_precision,
        )
    else:
        start_idx, end_idx = get_track_range(seq_folder, fast=True)
        # Verify the tracks directory actually exists
        tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"
        if not tracks_dir.exists():
            # Cache was stale, invalidate and retry
            cache_file = seq_folder / ".track_range"
            if cache_file.exists():
                cache_file.unlink()
            start_idx, end_idx = get_track_range(seq_folder, fast=False)
            tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"
            if not tracks_dir.exists():
                raise FileNotFoundError(f"Tracks directory not found: {tracks_dir}")

    if ns.stage == "motion":
        # Enable profiling if requested
        if getattr(ns, 'enable_profiler', False):
            from torch.profiler import profile, ProfilerActivity, schedule
            # Save profiler traces in batch run directory if available
            if getattr(ns, 'run_dir', None):
                profiler_output_dir = Path(ns.run_dir) / "profiler_traces"
            else:
                # Fallback for standalone usage (not called from batch_infer.py)
                profiler_output_dir = seq_folder.parent / "profiler_traces"
            profiler_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"[PROFILER] Enabled. Output dir: {profiler_output_dir}")
            print(f"[PROFILER] Video: {Path(ns.video_path).stem}")

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=0, warmup=1, active=3, repeat=1),
                on_trace_ready=lambda p: (
                    print(f"[PROFILER] Trace ready, exporting to {profiler_output_dir / f'motion_trace_{Path(ns.video_path).stem}.json'}"),
                    p.export_chrome_trace(str(profiler_output_dir / f"motion_trace_{Path(ns.video_path).stem}.json"))
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                hawor_motion_estimation(stage_args, start_idx, end_idx, str(seq_folder), profiler=prof)
        else:
            hawor_motion_estimation(stage_args, start_idx, end_idx, str(seq_folder))
    elif ns.stage == "slam":
        hawor_slam(
            stage_args,
            start_idx,
            end_idx,
            any4d_batch_size=ns.any4d_batch_size,
            slam_backend=getattr(ns, "slam_backend", "dpvo"),
            depth_backend=getattr(ns, "depth_backend", None),
            depth_predict_all_frames=getattr(ns, "depth_predict_all_frames", True),
        )
    elif ns.stage == "infiller":
        tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"
        frame_chunks_all = joblib.load(tracks_dir / "frame_chunks_all.npy")
        hawor_infiller(stage_args, start_idx, end_idx, frame_chunks_all)
    elif ns.stage == "detect_track":
        pass
    else:
        raise ValueError(f"Unknown stage: {ns.stage}")

    validate_stage_output(ns.stage, seq_folder, start_idx, end_idx)

    # Create .done marker after successful validation
    done_marker = seq_folder / f".{ns.stage}.done"
    done_marker.touch()

    return {
        "status": "success",
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--gpu", default="", type=str)
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--input_type", type=str, default="file")
    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str, default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16, help="Number of DataLoader workers for parallel frame loading")
    parser.add_argument("--render_batch_size", type=int, default=8, help="Batch size for rendering phase")
    parser.add_argument(
        "--any4d_batch_size",
        type=int,
        default=48,
        help="Batch size for Any4D depth in SLAM (env HAWOR_ANY4D_BATCH_SIZE overrides)",
    )
    parser.add_argument("--detect_batch_size", type=int, default=128, help="Batch size for YOLO detection (default 128)")
    parser.add_argument("--detect_io_workers", type=int, default=8, help="Number of DataLoader workers for parallel frame loading")
    parser.add_argument("--detect_device", type=str, default="cuda:0", help="Device for YOLO detector (e.g., cuda:0)")
    # Default: disable FP16 to improve numerical stability/debuggability across envs.
    parser.add_argument("--detect_half_precision", action="store_true", default=False, help="Use FP16 for YOLO detector (2x faster)")
    parser.add_argument("--no-detect_half_precision", dest="detect_half_precision", action="store_false", help="Disable FP16 for YOLO")
    parser.add_argument("--slam_backend", type=str, default="dpvo", choices=["dpvo"], help="SLAM backend for the 'slam' stage (only 'dpvo' is supported)")
    parser.add_argument(
        "--depth_backend",
        type=str,
        default=None,
        choices=["any4d"],
        help="Depth backend for SLAM scale (dpvo): only 'any4d' is supported; omit → env HAWOR_DEPTH_BACKEND",
    )
    parser.add_argument(
        "--any4d",
        action="store_true",
        help="Shorthand for --depth_backend any4d",
    )
    parser.add_argument(
        "--no_depth_predict_all_frames",
        dest="depth_predict_all_frames",
        action="store_false",
        help="SLAM: keyframe-only depth (faster; no dense_depth_*.npz). Default: dense on.",
    )
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true", help="Ignore existing outputs and rerun this stage")
    parser.add_argument("--video_list", type=str, help="Optional file with one video path per line for persistent worker mode")
    parser.add_argument("--persistent_worker", action="store_true", help="Run as long-lived stage worker for multiple videos")
    parser.add_argument("--enable_profiler", action="store_true", help="Enable torch profiler to diagnose performance bottlenecks")
    parser.add_argument("--run_dir", type=str, help="Batch run directory for output organization")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    if getattr(args, "any4d", False):
        args.depth_backend = "any4d"

    if args.persistent_worker:
        if not args.video_list:
            raise ValueError("--video_list is required when --persistent_worker is set")
        success = worker_runtime_loop(args)
        sys.exit(0 if success else 1)

    started_at = time.time()
    common_fields = {
        "video": args.video_path,
        "stage": args.stage,
        "gpu": args.gpu,
    }

    emit_event("stage_start", **common_fields)
    try:
        result = run_stage(args)
        emit_event(
            "stage_end",
            **common_fields,
            status=result.get("status", "success"),
            elapsed_sec=round(time.time() - started_at, 3),
            reason=result.get("reason"),
            start_idx=result.get("start_idx"),
            end_idx=result.get("end_idx"),
        )
        sys.exit(0)
    except Exception as err:
        emit_event(
            "stage_end",
            **common_fields,
            status="failed",
            elapsed_sec=round(time.time() - started_at, 3),
            error=str(err),
        )
        traceback.print_exc()
        sys.exit(1)
