#!/usr/bin/env python3
"""Legacy BuildAI-oriented VLA WebDataset builder."""

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .webdataset_discovery import (  # noqa: E402
    discover_episode_stats,
    discover_episodes,
    get_episode_feature_cache_path,
    load_episode_stats,
    load_or_build_frame_index,
    parse_factory_range,
)
from .webdataset_annotation import (  # noqa: E402
    DEFAULT_ANNOTATION_SUFFIX,
    attach_or_filter_episode_instructions,
)
from .webdataset_features import (  # noqa: E402
    DEFAULT_INTRINSIC,
    FINGERTIP_INDICES,
    LOWDIM_SIZE,
    build_mano_models,
    load_episode_features,
    run_infill_for_episode,
    run_mano_forward,
)
from .webdataset_geometry import (  # noqa: E402
    axis_angle_to_rot6d,
    interpolate_extrinsics,
    normalize_slam_keyframes,
    quat_to_4x4,
)
from .webdataset_workers import (  # noqa: E402
    _worker_init,
    _worker_prepare_episode_feature_batch,
    _worker_prepare_episode_features,
    _worker_process_shard,
    normalize_mano_devices,
)
from .webdataset_writer import add_sample_to_tar, iter_episode_samples, plan_shards  # noqa: E402

__all__ = [
    "DEFAULT_INTRINSIC",
    "DEFAULT_ANNOTATION_SUFFIX",
    "FINGERTIP_INDICES",
    "LOWDIM_SIZE",
    "_worker_init",
    "_worker_prepare_episode_feature_batch",
    "_worker_prepare_episode_features",
    "_worker_process_shard",
    "add_sample_to_tar",
    "attach_or_filter_episode_instructions",
    "axis_angle_to_rot6d",
    "build_mano_models",
    "discover_episode_stats",
    "discover_episodes",
    "get_episode_feature_cache_path",
    "interpolate_extrinsics",
    "iter_episode_samples",
    "load_episode_features",
    "load_episode_stats",
    "load_or_build_frame_index",
    "main",
    "normalize_mano_devices",
    "parse_factory_range",
    "normalize_slam_keyframes",
    "plan_shards",
    "quat_to_4x4",
    "run_infill_for_episode",
    "run_mano_forward",
]

PRECOMPUTE_EPISODES_PER_BATCH = 128
PRECOMPUTE_FRAMES_PER_BATCH = 16384


def build_parser():
    """Build CLI parser for WebDataset export."""
    parser = argparse.ArgumentParser(
        description="Legacy BuildAI-oriented builder. Prefer scripts/run_dataset_pipeline.py or scripts/build/build_vla_from_manifest.py"
    )
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--episode_list", default=None, help="Text file with one episode path per line")
    parser.add_argument("--frames_per_shard", type=int, default=10000)
    parser.add_argument("--factory_range", default=None, help="Inclusive factory range like 1-50 for 10K BuildAI layout")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for testing")
    parser.add_argument(
        "--preprocess_workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of workers for episode stats and annotation preprocessing",
    )
    parser.add_argument("--mano_device", default="cuda:0", help="Device for MANO forward pass")
    parser.add_argument("--mano_gpus", default=None, help="Comma-separated GPU ids for parallel MANO workers, e.g. 0,1,2,3")
    parser.add_argument("--mano_dir", default=None, help="Directory containing MANO_RIGHT.pkl and MANO_LEFT.pkl")
    parser.add_argument("--rescan", action="store_true", help="Force rescan episodes and frame indexes")
    parser.add_argument(
        "--feature_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache per-episode lowdim features under output_dir for faster reruns",
    )
    parser.add_argument(
        "--precompute_features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Precompute per-episode lowdim features before shard writing to improve GPU utilization and rerun speed",
    )
    parser.add_argument("--writer_workers", type=int, default=8, help="Number of parallel shard writers")
    parser.add_argument("--shard_manifest_out", default=None, help="Optional JSON manifest of planned shards")
    parser.add_argument("--auto_infill", action="store_true", help="Run infill for missing world_space_res.pth")
    parser.add_argument(
        "--checkpoint",
        default="weights/hawor/checkpoints/hawor.ckpt",
        help="HaWoR checkpoint path (required if --auto_infill)",
    )
    parser.add_argument(
        "--infiller_weight",
        default="weights/hawor/checkpoints/infiller.pt",
        help="Infiller weight path (required if --auto_infill)",
    )
    parser.add_argument(
        "--annotation_suffix",
        default=DEFAULT_ANNOTATION_SUFFIX,
        help="Suffix for per-episode annotation JSON files",
    )
    parser.add_argument(
        "--allow_missing_annotation",
        action="store_true",
        help="Keep episodes even when annotation is missing or invalid",
    )
    return parser


def normalize_args(args):
    """Normalize deprecated aliases and validate simple invariants."""
    if args.preprocess_workers < 1:
        raise ValueError("--preprocess_workers must be >= 1")
    parse_factory_range(args.factory_range)
    return args.writer_workers


def validate_auto_infill_args(args):
    """Validate auto-infill dependencies."""
    if not args.auto_infill:
        return True
    if not args.checkpoint or not args.infiller_weight:
        print("Error: --auto_infill requires --checkpoint and --infiller_weight")
        return False
    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return False
    if not os.path.exists(args.infiller_weight):
        print(f"Error: infiller_weight not found: {args.infiller_weight}")
        return False
    return True


def maybe_rescan_episode_cache(cache_file, rescan):
    """Drop stale episode cache when requested."""
    if rescan and os.path.exists(cache_file):
        os.remove(cache_file)


def maybe_run_auto_infill(args, cache_file, writer_workers):
    """Run infiller for episodes missing world-space results."""
    if not args.auto_infill:
        return

    all_episodes = discover_episodes(
        args.input_dir,
        episode_list=args.episode_list,
        max_episodes=args.max_episodes,
        require_world_res=False,
        factory_range=args.factory_range,
    )
    missing_infill = [
        ep
        for ep in all_episodes
        if not os.path.exists(os.path.join(ep["crop_dir"], "world_space_res.pth"))
    ]
    if missing_infill:
        if len(missing_infill) > max(8, writer_workers):
            print(
                "Warning: many episodes need infill. For large runs, prefer "
                "`scripts/batch_infer.py --stages infiller` before building WebDataset."
            )
        print(f"Running infill for {len(missing_infill)} episodes...")
        infill_failures = []
        for ep in tqdm(missing_infill, desc="Infill"):
            ok = run_infill_for_episode(ep["crop_dir"], args.checkpoint, args.infiller_weight, args.mano_device)
            if not ok:
                infill_failures.append(ep["crop_dir"])
        if infill_failures:
            print(
                f"WARNING: infill failed for {len(infill_failures)}/{len(missing_infill)} episodes "
                f"(see errors above). First few: {infill_failures[:5]}"
            )

    if os.path.exists(cache_file):
        os.remove(cache_file)


def prepare_episode_stats(args, cache_file):
    """Discover, validate, and expand episodes before shard planning."""
    episodes = discover_episodes(
        args.input_dir,
        args.episode_list,
        args.max_episodes,
        cache_file=cache_file,
        factory_range=args.factory_range,
    )
    print(f"Found {len(episodes)} episodes with world_space_res.pth")
    if not episodes:
        return None, None

    print("Collecting episode stats...")
    episode_stats = discover_episode_stats(
        episodes,
        rescan_frame_index=args.rescan,
        workers=args.preprocess_workers,
    )
    if not episode_stats:
        return episodes, None

    episode_stats, annotation_stats = attach_or_filter_episode_instructions(
        episode_stats,
        annotation_suffix=args.annotation_suffix,
        allow_missing_annotation=args.allow_missing_annotation,
        workers=args.preprocess_workers,
    )
    print(
        "Annotation filter:"
        f" kept={annotation_stats['kept']}"
        f" filtered={annotation_stats['filtered']}"
        f" missing={annotation_stats['missing_annotation']}"
        f" invalid_json={annotation_stats['invalid_json']}"
        f" invalid_status={annotation_stats['invalid_status']}"
        f" empty_instruction={annotation_stats['empty_instruction']}"
    )
    if not episode_stats:
        return episodes, None

    return episode_stats, episode_stats


def prepare_feature_cache_dir(args, episodes, episode_stats):
    """Create feature cache directory for reusable episode features."""
    feature_cache_dir = None
    if args.feature_cache:
        print(
            f"Episode feature cache enabled for {len(episode_stats)} episode entries"
            f" at {os.path.join(args.output_dir, '_episode_feature_cache')}"
        )
        feature_cache_dir = os.path.join(args.output_dir, "_episode_feature_cache")
        os.makedirs(feature_cache_dir, exist_ok=True)
    return feature_cache_dir


def _make_precompute_episode_task(ep):
    task = {
        "crop_dir": ep["crop_dir"],
        "episode_id": ep["episode_id"],
        "num_valid_frames": int(ep["num_valid_frames"]),
    }
    frame_ids = ep.get("frame_ids")
    if frame_ids is not None:
        task["frame_ids"] = frame_ids
    return task


def _build_precompute_batches(episode_stats):
    batches = []
    current_batch = []
    current_frames = 0

    def flush():
        nonlocal current_batch, current_frames
        if not current_batch:
            return
        batches.append(current_batch)
        current_batch = []
        current_frames = 0

    for ep in episode_stats:
        task = _make_precompute_episode_task(ep)
        task_frames = int(task["num_valid_frames"])
        if current_batch and (
            len(current_batch) >= PRECOMPUTE_EPISODES_PER_BATCH
            or current_frames + task_frames > PRECOMPUTE_FRAMES_PER_BATCH
        ):
            flush()
        current_batch.append(task)
        current_frames += task_frames
        if len(current_batch) >= PRECOMPUTE_EPISODES_PER_BATCH or current_frames >= PRECOMPUTE_FRAMES_PER_BATCH:
            flush()

    flush()
    return batches


def precompute_episode_features(episode_stats, mano_device_specs, args, feature_cache_dir):
    if not feature_cache_dir or not args.precompute_features:
        return None

    if not episode_stats:
        return {
            "episodes_ok": 0,
            "episodes_failed": 0,
            "frames_cached": 0,
            "workers": 0,
        }

    worker_count = max(1, len(mano_device_specs))
    precompute_batches = _build_precompute_batches(episode_stats)
    total_frames = sum(int(ep["num_valid_frames"]) for ep in episode_stats)
    print(
        f"Precomputing episode features with {worker_count} worker(s)"
        f" on {', '.join(mano_device_specs)}"
        f" across {len(precompute_batches)} batch(es)"
        f" for {len(episode_stats)} episode(s) / {total_frames} frame(s) ..."
    )
    totals = {
        "episodes_ok": 0,
        "episodes_failed": 0,
        "frames_cached": 0,
        "workers": worker_count,
        "batches": len(precompute_batches),
    }
    started_at = time.perf_counter()

    if worker_count <= 1:
        _worker_init(
            mano_device_specs,
            args.mano_dir,
            args.rescan,
            feature_cache_dir,
            require_feature_cache=False,
            eager_model_init=True,
        )
        result_iter = (_worker_prepare_episode_feature_batch(batch) for batch in precompute_batches)
    else:
        mp_context = get_context("spawn")
        with mp_context.Pool(
            worker_count,
            initializer=_worker_init,
            initargs=(mano_device_specs, args.mano_dir, args.rescan, feature_cache_dir, False, True),
        ) as pool:
            result_iter = pool.imap_unordered(_worker_prepare_episode_feature_batch, precompute_batches, chunksize=1)
            with tqdm(total=len(episode_stats), desc="Episode features") as pbar:
                for result in result_iter:
                    totals["episodes_ok"] += int(result["episodes_ok"])
                    totals["episodes_failed"] += int(result["episodes_failed"])
                    totals["frames_cached"] += int(result["frames_cached"])
                    pbar.update(int(result["episodes_total"]))
                    elapsed = max(time.perf_counter() - started_at, 1e-6)
                    pbar.set_postfix(
                        ep_s=f"{totals['episodes_ok'] / elapsed:.2f}",
                        frame_s=f"{totals['frames_cached'] / elapsed:.1f}",
                    )
            elapsed = time.perf_counter() - started_at
            print(
                f"Episode features throughput: {totals['episodes_ok'] / max(elapsed, 1e-6):.2f} ep/s, "
                f"{totals['frames_cached'] / max(elapsed, 1e-6):.1f} frame/s"
            )
            return totals

    with tqdm(total=len(episode_stats), desc="Episode features") as pbar:
        for result in result_iter:
            totals["episodes_ok"] += int(result["episodes_ok"])
            totals["episodes_failed"] += int(result["episodes_failed"])
            totals["frames_cached"] += int(result["frames_cached"])
            pbar.update(int(result["episodes_total"]))
            elapsed = max(time.perf_counter() - started_at, 1e-6)
            pbar.set_postfix(
                ep_s=f"{totals['episodes_ok'] / elapsed:.2f}",
                frame_s=f"{totals['frames_cached'] / elapsed:.1f}",
            )
    elapsed = time.perf_counter() - started_at
    print(
        f"Episode features throughput: {totals['episodes_ok'] / max(elapsed, 1e-6):.2f} ep/s, "
        f"{totals['frames_cached'] / max(elapsed, 1e-6):.1f} frame/s"
    )
    return totals


def write_shard_manifest(shard_tasks, shard_manifest_out):
    """Write planned shard manifest when requested."""
    if not shard_manifest_out:
        return
    with open(shard_manifest_out, "w") as f:
        json.dump(shard_tasks, f, ensure_ascii=False, indent=2)
    print(f"Wrote shard manifest to {shard_manifest_out}")


def resolve_mano_runtime(args, writer_workers):
    """Resolve MANO device placement and worker caps."""
    mano_device = torch.device(args.mano_device if torch.cuda.is_available() else "cpu")
    mano_device_specs = normalize_mano_devices(str(mano_device), args.mano_gpus if mano_device.type == "cuda" else None)

    if mano_device.type == "cuda":
        if len(mano_device_specs) > 1:
            if writer_workers > len(mano_device_specs):
                print(
                    f"Capping shard workers from {writer_workers} to {len(mano_device_specs)} "
                    f"to match MANO GPU workers: {', '.join(mano_device_specs)}"
                )
                writer_workers = len(mano_device_specs)
        elif writer_workers > 1:
            print(
                f"MANO device {mano_device} is CUDA with a single GPU worker; capping shard workers "
                f"from {writer_workers} to 1 to avoid GPU contention. Use --mano_gpus for multi-GPU writing."
            )
            writer_workers = 1

    return mano_device, mano_device_specs, writer_workers


def resolve_writer_runtime(writer_workers, feature_cache_precomputed):
    if feature_cache_precomputed:
        return torch.device("cpu"), ["cpu"], writer_workers
    return None, None, writer_workers


def print_writer_config(writer_workers, mano_device_specs):
    """Print worker allocation summary."""
    if len(mano_device_specs) > 1:
        print(
            f"Writing shards with {writer_workers} worker(s) across MANO GPUs: "
            f"{', '.join(mano_device_specs)}"
        )
    else:
        print(f"Writing shards with {writer_workers} worker(s) on MANO device {mano_device_specs[0]}...")


def run_shard_writers(shard_tasks, writer_workers, mano_device, mano_device_specs, args, feature_cache_dir):
    """Execute shard writers and aggregate progress."""
    totals = {
        "total_frames": 0,
        "total_shards": 0,
        "total_episodes_written": 0,
        "total_skipped": 0,
        "total_shard_elapsed_sec": 0.0,
    }
    pool = None
    started_at = time.perf_counter()

    if writer_workers <= 1:
        _worker_init(
            mano_device_specs,
            args.mano_dir,
            args.rescan,
            feature_cache_dir,
            require_feature_cache=bool(feature_cache_dir and args.precompute_features),
            eager_model_init=not bool(feature_cache_dir and args.precompute_features),
        )
        results_iter = (_worker_process_shard(task) for task in shard_tasks)
    else:
        mp_context = get_context("spawn") if mano_device.type == "cuda" else get_context()
        pool = mp_context.Pool(
            writer_workers,
            initializer=_worker_init,
            initargs=(
                mano_device_specs,
                args.mano_dir,
                args.rescan,
                feature_cache_dir,
                bool(feature_cache_dir and args.precompute_features),
                not bool(feature_cache_dir and args.precompute_features),
            ),
        )
        results_iter = pool.imap_unordered(_worker_process_shard, shard_tasks)

    try:
        with tqdm(results_iter, total=len(shard_tasks), desc="Shards") as pbar:
            for result in pbar:
                totals["total_frames"] += result["frames_written"]
                totals["total_shards"] += 1 if result["frames_written"] > 0 else 0
                totals["total_episodes_written"] += result["episodes_written"]
                totals["total_skipped"] += result["skipped_episodes"]
                totals["total_shard_elapsed_sec"] += float(result.get("elapsed_sec", 0.0))
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                pbar.set_postfix(
                    shard_s=f"{totals['total_shards'] / elapsed:.2f}",
                    frame_s=f"{totals['total_frames'] / elapsed:.1f}",
                    last_s=f"{float(result.get('elapsed_sec', 0.0)):.2f}",
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = max(time.perf_counter() - started_at, 1e-6)
    print(
        f"Shard throughput: {totals['total_shards'] / elapsed:.2f} shard/s, "
        f"{totals['total_frames'] / elapsed:.1f} frame/s, "
        f"avg shard worker time={totals['total_shard_elapsed_sec'] / max(len(shard_tasks), 1):.2f}s"
    )
    return totals


def print_summary(output_dir, totals):
    """Print final export summary."""
    print("\nDone!")
    print(f"  Episodes touched: {totals['total_episodes_written']}")
    print(f"  Skipped episode slices: {totals['total_skipped']}")
    print(f"  Frames: {totals['total_frames']}")
    print(f"  Shards: {totals['total_shards']}")
    print(f"  Output: {output_dir}")


def main():
    started_at = time.perf_counter()
    args = build_parser().parse_args()
    writer_workers = normalize_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    if not validate_auto_infill_args(args):
        return

    cache_file = os.path.join(args.input_dir, "_vla_episodes_cache.json")
    maybe_rescan_episode_cache(cache_file, args.rescan)
    maybe_run_auto_infill(args, cache_file, writer_workers)

    prepare_started_at = time.perf_counter()
    episodes, episode_stats = prepare_episode_stats(args, cache_file)
    prepare_elapsed = time.perf_counter() - prepare_started_at
    if not episodes:
        print("No episodes found!")
        return
    if not episode_stats:
        print("No valid episodes with extracted frames found!")
        return

    feature_cache_dir = prepare_feature_cache_dir(args, episodes, episode_stats)
    shard_tasks = plan_shards(episode_stats, args.frames_per_shard, args.output_dir)
    print(f"Planned {len(shard_tasks)} shards from {sum(ep['num_valid_frames'] for ep in episode_stats)} frames")
    write_shard_manifest(shard_tasks, args.shard_manifest_out)

    mano_device, mano_device_specs, writer_workers = resolve_mano_runtime(args, writer_workers)
    precompute_started_at = time.perf_counter()
    precompute_stats = precompute_episode_features(episode_stats, mano_device_specs, args, feature_cache_dir)
    precompute_elapsed = time.perf_counter() - precompute_started_at
    feature_cache_precomputed = bool(feature_cache_dir and args.precompute_features)
    if precompute_stats is not None and precompute_stats["episodes_failed"] > 0:
        raise RuntimeError(
            f"Episode feature precompute failed for {precompute_stats['episodes_failed']} episode(s); aborting build"
        )
    if feature_cache_precomputed:
        mano_device, mano_device_specs, writer_workers = resolve_writer_runtime(writer_workers, True)
    print_writer_config(writer_workers, mano_device_specs)
    build_started_at = time.perf_counter()
    totals = run_shard_writers(
        shard_tasks,
        writer_workers,
        mano_device,
        mano_device_specs,
        args,
        feature_cache_dir,
    )
    build_elapsed = time.perf_counter() - build_started_at
    print_summary(args.output_dir, totals)
    if precompute_stats is not None:
        print(
            "Feature cache:"
            f" ok={precompute_stats['episodes_ok']}"
            f" failed={precompute_stats['episodes_failed']}"
            f" frames={precompute_stats['frames_cached']}"
            f" batches={precompute_stats['batches']}"
            f" workers={precompute_stats['workers']}"
            f" elapsed={precompute_elapsed:.1f}s"
        )
    print(
        f"Timings: prepare={prepare_elapsed:.1f}s build={build_elapsed:.1f}s total={time.perf_counter() - started_at:.1f}s"
    )


if __name__ == "__main__":
    main()
