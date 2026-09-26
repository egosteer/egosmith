"""Top-level manifest build orchestration."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from multiprocessing import get_context

import torch
from tqdm import tqdm

from lib.pipeline.clips.annotation_protocol import write_annotation_issue_report

from .cache import mano_asset_identity
from .episodes import descriptor_uses_native_features, prepare_manifest_episodes
from .writer import (
    normalize_mano_devices,
    plan_manifest_shards,
    repeat_manifest_episodes,
    shard_task_digest,
    worker_init,
    worker_process_shard,
)

# Records, per completed shard, the digest of what it was built from (see shard_task_digest). --resume
# reuses a shard only when that digest matches the current plan; a directory without this file
# (built before it existed) is rebuilt in full. Version 2 digests include the upstream artifact identities;
# a version-1 plan is treated as missing.
SHARD_PLAN_FILENAME = "_shard_plan.json"
SHARD_PLAN_VERSION = 2


def _load_shard_plan(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != SHARD_PLAN_VERSION or not isinstance(data.get("shards"), dict):
        return {}
    return data["shards"]


def _write_shard_plan(path: str, shards: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"version": SHARD_PLAN_VERSION, "shards": shards}, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _retire_stale_shard(path: str) -> str:
    """Rename (never delete) a shard that is not part of the current build to ``<name>.stale``."""
    target = f"{path}.stale"
    k = 1
    while os.path.exists(target):
        target = f"{path}.{k}.stale"
        k += 1
    os.replace(path, target)
    return target


def run_manifest_build(
    *,
    manifest_path: str,
    output_dir: str,
    annotation_root: str | None,
    annotation_suffix: str,
    require_annotation: bool,
    max_episodes: int | None,
    repeat_episodes: int,
    preprocess_workers: int,
    writer_workers: int,
    frames_per_shard: int,
    mano_device: str,
    mano_gpus: str | None,
    mano_dir: str | None,
    feature_cache_dir: str | None,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
    export_depth: bool = False,
    annotation_issue_report_out: str | None = None,
    resume: bool = False,
):
    episodes, prepare_stats, annotation_issues = prepare_manifest_episodes(
        manifest_path,
        annotation_root=annotation_root,
        annotation_suffix=annotation_suffix,
        require_annotation=require_annotation,
        max_episodes=max_episodes,
        preprocess_workers=preprocess_workers,
        source_fps=source_fps,
        target_fps=target_fps,
        interpolate_labels=interpolate_labels,
    )
    # Input shards living where this build writes would be overwritten or retired as stale mid-build.
    # Checked for the file itself and for its directory entry: a shard-*.tar symlink inside the
    # output dir that points elsewhere would still be renamed to .stale by the cleanup below.
    resolved_output_dir = Path(output_dir).resolve()
    colliding_inputs = sorted({
        str(shard_path)
        for episode in episodes
        if (shard_path := getattr(episode["descriptor"], "shard_path", None))
        and (
            Path(shard_path).resolve().parent == resolved_output_dir
            or Path(os.path.abspath(shard_path)).parent.resolve() == resolved_output_dir
        )
    })
    if colliding_inputs:
        raise ValueError(
            f"Input shard(s) are in the output directory {resolved_output_dir}; "
            f"input and output directories must differ: {', '.join(colliding_inputs[:5])}"
            + (f" (and {len(colliding_inputs) - 5} more)" if len(colliding_inputs) > 5 else "")
        )
    annotation_issue_report_path = None
    if annotation_root and (annotation_issues or annotation_issue_report_out):
        report_path = annotation_issue_report_out or os.path.join(output_dir, "_annotation_issues.json")
        annotation_issue_report_path = write_annotation_issue_report(
            report_path,
            annotation_root=annotation_root,
            annotation_suffix=annotation_suffix,
            issues=annotation_issues,
            context={
                "manifest_path": str(Path(manifest_path).resolve()),
                "output_dir": str(Path(output_dir).resolve()),
                "require_annotation": bool(require_annotation),
            },
        )
        if annotation_issues:
            print(
                "Warning: "
                f"{len(annotation_issues)} clip(s) have missing/invalid/empty annotations; "
                f"report written to {annotation_issue_report_path}"
            )

    if not episodes:
        hint = ""
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest_is_empty = not any(line.strip() for line in handle)
        except OSError:
            manifest_is_empty = False
        if manifest_is_empty:
            hint = (
                f" The manifest {manifest_path} lists no clips; if it is the filter stage's output, quality "
                "filtering dropped every clip (reasons in filter_report.json next to it)."
            )
        raise RuntimeError(f"No valid manifest episodes found: {prepare_stats}{hint}")

    if export_depth:
        for episode in episodes:
            episode["export_depth"] = True

    repeated = repeat_manifest_episodes(episodes, repeat_episodes)
    shard_tasks = plan_manifest_shards(repeated, frames_per_shard, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    skip_mano_models = bool(repeated) and all(
        descriptor_uses_native_features(ep["descriptor"]) for ep in repeated
    )

    # the MANO assets' content, not only the mano_dir string: replacing them in place changes lowdim/mano samples
    options_digest = hashlib.sha1(
        json.dumps({"mano_dir": mano_dir, "mano_assets": mano_asset_identity(mano_dir)}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    shard_digests = {task["output_path"]: shard_task_digest(task, options_digest) for task in shard_tasks}
    shard_plan_path = os.path.join(output_dir, SHARD_PLAN_FILENAME)
    existing_shard_tasks = []
    pending_shard_tasks = shard_tasks
    if resume:
        recorded = _load_shard_plan(shard_plan_path)
        if not recorded and any(os.path.exists(task["output_path"]) for task in shard_tasks):
            print(f"Warning: --resume found no usable {SHARD_PLAN_FILENAME} in {output_dir}; rebuilding every shard")
        existing_shard_tasks = [
            task
            for task in shard_tasks
            if recorded.get(os.path.basename(task["output_path"])) == shard_digests[task["output_path"]]
            and os.path.exists(task["output_path"])
            and os.path.getsize(task["output_path"]) > 0
        ]
        existing_output_paths = {task["output_path"] for task in existing_shard_tasks}
        pending_shard_tasks = [task for task in shard_tasks if task["output_path"] not in existing_output_paths]

    # Shards left by an earlier build with more shards would be read by every consumer (they collect *.tar).
    planned_paths = {os.path.abspath(task["output_path"]) for task in shard_tasks}
    for path in sorted(glob.glob(os.path.join(output_dir, "shard-*.tar"))):
        if os.path.abspath(path) not in planned_paths:
            print(f"Warning: {path} is not part of this build's shard plan; renamed to {_retire_stale_shard(path)}")
    # Only verified shards are recorded before building: a shard is added once it has been rewritten.
    completed_shards = {os.path.basename(task["output_path"]): shard_digests[task["output_path"]] for task in existing_shard_tasks}
    _write_shard_plan(shard_plan_path, completed_shards)

    resolved_feature_cache_dir = feature_cache_dir
    if resolved_feature_cache_dir is None and repeat_episodes > 1:
        resolved_feature_cache_dir = os.path.join(output_dir, "_episode_feature_cache")
    if resolved_feature_cache_dir:
        os.makedirs(resolved_feature_cache_dir, exist_ok=True)

    mano_device_obj = torch.device(mano_device if torch.cuda.is_available() else "cpu")
    mano_device_specs = normalize_mano_devices(str(mano_device_obj), mano_gpus if mano_device_obj.type == "cuda" else None)
    if mano_device_obj.type == "cuda" and len(mano_device_specs) == 1:
        writer_workers = min(writer_workers, 1)
    elif mano_device_obj.type == "cuda":
        writer_workers = min(writer_workers, len(mano_device_specs))

    totals = {
        "frames_written": 0,
        "episodes_written": 0,
        "skipped_episodes": 0,
        "skipped_clips": 0,
        "shards_written": 0,
        "shards_reused": len(existing_shard_tasks),
        "frames_reused": int(sum(task["frame_count"] for task in existing_shard_tasks)),
    }
    skipped_clip_details = []

    if pending_shard_tasks:
        if writer_workers <= 1:
            worker_init(mano_device_specs, mano_dir, resolved_feature_cache_dir, skip_mano_models)
            result_iter = (worker_process_shard(task) for task in pending_shard_tasks)
        else:
            mp_context = get_context("spawn") if mano_device_obj.type == "cuda" else get_context()
            # ProcessPoolExecutor, not multiprocessing.Pool: a worker killed by the OOM killer raises
            # BrokenProcessPool here instead of hanging the build forever.
            pool = ProcessPoolExecutor(
                max_workers=writer_workers,
                mp_context=mp_context,
                initializer=worker_init,
                initargs=(mano_device_specs, mano_dir, resolved_feature_cache_dir, skip_mano_models),
            )
            futures = [pool.submit(worker_process_shard, task) for task in pending_shard_tasks]
            result_iter = (future.result() for future in as_completed(futures))  # completion order, like imap_unordered

        try:
            for result in tqdm(result_iter, total=len(pending_shard_tasks), desc="Build shards"):
                if result["frames_written"] > 0:
                    completed_shards[os.path.basename(result["output_path"])] = shard_digests[result["output_path"]]
                    _write_shard_plan(shard_plan_path, completed_shards)
                elif os.path.exists(result["output_path"]):
                    # this build left the shard empty, so the file there is from an earlier plan
                    print(f"Warning: {result['output_path']} received no frames in this build; renamed to {_retire_stale_shard(result['output_path'])}")
                totals["frames_written"] += result["frames_written"]
                totals["episodes_written"] += result["episodes_written"]
                totals["skipped_episodes"] += result["skipped_episodes"]
                totals["skipped_clips"] += result.get("skipped_clips", 0)
                totals["shards_written"] += 1 if result["frames_written"] > 0 else 0
                skipped_clip_details.extend(result.get("skipped_clip_details", []))
        finally:
            if writer_workers > 1:
                pool.shutdown(wait=True, cancel_futures=True)

    return {
        "prepare_stats": prepare_stats,
        "totals": totals,
        "skipped_clip_details": skipped_clip_details,
        "annotation_issue_report_path": annotation_issue_report_path,
        "annotation_issue_count": len(annotation_issues),
        "planned_shards": len(shard_tasks),
        "pending_shards": len(pending_shard_tasks),
        "planned_frames": sum(ep["num_valid_frames"] for ep in repeated),
        "feature_cache_dir": resolved_feature_cache_dir,
    }
