"""Top-level manifest build orchestration."""

from __future__ import annotations

import os
from pathlib import Path
from multiprocessing import get_context

import torch
from tqdm import tqdm

from lib.pipeline.clips.annotation_protocol import write_annotation_issue_report

from .episodes import descriptor_uses_native_features, prepare_manifest_episodes
from .writer import (
    normalize_mano_devices,
    plan_manifest_shards,
    repeat_manifest_episodes,
    worker_init,
    worker_process_shard,
)


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
        raise RuntimeError(f"No valid manifest episodes found: {prepare_stats}")

    if export_depth:
        for episode in episodes:
            episode["export_depth"] = True

    repeated = repeat_manifest_episodes(episodes, repeat_episodes)
    shard_tasks = plan_manifest_shards(repeated, frames_per_shard, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    skip_mano_models = bool(repeated) and all(
        descriptor_uses_native_features(ep["descriptor"]) for ep in repeated
    )

    existing_shard_tasks = []
    pending_shard_tasks = shard_tasks
    if resume:
        existing_shard_tasks = [
            task
            for task in shard_tasks
            if os.path.exists(task["output_path"]) and os.path.getsize(task["output_path"]) > 0
        ]
        existing_output_paths = {task["output_path"] for task in existing_shard_tasks}
        pending_shard_tasks = [task for task in shard_tasks if task["output_path"] not in existing_output_paths]

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
            pool = mp_context.Pool(
                writer_workers,
                initializer=worker_init,
                initargs=(mano_device_specs, mano_dir, resolved_feature_cache_dir, skip_mano_models),
            )
            result_iter = pool.imap_unordered(worker_process_shard, pending_shard_tasks)

        try:
            for result in tqdm(result_iter, total=len(pending_shard_tasks), desc="Build shards"):
                totals["frames_written"] += result["frames_written"]
                totals["episodes_written"] += result["episodes_written"]
                totals["skipped_episodes"] += result["skipped_episodes"]
                totals["skipped_clips"] += result.get("skipped_clips", 0)
                totals["shards_written"] += 1 if result["frames_written"] > 0 else 0
                skipped_clip_details.extend(result.get("skipped_clip_details", []))
        finally:
            if writer_workers > 1:
                pool.close()
                pool.join()

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
