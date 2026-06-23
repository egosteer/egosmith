#!/usr/bin/env python3
"""Validate manifest, stage outputs, annotations, and final dataset shards."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def get_parser():
    parser = argparse.ArgumentParser(description="Validate a whole-pipeline run")
    parser.add_argument("--descriptor_manifest", type=str, required=True, help="Frozen clip manifest JSONL")
    parser.add_argument("--annotation_root", type=str, default=None, help="Clip annotation sidecar directory")
    parser.add_argument(
        "--annotation_suffix",
        type=str,
        default=".annotation.json",
        help="Annotation sidecar suffix, e.g. .annotation.json or _qwen-annotation.json",
    )
    parser.add_argument("--dataset_dir", type=str, default=None, help="Final dataset shard directory")
    parser.add_argument(
        "--stages",
        type=str,
        default="detect_track,motion,slam,infiller",
        help="Comma-separated stages to validate",
    )
    parser.add_argument("--max_clips", type=int, default=None, help="Limit manifest clips for quick validation; <=0 means all clips")
    parser.add_argument(
        "--dataset_sample_checks",
        type=int,
        default=0,
        help="How many output samples to inspect for dataset sanity; <=0 means full dataset scan",
    )
    parser.add_argument(
        "--decode_images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Decode JPEGs during dataset sanity checks",
    )
    parser.add_argument(
        "--allow_empty_instruction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not fail dataset sanity when instruction/language fields are empty.",
    )
    parser.add_argument(
        "--require_depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail dataset sanity when built samples are missing depth payloads.",
    )
    parser.add_argument(
        "--depth_action_consistency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Summarize projected hand-depth versus exported depth maps when depth payloads exist.",
    )
    parser.add_argument(
        "--depth_action_report_out",
        type=str,
        default=None,
        help="Optional JSON path for the depth/action consistency summary.",
    )
    return parser


def validate_manifest_outputs(records, stages):
    from lib.pipeline.exporters.manifest_vla import descriptor_uses_native_features, load_descriptor_episode_features, prepare_manifest_record_for_build
    from lib.pipeline.slam.native_depth import validate_native_depth_output
    from lib.pipeline.proc.stage_api import get_track_range, validate_stage_output

    stats = {
        "clips_total": len(records),
        "clips_ok": 0,
        "clips_failed": 0,
        "stage_failures": {stage: 0 for stage in stages},
        "failure_examples": [],
        "track_range_failures": [],
    }

    for record in records:
        seq_folder = Path(record.descriptor.seq_folder)
        clip_ok = True
        if descriptor_uses_native_features(record.descriptor):
            if any(stage not in ("native_features", "native_depth") for stage in stages):
                stats["clips_failed"] += 1
                if len(stats["failure_examples"]) < 64:
                    stats["failure_examples"].append(
                        {
                            "clip_id": record.clip_id,
                            "seq_folder": str(seq_folder),
                            "stage": ",".join(stages),
                            "error": "native feature descriptors require --stages native_features",
                        }
                    )
                continue
            if "native_depth" in stages:
                try:
                    validate_native_depth_output(
                        record.descriptor.seq_folder,
                        expected_frame_count=int(record.descriptor.frame_count),
                    )
                except Exception as error:
                    stats["stage_failures"].setdefault("native_depth", 0)
                    stats["stage_failures"]["native_depth"] += 1
                    stats["clips_failed"] += 1
                    if len(stats["failure_examples"]) < 64:
                        stats["failure_examples"].append(
                            {
                                "clip_id": record.clip_id,
                                "seq_folder": str(seq_folder),
                                "stage": "native_depth",
                                "error": str(error),
                            }
                        )
                    continue
            episode, error_code = prepare_manifest_record_for_build(
                record,
                require_annotation=False,
                annotation_root=None,
                annotation_suffix=".annotation.json",
                source_fps=30.0,
                target_fps=30.0,
                interpolate_labels=False,
            )
            if episode is None or load_descriptor_episode_features(
                episode,
                None,
                None,
                None,
                feature_cache_dir=None,
                mano_dir=None,
                source_fps=30.0,
                target_fps=30.0,
                interpolate_labels=False,
            ) is None:
                stats["stage_failures"].setdefault("native_features", 0)
                stats["stage_failures"]["native_features"] += 1
                stats["clips_failed"] += 1
                if len(stats["failure_examples"]) < 64:
                    stats["failure_examples"].append(
                        {
                            "clip_id": record.clip_id,
                            "seq_folder": str(seq_folder),
                            "stage": "native_features",
                            "error": str(error_code or "invalid native features"),
                        }
                    )
                continue
            stats["clips_ok"] += 1
            continue
        try:
            start_idx, end_idx = get_track_range(seq_folder, fast=False)
        except Exception as error:
            for stage in stages:
                stats["stage_failures"][stage] += 1
            stats["clips_failed"] += 1
            if len(stats["track_range_failures"]) < 32:
                stats["track_range_failures"].append(
                    {
                        "clip_id": record.clip_id,
                        "seq_folder": str(seq_folder),
                        "error": str(error),
                    }
                )
            continue
        for stage in stages:
            try:
                validate_stage_output(stage, seq_folder, start_idx, end_idx)
            except Exception as error:
                stats["stage_failures"][stage] += 1
                clip_ok = False
                if len(stats["failure_examples"]) < 64:
                    stats["failure_examples"].append(
                        {
                            "clip_id": record.clip_id,
                            "seq_folder": str(seq_folder),
                            "stage": stage,
                            "error": str(error),
                        }
                    )
        if clip_ok:
            stats["clips_ok"] += 1
        else:
            stats["clips_failed"] += 1
    return stats


def validate_annotations(records, annotation_root, annotation_suffix):
    from lib.pipeline.clips.annotation_protocol import load_clip_annotation

    if not annotation_root:
        return None

    stats = {
        "valid": 0,
        "missing_annotation": 0,
        "invalid_json": 0,
        "invalid_status": 0,
        "empty_instruction": 0,
    }
    for record in records:
        annotation, error_code, _ = load_clip_annotation(
            annotation_root,
            record.clip_id,
            annotation_suffix=annotation_suffix,
        )
        if annotation is not None:
            stats["valid"] += 1
        else:
            stats[error_code] = stats.get(error_code, 0) + 1
    return stats


def validate_dataset(dataset_dir, sample_checks, *, decode_images: bool, allow_empty_instruction: bool, require_depth: bool):
    from lib.pipeline.quality.wds_sanity import analyze_webdataset

    if not dataset_dir:
        return None

    limit = None if sample_checks is None or int(sample_checks) <= 0 else int(sample_checks)
    report = analyze_webdataset(
        source_shard_dir=str(Path(dataset_dir)),
        sample_limit=limit,
        decode_images=bool(decode_images),
        allow_empty_instruction=bool(allow_empty_instruction),
        require_depth=bool(require_depth),
        max_issue_examples=64,
    )
    report["full_dataset_scan"] = limit is None
    return report


def validate_depth_action_consistency(dataset_dir, sample_checks, *, enabled: bool, report_out: str | None):
    if not dataset_dir or not enabled:
        return None
    from lib.pipeline.slam.depth_action_consistency import (
        analyze_depth_action_consistency,
        write_depth_action_consistency_report,
    )

    limit = None if sample_checks is None or int(sample_checks) <= 0 else int(sample_checks)
    report = analyze_depth_action_consistency(
        dataset_dir=str(Path(dataset_dir)),
        sample_limit=limit,
        max_examples=64,
    )
    if report_out:
        report["report_path"] = write_depth_action_consistency_report(report, report_out)
    return report


def summarize_validation_failures(summary: dict) -> list[str]:
    failures = []

    stage_stats = summary.get("stages") or {}
    if int(stage_stats.get("clips_failed", 0)) > 0:
        failures.append(f"stage validation failed for {stage_stats.get('clips_failed', 0)} clips")

    annotation_stats = summary.get("annotations")
    if annotation_stats:
        annotation_bad = sum(
            int(annotation_stats.get(key, 0))
            for key in ("missing_annotation", "invalid_json", "invalid_status", "empty_instruction")
        )
        if annotation_bad > 0:
            failures.append(f"annotation validation found {annotation_bad} problematic clips")

    dataset_stats = summary.get("dataset")
    if dataset_stats:
        checks = dataset_stats.get("checks") or {}
        dataset_bad = sum(int(value) for value in checks.values())
        if dataset_bad > 0:
            failures.append(f"dataset sanity found {dataset_bad} issues across built samples")

    return failures


def main():
    args = get_parser().parse_args()
    from lib.pipeline.clips.clip_manifest import load_clip_manifest

    records = load_clip_manifest(args.descriptor_manifest)
    if args.max_clips is not None and int(args.max_clips) > 0:
        records = records[: args.max_clips]

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    summary = {
        "manifest": {
            "path": str(Path(args.descriptor_manifest).resolve()),
            "clips_checked": len(records),
        },
        "stages": validate_manifest_outputs(records, stages),
        "annotations": validate_annotations(records, args.annotation_root, args.annotation_suffix),
        "dataset": validate_dataset(
            args.dataset_dir,
            args.dataset_sample_checks,
            decode_images=bool(args.decode_images),
            allow_empty_instruction=bool(args.allow_empty_instruction),
            require_depth=bool(args.require_depth),
        ),
        "depth_action_consistency": validate_depth_action_consistency(
            args.dataset_dir,
            args.dataset_sample_checks,
            enabled=bool(args.depth_action_consistency),
            report_out=args.depth_action_report_out,
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failures = summarize_validation_failures(summary)
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
