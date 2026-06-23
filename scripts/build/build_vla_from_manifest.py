#!/usr/bin/env python3
"""Build final VLA WebDataset from a frozen clip manifest snapshot."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Advanced helper: build VLA dataset from frozen clip manifest + HaWoR outputs"
    )
    parser.add_argument("--descriptor_manifest", type=str, required=True, help="Frozen clip manifest JSONL path")
    parser.add_argument("--output_dir", type=str, required=True, help="Final WebDataset output directory")
    parser.add_argument("--annotation_root", type=str, default=None, help="Clip annotation sidecar directory")
    parser.add_argument(
        "--annotation_suffix",
        type=str,
        default=".annotation.json",
        help="Annotation sidecar suffix, e.g. .annotation.json or _qwen-annotation.json",
    )
    parser.add_argument("--require_annotation", action="store_true", help="Drop clips with missing or invalid annotations")
    parser.add_argument(
        "--annotation_issue_report_out",
        type=str,
        default=None,
        help="Optional JSON path for missing/invalid annotation report; defaults to <output_dir>/_annotation_issues.json when issues exist",
    )
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for testing")
    parser.add_argument("--repeat_episodes", type=int, default=1, help="Repeat the manifest entries this many times")
    parser.add_argument("--preprocess_workers", type=int, default=8, help="Workers for manifest preparation")
    parser.add_argument("--writer_workers", type=int, default=4, help="Workers for shard writing")
    parser.add_argument("--frames_per_shard", type=int, default=10000, help="Approximate frame budget per shard")
    parser.add_argument("--mano_device", type=str, default="cuda:0", help="Device for MANO forward pass")
    parser.add_argument("--mano_gpus", type=str, default=None, help="Optional comma-separated GPU list for MANO workers")
    parser.add_argument("--mano_dir", type=str, default=None, help="Optional MANO model directory")
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default=None,
        help="Optional shared feature cache directory for precomputed lowdim/MANO episode features",
    )
    parser.add_argument("--source_fps", type=float, default=5.0, help="FPS of the existing stage outputs referenced by seq_folder")
    parser.add_argument("--target_fps", type=float, default=30.0, help="FPS of the RGB frames referenced by descriptor_manifest")
    parser.add_argument(
        "--interpolate_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpolate source labels onto descriptor frames instead of truncating to the source sequence length",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip non-empty shard tar files that already exist in output_dir",
    )
    parser.add_argument(
        "--export_depth",
        action="store_true",
        help="Export per-frame depth as {key}.depth.npy encoded as uint16 millimeters when seq_folder depth artifacts exist",
    )
    return parser


def main():
    args = get_parser().parse_args()
    from lib.pipeline.exporters.manifest_vla import run_manifest_build

    result = run_manifest_build(
        manifest_path=args.descriptor_manifest,
        output_dir=args.output_dir,
        annotation_root=args.annotation_root,
        annotation_suffix=args.annotation_suffix,
        require_annotation=args.require_annotation,
        max_episodes=args.max_episodes,
        repeat_episodes=args.repeat_episodes,
        preprocess_workers=args.preprocess_workers,
        writer_workers=args.writer_workers,
        frames_per_shard=args.frames_per_shard,
        mano_device=args.mano_device,
        mano_gpus=args.mano_gpus,
        mano_dir=args.mano_dir,
        feature_cache_dir=args.feature_cache_dir,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        interpolate_labels=args.interpolate_labels,
        export_depth=args.export_depth,
        annotation_issue_report_out=args.annotation_issue_report_out,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
