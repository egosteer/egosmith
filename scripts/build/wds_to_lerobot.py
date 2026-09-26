#!/usr/bin/env python3
"""Convert EgoSmith WebDataset shards (build-stage output) into a LeRobot v3.0 dataset.

Example::

    python scripts/build/wds_to_lerobot.py \
        --wds_dir /data/egosmith/final_dataset \
        --output_dir /data/egosmith/lerobot/egosmith_v1 \
        --fps 30 --workers 8 --validate

Depth (``*.depth.npy``) is never exported. See ``docs/dataset_format.md`` ("LeRobot export") for
the feature layout and the coordinate-frame convention.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# src-layout: first-party packages live under src/; scripts/ stays importable from root.
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def get_parser() -> argparse.ArgumentParser:
    from lib.pipeline.exporters.lerobot_export import (
        DEFAULT_CHUNKS_SIZE,
        DEFAULT_DATA_FILES_SIZE_IN_MB,
        DEFAULT_VIDEO_FILES_SIZE_IN_MB,
        HAND_FRAMES,
        STATE_LAYOUTS,
        TASK_SOURCES,
    )

    parser = argparse.ArgumentParser(description="Convert EgoSmith WebDataset shards to LeRobot v3.0 (no depth)")
    parser.add_argument("--wds_dir", type=str, nargs="+", required=True, help="Shard directories and/or .tar files")
    parser.add_argument("--output_dir", type=str, required=True, help="LeRobot dataset root to create")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate of the shard frames (build target_fps)")
    parser.add_argument("--robot_type", type=str, default="egosmith_human", help="info.json robot_type")
    parser.add_argument(
        "--state_layout",
        choices=STATE_LAYOUTS,
        default="egosteer74",
        help="egosteer74: EgoSteer 74-d (robot dims [0:26] padded); hawor48: hand block only",
    )
    parser.add_argument("--pad_value", type=float, default=0.0, help="Fill value for the padded robot dims (egosteer74)")
    parser.add_argument(
        "--hand_frame",
        choices=HAND_FRAMES,
        default="world",
        help="world (default): keep the SLAM world frame (per-frame w2c exported alongside); camera: express state/action in the current frame's head camera (EgoSteer world==head cam)",
    )
    parser.add_argument("--task_source", choices=TASK_SOURCES, default="dataset_name", help="Which meta field becomes the LeRobot task (default dataset_name: downstream tools read tasks[0] back as the dataset name; the sentences stay in instructions/language)")
    parser.add_argument("--task_name", type=str, default=None, help="Use this task / dataset name for every episode (overrides --task_source)")
    parser.add_argument("--split_order", type=str, default="train,val", help="Episode ordering by split; other splits follow")
    parser.add_argument("--no_mano", action="store_true", help="Drop the observation.mano (110-d) feature")
    parser.add_argument("--mano_policy", choices=("auto", "require"), default="auto", help="auto: no mano.npy anywhere -> no MANO feature; some episodes lack it -> skip those (recorded in provenance); require: fail before writing anything")
    parser.add_argument("--no_extrinsic", action="store_true", help="Drop the per-frame world->camera feature (only with --hand_frame camera; rejected with world)")
    parser.add_argument("--no_presence", action="store_true", help="Drop the observation.hand_presence feature")
    parser.add_argument("--no_video", action="store_true", help="Labels-only dataset: no observation.images.head, no ffmpeg needed")
    parser.add_argument("--descriptor_manifest", type=str, default=None, help="Frozen clip manifest JSONL; writes meta/source_frames.parquet (rehydration index)")
    parser.add_argument("--source_map", type=str, default=None, help="Source-map parquet (episode key -> original media + frame offset); adds source_frame_index/source_frame_observed and the source_media* episode columns")
    parser.add_argument("--source_map_policy", choices=("drop", "keep"), default="drop", help="Episodes without a usable source-map row: drop them (default) or keep with source_frame_index=-1")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg", help="ffmpeg binary (needs libx264)")
    parser.add_argument("--ffmpeg_threads", type=int, default=4)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--gop", type=int, default=15)
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--chunks_size", type=int, default=DEFAULT_CHUNKS_SIZE)
    parser.add_argument("--data_files_size_in_mb", type=int, default=DEFAULT_DATA_FILES_SIZE_IN_MB)
    parser.add_argument("--video_files_size_in_mb", type=int, default=DEFAULT_VIDEO_FILES_SIZE_IN_MB)
    parser.add_argument("--video_size_ratio", type=float, default=1.0, help="Planning estimate: h264 bytes per source JPEG byte")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes (testing)")
    parser.add_argument("--workers", type=int, default=0, help="Parallel file pairs, each with its own ffmpeg (0 = auto: cpu_count // ffmpeg_threads)")
    parser.add_argument("--resume", action="store_true", help="Skip file groups whose results already exist")
    parser.add_argument("--overwrite", action="store_true", help="Delete a non-empty output_dir first")
    parser.add_argument("--validate", action="store_true", help="Run structural checks (and the lerobot loader if installed) afterwards")
    parser.add_argument("--validate_only", action="store_true", help="Only validate an existing output_dir")
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser


def main() -> int:
    args = get_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    from lib.pipeline.exporters.lerobot_export import ConvertConfig, _ffprobe_for, convert_wds_to_lerobot, validate_lerobot_dataset

    output_dir = Path(args.output_dir)
    if not args.validate_only:
        if args.overwrite and output_dir.exists():
            shutil.rmtree(output_dir)
        cfg = ConvertConfig(
            fps=args.fps,
            robot_type=args.robot_type,
            state_layout=args.state_layout,
            hand_frame=args.hand_frame,
            pad_value=args.pad_value,
            task_source=args.task_source,
            task_name=args.task_name,
            split_order=tuple(s.strip() for s in args.split_order.split(",") if s.strip()),
            include_mano=not args.no_mano,
            mano_policy=args.mano_policy,
            include_extrinsic=not args.no_extrinsic,
            include_presence=not args.no_presence,
            include_video=not args.no_video,
            descriptor_manifest=args.descriptor_manifest,
            source_map=args.source_map,
            source_map_policy=args.source_map_policy,
            ffmpeg=args.ffmpeg,
            ffmpeg_threads=args.ffmpeg_threads,
            crf=args.crf,
            gop=args.gop,
            preset=args.preset,
            chunks_size=args.chunks_size,
            data_files_size_in_mb=args.data_files_size_in_mb,
            video_files_size_in_mb=args.video_files_size_in_mb,
            video_size_ratio=args.video_size_ratio,
            max_episodes=args.max_episodes,
            workers=args.workers,
            resume=args.resume,
        )
        summary = convert_wds_to_lerobot(args.wds_dir, str(output_dir), cfg)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.validate or args.validate_only:
        report = validate_lerobot_dataset(str(output_dir), ffprobe=_ffprobe_for(args.ffmpeg), ffmpeg=args.ffmpeg)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
