#!/usr/bin/env python3
"""Re-attach the video stream to a labels-only LeRobot dataset.

    python scripts/build/lerobot_rehydrate_video.py \
        --labels_root /data/lerobot/egosmith_ego4d_labels \
        --frames_root /data/ego4d_frames \
        --output_dir /data/lerobot/egosmith_ego4d_full

``frames_root/<clip_id>/<frame_name>`` must hold the JPEG frames listed in
``<labels_root>/meta/source_frames.parquet`` (extract them from the original videos with the same
fps / clipping as the EgoSmith prepare stage). Use ``--layout clip_name`` when the folders are named
after the source clip instead of the clip id. Afterwards run the normal verification
and can be checked with the converter's built-in validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rehydrate a labels-only LeRobot dataset with video from source frames")
    parser.add_argument("--labels_root", type=str, required=True)
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layout", choices=("clip_id", "clip_name"), default="clip_id")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg")
    parser.add_argument("--ffmpeg_threads", type=int, default=4)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--gop", type=int, default=15)
    parser.add_argument("--preset", type=str, default="medium")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from lib.pipeline.exporters.lerobot_rehydrate import rehydrate_video

    summary = rehydrate_video(args.labels_root, args.frames_root, args.output_dir, layout=args.layout, ffmpeg=args.ffmpeg, ffmpeg_threads=args.ffmpeg_threads, crf=args.crf, gop=args.gop, preset=args.preset)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
