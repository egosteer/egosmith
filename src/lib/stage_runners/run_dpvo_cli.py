"""
Standalone CLI: run DPVO on one video and write ``dpvo_raw_*.npz`` cache.

This is **not** a wrapper around ``subprocess`` — same as ``hawor_slam`` DPVO branch,
which calls ``run_dpvo_slam`` in-process. Use this script when you want to precompute
or debug DPVO outside the full pipeline.

Npz timing keys: ``dpvo_vo_wall_sec`` (preferred) and ``dpvo_subprocess_sec`` (legacy
alias, same value) for compatibility with older caches and ``hawor_slam`` loaders.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# put src/ (where the first-party packages live) on sys.path so `import lib.*` resolves.
# src/lib/stage_runners/run_dpvo_cli.py -> parents[2] is src/ (parents[3] would be the repo root).
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[2]  # .../EgoSmith/src
PROJECT_ROOT = SRC_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.io.frame_source import build_frame_source  # noqa: E402
from lib.pipeline.slam.slam_geom_utils import est_calib  # noqa: E402
from lib.pipeline.slam.dpvo_slam import run_dpvo_slam  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Run DPVO SLAM for a single video (writes npz cache, same format as hawor_slam)."
    )
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    video_root = os.path.dirname(args.video_path)
    video = os.path.basename(args.video_path).split(".")[0]
    video_folder = os.path.join(video_root, video)
    tracks_dir = os.path.join(
        video_folder, f"tracks_{args.start_idx}_{args.end_idx}"
    )

    # Load masks from detect_track output
    masks_path = os.path.join(tracks_dir, "model_masks.npy")
    masks = np.load(masks_path, allow_pickle=True)
    masks = torch.from_numpy(masks)

    # Build frame source and calibration (same convention as masked_droid_slam)
    frame_source = build_frame_source(args.video_path)
    calib = np.array(est_calib(frame_source))

    t0 = time.time()
    traj, disps, tstamp, tstamp_disps = run_dpvo_slam(
        frame_source, masks=masks, calib=calib
    )
    dpvo_vo_wall_sec = time.time() - t0
    _sec_arr = np.array([dpvo_vo_wall_sec], dtype=np.float64)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez(
        args.output_path,
        tstamp=tstamp,
        disps=disps,
        traj=traj,
        tstamp_disps=tstamp_disps,
        dpvo_vo_wall_sec=_sec_arr,
        dpvo_subprocess_sec=_sec_arr,
    )


if __name__ == "__main__":
    main()
