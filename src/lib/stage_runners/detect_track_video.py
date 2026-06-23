"""Standalone detection + tracking on a single video.

Related pipeline stage: lib/pipeline/stages/detect_track.py
This script is an independent executable with its own argument parsing.
"""
import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + '/../..')

from lib.pipeline.io.frame_source import build_frame_source
from lib.pipeline.hands.detect_track_batched import detect_track
from lib.pipeline.proc.logging_setup import QUIET_MODE, vprint  # noqa: F401


def detect_track_video(args, detector_runner=None, force=False, detect_batch_size=128, num_io_workers=8, device='cuda:0', half_precision=True):
    file = args.video_path
    root = os.path.dirname(file)
    seq = os.path.basename(file).split('.')[0]

    seq_folder = f'{root}/{seq}'
    os.makedirs(seq_folder, exist_ok=True)
    vprint(f'Running detect_track on {file} ...')

    frame_source = build_frame_source(file)

    ##### Detection + Track #####
    vprint('Detect and Track ...')

    start_idx = 0
    end_idx = len(frame_source)

    if (not force) and os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy'):
        vprint(f"skip track for {start_idx}_{end_idx}")
        return start_idx, end_idx, seq_folder, frame_source

    # Invalidate track range cache since we're (re)running detect_track
    cache_file = f'{seq_folder}/.track_range'
    if os.path.exists(cache_file):
        os.remove(cache_file)

    os.makedirs(f"{seq_folder}/tracks_{start_idx}_{end_idx}", exist_ok=True)
    boxes_, tracks_ = detect_track(
        frame_source,
        thresh=0.35,
        edge_margin_ratio=0.1,
        min_edge_conf=0.4,
        hand_det_model=detector_runner,
        detect_batch_size=detect_batch_size,
        num_io_workers=num_io_workers,
        device=device,
        half_precision=half_precision,
    )
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy', boxes_)
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy', tracks_)

    return start_idx, end_idx, seq_folder, frame_source


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--detect_batch_size", type=int, default=128)
    parser.add_argument("--detect_io_workers", type=int, default=8)
    args = parser.parse_args()

    detect_track_video(args, detect_batch_size=args.detect_batch_size, num_io_workers=args.detect_io_workers)
