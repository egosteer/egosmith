"""Assemble the per-frame 116-d lowdim vector, presence mask, and exportable frame-id set."""

import os

import numpy as np
import torch

from .webdataset_discovery import load_or_build_frame_index

LOWDIM_SIZE = 116


def export_frame_count_with_action(num_frames: int) -> int:
    return max(int(num_frames) - 1, 0)


def _shift_next_frame_action(state):
    action = torch.zeros_like(state)
    action[:-1] = state[1:]
    return action


def _compute_presence_per_frame(pred_valid, num_frames):
    if isinstance(pred_valid, np.ndarray):
        valid = pred_valid.astype(np.float32)
    else:
        valid = pred_valid.float().cpu().numpy()
    if valid.ndim == 1:
        valid = np.tile(valid[:, None], (1, num_frames))
    return ((valid[0] > 0.5).astype(int)) | (((valid[1] > 0.5).astype(int)) << 1)


def _build_lowdim_features(wrist_state, hand_state, extrinsics, intrinsic):
    wrist_action = _shift_next_frame_action(wrist_state)
    hand_action = _shift_next_frame_action(hand_state)
    num_frames = int(wrist_state.shape[0])

    lowdim_all = np.concatenate(
        [
            wrist_state.cpu().numpy().astype(np.float32),
            hand_state.cpu().numpy().astype(np.float32),
            wrist_action.cpu().numpy().astype(np.float32),
            hand_action.cpu().numpy().astype(np.float32),
            extrinsics.reshape(num_frames, 16),
            np.tile(intrinsic, (num_frames, 1)),
        ],
        axis=-1,
    )
    assert lowdim_all.shape == (num_frames, LOWDIM_SIZE), f"lowdim shape mismatch: {lowdim_all.shape}"
    return lowdim_all


def _build_episode_data(extracted_dir, num_frames, lowdim_all, presence_per_frame, rescan_frame_index):
    frame_index = load_or_build_frame_index(extracted_dir, rescan=rescan_frame_index)
    export_frame_count = export_frame_count_with_action(num_frames)
    frame_ids = sorted(frame_idx for frame_idx in frame_index if frame_idx < export_frame_count)
    if not frame_ids:
        return None

    return {
        "frame_index": frame_index,
        "frame_ids": frame_ids,
        "lowdim_all": lowdim_all,
        "presence_per_frame": presence_per_frame,
    }


def _build_episode_data_from_known_frame_ids(extracted_dir, frame_ids, lowdim_all, presence_per_frame):
    export_frame_count = export_frame_count_with_action(int(lowdim_all.shape[0]))
    valid_frame_ids = [int(frame_idx) for frame_idx in frame_ids if int(frame_idx) < export_frame_count]
    if not valid_frame_ids:
        return None

    frame_index = {
        frame_idx: os.path.join(extracted_dir, f"{frame_idx}.jpg")
        for frame_idx in valid_frame_ids
    }
    return {
        "frame_index": frame_index,
        "frame_ids": valid_frame_ids,
        "lowdim_all": lowdim_all,
        "presence_per_frame": presence_per_frame,
    }
