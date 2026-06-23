"""Per-stage output validation (full + fast), used to decide stage completeness.

Each stage has a strict validator (raises on bad output) and a fast validator (cheap existence
check returning bool). Registered in two dispatch dicts and exposed via validate_stage_output[_fast].
"""

import sys
from pathlib import Path

import numpy as np

from .stage_paths import get_tracks_dir


def _cleanup_incomplete_motion_output(seq_folder: Path, start_idx: int, end_idx: int):
    tracks_dir = get_tracks_dir(seq_folder, start_idx, end_idx)
    frame_chunks_file = tracks_dir / "frame_chunks_all.npy"
    model_masks_file = tracks_dir / "model_masks.npy"

    if frame_chunks_file.exists() and not model_masks_file.exists():
        print(f"Warning: Incomplete motion output detected for {seq_folder}", file=sys.stderr)
        print("  - frame_chunks_all.npy exists but model_masks.npy missing", file=sys.stderr)
        print("  - Removing incomplete output to force re-run", file=sys.stderr)
        frame_chunks_file.unlink()
        raise AssertionError("Incomplete motion output - removed and will retry")


def _validate_detect_track_output(seq_folder: Path, start_idx: int, end_idx: int):
    tracks_dir = get_tracks_dir(seq_folder, start_idx, end_idx)
    assert (tracks_dir / "model_boxes.npy").exists(), "model_boxes.npy missing"
    assert (tracks_dir / "model_tracks.npy").exists(), "model_tracks.npy missing"


def _validate_motion_output(seq_folder: Path, start_idx: int, end_idx: int):
    _cleanup_incomplete_motion_output(seq_folder, start_idx, end_idx)

    tracks_dir = get_tracks_dir(seq_folder, start_idx, end_idx)
    frame_chunks_file = tracks_dir / "frame_chunks_all.npy"
    model_masks_file = tracks_dir / "model_masks.npy"
    assert frame_chunks_file.exists(), "frame_chunks_all.npy missing"
    assert model_masks_file.exists(), "model_masks.npy missing"
    with open(model_masks_file, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        shape, _fortran, _dtype = np.lib.format._read_array_header(handle, version)
    assert len(shape) == 3, f"model_masks should be (T,H,W), got shape {shape}"


def _validate_slam_output(seq_folder: Path, start_idx: int, end_idx: int):
    slam_file = seq_folder / "SLAM" / f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    assert slam_file.exists(), "SLAM npz missing"
    data = np.load(slam_file, allow_pickle=True)
    assert "traj" in data and "scale" in data, "invalid SLAM npz keys"
    traj = np.asarray(data["traj"])
    scale = np.asarray(data["scale"])
    assert np.isfinite(traj).all(), "traj contains non-finite values"
    assert np.isfinite(scale).all(), "scale contains non-finite values"


def _validate_infiller_output(seq_folder: Path, _start_idx: int, _end_idx: int):
    from lib.pipeline.io.result_io import final_artifact_exists, load_pose_arrays

    assert final_artifact_exists(seq_folder), "final result (result.npz or world_space_res.pth) missing"
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = load_pose_arrays(seq_folder)
    assert pred_trans.shape[0] == 2 and pred_trans.shape[-1] == 3, "pred_trans shape invalid"
    assert pred_rot.shape[0] == 2 and pred_rot.shape[-1] == 3, "pred_rot shape invalid"
    assert pred_hand_pose.shape[0] == 2 and pred_hand_pose.shape[-1] == 45, "pred_hand_pose shape invalid"
    assert pred_betas.shape[0] == 2 and pred_betas.shape[-1] == 10, "pred_betas shape invalid"
    assert pred_valid.shape[0] == 2, "pred_valid shape invalid"
    frame_count = pred_trans.shape[1]
    assert pred_rot.shape[1] == frame_count, "pred_rot frame count invalid"
    assert pred_hand_pose.shape[1] == frame_count, "pred_hand_pose frame count invalid"
    assert pred_betas.shape[1] == frame_count, "pred_betas frame count invalid"
    assert pred_valid.shape[1] == frame_count, "pred_valid frame count invalid"
    assert np.isfinite(np.asarray(pred_trans)).all(), "pred_trans contains non-finite values"
    assert np.isfinite(np.asarray(pred_rot)).all(), "pred_rot contains non-finite values"
    assert np.isfinite(np.asarray(pred_hand_pose)).all(), "pred_hand_pose contains non-finite values"
    assert np.isfinite(np.asarray(pred_betas)).all(), "pred_betas contains non-finite values"
    assert np.isfinite(np.asarray(pred_valid)).all(), "pred_valid contains non-finite values"


def _validate_detect_track_output_fast(seq_folder: Path, start_idx: int, end_idx: int):
    tracks_dir = get_tracks_dir(seq_folder, start_idx, end_idx)
    return (tracks_dir / "model_boxes.npy").exists() and (tracks_dir / "model_tracks.npy").exists()


def _validate_motion_output_fast(seq_folder: Path, start_idx: int, end_idx: int):
    tracks_dir = get_tracks_dir(seq_folder, start_idx, end_idx)
    return (tracks_dir / "frame_chunks_all.npy").exists() and (tracks_dir / "model_masks.npy").exists()


def _validate_slam_output_fast(seq_folder: Path, start_idx: int, end_idx: int):
    slam_file = seq_folder / "SLAM" / f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    return slam_file.exists()


def _validate_infiller_output_fast(seq_folder: Path, _start_idx: int, _end_idx: int):
    from lib.pipeline.io.result_io import final_artifact_exists

    return final_artifact_exists(seq_folder)


_STAGE_VALIDATORS = {
    "detect_track": _validate_detect_track_output,
    "motion": _validate_motion_output,
    "slam": _validate_slam_output,
    "infiller": _validate_infiller_output,
}

_FAST_STAGE_VALIDATORS = {
    "detect_track": _validate_detect_track_output_fast,
    "motion": _validate_motion_output_fast,
    "slam": _validate_slam_output_fast,
    "infiller": _validate_infiller_output_fast,
}


def validate_stage_output(stage: str, seq_folder: Path, start_idx: int, end_idx: int):
    validator = _STAGE_VALIDATORS.get(stage)
    if validator is None:
        raise ValueError(f"Unknown stage: {stage}")
    validator(seq_folder, start_idx, end_idx)


def validate_stage_output_fast(stage: str, seq_folder: Path, start_idx: int, end_idx: int):
    validator = _FAST_STAGE_VALIDATORS.get(stage)
    if validator is None:
        return False
    return validator(seq_folder, start_idx, end_idx)
