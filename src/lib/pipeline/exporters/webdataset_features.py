"""Feature extraction helpers for WebDataset export.

Facade module: the implementation is split into cohesive submodules for readability —

- ``camera_features``  — SLAM trajectory -> per-frame extrinsics + intrinsic
- ``mano_features``    — MANO model build/forward + wrist/hand joint-state features
- ``lowdim_assembly`` — assemble the 116-d lowdim vector / presence / exportable frame ids
- ``episode_cache``   — world-space prediction load + on-disk feature cache

This module keeps the ``load_episode_features`` orchestrator and the per-episode infiller
fallback, and re-exports the full public surface so existing imports keep working unchanged.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from lib.pipeline.hands.hand_depth_align import HandDepthAlignConfig
from lib.pipeline.proc.logging_setup import get_logger
from .mano_codec import build_mano_pca_frame_features

from .camera_features import (
    DEFAULT_INTRINSIC,
    InvalidCameraDataError,
    _ensure_finite_array,
    _load_episode_camera_features,
    _log_slam_warning,
)
from .episode_cache import (
    EPISODE_FEATURE_CACHE_VERSION,
    _effective_cache_version,
    _load_cached_episode_features,
    _load_world_space_prediction,
    _to_float_tensor,
    _write_episode_feature_cache,
)
from .lowdim_assembly import (
    LOWDIM_SIZE,
    _build_episode_data,
    _build_episode_data_from_known_frame_ids,
    _build_lowdim_features,
    _compute_presence_per_frame,
    _shift_next_frame_action,
    export_frame_count_with_action,
)
from .mano_features import (
    FINGERTIP_INDICES,
    _compute_hand_joints,
    _compute_hand_state,
    _compute_joint_states,
    _compute_wrist_state,
    build_mano_models,
    run_mano_forward,
)

_logger = get_logger("exporters.webdataset_features")

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def load_episode_features(
    ep,
    mano_right,
    mano_left,
    device,
    rescan_frame_index=False,
    feature_cache_dir=None,
    require_cache=False,
    mano_dir=None,
):
    """Load one episode and compute per-frame lowdim features."""
    crop_dir = ep["crop_dir"]
    world_res_path = os.path.join(crop_dir, "world_space_res.pth")
    extracted_dir = os.path.join(crop_dir, "extracted_images")

    # After precompute, shard writers call with require_cache=True; allow disk cache load
    # even if --rescan was used (precompute already rewrote .joblib for this run).
    if feature_cache_dir and (not rescan_frame_index or require_cache):
        cached = _load_cached_episode_features(ep, extracted_dir, feature_cache_dir)
        if cached is not None:
            return cached
        if require_cache:
            raise RuntimeError(f"Missing episode feature cache for {crop_dir}")

    if require_cache:
        raise RuntimeError(f"Feature cache mode requires --feature_cache for {crop_dir}")

    prediction = _load_world_space_prediction(ep, world_res_path)
    if prediction is None:
        return None

    pred_trans = prediction["pred_trans"]
    pred_rot = prediction["pred_rot"]
    pred_hand_pose = prediction["pred_hand_pose"]
    pred_betas = prediction["pred_betas"]
    pred_valid = prediction["pred_valid"]
    num_frames = int(pred_trans.shape[1])
    mano_all = build_mano_pca_frame_features(
        pred_hand_pose.cpu().numpy(),
        pred_betas.cpu().numpy(),
        mano_dir=mano_dir,
    )

    # Camera + presence are needed before joint states so the optional hand-depth
    # alignment can ray-scale the joints (it consumes per-frame extrinsics).
    try:
        extrinsics, intrinsic = _load_episode_camera_features(ep, num_frames)
    except InvalidCameraDataError as error:
        _log_slam_warning("invalid_camera_episode", f"Skip {ep['episode_id']}: invalid camera features: {error}")
        return None
    presence_per_frame = _compute_presence_per_frame(pred_valid, num_frames)

    align_ctx = {
        "cfg": HandDepthAlignConfig.from_env(),
        "extrinsics": extrinsics,
        "crop_dir": crop_dir,
        "presence": presence_per_frame,
    }
    wrist_state, hand_state = _compute_joint_states(
        pred_trans,
        pred_rot,
        pred_hand_pose,
        pred_betas,
        mano_right,
        mano_left,
        device,
        align_ctx=align_ctx,
    )
    lowdim_all = _build_lowdim_features(wrist_state, hand_state, extrinsics, intrinsic)

    if not rescan_frame_index and ep.get("frame_ids"):
        episode_data = _build_episode_data_from_known_frame_ids(
            extracted_dir,
            ep["frame_ids"],
            lowdim_all,
            presence_per_frame,
        )
    else:
        episode_data = _build_episode_data(
            extracted_dir,
            num_frames,
            lowdim_all,
            presence_per_frame,
            rescan_frame_index=rescan_frame_index,
        )
    if episode_data is None:
        return None
    episode_data["mano_all"] = mano_all

    _write_episode_feature_cache(ep, feature_cache_dir, episode_data)
    return episode_data


def run_infill_for_episode(crop_dir, checkpoint, infiller_weight, device):
    """Run infiller as a small-scale fallback for a single episode."""
    seq_folder = Path(crop_dir)
    world_res = seq_folder / "world_space_res.pth"
    if world_res.exists():
        return True

    gpu = "" if str(device).startswith("cpu") else (device.split(":")[-1] if ":" in device else device)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(f"{seq_folder}\n")
        video_list_path = handle.name

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "batch_worker.py"),
        "--stage",
        "infiller",
        "--video_list",
        video_list_path,
        "--gpu",
        str(gpu),
        "--checkpoint",
        checkpoint,
        "--infiller_weight",
        infiller_weight,
    ]
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    finally:
        if os.path.exists(video_list_path):
            os.remove(video_list_path)
    produced = world_res.exists()
    if result.returncode != 0 or not produced:
        # Visible at ERROR even under quiet mode. Return real success so the caller
        # can count/skip failures instead of silently treating them as done.
        _logger.error(
            "Infill failed for %s (returncode=%s, world_space_res produced=%s):\n%s",
            seq_folder.name, result.returncode, produced, result.stdout,
        )
        return False
    return True
