"""World-space prediction loading and the on-disk per-episode feature cache.

The cache version is augmented with the hand-depth-align signature so toggling alignment never
reuses a stale cache.
"""

import os

import joblib
import numpy as np
import torch

from lib.pipeline.hands.hand_depth_align import HandDepthAlignConfig
from lib.pipeline.io import result_io
from .webdataset_discovery import get_episode_feature_cache_path, load_or_build_frame_index

EPISODE_FEATURE_CACHE_VERSION = 9


def _effective_cache_version():
    """Cache version augmented with the hand-depth-align signature so that turning
    alignment on/off (or changing its params) never reuses a stale feature cache."""
    sig = HandDepthAlignConfig.from_env().signature()
    return EPISODE_FEATURE_CACHE_VERSION if sig == "off" else f"{EPISODE_FEATURE_CACHE_VERSION}:hda:{sig}"


def _to_float_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.tensor(np.array(value), dtype=torch.float32)


def _load_cached_episode_features(ep, extracted_dir, feature_cache_dir):
    if not feature_cache_dir:
        return None

    cache_path = get_episode_feature_cache_path(ep, feature_cache_dir)
    if not os.path.exists(cache_path):
        return None

    try:
        cached = joblib.load(cache_path)
    except Exception:
        return None

    if cached.get("cache_version") != _effective_cache_version() or cached.get("crop_dir") != ep["crop_dir"]:
        return None

    frame_index = load_or_build_frame_index(extracted_dir, rescan=False)
    if not frame_index:
        return None

    return {
        "frame_index": frame_index,
        "frame_ids": cached["frame_ids"],
        "lowdim_all": cached["lowdim_all"],
        "mano_all": cached["mano_all"],
        "presence_per_frame": cached["presence_per_frame"],
    }


def _load_world_space_prediction(ep, world_res_path):
    try:
        # Prefers the consolidated result.npz; falls back to legacy world_space_res.pth.
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = result_io.load_pose_arrays(world_res_path)
    except Exception as error:
        print(f"  Skip {ep['episode_id']}: failed to load world-space prediction: {error}")
        return None

    return {
        "pred_trans": _to_float_tensor(pred_trans),
        "pred_rot": _to_float_tensor(pred_rot),
        "pred_hand_pose": _to_float_tensor(pred_hand_pose),
        "pred_betas": _to_float_tensor(pred_betas),
        "pred_valid": pred_valid,
    }


def _write_episode_feature_cache(ep, feature_cache_dir, episode_data):
    if not feature_cache_dir:
        return

    cache_path = get_episode_feature_cache_path(ep, feature_cache_dir)
    cache_tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    cache_payload = {
        "cache_version": _effective_cache_version(),
        "crop_dir": ep["crop_dir"],
        "frame_ids": episode_data["frame_ids"],
        "lowdim_all": episode_data["lowdim_all"],
        "mano_all": episode_data["mano_all"],
        "presence_per_frame": episode_data["presence_per_frame"].astype(np.uint8),
    }
    try:
        os.makedirs(feature_cache_dir, exist_ok=True)
        joblib.dump(cache_payload, cache_tmp_path)
        os.replace(cache_tmp_path, cache_path)
    except OSError:
        if os.path.exists(cache_tmp_path):
            os.remove(cache_tmp_path)
