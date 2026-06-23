"""Feature cache helpers for manifest-based build/export."""

from __future__ import annotations

import hashlib
import os

import joblib
import numpy as np

MANIFEST_FEATURE_CACHE_VERSION = 11


def feature_cache_path(seq_folder: str, feature_cache_dir: str) -> str:
    digest = hashlib.md5(seq_folder.encode("utf-8")).hexdigest()
    return os.path.join(feature_cache_dir, f"{digest}.joblib")


def load_cached_features(
    seq_folder: str,
    frame_count: int,
    feature_cache_dir: str,
    *,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
):
    if not feature_cache_dir:
        return None
    path = feature_cache_path(seq_folder, feature_cache_dir)
    if not os.path.exists(path):
        return None
    try:
        payload = joblib.load(path)
    except Exception:
        return None
    if (
        payload.get("cache_version") != MANIFEST_FEATURE_CACHE_VERSION
        or payload.get("seq_folder") != seq_folder
        or payload.get("frame_count") != frame_count
        or float(payload.get("source_fps", -1.0)) != float(source_fps)
        or float(payload.get("target_fps", -1.0)) != float(target_fps)
        or bool(payload.get("interpolate_labels", False)) != bool(interpolate_labels)
    ):
        return None
    return {
        "frame_count": payload["frame_count"],
        "lowdim_all": payload["lowdim_all"],
        "mano_all": payload["mano_all"],
        "presence_per_frame": payload["presence_per_frame"],
    }


def write_cached_features(
    seq_folder: str,
    feature_cache_dir: str,
    episode_data: dict,
    *,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
) -> None:
    if not feature_cache_dir:
        return
    os.makedirs(feature_cache_dir, exist_ok=True)
    path = feature_cache_path(seq_folder, feature_cache_dir)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    payload = {
        "cache_version": MANIFEST_FEATURE_CACHE_VERSION,
        "seq_folder": seq_folder,
        "frame_count": episode_data["frame_count"],
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "interpolate_labels": bool(interpolate_labels),
        "lowdim_all": episode_data["lowdim_all"],
        "mano_all": episode_data["mano_all"],
        "presence_per_frame": episode_data["presence_per_frame"].astype(np.uint8),
    }
    try:
        joblib.dump(payload, tmp_path)
        os.replace(tmp_path, path)
    except OSError:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
