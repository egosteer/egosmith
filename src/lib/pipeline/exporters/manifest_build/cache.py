"""Feature cache helpers for manifest-based build/export."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from lib.pipeline.exporters.mano_codec import _mano_pkl_path
from lib.pipeline.hands.hand_depth_align import HandDepthAlignConfig
from lib.pipeline.slam.depth_artifacts import _discover_track_range

MANIFEST_FEATURE_CACHE_VERSION = 11

# Upstream artifacts under a seq folder whose content the cached features are derived from.
_UPSTREAM_ARTIFACTS = ("world_space_res.pth", "result.npz")
# SLAM/*.npz written by the feature computation itself (diagnostic sidecars, not inputs): counting
# them would invalidate the entry that the same computation is about to write.
_SELF_WRITTEN_SLAM_PREFIXES = ("hand_depth_align_",)


def _file_identity(path: Path):
    try:
        stat = path.stat()
    except OSError:
        return None
    return [int(stat.st_size), int(stat.st_mtime_ns)]


@lru_cache(maxsize=16)
def _content_sha1(path: str, size: int, mtime_ns: int) -> str:
    """sha1 of a file's bytes; (size, mtime_ns) are part of the memo key so a rewritten file is re-hashed."""
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mano_asset_identity(mano_dir: str | None) -> dict:
    """(size, sha1) of the MANO_LEFT/RIGHT.pkl files the PCA codec and joint model read for ``mano_dir``.
    Content only, no path: replacing the assets in place changes it, moving the same files does not.
    An unresolvable MANO root (e.g. a native-features build without MANO) is recorded as None."""
    identity = {}
    for side in ("left", "right"):
        try:
            path = Path(_mano_pkl_path(side, mano_dir))
            stat = path.stat()
            identity[side] = [int(stat.st_size), _content_sha1(str(path), int(stat.st_size), int(stat.st_mtime_ns))]
        except (OSError, ValueError):
            identity[side] = None
    return identity


def _hand_mask_path(root: Path) -> Path | None:
    """The tracks_*/model_masks.npy hand-depth alignment reads for this seq folder (same lookup as
    hand_depth_align._load_hand_masks), or None when no track range is known."""
    track_range = _discover_track_range(root)
    if track_range is None:
        return None
    start, end = track_range
    return root / f"tracks_{start}_{end}" / "model_masks.npy"


def feature_cache_dependencies(seq_folder: str, *, extra_files: tuple = (), include_mano_assets: bool = False, mano_dir: str | None = None) -> dict:
    """Identity (size, mtime_ns) of every upstream artifact the features are computed from,
    plus the hand-depth-align settings. A cache entry is only valid while this is unchanged,
    so re-running infer/SLAM (same frame count) or changing alignment invalidates it.

    With hand-depth alignment on, its hand masks are an input too. ``include_mano_assets`` adds the
    content identity of the MANO assets (joint model + PCA codec) for ``mano_dir``."""
    root = Path(seq_folder)
    files = [root / name for name in _UPSTREAM_ARTIFACTS]
    slam_dir = root / "SLAM"
    if slam_dir.is_dir():
        files.extend(sorted(p for p in slam_dir.glob("*.npz") if not p.name.startswith(_SELF_WRITTEN_SLAM_PREFIXES)))
    artifacts = {}
    for path in files:
        identity = _file_identity(path)
        if identity is not None:
            artifacts[str(path.relative_to(root))] = identity
    for path in extra_files:
        if path:
            artifacts[str(path)] = _file_identity(Path(path))
    hda = HandDepthAlignConfig.from_env()
    dependencies = {
        "artifacts": artifacts,
        "hand_depth_align": hda.signature(),
    }
    if hda.enable:
        mask_path = _hand_mask_path(root)
        # recorded even when absent, so masks appearing later (alignment then applies) invalidate the entry
        dependencies["hand_masks"] = None if mask_path is None else [str(mask_path.relative_to(root)), _file_identity(mask_path)]
    if include_mano_assets:
        dependencies["mano_assets"] = mano_asset_identity(mano_dir)
    return dependencies


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
    dependencies: dict | None = None,
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
        # entries written before dependencies were recorded lack the key -> miss
        or payload.get("dependencies") != (dependencies if dependencies is not None else feature_cache_dependencies(seq_folder))
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
    dependencies: dict | None = None,
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
        "dependencies": dependencies if dependencies is not None else feature_cache_dependencies(seq_folder),
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
