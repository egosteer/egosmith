"""Consolidated per-clip final result I/O (``result.npz``).

Historically a finished clip left two separate final artifacts:

* ``world_space_res.pth`` -- a joblib list ``[pred_trans, pred_rot,
  pred_hand_pose, pred_betas, pred_valid]`` (world-space MANO params), and
* ``SLAM/dense_depth_any4d_*.npz`` -- the per-frame metric depth.

This module consolidates both into a single self-contained ``result.npz`` so a
cleaned clip carries everything (poses + depth) in one file, and the separate
intermediates can be removed.

To avoid breaking the many existing readers of ``world_space_res.pth`` and the
depth npz, the consolidation is additive + backward compatible:

* :func:`consolidate_result` writes ``result.npz`` from the legacy artifacts.
* :func:`load_pose_arrays` and :func:`load_result_depth` read ``result.npz`` when
  present and otherwise fall back to the legacy files, so the central build
  loaders work for both old and new runs.

Pure numpy/joblib (no torch import); tensors are converted via duck typing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from lib.pipeline.proc.logging_setup import get_logger

_logger = get_logger("result_io")

RESULT_FILENAME = "result.npz"
RESULT_SCHEMA = "hawor_result_v1"
DEPTH_ENCODING = "uint16_mm"

_POSE_KEYS = ("pred_trans", "pred_rot", "pred_hand_pose", "pred_betas", "pred_valid")


def result_path(seq_folder) -> Path:
    return Path(seq_folder) / RESULT_FILENAME


def result_exists(seq_folder) -> bool:
    return result_path(seq_folder).is_file()


def _to_numpy(value):
    """Convert a torch tensor / array-like to a numpy array (no torch import)."""
    if value is None:
        return None
    # torch.Tensor exposes detach()/cpu()/numpy(); duck-type to avoid importing torch.
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        npy = getattr(value, "numpy", None)
        if callable(npy):
            return npy()
    return np.asarray(value)


def save_result(
    seq_folder,
    *,
    pred_trans,
    pred_rot,
    pred_hand_pose,
    pred_betas,
    pred_valid,
    depth_frame_indices=None,
    depths_uint16=None,
    depth_height: Optional[int] = None,
    depth_width: Optional[int] = None,
) -> Path:
    """Write the consolidated ``result.npz`` (poses + optional depth)."""
    out = result_path(seq_folder)
    payload = {
        "schema": np.array(RESULT_SCHEMA),
        "pred_trans": _to_numpy(pred_trans),
        "pred_rot": _to_numpy(pred_rot),
        "pred_hand_pose": _to_numpy(pred_hand_pose),
        "pred_betas": _to_numpy(pred_betas),
        "pred_valid": _to_numpy(pred_valid),
    }
    if depths_uint16 is not None and depth_frame_indices is not None:
        payload["depths_uint16"] = np.asarray(depths_uint16, dtype=np.uint16)
        payload["depth_frame_indices"] = np.asarray(depth_frame_indices, dtype=np.int64)
        payload["depth_encoding"] = np.array(DEPTH_ENCODING)
        if depth_height is not None:
            payload["depth_height"] = np.int32(depth_height)
        if depth_width is not None:
            payload["depth_width"] = np.int32(depth_width)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **payload)
    return out


def _read_legacy_depth(seq_folder, start_idx, end_idx):
    """Read (frame_indices, depths_uint16, h, w) from the legacy dense-depth npz."""
    slam = Path(seq_folder) / "SLAM"
    candidates = [
        slam / f"dense_depth_any4d_{start_idx}_{end_idx}.npz",
        slam / f"dense_depth_any4d_all_{start_idx}_{end_idx}.npz",
        slam / f"dense_depth_any4d_keyframes_{start_idx}_{end_idx}.npz",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                if "depths_uint16" not in data.files or "frame_indices" not in data.files:
                    continue
                depths = np.asarray(data["depths_uint16"], dtype=np.uint16)
                indices = np.asarray(data["frame_indices"], dtype=np.int64)
                h = int(data["height"]) if "height" in data.files else (depths.shape[1] if depths.ndim == 3 else None)
                w = int(data["width"]) if "width" in data.files else (depths.shape[2] if depths.ndim == 3 else None)
                return indices, depths, h, w
        except Exception as error:  # pragma: no cover - defensive
            _logger.warning("Could not read legacy depth %s: %s", path, error)
    return None


def consolidate_result(seq_folder, *, start_idx: int, end_idx: int) -> Optional[Path]:
    """Build ``result.npz`` from the legacy ``world_space_res.pth`` + depth npz.

    Returns the path on success, or ``None`` if the pose file is absent. Depth is
    embedded when available; otherwise the result holds poses only (logged).
    """
    import joblib  # local import: not needed for pure-load paths/tests

    seq_folder = Path(seq_folder)
    world_res = seq_folder / "world_space_res.pth"
    if not world_res.is_file():
        _logger.warning("consolidate_result: %s missing; skipping", world_res)
        return None
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_res)

    depth = _read_legacy_depth(seq_folder, start_idx, end_idx)
    kwargs = {}
    if depth is not None:
        indices, depths, h, w = depth
        kwargs = {
            "depth_frame_indices": indices,
            "depths_uint16": depths,
            "depth_height": h,
            "depth_width": w,
        }
    else:
        _logger.warning(
            "consolidate_result: no depth artifact for %s [%s,%s]; result.npz will hold poses only",
            seq_folder.name, start_idx, end_idx,
        )
    return save_result(
        seq_folder,
        pred_trans=pred_trans,
        pred_rot=pred_rot,
        pred_hand_pose=pred_hand_pose,
        pred_betas=pred_betas,
        pred_valid=pred_valid,
        **kwargs,
    )


def load_pose_arrays(world_res_path):
    """Return ``[pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid]``.

    Prefers a sibling/own ``result.npz`` (new format), falling back to the legacy
    ``world_space_res.pth`` joblib list. ``world_res_path`` may point at either the
    legacy ``.pth`` or its containing seq folder.
    """
    p = Path(world_res_path)
    seq_folder = p.parent if p.suffix else p
    rp = result_path(seq_folder)
    if rp.is_file():
        with np.load(rp, allow_pickle=True) as data:
            return [data[key] for key in _POSE_KEYS]
    # Legacy fallback.
    import joblib

    legacy = p if p.suffix == ".pth" else (seq_folder / "world_space_res.pth")
    return list(joblib.load(legacy))


def load_result_depth(seq_folder) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(frame_indices, depths_uint16)`` from ``result.npz`` if present."""
    rp = result_path(seq_folder)
    if not rp.is_file():
        return None
    with np.load(rp, allow_pickle=True) as data:
        if "depths_uint16" not in data.files or "depth_frame_indices" not in data.files:
            return None
        return (
            np.asarray(data["depth_frame_indices"], dtype=np.int64),
            np.asarray(data["depths_uint16"], dtype=np.uint16),
        )


def final_artifact_exists(seq_folder) -> bool:
    """A clip is complete if it has the consolidated result OR the legacy pose file."""
    seq_folder = Path(seq_folder)
    return result_exists(seq_folder) or (seq_folder / "world_space_res.pth").is_file()
