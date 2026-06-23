"""Helpers for loading per-frame depth artifacts for export/rewrite."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from lib.pipeline.slam.native_depth import get_native_depth_output_path


DEPTH_EXPORT_ENCODING = "uint16_mm"
DEPTH_EXPORT_SCHEMA = "metric_depth_uint16_mm_v1"
DEPTH_EXPORT_DTYPE = np.dtype(np.uint16)


def _dense_depth_cache_candidates(seq_folder: str | Path, start_idx: int, end_idx: int) -> list[Path]:
    root = Path(seq_folder) / "SLAM"
    return [
        root / f"dense_depth_any4d_{start_idx}_{end_idx}.npz",
        root / f"dense_depth_any4d_all_{start_idx}_{end_idx}.npz",
        root / f"dense_depth_any4d_keyframes_{start_idx}_{end_idx}.npz",
    ]


def _read_cached_track_range(seq_folder: str | Path) -> tuple[int, int] | None:
    cache_file = Path(seq_folder) / ".track_range"
    if not cache_file.is_file():
        return None
    try:
        raw = cache_file.read_text(encoding="utf-8").strip()
        start_idx, end_idx = raw.split(",", 1)
        return int(start_idx), int(end_idx)
    except (OSError, ValueError):
        return None


def _discover_track_range(seq_folder: str | Path) -> tuple[int, int] | None:
    cached = _read_cached_track_range(seq_folder)
    if cached is not None:
        return cached
    root = Path(seq_folder)
    candidates = []
    for track_dir in root.glob("tracks_*_*"):
        if not track_dir.is_dir():
            continue
        parts = track_dir.name.split("_")
        if len(parts) != 3:
            continue
        try:
            candidates.append((int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[1], item[0]))
    return candidates[-1]


def _load_depth_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if "frame_indices" not in payload.files:
            raise ValueError(f"Depth artifact missing frame_indices: {path}")
        frame_indices = np.asarray(payload["frame_indices"], dtype=np.int64).reshape(-1)
        if "depths_uint16" in payload.files:
            depths = payload["depths_uint16"].astype(np.float32) * 1e-3
        elif "depths" in payload.files:
            depths = np.asarray(payload["depths"], dtype=np.float32)
        elif "pred_depths" in payload.files:
            depths = np.asarray(payload["pred_depths"], dtype=np.float32)
        else:
            raise ValueError(f"Depth artifact missing depths payload: {path}")

    if depths.ndim != 3:
        raise ValueError(f"Depth artifact must have shape (T,H,W), got {depths.shape} from {path}")
    if depths.shape[0] != frame_indices.shape[0]:
        raise ValueError(
            f"Depth artifact frame count mismatch: frame_indices={frame_indices.shape[0]}, depths={depths.shape[0]} ({path})"
        )
    if not np.isfinite(depths).all():
        raise ValueError(f"Depth artifact contains non-finite values: {path}")
    if (depths < 0.0).any():
        raise ValueError(f"Depth artifact contains negative depths: {path}")
    return frame_indices, depths


def _load_consolidated_result_depths(seq_folder: Path, expected_frame_count: int, expected_indices: np.ndarray):
    """Read depth from the consolidated result.npz, if present and aligned."""
    from lib.pipeline.io import result_io

    depth = result_io.load_result_depth(seq_folder)
    if depth is None:
        return None
    frame_indices, depths_uint16 = depth
    depths = depths_uint16.astype(np.float32) * 1e-3
    if depths.ndim != 3 or frame_indices.shape[0] < expected_frame_count:
        return None
    if not np.array_equal(frame_indices[:expected_frame_count], expected_indices):
        return None
    return np.asarray(depths[:expected_frame_count], dtype=np.float32)


def load_export_depths(seq_folder: str | Path, expected_frame_count: int) -> np.ndarray:
    """Load per-frame metric depth aligned to exported frame indices [0..T-1]."""
    seq_folder = Path(seq_folder)
    expected_frame_count = int(expected_frame_count)
    expected_indices = np.arange(expected_frame_count, dtype=np.int64)

    # Prefer the consolidated result.npz (new single-file output).
    consolidated = _load_consolidated_result_depths(seq_folder, expected_frame_count, expected_indices)
    if consolidated is not None:
        return consolidated

    native_path = get_native_depth_output_path(seq_folder)
    candidate_artifacts = []
    if native_path.is_file():
        candidate_artifacts.append(native_path)
    track_range = _discover_track_range(seq_folder)
    start_idx, end_idx = track_range if track_range is not None else (None, None)
    if start_idx is not None and end_idx is not None:
        candidate_artifacts.extend(_dense_depth_cache_candidates(seq_folder, int(start_idx), int(end_idx)))

    last_error: Exception | None = None
    for path in candidate_artifacts:
        if not Path(path).is_file():
            continue
        try:
            frame_indices, depths = _load_depth_npz(path)
        except Exception as error:
            last_error = error
            continue
        if frame_indices.shape[0] < expected_frame_count:
            last_error = ValueError(
                f"Depth artifact shorter than requested frames: expected={expected_frame_count}, got={frame_indices.shape[0]} ({path})"
            )
            continue
        if not np.array_equal(frame_indices[:expected_frame_count], expected_indices):
            last_error = ValueError(
                f"Depth artifact frame indices do not match export frames 0..{expected_frame_count - 1}: {path}"
            )
            continue
        return np.asarray(depths[:expected_frame_count], dtype=np.float32)

    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"No depth artifact found for {seq_folder}")


def depth_to_uint16_mm(depth_frame: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_frame, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth frame must have shape (H,W), got {depth.shape}")
    if not np.isfinite(depth).all():
        raise ValueError("Depth frame contains non-finite values")
    if (depth < 0.0).any():
        raise ValueError("Depth frame contains negative values")
    depth_mm = np.clip(np.round(depth * 1000.0), 0.0, 65535.0).astype(DEPTH_EXPORT_DTYPE)
    return depth_mm


def encode_depth_npy(depth_frame: np.ndarray) -> bytes:
    depth_mm = depth_to_uint16_mm(depth_frame)
    buf = io.BytesIO()
    np.save(buf, depth_mm, allow_pickle=False)
    return buf.getvalue()
