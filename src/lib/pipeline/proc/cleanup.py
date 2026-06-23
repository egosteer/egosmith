"""Retention / cleanup of per-clip stage intermediates.

A single 1-minute clip can leave many GB of stage intermediates beside its
outputs (stage-3 frame caches, dense-depth + DPVO caches, per-frame masks,
cam-space dumps, extracted frames). Once a clip's final product exists, almost
all of that is dead weight and easily fills disks.

This module removes those intermediates while preserving an explicit set of
*final artifacts*. It is deliberately:

* **Explicit** about what is preserved vs removed (no "delete everything but..."
  guesswork) -- preserved patterns are listed and the depth artifact is kept by
  default so a cleaned clip still carries depth.
* **Robust to partial failure** -- each removal is isolated; one failure never
  aborts the rest. A :class:`CleanupReport` records what happened.
* **Pure filesystem** -- no torch/cv2, so it is unit-testable without the deps.

Retention levels (``keep_intermediates``):

* ``all``  -- keep everything (no cleanup). Backwards-compatible default for
  callers that don't opt in.
* ``slam`` -- remove only the heavy, redundant SLAM caches + stage-3 frames;
  keep masks/cam-space (so an infiller re-run is cheap) and depth.
* ``none`` -- keep only final artifacts; remove all intermediates including
  ``.done`` markers. (A completed clip is then identified by its final artifact
  existing, not by a marker.)

IMPORTANT ordering note: in the dataset pipeline the ``build`` stage still needs
the depth artifact *and* the extracted frames, so aggressive cleanup (and
``remove_frames``) must run only after the final consuming stage, not right
after infiller. Callers pass ``remove_frames`` explicitly for that reason.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from lib.pipeline.proc.logging_setup import get_logger
from lib.pipeline.io.result_io import RESULT_FILENAME, final_artifact_exists, result_exists

_logger = get_logger("cleanup")

RETENTION_LEVELS = ("all", "slam", "none")
DEFAULT_RETENTION = "none"

# Final artifacts -- never removed by cleanup (the products of a successful run).
_FINAL_FILE_NAMES = ("world_space_res.pth", "est_focal.txt", "slam_backend.txt")
_FINAL_SLAM_GLOBS = (
    "hawor_slam_w_scale_*.npz",  # scaled SLAM trajectory (final)
    "dense_depth_any4d_*.npz",   # exported dense depth (kept so depth survives cleanup)
)

# Heavy, redundant SLAM caches -- removable at 'slam' and 'none'.
_HEAVY_SLAM_GLOBS = (
    "dpvo_raw_*.npz",
    "any4d_depth_dpvo_*.npz",
    "any4d_stitch_cf_*.npz",
    "hand_anchor_k_*.npz",
)


@dataclass
class CleanupReport:
    removed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    freed_bytes: int = 0

    def summary(self) -> str:
        gb = self.freed_bytes / (1024 ** 3)
        return (
            f"cleanup: removed {len(self.removed)} item(s), freed {gb:.2f} GB, "
            f"{len(self.errors)} error(s)"
        )


def _path_size(path: Path) -> int:
    try:
        if path.is_dir():
            total = 0
            for p in path.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
            return total
        return path.stat().st_size
    except OSError:
        return 0


def _remove(path: Path, report: CleanupReport, *, dry_run: bool) -> None:
    if not path.exists():
        return
    size = _path_size(path)
    if dry_run:
        report.skipped.append(f"would remove {path}")
        report.freed_bytes += size
        return
    try:
        if path.is_dir():
            import shutil

            shutil.rmtree(path)
        else:
            path.unlink()
        report.removed.append(str(path))
        report.freed_bytes += size
    except OSError as error:
        report.errors.append(f"{path}: {error}")
        _logger.warning("cleanup could not remove %s: %s", path, error)


def _is_dense_depth_all_frames(path: Path) -> bool:
    # Keep the full dense-depth export but drop legacy keyframe-only variants which
    # are redundant once the all-frames export exists.
    name = path.name
    return name.startswith("dense_depth_any4d_") and "keyframes" not in name


def cleanup_seq_folder(
    seq_folder,
    *,
    level: str = DEFAULT_RETENTION,
    tmp_root: Optional[str] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    keep_depth: bool = True,
    remove_frames: bool = False,
    dry_run: bool = False,
) -> CleanupReport:
    """Remove stage intermediates under ``seq_folder`` per the retention ``level``.

    ``keep_depth`` (default True) preserves the dense-depth export so a cleaned
    clip still carries depth; set False only when depth is already captured
    elsewhere (e.g. folded into the final WebDataset shard).

    ``remove_frames`` also clears ``extracted_images/`` -- pass True only after the
    final frame-consuming stage (e.g. WebDataset build) has run.

    ``tmp_root`` lets cleanup also purge this clip's stage-3 frame cache.
    """
    report = CleanupReport()
    if level not in RETENTION_LEVELS:
        raise ValueError(f"Unknown retention level {level!r}; expected one of {RETENTION_LEVELS}")
    seq_folder = Path(seq_folder)
    if level == "all" or not seq_folder.is_dir():
        return report

    slam_dir = seq_folder / "SLAM"

    # Heavy redundant SLAM caches (both 'slam' and 'none').
    for pattern in _HEAVY_SLAM_GLOBS:
        for path in slam_dir.glob(pattern):
            _remove(path, report, dry_run=dry_run)

    # Depth is redundant on disk once it has been consolidated into result.npz
    # (or when the caller says depth is captured elsewhere, e.g. the WebDataset).
    consolidated = result_exists(seq_folder)
    if consolidated or not keep_depth:
        for path in slam_dir.glob("dense_depth_any4d_*.npz"):
            _remove(path, report, dry_run=dry_run)
    else:
        # Even when keeping depth, legacy keyframe-only caches are redundant.
        for path in slam_dir.glob("dense_depth_any4d_keyframes_*.npz"):
            _remove(path, report, dry_run=dry_run)

    # Stage-3 frame materialization cache (shared scratch root).
    if tmp_root and start_idx is not None and end_idx is not None:
        from lib.pipeline.io.workspace import stage3_frame_cache_dir

        cache_dir = Path(stage3_frame_cache_dir(tmp_root, str(seq_folder), int(start_idx), int(end_idx)))
        _remove(cache_dir, report, dry_run=dry_run)

    if level == "none":
        # Legacy pose file is redundant once consolidated into result.npz.
        if consolidated:
            _remove(seq_folder / "world_space_res.pth", report, dry_run=dry_run)
        # Per-frame tracks/masks/chunks: not needed once the final product exists.
        for track_dir in seq_folder.glob("tracks_*_*"):
            _remove(track_dir, report, dry_run=dry_run)
        # Cam-space dumps + cache.
        _remove(seq_folder / "cam_space", report, dry_run=dry_run)
        _remove(seq_folder / "cam_space_cache.joblib", report, dry_run=dry_run)
        # Track-range cache + stage done markers (completion is identified by the
        # final artifact existing, not by a marker, so removing markers is safe).
        _remove(seq_folder / ".track_range", report, dry_run=dry_run)
        for marker in seq_folder.glob(".*.done"):
            _remove(marker, report, dry_run=dry_run)

    if remove_frames:
        _remove(seq_folder / "extracted_images", report, dry_run=dry_run)

    if report.removed or report.skipped:
        _logger.info("%s for %s", report.summary(), seq_folder.name)
    return report


# Re-exported: a clip is complete if it has the consolidated result.npz OR the
# legacy world_space_res.pth. Used for resume detection after markers are removed.
__all__ = ["cleanup_seq_folder", "CleanupReport", "final_artifact_exists", "RETENTION_LEVELS"]
