"""Single source of truth for camera intrinsics across all pipeline stages.

Historically the focal length was resolved independently in several places, each with
slightly different default/persist behavior:
  * ``stages/slam.py:_resolve_focal`` + ``_build_calibration``
  * ``stages/hawor_motion_stage.py:_resolve_img_focal``
  * a duplicate inline copy in the motion cache-skip path
  * ``slam_geom_utils.est_calib`` (image-dimension heuristic, always overridden)
That divergence silently baked a wrong focal (the blind ``600`` default) into entire runs.

This module collapses all of it into ONE entry point: ``resolve_calibration``. Stages take
the focal as ``calib[0]`` and the principal point as ``calib[2:4]`` — there is no separate
focal-resolution path to drift out of sync.
"""

import os
from typing import Optional

import numpy as np

from lib.pipeline.proc.logging_setup import get_logger

_logger = get_logger("intrinsics")

EST_FOCAL_FILENAME = "est_focal.txt"


def read_recorded_focal(seq_folder: str) -> Optional[float]:
    """Pure accessor: the focal recorded in ``est_focal.txt``, or ``None`` if absent/invalid.

    This is NOT a resolver — it applies no precedence, no default, and never writes. Use it
    only to report what is on record; use ``resolve_calibration`` to actually decide a focal.
    """
    path = os.path.join(seq_folder, EST_FOCAL_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = float(handle.read())
    except Exception:
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return float(value)


def _persist_focal(seq_folder: str, focal: float) -> None:
    path = os.path.join(seq_folder, EST_FOCAL_FILENAME)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(str(float(focal)))
    except OSError as error:
        _logger.warning("Could not persist focal to %s (%s).", path, error)


def _frame_dims(frame_source):
    image = frame_source.get_frame(0, rgb=False)
    height, width = image.shape[:2]
    return int(height), int(width)


def resolve_calibration(
    frame_source,
    seq_folder: str,
    *,
    requested_focal: Optional[float] = None,
    persist: bool = True,
):
    """THE single interface for camera intrinsics. Returns ``[fx, fy, cx, cy]`` (floats).

    ``fx == fy == focal``; ``cx, cy`` are the image center. Focal precedence (highest first):

      1. ``requested_focal``  — explicit ``--img_focal`` / config value (authoritative).
      2. ``est_focal.txt``    — a focal previously recorded in ``seq_folder``.
      3. egocentric default ``focal = W/2`` (~90° HFOV) — emitted with a LOUD warning.

    The ``W/2`` default is NEVER persisted, so an un-calibrated clip keeps warning on every
    run and the presence of ``est_focal.txt`` unambiguously means "a real focal was given".
    An explicit ``requested_focal`` IS persisted (sticky calibration across reruns).
    """
    height, width = _frame_dims(frame_source)
    cx, cy = width / 2.0, height / 2.0

    if requested_focal is not None and float(requested_focal) > 0:
        focal = float(requested_focal)
        if persist:
            _persist_focal(seq_folder, focal)
        return [focal, focal, cx, cy]

    recorded = read_recorded_focal(seq_folder)
    if recorded is not None:
        return [recorded, recorded, cx, cy]

    # Egocentric wide-angle default: HFOV = 2*atan((W/2)/(W/2)) = 90°. Far closer to the
    # real wide-angle ego cameras than the old blind 600 (which implies a ~42° telephoto).
    focal = width / 2.0
    _logger.warning(
        "No focal provided for %s and no %s found -> using egocentric wide-angle default "
        "focal=W/2=%.1f (assumes ~90 deg HFOV). This is a GUESS and biases depth/SLAM/hand if "
        "the true camera differs. Pass --img_focal or write %s with the real value.",
        seq_folder, EST_FOCAL_FILENAME, focal, EST_FOCAL_FILENAME,
    )
    return [focal, focal, cx, cy]
