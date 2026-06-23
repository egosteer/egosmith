"""Shared constants for the quality-metric helpers (lowdim schema + sanity tolerances).

Leaf module: imported by every other ``quality`` submodule, imports nothing from the package.
"""

from __future__ import annotations

import re


LOWDIM_SIZE = 116
LEFT_HAND_TRANSLATION_SLICE = slice(0, 3)
RIGHT_HAND_TRANSLATION_SLICE = slice(3, 6)
LEFT_ROOT_ROT6D_SLICE = slice(6, 12)
RIGHT_ROOT_ROT6D_SLICE = slice(12, 18)
LEFT_FINGERTIPS_SLICE = slice(18, 33)
RIGHT_FINGERTIPS_SLICE = slice(33, 48)
NEXT_LEFT_ROOT_ROT6D_SLICE = slice(54, 60)
NEXT_RIGHT_ROOT_ROT6D_SLICE = slice(60, 66)
EXTRINSIC_SLICE = slice(96, 112)
INTRINSIC_SLICE = slice(112, 116)
FRAME_INDEX_PATTERN = re.compile(r"_f(\d+)$")
CAMERA_AXES = ("x", "y", "z")
LOWDIM_ROT6D_SLICES = (
    LEFT_ROOT_ROT6D_SLICE,
    RIGHT_ROOT_ROT6D_SLICE,
    NEXT_LEFT_ROOT_ROT6D_SLICE,
    NEXT_RIGHT_ROOT_ROT6D_SLICE,
)
ROT6D_UNIT_NORM_TOL = 0.2
ROT6D_ORTHOGONALITY_TOL = 0.2
ROT6D_MIN_CROSS_NORM = 0.5
EXTRINSIC_BOTTOM_ROW_TOL = 1e-3
EXTRINSIC_ROTATION_ORTHO_FROB_TOL = 0.2
EXTRINSIC_ROTATION_DET_TOL = 0.2
