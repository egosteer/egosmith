"""Shared quality-metric helpers for manifest and WebDataset filtering.

Facade module: the implementation is split into cohesive submodules for readability —

- ``constants``     — lowdim schema slices + sanity tolerances
- ``lowdim``        — per-frame sample decode / numeric-sanity validation / metadata parsing
- ``kinematics``    — geometry + per-frame motion-step computations
- ``projection``    — image-projection + in-frame / off-screen classification
- ``accumulator``   — per-clip stats lifecycle (init / update / finalize)
- ``distribution``  — dataset-level distribution summaries (percentile / IQR)
- ``thresholds``    — auto threshold resolution
- ``decision``      — keep/reject decision

This module re-exports the full public surface, so ``from lib.pipeline.quality.quality_metrics
import X`` keeps working unchanged. New code may import directly from the submodules.
"""

from __future__ import annotations

from .accumulator import (
    finalize_clip_quality_metrics,
    new_clip_quality_stats,
    update_clip_quality_stats,
)
from .constants import (
    CAMERA_AXES,
    EXTRINSIC_BOTTOM_ROW_TOL,
    EXTRINSIC_ROTATION_DET_TOL,
    EXTRINSIC_ROTATION_ORTHO_FROB_TOL,
    EXTRINSIC_SLICE,
    FRAME_INDEX_PATTERN,
    INTRINSIC_SLICE,
    LEFT_FINGERTIPS_SLICE,
    LEFT_HAND_TRANSLATION_SLICE,
    LEFT_ROOT_ROT6D_SLICE,
    LOWDIM_ROT6D_SLICES,
    LOWDIM_SIZE,
    NEXT_LEFT_ROOT_ROT6D_SLICE,
    NEXT_RIGHT_ROOT_ROT6D_SLICE,
    RIGHT_FINGERTIPS_SLICE,
    RIGHT_HAND_TRANSLATION_SLICE,
    RIGHT_ROOT_ROT6D_SLICE,
    ROT6D_MIN_CROSS_NORM,
    ROT6D_ORTHOGONALITY_TOL,
    ROT6D_UNIT_NORM_TOL,
)
from .decision import _camera_space_bounds_exceeded, decide_clip_quality
from .distribution import summarize_iqr_distribution, summarize_metric_distribution
from .kinematics import (
    _abs_axis_metrics,
    _fingers_relative_to_wrist,
    camera_space_abs_metrics,
    camera_space_axis_metrics,
    max_camera_step,
    max_translation_step,
    transform_points_world_to_camera,
    windowed_wrist_camera_extremes,
)
from .lowdim import (
    _extrinsic_is_sane,
    _intrinsic_is_sane,
    _rot6d_is_sane,
    _rot6d_to_rotmat,
    decode_lowdim,
    extract_lowdim_components,
    is_finite_array,
    parse_frame_index,
    parse_instruction_metadata,
    validate_lowdim_numeric_sanity,
)
from .projection import classify_hand_projection, project_points_world_to_image
from .thresholds import (
    _camera_space_axis_abs_cap_bounds,
    _camera_space_bound_metrics,
    _merge_camera_space_bounds,
    resolve_auto_quality_thresholds,
)

__all__ = [
    # constants
    "LOWDIM_SIZE",
    "LEFT_HAND_TRANSLATION_SLICE",
    "RIGHT_HAND_TRANSLATION_SLICE",
    "LEFT_ROOT_ROT6D_SLICE",
    "RIGHT_ROOT_ROT6D_SLICE",
    "LEFT_FINGERTIPS_SLICE",
    "RIGHT_FINGERTIPS_SLICE",
    "NEXT_LEFT_ROOT_ROT6D_SLICE",
    "NEXT_RIGHT_ROOT_ROT6D_SLICE",
    "EXTRINSIC_SLICE",
    "INTRINSIC_SLICE",
    "FRAME_INDEX_PATTERN",
    "CAMERA_AXES",
    "LOWDIM_ROT6D_SLICES",
    "ROT6D_UNIT_NORM_TOL",
    "ROT6D_ORTHOGONALITY_TOL",
    "ROT6D_MIN_CROSS_NORM",
    "EXTRINSIC_BOTTOM_ROW_TOL",
    "EXTRINSIC_ROTATION_ORTHO_FROB_TOL",
    "EXTRINSIC_ROTATION_DET_TOL",
    # lowdim
    "parse_instruction_metadata",
    "is_finite_array",
    "parse_frame_index",
    "decode_lowdim",
    "validate_lowdim_numeric_sanity",
    "extract_lowdim_components",
    # kinematics
    "max_translation_step",
    "max_camera_step",
    "transform_points_world_to_camera",
    "camera_space_abs_metrics",
    "camera_space_axis_metrics",
    "windowed_wrist_camera_extremes",
    # projection
    "project_points_world_to_image",
    "classify_hand_projection",
    # accumulator
    "new_clip_quality_stats",
    "update_clip_quality_stats",
    "finalize_clip_quality_metrics",
    # distribution
    "summarize_metric_distribution",
    "summarize_iqr_distribution",
    # thresholds
    "resolve_auto_quality_thresholds",
    # decision
    "decide_clip_quality",
]
