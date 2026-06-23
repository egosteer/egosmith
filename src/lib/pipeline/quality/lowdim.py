"""Per-frame sample decoding, numeric-sanity validation, and metadata parsing.

Covers the 116-d lowdim vector (decode, rot6d/extrinsic/intrinsic sanity, component extraction)
plus the instruction-metadata and frame-key parsing used alongside it.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    EXTRINSIC_BOTTOM_ROW_TOL,
    EXTRINSIC_ROTATION_DET_TOL,
    EXTRINSIC_ROTATION_ORTHO_FROB_TOL,
    EXTRINSIC_SLICE,
    FRAME_INDEX_PATTERN,
    INTRINSIC_SLICE,
    LEFT_FINGERTIPS_SLICE,
    LEFT_HAND_TRANSLATION_SLICE,
    LOWDIM_ROT6D_SLICES,
    LOWDIM_SIZE,
    RIGHT_FINGERTIPS_SLICE,
    RIGHT_HAND_TRANSLATION_SLICE,
    ROT6D_MIN_CROSS_NORM,
    ROT6D_ORTHOGONALITY_TOL,
    ROT6D_UNIT_NORM_TOL,
)


def parse_instruction_metadata(meta: dict | None) -> dict:
    """Normalize one frame/episode instruction payload and expose fail-closed flags."""
    if not isinstance(meta, dict):
        return {
            "instruction_num": 0,
            "instructions": [],
            "slots": [],
            "effective_slots": [],
            "missing_instruction": False,
            "empty_instruction": False,
            "instruction_num_mismatch": False,
        }

    raw_instruction_num = meta.get("instruction_num", 0)
    try:
        instruction_num = max(0, int(raw_instruction_num))
    except Exception:
        instruction_num = 0

    raw_instruction = meta.get("instruction", [])
    if isinstance(raw_instruction, str):
        slots = [raw_instruction]
    elif isinstance(raw_instruction, (list, tuple)):
        slots = list(raw_instruction)
    else:
        slots = []

    effective_slots = slots[:instruction_num]
    instructions = [str(item).strip() for item in effective_slots if str(item).strip()]
    missing_instruction = instruction_num <= 0
    empty_instruction = instruction_num > 0 and len(instructions) == 0
    instruction_num_mismatch = instruction_num > 0 and (
        len(slots) < instruction_num or len(instructions) != min(instruction_num, len(effective_slots))
    )
    return {
        "instruction_num": int(instruction_num),
        "instructions": instructions,
        "slots": slots,
        "effective_slots": effective_slots,
        "missing_instruction": bool(missing_instruction),
        "empty_instruction": bool(empty_instruction),
        "instruction_num_mismatch": bool(instruction_num_mismatch),
    }


def is_finite_array(value) -> bool:
    array = np.asarray(value)
    return bool(np.isfinite(array).all())


def parse_frame_index(sample_key: str) -> int:
    match = FRAME_INDEX_PATTERN.search(sample_key)
    if not match:
        raise ValueError(f"Failed to parse frame index from sample key: {sample_key}")
    return int(match.group(1))


def decode_lowdim(lowdim_bytes: bytes) -> np.ndarray:
    import io

    array = np.load(io.BytesIO(lowdim_bytes), allow_pickle=False)
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.shape != (LOWDIM_SIZE,):
        raise ValueError(f"Expected lowdim shape {(LOWDIM_SIZE,)}, got {array.shape}")
    return array


def _rot6d_is_sane(rot6d: np.ndarray) -> bool:
    array = np.asarray(rot6d, dtype=np.float32).reshape(-1)
    if array.shape != (6,) or not np.isfinite(array).all():
        return False
    col_a = array[:3]
    col_b = array[3:]
    norm_a = float(np.linalg.norm(col_a))
    norm_b = float(np.linalg.norm(col_b))
    if norm_a <= 1e-8 or norm_b <= 1e-8:
        return False
    if abs(norm_a - 1.0) > ROT6D_UNIT_NORM_TOL or abs(norm_b - 1.0) > ROT6D_UNIT_NORM_TOL:
        return False
    unit_a = col_a / norm_a
    unit_b = col_b / norm_b
    dot_abs = abs(float(np.dot(unit_a, unit_b)))
    if dot_abs > ROT6D_ORTHOGONALITY_TOL:
        return False
    cross_norm = float(np.linalg.norm(np.cross(unit_a, unit_b)))
    if cross_norm < ROT6D_MIN_CROSS_NORM:
        return False
    return True


def _rot6d_to_rotmat(rot6d) -> np.ndarray:
    """Single 6D rotation -> 3x3 matrix via Gram-Schmidt.

    Matches lib/pipeline/exporters/mano_codec.rot6d_to_rotmat exactly (columns [b1, b2, b3]) so
    wrist-rotation deltas are consistent with the stored root rot6d. Kept local to avoid importing
    the exporters package (and its quality_metrics dependency) into this widely-used module.
    """
    row = np.asarray(rot6d, dtype=np.float32).reshape(6)
    a1 = row[:3]
    a2 = row[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32)


def _extrinsic_is_sane(extrinsic: np.ndarray) -> bool:
    matrix = np.asarray(extrinsic, dtype=np.float32).reshape(4, 4)
    if not np.isfinite(matrix).all():
        return False
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=EXTRINSIC_BOTTOM_ROW_TOL):
        return False
    rotation = matrix[:3, :3].astype(np.float64)
    det = float(np.linalg.det(rotation))
    if not np.isfinite(det) or abs(det - 1.0) > EXTRINSIC_ROTATION_DET_TOL:
        return False
    ortho_err = float(np.linalg.norm(rotation.T @ rotation - np.eye(3, dtype=np.float64), ord="fro"))
    if ortho_err > EXTRINSIC_ROTATION_ORTHO_FROB_TOL:
        return False
    return True


def _intrinsic_is_sane(intrinsic: np.ndarray) -> bool:
    array = np.asarray(intrinsic, dtype=np.float32).reshape(-1)
    if array.shape != (4,) or not np.isfinite(array).all():
        return False
    return float(array[0]) > 0.0 and float(array[1]) > 0.0


def validate_lowdim_numeric_sanity(lowdim: np.ndarray) -> dict:
    array = np.asarray(lowdim, dtype=np.float32).reshape(-1)
    if array.shape != (LOWDIM_SIZE,):
        raise ValueError(f"Expected lowdim shape {(LOWDIM_SIZE,)}, got {array.shape}")

    invalid_rot6d = any(not _rot6d_is_sane(array[rot_slice]) for rot_slice in LOWDIM_ROT6D_SLICES)
    invalid_extrinsic = not _extrinsic_is_sane(array[EXTRINSIC_SLICE].reshape(4, 4))
    invalid_intrinsic = not _intrinsic_is_sane(array[INTRINSIC_SLICE])
    issues = []
    if invalid_rot6d:
        issues.append("invalid_rot6d")
    if invalid_extrinsic:
        issues.append("invalid_extrinsic")
    if invalid_intrinsic:
        issues.append("invalid_intrinsic")
    return {
        "valid": not issues,
        "invalid_rot6d": bool(invalid_rot6d),
        "invalid_extrinsic": bool(invalid_extrinsic),
        "invalid_intrinsic": bool(invalid_intrinsic),
        "issues": issues,
    }


def extract_lowdim_components(lowdim: np.ndarray) -> dict:
    array = np.asarray(lowdim, dtype=np.float32).reshape(-1)
    if array.shape != (LOWDIM_SIZE,):
        raise ValueError(f"Expected lowdim shape {(LOWDIM_SIZE,)}, got {array.shape}")
    return {
        "left_translation": array[LEFT_HAND_TRANSLATION_SLICE],
        "right_translation": array[RIGHT_HAND_TRANSLATION_SLICE],
        "left_fingertips": array[LEFT_FINGERTIPS_SLICE].reshape(5, 3),
        "right_fingertips": array[RIGHT_FINGERTIPS_SLICE].reshape(5, 3),
        "extrinsic": array[EXTRINSIC_SLICE].reshape(4, 4),
        "intrinsic": array[INTRINSIC_SLICE].reshape(4),
    }
