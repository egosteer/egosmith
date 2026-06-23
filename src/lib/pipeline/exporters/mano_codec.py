"""Helpers for storing MANO in WebDataset and replaying it with manopth."""

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

MANO_PCA_DIMS = 45
MANO_BETA_DIMS = 10
MANO_HAND_PARAM_DIMS = MANO_PCA_DIMS + MANO_BETA_DIMS
MANO_SAMPLE_SHAPE = (2, MANO_HAND_PARAM_DIMS)
MANO_SCHEMA = "manopth_pca45_flatmean_center0_v1"
MANO_CENTER_IDX = 0
MANO_FLAT_HAND_MEAN = True


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_mano_root_for_side(side: str) -> Path:
    if side.lower() == "left":
        return (_project_root() / "_DATA" / "data_left" / "mano_left").resolve()
    return (_project_root() / "_DATA" / "data" / "mano").resolve()


def _resolve_mano_root(mano_dir: str | None = None) -> Path:
    if mano_dir is None:
        candidates = [
            _project_root() / "_DATA" / "mano_models",
            _project_root() / "_DATA" / "data" / "mano",
            _project_root() / "_DATA" / "data",
        ]
    else:
        base = Path(mano_dir).expanduser()
        candidates = []
        if base.is_file():
            candidates.append(base.parent)
        else:
            candidates.extend([base, base / "models", base.parent])

    for candidate in candidates:
        if (candidate / "MANO_RIGHT.pkl").exists() and (candidate / "MANO_LEFT.pkl").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Failed to resolve MANO model root. Expected one directory containing "
        "MANO_RIGHT.pkl and MANO_LEFT.pkl, or use side-specific default _DATA paths."
    )


def _resolve_mano_root_for_side(side: str, mano_dir: str | None = None) -> Path:
    side_name = side.lower()
    if side_name not in {"left", "right"}:
        raise ValueError(f"Unsupported MANO side: {side}")
    if mano_dir is None:
        root = _default_mano_root_for_side(side_name)
        file_name = "MANO_LEFT.pkl" if side_name == "left" else "MANO_RIGHT.pkl"
        if (root / file_name).exists():
            return root
    return _resolve_mano_root(mano_dir)


def _mano_pkl_path(side: str, mano_dir: str | None = None) -> str:
    side_name = side.lower()
    if side_name not in {"left", "right"}:
        raise ValueError(f"Unsupported MANO side: {side}")
    file_name = "MANO_LEFT.pkl" if side_name == "left" else "MANO_RIGHT.pkl"
    return str(_resolve_mano_root_for_side(side_name, mano_dir) / file_name)


@lru_cache(maxsize=8)
def _load_pca_codec(side: str, mano_root: str, flat_hand_mean: bool, ncomps: int):
    side_name = side.lower()
    file_name = "MANO_LEFT.pkl" if side_name == "left" else "MANO_RIGHT.pkl"
    with open(Path(mano_root) / file_name, "rb") as handle:
        data = pickle.load(handle, encoding="latin1")

    components = np.asarray(data["hands_components"], dtype=np.float32)[:ncomps]
    if components.ndim != 2 or components.shape[1] != 45:
        raise ValueError(f"Invalid MANO PCA component shape for {side}: {components.shape}")

    if flat_hand_mean:
        mean = np.zeros((45,), dtype=np.float32)
    else:
        mean = np.asarray(data["hands_mean"], dtype=np.float32).reshape(-1)[:45]
    projector = np.linalg.pinv(components).astype(np.float32)
    return components.astype(np.float32), mean.astype(np.float32), projector


def hand_pose_axis_angle_to_pca(
    hand_pose_axis_angle,
    *,
    side: str,
    mano_dir: str | None = None,
    flat_hand_mean: bool = MANO_FLAT_HAND_MEAN,
    ncomps: int = MANO_PCA_DIMS,
) -> np.ndarray:
    array = np.asarray(hand_pose_axis_angle, dtype=np.float32)
    if array.shape[-1] != 45:
        raise ValueError(f"Expected hand pose axis-angle shape (..., 45), got {array.shape}")
    components, mean, projector = _load_pca_codec(side, str(_resolve_mano_root_for_side(side, mano_dir)), flat_hand_mean, ncomps)
    del components
    flat = array.reshape(-1, 45)
    coeffs = (flat - mean[None, :]) @ projector
    return coeffs.reshape(array.shape[:-1] + (ncomps,)).astype(np.float32)


def hand_pose_pca_to_axis_angle(
    hand_pose_pca,
    *,
    side: str,
    mano_dir: str | None = None,
    flat_hand_mean: bool = MANO_FLAT_HAND_MEAN,
    ncomps: int = MANO_PCA_DIMS,
) -> np.ndarray:
    coeffs = np.asarray(hand_pose_pca, dtype=np.float32)
    if coeffs.shape[-1] != ncomps:
        raise ValueError(f"Expected hand pose PCA shape (..., {ncomps}), got {coeffs.shape}")
    components, mean, _ = _load_pca_codec(side, str(_resolve_mano_root_for_side(side, mano_dir)), flat_hand_mean, ncomps)
    flat = coeffs.reshape(-1, ncomps)
    hand_pose = flat @ components + mean[None, :]
    return hand_pose.reshape(coeffs.shape[:-1] + (45,)).astype(np.float32)


def build_mano_pca_frame_features(pred_hand_pose, pred_betas, *, mano_dir: str | None = None) -> np.ndarray:
    hand_pose = np.asarray(pred_hand_pose, dtype=np.float32)
    betas = np.asarray(pred_betas, dtype=np.float32)
    if hand_pose.ndim != 3 or hand_pose.shape[0] != 2 or hand_pose.shape[-1] != 45:
        raise ValueError(f"Expected pred_hand_pose shape (2, T, 45), got {hand_pose.shape}")
    if betas.ndim != 3 or betas.shape[0] != 2 or betas.shape[-1] != 10:
        raise ValueError(f"Expected pred_betas shape (2, T, 10), got {betas.shape}")
    if hand_pose.shape[1] != betas.shape[1]:
        raise ValueError(f"Hand pose / betas frame mismatch: {hand_pose.shape} vs {betas.shape}")

    num_frames = int(hand_pose.shape[1])
    output = np.empty((num_frames,) + MANO_SAMPLE_SHAPE, dtype=np.float32)
    output[:, 0, :MANO_PCA_DIMS] = hand_pose_axis_angle_to_pca(
        hand_pose[0],
        side="left",
        mano_dir=mano_dir,
    )
    output[:, 1, :MANO_PCA_DIMS] = hand_pose_axis_angle_to_pca(
        hand_pose[1],
        side="right",
        mano_dir=mano_dir,
    )
    output[:, 0, MANO_PCA_DIMS:] = betas[0]
    output[:, 1, MANO_PCA_DIMS:] = betas[1]
    return output


def decode_mano_sample_array(mano_array) -> dict[str, np.ndarray]:
    array = np.asarray(mano_array, dtype=np.float32)
    if array.shape == MANO_SAMPLE_SHAPE:
        sample = array
    elif array.size == int(np.prod(MANO_SAMPLE_SHAPE)):
        sample = array.reshape(MANO_SAMPLE_SHAPE)
    else:
        raise ValueError(f"Expected MANO sample shape {MANO_SAMPLE_SHAPE} or flat size 110, got {array.shape}")
    return {
        "left_pose_pca": sample[0, :MANO_PCA_DIMS].astype(np.float32, copy=False),
        "left_betas": sample[0, MANO_PCA_DIMS:].astype(np.float32, copy=False),
        "right_pose_pca": sample[1, :MANO_PCA_DIMS].astype(np.float32, copy=False),
        "right_betas": sample[1, MANO_PCA_DIMS:].astype(np.float32, copy=False),
    }


def rot6d_to_rotmat(rot6d) -> np.ndarray:
    array = np.asarray(rot6d, dtype=np.float32)
    if array.shape[-1] != 6:
        raise ValueError(f"Expected rot6d shape (..., 6), got {array.shape}")
    flat = array.reshape(-1, 6)
    mats = np.empty((flat.shape[0], 3, 3), dtype=np.float32)
    for idx, row in enumerate(flat):
        a1 = row[:3]
        a2 = row[3:]
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        a2 = a2 - np.dot(b1, a2) * b1
        b2 = a2 / (np.linalg.norm(a2) + 1e-8)
        b3 = np.cross(b1, b2)
        mats[idx] = np.stack([b1, b2, b3], axis=1)
    return mats.reshape(array.shape[:-1] + (3, 3)).astype(np.float32)


def rot6d_to_axis_angle(rot6d) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    rotmat = rot6d_to_rotmat(rot6d)
    flat = rotmat.reshape(-1, 3, 3)
    rotvec = Rotation.from_matrix(flat).as_rotvec().astype(np.float32)
    return rotvec.reshape(rotmat.shape[:-2] + (3,))


def build_manopth_models(
    device,
    *,
    mano_dir: str | None = None,
    center_idx: int = MANO_CENTER_IDX,
    flat_hand_mean: bool = MANO_FLAT_HAND_MEAN,
    ncomps: int = MANO_PCA_DIMS,
):
    try:
        from manopth.manolayer import ManoLayer
    except ImportError as error:
        raise ImportError("manopth is required for MANO replay. Install it in the runtime environment.") from error

    mano_root = str(_resolve_mano_root(mano_dir))
    common_kwargs = {
        "mano_root": mano_root,
        "use_pca": True,
        "ncomps": ncomps,
        "flat_hand_mean": flat_hand_mean,
        "center_idx": center_idx,
    }
    mano_right = ManoLayer(side="right", **common_kwargs).to(device)
    mano_left = ManoLayer(side="left", **common_kwargs).to(device)
    return mano_right, mano_left


def run_manopth_mano(
    mano_layer,
    *,
    wrist_world,
    root_rot_axis_angle,
    hand_pose_pca,
    betas,
    device,
):
    import torch

    wrist = torch.as_tensor(np.asarray(wrist_world), dtype=torch.float32, device=device).reshape(-1, 3)
    root = torch.as_tensor(np.asarray(root_rot_axis_angle), dtype=torch.float32, device=device).reshape(-1, 3)
    pose = torch.as_tensor(np.asarray(hand_pose_pca), dtype=torch.float32, device=device).reshape(-1, MANO_PCA_DIMS)
    shape = torch.as_tensor(np.asarray(betas), dtype=torch.float32, device=device).reshape(-1, MANO_BETA_DIMS)
    pose_coeffs = torch.cat([root, pose], dim=-1)

    with torch.no_grad():
        verts, joints = mano_layer(pose_coeffs, shape)
        wrist_offset = wrist - joints[:, MANO_CENTER_IDX, :]
        verts = verts + wrist_offset[:, None, :]
        joints = joints + wrist_offset[:, None, :]
    return verts.detach().cpu().numpy(), joints.detach().cpu().numpy()


def mano_meta_fields() -> dict[str, object]:
    return {
        "mano_schema": MANO_SCHEMA,
        "mano_pose_representation": "per_hand_pca_45d",
        "mano_betas_representation": "per_hand_betas_10d",
        "mano_flat_hand_mean": True,
        "mano_center_idx": MANO_CENTER_IDX,
        "mano_root_rotation_source": "lowdim_root_rot6d",
        "mano_wrist_translation_source": "lowdim_wrist_world",
    }
