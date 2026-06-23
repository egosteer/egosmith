"""FPHA skeleton loading, geometry transforms, and right-hand MANO fitting."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from hawor.utils.geometry import aa_to_rotmat
from lib.pipeline.hands.mano_runtime import get_mano_cfg
from lib.models.mano_wrapper import MANO
from lib.pipeline.exporters.webdataset_features import _load_episode_camera_features


_CLIP_ID_RE = re.compile(r"^(?:FPHA_)?(Subject_\d+)_(.+)_(\d+)$")
_FPHA_REORDER_IDX = np.array(
    [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20],
    dtype=np.int64,
)
_DEPTH_TO_RGB_EXTRINSIC_MM = np.array(
    [
        [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
        [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
        [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def parse_fpha_clip_identity(clip_id: str) -> tuple[str, str, str]:
    match = _CLIP_ID_RE.match(str(clip_id))
    if match is None:
        raise ValueError(f"Invalid FPHA clip_id: {clip_id}")
    subject, action, trial = match.groups()
    return subject, action, trial


def resolve_fpha_skeleton_root(*, tar_root: str | Path | None = None, skeleton_root: str | Path | None = None) -> Path:
    if skeleton_root:
        requested = Path(skeleton_root).expanduser()
        candidates = [requested]
        name = requested.name
        parent = requested.parent
        if name == "Hand_pose_annotation_v1_1":
            candidates.append(parent / "Hand_pose_annotation_v1")
        elif name == "Hand_pose_annotation_v1":
            candidates.append(parent / "Hand_pose_annotation_v1_1")

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_dir():
                return resolved
        raise FileNotFoundError(
            "FPHA skeleton_root not found: "
            f"{requested.resolve()}; tried {[str(candidate.resolve()) for candidate in candidates]}"
        )

    if tar_root is None:
        raise ValueError("Either tar_root or skeleton_root must be provided")

    tar_root_path = Path(tar_root).expanduser().resolve()
    dataset_root = tar_root_path.parent.parent
    candidates = [
        dataset_root / "Hand_pose_annotation_v1_1",
        dataset_root / "Hand_pose_annotation_v1",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Failed to resolve FPHA skeleton root from {tar_root_path}; tried {candidates}"
    )


def resolve_fpha_skeleton_path(
    *,
    clip_id: str,
    skeleton_root: str | Path | None = None,
    tar_root: str | Path | None = None,
    subject: str | None = None,
    action: str | None = None,
    trial: str | int | None = None,
) -> Path:
    if subject is None or action is None or trial is None:
        subject, action, parsed_trial = parse_fpha_clip_identity(clip_id)
        trial = parsed_trial if trial is None else trial
    root = resolve_fpha_skeleton_root(tar_root=tar_root, skeleton_root=skeleton_root)
    return (root / str(subject) / str(action) / str(trial) / "skeleton.txt").resolve()


def load_fpha_skeleton_sequence(
    *,
    clip_id: str,
    skeleton_root: str | Path | None = None,
    tar_root: str | Path | None = None,
    subject: str | None = None,
    action: str | None = None,
    trial: str | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    skeleton_path = resolve_fpha_skeleton_path(
        clip_id=clip_id,
        skeleton_root=skeleton_root,
        tar_root=tar_root,
        subject=subject,
        action=action,
        trial=trial,
    )
    if not skeleton_path.is_file():
        raise FileNotFoundError(f"FPHA skeleton.txt not found: {skeleton_path}")

    skeleton_vals = np.loadtxt(skeleton_path, dtype=np.float32)
    if skeleton_vals.ndim == 1:
        skeleton_vals = skeleton_vals[None, :]
    if skeleton_vals.shape[1] != 64:
        raise ValueError(f"Unexpected skeleton.txt shape {skeleton_vals.shape} in {skeleton_path}")
    frame_ids = skeleton_vals[:, 0].astype(np.int64)
    joints_depth_mm = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, 3)
    joints_depth_mm = joints_depth_mm[:, _FPHA_REORDER_IDX, :]
    return frame_ids, joints_depth_mm


def _align_skeleton_rows(frame_ids: np.ndarray, joints_depth_mm: np.ndarray, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    aligned = np.zeros((num_frames, 21, 3), dtype=np.float32)
    valid = np.zeros((num_frames,), dtype=bool)

    frame_ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    joints_depth_mm = np.asarray(joints_depth_mm, dtype=np.float32)
    if len(frame_ids) != len(joints_depth_mm):
        raise ValueError("frame_ids and joints_depth_mm length mismatch")

    if len(frame_ids) == 0:
        return aligned, valid

    candidate_ids = frame_ids.copy()
    if candidate_ids.min() == 1 and candidate_ids.max() <= num_frames:
        candidate_ids = candidate_ids - 1
    elif candidate_ids.min() != 0 or candidate_ids.max() >= num_frames:
        candidate_ids = np.arange(len(frame_ids), dtype=np.int64)

    for row_idx, frame_idx in enumerate(candidate_ids):
        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx >= num_frames:
            continue
        aligned[frame_idx] = joints_depth_mm[row_idx]
        valid[frame_idx] = True
    return aligned, valid


def depth_joints_to_rgb_camera_meters(joints_depth_mm: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints_depth_mm, dtype=np.float32)
    hom = np.concatenate([joints, np.ones(joints.shape[:-1] + (1,), dtype=np.float32)], axis=-1)
    rgb_mm = np.einsum("ij,tkj->tki", _DEPTH_TO_RGB_EXTRINSIC_MM, hom)[..., :3]
    return rgb_mm.astype(np.float32) * 1e-3


def load_sequence_c2w(seq_folder: str | Path, *, clip_id: str, num_frames: int) -> np.ndarray:
    seq_folder_path = Path(seq_folder)
    slam_files = sorted((seq_folder_path / "SLAM").glob("hawor_slam_w_scale_*.npz"))
    if not slam_files:
        raise FileNotFoundError(f"Missing FPHA SLAM output under {seq_folder_path / 'SLAM'}")

    extrinsics, _intrinsic = _load_episode_camera_features(
        {"crop_dir": str(seq_folder_path), "episode_id": clip_id},
        num_frames,
    )
    return np.linalg.inv(extrinsics).astype(np.float32)


def camera_joints_to_world(joints_rgb_camera_m: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints_rgb_camera_m, dtype=np.float32)
    hom = np.concatenate([joints, np.ones(joints.shape[:-1] + (1,), dtype=np.float32)], axis=-1)
    world = np.einsum("tij,tkj->tki", c2w, hom)[..., :3]
    return world.astype(np.float32)


def build_fpha_world_targets(
    *,
    clip_id: str,
    seq_folder: str | Path,
    num_frames: int,
    tar_root: str | Path | None = None,
    skeleton_root: str | Path | None = None,
    subject: str | None = None,
    action: str | None = None,
    trial: str | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frame_ids, joints_depth_mm = load_fpha_skeleton_sequence(
        clip_id=clip_id,
        skeleton_root=skeleton_root,
        tar_root=tar_root,
        subject=subject,
        action=action,
        trial=trial,
    )
    aligned_depth_mm, valid = _align_skeleton_rows(frame_ids, joints_depth_mm, num_frames)
    if not valid.any():
        raise RuntimeError(f"No usable FPHA skeleton frames for {clip_id}")
    joints_rgb_camera_m = depth_joints_to_rgb_camera_meters(aligned_depth_mm)
    c2w = load_sequence_c2w(seq_folder, clip_id=clip_id, num_frames=num_frames)
    joints_world = camera_joints_to_world(joints_rgb_camera_m, c2w)
    return joints_world, valid


def _build_right_mano(device: torch.device) -> MANO:
    mano = MANO(**get_mano_cfg(is_right=True)).to(device)
    mano.eval()
    return mano


def _compute_palm_frame(joints: np.ndarray) -> np.ndarray:
    wrist = joints[0]
    index_mcp = joints[5] - wrist
    middle_mcp = joints[9] - wrist
    pinky_mcp = joints[17] - wrist

    forward = middle_mcp / (np.linalg.norm(middle_mcp) + 1e-8)
    normal = np.cross(index_mcp, pinky_mcp)
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    lateral = np.cross(forward, normal)
    lateral = lateral / (np.linalg.norm(lateral) + 1e-8)
    normal = np.cross(lateral, forward)
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    return np.stack([forward, lateral, normal], axis=1).astype(np.float32)


def _estimate_initial_global_rot(target_world_joints: np.ndarray, canonical_joints: np.ndarray) -> np.ndarray:
    canonical_frame = _compute_palm_frame(canonical_joints)
    canonical_to_world = []
    for joints in np.asarray(target_world_joints, dtype=np.float32):
        target_frame = _compute_palm_frame(joints)
        rotmat = target_frame @ canonical_frame.T
        canonical_to_world.append(rotmat.astype(np.float32))
    rotmats = np.stack(canonical_to_world, axis=0)
    return Rotation.from_matrix(rotmats).as_rotvec().astype(np.float32)


def _uniform_frame_subset(num_frames: int, sample_size: int) -> np.ndarray:
    if num_frames <= sample_size:
        return np.arange(num_frames, dtype=np.int64)
    return np.linspace(0, num_frames - 1, sample_size, dtype=np.int64)


def _build_joint_weights(device: torch.device) -> torch.Tensor:
    weights = np.ones((21,), dtype=np.float32)
    weights[0] = 2.0
    weights[[5, 9, 13, 17, 1]] = 2.0
    weights[[4, 8, 12, 16, 20]] = 3.0
    return torch.as_tensor(weights, dtype=torch.float32, device=device).view(1, 21, 1)


def _forward_right_mano(
    mano: MANO,
    global_orient: torch.Tensor,
    hand_pose: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    num_frames = int(global_orient.shape[0])
    global_orient_rot = aa_to_rotmat(global_orient).view(num_frames, 1, 3, 3)
    hand_pose_rot = aa_to_rotmat(hand_pose.view(num_frames * 15, 3)).view(num_frames, 15, 3, 3)
    output = mano(
        global_orient=global_orient_rot,
        hand_pose=hand_pose_rot,
        betas=betas,
        pose2rot=False,
    )
    return output.joints[:, :21, :]


def _align_wrist_translation(target: torch.Tensor, joints_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    transl = target[:, 0, :] - joints_local[:, 0, :]
    joints_world = joints_local + transl[:, None, :]
    return transl, joints_world


def fit_right_hand_mano_sequence(
    target_world_joints: np.ndarray,
    *,
    device: str | torch.device = "cuda:0",
    num_iters: int = 180,
    lr: float = 1e-2,
    pose_reg: float = 1e-4,
    shape_reg: float = 1e-3,
    temporal_reg: float = 1e-3,
    shape_iters: int = 120,
    shape_sample_size: int = 96,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = torch.as_tensor(np.asarray(target_world_joints), dtype=torch.float32)
    if target.ndim != 3 or target.shape[1:] != (21, 3):
        raise ValueError(f"Expected target_world_joints shape (T, 21, 3), got {tuple(target.shape)}")

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    target = target.to(requested_device)
    num_frames = int(target.shape[0])
    mano = _build_right_mano(requested_device)
    joint_weights = _build_joint_weights(requested_device)

    with torch.no_grad():
        canonical_joints = _forward_right_mano(
            mano,
            torch.zeros((1, 3), dtype=torch.float32, device=requested_device),
            torch.zeros((1, 45), dtype=torch.float32, device=requested_device),
            torch.zeros((1, 10), dtype=torch.float32, device=requested_device),
        )[0].detach().cpu().numpy()

    init_global_rot_np = _estimate_initial_global_rot(target.detach().cpu().numpy(), canonical_joints)
    init_global_rot = torch.as_tensor(init_global_rot_np, dtype=torch.float32, device=requested_device)

    shared_betas = torch.zeros((1, 10), dtype=torch.float32, device=requested_device)
    if shape_iters > 0 and num_frames > 0:
        shape_indices = _uniform_frame_subset(num_frames, int(shape_sample_size))
        shape_target = target[shape_indices]
        shape_global = init_global_rot[shape_indices]
        shape_pose = torch.zeros((len(shape_indices), 45), dtype=torch.float32, device=requested_device)
        shape_betas = torch.nn.Parameter(torch.zeros((1, 10), dtype=torch.float32, device=requested_device))
        shape_optimizer = torch.optim.Adam([shape_betas], lr=5e-3)

        for _ in range(int(shape_iters)):
            shape_optimizer.zero_grad(set_to_none=True)
            local_joints = _forward_right_mano(
                mano,
                shape_global,
                shape_pose,
                shape_betas.expand(len(shape_indices), -1),
            )
            _trans, joints_world = _align_wrist_translation(shape_target, local_joints)
            loss = ((joints_world - shape_target).abs() * joint_weights).mean()
            loss = loss + float(shape_reg) * shape_betas.square().mean()
            loss.backward()
            shape_optimizer.step()
        shared_betas = shape_betas.detach()

    global_orient_all = torch.empty((num_frames, 3), dtype=torch.float32, device=requested_device)
    hand_pose_all = torch.empty((num_frames, 45), dtype=torch.float32, device=requested_device)
    transl_all = torch.empty((num_frames, 3), dtype=torch.float32, device=requested_device)

    chunk_size = max(1, int(chunk_size))
    for chunk_start in range(0, num_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_frames)
        chunk_target = target[chunk_start:chunk_end]
        chunk_count = int(chunk_end - chunk_start)

        global_orient = torch.nn.Parameter(init_global_rot[chunk_start:chunk_end].clone())
        hand_pose = torch.nn.Parameter(torch.zeros((chunk_count, 45), dtype=torch.float32, device=requested_device))
        optimizer = torch.optim.Adam([global_orient, hand_pose], lr=float(lr))

        phase_one = int(num_iters * 0.5)
        phase_two = int(num_iters * 0.3)
        phase_three = max(int(num_iters) - phase_one - phase_two, 1)
        phase_boundaries = (phase_one, phase_one + phase_two, phase_one + phase_two + phase_three)

        for iter_idx in range(int(num_iters)):
            if iter_idx == phase_boundaries[0] or iter_idx == phase_boundaries[1]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.3

            optimizer.zero_grad(set_to_none=True)
            local_joints = _forward_right_mano(
                mano,
                global_orient,
                hand_pose,
                shared_betas.expand(chunk_count, -1),
            )
            transl, joints_world = _align_wrist_translation(chunk_target, local_joints)

            joint_l1 = (joints_world - chunk_target).abs() * joint_weights
            fingertip_l2 = (
                joints_world[:, [4, 8, 12, 16, 20], :] - chunk_target[:, [4, 8, 12, 16, 20], :]
            ).norm(dim=-1).mean()
            mcp_l2 = (
                joints_world[:, [1, 5, 9, 13, 17], :] - chunk_target[:, [1, 5, 9, 13, 17], :]
            ).norm(dim=-1).mean()

            loss = joint_l1.mean() + 2.0 * fingertip_l2 + 1.0 * mcp_l2
            loss = loss + float(pose_reg) * hand_pose.square().mean()
            loss = loss + 5e-3 * (global_orient - init_global_rot[chunk_start:chunk_end]).square().mean()
            if chunk_count > 1 and temporal_reg > 0:
                loss = loss + float(temporal_reg) * (
                    hand_pose[1:].sub(hand_pose[:-1]).square().mean()
                    + global_orient[1:].sub(global_orient[:-1]).square().mean()
                    + transl[1:].sub(transl[:-1]).square().mean()
                )
            optimizer.step()

        with torch.no_grad():
            final_local = _forward_right_mano(
                mano,
                global_orient,
                hand_pose,
                shared_betas.expand(chunk_count, -1),
            )
            final_transl, _final_world = _align_wrist_translation(chunk_target, final_local)
            global_orient_all[chunk_start:chunk_end] = global_orient
            hand_pose_all[chunk_start:chunk_end] = hand_pose
            transl_all[chunk_start:chunk_end] = final_transl

    return (
        transl_all.detach().cpu().numpy().astype(np.float32),
        global_orient_all.detach().cpu().numpy().astype(np.float32),
        hand_pose_all.detach().cpu().numpy().astype(np.float32),
        shared_betas.detach().cpu().numpy().astype(np.float32),
    )


def build_fpha_right_hand_prediction(
    *,
    clip_id: str,
    seq_folder: str | Path,
    num_frames: int,
    tar_root: str | Path | None = None,
    skeleton_root: str | Path | None = None,
    subject: str | None = None,
    action: str | None = None,
    trial: str | int | None = None,
    device: str | torch.device = "cuda:0",
    num_iters: int = 120,
    lr: float = 5e-2,
    pose_reg: float = 1e-4,
    shape_reg: float = 1e-3,
    temporal_reg: float = 1e-3,
    shape_iters: int = 120,
    shape_sample_size: int = 96,
    chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    joints_world, valid = build_fpha_world_targets(
        clip_id=clip_id,
        seq_folder=seq_folder,
        num_frames=num_frames,
        tar_root=tar_root,
        skeleton_root=skeleton_root,
        subject=subject,
        action=action,
        trial=trial,
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        raise RuntimeError(f"No valid FPHA skeleton frames for {clip_id}")

    fit_transl, fit_root, fit_hand_pose, fit_betas = fit_right_hand_mano_sequence(
        joints_world[valid_indices],
        device=device,
        num_iters=num_iters,
        lr=lr,
        pose_reg=pose_reg,
        shape_reg=shape_reg,
        temporal_reg=temporal_reg,
        shape_iters=shape_iters,
        shape_sample_size=shape_sample_size,
        chunk_size=chunk_size,
    )

    pred_trans = np.zeros((num_frames, 3), dtype=np.float32)
    pred_rot = np.zeros((num_frames, 3), dtype=np.float32)
    pred_hand_pose = np.zeros((num_frames, 45), dtype=np.float32)
    pred_betas = np.zeros((num_frames, 10), dtype=np.float32)
    pred_valid = np.zeros((num_frames,), dtype=np.float32)

    pred_trans[valid_indices] = fit_transl
    pred_rot[valid_indices] = fit_root
    pred_hand_pose[valid_indices] = fit_hand_pose
    pred_betas[valid_indices] = np.repeat(fit_betas, len(valid_indices), axis=0)
    pred_valid[valid_indices] = 1.0
    return {
        "pred_trans": pred_trans,
        "pred_rot": pred_rot,
        "pred_hand_pose": pred_hand_pose,
        "pred_betas": pred_betas,
        "pred_valid": pred_valid,
        "valid_mask": valid,
    }
