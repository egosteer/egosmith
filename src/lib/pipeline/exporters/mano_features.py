"""MANO model construction + forward pass and per-frame wrist/hand joint-state features.

Optionally ray-scales the computed joints to match the Any4D depth map (fail-open hand-depth
alignment) before assembling the wrist-state / hand-state vectors.
"""

import numpy as np
import torch

from lib.pipeline.hands.hand_depth_align import align_hand_joints
from .webdataset_geometry import axis_angle_to_rot6d

FINGERTIP_INDICES = [4, 8, 12, 16, 20]


def run_mano_forward(mano_model, trans, root_orient, hand_pose, betas, device):
    """Run MANO forward pass, return joints (1, T, J, 3) on CPU."""
    from hawor.utils.geometry import aa_to_rotmat

    batch_size, num_frames, _ = root_orient.shape
    num_joints = 15

    params = {
        "global_orient": aa_to_rotmat(root_orient.reshape(batch_size * num_frames, 3)).view(batch_size * num_frames, 1, 3, 3),
        "hand_pose": aa_to_rotmat(hand_pose.reshape(batch_size * num_frames * num_joints, 3)).view(batch_size * num_frames, num_joints, 3, 3),
        "transl": trans.reshape(batch_size * num_frames, 3),
        "betas": betas.reshape(batch_size * num_frames, -1),
    }

    with torch.no_grad():
        output = mano_model(**{k: v.float().to(device) for k, v in params.items()}, pose2rot=False)

    return output.joints.reshape(batch_size, num_frames, -1, 3).cpu()


def build_mano_models(device, mano_dir=None):
    """Create right and left MANO models."""
    from lib.models.mano_wrapper import MANO
    from lib.pipeline.hands.mano_runtime import resolve_mano_model_dir

    if mano_dir is None:
        mano_right_dir = str(resolve_mano_model_dir(is_right=True))
        mano_left_dir = str(resolve_mano_model_dir(is_right=False))
    else:
        mano_right_dir = mano_dir
        mano_left_dir = mano_dir

    mano_right = MANO(
        data_dir=mano_right_dir,
        model_path=mano_right_dir,
        gender="neutral",
        num_hand_joints=15,
        create_body_pose=False,
    ).to(device)

    mano_left = MANO(
        data_dir=mano_left_dir,
        model_path=mano_left_dir,
        gender="neutral",
        num_hand_joints=15,
        create_body_pose=False,
        is_rhand=False,
    ).to(device)
    mano_left.shapedirs[:, 0, :] *= -1
    return mano_right, mano_left


def _compute_wrist_state(left_joints, right_joints, pred_rot):
    rot6d = axis_angle_to_rot6d(pred_rot.float())
    return torch.cat(
        [left_joints[:, 0, :].float(), right_joints[:, 0, :].float(), rot6d[0], rot6d[1]],
        dim=-1,
    )


def _compute_hand_joints(mano_model, pred_trans, pred_rot, pred_hand_pose, pred_betas, hand_index, device):
    num_frames = int(pred_trans.shape[1])
    hand_pose = pred_hand_pose[hand_index].float().reshape(1, num_frames, 15, 3)
    output = run_mano_forward(
        mano_model,
        pred_trans[hand_index].float().unsqueeze(0),
        pred_rot[hand_index].float().unsqueeze(0),
        hand_pose,
        pred_betas[hand_index].float().unsqueeze(0),
        device,
    )
    return output[0]


def _compute_hand_state(left_joints, right_joints):
    num_frames = int(left_joints.shape[0])
    left_tips = left_joints[:, FINGERTIP_INDICES, :]
    right_tips = right_joints[:, FINGERTIP_INDICES, :]
    return torch.cat(
        [left_tips.reshape(num_frames, 15), right_tips.reshape(num_frames, 15)],
        dim=-1,
    )


def _compute_joint_states(pred_trans, pred_rot, pred_hand_pose, pred_betas, mano_right, mano_left, device,
                          align_ctx=None):
    num_frames = int(pred_trans.shape[1])
    right_joints = _compute_hand_joints(
        mano_right,
        pred_trans,
        pred_rot,
        pred_hand_pose,
        pred_betas,
        hand_index=1,
        device=device,
    )
    left_joints = _compute_hand_joints(
        mano_left,
        pred_trans,
        pred_rot,
        pred_hand_pose,
        pred_betas,
        hand_index=0,
        device=device,
    )
    # Optional: ray-scale hand joints so their depth matches the Any4D depth map
    # (fail-open; unchanged when disabled or artifacts missing).
    if align_ctx is not None and align_ctx.get("cfg") is not None and align_ctx["cfg"].enable:
        left_np, right_np, _diag = align_hand_joints(
            left_joints.cpu().numpy(),
            right_joints.cpu().numpy(),
            align_ctx["extrinsics"],
            align_ctx["crop_dir"],
            align_ctx["presence"],
            align_ctx["cfg"],
        )
        left_joints = torch.from_numpy(np.ascontiguousarray(left_np)).to(left_joints)
        right_joints = torch.from_numpy(np.ascontiguousarray(right_np)).to(right_joints)
    wrist_state = _compute_wrist_state(left_joints, right_joints, pred_rot)
    hand_state = _compute_hand_state(left_joints, right_joints)
    return wrist_state, hand_state
