"""Torch-free numpy MANO LBS forward.

Lifted (verbatim) from scripts/visualize_alignment_rerun.py so both the rerun viz and the hand-shape
stabilization (lib/pipeline/hands/hand_shape_stabilize.py) share one implementation. Loads the
chumpy-backed MANO_*.pkl without the chumpy package and runs a plain numpy LBS forward.
"""
from __future__ import annotations

import pickle
import sys
import types

import numpy as np


def _install_chumpy_stub():
    """Let pickle load the chumpy-backed MANO_*.pkl without the chumpy package.

    The pkl references chumpy.ch.Ch and chumpy.reordering.Select. We restore them as
    plain holders; their numpy payload lives in __dict__ ('x' for Ch; 'a'/'idxs' for
    Select). We extract the arrays manually afterwards.
    """
    if "chumpy" in sys.modules:
        return
    chumpy = types.ModuleType("chumpy")
    chumpy.__path__ = []
    ch = types.ModuleType("chumpy.ch")
    reordering = types.ModuleType("chumpy.reordering")

    class Ch:
        def __setstate__(self, s):
            self.__dict__.update(s if isinstance(s, dict) else {})

    class Select:
        def __setstate__(self, s):
            self.__dict__.update(s if isinstance(s, dict) else {})

    ch.Ch = Ch
    reordering.Select = Select
    chumpy.Ch = Ch
    chumpy.ch = ch
    chumpy.reordering = reordering
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.ch"] = ch
    sys.modules["chumpy.reordering"] = reordering


def _ch_data(obj):
    """Return the numpy payload of a chumpy Ch / Select holder, else np.asarray(obj)."""
    if hasattr(obj, "__dict__"):
        d = obj.__dict__
        if "x" in d:  # Ch
            return np.asarray(d["x"])
        if "a" in d and "idxs" in d:  # Select (reordering of a.x)
            base = _ch_data(d["a"]).flatten()
            return base[np.asarray(d["idxs"])]
    return np.asarray(obj)


def load_mano(mano_pkl, n_betas=10):
    """Load MANO components needed for an LBS forward (numpy)."""
    _install_chumpy_stub()
    with open(mano_pkl, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    v_template = _ch_data(d["v_template"]).reshape(778, 3).astype(np.float64)
    # shapedirs: select picks (778,3,10) out of the full (778,3,20)
    shapedirs = _ch_data(d["shapedirs"]).reshape(778, 3, -1)[:, :, :n_betas].astype(np.float64)
    posedirs = _ch_data(d["posedirs"]).reshape(778, 3, -1).astype(np.float64)  # (778,3,135)
    J_reg = d["J_regressor"]
    J_regressor = (J_reg.toarray() if hasattr(J_reg, "toarray") else np.asarray(J_reg)).astype(np.float64)
    weights = _ch_data(d["weights"]).reshape(778, 16).astype(np.float64)
    kintree = np.asarray(d["kintree_table"]).astype(np.int64)  # (2,16)
    faces = np.asarray(d["f"]).astype(np.int64)  # (1538,3)
    parents = kintree[0].copy()
    parents[0] = -1
    return dict(
        v_template=v_template, shapedirs=shapedirs, posedirs=posedirs,
        J_regressor=J_regressor, weights=weights, parents=parents, faces=faces,
    )


def _rodrigues(rotvecs):
    """(N,3) axis-angle -> (N,3,3) rotation matrices."""
    rotvecs = np.asarray(rotvecs, np.float64).reshape(-1, 3)
    theta = np.linalg.norm(rotvecs, axis=1, keepdims=True)
    theta_safe = np.where(theta < 1e-8, 1.0, theta)
    k = rotvecs / theta_safe
    K = np.zeros((rotvecs.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = np.eye(3)[None]
    s = np.sin(theta)[:, :, None]
    c = np.cos(theta)[:, :, None]
    R = I + s * K + (1 - c) * (K @ K)
    R[theta[:, 0] < 1e-8] = np.eye(3)
    return R


def mano_forward(mano, betas, global_orient_aa, hand_pose_aa, transl, return_joints=False):
    """Numpy MANO LBS forward. Mirrors smplx MANOLayer with pose2rot=True semantics,
    flat_hand_mean (hands_mean NOT added — pipeline uses pose2rot=False on rotmats,
    where hands_mean is never applied).

    betas: (10,)  global_orient_aa: (3,)  hand_pose_aa: (15,3)  transl: (3,)
    returns vertices (778,3) in the same frame as transl; if return_joints, also the
    16 MANO joints (J_regressor @ posed verts) — joint 0 is the wrist root_loc used by
    cam2world_convert.
    """
    v_template = mano["v_template"]
    shapedirs = mano["shapedirs"]
    posedirs = mano["posedirs"]
    J_regressor = mano["J_regressor"]
    weights = mano["weights"]
    parents = mano["parents"]

    betas = np.asarray(betas, np.float64).reshape(-1)[: shapedirs.shape[2]]
    # shape blend
    v_shaped = v_template + np.einsum("vck,k->vc", shapedirs, betas)  # (778,3)
    J = J_regressor @ v_shaped  # (16,3)

    full_pose_aa = np.concatenate(
        [np.asarray(global_orient_aa, np.float64).reshape(1, 3),
         np.asarray(hand_pose_aa, np.float64).reshape(15, 3)], axis=0)  # (16,3)
    R = _rodrigues(full_pose_aa)  # (16,3,3)

    # pose blend shapes: (R[1:] - I) flattened, dot posedirs
    pose_feat = (R[1:] - np.eye(3)[None]).reshape(-1)  # 15*9 = 135
    v_posed = v_shaped + np.einsum("vck,k->vc", posedirs, pose_feat)

    # build per-joint global transforms (rigid kinematic chain)
    G = np.zeros((16, 4, 4))
    G[0, :3, :3] = R[0]
    G[0, :3, 3] = J[0]
    G[0, 3, 3] = 1.0
    for i in range(1, 16):
        T_local = np.eye(4)
        T_local[:3, :3] = R[i]
        T_local[:3, 3] = J[i] - J[parents[i]]
        G[i] = G[parents[i]] @ T_local
    # remove rest-pose joint offset
    for i in range(16):
        Jh = np.array([J[i, 0], J[i, 1], J[i, 2], 0.0])
        G[i, :, 3] = G[i, :, 3] - G[i] @ Jh

    T = np.einsum("vj,jab->vab", weights, G)  # (778,4,4)
    v_h = np.concatenate([v_posed, np.ones((778, 1))], axis=1)  # (778,4)
    v_out = np.einsum("vab,vb->va", T, v_h)[:, :3]
    transl = np.asarray(transl, np.float64).reshape(1, 3)
    verts = v_out + transl
    if return_joints:
        joints = (J_regressor @ v_out) + transl  # (16,3); joint 0 = wrist root_loc
        return verts, joints
    return verts
