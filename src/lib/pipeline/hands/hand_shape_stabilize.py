"""Per-clip hand-shape stabilization: constant betas + closed-form depth (trans) compensation.

HaWoR predicts per-frame MANO betas that wobble ~±4-5% in overall SIZE. The real hand shape is
constant within a clip; the wobble is monocular size-depth-ambiguity noise (2D constrains only the
size/depth RATIO, not absolute size). We therefore impose ONE per-clip shape (median betas) and
scale each frame's cam-frame translation by the shape-size ratio ``f``. Because scaling a global
size by ``f`` and the whole cam-frame translation by the same ``f`` scales every cam point about the
camera origin, the perspective projection is UNCHANGED for the global-size dimension:

    x = K (f X) / (f z) = K X / z

So the 2D overlay is preserved exactly for the (dominant, depth-ambiguous) size dimension; the only
residual is the proportion part of beta, which is 2D-constrained and already stable across frames →
small and smooth (no per-frame switching/jumps). Side effect: the compensated depth
``f_t * trans_z`` is DE-NOISED (the per-frame size wobble cancels), which also steadies the metric
anchor (it reads ``trans_z``).

Gated OFF by default; enable with HAWOR_HAND_SHAPE_STABILIZE=1.
"""
from __future__ import annotations

import json
import os

import numpy as np

from lib.utils.mano_numpy import load_mano, mano_forward

_REST_ORIENT = np.zeros(3, np.float64)
_REST_POSE = np.zeros((15, 3), np.float64)
_ZERO_TRANS = np.zeros(3, np.float64)


def hand_shape_stabilize_enabled() -> bool:
    return str(os.environ.get("HAWOR_HAND_SHAPE_STABILIZE", "")).strip().lower() in ("1", "true", "yes", "on")


def load_default_mano():
    """Load the repo MANO model for the size metric. Size is mirror-symmetric, so one model
    (RIGHT) is fine for the left/right size RATIO that drives the depth compensation."""
    # src/lib/pipeline/hands/hand_shape_stabilize.py -> 5 dirname() reach the repo
    # root (4 would stop at src/); _DATA lives at the repo root, not under src/.
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    return load_mano(os.path.join(here, "_DATA", "data", "mano", "MANO_RIGHT.pkl"))


def shape_size(mano, betas) -> float:
    """Global size proxy = mean joint-to-wrist distance of the REST-pose hand for these betas.

    Evaluated at the rest pose so it depends on shape only (pose-independent); the RATIO of two
    shapes' sizes is the global-scale factor used for the depth compensation.
    """
    _, joints = mano_forward(mano, betas, _REST_ORIENT, _REST_POSE, _ZERO_TRANS, return_joints=True)
    return float(np.mean(np.linalg.norm(joints - joints[0:1], axis=1)))


def stabilize_betas_trans(
    betas: np.ndarray,
    trans: np.ndarray,
    mano,
    *,
    f_lo: float = 0.5,
    f_hi: float = 2.0,
):
    """Stabilize ONE hand's shape over a clip.

    ``betas`` (T,10), ``trans`` (T,3) cam-frame. Returns (betas_out (T,10), trans_out (T,3), info):
      betas_out = median(betas) broadcast to all T frames (one constant shape).
      trans_out = f_t * trans, f_t = shape_size(b*) / shape_size(betas_t)  (closed form, clipped).
    Fail-open: on too few/invalid frames returns the inputs unchanged with info['applied']=False.
    """
    betas = np.asarray(betas, np.float64).reshape(-1, betas.shape[-1])
    trans = np.asarray(trans, np.float64).reshape(-1, 3)
    n = betas.shape[0]
    info = {"applied": False, "reason": "", "n_frames": int(n)}
    if n == 0 or trans.shape[0] != n:
        info["reason"] = "empty or mismatched"
        return betas, trans, info

    b_star = np.median(betas, axis=0)
    s_star = shape_size(mano, b_star)
    if not np.isfinite(s_star) or s_star <= 0:
        info["reason"] = "invalid median-shape size"
        return betas, trans, info

    f = np.ones(n, np.float64)
    for t in range(n):
        s_t = shape_size(mano, betas[t])
        if np.isfinite(s_t) and s_t > 1e-9:
            f[t] = float(np.clip(s_star / s_t, f_lo, f_hi))

    betas_out = np.repeat(b_star[None, :], n, axis=0)
    trans_out = trans * f[:, None]
    info.update(
        applied=True,
        b_star=b_star.astype(np.float32),
        f_min=float(f.min()), f_max=float(f.max()),
        size_var_before=float(np.var([shape_size(mano, betas[t]) for t in range(n)])),
    )
    return betas_out, trans_out, info


def _cam_chunk_path(seq_folder: str, hand_idx: int, frame_ck) -> str:
    frame_ck = np.asarray(frame_ck).reshape(-1)
    key = f"{int(frame_ck[0])}_{int(frame_ck[-1])}"
    return os.path.join(seq_folder, "cam_space", str(hand_idx), f"{key}.json")


def stabilize_cam_space_clip(seq_folder: str, frame_chunks_all, mano=None) -> dict:
    """Rewrite the cam_space chunk JSONs in place with the per-clip stabilized shape + depth.

    For each hand, concatenates all chunks' (init_betas, init_trans), runs ``stabilize_betas_trans``
    (median shape over the whole clip + per-frame trans×f), then writes the stabilized betas/trans
    back into each chunk JSON (preserving the (1,n,*) layout and untouched orient/pose). Persists the
    per-clip ``b*`` as a sidecar so the infiller/world step can reuse it for filled frames.

    Returns {hand_idx: info}. The caller should invalidate the cam_space cache afterwards.
    """
    mano = mano or load_default_mano()
    out = {}
    for hand_idx in (0, 1):
        entries = []  # (path, json_dict, n) in frame order
        betas_list, trans_list = [], []
        for frame_ck in frame_chunks_all.get(hand_idx, []):
            if np.asarray(frame_ck).reshape(-1).size == 0:
                continue
            path = _cam_chunk_path(seq_folder, hand_idx, frame_ck)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            betas = np.asarray(d["init_betas"], np.float64).reshape(-1, 10)
            trans = np.asarray(d["init_trans"], np.float64).reshape(-1, 3)
            if betas.shape[0] == 0 or betas.shape[0] != trans.shape[0]:
                continue
            entries.append((path, d, betas.shape[0]))
            betas_list.append(betas)
            trans_list.append(trans)
        if not entries:
            continue

        b_out, t_out, info = stabilize_betas_trans(
            np.concatenate(betas_list, 0), np.concatenate(trans_list, 0), mano,
        )
        if not info.get("applied"):
            out[hand_idx] = {"applied": False, "reason": info.get("reason", "")}
            continue

        off = 0
        for path, d, n in entries:
            d["init_betas"] = b_out[off:off + n][None, :, :].tolist()  # (1,n,10)
            d["init_trans"] = t_out[off:off + n][None, :, :].tolist()  # (1,n,3)
            off += n
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=1)

        b_star = info["b_star"]
        try:
            np.save(os.path.join(seq_folder, "cam_space", str(hand_idx), "shape_stabilized_beta.npy"),
                    np.asarray(b_star, np.float32))
        except Exception:
            pass
        out[hand_idx] = {"applied": True, "n_frames": off,
                         "f_min": info["f_min"], "f_max": info["f_max"]}
    return out
