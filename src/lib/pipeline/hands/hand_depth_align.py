"""Align reconstructed hand depth to the Any4D metric depth map (ray-scale + smooth α(t)).

Why: the metric ``scale`` fit in ``est_scale`` only calibrates the *camera trajectory*
on background pixels (hands are masked out); it does NOT change the hand's depth
relative to the camera. The hand's camera-frame depth comes from HaWoR's hand-size
prior, independent of Any4D, so hand-action depth is often inconsistent with the
metric depth map.

This module makes the hand depth agree with the depth map while:
  * keeping the 2D image overlay exact  -> we scale the hand *along the camera ray*
    (uniform scaling about the camera center: ``p' = cam + α·(p - cam)``), which
    leaves the image projection unchanged and only moves depth;
  * preserving action dynamics           -> α(t) is a temporally-SMOOTH scale
    (Whittaker / second-difference regularized, robust IRLS). With the second-
    difference penalty, λ→∞ converges to the best AFFINE α(t) (a slow linear drift,
    no wiggle); a strict constant global scale is used only as the insufficient-data
    fallback. Per-frame snapping (which would inject Any4D noise and make the hand
    size "breathe") is explicitly avoided.

The fit target is sampled from the metric depth map over the hand mask region; the
source is the camera-frame z of the reconstructed hand joints. Diagnostics (r(t),
residuals, a per-clip consistency score) are always produced for tuning / filtering.

Gated OFF by default; enable via env (see ``HandDepthAlignConfig.from_env``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:  # optional; only needed for mask/depth resolution matching
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from lib.pipeline.quality.quality_metrics import transform_points_world_to_camera
from lib.pipeline.slam.depth_artifacts import load_export_depths, _discover_track_range
from lib.pipeline.hands.scale_solver import fit_smooth_scale


@dataclass(frozen=True)
class HandDepthAlignConfig:
    enable: bool = False
    lam: float = 50.0            # smoothness weight (λ→∞ -> single global scale)
    sigma: float = 0.10          # GMoF robust scale, meters
    depth_min: float = 0.10      # valid depth band (m)
    depth_max: float = 3.00
    min_valid_frames: int = 8
    min_mask_pixels: int = 50
    irls_iters: int = 5

    @staticmethod
    def _f(name: str, default: float) -> float:
        raw = os.environ.get(name)
        try:
            return float(raw) if raw not in (None, "") else default
        except ValueError:
            return default

    @classmethod
    def from_env(cls) -> "HandDepthAlignConfig":
        raw = os.environ.get("HAWOR_HAND_DEPTH_ALIGN", "")
        enable = str(raw).strip().lower() in ("1", "true", "yes", "on")
        return cls(
            enable=enable,
            lam=cls._f("HAWOR_HDA_LAM", 50.0),
            sigma=cls._f("HAWOR_HDA_SIGMA", 0.10),
            depth_min=cls._f("HAWOR_HDA_DEPTH_MIN", 0.10),
            depth_max=cls._f("HAWOR_HDA_DEPTH_MAX", 3.00),
            min_valid_frames=int(cls._f("HAWOR_HDA_MIN_VALID", 8)),
            min_mask_pixels=int(cls._f("HAWOR_HDA_MIN_MASK_PX", 50)),
            irls_iters=int(cls._f("HAWOR_HDA_IRLS", 5)),
        )

    def signature(self) -> str:
        """Stable string for cache invalidation when alignment is enabled."""
        if not self.enable:
            return "off"
        return (f"on:lam{self.lam:g}:sig{self.sigma:g}:db{self.depth_min:g}-{self.depth_max:g}"
                f":mv{self.min_valid_frames}:mp{self.min_mask_pixels}")


def _cam_positions(extrinsics: np.ndarray) -> np.ndarray:
    """Camera centers in world from per-frame w2c extrinsics. extrinsics: (T,4,4)."""
    ext = np.asarray(extrinsics, dtype=np.float32).reshape(-1, 4, 4)
    R = ext[:, :3, :3]
    t = ext[:, :3, 3]
    # cam_pos = -R^T t
    return -np.einsum("tji,tj->ti", R, t).astype(np.float32)


def _load_hand_masks(crop_dir: str, num_frames: int) -> Optional[np.ndarray]:
    rng = _discover_track_range(crop_dir)
    if rng is None:
        return None
    s, e = rng
    path = Path(crop_dir) / f"tracks_{s}_{e}" / "model_masks.npy"
    if not path.is_file():
        return None
    try:
        masks = np.load(str(path), allow_pickle=True)
    except Exception:
        return None
    masks = np.asarray(masks)
    if masks.ndim == 4:  # (frames, n_hands, H, W) -> union over hands
        masks = masks.any(axis=1)
    if masks.ndim != 3 or masks.shape[0] < num_frames:
        return None
    return masks[:num_frames]


def _resize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    m = mask.astype(np.uint8)
    if m.shape == (h, w):
        return m > 0
    if cv2 is not None:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return m > 0
    # nearest-neighbour fallback without cv2
    ys = (np.linspace(0, mask.shape[0] - 1, h)).astype(int)
    xs = (np.linspace(0, mask.shape[1] - 1, w)).astype(int)
    return mask[np.ix_(ys, xs)] > 0


def sample_depth_targets(
    left_joints: np.ndarray,
    right_joints: np.ndarray,
    extrinsics: np.ndarray,
    depth_maps: np.ndarray,
    masks: np.ndarray,
    presence: np.ndarray,
    cfg: HandDepthAlignConfig,
):
    """Per-frame (z_hawor, d_map, valid).

    z_hawor : robust cam-frame z of the hand joints (both hands where present).
    d_map   : robust median of the metric depth map over the hand-mask region.
    """
    T = depth_maps.shape[0]
    z = np.full(T, np.nan, np.float32)
    d = np.full(T, np.nan, np.float32)
    valid = np.zeros(T, bool)
    pres = np.asarray(presence)

    for t in range(T):
        ext = extrinsics[t]
        pts = []
        # presence bit0 = left, bit1 = right (see _compute_presence_per_frame)
        if int(pres[t]) & 1:
            pts.append(left_joints[t])
        if int(pres[t]) & 2:
            pts.append(right_joints[t])
        if not pts:
            continue
        pts = np.concatenate(pts, axis=0)
        cam = transform_points_world_to_camera(pts, ext)
        cam_z = cam[:, 2]
        cam_z = cam_z[np.isfinite(cam_z) & (cam_z > 0)]
        if cam_z.size == 0:
            continue
        z_t = float(np.median(cam_z))

        dm = depth_maps[t]
        mk = _resize_mask(masks[t], dm.shape)
        sel = mk & np.isfinite(dm) & (dm >= cfg.depth_min) & (dm <= cfg.depth_max)
        if int(sel.sum()) < cfg.min_mask_pixels:
            continue
        d_t = float(np.median(dm[sel]))

        z[t], d[t], valid[t] = z_t, d_t, True
    return z, d, valid


def fit_alpha_smooth(z: np.ndarray, d: np.ndarray, valid: np.ndarray, cfg: HandDepthAlignConfig):
    """Smoothness-regularized robust fit of α(t) so that α(t)·z(t) ≈ d(t).

    Thin wrapper over the shared ``fit_smooth_scale`` core (second-difference +
    Geman-McClure IRLS). λ→∞ -> best affine α(t); insufficient valid frames ->
    constant median(d/z). Returns (alpha[T], info) with ``corr_z_d`` aliased.
    """
    alpha, info = fit_smooth_scale(
        z, d, valid,
        lam=cfg.lam,
        sigma=cfg.sigma,
        irls_iters=cfg.irls_iters,
        min_valid_frames=cfg.min_valid_frames,
    )
    if "corr_a_b" in info:
        info["corr_z_d"] = info.pop("corr_a_b")
    return alpha, info


def apply_ray_scale(joints: np.ndarray, cam_pos: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """joints'(t) = cam_pos(t) + α(t)·(joints(t) - cam_pos(t)).  joints:(T,J,3)."""
    a = alpha.reshape(-1, 1, 1)
    cp = cam_pos.reshape(-1, 1, 3)
    return (cp + a * (joints - cp)).astype(joints.dtype)


def align_hand_joints(
    left_joints: np.ndarray,
    right_joints: np.ndarray,
    extrinsics: np.ndarray,
    crop_dir: str,
    presence: np.ndarray,
    cfg: HandDepthAlignConfig,
):
    """Ray-scale hand joints so their depth matches the Any4D depth map.

    Returns (left', right', diag). On any missing artifact / insufficient data the
    joints are returned unchanged with diag['applied']=False (fail-open).
    """
    diag = {"applied": False, "reason": ""}
    if not cfg.enable:
        diag["reason"] = "disabled"
        return left_joints, right_joints, diag

    left = np.asarray(left_joints, dtype=np.float32)
    right = np.asarray(right_joints, dtype=np.float32)
    T = left.shape[0]

    try:
        depth_maps = load_export_depths(crop_dir, T)  # (T,H,W) meters
    except Exception as error:
        diag["reason"] = f"no depth artifact: {error}"
        return left_joints, right_joints, diag
    masks = _load_hand_masks(crop_dir, T)
    if masks is None:
        diag["reason"] = "no hand masks"
        return left_joints, right_joints, diag

    z, d, valid = sample_depth_targets(left, right, extrinsics, depth_maps, masks, presence, cfg)
    alpha, info = fit_alpha_smooth(z, d, valid, cfg)
    diag.update(info)
    if not info.get("applied", False):
        return left_joints, right_joints, diag

    cam_pos = _cam_positions(extrinsics)
    left_a = apply_ray_scale(left, cam_pos, alpha)
    right_a = apply_ray_scale(right, cam_pos, alpha)

    diag["alpha"] = alpha
    diag["r"] = np.where(valid & (z > 0), d / np.maximum(z, 1e-6), np.nan).astype(np.float32)
    diag["z_hawor"] = z
    diag["d_map"] = d
    diag["valid"] = valid
    _write_sidecar(crop_dir, diag)
    return left_a, right_a, diag


def _write_sidecar(crop_dir: str, diag: dict) -> None:
    rng = _discover_track_range(crop_dir)
    if rng is None:
        return
    s, e = rng
    slam_dir = Path(crop_dir) / "SLAM"
    try:
        slam_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            slam_dir / f"hand_depth_align_{s}_{e}.npz",
            alpha=diag.get("alpha"),
            r=diag.get("r"),
            z_hawor=diag.get("z_hawor"),
            d_map=diag.get("d_map"),
            valid=diag.get("valid"),
            residual_rel_median=diag.get("residual_rel_median", np.nan),
            residual_rel_p90=diag.get("residual_rel_p90", np.nan),
            corr_z_d=diag.get("corr_z_d", np.nan),
        )
    except Exception:
        pass
