"""Shared robust + temporally-smooth scalar-curve solver.

This is the single mechanism reused across the metric-scale pipeline:
  * hand-depth alignment  (α(t) so that α·z_hawor ≈ d_map)
  * camera scale s(t)      (smooth the per-keyframe DPVO-vs-depth scale)
  * Any4D batch stitching  (smooth the per-batch multiplicative scale)

It fits a per-sample scalar curve ``α(t)`` minimizing

    min_α  Σ_valid w(t)·(α(t)·a(t) - b(t))²  +  λ Σ (Δ²α)²

via Iteratively Reweighted Least Squares with a Geman-McClure robust weight.
The second-difference penalty makes α(t) temporally smooth: λ→∞ converges to the
best AFFINE α(t) (slow linear drift, no per-frame wiggle); per-frame snapping
(which would inject depth noise) is avoided. Insufficient valid data falls back
to a constant median(b/a).

``a`` and ``b`` are per-frame observations. For hand alignment a=z_hawor, b=d_map.
For camera scale, set a=1 and b=per-keyframe scale to simply smooth a noisy curve.
"""
from __future__ import annotations

import numpy as np


def second_difference(T: int) -> np.ndarray:
    """(T-2, T) second-difference operator; empty for T<3."""
    if T < 3:
        return np.zeros((0, T), np.float64)
    D = np.zeros((T - 2, T), np.float64)
    for i in range(T - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def gmof_weight(res: np.ndarray, sigma: float) -> np.ndarray:
    """IRLS weight for Geman-McClure: ρ'(x)/x = 2σ⁴/(σ²+x²)² (constant dropped)."""
    s2 = sigma * sigma
    return (s2 * s2) / np.square(s2 + np.square(res))


def fit_smooth_scale(
    a: np.ndarray,
    b: np.ndarray,
    valid: np.ndarray,
    *,
    lam: float,
    sigma: float,
    irls_iters: int = 5,
    min_valid_frames: int = 8,
    clip_lo: float = 1e-3,
    clip_hi: float = 1e3,
):
    """Smoothness-regularized robust fit of α(t) so that α(t)·a(t) ≈ b(t).

    Returns (alpha[T] float32, info dict). On insufficient valid frames or a failed
    solve, returns a constant median(b/a) and info['applied']=False.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    T = a.shape[0]
    info = {"applied": False, "reason": "", "n_valid": int(valid.sum())}

    r = np.where(valid & (a > 0), b / np.maximum(a, 1e-6), np.nan)
    r_med = float(np.nanmedian(r)) if np.any(valid) else 1.0
    if not np.isfinite(r_med) or r_med <= 0:
        r_med = 1.0

    if int(valid.sum()) < min_valid_frames:
        info["reason"] = f"insufficient valid frames ({int(valid.sum())}<{min_valid_frames})"
        return np.full(T, r_med, np.float32), info

    aa = np.where(valid, a, 0.0)
    bb = np.where(valid, b, 0.0)
    vmask = valid.astype(np.float64)

    D2 = second_difference(T)
    P = lam * (D2.T @ D2) if D2.size else np.zeros((T, T))
    P = P + 1e-6 * np.eye(T)  # tiny ridge for numerical PD

    alpha = np.full(T, r_med, np.float64)
    for _ in range(max(1, irls_iters)):
        res = alpha * aa - bb
        w = gmof_weight(res, sigma) * vmask
        A = np.diag(w * aa * aa)
        rhs = w * aa * bb
        try:
            alpha = np.linalg.solve(A + P, rhs)
        except np.linalg.LinAlgError:
            alpha = np.full(T, r_med, np.float64)
            info["reason"] = "linear solve failed; fell back to global median"
            break
    alpha = np.clip(alpha, clip_lo, clip_hi)

    res_after = (alpha * aa - bb)[valid]
    rel = np.abs(res_after) / np.maximum(np.abs(bb[valid]), 1e-6)
    a_v, b_v = a[valid], b[valid]
    corr = (
        float(np.corrcoef(a_v, b_v)[0, 1])
        if a_v.size > 1 and np.std(a_v) > 0 and np.std(b_v) > 0
        else float("nan")
    )
    info.update(
        applied=True,
        r_median=r_med,
        residual_rel_median=float(np.median(rel)) if rel.size else float("nan"),
        residual_rel_p90=float(np.percentile(rel, 90)) if rel.size else float("nan"),
        corr_a_b=corr,
        alpha_min=float(alpha.min()),
        alpha_max=float(alpha.max()),
    )
    return alpha.astype(np.float32), info
