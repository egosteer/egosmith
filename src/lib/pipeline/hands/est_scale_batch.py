"""Batched metric-scale estimation (EgoSmith first-party).

`est_scale_hybrid_batch` is an EgoSmith addition: it keeps the exact algorithm and
convergence of HaWoR's per-keyframe `est_scale.est_scale_hybrid` (iterative median
init + Geman-McClure-robust 1D BFGS), but transfers all keyframes to the GPU in a
single pass and runs the N independent 1D optimizations together. This lives in a
first-party module (not in the obtained HaWoR base `est_scale.py`) so it ships as
EgoSmith code.
"""

import cv2
import numpy as np
import torch
from torchmin import minimize


def _gmof(x, sigma=100):
    """Geman-McClure robust error (same as est_scale.gmof; duplicated to keep this
    first-party module independent of the obtained HaWoR base est_scale.py)."""
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def est_scale_hybrid_gpu(slam_depth, pred_depth, sigma=0.5, msk=None, near_thresh=0,
                         far_thresh=10):
    """GPU variant of HaWoR's est_scale.est_scale_hybrid (single keyframe).

    Identical algorithm (iterative-median init + Geman-McClure-robust 1D BFGS);
    the only EgoSmith change vs upstream is running stage-2 BFGS on the GPU. Kept
    first-party (not in the obtained base est_scale.py) and used on the fallback
    path when the batched estimator returns NaN, so behavior matches what the
    pipeline currently produces.
    """
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    # Stage 1: Iterative steps
    s = pred_depth / slam_depth

    robust = (msk < 0.5) * (near_thresh < pred_depth) * (pred_depth < far_thresh)
    s_est = s[robust]
    scale = np.median(s_est)

    for _ in range(10):
        slam_depth_0 = slam_depth * scale
        robust = (msk < 0.5) * (0 < slam_depth_0) * (slam_depth_0 < far_thresh) * (near_thresh < pred_depth) * (pred_depth < far_thresh)
        s_est = s[robust]
        scale = np.median(s_est)

    # Stage 2: Robust optimization on GPU
    robust = (msk < 0.5) * (0 < slam_depth_0) * (slam_depth_0 < far_thresh) * (near_thresh < pred_depth) * (pred_depth < far_thresh)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pm = torch.from_numpy(pred_depth[robust]).to(device)
    sm = torch.from_numpy(slam_depth[robust]).to(device)

    def f(x):
        loss = sm * x - pm
        loss = _gmof(loss, sigma=sigma).mean()
        return loss

    x0 = torch.tensor([scale], device=device)
    result = minimize(f, x0, method='bfgs')
    scale = result.x.detach().cpu().item()

    return scale


def est_scale_hybrid_batch(slam_depths, pred_depths, sigma=0.5, masks=None,
                           near_thresh=0.4, far_thresh=0.7):
    """Batch scale estimation with single GPU transfer + N independent 1D BFGS.

    Same algorithm and convergence as est_scale_hybrid, but:
    - Stage 1 (iterative median) runs per-keyframe on CPU (unchanged)
    - Stage 2 transfers all filtered data to GPU in one batch, then runs
      N independent 1D BFGS optimizations (each with 1 scalar variable)

    This avoids N separate CPU->GPU transfers while keeping each optimization
    independent (no cross-keyframe coupling in the Hessian).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = len(slam_depths)

    # Stage 1: iterative median per keyframe (CPU, fast)
    init_scales = []
    sm_arrays = []
    pm_arrays = []
    for i in range(N):
        slam_depth = slam_depths[i]
        pred_depth = pred_depths[i]
        if masks is not None:
            msk = cv2.resize(masks[i], (pred_depth.shape[1], pred_depth.shape[0]))
        else:
            msk = np.zeros_like(pred_depth)

        s = pred_depth / slam_depth
        nt, ft = near_thresh, far_thresh
        robust = (msk < 0.5) * (nt < pred_depth) * (pred_depth < ft)
        s_est = s[robust]
        if s_est.size == 0:
            init_scales.append(float('nan'))
            sm_arrays.append(None)
            pm_arrays.append(None)
            continue
        scale = np.median(s_est)

        for _ in range(10):
            slam_depth_0 = slam_depth * scale
            robust = (msk < 0.5) * (0 < slam_depth_0) * (slam_depth_0 < ft) * (nt < pred_depth) * (pred_depth < ft)
            s_est = s[robust]
            if s_est.size == 0:
                break
            scale = np.median(s_est)

        robust = (msk < 0.5) * (0 < slam_depth_0) * (slam_depth_0 < ft) * (nt < pred_depth) * (pred_depth < ft)
        sm_filtered = slam_depth[robust]
        pm_filtered = pred_depth[robust]

        if sm_filtered.size == 0:
            init_scales.append(float('nan'))
            sm_arrays.append(None)
            pm_arrays.append(None)
        else:
            init_scales.append(scale)
            sm_arrays.append(sm_filtered)
            pm_arrays.append(pm_filtered)

    # Separate valid vs NaN keyframes
    valid_idx = [i for i in range(N) if not np.isnan(init_scales[i])]
    if len(valid_idx) == 0:
        return [float('nan')] * N

    # Stage 2: batch transfer to GPU, then N independent 1D BFGS
    # Transfer all data in one pass to avoid per-keyframe CPU->GPU overhead
    sm_gpu = [torch.from_numpy(sm_arrays[i].ravel()).to(device) for i in valid_idx]
    pm_gpu = [torch.from_numpy(pm_arrays[i].ravel()).to(device) for i in valid_idx]

    sigma_sq = sigma ** 2
    scales = [float('nan')] * N
    for j, i in enumerate(valid_idx):
        sm = sm_gpu[j]
        pm = pm_gpu[j]

        def f(x, _sm=sm, _pm=pm):
            residuals = _sm * x - _pm
            r_sq = residuals ** 2
            return (sigma_sq * r_sq / (sigma_sq + r_sq)).mean()

        x0 = torch.tensor([init_scales[i]], device=device)
        result = minimize(f, x0, method='bfgs')
        scales[i] = result.x.detach().cpu().item()

    return scales
