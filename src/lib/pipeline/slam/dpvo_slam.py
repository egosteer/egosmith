"""DPVO adapter that returns trajectory and disparity in HaWoR stage3 format."""

import hashlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


# src/lib/pipeline/slam/dpvo_slam.py -> parents[4] is the repo root (parents[3] is src/).
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DPVO_ROOT = PROJECT_ROOT / "thirdparty" / "DPVO"
if str(DPVO_ROOT) not in sys.path:
    sys.path.insert(0, str(DPVO_ROOT))


# Bump when the layout/semantics of the disparity maps returned by run_dpvo_slam
# change; cached dpvo_raw_*.npz files stamped with an older version are rerun.
#   1 (implicit, unstamped): patch grid coords splatted onto a full-res canvas
#      (everything in the top-left 1/16) + constant median fill.
#   2: correct 1/RES grid placement through downscale+crop, sparse (0 = invalid).
#   3: DPVO input geometry derived from the original frame size instead of the
#      (smaller) output raster size, so patches map to the right output pixels.
DPVO_DISP_RASTER_VERSION = 3

# Seed of the first DPVO attempt; attempt k uses DPVO_SEED + k (HAWOR_DPVO_ATTEMPTS, default 3).
DPVO_SEED = 0
DPVO_DEFAULT_ATTEMPTS = 3


class DPVODivergedError(RuntimeError):
    """DPVO returned non-finite poses in every attempt."""


# Environment overrides applied to the DPVO config in run_dpvo_slam.
DPVO_ENV_TO_CFG = {
    "HAWOR_DPVO_BUFFER_SIZE": "BUFFER_SIZE",
    "HAWOR_DPVO_PATCHES_PER_FRAME": "PATCHES_PER_FRAME",
    "HAWOR_DPVO_REMOVAL_WINDOW": "REMOVAL_WINDOW",
    "HAWOR_DPVO_OPTIMIZATION_WINDOW": "OPTIMIZATION_WINDOW",
    "HAWOR_DPVO_PATCH_LIFETIME": "PATCH_LIFETIME",
    "HAWOR_DPVO_KEYFRAME_INDEX": "KEYFRAME_INDEX",
    "HAWOR_DPVO_KEYFRAME_THRESH": "KEYFRAME_THRESH",
    "HAWOR_DPVO_MIXED_PRECISION": "MIXED_PRECISION",
    # Proximity (mid-term) loop closure = DPV-SLAM. Goes through the main BA; needs NO
    # retrieval/DBoW2/ORBvoc, no rebuild, and leaves terminate()'s output contract intact
    # (globally-corrected poses + patches/disps). Gated: default off (cfg defaults False).
    # Classical (CLASSIC_LOOP_CLOSURE) is intentionally NOT exposed here — it needs heavy
    # extra deps + an async PGO process; revisit only if proximity is insufficient.
    "HAWOR_DPVO_LOOP_CLOSURE": "LOOP_CLOSURE",
    "HAWOR_DPVO_MAX_EDGE_AGE": "MAX_EDGE_AGE",
    "HAWOR_DPVO_GLOBAL_OPT_FREQ": "GLOBAL_OPT_FREQ",
    "HAWOR_DPVO_BACKEND_THRESH": "BACKEND_THRESH",
}


def _dpvo_weight_path() -> str:
    return str(DPVO_ROOT / "models" / "dpvo.pth")


def dpvo_config_identity() -> dict:
    """What, besides the frames/calibration, decides run_dpvo_slam's output: the DPVO config file,
    the environment overrides that are set, and the checkpoint (path inside DPVO_ROOT + size, so
    the same weights in another checkout match). Used to key the dpvo_raw cache."""
    config_path = DPVO_ROOT / "config" / "default.yaml"
    try:
        config_bytes = config_path.read_bytes()
    except OSError:
        config_bytes = None
    weight_path = _dpvo_weight_path()
    try:
        weight_size = int(os.path.getsize(weight_path))
    except OSError:
        weight_size = None
    return {
        "config_sha1": hashlib.sha1(config_bytes).hexdigest() if config_bytes is not None else None,
        "env": {name: os.environ[name] for name in sorted(DPVO_ENV_TO_CFG) if os.environ.get(name)},
        "weights": [os.path.relpath(weight_path, str(DPVO_ROOT)), weight_size],
        "raster_version": DPVO_DISP_RASTER_VERSION,
    }


def dpvo_cache_is_stale(cache_path) -> bool:
    """True if a dpvo_raw_*.npz was written by an older disparity rasterizer (or is unreadable)."""
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            if "disp_raster_version" not in cached.files:
                return True
            return int(np.asarray(cached["disp_raster_version"]).reshape(-1)[0]) != DPVO_DISP_RASTER_VERSION
    except Exception:
        return True


def _processed_frame_count(frame_source, stride=1, frame_indices=None) -> int:
    stride = max(1, int(stride))
    if frame_indices is None:
        total = len(frame_source)
    else:
        total = len(frame_indices)
    return max(0, (int(total) + stride - 1) // stride)


def _resolve_dpvo_buffer_size(current_buffer_size: int, *, frame_source, stride=1, frame_indices=None) -> int:
    env_raw = os.environ.get("HAWOR_DPVO_BUFFER_SIZE")
    if env_raw is not None and str(env_raw).strip():
        return max(32, int(env_raw))

    processed_frames = _processed_frame_count(
        frame_source,
        stride=stride,
        frame_indices=frame_indices,
    )
    # DPVO checks `(self.n + 1) >= self.N` before inserting a new frame, so
    # the buffer must be strictly larger than the number of frames we plan to feed.
    required = max(int(current_buffer_size), processed_frames + 32)
    if required <= 1024:
        align = 128
    elif required <= 4096:
        align = 256
    else:
        align = 512
    return int(((required + align - 1) // align) * align)


def _dpvo_input_geometry(height, width, max_size=800):
    """Geometry of the frame DPVO actually sees for an original (height, width) frame.

    Returns (scale, resized_h, resized_w, crop_h, crop_w): the frame is first
    downscaled so its long side is <= max_size, then cropped (top-left anchored)
    to a multiple of 16. The disparity rasterizer must undo exactly this.
    """
    scale = min(max_size / max(height, width), 1.0)
    if scale < 1.0:
        resized_h = int(height * scale)
        resized_w = int(width * scale)
    else:
        resized_h, resized_w = height, width
    crop_h = resized_h - resized_h % 16
    crop_w = resized_w - resized_w % 16
    return scale, resized_h, resized_w, crop_h, crop_w


def _frame_stream(frame_source, calib, stride=1, max_size=800, frame_indices=None):
    """Yield DPVO-ready frames and intrinsics from a generic frame_source."""
    fx, fy, cx, cy = np.array(calib[:4], dtype=np.float64)
    if frame_indices is None:
        frame_pairs = [(idx, idx) for idx in range(0, len(frame_source), stride)]
    else:
        frame_pairs = [
            (local_idx, int(frame_indices[local_idx]))
            for local_idx in range(0, len(frame_indices), stride)
        ]

    for local_idx, t in frame_pairs:
        image = frame_source.get_frame(local_idx, rgb=False)
        if image is None:
            break
        height, width = image.shape[:2]

        scale, new_h, new_w, crop_h, crop_w = _dpvo_input_geometry(height, width, max_size)
        if scale < 1.0:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            fx_s, fy_s = fx * scale, fy * scale
            cx_s, cy_s = cx * scale, cy * scale
        else:
            fx_s, fy_s, cx_s, cy_s = fx, fy, cx, cy

        image = image[:crop_h, :crop_w]
        intrinsics = np.array([fx_s, fy_s, cx_s, cy_s], dtype=np.float64)
        yield t, image, intrinsics


def _poses_to_traj(poses):
    """Return poses in HaWoR traj format [tx, ty, tz, qx, qy, qz, qw]."""
    return np.asarray(poses, dtype=np.float32)


def rasterize_patch_disps(patches, ht, wd, res, out_hw, resized_hw):
    """Rasterize sparse DPVO patch disparities into per-keyframe maps at the original frame size.

    patches: (n, M, 3, P, P) array straight from ``slam.pg.patches_``. Rows 0/1 are
    x/y on DPVO's 1/``res`` feature grid (RES=4), row 2 is the patch inverse depth
    (one value shared by the whole PxP block, so the block is splatted as a whole).
    ht/wd: the cropped DPVO input size; resized_hw: the downscaled-but-uncropped
    size; out_hw: the original frame size the maps are returned at.

    Cells no patch touches are 0 (= invalid); consumers must treat disp <= 0 as
    "no SLAM depth here" rather than as a measurement.
    """
    patches = np.asarray(patches, dtype=np.float32)
    n = patches.shape[0]
    lo_h, lo_w = ht // res, wd // res
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    resized_h, resized_w = int(resized_hw[0]), int(resized_hw[1])
    out = np.zeros((n, out_h, out_w), dtype=np.float32)

    for i in range(n):
        x = patches[i, :, 0].reshape(-1)
        y = patches[i, :, 1].reshape(-1)
        disp = patches[i, :, 2].reshape(-1)
        valid = np.isfinite(disp) & (disp > 1e-6) & np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        ix = np.round(x[valid]).astype(int)
        iy = np.round(y[valid]).astype(int)
        disp = disp[valid]
        inside = (ix >= 0) & (ix < lo_w) & (iy >= 0) & (iy < lo_h)
        if not np.any(inside):
            continue
        ix, iy, disp = ix[inside], iy[inside], disp[inside]

        disp_sum = np.zeros((lo_h, lo_w), dtype=np.float64)
        count = np.zeros((lo_h, lo_w), dtype=np.int32)
        np.add.at(disp_sum, (iy, ix), disp)
        np.add.at(count, (iy, ix), 1)
        lo = np.where(count > 0, disp_sum / np.maximum(count, 1), 0.0).astype(np.float32)

        # 1/res grid -> DPVO input pixels (block fill), placed on the uncropped resized canvas.
        canvas = np.zeros((resized_h, resized_w), dtype=np.float32)
        full = np.repeat(np.repeat(lo, res, axis=0), res, axis=1)
        fh, fw = min(full.shape[0], resized_h), min(full.shape[1], resized_w)
        canvas[:fh, :fw] = full[:fh, :fw]

        if (resized_h, resized_w) != (out_h, out_w):
            canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        out[i] = canvas

    return out


def _build_disps_from_patches(slam, src_hw, out_hw):
    """Rasterize the patch graph of a finished DPVO run to (n_keyframes, *out_hw).

    src_hw is the original frame size DPVO's input was derived from (see
    _frame_stream); out_hw is the raster size the maps are returned at.
    """
    n_keyframes = int(slam.n)
    patches = slam.pg.patches_[:n_keyframes].detach().cpu().numpy()
    _, resized_h, resized_w, _, _ = _dpvo_input_geometry(int(src_hw[0]), int(src_hw[1]), max_size=800)
    return rasterize_patch_disps(
        patches,
        ht=int(slam.ht),
        wd=int(slam.wd),
        res=int(slam.RES),
        out_hw=(int(out_hw[0]), int(out_hw[1])),
        resized_hw=(resized_h, resized_w),
    )


def run_dpvo_slam(imagedir, masks, calib=None, stride=1, frame_indices=None):
    """Run DPVO and return trajectory/disparity arrays for stage3 scale estimation."""
    del masks  # DPVO itself does not consume the hand masks.

    from dpvo.config import cfg
    from dpvo.dpvo import DPVO

    from lib.pipeline.io.frame_source import build_frame_source
    from lib.pipeline.slam.slam_geom_utils import est_calib, get_dimention

    frame_source = imagedir
    if not (hasattr(imagedir, "get_frame") and hasattr(imagedir, "__len__")):
        frame_source = build_frame_source(imagedir)

    if calib is None:
        calib = np.array(est_calib(frame_source))
    calib = np.array(calib[:4], dtype=np.float64)

    config_path = DPVO_ROOT / "config" / "default.yaml"
    if config_path.exists():
        cfg.merge_from_file(str(config_path))

    for env_name, cfg_name in DPVO_ENV_TO_CFG.items():
        if env_name not in os.environ:
            continue
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        current = getattr(cfg, cfg_name)
        if isinstance(current, bool):
            setattr(cfg, cfg_name, raw.strip().lower() in ("1", "true", "yes", "y", "on"))
        elif isinstance(current, int):
            setattr(cfg, cfg_name, int(raw))
        else:
            setattr(cfg, cfg_name, float(raw))

    print(f"[dpvo] LOOP_CLOSURE={bool(cfg.LOOP_CLOSURE)} "
          f"(MAX_EDGE_AGE={int(cfg.MAX_EDGE_AGE)}, GLOBAL_OPT_FREQ={int(cfg.GLOBAL_OPT_FREQ)}, "
          f"BACKEND_THRESH={float(cfg.BACKEND_THRESH)})", flush=True)

    cfg.BUFFER_SIZE = _resolve_dpvo_buffer_size(
        int(cfg.BUFFER_SIZE),
        frame_source=frame_source,
        stride=stride,
        frame_indices=frame_indices,
    )

    weight_path = _dpvo_weight_path()
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"[DPVO] DPVO checkpoint not found: {weight_path}. "
            "Download it (see README 'Weights' or run scripts/setup/download_weights.sh)."
        )

    def _run_once():
        slam = None
        with torch.inference_mode():
            for t, image, intrinsics in _frame_stream(frame_source, calib, stride, max_size=800, frame_indices=frame_indices):
                image_t = torch.from_numpy(image).permute(2, 0, 1).float().cuda()
                intrinsics_t = torch.from_numpy(intrinsics).float().cuda()

                if slam is None:
                    _, height, width = image_t.shape
                    slam = DPVO(cfg, weight_path, ht=height, wd=width, viz=False)

                slam(t, image_t, intrinsics_t)

            poses, tstamps = slam.terminate()
            traj = _poses_to_traj(poses)
            out_h, out_w = get_dimention(frame_source)
            src_h, src_w = frame_source.get_frame(0, rgb=False).shape[:2]
            disps = _build_disps_from_patches(slam, (src_h, src_w), (out_h, out_w))

            patch_tstamps = slam.pg.tstamps_
            if hasattr(patch_tstamps, "detach"):
                patch_tstamps = patch_tstamps.detach().cpu().numpy()
            else:
                patch_tstamps = np.asarray(patch_tstamps)
            tstamps_disps = patch_tstamps[: int(slam.n)].reshape(-1)

        del slam
        torch.cuda.empty_cache()

        return traj, disps, tstamps, tstamps_disps

    return _run_dpvo_attempts(_run_once)


def _run_dpvo_attempts(run_once):
    """Run ``run_once`` (one full DPVO pass) under fixed seeds until the poses are finite.

    DPVO picks its patch centroids at random (CENTROID_SEL_STRAT=RANDOM) and, on some clips,
    one draw in a few makes the pose graph diverge to non-finite poses. Each attempt uses a
    fixed seed (reproducible output) and a diverged run is retried with the next seed.
    """
    attempts = max(1, int(os.environ.get("HAWOR_DPVO_ATTEMPTS", DPVO_DEFAULT_ATTEMPTS)))
    for attempt in range(attempts):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            torch.manual_seed(DPVO_SEED + attempt)
            traj, disps, tstamps, tstamps_disps = run_once()
        if np.isfinite(traj).all():
            return traj, disps, tstamps, tstamps_disps
        bad = int((~np.isfinite(traj).all(axis=1)).sum())
        print(f"[dpvo] attempt {attempt + 1}/{attempts} diverged ({bad}/{len(traj)} non-finite poses)", flush=True)
    raise DPVODivergedError(
        f"DPVO diverged: non-finite poses in all {attempts} attempt(s) (seeds {DPVO_SEED}..{DPVO_SEED + attempts - 1})"
    )
