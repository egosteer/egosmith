"""DPVO adapter that returns trajectory and disparity in HaWoR stage3 format."""

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

        scale = min(max_size / max(height, width), 1.0)
        if scale < 1.0:
            new_h = int(height * scale)
            new_w = int(width * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            height, width = image.shape[:2]
            fx_s, fy_s = fx * scale, fy * scale
            cx_s, cy_s = cx * scale, cy * scale
        else:
            fx_s, fy_s, cx_s, cy_s = fx, fy, cx, cy

        image = image[: height - height % 16, : width - width % 16]
        intrinsics = np.array([fx_s, fy_s, cx_s, cy_s], dtype=np.float64)
        yield t, image, intrinsics


def _poses_to_traj(poses):
    """Return poses in HaWoR traj format [tx, ty, tz, qx, qy, qz, qw]."""
    return np.asarray(poses, dtype=np.float32)


def _build_disps_from_patches(slam, height, width):
    """Rasterize sparse DPVO patch disparities into dense-ish per-keyframe maps."""
    n_keyframes = slam.n
    ht, wd = slam.ht, slam.wd
    disps_list = []
    patches = slam.pg.patches_.cpu().numpy()[:n_keyframes]

    for i in range(n_keyframes):
        x = patches[i, :, 0, 1, 1]
        y = patches[i, :, 1, 1, 1]
        disp = patches[i, :, 2, 1, 1]
        valid = disp > 1e-6
        if not np.any(valid):
            disps_list.append(np.ones((ht, wd), dtype=np.float32) * 0.01)
            continue

        x, y, disp = x[valid], y[valid], disp[valid]
        disp_map = np.zeros((ht, wd), dtype=np.float32)
        count_map = np.zeros((ht, wd), dtype=np.int32)
        iy = np.clip(np.round(y).astype(int), 0, ht - 1)
        ix = np.clip(np.round(x).astype(int), 0, wd - 1)
        np.add.at(disp_map, (iy, ix), disp)
        np.add.at(count_map, (iy, ix), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            disp_map = np.where(count_map > 0, disp_map / count_map, 0)

        median_disp = np.median(disp)
        disp_map[disp_map <= 0] = median_disp
        disps_list.append(disp_map.astype(np.float32))

    if (height, width) != (ht, wd):
        disps_list = [
            cv2.resize(disp_map, (width, height), interpolation=cv2.INTER_LINEAR)
            for disp_map in disps_list
        ]

    return np.stack(disps_list, axis=0)


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

    env_to_cfg = {
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
    for env_name, cfg_name in env_to_cfg.items():
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

    weight_path = str(DPVO_ROOT / "models" / "dpvo.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"[DPVO] DPVO checkpoint not found: {weight_path}. "
            "Download it (see README 'Weights' or run scripts/setup/download_weights.sh)."
        )

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
        disps = _build_disps_from_patches(slam, out_h, out_w)

        patch_tstamps = slam.pg.tstamps_
        if hasattr(patch_tstamps, "detach"):
            patch_tstamps = patch_tstamps.detach().cpu().numpy()
        else:
            patch_tstamps = np.asarray(patch_tstamps)
        tstamps_disps = patch_tstamps[: int(slam.n)].reshape(-1)

    del slam
    torch.cuda.empty_cache()

    return traj, disps, tstamps, tstamps_disps
