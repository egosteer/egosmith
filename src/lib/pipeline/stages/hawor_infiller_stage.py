"""Infiller stage: fills missing hand motion and exports the world-space result (world_space_res.pth)."""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.eval_utils.custom_utils import cam2world_convert
from lib.pipeline.slam.slam_cam import load_slam_cam, validate_dense_slam_export
from lib.eval_utils.filling_utils import filling_postprocess, filling_preprocess
from lib.pipeline.io.frame_source import build_frame_source
from lib.pipeline.tools import parse_chunks_hand_frame
from lib.pipeline.io.workspace import resolve_seq_folder
from lib.pipeline.io import result_io

from .hawor_cache import _load_or_build_cam_space_cache, _slice_cam_space_pred_dict
from .hawor_common import QUIET_MODE, vprint
from .hawor_runtime import build_infiller_runner

INFILLER_DEBUG_WINDOWS = os.environ.get("HAWOR_INFILLER_VERBOSE_WINDOWS", "0") == "1"

# Set HAWOR_INFILLER_NO_SANITIZE=1 to skip finite cleanup (debug / A/B).
_INFILLER_SANITIZE = os.environ.get("HAWOR_INFILLER_NO_SANITIZE", "0").strip() != "1"


@dataclass
class InfillerState:
    pred_trans: torch.Tensor
    pred_rot: torch.Tensor
    pred_hand_pose: torch.Tensor
    pred_betas: torch.Tensor
    pred_valid: torch.Tensor
    num_frames: int
    max_slam_frames: int
    cam_space_cache: dict
    r_c2w_sla_all: torch.Tensor
    t_c2w_sla_all: torch.Tensor
    slam_path: str
    use_dpvo_infiller: bool


def _use_dpvo_infiller_mode(seq_folder: str) -> bool:
    """Consistent with lib/stage_runners/hawor_video.py: for dpvo, interpolate the camera per video frame by tstamp."""
    if os.environ.get("HAWOR_INFILLER_DPVO_MODE", "").strip() == "1":
        return True
    backend_txt = os.path.join(seq_folder, "SLAM", "slam_backend.txt")
    if os.path.isfile(backend_txt):
        with open(backend_txt, "r", encoding="utf-8") as bf:
            if bf.read().strip().lower() == "dpvo":
                return True
    return False


def _interp_and_smooth_hand_time(x: np.ndarray, *, smooth_window: int = 5) -> tuple[np.ndarray, int]:
    """
    Repair NaN/Inf in array shaped (2, T, D) by:
      1) time interpolation using nearest valid-frame anchors (per hand, per dim)
      2) light temporal smoothing, applied only to frames that were non-finite.

    This aims to be more visually continuous than pure forward-fill.
    """
    arr = np.asarray(x, dtype=np.float64).copy()
    bad_before = int((~np.isfinite(arr)).sum())

    H, T, D = arr.shape
    smooth_window = int(smooth_window)
    if smooth_window <= 1:
        smooth_window = 1
    # Ensure odd window for symmetric padding.
    if smooth_window % 2 == 0:
        smooth_window += 1

    # valid_time[h, t] means all D components for this hand/time are finite.
    valid_time = np.all(np.isfinite(arr), axis=-1)  # (H, T)

    for h in range(H):
        vt = valid_time[h]  # (T,)
        if int(vt.sum()) < 2:
            arr[h] = np.nan_to_num(arr[h], nan=0.0, posinf=0.0, neginf=0.0)
            continue

        invalid_time = ~vt
        t_valid = np.where(vt)[0].astype(np.int64)
        t_all = np.arange(T, dtype=np.float64)

        out_h = arr[h]  # (T, D)
        for d in range(D):
            y = out_h[:, d]  # (T,)
            y_valid = y[t_valid]
            y_interp = np.interp(t_all, t_valid.astype(np.float64), y_valid.astype(np.float64)).astype(np.float64)
            # keep anchors exactly
            y_interp[vt] = y[vt]

            if smooth_window > 1 and np.any(invalid_time):
                k = smooth_window
                pad_left = k // 2
                pad_right = (k - 1) - pad_left
                y_pad = np.pad(y_interp, (pad_left, pad_right), mode="edge")
                kernel = np.ones(k, dtype=np.float64) / float(k)
                y_smooth = np.convolve(y_pad, kernel, mode="valid")
                y_interp[invalid_time] = y_smooth[invalid_time]

            out_h[:, d] = y_interp

        arr[h] = out_h

    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32), bad_before


def _sanitize_infiller_tensors(state: InfillerState) -> None:
    """Ensure saved world-space tensors are finite (Stage 4 guardrail)."""
    if not _INFILLER_SANITIZE:
        return
    names = ("pred_trans", "pred_betas", "pred_rot", "pred_hand_pose")
    for name in names:
        t = getattr(state, name, None)
        if t is None:
            continue
        arr_in = t.detach().cpu().numpy()
        if arr_in.ndim < 2 or arr_in.shape[0] != 2:
            continue
        smooth_window = int(os.environ.get("HAWOR_INFILLER_SANITIZE_SMOOTH_WINDOW", "5"))
        fixed, n_bad = _interp_and_smooth_hand_time(arr_in, smooth_window=smooth_window)
        if n_bad > 0 and not QUIET_MODE:
            vprint(f"[infiller] sanitized {n_bad} non-finite values in {name} (time-interp + light smooth)")
        t.copy_(torch.from_numpy(fixed))


def infiller_debug(*args, **kwargs):
    if INFILLER_DEBUG_WINDOWS and not QUIET_MODE:
        print(*args, **kwargs)


def _prepare_infiller_window(
    frame_ck,
    pred_trans,
    pred_rot,
    pred_hand_pose,
    pred_betas,
    pred_valid,
    num_frames,
    filling_length,
):
    start_shift = -1
    while frame_ck[0] + start_shift >= 0 and pred_valid[:, frame_ck[0] + start_shift].sum() != 2:
        start_shift -= 1

    frame_start = int(frame_ck[0])
    filling_net_start = max(0, frame_start + start_shift)
    filling_net_end = min(num_frames - 1, filling_net_start + filling_length)
    if filling_net_end <= filling_net_start:
        return None

    seq_valid = pred_valid[:, filling_net_start:filling_net_end]
    filling_seq = {
        "trans": pred_trans[:, filling_net_start:filling_net_end].numpy(),
        "rot": pred_rot[:, filling_net_start:filling_net_end].numpy(),
        "hand_pose": pred_hand_pose[:, filling_net_start:filling_net_end].numpy(),
        "betas": pred_betas[:, filling_net_start:filling_net_end].numpy(),
        "valid": seq_valid,
    }
    filling_input, transform_w_canon = filling_preprocess(filling_seq)
    filling_input = np.asarray(filling_input, dtype=np.float32)

    t_original = filling_input.shape[0]
    if t_original == 0:
        return None

    if t_original < filling_length:
        pad_length = filling_length - t_original
        padding = np.repeat(filling_input[-1:, :], pad_length, axis=0)
        filling_input = np.concatenate([filling_input, padding], axis=0)
        seq_valid_padding = np.concatenate([seq_valid, np.ones((2, pad_length), dtype=bool)], axis=1)
    else:
        seq_valid_padding = seq_valid

    return {
        "filling_net_start": filling_net_start,
        "filling_net_end": filling_net_end,
        "seq_valid": seq_valid,
        "seq_valid_padding": seq_valid_padding,
        "filling_seq": filling_seq,
        "filling_input": filling_input,
        "transform_w_canon": transform_w_canon,
        "t_original": t_original,
    }


def _flush_infiller_windows(
    pending_windows,
    filling_model,
    src_mask,
    device,
    horizon,
    pred_trans,
    pred_rot,
    pred_hand_pose,
    pred_betas,
    pred_valid,
):
    if not pending_windows:
        return {
            "batch_size": 0,
            "forward_time": 0.0,
            "postprocess_time": 0.0,
        }

    batch_size = len(pending_windows)
    batch_inputs = np.stack([window["filling_input"] for window in pending_windows], axis=1)
    valid_both = np.stack([window["seq_valid_padding"].all(axis=0) for window in pending_windows], axis=1)

    filling_input = torch.from_numpy(batch_inputs).to(device)
    valid_tensor = torch.from_numpy(valid_both).to(device=device)

    data_mask = torch.zeros((horizon, batch_size, 1), device=device, dtype=filling_input.dtype)
    data_mask[valid_tensor] = 1

    valid_atten = valid_tensor.transpose(0, 1).unsqueeze(1)
    atten_mask = torch.ones((batch_size, 1, horizon, horizon), device=device, dtype=torch.bool)
    atten_mask[valid_atten.unsqueeze(2).expand(-1, -1, horizon, -1)] = False

    t_forward = time.time()
    with torch.no_grad():
        batch_output = filling_model(filling_input, src_mask, data_mask, atten_mask)
    forward_time = time.time() - t_forward

    batch_output = batch_output.permute(1, 0, 2).cpu().detach()

    t_postprocess = time.time()
    for window_idx, window in enumerate(pending_windows):
        output_ck = batch_output[window_idx, : window["t_original"]].reshape(window["t_original"], 2, -1)
        filling_output = filling_postprocess(output_ck, window["transform_w_canon"])

        filling_seq = window["filling_seq"]
        seq_valid = window["seq_valid"]
        filling_seq["trans"][~seq_valid] = filling_output["trans"][~seq_valid]
        filling_seq["rot"][~seq_valid] = filling_output["rot"][~seq_valid]
        filling_seq["hand_pose"][~seq_valid] = filling_output["hand_pose"][~seq_valid]
        filling_seq["betas"][~seq_valid] = filling_output["betas"][~seq_valid]

        start = window["filling_net_start"]
        end = window["filling_net_end"]
        pred_trans[:, start:end] = torch.from_numpy(filling_seq["trans"]).float()
        pred_rot[:, start:end] = torch.from_numpy(filling_seq["rot"]).float()
        pred_hand_pose[:, start:end] = torch.from_numpy(filling_seq["hand_pose"]).float()
        pred_betas[:, start:end] = torch.from_numpy(filling_seq["betas"]).float()
        pred_valid[:, start:end] = True

    return {
        "batch_size": batch_size,
        "forward_time": forward_time,
        "postprocess_time": time.time() - t_postprocess,
    }


def _resolve_seq_folder(args, seq_folder):
    if seq_folder is not None:
        return seq_folder
    return str(resolve_seq_folder(video_path=args.video_path))


def _prepare_infiller_state(seq_folder, start_idx, end_idx, frame_chunks_all, frame_source, rebuild_cam_space_cache):
    num_frames = len(frame_source)
    return _prepare_infiller_state_with_cache(
        seq_folder,
        start_idx,
        end_idx,
        frame_chunks_all,
        num_frames=num_frames,
        rebuild_cam_space_cache=rebuild_cam_space_cache,
    )


def _prepare_infiller_state_with_cache(
    seq_folder,
    start_idx,
    end_idx,
    frame_chunks_all,
    *,
    num_frames,
    rebuild_cam_space_cache,
    cam_space_cache=None,
):
    slam_path = os.path.join(seq_folder, "SLAM", f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    use_dpvo_infiller = _use_dpvo_infiller_mode(seq_folder)
    if use_dpvo_infiller and not QUIET_MODE:
        vprint("[infiller] DPVO: dense per-frame SLAM cameras from hawor_slam_w_scale npz use direct frame lookup")
        validate_dense_slam_export(slam_path)

    _r_w2c_sla_all, _t_w2c_sla_all, r_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    pred_trans = torch.zeros(2, num_frames, 3)
    pred_rot = torch.zeros(2, num_frames, 3)
    pred_hand_pose = torch.zeros(2, num_frames, 45)
    pred_betas = torch.zeros(2, num_frames, 10)
    pred_valid = torch.zeros((2, pred_betas.size(1)))
    # Sparse traj (DPVO): the video frame index is NOT the traj row index; projection uses interpolation, here max_slam_frames is only an upper bound on video length.
    if use_dpvo_infiller:
        max_slam_frames = num_frames
    else:
        max_slam_frames = min(pred_trans.shape[1], r_c2w_sla_all.shape[0], t_c2w_sla_all.shape[0])
    if cam_space_cache is None:
        cam_space_cache = _load_or_build_cam_space_cache(
            seq_folder,
            frame_chunks_all,
            rebuild=rebuild_cam_space_cache,
        )
    return InfillerState(
        pred_trans=pred_trans,
        pred_rot=pred_rot,
        pred_hand_pose=pred_hand_pose,
        pred_betas=pred_betas,
        pred_valid=pred_valid,
        num_frames=num_frames,
        max_slam_frames=max_slam_frames,
        cam_space_cache=cam_space_cache,
        r_c2w_sla_all=r_c2w_sla_all,
        t_c2w_sla_all=t_c2w_sla_all,
        slam_path=slam_path,
        use_dpvo_infiller=use_dpvo_infiller,
    )


def _project_cam_space_chunks_to_world(state, frame_chunks_all):
    for idx in [0, 1]:
        frame_chunks = frame_chunks_all[idx]
        if len(frame_chunks) == 0:
            continue

        for frame_ck in frame_chunks:
            frame_ck = np.asarray(frame_ck)
            original_key = f"{int(frame_ck[0])}_{int(frame_ck[-1])}"
            # DPVO: sparse traj, interpolate per video frame by tstamp; upper bound num_frames. Non-DPVO: frame index must fall within the traj row range.
            upper = state.num_frames if state.use_dpvo_infiller else state.max_slam_frames
            valid_frame_mask = frame_ck < upper
            if valid_frame_mask.sum() == 0:
                continue

            pred_dict = state.cam_space_cache[idx][original_key]
            pred_dict = _slice_cam_space_pred_dict(pred_dict, valid_frame_mask)
            frame_ck = frame_ck[valid_frame_mask]
            infiller_debug(f"from frame {frame_ck[0]} to {frame_ck[-1]}")
            data_out = {name: torch.from_numpy(value) for name, value in pred_dict.items()}

            if state.use_dpvo_infiller:
                slam_frame_idx = np.clip(frame_ck, 0, state.r_c2w_sla_all.shape[0] - 1)
                r_c2w_sla = state.r_c2w_sla_all[slam_frame_idx]
                t_c2w_sla = state.t_c2w_sla_all[slam_frame_idx]
            else:
                r_c2w_sla = state.r_c2w_sla_all[frame_ck]
                t_c2w_sla = state.t_c2w_sla_all[frame_ck]
            data_world = cam2world_convert(r_c2w_sla, t_c2w_sla, data_out, "right" if idx > 0 else "left")

            state.pred_trans[[idx], frame_ck] = data_world["init_trans"]
            state.pred_rot[[idx], frame_ck] = data_world["init_root_orient"]
            state.pred_hand_pose[[idx], frame_ck] = data_world["init_hand_pose"].flatten(-2)
            state.pred_betas[[idx], frame_ck] = data_world["init_betas"]
            state.pred_valid[[idx], frame_ck] = 1


def _run_infiller_pass(state, filling_model, src_mask, device, horizon, window_batch_size):
    idx_to_hand = ["left", "right"]
    filling_length = 120
    timing = {
        "prepare_windows": 0.0,
        "model_forward": 0.0,
        "postprocess": 0.0,
    }
    total_windows = 0

    frame_list = torch.tensor(list(range(state.pred_trans.size(1))))
    state.pred_valid = (state.pred_valid > 0).numpy()
    pred_valid_numpy = state.pred_valid
    for idx in [1, 0]:
        observed_count = int(pred_valid_numpy[idx].sum())
        if observed_count == 0:
            if not QUIET_MODE:
                vprint(
                    f"[infiller] skip {idx_to_hand[idx]} hand: "
                    "no observed cam-space frames, keep hand invalid instead of hallucinating"
                )
            continue
        missing = ~pred_valid_numpy[idx]
        frame = frame_list[missing]
        frame_chunks = parse_chunks_hand_frame(frame)
        pending_windows = []

        infiller_debug(f"run infiller on {idx_to_hand[idx]} hand ...")
        for frame_ck in tqdm(frame_chunks, disable=QUIET_MODE):
            t_window = time.time()
            window = _prepare_infiller_window(
                frame_ck,
                state.pred_trans,
                state.pred_rot,
                state.pred_hand_pose,
                state.pred_betas,
                pred_valid_numpy,
                state.num_frames,
                filling_length,
            )
            timing["prepare_windows"] += time.time() - t_window
            if window is None:
                continue

            total_windows += 1
            pending_windows.append(window)
            infiller_debug(
                f"queue infiller window {window['filling_net_start']} to "
                f"{min(state.num_frames - 1, window['filling_net_start'] + filling_length)}"
            )

            if len(pending_windows) >= window_batch_size:
                flush_stats = _flush_infiller_windows(
                    pending_windows,
                    filling_model,
                    src_mask,
                    device,
                    horizon,
                    state.pred_trans,
                    state.pred_rot,
                    state.pred_hand_pose,
                    state.pred_betas,
                    pred_valid_numpy,
                )
                timing["model_forward"] += float(flush_stats["forward_time"])
                timing["postprocess"] += float(flush_stats["postprocess_time"])
                pending_windows = []

        if pending_windows:
            flush_stats = _flush_infiller_windows(
                pending_windows,
                filling_model,
                src_mask,
                device,
                horizon,
                state.pred_trans,
                state.pred_rot,
                state.pred_hand_pose,
                state.pred_betas,
                pred_valid_numpy,
            )
            timing["model_forward"] += float(flush_stats["forward_time"])
            timing["postprocess"] += float(flush_stats["postprocess_time"])

    return total_windows, timing


def _save_infiller_result(seq_folder, state, total_windows, window_batch_size, timing, load_cam_space_time, start_idx=None, end_idx=None):
    t_save = time.time()
    _sanitize_infiller_tensors(state)
    save_path = os.path.join(seq_folder, "world_space_res.pth")
    joblib.dump(
        [state.pred_trans, state.pred_rot, state.pred_hand_pose, state.pred_betas, state.pred_valid],
        save_path,
    )
    # Also write the consolidated single-file result (poses + depth). The legacy
    # .pth stays for now so existing readers keep working; cleanup later removes it
    # in favor of result.npz.
    if start_idx is not None and end_idx is not None:
        try:
            depth = result_io._read_legacy_depth(seq_folder, start_idx, end_idx)
            depth_kwargs = {}
            if depth is not None:
                indices, depths, h, w = depth
                depth_kwargs = {
                    "depth_frame_indices": indices,
                    "depths_uint16": depths,
                    "depth_height": h,
                    "depth_width": w,
                }
            result_io.save_result(
                seq_folder,
                pred_trans=state.pred_trans,
                pred_rot=state.pred_rot,
                pred_hand_pose=state.pred_hand_pose,
                pred_betas=state.pred_betas,
                pred_valid=state.pred_valid,
                **depth_kwargs,
            )
        except Exception as error:
            # Consolidation is best-effort; the legacy .pth + depth npz remain the
            # source of truth if it fails. Make the failure visible.
            result_io._logger.warning("Failed to write consolidated result.npz for %s: %s", seq_folder, error)
    save_time = time.time() - t_save
    print(
        f"[infiller] {os.path.basename(seq_folder)} windows={total_windows} "
        f"batch_size={window_batch_size} "
        f"load_cam_space={load_cam_space_time:.2f}s "
        f"prepare={timing['prepare_windows']:.2f}s "
        f"forward={timing['model_forward']:.2f}s "
        f"postprocess={timing['postprocess']:.2f}s "
        f"save={save_time:.2f}s"
    )
    return save_time


def run_infiller_for_video(
    args,
    start_idx,
    end_idx,
    frame_chunks_all,
    infiller_runner=None,
    frame_source=None,
    seq_folder=None,
    num_frames=None,
    cam_space_cache=None,
    return_timing=False,
):
    infiller_runner = infiller_runner or build_infiller_runner(args.infiller_weight)
    filling_model = infiller_runner["model"]
    device = infiller_runner["device"]
    horizon = infiller_runner["horizon"]
    src_mask = infiller_runner["src_mask"]
    window_batch_size = max(1, int(getattr(args, "infiller_window_batch_size", 64)))
    rebuild_cam_space_cache = bool(getattr(args, "rebuild_cam_space_cache", False))

    seq_folder = _resolve_seq_folder(args, seq_folder)
    if num_frames is None:
        if frame_source is None:
            frame_source = build_frame_source(args.video_path)
        num_frames = len(frame_source)

    cache_path = os.path.join(seq_folder, "cam_space_cache.joblib")
    cache_hit = bool(cam_space_cache is not None)
    if not cache_hit and not rebuild_cam_space_cache and os.path.exists(cache_path):
        cache_hit = True
    t_load = time.time()
    state = _prepare_infiller_state_with_cache(
        seq_folder,
        start_idx,
        end_idx,
        frame_chunks_all,
        num_frames=num_frames,
        rebuild_cam_space_cache=rebuild_cam_space_cache,
        cam_space_cache=cam_space_cache,
    )
    load_cam_space_time = time.time() - t_load

    t_project = time.time()
    _project_cam_space_chunks_to_world(state, frame_chunks_all)
    project_world_time = time.time() - t_project
    total_windows, timing = _run_infiller_pass(
        state,
        filling_model,
        src_mask,
        device,
        horizon,
        window_batch_size=window_batch_size,
    )
    save_time = _save_infiller_result(
        seq_folder,
        state,
        total_windows,
        window_batch_size,
        timing,
        load_cam_space_time,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    if return_timing:
        return {
            "timing": {
                "load_cam_space": float(load_cam_space_time),
                "project_world": float(project_world_time),
                "prepare_windows": float(timing["prepare_windows"]),
                "model_forward": float(timing["model_forward"]),
                "postprocess": float(timing["postprocess"]),
                "save": float(save_time),
                "total": float(
                    load_cam_space_time
                    + project_world_time
                    + timing["prepare_windows"]
                    + timing["model_forward"]
                    + timing["postprocess"]
                    + save_time
                ),
            },
            "stats": {
                "frame_count": int(state.num_frames),
                "chunk_count": int(sum(len(frame_chunks_all.get(idx, [])) for idx in [0, 1])),
                "window_count": int(total_windows),
                "cam_space_cache_hit": bool(cache_hit),
                "dpvo_infiller_mode": bool(state.use_dpvo_infiller),
            },
        }
    return state.pred_trans, state.pred_rot, state.pred_hand_pose, state.pred_betas, state.pred_valid


def hawor_infiller(args, start_idx, end_idx, frame_chunks_all):
    return run_infiller_for_video(args, start_idx, end_idx, frame_chunks_all, infiller_runner=None)
