"""On-disk caching helpers for per-clip track and stage outputs."""

import json
import os
import sys

import joblib
import numpy as np

from .hawor_common import vprint


def _get_tracks_dir(seq_folder, start_idx, end_idx):
    return os.path.join(seq_folder, f"tracks_{start_idx}_{end_idx}")


def _get_motion_output_paths(seq_folder, start_idx, end_idx):
    tracks_dir = _get_tracks_dir(seq_folder, start_idx, end_idx)
    return (
        tracks_dir,
        os.path.join(tracks_dir, "frame_chunks_all.npy"),
        os.path.join(tracks_dir, "model_masks.npy"),
    )


def _save_cam_space_json(data_out_cpu, seq_folder, idx, frame_ck_first, frame_ck_last):
    pred_dict = {key: value.tolist() for key, value in data_out_cpu.items()}
    pred_path = os.path.join(seq_folder, "cam_space", str(idx), f"{frame_ck_first}_{frame_ck_last}.json")
    cam_dir = os.path.join(seq_folder, "cam_space", str(idx))
    if not os.path.exists(cam_dir):
        os.makedirs(cam_dir)
    with open(pred_path, "w") as handle:
        json.dump(pred_dict, handle, indent=1)


def _save_motion_outputs(model_masks, frame_chunks_all, model_masks_file, frame_chunks_file, output_dir):
    def _save_masks():
        np.save(model_masks_file, model_masks)
        if not os.path.exists(model_masks_file):
            raise IOError(f"File not found after save: {model_masks_file}")
        file_size = os.path.getsize(model_masks_file)
        if file_size == 0:
            raise IOError(f"File is empty after save: {model_masks_file}")
        vprint(f"Saved model_masks.npy ({model_masks.shape}, {model_masks.dtype}, {file_size} bytes)")

    def _save_chunks():
        joblib.dump(frame_chunks_all, frame_chunks_file)
        if not os.path.exists(frame_chunks_file):
            raise IOError(f"File not found after save: {frame_chunks_file}")
        file_size = os.path.getsize(frame_chunks_file)
        if file_size == 0:
            raise IOError(f"File is empty after save: {frame_chunks_file}")
        vprint(f"Saved frame_chunks_all.npy ({file_size} bytes)")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as save_pool:
        mask_future = save_pool.submit(_save_masks)
        chunks_future = save_pool.submit(_save_chunks)
        try:
            mask_future.result()
        except Exception as error:
            print(f"ERROR: Failed to save model_masks.npy: {error}", file=sys.stderr)
            print(f"  Path: {model_masks_file}", file=sys.stderr)
            print(f"  Directory exists: {os.path.exists(output_dir)}", file=sys.stderr)
            raise
        try:
            chunks_future.result()
        except Exception as error:
            print(f"ERROR: Failed to save frame_chunks_all.npy: {error}", file=sys.stderr)
            print(f"  Path: {frame_chunks_file}", file=sys.stderr)
            raise


def _load_or_build_cam_space_cache(seq_folder, frame_chunks_all, rebuild=False):
    cache_path = os.path.join(seq_folder, "cam_space_cache.joblib")
    if os.path.exists(cache_path) and not rebuild:
        try:
            return joblib.load(cache_path)
        except Exception:
            vprint(f"cam_space cache is invalid, rebuilding: {cache_path}")

    cache = {0: {}, 1: {}}
    for idx in [0, 1]:
        for frame_ck in frame_chunks_all.get(idx, []):
            frame_ck = np.asarray(frame_ck)
            if frame_ck.size == 0:
                continue
            key = f"{int(frame_ck[0])}_{int(frame_ck[-1])}"
            pred_path = os.path.join(seq_folder, "cam_space", str(idx), f"{key}.json")
            with open(pred_path, "r") as handle:
                pred_dict = json.load(handle)
            cache[idx][key] = {name: np.asarray(value, dtype=np.float32) for name, value in pred_dict.items()}

    joblib.dump(cache, cache_path)
    return cache


def _invalidate_cam_space_cache(seq_folder):
    cache_path = os.path.join(seq_folder, "cam_space_cache.joblib")
    if os.path.exists(cache_path):
        os.remove(cache_path)


def _slice_cam_space_pred_dict(pred_dict, valid_frame_mask):
    if valid_frame_mask is None or bool(np.all(valid_frame_mask)):
        return pred_dict

    sliced = {}
    for name, value in pred_dict.items():
        value = np.asarray(value)
        if value.ndim >= 2 and value.shape[1] == len(valid_frame_mask):
            sliced[name] = value[:, valid_frame_mask]
        else:
            sliced[name] = value
    return sliced
