"""Multiprocessing worker helpers for WebDataset export."""

import os
import tarfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import current_process

import torch

from .webdataset_features import build_mano_models, load_episode_features
from .webdataset_writer import add_prepared_sample_to_tar, iter_episode_samples, prepare_sample_payload

_worker_mano_right = None
_worker_mano_left = None
_worker_device = None
_worker_rescan_frame_index = False
_worker_feature_cache_dir = None
_worker_episode_cache = {}
_worker_mano_dir = None
_worker_require_feature_cache = False

WRITE_PREFETCH_THREADS = 4
WRITE_PREFETCH_DEPTH = 16


def normalize_mano_devices(mano_device, mano_gpus):
    """Normalize MANO worker device list."""
    if mano_gpus:
        devices = []
        for gpu in mano_gpus.split(","):
            gpu = gpu.strip()
            if not gpu:
                continue
            if gpu.startswith("cuda:"):
                devices.append(gpu)
            else:
                devices.append(f"cuda:{gpu}")
        if devices:
            return devices
    return [mano_device]


def _ensure_worker_mano_models():
    global _worker_mano_right, _worker_mano_left
    if _worker_mano_right is not None and _worker_mano_left is not None:
        return
    _worker_mano_right, _worker_mano_left = build_mano_models(_worker_device, mano_dir=_worker_mano_dir)
    _worker_mano_right.eval()
    _worker_mano_left.eval()


def _worker_init(device_specs, mano_dir, rescan_frame_index, feature_cache_dir, require_feature_cache=False, eager_model_init=True):
    global _worker_mano_right, _worker_mano_left, _worker_device
    global _worker_rescan_frame_index, _worker_feature_cache_dir, _worker_episode_cache
    global _worker_mano_dir, _worker_require_feature_cache

    identity = current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    device_str = device_specs[worker_idx % len(device_specs)]
    _worker_device = torch.device(device_str)
    _worker_mano_dir = mano_dir
    _worker_mano_right = None
    _worker_mano_left = None
    _worker_rescan_frame_index = rescan_frame_index
    _worker_feature_cache_dir = feature_cache_dir
    _worker_episode_cache = {}
    _worker_require_feature_cache = require_feature_cache
    if eager_model_init:
        _ensure_worker_mano_models()


def _worker_prepare_episode_features(episode_slice):
    cache_key = episode_slice["crop_dir"]
    if cache_key in _worker_episode_cache:
        episode_data = _worker_episode_cache[cache_key]
    else:
        _ensure_worker_mano_models()
        episode_data = load_episode_features(
            episode_slice,
            _worker_mano_right,
            _worker_mano_left,
            _worker_device,
            rescan_frame_index=_worker_rescan_frame_index,
            feature_cache_dir=_worker_feature_cache_dir,
            require_cache=False,
            mano_dir=_worker_mano_dir,
        )
        _worker_episode_cache[cache_key] = episode_data

    if episode_data is None:
        return {
            "episode_id": episode_slice["episode_id"],
            "cached": False,
            "ok": False,
            "num_frames": 0,
        }

    return {
        "episode_id": episode_slice["episode_id"],
        "cached": True,
        "ok": True,
        "num_frames": len(episode_data["frame_ids"]),
    }


def _worker_prepare_episode_feature_batch(batch):
    episodes_ok = 0
    episodes_failed = 0
    frames_cached = 0

    for episode_slice in batch:
        result = _worker_prepare_episode_features(episode_slice)
        episodes_ok += 1 if result["ok"] else 0
        episodes_failed += 0 if result["ok"] else 1
        frames_cached += int(result["num_frames"])

    return {
        "episodes_total": len(batch),
        "episodes_ok": episodes_ok,
        "episodes_failed": episodes_failed,
        "frames_cached": frames_cached,
    }


def _worker_process_shard(task):
    """Build one shard in a worker process and write directly to disk."""
    started_at = time.perf_counter()
    frames_written = 0
    skipped_episodes = 0
    touched_episodes = set()
    tar_writer = None
    output_path = task["output_path"]
    tmp_path = task["tmp_path"]

    try:
        with ThreadPoolExecutor(max_workers=WRITE_PREFETCH_THREADS) as prefetch_pool:
            for episode_slice in task["episode_slices"]:
                cache_key = episode_slice["crop_dir"]
                if cache_key not in _worker_episode_cache:
                    if not _worker_require_feature_cache:
                        _ensure_worker_mano_models()
                    _worker_episode_cache[cache_key] = load_episode_features(
                        episode_slice,
                        _worker_mano_right,
                        _worker_mano_left,
                        _worker_device,
                        rescan_frame_index=_worker_rescan_frame_index,
                        feature_cache_dir=_worker_feature_cache_dir,
                        require_cache=_worker_require_feature_cache,
                        mano_dir=_worker_mano_dir,
                    )

                episode_data = _worker_episode_cache[cache_key]
                if episode_data is None:
                    skipped_episodes += 1
                    continue

                sample_iter = iter_episode_samples(
                    episode_slice,
                    episode_data,
                    episode_slice["frame_start"],
                    episode_slice["frame_end"],
                )
                pending = deque()

                def submit_next():
                    try:
                        sample = next(sample_iter)
                    except StopIteration:
                        return False
                    pending.append(prefetch_pool.submit(prepare_sample_payload, *sample))
                    return True

                for _ in range(WRITE_PREFETCH_DEPTH):
                    if not submit_next():
                        break

                while pending:
                    key, image_bytes, lowdim_bytes, mano_bytes, meta_bytes = pending.popleft().result()
                    if tar_writer is None:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        tar_writer = tarfile.open(tmp_path, "w")
                    add_prepared_sample_to_tar(tar_writer, key, image_bytes, lowdim_bytes, mano_bytes, meta_bytes)
                    frames_written += 1
                    submit_next()

                touched_episodes.add(cache_key)
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if tar_writer is not None:
        tar_writer.close()

    if frames_written == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    else:
        os.replace(tmp_path, output_path)

    return {
        "shard_idx": task["shard_idx"],
        "frames_written": frames_written,
        "episodes_written": len(touched_episodes),
        "skipped_episodes": skipped_episodes,
        "output_path": output_path,
        "elapsed_sec": time.perf_counter() - started_at,
    }
