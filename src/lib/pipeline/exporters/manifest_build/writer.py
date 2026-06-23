"""Shard planning, encoding, and worker execution for manifest-based build/export."""

from __future__ import annotations

import io
import json
import os
import tarfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import current_process

import torch

from lib.pipeline.slam.depth_artifacts import DEPTH_EXPORT_ENCODING, DEPTH_EXPORT_SCHEMA, encode_depth_npy
from lib.pipeline.exporters.mano_codec import mano_meta_fields
from lib.pipeline.exporters.shard_io import encode_array_npy, encode_lowdim_npy
from lib.pipeline.exporters.webdataset_workers import normalize_mano_devices
from lib.pipeline.io.frame_sources import (
    build_frame_bytes_reader,
    classify_descriptor_storage,
    is_light_tar_descriptor,
)
from lib.pipeline.exporters.webdataset_features import build_mano_models

from .episodes import load_descriptor_episode_features

WRITE_PREFETCH_THREADS = 4
WRITE_PREFETCH_DEPTH = 16

_worker_mano_right = None
_worker_mano_left = None
_worker_device = None
_worker_mano_dir = None
_worker_feature_cache_dir = None
_worker_episode_cache = {}
_worker_shard_fd_cache = {}
_worker_shard_tar_cache = {}


def plan_manifest_shards(episodes: list[dict], frames_per_shard: int, output_dir: str):
    tasks = []
    shard_slices = []
    shard_frame_count = 0
    shard_idx = 0

    def flush_current():
        nonlocal shard_slices, shard_frame_count, shard_idx
        if not shard_slices:
            return
        output_path = os.path.join(output_dir, f"shard-{shard_idx:06d}.tar")
        tasks.append(
            {
                "shard_idx": shard_idx,
                "output_path": output_path,
                "tmp_path": f"{output_path}.tmp",
                "frame_count": shard_frame_count,
                "episode_slices": shard_slices,
            }
        )
        shard_idx += 1
        shard_slices = []
        shard_frame_count = 0

    for ep in episodes:
        num_frames = ep["num_valid_frames"]
        if shard_slices and shard_frame_count + num_frames > frames_per_shard:
            flush_current()
        shard_slices.append(
            {
                "seq_folder": ep["seq_folder"],
                "episode_id": ep["episode_id"],
                "episode_index": ep["episode_index"],
                "clip_id": ep["clip_id"],
                "source_id": ep["source_id"],
                "split": ep["split"],
                "instruction": list(ep.get("instruction", [])),
                "instruction_num": int(ep.get("instruction_num", 0)),
                "language": ep.get("language"),
                "descriptor": ep["descriptor"],
                "frame_start": 0,
                "frame_end": num_frames,
                "source_fps": float(ep.get("source_fps", 5.0)),
                "target_fps": float(ep.get("target_fps", 30.0)),
                "interpolate_labels": bool(ep.get("interpolate_labels", False)),
                "export_depth": bool(ep.get("export_depth", False)),
            }
        )
        shard_frame_count += num_frames
        if shard_frame_count >= frames_per_shard:
            flush_current()

    flush_current()
    return tasks


def repeat_manifest_episodes(episodes: list[dict], repeat_count: int) -> list[dict]:
    repeated = []
    for repeat_idx in range(repeat_count):
        for ep in episodes:
            ep_copy = dict(ep)
            ep_copy["source_episode_index"] = ep["episode_index"]
            if repeat_count > 1:
                ep_copy["repeat_index"] = repeat_idx
            ep_copy["episode_index"] = len(repeated)
            repeated.append(ep_copy)
    return repeated


def prepare_sample_payload_from_bytes(key: str, image_bytes: bytes, lowdim, mano, meta_bytes: bytes, depth=None):
    return (
        key,
        image_bytes,
        encode_lowdim_npy(lowdim),
        encode_array_npy(mano),
        None if depth is None else encode_depth_npy(depth),
        meta_bytes,
    )


def add_prepared_sample_bytes_to_tar(
    tar_writer,
    key: str,
    image_bytes: bytes,
    lowdim_bytes: bytes,
    mano_bytes: bytes,
    depth_bytes: bytes | None,
    meta_bytes: bytes,
) -> None:
    img_info = tarfile.TarInfo(name=f"{key}.image.jpg")
    img_info.size = len(image_bytes)
    tar_writer.addfile(img_info, io.BytesIO(image_bytes))

    lowdim_info = tarfile.TarInfo(name=f"{key}.lowdim.npy")
    lowdim_info.size = len(lowdim_bytes)
    tar_writer.addfile(lowdim_info, io.BytesIO(lowdim_bytes))

    mano_info = tarfile.TarInfo(name=f"{key}.mano.npy")
    mano_info.size = len(mano_bytes)
    tar_writer.addfile(mano_info, io.BytesIO(mano_bytes))

    if depth_bytes is not None:
        depth_info = tarfile.TarInfo(name=f"{key}.depth.npy")
        depth_info.size = len(depth_bytes)
        tar_writer.addfile(depth_info, io.BytesIO(depth_bytes))

    meta_info = tarfile.TarInfo(name=f"{key}.meta.json")
    meta_info.size = len(meta_bytes)
    tar_writer.addfile(meta_info, io.BytesIO(meta_bytes))


def build_manifest_meta_prefix(episode_slice: dict) -> bytes:
    descriptor = episode_slice.get("descriptor")
    descriptor_extra = getattr(descriptor, "extra", None) or {}
    mano_fields = mano_meta_fields()
    if descriptor_extra.get("mano_schema"):
        mano_fields["mano_schema"] = descriptor_extra["mano_schema"]
    meta = {
        "dataset_name": episode_slice["source_id"],
        "clip_id": episode_slice["clip_id"],
        "episode_index": episode_slice["episode_index"],
        "split": episode_slice["split"],
        "instruction": list(episode_slice.get("instruction", [])),
        "instruction_num": int(episode_slice.get("instruction_num", 0)),
        "language": episode_slice.get("language"),
        "lowdim_schema": descriptor_extra.get("lowdim_schema") or "hawor_wrist_world_v2",
        "native_feature_source": descriptor_extra.get("native_feature_source"),
        "wrist_translation_semantics": "mano_joint_0_world",
        "camera_extrinsic_convention": "w2c",
        **mano_fields,
    }
    if episode_slice.get("export_depth"):
        meta["depth_schema"] = DEPTH_EXPORT_SCHEMA
        meta["depth_encoding"] = DEPTH_EXPORT_ENCODING
    return (json.dumps(meta, ensure_ascii=False, separators=(",", ":"))[:-1] + ',"presence":').encode("utf-8")


def worker_init(device_specs, mano_dir, feature_cache_dir, skip_mano_models=False):
    global _worker_mano_right, _worker_mano_left, _worker_device, _worker_mano_dir
    global _worker_feature_cache_dir, _worker_episode_cache
    global _worker_shard_fd_cache, _worker_shard_tar_cache

    identity = current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    device_str = device_specs[worker_idx % len(device_specs)]
    _worker_device = torch.device(device_str)
    _worker_mano_dir = mano_dir
    if skip_mano_models:
        _worker_mano_right = None
        _worker_mano_left = None
    else:
        _worker_mano_right, _worker_mano_left = build_mano_models(_worker_device, mano_dir=mano_dir)
        _worker_mano_right.eval()
        _worker_mano_left.eval()
    _worker_feature_cache_dir = feature_cache_dir
    _worker_episode_cache = {}
    _worker_shard_fd_cache = {}
    _worker_shard_tar_cache = {}


def classify_light_descriptor_error(error: Exception) -> str:
    if isinstance(error, KeyError):
        return "missing_member"
    if isinstance(error, IndexError):
        return "invalid_light_descriptor"
    if isinstance(error, ValueError):
        return "invalid_light_descriptor"

    message = str(error).lower()
    if "failed to extract" in message or "no item named" in message:
        return "missing_member"
    if "short read" in message:
        return "read_error"
    return "read_error"


def worker_process_shard(task):
    frames_written = 0
    skipped_episodes = 0
    skipped_clips = []
    touched_episodes = set()
    tar_writer = None

    try:
        with ThreadPoolExecutor(max_workers=WRITE_PREFETCH_THREADS) as prefetch_pool:
            for episode_slice in task["episode_slices"]:
                cache_key = episode_slice["seq_folder"]
                if cache_key not in _worker_episode_cache:
                    _worker_episode_cache[cache_key] = load_descriptor_episode_features(
                        episode_slice,
                        _worker_mano_right,
                        _worker_mano_left,
                        _worker_device,
                        _worker_feature_cache_dir,
                        _worker_mano_dir,
                        source_fps=float(episode_slice.get("source_fps", 5.0)),
                        target_fps=float(episode_slice.get("target_fps", 30.0)),
                        interpolate_labels=bool(episode_slice.get("interpolate_labels", False)),
                        export_depth=bool(episode_slice.get("export_depth", False)),
                    )

                episode_data = _worker_episode_cache[cache_key]
                if episode_data is None:
                    skipped_episodes += 1
                    continue

                descriptor = episode_slice["descriptor"]
                read_frame_bytes = build_frame_bytes_reader(
                    descriptor,
                    shard_fd_cache=_worker_shard_fd_cache,
                    shard_tar_cache=_worker_shard_tar_cache,
                )
                clip_samples = []
                pending = deque()
                meta_prefix = build_manifest_meta_prefix(episode_slice)

                try:
                    frame_end = min(int(episode_slice["frame_end"]), int(episode_data["frame_count"]))
                    for frame_idx in range(int(episode_slice["frame_start"]), frame_end):
                        image_bytes = read_frame_bytes(frame_idx)
                        presence = int(episode_data["presence_per_frame"][frame_idx])
                        key = f"{episode_slice['clip_id']}_f{frame_idx:06d}"
                        meta_bytes = meta_prefix + str(presence).encode("ascii") + b"}"
                        pending.append(
                            prefetch_pool.submit(
                                prepare_sample_payload_from_bytes,
                                key,
                                image_bytes,
                                episode_data["lowdim_all"][frame_idx],
                                episode_data["mano_all"][frame_idx],
                                meta_bytes,
                                None if not episode_slice.get("export_depth", False) else episode_data["depth_all"][frame_idx],
                            )
                        )
                        if len(pending) >= WRITE_PREFETCH_DEPTH:
                            clip_samples.append(pending.popleft().result())

                    while pending:
                        clip_samples.append(pending.popleft().result())
                except Exception as error:
                    for future in pending:
                        future.cancel()
                    if is_light_tar_descriptor(descriptor):
                        skipped_clips.append(
                            {
                                "clip_id": episode_slice["clip_id"],
                                "shard_path": descriptor.shard_path,
                                "descriptor_path": classify_descriptor_storage(descriptor),
                                "reason": classify_light_descriptor_error(error),
                                "error": str(error),
                            }
                        )
                        continue
                    raise

                if tar_writer is None and clip_samples:
                    os.makedirs(os.path.dirname(task["output_path"]), exist_ok=True)
                    tar_writer = tarfile.open(task["tmp_path"], "w")

                for key, image_bytes, lowdim_bytes, mano_bytes, depth_bytes, meta_bytes in clip_samples:
                    add_prepared_sample_bytes_to_tar(
                        tar_writer,
                        key,
                        image_bytes,
                        lowdim_bytes,
                        mano_bytes,
                        depth_bytes,
                        meta_bytes,
                    )
                    frames_written += 1

                touched_episodes.add(cache_key)
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if os.path.exists(task["tmp_path"]):
            os.remove(task["tmp_path"])
        raise

    if tar_writer is not None:
        tar_writer.close()

    if frames_written == 0:
        if os.path.exists(task["tmp_path"]):
            os.remove(task["tmp_path"])
    else:
        os.replace(task["tmp_path"], task["output_path"])

    return {
        "shard_idx": task["shard_idx"],
        "frames_written": frames_written,
        "episodes_written": len(touched_episodes),
        "skipped_episodes": skipped_episodes,
        "skipped_clips": len(skipped_clips),
        "skipped_clip_details": skipped_clips,
        "output_path": task["output_path"],
    }


__all__ = [
    "normalize_mano_devices",
    "plan_manifest_shards",
    "repeat_manifest_episodes",
    "worker_init",
    "worker_process_shard",
]
