"""Reusable backend for dataset viewers.

This module keeps sample discovery and MANO loading separate from any
particular frontend so tools can reuse the same episode selection logic.
"""

from __future__ import annotations

import sys
import tarfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from tools.ops import webdataset_visualizer as wv


@dataclass
class EpisodeFrame:
    summary: wv.SampleSummary
    sample: dict
    meta: Optional[dict]
    lowdim_array: np.ndarray
    mano_array: Optional[np.ndarray]
    presence: Optional[int]
    frame_idx: int


def _new_header(sample_key: str) -> dict:
    return {
        "key": sample_key,
        "meta_bytes": None,
        "fields_present": set(),
    }


def iter_shard_sample_headers(shard_path: str):
    """Yield metadata-only sample headers without reading image/lowdim payloads."""
    current_header = None
    with tarfile.open(shard_path, "r:*") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue

            sample_key, _, field_name = wv.split_sample_member_name(member.name)
            if sample_key is None:
                continue

            if current_header is None:
                current_header = _new_header(sample_key)
            elif current_header["key"] != sample_key:
                yield current_header
                current_header = _new_header(sample_key)

            current_header["fields_present"].add(field_name)
            if field_name != "meta_bytes":
                continue

            member_file = tar_reader.extractfile(member)
            if member_file is not None:
                current_header["meta_bytes"] = member_file.read()

    if current_header is not None:
        yield current_header


def build_sample_summary_from_header(header: dict, shard_path: str, sample_id: int) -> wv.SampleSummary:
    meta, meta_error = wv.decode_meta(header["meta_bytes"])
    missing_fields = [
        field_name
        for field_name in wv.REQUIRED_SAMPLE_FIELDS
        if field_name not in header["fields_present"]
    ]
    instruction_text = wv.truncate_text(wv.normalize_instruction(meta))
    clip_id = None if meta is None else meta.get("clip_id")
    instruction_num = None if meta is None else meta.get("instruction_num")
    presence = None if meta is None else meta.get("presence")
    broken = bool(missing_fields) or meta_error is not None
    return wv.SampleSummary(
        id=sample_id,
        key=header["key"],
        episode_key=str(clip_id) if clip_id is not None else wv.sample_key_to_episode_key(header["key"]),
        shard_path=shard_path,
        shard_name=Path(shard_path).name,
        clip_id=str(clip_id) if clip_id is not None else None,
        instruction_preview=instruction_text,
        instruction_num=int(instruction_num) if isinstance(instruction_num, (int, np.integer)) else instruction_num,
        presence=int(presence) if isinstance(presence, (int, np.integer)) else presence,
        broken=broken,
        missing_fields=missing_fields,
    )


def scan_sample_summaries(
    tar_paths: list[str],
    *,
    sample_limit: Optional[int],
    episode_limit: Optional[int],
    filter_key: str,
    filter_presence: Optional[int],
) -> list[wv.SampleSummary]:
    entries: list[wv.SampleSummary] = []
    filter_key_lower = filter_key.lower()
    seen_episodes: set[str] = set()

    print(f"Scanning {len(tar_paths)} tar shard(s) for sample metadata...", flush=True)
    for shard_idx, shard_path in enumerate(tar_paths, start=1):
        matched_in_shard = 0
        print(f"[scan] {shard_idx}/{len(tar_paths)} {Path(shard_path).name}", flush=True)
        for header in iter_shard_sample_headers(shard_path):
            summary = build_sample_summary_from_header(header, shard_path, len(entries))
            if filter_key_lower:
                haystack = " ".join(
                    [
                        summary.key,
                        summary.episode_key,
                        summary.clip_id or "",
                        summary.instruction_preview or "",
                        summary.shard_name,
                    ]
                ).lower()
                if filter_key_lower not in haystack:
                    continue
            if filter_presence is not None and summary.presence != filter_presence:
                continue

            if episode_limit is not None and summary.episode_key not in seen_episodes and len(seen_episodes) >= episode_limit:
                print(f"Reached episode limit {episode_limit}; stopping scan.", flush=True)
                return entries

            entries.append(summary)
            seen_episodes.add(summary.episode_key)
            matched_in_shard += 1
            if len(entries) <= 5 or len(entries) % 500 == 0:
                print(
                    f"  matched={len(entries)} episodes={len(seen_episodes)} current_shard={matched_in_shard}",
                    flush=True,
                )
            if sample_limit is not None and len(entries) >= sample_limit:
                print(f"Reached sample limit {sample_limit}; stopping scan.", flush=True)
                return entries
        print(
            (
                f"  done shard {shard_idx}/{len(tar_paths)} matched_in_shard={matched_in_shard} "
                f"episodes={len(seen_episodes)} total={len(entries)}"
            ),
            flush=True,
        )

    print(f"Finished scan: {len(entries)} matched sample(s).", flush=True)
    return entries


def summarize_episode_candidates(summaries: list[wv.SampleSummary]) -> list[dict]:
    ordered = []
    by_key: dict[str, dict] = {}
    for summary in summaries:
        item = by_key.get(summary.episode_key)
        if item is None:
            item = {
                "episode_key": summary.episode_key,
                "clip_id": summary.clip_id,
                "sample_count": 0,
                "shards": set(),
                "instruction_preview": summary.instruction_preview,
            }
            by_key[summary.episode_key] = item
            ordered.append(item)
        item["sample_count"] += 1
        item["shards"].add(summary.shard_name)
    return ordered


def select_episode_key(
    summaries: list[wv.SampleSummary],
    *,
    clip_id: Optional[str] = None,
    episode_key: Optional[str] = None,
    episode_index: Optional[int] = None,
) -> str:
    candidates = summaries
    if clip_id:
        candidates = [summary for summary in candidates if summary.clip_id == clip_id]
        if not candidates:
            raise ValueError(f"No samples matched clip_id={clip_id!r}")
    if episode_key:
        candidates = [summary for summary in candidates if summary.episode_key == episode_key]
        if not candidates:
            raise ValueError(f"No samples matched episode_key={episode_key!r}")

    episodes = summarize_episode_candidates(candidates)
    if not episodes:
        raise ValueError("No episodes matched the current filters.")

    if episode_index is not None:
        if episode_index < 1 or episode_index > len(episodes):
            raise ValueError(f"episode_index must be in [1, {len(episodes)}], got {episode_index}")
        return str(episodes[episode_index - 1]["episode_key"])

    if len(episodes) == 1:
        return str(episodes[0]["episode_key"])

    if not sys.stdin.isatty():
        raise ValueError(
            f"Matched {len(episodes)} episodes. Pass --clip-id, --episode-key, or --episode-index to disambiguate."
        )

    print("Matched multiple episodes; choose one:", flush=True)
    for idx, item in enumerate(episodes, start=1):
        shard_list = ",".join(sorted(item["shards"]))
        preview = item["instruction_preview"] or "-"
        print(
            f"  {idx}. episode={item['episode_key']} clip_id={item['clip_id'] or '-'} "
            f"frames={item['sample_count']} shard={shard_list} instruction={preview}",
            flush=True,
        )

    while True:
        raw = input(f"Select episode [1-{len(episodes)}]: ").strip()
        if not raw:
            continue
        try:
            chosen = int(raw)
        except ValueError:
            print("Please enter an integer index.", flush=True)
            continue
        if 1 <= chosen <= len(episodes):
            return str(episodes[chosen - 1]["episode_key"])
        print(f"Index out of range: {chosen}", flush=True)


class EpisodeViewerBackend:
    """Shared backend for single-episode viewers."""

    def __init__(
        self,
        summaries: list[wv.SampleSummary],
        *,
        descriptor_manifest: Optional[str] = None,
        mano_dir: Optional[str] = None,
        mano_device: str = "cpu",
    ):
        self.summaries = summaries
        self.descriptor_manifest = descriptor_manifest
        self.mano_dir = mano_dir
        self.mano_device = mano_device
        self.clip_to_seq_folder = self._load_clip_lookup(descriptor_manifest)
        self._mano_runtime = None
        self._mano_sample_cache: dict[int, dict] = {}
        self._cache_lock = threading.RLock()

    def _load_clip_lookup(self, manifest_path: Optional[str]) -> dict[str, str]:
        if not manifest_path:
            return {}

        from lib.pipeline.clips.clip_manifest import load_clip_manifest

        print(f"Loading descriptor manifest: {manifest_path}", flush=True)
        records = load_clip_manifest(manifest_path)
        clip_to_seq = {record.clip_id: record.descriptor.seq_folder for record in records}
        print(f"Loaded {len(clip_to_seq)} clip -> seq_folder mappings", flush=True)
        return clip_to_seq

    def require_mano_manifest(self):
        return None

    def _ensure_mano_runtime(self):
        if self._mano_runtime is not None:
            return self._mano_runtime

        import torch

        requested = self.mano_device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print(f"MANO device {requested} requested but CUDA is unavailable; falling back to cpu.", flush=True)
            requested = "cpu"
        device = torch.device(requested)
        print(f"Initializing MANO runtime on {device} ...", flush=True)
        mano_right, mano_left = wv.build_manopth_models(
            device,
            mano_dir=self.mano_dir,
            center_idx=wv.MANO_CENTER_IDX,
            flat_hand_mean=wv.MANO_FLAT_HAND_MEAN,
            ncomps=wv.MANO_PCA_DIMS,
        )
        self._mano_runtime = {
            "device": device,
            "mano_right": mano_right,
            "mano_left": mano_left,
        }
        return self._mano_runtime

    def _compute_mano_sample_cache(self, summary: wv.SampleSummary, lowdim_array: np.ndarray, mano_array: np.ndarray) -> dict:
        with self._cache_lock:
            cached = self._mano_sample_cache.get(summary.id)
        if cached is not None:
            return cached

        cache = wv._build_mano_frame_from_sample(lowdim_array, mano_array, self._ensure_mano_runtime())
        with self._cache_lock:
            self._mano_sample_cache[summary.id] = cache
        return cache

    def _get_mano_frame(self, summary: wv.SampleSummary, lowdim_array: np.ndarray, mano_array: np.ndarray) -> dict:
        mano_frame = self._compute_mano_sample_cache(summary, lowdim_array, mano_array)
        return {
            "frame_idx": wv.parse_frame_index(summary.key),
            "c2w": mano_frame["c2w"],
            "intrinsic": mano_frame["intrinsic"],
            "camera_convention": mano_frame["camera_convention"],
            "left_verts": mano_frame["left_verts"],
            "left_joints": mano_frame["left_joints"],
            "right_verts": mano_frame["right_verts"],
            "right_joints": mano_frame["right_joints"],
        }

    def build_keypoint_frame(self, summary: wv.SampleSummary, lowdim_array: np.ndarray, mano_array: Optional[np.ndarray] = None) -> dict:
        fields = wv._decode_lowdim_fields(lowdim_array)
        notes = ["All 3D lowdim fields are stored in the HaWoR/SLAM world frame."]
        c2w, _ = wv._resolve_camera_c2w(fields["camera_w2c"])
        keypoint_frame = {
            "c2w": c2w,
            "intrinsic": fields["camera_intrinsic"],
            "left_wrist": fields["left_wrist_world"],
            "right_wrist": fields["right_wrist_world"],
            "left_tips": fields["left_fingertips_world"],
            "right_tips": fields["right_fingertips_world"],
            "left_rotmat": wv.rot6_to_rotmat(fields["left_root_rot6d"]),
            "right_rotmat": wv.rot6_to_rotmat(fields["right_root_rot6d"]),
            "anchor_source": "lowdim_wrist_world",
            "notes": notes,
        }
        keypoint_frame["hands"] = [
            {
                "side": "left",
                "wrist": np.asarray(keypoint_frame["left_wrist"], dtype=np.float32),
                "tips": np.asarray(keypoint_frame["left_tips"], dtype=np.float32),
                "rotmat": np.asarray(keypoint_frame["left_rotmat"], dtype=np.float32),
                "color": (255, 0, 255),
                "label": "L-lowdim",
                "draw_axes": True,
            },
            {
                "side": "right",
                "wrist": np.asarray(keypoint_frame["right_wrist"], dtype=np.float32),
                "tips": np.asarray(keypoint_frame["right_tips"], dtype=np.float32),
                "rotmat": np.asarray(keypoint_frame["right_rotmat"], dtype=np.float32),
                "color": (0, 255, 0),
                "label": "R-lowdim",
                "draw_axes": True,
            },
        ]
        notes.append("camera extrinsic is interpreted as fixed world-to-camera (w2c) and inverted for display.")
        return keypoint_frame

    def build_mano_frame(self, summary: wv.SampleSummary, lowdim_array: np.ndarray, mano_array: np.ndarray) -> dict:
        return self._get_mano_frame(summary, lowdim_array, mano_array)

    def load_episode_frames(self, episode_key: str) -> list[EpisodeFrame]:
        episode_summaries = [summary for summary in self.summaries if summary.episode_key == episode_key]
        if not episode_summaries:
            raise ValueError(f"Episode not found: {episode_key}")

        by_shard: dict[str, set[str]] = {}
        for summary in episode_summaries:
            by_shard.setdefault(summary.shard_path, set()).add(summary.key)

        loaded_samples: dict[tuple[str, str], dict] = {}
        for shard_path, target_keys in by_shard.items():
            for sample in wv.iter_shard_samples(shard_path):
                sample_key = sample["key"]
                if sample_key in target_keys:
                    loaded_samples[(shard_path, sample_key)] = sample
                    if len([key for key in loaded_samples if key[0] == shard_path]) >= len(target_keys):
                        break

        frames: list[EpisodeFrame] = []
        for summary in sorted(episode_summaries, key=lambda item: wv.parse_frame_index(item.key)):
            sample = loaded_samples.get((summary.shard_path, summary.key))
            if sample is None:
                raise KeyError(f"Failed to load sample {summary.key} from shard {summary.shard_path}")
            if sample["image_bytes"] is None:
                raise ValueError(f"missing image.jpg for sample {summary.key}")

            meta, _ = wv.decode_meta(sample["meta_bytes"])
            lowdim_summary = wv.summarize_lowdim(sample["lowdim_bytes"])
            if "error" in lowdim_summary:
                raise ValueError(f"Failed to load lowdim for sample {summary.key}: {lowdim_summary['error']}")
            mano_summary = wv.summarize_mano(sample.get("mano_bytes"))
            frames.append(
                EpisodeFrame(
                    summary=summary,
                    sample=sample,
                    meta=meta,
                    lowdim_array=lowdim_summary["array"],
                    mano_array=mano_summary.get("array"),
                    presence=None if meta is None else meta.get("presence"),
                    frame_idx=wv.parse_frame_index(summary.key),
                )
            )
        return frames
