"""Lightweight clip discovery and indexing for WebDataset factory directories."""

from __future__ import annotations

import json
import os
import pickle
import re
import tarfile
import gzip
from pathlib import Path
from typing import Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        del args, kwargs
        return iterable if iterable is not None else []

from lib.pipeline.datasets.descriptors import ClipDescriptor as VideoDescriptor


_FRAME_RE = re.compile(r"^(.+)_(f\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
CLIP_INDEX_FORMAT_VERSION = 3
CLIP_OFFSET_INDEX_FORMAT_VERSION = 1


def _list_tar_shards(factory_dir: str) -> list[str]:
    return sorted(name for name in os.listdir(factory_dir) if name.endswith(".tar"))


def _parse_frame_name(name: str):
    """Parse a frame filename into (clip_id, frame_sort_key, extension)."""
    match = _FRAME_RE.match(name)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def _clip_index_cache_path(factory_dir: str) -> str:
    return os.path.join(factory_dir, "_clip_index.json")


def _legacy_index_cache_path(factory_dir: str) -> str:
    return os.path.join(factory_dir, "_video_index.json")


def _clip_offset_index_path(factory_dir: str) -> str:
    return os.path.join(factory_dir, "_clip_offset_index.pkl.gz")


def _new_clip_summary(shard_file: str, clip_id: str, frame_sort: str, ext: str, video_name: str | None = None) -> dict:
    frame_idx = int(frame_sort[1:])
    return {
        "shard": shard_file,
        "video_name": video_name or clip_id,
        "frame_count": 0,
        "frame_ext": f".{ext.lower()}",
        "frame_index_width": max(1, len(frame_sort) - 1),
        "frame_start_idx": frame_idx,
        "_min_frame_idx": frame_idx,
        "_max_frame_idx": frame_idx,
    }


def _basename(name: str) -> str:
    return Path(name).name


def _legacy_frame_name(frame_entry) -> Optional[str]:
    if isinstance(frame_entry, str):
        return frame_entry
    if isinstance(frame_entry, dict):
        name = frame_entry.get("name")
        if isinstance(name, str):
            return name
    return None


def _summarize_legacy_clip_entry(clip_id: str, info: dict) -> Optional[dict]:
    shard_file = info.get("shard")
    if not isinstance(shard_file, str) or not shard_file:
        return None

    if "frame_count" in info:
        frame_count = int(info.get("frame_count") or 0)
        return {
            "shard": shard_file,
            "video_name": info.get("video_name") or clip_id,
            "frame_count": frame_count,
            "frame_ext": info.get("frame_ext") or ".jpg",
            "frame_index_width": int(info.get("frame_index_width") or 6),
            "frame_start_idx": int(info.get("frame_start_idx") or 0),
        }

    frames = info.get("frames") or []
    if not isinstance(frames, list):
        return None

    frame_count = int(info.get("num_frames") or len(frames))
    frame_ext = ".jpg"
    frame_index_width = 6
    frame_start_idx = 0
    min_frame_idx = None
    max_frame_idx = None

    for frame_entry in frames:
        frame_name = _legacy_frame_name(frame_entry)
        if not frame_name:
            continue
        parsed = _parse_frame_name(_basename(frame_name))
        if parsed is None:
            continue
        _, frame_sort, ext = parsed
        frame_idx = int(frame_sort[1:])
        if min_frame_idx is None or frame_idx < min_frame_idx:
            min_frame_idx = frame_idx
            frame_start_idx = frame_idx
        if max_frame_idx is None or frame_idx > max_frame_idx:
            max_frame_idx = frame_idx
        frame_ext = f".{ext.lower()}"
        frame_index_width = max(1, len(frame_sort) - 1)

    if min_frame_idx is None or max_frame_idx is None:
        return None
    if (max_frame_idx - min_frame_idx + 1) != frame_count:
        return None

    return {
        "shard": shard_file,
        "video_name": info.get("video_name") or clip_id,
        "frame_count": frame_count,
        "frame_ext": frame_ext,
        "frame_index_width": frame_index_width,
        "frame_start_idx": frame_start_idx,
    }


def _summarize_legacy_index(index: dict, factory_dir: str) -> Optional[dict]:
    entries = index.get("clips")
    if not isinstance(entries, dict):
        entries = index.get("videos")
    if not isinstance(entries, dict) or not entries:
        return None

    clips = {}
    for clip_id, info in entries.items():
        if not isinstance(info, dict):
            return None
        summary = _summarize_legacy_clip_entry(clip_id, info)
        if summary is None:
            return None
        clips[clip_id] = summary

    shards = index.get("shards")
    if not isinstance(shards, list) or not shards:
        shards = _list_tar_shards(factory_dir)

    return {
        "format_version": CLIP_INDEX_FORMAT_VERSION,
        "clips": clips,
        "shards": shards,
        "num_videos": len(clips),
        "num_shards": len(shards),
    }


def _is_clip_index_stale(index: dict, factory_dir: str) -> bool:
    current_shards = _list_tar_shards(factory_dir)
    if current_shards != index.get("shards", []):
        return True
    if int(index.get("format_version", 0)) != CLIP_INDEX_FORMAT_VERSION:
        return True
    clips = index.get("clips", {})
    if clips:
        first_clip = next(iter(clips.values()))
        required = ("shard", "frame_count", "frame_ext", "frame_index_width", "frame_start_idx")
        if any(key not in first_clip for key in required):
            return True
    return False


def _load_json_file(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def _write_json_file(path: str, payload: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    except OSError:
        pass


def _load_pickle_file(path: str):
    try:
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _write_pickle_file(path: str, payload) -> None:
    try:
        with gzip.open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:
        pass


def build_video_index(factory_dir: str) -> dict:
    """Scan tar shards and build a lightweight clip-level index."""
    factory_dir = str(factory_dir)
    factory_path = Path(factory_dir)
    shard_files = _list_tar_shards(factory_dir)
    if not shard_files:
        raise FileNotFoundError(f"No tar shards found in {factory_dir}")

    clips: Dict[str, dict] = {}
    clip_offsets: Dict[str, list[list[int]]] = {}
    skipped_cross_shard_clips: set[str] = set()
    n_frames_total = 0
    pbar = tqdm(shard_files, desc="Scanning shards", unit="shard")
    for shard_file in pbar:
        shard_path = str(factory_path / shard_file)
        json_meta: Dict[str, str] = {}
        with tarfile.open(shard_path, "r|") as tar_reader:
            for member in tar_reader:
                if not member.isfile():
                    continue

                member_name = _basename(member.name)
                if member_name.endswith((".jpg", ".jpeg", ".png")):
                    parsed = _parse_frame_name(member_name)
                    if parsed is None:
                        continue
                    clip_id, frame_sort, ext = parsed

                    if clip_id in skipped_cross_shard_clips:
                        continue

                    if clip_id not in clips:
                        clips[clip_id] = _new_clip_summary(
                            shard_file,
                            clip_id,
                            frame_sort,
                            ext,
                            video_name=json_meta.pop(clip_id, None),
                        )
                        clip_offsets[clip_id] = []
                    clip_summary = clips[clip_id]
                    if clip_summary["shard"] != shard_file:
                        skipped_cross_shard_clips.add(clip_id)
                        clips.pop(clip_id, None)
                        clip_offsets.pop(clip_id, None)
                        continue

                    frame_idx = int(frame_sort[1:])
                    if clip_summary["frame_ext"] != f".{ext.lower()}":
                        raise RuntimeError(
                            f"Clip {clip_id} has mixed frame extensions: "
                            f"{clip_summary['frame_ext']} and .{ext.lower()}"
                        )

                    clip_summary["frame_count"] += 1
                    clip_summary["frame_start_idx"] = min(clip_summary["frame_start_idx"], frame_idx)
                    clip_summary["_min_frame_idx"] = min(int(clip_summary["_min_frame_idx"]), frame_idx)
                    clip_summary["_max_frame_idx"] = max(int(clip_summary["_max_frame_idx"]), frame_idx)
                    clip_summary["frame_index_width"] = max(
                        int(clip_summary["frame_index_width"]),
                        max(1, len(frame_sort) - 1),
                    )
                    clip_offsets[clip_id].append(
                        [frame_idx, int(member.offset_data), int(member.size)]
                    )
                    n_frames_total += 1
                    if n_frames_total % 500 == 0:
                        pbar.set_postfix(clips=len(clips), frames=n_frames_total)
                elif member_name.endswith(".json"):
                    base_name = member_name[:-5]
                    parsed = _parse_frame_name(f"{base_name}.jpg")
                    if parsed is None:
                        continue
                    clip_id = parsed[0]
                    try:
                        payload = tar_reader.extractfile(member)
                        if payload is None:
                            continue
                        meta = json.loads(payload.read())
                    except Exception:
                        continue
                    video_name = meta.get("video_name", "")
                    if not video_name:
                        continue
                    if clip_id in clips and not clips[clip_id]["video_name"]:
                        clips[clip_id]["video_name"] = video_name
                    else:
                        json_meta[clip_id] = video_name

    pbar.set_postfix(clips=len(clips), frames=n_frames_total)
    pbar.close()

    if skipped_cross_shard_clips:
        skipped_preview = ", ".join(sorted(skipped_cross_shard_clips)[:8])
        print(
            (
                f"Warning: skipped {len(skipped_cross_shard_clips)} clip(s) in {factory_dir} "
                f"because they span multiple shards. "
                f"Examples: {skipped_preview}"
            ),
            flush=True,
        )

    for clip_id, clip_summary in clips.items():
        min_frame_idx = int(clip_summary.pop("_min_frame_idx"))
        max_frame_idx = int(clip_summary.pop("_max_frame_idx"))
        expected_count = max_frame_idx - min_frame_idx + 1
        if expected_count != int(clip_summary["frame_count"]):
            raise RuntimeError(
                f"Clip {clip_id} in {factory_dir} has non-contiguous frame indices: "
                f"start={min_frame_idx}, end={max_frame_idx}, frame_count={clip_summary['frame_count']}"
            )
        clip_summary["frame_start_idx"] = min_frame_idx
        offsets = clip_offsets.get(clip_id) or []
        offsets.sort(key=lambda item: int(item[0]))
        if len(offsets) != expected_count:
            raise RuntimeError(
                f"Clip {clip_id} in {factory_dir} has inconsistent offset count: "
                f"expected={expected_count}, got={len(offsets)}"
            )
        if [int(item[0]) for item in offsets] != list(range(min_frame_idx, max_frame_idx + 1)):
            raise RuntimeError(
                f"Clip {clip_id} in {factory_dir} has non-contiguous offset entries: "
                f"start={min_frame_idx}, end={max_frame_idx}"
            )
        clip_offsets[clip_id] = [[int(item[1]), int(item[2])] for item in offsets]

    _write_pickle_file(
        _clip_offset_index_path(factory_dir),
        {
            "format_version": CLIP_OFFSET_INDEX_FORMAT_VERSION,
            "clips": clip_offsets,
            "shards": shard_files,
        },
    )

    return {
        "format_version": CLIP_INDEX_FORMAT_VERSION,
        "clips": clips,
        "shards": shard_files,
        "num_videos": len(clips),
        "num_shards": len(shard_files),
        "skipped_cross_shard_clips": sorted(skipped_cross_shard_clips),
    }


def load_or_build_index(factory_dir: str, force_rebuild: bool = False) -> dict:
    """Load a lightweight clip index or build it from legacy cache / tar shards."""
    clip_index_path = _clip_index_cache_path(factory_dir)
    if not force_rebuild and os.path.exists(clip_index_path):
        clip_index = _load_json_file(clip_index_path)
        if clip_index is not None and not _is_clip_index_stale(clip_index, factory_dir):
            return clip_index

    legacy_index_path = _legacy_index_cache_path(factory_dir)
    if not force_rebuild and os.path.exists(legacy_index_path):
        legacy_index = _load_json_file(legacy_index_path)
        if legacy_index is not None:
            clip_index = _summarize_legacy_index(legacy_index, factory_dir)
            if clip_index is not None and not _is_clip_index_stale(clip_index, factory_dir):
                _write_json_file(clip_index_path, clip_index)
                return clip_index

    clip_index = build_video_index(factory_dir)
    _write_json_file(clip_index_path, clip_index)
    return clip_index


def load_clip_frame_offsets(factory_dir: str, clip_id: str) -> Optional[list[list[int]]]:
    factory_dir = str(Path(factory_dir).resolve())
    offset_index = _load_pickle_file(_clip_offset_index_path(factory_dir))
    if not isinstance(offset_index, dict):
        return None
    if int(offset_index.get("format_version", 0)) != CLIP_OFFSET_INDEX_FORMAT_VERSION:
        return None
    clips = offset_index.get("clips")
    if not isinstance(clips, dict):
        return None
    offsets = clips.get(clip_id)
    if offsets is None:
        return None
    return [[int(offset), int(size)] for offset, size in offsets]


def collect_videos_from_factory(factory_dir: str) -> List[VideoDescriptor]:
    """Collect all clips from a shard directory as lightweight descriptors."""
    factory_dir = str(Path(factory_dir).resolve())
    index = load_or_build_index(factory_dir)

    descriptors = []
    for clip_id, info in sorted(index["clips"].items()):
        shard_path = os.path.join(factory_dir, info["shard"])
        seq_folder = os.path.join(factory_dir, "outputs", clip_id)
        descriptors.append(
            VideoDescriptor(
                clip_id=clip_id,
                clip_name=info.get("video_name") or clip_id,
                storage_kind="tar_shard",
                root_dir=factory_dir,
                shard_path=shard_path,
                frame_names=[],
                seq_folder=seq_folder,
                frame_offsets=None,
                frame_count_override=int(info.get("frame_count", 0)),
                extra={
                    "frame_ext": info.get("frame_ext", ".jpg"),
                    "frame_start_idx": int(info.get("frame_start_idx", 0)),
                    "frame_index_width": int(info.get("frame_index_width", 6)),
                },
            )
        )
    return descriptors


def collect_videos_from_factories(factory_dirs: List[str]) -> List[VideoDescriptor]:
    """Collect clips from multiple shard directories."""
    all_descriptors = []
    for factory_dir in factory_dirs:
        all_descriptors.extend(collect_videos_from_factory(factory_dir))
    return all_descriptors
