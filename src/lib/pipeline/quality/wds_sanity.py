"""Standalone WebDataset sanity-check helpers."""

from __future__ import annotations

import io
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from lib.pipeline.exporters.webdataset_rewriter import iter_shard_paths, iter_shard_samples
from lib.pipeline.quality.quality_metrics import (
    LOWDIM_SIZE,
    decode_lowdim,
    parse_frame_index,
    parse_instruction_metadata,
    validate_lowdim_numeric_sanity,
)

LOWDIM_STATE_SLICE = slice(0, 48)
LOWDIM_ACTION_SLICE = slice(48, 96)
LOWDIM_EXTRINSIC_SLICE = slice(96, 112)
LOWDIM_INTRINSIC_SLICE = slice(112, 116)
REQUIRED_META_KEYS = ("clip_id", "instruction", "instruction_num", "presence")
HARD_FILTER_ISSUES = {
    "missing_image",
    "missing_lowdim",
    "missing_meta",
    "missing_depth",
    "invalid_meta",
    "missing_meta_keys",
    "lowdim_decode_failure",
    "nonfinite_lowdim",
    "invalid_rot6d",
    "invalid_state",
    "invalid_action",
    "invalid_extrinsic",
    "invalid_intrinsic",
    "missing_instruction",
    "empty_instruction",
    "instruction_num_mismatch",
}


def build_lowdim_dimension_names() -> list[str]:
    names: list[str] = []

    def add_wrist(prefix: str) -> None:
        names.extend(
            [
                f"{prefix}.left_wrist.x",
                f"{prefix}.left_wrist.y",
                f"{prefix}.left_wrist.z",
                f"{prefix}.right_wrist.x",
                f"{prefix}.right_wrist.y",
                f"{prefix}.right_wrist.z",
            ]
        )
        names.extend(f"{prefix}.left_root_rot6d.{idx}" for idx in range(6))
        names.extend(f"{prefix}.right_root_rot6d.{idx}" for idx in range(6))

    def add_fingertips(prefix: str) -> None:
        axes = ("x", "y", "z")
        for hand in ("left", "right"):
            for finger_idx in range(5):
                for axis in axes:
                    names.append(f"{prefix}.{hand}_fingertip{finger_idx}.{axis}")

    add_wrist("state")
    add_fingertips("state")
    add_wrist("action")
    add_fingertips("action")
    for row in range(4):
        for col in range(4):
            names.append(f"camera_w2c.r{row}c{col}")
    names.extend(("camera_intrinsic.fx", "camera_intrinsic.fy", "camera_intrinsic.cx", "camera_intrinsic.cy"))
    if len(names) != LOWDIM_SIZE:
        raise AssertionError(f"Expected {LOWDIM_SIZE} lowdim names, got {len(names)}")
    return names


LOWDIM_DIMENSION_NAMES = build_lowdim_dimension_names()


def clip_id_from_sample(sample: dict, meta: dict | None) -> str:
    if isinstance(meta, dict):
        clip_id = meta.get("clip_id")
        if clip_id:
            return str(clip_id)
    return sample["key"].rsplit("_f", 1)[0]


def missing_sample_fields(sample: dict) -> list[str]:
    missing = []
    for field_name in ("image_bytes", "lowdim_bytes", "meta_bytes"):
        if sample.get(field_name) is None:
            missing.append(field_name)
    return missing


def build_issue_record(reason: str, *, clip_id: str, sample_key: str, shard_name: str, detail=None) -> dict:
    record = {
        "reason": reason,
        "clip_id": clip_id,
        "sample_key": sample_key,
        "shard_name": shard_name,
    }
    if detail is not None:
        record["detail"] = detail
    return record


@dataclass
class EpisodeRenderFrame:
    image_bytes: bytes
    frame_index: int


@dataclass
class EpisodeAccumulator:
    clip_id: str
    shard_name: str
    buffer_images: bool = False
    frames_total: int = 0
    issue_reasons: set[str] = field(default_factory=set)
    instruction_preview: str = ""
    frames: list[EpisodeRenderFrame] = field(default_factory=list)

    def add_frame(self, image_bytes: bytes | None, frame_index: int) -> None:
        if self.buffer_images and image_bytes is not None:
            self.frames.append(EpisodeRenderFrame(image_bytes=image_bytes, frame_index=frame_index))

    def mark_issue(self, reason: str) -> None:
        self.issue_reasons.add(reason)

    def to_summary(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "shard_name": self.shard_name,
            "frames_total": self.frames_total,
            "issue_reasons": sorted(self.issue_reasons),
            "instruction_preview": self.instruction_preview,
        }


class LowdimStatCollector:
    """Exact per-dimension stats with temp-file backing for percentiles."""

    def __init__(self, *, temp_dir: str | None = None):
        temp_root = None if temp_dir is None else str(Path(temp_dir))
        handle = tempfile.NamedTemporaryFile(prefix="wds_sanity_lowdim_", suffix=".bin", dir=temp_root, delete=False)
        self._temp_path = Path(handle.name)
        self._handle = handle
        self._count = 0
        self._sum = np.zeros((LOWDIM_SIZE,), dtype=np.float64)
        self._min = np.full((LOWDIM_SIZE,), np.inf, dtype=np.float64)
        self._max = np.full((LOWDIM_SIZE,), -np.inf, dtype=np.float64)

    @property
    def temp_path(self) -> Path:
        return self._temp_path

    @property
    def count(self) -> int:
        return int(self._count)

    def add(self, lowdim: np.ndarray) -> None:
        array = np.asarray(lowdim, dtype=np.float32).reshape(-1)
        if array.shape != (LOWDIM_SIZE,):
            raise ValueError(f"Expected lowdim shape {(LOWDIM_SIZE,)}, got {array.shape}")
        self._handle.write(np.ascontiguousarray(array).tobytes())
        self._count += 1
        float64 = array.astype(np.float64)
        self._sum += float64
        self._min = np.minimum(self._min, float64)
        self._max = np.maximum(self._max, float64)

    def finalize(self) -> dict:
        self._handle.flush()
        self._handle.close()

        if self._count <= 0:
            return {
                "count": 0,
                "dimensions": [],
            }

        matrix = np.memmap(self._temp_path, dtype=np.float32, mode="r", shape=(self._count, LOWDIM_SIZE))
        q01 = np.percentile(matrix, 1.0, axis=0)
        q99 = np.percentile(matrix, 99.0, axis=0)
        avg = self._sum / float(self._count)
        dimensions = []
        for idx, name in enumerate(LOWDIM_DIMENSION_NAMES):
            dimensions.append(
                {
                    "index": idx,
                    "name": name,
                    "min": float(self._min[idx]),
                    "max": float(self._max[idx]),
                    "avg": float(avg[idx]),
                    "q01": float(q01[idx]),
                    "q99": float(q99[idx]),
                }
            )
        return {
            "count": int(self._count),
            "dimensions": dimensions,
        }

    def cleanup(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass
        if self._temp_path.exists():
            self._temp_path.unlink()


def init_sanity_report(
    *,
    source_shard_dir: str,
    selected_shards: list[str],
    start_shard: int,
    end_shard: int,
    total_shards_available: int,
) -> dict:
    return {
        "source_shard_dir": str(Path(source_shard_dir).resolve()),
        "shard_selection": {
            "start_shard": int(start_shard),
            "end_shard": int(end_shard),
            "selected_shards": len(selected_shards),
            "total_shards_available": int(total_shards_available),
            "shards": [Path(path).name for path in selected_shards],
        },
        "summary": {
            "samples_total": 0,
            "episodes_total": 0,
            "issue_episodes": 0,
            "hard_filter_drop_episodes": 0,
            "valid_lowdim_frames": 0,
        },
        "checks": {
            "missing_image_samples": 0,
            "missing_lowdim_samples": 0,
            "missing_meta_samples": 0,
            "meta_parse_failures": 0,
            "meta_missing_required_keys": 0,
            "image_decode_failures": 0,
            "lowdim_decode_failures": 0,
            "nonfinite_lowdim_frames": 0,
            "invalid_rot6d_frames": 0,
            "invalid_state_frames": 0,
            "invalid_action_frames": 0,
            "invalid_extrinsic_frames": 0,
            "invalid_intrinsic_frames": 0,
            "missing_instruction_frames": 0,
            "empty_instruction_frames": 0,
            "instruction_num_mismatch_frames": 0,
            "missing_mano_samples": 0,
            "missing_depth_samples": 0,
        },
        "issue_examples": [],
        "hard_filter_reason_counts": {},
        "episode_examples": {
            "problematic": [],
            "clean": [],
        },
        "renders": [],
    }


def iter_selected_shards(source_shard_dir: str, start_shard: int, end_shard: int | None) -> tuple[list[str], int, int]:
    shard_paths = list(iter_shard_paths(source_shard_dir))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_shard_dir}")
    total_shards = len(shard_paths)
    if start_shard < 0 or start_shard >= total_shards:
        raise ValueError(f"--start_shard {start_shard} is out of range [0, {total_shards})")
    resolved_end = total_shards if end_shard is None else int(end_shard)
    if resolved_end < start_shard or resolved_end > total_shards:
        raise ValueError(f"--end_shard {resolved_end} is out of range [{start_shard}, {total_shards}]")
    selected = shard_paths[start_shard:resolved_end]
    if not selected:
        raise RuntimeError(f"No shards selected in range [{start_shard}, {resolved_end}) from {source_shard_dir}")
    return selected, total_shards, resolved_end


def decode_image_shape(image_bytes: bytes) -> tuple[int, int] | None:
    import cv2

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    return int(height), int(width)


def make_frame_overlay(image_bgr: np.ndarray, footer: str | None) -> np.ndarray:
    if not footer:
        return image_bgr
    import cv2

    canvas = image_bgr.copy()
    cv2.rectangle(canvas, (0, max(0, canvas.shape[0] - 30)), (canvas.shape[1], canvas.shape[0]), (16, 16, 16), -1)
    cv2.putText(
        canvas,
        footer,
        (12, canvas.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render_episode_video(
    episode: EpisodeAccumulator,
    *,
    output_dir: str,
    max_frames: int | None = None,
    fps: int = 15,
    overlay_text: bool = True,
) -> Optional[dict]:
    if not episode.frames:
        return None

    import cv2

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    first = cv2.imdecode(np.frombuffer(episode.frames[0].image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        return None
    height, width = first.shape[:2]
    output_path = output_root / f"{episode.clip_id}.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for idx, frame in enumerate(episode.frames):
            if max_frames is not None and idx >= max_frames:
                break
            image = cv2.imdecode(np.frombuffer(frame.image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            footer = None
            if overlay_text:
                footer = f"{episode.clip_id} | frame={frame.frame_index}"
            writer.write(make_frame_overlay(image, footer))
    finally:
        writer.release()

    return {
        "clip_id": episode.clip_id,
        "path": str(output_path.resolve()),
        "issue_reasons": sorted(episode.issue_reasons),
    }


def analyze_webdataset(
    *,
    source_shard_dir: str,
    start_shard: int = 0,
    end_shard: int | None = None,
    sample_limit: int | None = None,
    episode_limit: int | None = None,
    render_dir: str | None = None,
    render_episodes: int = 0,
    render_max_frames: int | None = None,
    render_fps: int = 15,
    decode_images: bool = True,
    allow_empty_instruction: bool = False,
    require_depth: bool = False,
    temp_dir: str | None = None,
    max_issue_examples: int = 32,
) -> dict:
    selected_shards, total_shards, resolved_end = iter_selected_shards(source_shard_dir, start_shard, end_shard)
    report = init_sanity_report(
        source_shard_dir=source_shard_dir,
        selected_shards=selected_shards,
        start_shard=start_shard,
        end_shard=resolved_end,
        total_shards_available=total_shards,
    )

    lowdim_stats = LowdimStatCollector(temp_dir=temp_dir)
    problematic_render_candidates: list[EpisodeAccumulator] = []
    clean_render_candidates: list[EpisodeAccumulator] = []
    current_episode: EpisodeAccumulator | None = None

    def append_issue(reason: str, *, clip_id: str, sample_key: str, shard_name: str, detail=None) -> None:
        if len(report["issue_examples"]) >= max_issue_examples:
            return
        report["issue_examples"].append(
            build_issue_record(reason, clip_id=clip_id, sample_key=sample_key, shard_name=shard_name, detail=detail)
        )

    def flush_episode() -> bool:
        nonlocal current_episode
        if current_episode is None:
            return False
        report["summary"]["episodes_total"] += 1
        is_problematic = bool(current_episode.issue_reasons)
        hard_filter_reasons = sorted(current_episode.issue_reasons & HARD_FILTER_ISSUES)
        if hard_filter_reasons:
            report["summary"]["hard_filter_drop_episodes"] += 1
            for reason in hard_filter_reasons:
                report["hard_filter_reason_counts"][reason] = int(report["hard_filter_reason_counts"].get(reason, 0)) + 1
        if is_problematic:
            report["summary"]["issue_episodes"] += 1
            if len(report["episode_examples"]["problematic"]) < 8:
                report["episode_examples"]["problematic"].append(current_episode.to_summary())
            if render_dir and len(problematic_render_candidates) < max(0, render_episodes):
                problematic_render_candidates.append(current_episode)
        else:
            if len(report["episode_examples"]["clean"]) < 8:
                report["episode_examples"]["clean"].append(current_episode.to_summary())
            if render_dir and len(clean_render_candidates) < max(0, render_episodes):
                clean_render_candidates.append(current_episode)
        stop = episode_limit is not None and report["summary"]["episodes_total"] >= int(episode_limit)
        current_episode = None
        return stop

    try:
        for shard_path in selected_shards:
            shard_name = Path(shard_path).name
            for sample in iter_shard_samples(shard_path):
                if sample_limit is not None and report["summary"]["samples_total"] >= int(sample_limit):
                    break
                report["summary"]["samples_total"] += 1

                meta = None
                sample_key = sample["key"]
                missing_fields = missing_sample_fields(sample)
                if sample.get("meta_bytes") is None:
                    report["checks"]["missing_meta_samples"] += 1
                else:
                    try:
                        meta = json.loads(sample["meta_bytes"].decode("utf-8"))
                    except Exception as error:
                        report["checks"]["meta_parse_failures"] += 1
                        append_issue(
                            "meta_parse_failure",
                            clip_id=clip_id_from_sample(sample, None),
                            sample_key=sample_key,
                            shard_name=shard_name,
                            detail=str(error),
                        )

                clip_id = clip_id_from_sample(sample, meta)
                if current_episode is None or current_episode.clip_id != clip_id:
                    if flush_episode():
                        break
                    buffer_images = bool(render_dir) and (
                        len(problematic_render_candidates) < max(0, render_episodes)
                        or len(clean_render_candidates) < max(0, render_episodes)
                    )
                    current_episode = EpisodeAccumulator(clip_id=clip_id, shard_name=shard_name, buffer_images=buffer_images)

                current_episode.frames_total += 1
                try:
                    frame_index = parse_frame_index(sample_key)
                except Exception:
                    frame_index = current_episode.frames_total - 1

                if "image_bytes" in missing_fields:
                    report["checks"]["missing_image_samples"] += 1
                    current_episode.mark_issue("missing_image")
                    append_issue("missing_image", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
                else:
                    current_episode.add_frame(sample.get("image_bytes"), frame_index)
                    if decode_images:
                        try:
                            image_shape = decode_image_shape(sample["image_bytes"])
                        except Exception as error:
                            image_shape = None
                            append_issue(
                                "image_decode_failure",
                                clip_id=clip_id,
                                sample_key=sample_key,
                                shard_name=shard_name,
                                detail=str(error),
                            )
                        if image_shape is None:
                            report["checks"]["image_decode_failures"] += 1
                            current_episode.mark_issue("image_decode_failure")

                if sample.get("meta_bytes") is None:
                    current_episode.mark_issue("missing_meta")
                    append_issue("missing_meta", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)

                if "lowdim_bytes" in missing_fields:
                    report["checks"]["missing_lowdim_samples"] += 1
                    current_episode.mark_issue("missing_lowdim")
                    append_issue("missing_lowdim", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
                    lowdim = None
                else:
                    try:
                        lowdim = decode_lowdim(sample["lowdim_bytes"])
                    except Exception as error:
                        report["checks"]["lowdim_decode_failures"] += 1
                        current_episode.mark_issue("lowdim_decode_failure")
                        append_issue(
                            "lowdim_decode_failure",
                            clip_id=clip_id,
                            sample_key=sample_key,
                            shard_name=shard_name,
                            detail=str(error),
                        )
                        lowdim = None

                if sample.get("mano_bytes") is None:
                    report["checks"]["missing_mano_samples"] += 1
                if require_depth and sample.get("depth_bytes") is None:
                    report["checks"]["missing_depth_samples"] += 1
                    current_episode.mark_issue("missing_depth")
                    append_issue("missing_depth", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)

                if isinstance(meta, dict):
                    missing_meta_keys = [key for key in REQUIRED_META_KEYS if key not in meta]
                    if missing_meta_keys:
                        report["checks"]["meta_missing_required_keys"] += 1
                        current_episode.mark_issue("missing_meta_keys")
                        append_issue(
                            "missing_meta_keys",
                            clip_id=clip_id,
                            sample_key=sample_key,
                            shard_name=shard_name,
                            detail=missing_meta_keys,
                        )

                    parsed_instruction = parse_instruction_metadata(meta)
                    instruction_num = int(parsed_instruction["instruction_num"])
                    instructions = list(parsed_instruction["instructions"])
                    if not current_episode.instruction_preview and instructions:
                        current_episode.instruction_preview = instructions[0][:160]
                    if allow_empty_instruction and (
                        parsed_instruction["missing_instruction"] or parsed_instruction["empty_instruction"]
                    ):
                        pass
                    elif parsed_instruction["missing_instruction"]:
                        report["checks"]["missing_instruction_frames"] += 1
                        current_episode.mark_issue("missing_instruction")
                        append_issue("missing_instruction", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
                    elif parsed_instruction["empty_instruction"]:
                        report["checks"]["empty_instruction_frames"] += 1
                        current_episode.mark_issue("empty_instruction")
                        append_issue("empty_instruction", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
                    elif parsed_instruction["instruction_num_mismatch"]:
                        report["checks"]["instruction_num_mismatch_frames"] += 1
                        current_episode.mark_issue("instruction_num_mismatch")
                        append_issue(
                            "instruction_num_mismatch",
                            clip_id=clip_id,
                            sample_key=sample_key,
                            shard_name=shard_name,
                            detail={
                                "instruction_num": instruction_num,
                                "non_empty_slots": len(instructions),
                                "instruction": list(parsed_instruction["effective_slots"]),
                            },
                        )
                else:
                    current_episode.mark_issue("invalid_meta")

                if lowdim is None:
                    continue
                if not np.isfinite(lowdim).all():
                    report["checks"]["nonfinite_lowdim_frames"] += 1
                    current_episode.mark_issue("nonfinite_lowdim")
                    append_issue("nonfinite_lowdim", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
                    continue

                state = lowdim[LOWDIM_STATE_SLICE]
                action = lowdim[LOWDIM_ACTION_SLICE]
                extrinsic = lowdim[LOWDIM_EXTRINSIC_SLICE]
                intrinsic = lowdim[LOWDIM_INTRINSIC_SLICE]
                if not np.isfinite(state).all():
                    report["checks"]["invalid_state_frames"] += 1
                    current_episode.mark_issue("invalid_state")
                    append_issue("invalid_state", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
                    continue
                if not np.isfinite(action).all():
                    report["checks"]["invalid_action_frames"] += 1
                    current_episode.mark_issue("invalid_action")
                    append_issue("invalid_action", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
                    continue
                if not np.isfinite(extrinsic).all():
                    report["checks"]["invalid_extrinsic_frames"] += 1
                    current_episode.mark_issue("invalid_extrinsic")
                    append_issue("invalid_extrinsic", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
                    continue
                if (not np.isfinite(intrinsic).all()) or float(intrinsic[0]) <= 0.0 or float(intrinsic[1]) <= 0.0:
                    report["checks"]["invalid_intrinsic_frames"] += 1
                    current_episode.mark_issue("invalid_intrinsic")
                    append_issue("invalid_intrinsic", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
                    continue

                numeric_sanity = validate_lowdim_numeric_sanity(lowdim)
                if not numeric_sanity["valid"]:
                    if numeric_sanity["invalid_rot6d"]:
                        report["checks"]["invalid_rot6d_frames"] += 1
                        current_episode.mark_issue("invalid_rot6d")
                        append_issue("invalid_rot6d", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
                    if numeric_sanity["invalid_extrinsic"]:
                        report["checks"]["invalid_extrinsic_frames"] += 1
                        current_episode.mark_issue("invalid_extrinsic")
                        append_issue("invalid_extrinsic", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
                    if numeric_sanity["invalid_intrinsic"]:
                        report["checks"]["invalid_intrinsic_frames"] += 1
                        current_episode.mark_issue("invalid_intrinsic")
                        append_issue("invalid_intrinsic", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
                    continue

                lowdim_stats.add(lowdim)
                report["summary"]["valid_lowdim_frames"] += 1
            else:
                continue
            break
        else:
            flush_episode()
    finally:
        lowdim_report = lowdim_stats.finalize()
        report["lowdim_stats"] = lowdim_report
        lowdim_stats.cleanup()

    if current_episode is not None:
        flush_episode()

    if render_dir and render_episodes > 0:
        selected_render_episodes = problematic_render_candidates[:render_episodes]
        if len(selected_render_episodes) < render_episodes:
            needed = render_episodes - len(selected_render_episodes)
            selected_render_episodes.extend(clean_render_candidates[:needed])
        for episode in selected_render_episodes:
            render_result = render_episode_video(
                episode,
                output_dir=render_dir,
                max_frames=render_max_frames,
                fps=render_fps,
            )
            if render_result is not None:
                report["renders"].append(render_result)

    report["hard_filter_reason_counts"] = dict(sorted(report["hard_filter_reason_counts"].items()))
    return report
