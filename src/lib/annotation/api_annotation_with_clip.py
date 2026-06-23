#!/usr/bin/env python3
"""API-driven semantic clipping plus language annotation.

This module is the pipeline-facing version of the older HOT3D script. It sends
raw videos to a multimodal model once, asks for temporal action segments plus
language instructions, writes each returned segment as a clipped MP4, and emits
standard annotation sidecars whose names match the generic video_folder adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.annotation.api_annotation import (  # noqa: E402
    DEFAULT_ANNOTATION_SUFFIX,
    DEFAULT_MODEL,
    DEFAULT_TARGET_FPS,
    clean_json_text,
    load_api_keys,
)
from lib.pipeline.clips.annotation_protocol import HIERARCHY_KEYS, annotation_path  # noqa: E402


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
DEFAULT_PROMPT_FILE = (
    PROJECT_ROOT
    / "src"
    / "lib"
    / "annotation"
    / "prompts"
    / "with_clip"
    / "annotation_general_clip.txt"
)

_invalid_log_lock = threading.Lock()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use a multimodal API to segment raw videos, write clipped MP4s, and write annotation sidecars."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", default=None, help="Single raw input video.")
    source.add_argument("--video_root", default=None, help="Root containing raw videos.")
    parser.add_argument("--output_root", required=True, help="Directory for clipped MP4 files.")
    parser.add_argument("--annotation_root", required=True, help="Directory for segment annotation sidecars.")
    parser.add_argument("--annotation_suffix", default=DEFAULT_ANNOTATION_SUFFIX)
    parser.add_argument("--prompt_file", default=None, help=f"Defaults to {DEFAULT_PROMPT_FILE.as_posix()}.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DashScope multimodal model name.")
    parser.add_argument("--api_key", default=None, help="DashScope API key. Prefer DASHSCOPE_API_KEY.")
    parser.add_argument("--api_keys_file", default=None, help="Optional text file with one API key per line.")
    parser.add_argument("--target_fps", type=float, default=DEFAULT_TARGET_FPS, help="FPS hint sent to the API.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--min_segment_sec", type=float, default=0.2)
    parser.add_argument("--max_segment_sec", type=float, default=0.0, help="0 disables the maximum.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep_low_quality", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--report_out", default=None)
    return parser


def discover_videos(video_root: str | Path) -> list[Path]:
    root = Path(video_root).expanduser().resolve()
    if root.is_file() and root.suffix.lower() in VIDEO_EXTENSIONS:
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"video_root not found: {root}")
    videos: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(root.rglob(f"*{ext}"))
    return sorted(videos)


def load_prompt(prompt_file: str | Path | None) -> str:
    path = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
    return path.read_text(encoding="utf-8")


def extract_response_text(response) -> str | None:
    try:
        return response.output.choices[0].message.content[0]["text"]
    except Exception:
        return None


def call_qwen_safe(*, api_key: str, model: str, content_list: list[dict[str, Any]]) -> str | None:
    from dashscope import MultiModalConversation

    retry_count = 0
    while True:
        try:
            response = MultiModalConversation.call(
                api_key=api_key,
                model=model,
                messages=[{"role": "user", "content": content_list}],
                enable_thinking=False,
                temperature=1.0,
                top_p=0.9,
            )
            if response.status_code == 200:
                return extract_response_text(response)
            if response.status_code == 429:
                time.sleep(random.uniform(2, 4) * (1.1 ** min(retry_count, 10)))
                retry_count += 1
                continue
            return None
        except Exception:
            time.sleep(5)


def parse_segments(raw: str) -> list[dict[str, Any]]:
    payload = json.loads(clean_json_text(raw))
    if isinstance(payload, dict):
        if isinstance(payload.get("segments"), list):
            payload = payload["segments"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("API response must be a JSON array or an object with a segments array")
    return [item for item in payload if isinstance(item, dict)]


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in path.stem).strip("._") or "video"


def _clip_id_for_output(output_root: Path, clip_path: Path) -> str:
    relative_stem = clip_path.relative_to(output_root).with_suffix("")
    return "__".join(relative_stem.parts)


def _normalize_hierarchy(segment: dict[str, Any]) -> dict[str, str]:
    candidate = segment.get("language_instructions") or segment.get("hierarchy") or segment.get("global_analysis") or {}
    if not isinstance(candidate, dict):
        return {}
    hierarchy = {}
    for key in HIERARCHY_KEYS:
        value = candidate.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            hierarchy[key] = text
    return hierarchy


def _has_language_instruction(segment: dict[str, Any]) -> bool:
    return bool(_normalize_hierarchy(segment))


def _segment_bounds(segment: dict[str, Any]) -> tuple[float, float]:
    start = segment.get("start", segment.get("start_sec", segment.get("begin", 0.0)))
    end = segment.get("end", segment.get("end_sec", segment.get("finish", 0.0)))
    return float(start), float(end)


def _write_video_segment(source_video: Path, output_path: Path, *, start_sec: float, end_sec: float) -> dict[str, Any]:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for clipping: {source_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = max(start_frame + 1, int(round(end_sec * fps)))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create clipped video: {output_path}")
    written = 0
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _frame_idx in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()
    if written <= 0:
        output_path.unlink(missing_ok=True)
    return {
        "path": str(output_path),
        "frames": written,
        "fps": fps,
        "start_frame": start_frame,
        "end_frame": start_frame + written,
    }


def _annotation_payload(
    *,
    clip_id: str,
    clip_name: str,
    source_video: Path,
    segment: dict[str, Any],
    model: str,
    raw_text: str,
) -> dict[str, Any]:
    hierarchy = _normalize_hierarchy(segment)
    instruction = [hierarchy[key] for key in HIERARCHY_KEYS if hierarchy.get(key)]
    language = hierarchy.get("level5") or hierarchy.get("level2") or (instruction[-1] if instruction else None)
    is_good_quality = bool(segment.get("is_good_quality", True))
    return {
        "clip_id": clip_id,
        "clip_name": clip_name,
        "source_id": source_video.stem,
        "split": "train",
        "status": "Valid",
        "is_good_quality": is_good_quality,
        "instruction": instruction,
        "instruction_num": len(instruction),
        "language": language,
        "hierarchy": hierarchy,
        "global_analysis": hierarchy,
        "segment": {
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
        },
        "source_video": str(source_video),
        "annotation_model": model,
        "annotation_time": datetime.now().isoformat(),
        "raw_model_response": raw_text,
    }


def _write_invalid_log(annotation_root: Path, video_path: Path, reason: str) -> None:
    log_path = annotation_root / "_invalid" / "semantic_clip_errors.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _invalid_log_lock:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"video": str(video_path), "reason": reason, "time": datetime.now().isoformat()},
                    ensure_ascii=False,
                )
            )
            handle.write("\n")


def process_video(
    *,
    video_path: Path,
    source_root: Path | None,
    output_root: Path,
    annotation_root: Path,
    annotation_suffix: str,
    api_key: str,
    prompt: str,
    model: str,
    target_fps: float,
    min_segment_sec: float,
    max_segment_sec: float,
    keep_low_quality: bool,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    rel_parent = Path()
    if source_root is not None and video_path.is_relative_to(source_root):
        rel_parent = video_path.relative_to(source_root).parent
    video_url = f"file://{video_path.resolve()}"

    if dry_run:
        return {"video": str(video_path), "status": "dry_run", "segments": 0, "clips": []}

    response_text = call_qwen_safe(
        api_key=api_key,
        model=model,
        content_list=[
            {"video": video_url, "fps": float(target_fps)},
            {"text": prompt, "cache_control": {"type": "ephemeral"}},
        ],
    )
    if not response_text:
        _write_invalid_log(annotation_root, video_path, "empty_api_response")
        return {"video": str(video_path), "status": "failed", "error": "empty_api_response", "clips": []}

    try:
        segments = parse_segments(response_text)
    except Exception as error:
        _write_invalid_log(annotation_root, video_path, f"parse_error: {error}")
        return {"video": str(video_path), "status": "failed", "error": str(error), "clips": []}

    clips = []
    skipped_segments = []
    for index, segment in enumerate(segments):
        start_sec, end_sec = _segment_bounds(segment)
        duration = end_sec - start_sec
        if duration < min_segment_sec:
            skipped_segments.append({"index": index, "reason": "too_short", "start": start_sec, "end": end_sec})
            continue
        if max_segment_sec > 0 and duration > max_segment_sec:
            skipped_segments.append({"index": index, "reason": "too_long", "start": start_sec, "end": end_sec})
            continue
        if not keep_low_quality and segment.get("is_good_quality") is False:
            skipped_segments.append({"index": index, "reason": "low_quality", "start": start_sec, "end": end_sec})
            continue
        if not _has_language_instruction(segment):
            skipped_segments.append({"index": index, "reason": "empty_language_instruction", "start": start_sec, "end": end_sec})
            continue

        out_name = f"{_safe_stem(video_path)}_clip{index:03d}.mp4"
        out_path = output_root / rel_parent / out_name
        clip_id = _clip_id_for_output(output_root, out_path)
        ann_path = annotation_path(annotation_root, clip_id, annotation_suffix=annotation_suffix)
        if resume and out_path.exists() and ann_path.exists():
            clips.append({"clip_id": clip_id, "status": "skipped", "path": str(out_path)})
            continue

        write_meta = _write_video_segment(video_path, out_path, start_sec=start_sec, end_sec=end_sec)
        if write_meta["frames"] <= 0:
            continue
        payload = _annotation_payload(
            clip_id=clip_id,
            clip_name=out_path.relative_to(output_root).with_suffix("").as_posix(),
            source_video=video_path,
            segment={**segment, "start": start_sec, "end": end_sec},
            model=model,
            raw_text=response_text,
        )
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        ann_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        clips.append({"clip_id": clip_id, "status": "written", "annotation_path": str(ann_path), **write_meta})

    return {
        "video": str(video_path),
        "status": "processed",
        "segments": len(segments),
        "skipped_segments": skipped_segments,
        "clips": clips,
    }


def run_api_video_clipping(
    *,
    video_paths: Iterable[str | Path],
    output_root: str | Path,
    annotation_root: str | Path,
    prompt_file: str | Path | None = None,
    api_keys: list[str] | None = None,
    source_root: str | Path | None = None,
    annotation_suffix: str = DEFAULT_ANNOTATION_SUFFIX,
    model: str = DEFAULT_MODEL,
    target_fps: float = DEFAULT_TARGET_FPS,
    workers: int = 4,
    min_segment_sec: float = 0.2,
    max_segment_sec: float = 0.0,
    keep_low_quality: bool = False,
    resume: bool = True,
    dry_run: bool = False,
    report_out: str | Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    annotation_root = Path(annotation_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    annotation_root.mkdir(parents=True, exist_ok=True)
    source_root_path = Path(source_root).expanduser().resolve() if source_root else None
    videos = [Path(path).expanduser().resolve() for path in video_paths]
    prompt = load_prompt(prompt_file)
    if not api_keys and not dry_run:
        raise RuntimeError("No API key provided. Set DASHSCOPE_API_KEY, --api_key, or --api_keys_file.")
    api_keys = api_keys or ["dry-run"]

    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = []
        for index, video_path in enumerate(videos):
            futures.append(
                executor.submit(
                    process_video,
                    video_path=video_path,
                    source_root=source_root_path,
                    output_root=output_root,
                    annotation_root=annotation_root,
                    annotation_suffix=annotation_suffix,
                    api_key=api_keys[index % len(api_keys)],
                    prompt=prompt,
                    model=model,
                    target_fps=target_fps,
                    min_segment_sec=min_segment_sec,
                    max_segment_sec=max_segment_sec,
                    keep_low_quality=keep_low_quality,
                    resume=resume,
                    dry_run=dry_run,
                )
            )
        for future in tqdm(as_completed(futures), total=len(futures), desc="API clipping"):
            results.append(future.result())

    kept_clips = sum(len(item.get("clips") or []) for item in results)
    report = {
        "output_root": str(output_root),
        "annotation_root": str(annotation_root),
        "source_root": str(source_root_path) if source_root_path else None,
        "videos": results,
        "summary": {
            "source_videos": len(videos),
            "kept_clips": kept_clips,
            "failed_videos": sum(1 for item in results if item.get("status") == "failed"),
        },
    }
    report_path = Path(report_out) if report_out else annotation_root / "_api_clip_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    args = get_parser().parse_args(argv)
    if args.video:
        videos = [Path(args.video)]
        source_root = Path(args.video).parent
    else:
        videos = discover_videos(args.video_root)
        source_root = Path(args.video_root)
    if args.max_videos is not None:
        videos = videos[: max(0, int(args.max_videos))]

    api_keys = load_api_keys(args)
    report = run_api_video_clipping(
        video_paths=videos,
        output_root=args.output_root,
        annotation_root=args.annotation_root,
        prompt_file=args.prompt_file,
        api_keys=api_keys,
        source_root=source_root,
        annotation_suffix=args.annotation_suffix,
        model=args.model,
        target_fps=args.target_fps,
        workers=args.workers,
        min_segment_sec=args.min_segment_sec,
        max_segment_sec=args.max_segment_sec,
        keep_low_quality=args.keep_low_quality,
        resume=args.resume,
        dry_run=args.dry_run,
        report_out=args.report_out,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["summary"]["failed_videos"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
