#!/usr/bin/env python3
"""Generic clip-level language annotation stage for the dataset pipeline.

This script is intentionally driven by the pipeline's prepared clip manifest,
not by a dataset-specific folder layout. Any adapter that can produce
``clip_manifest.jsonl`` can use it as ``annotation.command``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.clips.annotation_protocol import HIERARCHY_KEYS, annotation_path  # noqa: E402
from lib.pipeline.clips.clip_manifest import ClipManifestRecord, load_clip_manifest  # noqa: E402
from lib.pipeline.io.frame_sources import build_frame_source_from_descriptor  # noqa: E402


DEFAULT_MODEL = "qwen3.5-plus"
DEFAULT_ANNOTATION_SUFFIX = ".annotation.json"
DEFAULT_TARGET_FPS = 5.0
DEFAULT_PROMPT_FILE = (
    PROJECT_ROOT
    / "src"
    / "lib"
    / "annotation"
    / "prompts"
    / "without_clip"
    / "annotation_industrial_egocentric.txt"
)


_invalid_log_lock = threading.Lock()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate pipeline clips with language sidecars using a multimodal API."
    )
    parser.add_argument(
        "--prepared_state",
        "--manifest",
        dest="manifest",
        required=True,
        help="Pipeline prepared clip manifest JSONL, usually {prepared_state}.",
    )
    parser.add_argument(
        "--annotation_root",
        required=True,
        help="Directory where <clip_id><annotation_suffix> sidecars are written.",
    )
    parser.add_argument(
        "--annotation_suffix",
        default=DEFAULT_ANNOTATION_SUFFIX,
        help="Output sidecar suffix. Must match build.annotation_suffix.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DashScope multimodal model name.")
    parser.add_argument(
        "--api_key",
        default=None,
        help="DashScope API key. Prefer DASHSCOPE_API_KEY or --api_keys_file for production.",
    )
    parser.add_argument(
        "--api_keys_file",
        default=None,
        help="Optional text file with one API key per line.",
    )
    parser.add_argument(
        "--prompt_file",
        default=None,
        help=f"Optional prompt file. Defaults to {DEFAULT_PROMPT_FILE.as_posix()}.",
    )
    parser.add_argument("--target_fps", type=float, default=DEFAULT_TARGET_FPS, help="FPS hint sent to the API.")
    parser.add_argument(
        "--materialized_video_fps",
        type=float,
        default=None,
        help="FPS used when frame descriptors must be materialized to temporary MP4. Defaults to descriptor FPS or target_fps.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel API workers.")
    parser.add_argument("--max_clips", type=int, default=None, help="Optional clip limit for tests/debugging.")
    parser.add_argument(
        "--clip_ids",
        default=None,
        help="Optional comma-separated clip_id allowlist.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Uniformly sample at most this many frames when materializing. 0 keeps all frames.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--keep_invalid",
        action="store_true",
        help="Write Invalid sidecars instead of only logging invalid clips.",
    )
    parser.add_argument(
        "--tmp_dir",
        default=None,
        help="Optional directory for temporary materialized MP4 clips.",
    )
    parser.add_argument(
        "--report_out",
        default=None,
        help="Optional JSON report path. Defaults to <annotation_root>/_annotation_report.json.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve clips and output paths without calling the API.",
    )
    return parser


def load_api_keys(args) -> list[str]:
    keys: list[str] = []
    if args.api_key:
        keys.append(str(args.api_key).strip())
    if args.api_keys_file:
        key_path = Path(args.api_keys_file)
        keys.extend(line.strip() for line in key_path.read_text(encoding="utf-8").splitlines())
    env_key = os.environ.get("DASHSCOPE_API_KEY")
    if env_key:
        keys.append(env_key.strip())
    keys = [key for key in keys if key]
    if not keys and not args.dry_run:
        raise RuntimeError("No API key provided. Set DASHSCOPE_API_KEY, --api_key, or --api_keys_file.")
    return keys or ["dry-run"]


def load_prompt(prompt_file: str | None) -> str:
    path = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
    return path.read_text(encoding="utf-8")


def select_records(records: list[ClipManifestRecord], *, clip_ids: str | None, max_clips: int | None):
    if clip_ids:
        allowed = {item.strip() for item in clip_ids.split(",") if item.strip()}
        records = [record for record in records if record.clip_id in allowed]
    if max_clips is not None:
        records = records[: max(0, int(max_clips))]
    return records


def clean_json_text(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_model_json(raw: str) -> dict[str, Any]:
    cleaned = clean_json_text(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _normalize_hierarchy(model_payload: dict[str, Any]) -> dict[str, str]:
    candidates = [
        model_payload.get("language_instructions"),
        model_payload.get("hierarchy"),
        model_payload.get("global_analysis"),
    ]
    segments = model_payload.get("segments")
    if isinstance(segments, dict):
        candidates.extend(
            [
                segments.get("language_instructions"),
                segments.get("hierarchy"),
                segments.get("global_analysis"),
            ]
        )

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        hierarchy = {}
        for key in HIERARCHY_KEYS:
            value = candidate.get(key)
            if value is None:
                continue
            value = str(value).strip()
            if value:
                hierarchy[key] = value
        if hierarchy:
            return hierarchy
    return {}


def build_pipeline_annotation_payload(
    *,
    record: ClipManifestRecord,
    model_payload: dict[str, Any],
    raw_text: str,
    model: str,
) -> dict[str, Any]:
    status = str(model_payload.get("status") or "").strip()
    if not status and isinstance(model_payload.get("segments"), dict):
        status = str(model_payload["segments"].get("status") or "").strip()
    status = "Invalid" if status.lower() == "invalid" else "Valid"

    is_good_quality = model_payload.get("is_good_quality")
    if is_good_quality is None and isinstance(model_payload.get("segments"), dict):
        is_good_quality = model_payload["segments"].get("is_good_quality")
    is_good_quality = bool(is_good_quality) if status == "Valid" else False

    hierarchy = _normalize_hierarchy(model_payload) if status == "Valid" else {}
    instruction = [hierarchy[key] for key in HIERARCHY_KEYS if hierarchy.get(key)]
    language = hierarchy.get("level5") or hierarchy.get("level2") or (instruction[-1] if instruction else None)

    return {
        "clip_id": record.clip_id,
        "clip_name": record.descriptor.clip_name,
        "source_id": record.source_id,
        "split": record.split,
        "status": status,
        "is_good_quality": is_good_quality,
        "instruction": instruction,
        "instruction_num": len(instruction),
        "language": language,
        "hierarchy": hierarchy,
        "global_analysis": hierarchy,
        "annotation_model": model,
        "annotation_time": datetime.now().isoformat(),
        "raw_model_response": raw_text,
    }


def _uniform_indices(frame_count: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or frame_count <= max_frames:
        return list(range(frame_count))
    if max_frames == 1:
        return [0]
    return sorted({round(i * (frame_count - 1) / (max_frames - 1)) for i in range(max_frames)})


def materialize_record_video(record: ClipManifestRecord, *, tmp_dir: str | None, fps: float, max_frames: int):
    """Yield a local video path for an arbitrary descriptor.

    Existing media files are used directly. Tar/image-sequence descriptors are
    encoded into a temporary MP4 so the multimodal API receives a normal video
    clip regardless of source dataset.
    """
    media_path = record.descriptor.media_path
    if media_path and Path(media_path).is_file():
        yield str(Path(media_path).resolve())
        return

    import cv2

    frame_source = build_frame_source_from_descriptor(record.descriptor)
    frame_count = len(frame_source)
    indices = _uniform_indices(frame_count, max_frames)
    if not indices:
        raise RuntimeError(f"Clip {record.clip_id} has no frames to materialize")

    first = frame_source.get_frame(indices[0], rgb=False)
    height, width = first.shape[:2]
    temp_root = Path(tmp_dir) if tmp_dir else None
    if temp_root:
        temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{record.clip_id}_",
        suffix=".mp4",
        dir=str(temp_root) if temp_root else None,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)

    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps) if fps > 0 else DEFAULT_TARGET_FPS,
        (width, height),
    )
    if not writer.isOpened():
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to create temporary annotation video: {temp_path}")

    try:
        for idx in indices:
            frame = first if idx == indices[0] else frame_source.get_frame(idx, rgb=False)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
        writer.release()
        yield str(temp_path.resolve())
    finally:
        try:
            writer.release()
        except Exception:
            pass
        temp_path.unlink(missing_ok=True)


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
                temperature=1.2,
                top_p=0.95,
            )
            if response.status_code == 200:
                return extract_response_text(response)
            if response.status_code == 429:
                time.sleep(random.uniform(2, 5) * (1.2 ** min(retry_count, 10)))
                retry_count += 1
                continue
            return None
        except Exception:
            time.sleep(5)


def write_invalid_log(annotation_root: Path, record: ClipManifestRecord, reason: str) -> None:
    log_path = annotation_root / "_invalid" / "invalid_summary.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _invalid_log_lock:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "clip_id": record.clip_id,
                        "source_id": record.source_id,
                        "reason": reason,
                        "time": datetime.now().isoformat(),
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")


def annotate_one_record(
    *,
    record: ClipManifestRecord,
    api_key: str,
    prompt: str,
    args,
) -> dict[str, Any]:
    output_path = annotation_path(args.annotation_root, record.clip_id, annotation_suffix=args.annotation_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and output_path.exists():
        return {"clip_id": record.clip_id, "status": "skipped", "output_path": str(output_path)}

    if args.dry_run:
        return {"clip_id": record.clip_id, "status": "dry_run", "output_path": str(output_path)}

    materialized_fps = (
        float(args.materialized_video_fps)
        if args.materialized_video_fps is not None
        else float(record.descriptor.fps or args.target_fps or DEFAULT_TARGET_FPS)
    )
    with next_materialized_video(record, tmp_dir=args.tmp_dir, fps=materialized_fps, max_frames=args.max_frames) as video_path:
        video_url = f"file://{Path(video_path).resolve()}"
        response_text = call_qwen_safe(
            api_key=api_key,
            model=args.model,
            content_list=[
                {"video": video_url, "fps": float(args.target_fps)},
                {"text": prompt, "cache_control": {"type": "ephemeral"}},
            ],
        )

    if not response_text:
        write_invalid_log(Path(args.annotation_root), record, "empty_api_response")
        return {"clip_id": record.clip_id, "status": "failed", "error": "empty_api_response"}

    try:
        model_payload = parse_model_json(response_text)
        payload = build_pipeline_annotation_payload(
            record=record,
            model_payload=model_payload,
            raw_text=response_text,
            model=args.model,
        )
        if payload["status"] == "Invalid":
            write_invalid_log(Path(args.annotation_root), record, response_text)
            if not args.keep_invalid:
                return {"clip_id": record.clip_id, "status": "invalid", "output_path": str(output_path)}
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "clip_id": record.clip_id,
            "status": "written" if payload["status"] == "Valid" else "invalid_written",
            "output_path": str(output_path),
        }
    except Exception as error:
        write_invalid_log(Path(args.annotation_root), record, f"{type(error).__name__}: {error}")
        return {"clip_id": record.clip_id, "status": "failed", "error": str(error)}


class next_materialized_video:
    def __init__(self, record: ClipManifestRecord, *, tmp_dir: str | None, fps: float, max_frames: int):
        self._record = record
        self._tmp_dir = tmp_dir
        self._fps = fps
        self._max_frames = max_frames
        self._generator = None
        self.path = None

    def __enter__(self):
        self._generator = materialize_record_video(
            self._record,
            tmp_dir=self._tmp_dir,
            fps=self._fps,
            max_frames=self._max_frames,
        )
        self.path = next(self._generator)
        return self.path

    def __exit__(self, exc_type, exc, tb):
        if self._generator is not None:
            try:
                next(self._generator)
            except StopIteration:
                pass
        return False


def summarize_results(results: list[dict[str, Any]], *, manifest: str, annotation_root: str) -> dict[str, Any]:
    summary = {
        "manifest": str(Path(manifest).resolve()),
        "annotation_root": str(Path(annotation_root).resolve()),
        "total": len(results),
        "written": 0,
        "invalid": 0,
        "invalid_written": 0,
        "failed": 0,
        "skipped": 0,
        "dry_run": 0,
    }
    for result in results:
        status = str(result.get("status") or "")
        if status in summary:
            summary[status] += 1
    return {"summary": summary, "results": results}


def main(argv: list[str] | None = None) -> None:
    args = get_parser().parse_args(argv)
    annotation_root = Path(args.annotation_root)
    annotation_root.mkdir(parents=True, exist_ok=True)

    records = select_records(
        load_clip_manifest(args.manifest),
        clip_ids=args.clip_ids,
        max_clips=args.max_clips,
    )
    prompt = load_prompt(args.prompt_file)
    api_keys = load_api_keys(args)

    results = []
    max_workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, record in enumerate(records):
            futures.append(
                executor.submit(
                    annotate_one_record,
                    record=record,
                    api_key=api_keys[index % len(api_keys)],
                    prompt=prompt,
                    args=args,
                )
            )
        for future in tqdm(as_completed(futures), total=len(futures), desc="Annotating clips"):
            results.append(future.result())

    report = summarize_results(results, manifest=args.manifest, annotation_root=args.annotation_root)
    report_path = Path(args.report_out) if args.report_out else annotation_root / "_annotation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    if report["summary"]["failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
