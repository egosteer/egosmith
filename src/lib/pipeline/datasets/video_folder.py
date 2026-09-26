"""Dataset adapter for video folders with extracted frame directories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lib.pipeline.datasets.base import AdapterPrepareResult, BaseDatasetAdapter, register_dataset_adapter
from lib.pipeline.datasets.descriptors import ClipDescriptor
from lib.pipeline.datasets.image_sequence import IMAGE_EXTENSIONS


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def _collect_videos(video_root: Path) -> list[Path]:
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(video_root.rglob(f"*{ext}"))
    return sorted(videos)


def _list_image_names(frame_dir: Path) -> list[str]:
    return sorted(path.name for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


# Written into the frame directory only after every frame of a video has been
# extracted; its absence means the directory is partial (or pre-dates markers).
EXTRACT_MARKER_NAME = ".extract_complete.json"


_CONTENT_BLOCK = 1 << 20


def video_content_digest(video_path: Path) -> str:
    """sha1 over the first and last 1 MiB of a video (the whole file when smaller) plus its size.

    Cheap content identity for a file that may be rewritten in place with the same path and size.
    """
    size = int(Path(video_path).stat().st_size)
    digest = hashlib.sha1(str(size).encode("ascii"))
    with open(video_path, "rb") as fh:
        if size <= 2 * _CONTENT_BLOCK:
            digest.update(fh.read())
        else:
            digest.update(fh.read(_CONTENT_BLOCK))
            fh.seek(size - _CONTENT_BLOCK)
            digest.update(fh.read(_CONTENT_BLOCK))
    return digest.hexdigest()


def _source_identity(video_path: Path, frame_ext: str) -> dict:
    stat = video_path.stat()
    return {
        "video": str(video_path.resolve()),
        "video_size": int(stat.st_size),
        # No mtime: clip.mode heuristic rewrites identical clip files on every run, and a
        # changed mtime alone must not force re-extraction or trip the stale-output guard.
        # The content digest catches a different video written to the same path with the same size.
        "video_content": video_content_digest(video_path),
        "frame_ext": frame_ext,
    }


def _read_extract_marker(frame_dir: Path) -> dict | None:
    marker_path = frame_dir / EXTRACT_MARKER_NAME
    if not marker_path.is_file():
        return None
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        return None
    if not isinstance(payload.get("frame_names"), list) or not payload["frame_names"]:
        return None
    return payload


def _write_extract_marker(frame_dir: Path, payload: dict) -> None:
    (frame_dir / EXTRACT_MARKER_NAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prior_marker_source(frame_dir: Path) -> dict | None:
    """Source identity recorded by an earlier (complete or invalidated) extraction."""
    marker_path = frame_dir / EXTRACT_MARKER_NAME
    if not marker_path.is_file():
        return None
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    source = payload.get("source") if isinstance(payload, dict) else None
    return source if isinstance(source, dict) else None


def _same_video(a: dict, b: dict) -> bool:
    """Same source video (path, size, content digest); the frame format is not part of it.
    A marker written before content digests were recorded is compared by (path, size) only."""
    keys = ("video", "video_size") + (("video_content",) if "video_content" in a and "video_content" in b else ())
    return all(a.get(key) == b.get(key) for key in keys)


def _extraction_complete(frame_dir: Path, identity: dict) -> bool:
    marker = _read_extract_marker(frame_dir)
    if marker is None:
        return False
    source = marker.get("source")
    if isinstance(source, dict) and "video_content" not in source and "video_content" in identity:
        # Marker from before content digests: it is trusted as before on (path, size, format), and the
        # current digest is recorded so any later same-size rewrite of the video is detected.
        if source != {k: v for k, v in identity.items() if k != "video_content"}:
            return False
        if not all((frame_dir / name).is_file() for name in marker["frame_names"]):
            return False
        _write_extract_marker(frame_dir, {**marker, "source": identity})
        return True
    if source != identity:
        return False
    return all((frame_dir / name).is_file() for name in marker["frame_names"])


@register_dataset_adapter
class VideoFolderDatasetAdapter(BaseDatasetAdapter):
    name = "video_folder"

    def prepare(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        runtimes_cfg: dict,
        context,
        run_logged=None,
    ) -> AdapterPrepareResult:
        del dataset_cfg, runtimes_cfg, context, run_logged
        if not bool(adapter_cfg.get("extract_frames", False)):
            return AdapterPrepareResult()

        video_root = Path(adapter_cfg.get("video_root") or paths_cfg.get("video_root", ""))
        if not video_root.is_dir():
            raise FileNotFoundError(f"video_root not found: {video_root}")
        frames_root = Path(adapter_cfg.get("frames_root") or paths_cfg.get("frames_root", video_root))
        frame_subdir = adapter_cfg.get("frame_subdir", "extracted_images")
        frame_ext = str(adapter_cfg.get("frame_ext", ".jpg")).lower()
        if not frame_ext.startswith("."):
            frame_ext = f".{frame_ext}"
        jpeg_quality = int(adapter_cfg.get("jpeg_quality", 95))
        resume = bool(adapter_cfg.get("resume", True))
        seq_folder_root = Path(adapter_cfg.get("seq_folder_root") or (frames_root / "outputs"))

        import cv2

        extracted = 0
        skipped = 0
        for video_path in _collect_videos(video_root):
            rel = video_path.relative_to(video_root)
            frame_dir = frames_root / rel.parent / video_path.stem / frame_subdir
            identity = _source_identity(video_path, frame_ext)
            if resume and frame_dir.is_dir() and _extraction_complete(frame_dir, identity):
                skipped += 1
                continue
            prior_source = _prior_marker_source(frame_dir)
            if prior_source is not None and not _same_video(prior_source, identity):
                # Same layout as build_descriptors: re-extracting here would pair the new
                # video's frames with the previous video's inference results.
                seq_folder = seq_folder_root / rel.parent / "__".join(rel.with_suffix("").parts)
                if seq_folder.is_dir() and any(seq_folder.iterdir()):
                    raise RuntimeError(
                        f"video_folder: {frame_dir} was extracted from a different video "
                        f"({prior_source.get('video')}) and {seq_folder} already holds its inference "
                        f"outputs; refusing to mix them with {video_path}. Use new frames_root / "
                        f"seq_folder_root locations, or manually clear {seq_folder} first."
                    )
            frame_dir.mkdir(parents=True, exist_ok=True)
            if (frame_dir / EXTRACT_MARKER_NAME).is_file():
                # Invalidate a stale marker before overwriting its frames; keep its source
                # so an interrupted re-extraction still knows which video the outputs are from.
                invalidated = {"complete": False}
                if prior_source is not None:
                    invalidated["source"] = prior_source
                _write_extract_marker(frame_dir, invalidated)
            frame_names = []
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video for frame extraction: {video_path}")
            write_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality] if frame_ext in {".jpg", ".jpeg"} else []
            idx = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    out_path = frame_dir / f"{idx:06d}{frame_ext}"
                    if not cv2.imwrite(str(out_path), frame, write_params):
                        raise RuntimeError(f"Failed to write frame: {out_path}")
                    frame_names.append(out_path.name)
                    idx += 1
            finally:
                cap.release()
            if idx <= 0:
                raise RuntimeError(f"No frames extracted from video: {video_path}")
            _write_extract_marker(frame_dir, {"complete": True, "source": identity, "frame_names": frame_names})
            extracted += 1
        return AdapterPrepareResult({"extracted_videos": extracted, "skipped_videos": skipped})

    def build_descriptors(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context=None,
        prepared=None,
    ):
        video_root = Path(adapter_cfg.get("video_root") or paths_cfg.get("video_root", ""))
        if not video_root.is_dir():
            raise FileNotFoundError(f"video_root not found: {video_root}")

        frames_root = Path(adapter_cfg.get("frames_root") or paths_cfg.get("frames_root", video_root))
        seq_folder_root = Path(adapter_cfg.get("seq_folder_root") or (frames_root / "outputs"))
        frame_subdir = adapter_cfg.get("frame_subdir", "extracted_images")

        descriptors = []
        for video_path in _collect_videos(video_root):
            relative_video = video_path.relative_to(video_root)
            relative_parent = relative_video.parent
            relative_stem = relative_video.with_suffix("")
            clip_id = "__".join(relative_stem.parts)
            clip_name = relative_stem.as_posix()
            frame_dir = frames_root / relative_parent / video_path.stem / frame_subdir
            if not frame_dir.is_dir():
                continue
            marker = _read_extract_marker(frame_dir)
            # A completed extraction lists exactly its own frames, so leftovers from an
            # earlier, longer/other-format extraction in the same directory never leak in.
            frame_names = list(marker["frame_names"]) if marker is not None else _list_image_names(frame_dir)
            if not frame_names:
                continue
            descriptors.append(
                ClipDescriptor.from_image_sequence(
                    clip_id=clip_id,
                    clip_name=clip_name,
                    root_dir=str(video_root.resolve()),
                    seq_folder=str((seq_folder_root / relative_parent / clip_id).resolve()),
                    frame_dir=str(frame_dir.resolve()),
                    frame_names=frame_names,
                    media_path=str(video_path.resolve()),
                    extra={"adapter": self.name},
                )
            )
        return descriptors
