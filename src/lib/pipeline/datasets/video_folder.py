"""Dataset adapter for video folders with extracted frame directories."""

from __future__ import annotations

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

        import cv2

        extracted = 0
        skipped = 0
        for video_path in _collect_videos(video_root):
            rel = video_path.relative_to(video_root)
            frame_dir = frames_root / rel.parent / video_path.stem / frame_subdir
            if resume and frame_dir.is_dir() and _list_image_names(frame_dir):
                skipped += 1
                continue
            frame_dir.mkdir(parents=True, exist_ok=True)
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
                    idx += 1
            finally:
                cap.release()
            if idx <= 0:
                raise RuntimeError(f"No frames extracted from video: {video_path}")
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
            frame_names = _list_image_names(frame_dir)
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
