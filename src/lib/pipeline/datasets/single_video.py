"""Dataset adapter for the simplified one-video pipeline entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

from lib.pipeline.datasets.base import (
    AdapterPrepareResult,
    AdapterValidationResult,
    BaseDatasetAdapter,
    register_dataset_adapter,
)
from lib.pipeline.datasets.descriptors import ClipDescriptor


def _frame_ext(adapter_cfg: dict) -> str:
    """Return the normalized, validated frame extension (default ``.jpg``)."""
    ext = str(adapter_cfg.get("frame_ext") or ".jpg").strip().lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported single_video frame_ext: {ext}")
    return ext


def _metadata_path(paths_cfg: dict) -> Path:
    """Path of the cached single-video descriptor under ``paths.output_root``."""
    output_root = paths_cfg.get("output_root")
    if not output_root:
        raise KeyError(
            "single_video adapter requires 'paths.output_root' to be configured"
        )
    return Path(output_root) / "prepare" / "single_video_descriptor.json"


def _descriptor_from_payload(payload: dict) -> ClipDescriptor:
    """Build a ClipDescriptor from cached metadata, validating its shape."""
    descriptor_payload = payload.get("descriptor")
    if not isinstance(descriptor_payload, dict):
        raise ValueError("single_video metadata is missing descriptor payload")
    return ClipDescriptor.from_dict(descriptor_payload)


def _load_prepared_descriptor(paths_cfg: dict) -> ClipDescriptor | None:
    """Load the cached descriptor if a prior prepare run wrote one, else None."""
    metadata_path = _metadata_path(paths_cfg)
    if not metadata_path.is_file():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return _descriptor_from_payload(payload)


def _descriptor_frames_exist(descriptor: ClipDescriptor) -> bool:
    """True only if every frame named by the descriptor is present on disk."""
    frame_dir = Path(descriptor.frame_dir or "")
    if not frame_dir.is_dir():
        return False
    return bool(descriptor.frame_names) and all((frame_dir / name).is_file() for name in descriptor.frame_names)


@register_dataset_adapter
class SingleVideoDatasetAdapter(BaseDatasetAdapter):
    name = "single_video"

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

        video_path = Path(adapter_cfg["video"]).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"single_video input not found: {video_path}")

        existing = _load_prepared_descriptor(paths_cfg)
        if bool(adapter_cfg.get("resume", False)) and existing is not None and _descriptor_frames_exist(existing):
            return AdapterPrepareResult(
                {
                    "descriptor": existing,
                    "fps": existing.fps,
                    "width": existing.width,
                    "height": existing.height,
                    "frame_count": existing.frame_count,
                    "metadata_path": str(_metadata_path(paths_cfg)),
                    "resumed": True,
                }
            )

        import cv2

        clip_id = str(adapter_cfg.get("clip_id") or video_path.stem)
        clip_name = str(adapter_cfg.get("clip_name") or clip_id)
        ext = _frame_ext(adapter_cfg)
        frames_root = Path(adapter_cfg.get("frames_root") or paths_cfg["frames_root"])
        seq_folder_root = Path(adapter_cfg.get("seq_folder_root") or paths_cfg["seq_folder_root"])
        frame_dir = frames_root / clip_id
        seq_folder = seq_folder_root / clip_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        seq_folder.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open single_video input: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        jpeg_quality = int(adapter_cfg.get("jpeg_quality", 95))
        write_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality] if ext in {".jpg", ".jpeg"} else []

        frame_names = []
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if width <= 0 or height <= 0:
                    height, width = frame.shape[:2]
                frame_name = f"{frame_idx:06d}{ext}"
                out_path = frame_dir / frame_name
                if not cv2.imwrite(str(out_path), frame, write_params):
                    raise RuntimeError(f"Failed to write extracted frame: {out_path}")
                frame_names.append(frame_name)
                frame_idx += 1
        finally:
            cap.release()

        if not frame_names:
            raise RuntimeError(f"No frames extracted from single_video input: {video_path}")

        descriptor = ClipDescriptor.from_image_sequence(
            clip_id=clip_id,
            clip_name=clip_name,
            root_dir=str(Path(paths_cfg["output_root"]).resolve()),
            seq_folder=str(seq_folder.resolve()),
            frame_dir=str(frame_dir.resolve()),
            frame_names=frame_names,
            media_path=str(video_path),
            fps=fps if fps > 0.0 else None,
            width=width if width > 0 else None,
            height=height if height > 0 else None,
            frame_count_override=len(frame_names),
            extra={
                "adapter": self.name,
                "source_kind": "single_video",
                "native_fps": fps if fps > 0.0 else None,
                "media_path": str(video_path),
            },
        )

        metadata_path = _metadata_path(paths_cfg)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "descriptor": descriptor.to_dict(),
                    "video": str(video_path),
                    "fps": descriptor.fps,
                    "width": descriptor.width,
                    "height": descriptor.height,
                    "frame_count": descriptor.frame_count,
                    "media_path": str(video_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return AdapterPrepareResult(
            {
                "descriptor": descriptor,
                "fps": descriptor.fps,
                "width": descriptor.width,
                "height": descriptor.height,
                "frame_count": descriptor.frame_count,
                "metadata_path": str(metadata_path),
                "resumed": False,
            }
        )

    def build_descriptors(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context=None,
        prepared=None,
    ):
        del dataset_cfg, adapter_cfg, context
        if prepared is not None and isinstance(prepared.payload.get("descriptor"), ClipDescriptor):
            return [prepared.payload["descriptor"]]
        descriptor = _load_prepared_descriptor(paths_cfg)
        if descriptor is None:
            raise FileNotFoundError(f"single_video descriptor metadata not found: {_metadata_path(paths_cfg)}")
        return [descriptor]

    def validate_source(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context=None,
        prepared=None,
        manifest_records=None,
    ) -> AdapterValidationResult:
        del dataset_cfg, adapter_cfg, context, manifest_records
        try:
            descriptor = (
                prepared.payload.get("descriptor")
                if prepared is not None and isinstance(prepared.payload.get("descriptor"), ClipDescriptor)
                else _load_prepared_descriptor(paths_cfg)
            )
            if descriptor is None:
                raise FileNotFoundError(f"single_video descriptor metadata not found: {_metadata_path(paths_cfg)}")
            if not _descriptor_frames_exist(descriptor):
                raise FileNotFoundError(f"single_video extracted frames are incomplete: {descriptor.frame_dir}")
        except Exception as error:
            return AdapterValidationResult(
                ok=False,
                summary={"adapter": self.name, "error": str(error)},
            )
        return AdapterValidationResult(
            ok=True,
            summary={
                "adapter": self.name,
                "clip_id": descriptor.clip_id,
                "frame_count": descriptor.frame_count,
                "fps": descriptor.fps,
                "width": descriptor.width,
                "height": descriptor.height,
                "media_path": descriptor.media_path,
            },
        )
