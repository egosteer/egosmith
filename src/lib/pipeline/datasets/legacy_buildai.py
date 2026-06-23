"""Dataset adapter for legacy BuildAI processed layouts."""

from __future__ import annotations

import re
from pathlib import Path

from lib.pipeline.datasets.base import AdapterValidationResult, BaseDatasetAdapter, register_dataset_adapter
from lib.pipeline.datasets.descriptors import ClipDescriptor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FACTORY_DIR_RE = re.compile(r"^factory_(\d+)$")


def _frame_sort_key(path: Path):
    stem = path.stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def _list_frame_names(frame_dir: Path) -> list[str]:
    files = [path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return [path.name for path in sorted(files, key=_frame_sort_key)]


def _matches_factory_range(factory_dir: Path, factory_range) -> bool:
    if factory_range is None:
        return True
    if isinstance(factory_range, str):
        start_str, end_str = factory_range.split("-", 1)
        start, end = int(start_str.strip()), int(end_str.strip())
    else:
        start, end = int(factory_range[0]), int(factory_range[1])

    match = FACTORY_DIR_RE.match(factory_dir.name)
    if match is None:
        return False
    factory_id = int(match.group(1))
    return start <= factory_id <= end


def _iter_legacy_seq_folders(processed_root: Path, factory_range=None):
    for factory_dir in sorted(processed_root.glob("factory_*")):
        if not factory_dir.is_dir():
            continue
        if not _matches_factory_range(factory_dir, factory_range):
            continue
        for worker_dir in sorted(factory_dir.glob("worker_*")):
            processed_dir = worker_dir / "processed"
            if not processed_dir.is_dir():
                continue
            for seq_folder in sorted(processed_dir.iterdir()):
                if seq_folder.is_dir():
                    yield factory_dir, worker_dir, processed_dir, seq_folder


@register_dataset_adapter
class LegacyBuildAIDatasetAdapter(BaseDatasetAdapter):
    name = "legacy_buildai"

    def build_descriptors(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context=None,
        prepared=None,
    ):
        processed_root = Path(
            adapter_cfg.get("processed_root")
            or paths_cfg.get("processed_root")
            or paths_cfg.get("legacy_buildai_processed_root", "")
        )
        if not processed_root.is_dir():
            raise FileNotFoundError(f"legacy BuildAI processed_root not found: {processed_root}")

        factory_range = adapter_cfg.get("factory_range")
        if factory_range is None and dataset_cfg.get("start_factory_id") is not None and dataset_cfg.get("end_factory_id") is not None:
            factory_range = (int(dataset_cfg["start_factory_id"]), int(dataset_cfg["end_factory_id"]))

        descriptors = []
        for factory_dir, worker_dir, processed_dir, seq_folder in _iter_legacy_seq_folders(processed_root, factory_range):
            frame_dir = seq_folder / "extracted_images"
            if not frame_dir.is_dir():
                continue
            frame_names = _list_frame_names(frame_dir)
            if not frame_names:
                continue

            clip_id = seq_folder.name
            media_path = processed_dir / f"{clip_id}.mp4"
            descriptors.append(
                ClipDescriptor.from_image_sequence(
                    clip_id=clip_id,
                    clip_name=clip_id,
                    root_dir=str(processed_dir.resolve()),
                    seq_folder=str(seq_folder.resolve()),
                    frame_dir=str(frame_dir.resolve()),
                    frame_names=frame_names,
                    media_path=str(media_path.resolve()) if media_path.is_file() else None,
                    fps=30.0,
                    extra={
                        "adapter": self.name,
                        "factory_dir": factory_dir.name,
                        "worker_dir": worker_dir.name,
                    },
                )
            )
        return descriptors

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
        processed_root = Path(
            adapter_cfg.get("processed_root")
            or paths_cfg.get("processed_root")
            or paths_cfg.get("legacy_buildai_processed_root", "")
        )
        if not processed_root.is_dir():
            return AdapterValidationResult(
                ok=False,
                summary={
                    "adapter": self.name,
                    "error": f"processed_root not found: {processed_root}",
                },
            )

        factory_range = adapter_cfg.get("factory_range")
        if factory_range is None and dataset_cfg.get("start_factory_id") is not None and dataset_cfg.get("end_factory_id") is not None:
            factory_range = (int(dataset_cfg["start_factory_id"]), int(dataset_cfg["end_factory_id"]))

        clip_count = 0
        clip_with_frames = 0
        clip_with_world = 0
        for _, _, _, seq_folder in _iter_legacy_seq_folders(processed_root, factory_range):
            clip_count += 1
            if (seq_folder / "extracted_images").is_dir():
                clip_with_frames += 1
            if (seq_folder / "world_space_res.pth").is_file():
                clip_with_world += 1

        return AdapterValidationResult(
            ok=clip_with_frames > 0,
            summary={
                "adapter": self.name,
                "processed_root": str(processed_root.resolve()),
                "factory_range": factory_range,
                "clip_count": clip_count,
                "clip_with_frames": clip_with_frames,
                "clip_with_world_res": clip_with_world,
            },
        )
