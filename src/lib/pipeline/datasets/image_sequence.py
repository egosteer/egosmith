"""Dataset adapter for clip directories that already contain frame images."""

from __future__ import annotations

from pathlib import Path

from lib.pipeline.datasets.base import BaseDatasetAdapter, register_dataset_adapter
from lib.pipeline.datasets.descriptors import ClipDescriptor


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _list_image_names(frame_dir: Path) -> list[str]:
    return sorted(path.name for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


@register_dataset_adapter
class ImageSequenceDatasetAdapter(BaseDatasetAdapter):
    name = "image_sequence"

    def build_descriptors(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context=None,
        prepared=None,
    ):
        sequence_root = Path(
            adapter_cfg.get("sequence_root")
            or paths_cfg.get("sequence_root")
            or paths_cfg.get("shard_root", "")
        )
        if not sequence_root.is_dir():
            raise FileNotFoundError(f"sequence_root not found: {sequence_root}")

        include_dirs = {str(item) for item in adapter_cfg.get("include_dirs", []) if str(item).strip()}
        seq_folder_root = Path(adapter_cfg.get("seq_folder_root") or (sequence_root / "outputs"))

        descriptors = []
        for child in sorted(sequence_root.iterdir()):
            if not child.is_dir():
                continue
            if child.resolve() == seq_folder_root.resolve():
                continue
            if include_dirs and child.name not in include_dirs:
                continue
            frame_names = _list_image_names(child)
            if not frame_names:
                continue
            clip_id = child.name
            descriptors.append(
                ClipDescriptor.from_image_sequence(
                    clip_id=clip_id,
                    clip_name=clip_id,
                    root_dir=str(sequence_root.resolve()),
                    seq_folder=str((seq_folder_root / clip_id).resolve()),
                    frame_dir=str(child.resolve()),
                    frame_names=frame_names,
                    extra={"adapter": self.name},
                )
            )
        return descriptors
