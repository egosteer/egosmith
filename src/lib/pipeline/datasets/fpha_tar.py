"""Dataset adapter for FPHA sequence-per-tar shards."""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

from lib.pipeline.datasets.base import AdapterValidationResult, BaseDatasetAdapter, register_dataset_adapter
from lib.pipeline.datasets.descriptors import ClipDescriptor


_TAR_NAME_RE = re.compile(r"^((?:FPHA_)?(Subject_\d+)_(.+)_(\d+))\.tar$")
_IMAGE_MEMBER_RE = re.compile(r"^(FPHA_(Subject_\d+)_(.+)_(\d+))_f(\d+)\.image\.jpg$", re.IGNORECASE)
_DEPTH_MEMBER_RE = re.compile(r"^(FPHA_(Subject_\d+)_(.+)_(\d+))_f(\d+)\.depth\.npy$", re.IGNORECASE)


def _parse_tar_identity(tar_path: Path) -> tuple[str, str, str, int]:
    match = _TAR_NAME_RE.match(tar_path.name)
    if match is None:
        raise ValueError(
            "FPHA tar filename must match "
            f"`Subject_<id>_<action>_<trial>.tar` or "
            f"`FPHA_Subject_<id>_<action>_<trial>.tar`, got: {tar_path.name}"
        )
    tar_clip_id, subject, action, trial = match.groups()
    clip_id = tar_clip_id if tar_clip_id.startswith("FPHA_") else f"FPHA_{tar_clip_id}"
    return clip_id, subject, action, int(trial)


def _collect_tar_frames(tar_path: Path) -> tuple[list[str], list[list[int]], int]:
    image_entries: list[tuple[int, str, list[int]]] = []
    depth_frame_indices: set[int] = set()

    with tarfile.open(tar_path, "r") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue
            member_name = member.name
            member_base = Path(member_name).name

            image_match = _IMAGE_MEMBER_RE.match(member_base)
            if image_match is not None:
                frame_idx = int(image_match.group(5))
                image_entries.append((frame_idx, member_name, [int(member.offset_data), int(member.size)]))
                continue

            depth_match = _DEPTH_MEMBER_RE.match(member_base)
            if depth_match is not None:
                depth_frame_indices.add(int(depth_match.group(5)))

    if not image_entries:
        raise RuntimeError(f"No `*.image.jpg` frames found in {tar_path}")

    image_entries.sort(key=lambda item: item[0])
    frame_indices = [item[0] for item in image_entries]
    expected_indices = list(range(frame_indices[0], frame_indices[0] + len(frame_indices)))
    if frame_indices != expected_indices:
        raise RuntimeError(
            f"Non-contiguous RGB frame indices in {tar_path}: "
            f"start={frame_indices[0]} count={len(frame_indices)}"
        )

    if depth_frame_indices and depth_frame_indices != set(frame_indices):
        raise RuntimeError(
            f"RGB/depth frame mismatch in {tar_path}: "
            f"rgb={len(frame_indices)} depth={len(depth_frame_indices)}"
        )

    frame_names = [item[1] for item in image_entries]
    frame_offsets = [item[2] for item in image_entries]
    return frame_names, frame_offsets, len(depth_frame_indices)


@register_dataset_adapter
class FPHATarDatasetAdapter(BaseDatasetAdapter):
    name = "fpha_tar"

    def build_descriptors(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context=None,
        prepared=None,
    ):
        tar_root = Path(
            adapter_cfg.get("tar_root")
            or adapter_cfg.get("shard_dir")
            or paths_cfg.get("tar_root")
            or paths_cfg.get("shard_root", "")
        )
        if not tar_root.is_dir():
            raise FileNotFoundError(f"tar_root not found: {tar_root}")

        seq_folder_root = Path(adapter_cfg.get("seq_folder_root") or (tar_root / "outputs"))

        descriptors = []
        for tar_path in sorted(tar_root.glob("*.tar")):
            clip_id, subject, action, trial = _parse_tar_identity(tar_path)
            frame_names, frame_offsets, depth_count = _collect_tar_frames(tar_path)
            descriptors.append(
                ClipDescriptor.from_tar_shard(
                    clip_id=clip_id,
                    clip_name=clip_id,
                    root_dir=str(tar_root.resolve()),
                    seq_folder=str((seq_folder_root / clip_id).resolve()),
                    shard_path=str(tar_path.resolve()),
                    frame_names=frame_names,
                    frame_offsets=frame_offsets,
                    extra={
                        "adapter": self.name,
                        "subject": subject,
                        "action": action,
                        "trial": trial,
                        "depth_frame_count": depth_count,
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
        tar_root = Path(
            adapter_cfg.get("tar_root")
            or adapter_cfg.get("shard_dir")
            or paths_cfg.get("tar_root")
            or paths_cfg.get("shard_root", "")
        )
        if not tar_root.is_dir():
            return AdapterValidationResult(
                ok=False,
                summary={
                    "adapter": self.name,
                    "error": f"tar_root not found: {tar_root}",
                },
            )

        tar_files = sorted(tar_root.glob("*.tar"))
        invalid_names = [path.name for path in tar_files if _TAR_NAME_RE.match(path.name) is None][:16]
        seq_folder_root = Path(adapter_cfg.get("seq_folder_root") or (tar_root / "outputs"))
        return AdapterValidationResult(
            ok=bool(tar_files) and not invalid_names,
            summary={
                "adapter": self.name,
                "tar_root": str(tar_root.resolve()),
                "seq_folder_root": str(seq_folder_root.resolve()),
                "tar_count": len(tar_files),
                "invalid_name_preview": invalid_names,
            },
        )
