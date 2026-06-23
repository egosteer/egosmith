"""Clip shard manifest helpers for source-agnostic dataset pipelines."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List

if TYPE_CHECKING:
    from lib.pipeline.datasets.descriptors import ClipDescriptor


@dataclass(frozen=True)
class ClipManifestRecord:
    clip_id: str
    source_id: str
    split: str
    descriptor: "ClipDescriptor"
    group_id: str
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "clip_id": self.clip_id,
                "source_id": self.source_id,
                "split": self.split,
                "group_id": self.group_id,
                "descriptor": self.descriptor.to_dict(),
                "metadata": self.metadata,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> "ClipManifestRecord":
        from lib.pipeline.datasets.descriptors import ClipDescriptor

        payload = json.loads(raw)
        return cls(
            clip_id=payload["clip_id"],
            source_id=payload["source_id"],
            split=payload["split"],
            group_id=payload.get("group_id") or Path(payload["descriptor"].get("root_dir") or payload["descriptor"].get("factory_dir") or "").name,
            descriptor=ClipDescriptor.from_dict(payload["descriptor"]),
            metadata=payload.get("metadata") or {},
        )


def discover_shard_dirs(shard_root: str | Path, include_dirs: Iterable[str] | None = None) -> List[str]:
    """Discover shard directories containing one or more tar files."""
    root = Path(shard_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Shard root not found: {root}")

    include_set = {str(value).strip() for value in include_dirs or [] if str(value).strip()}
    shard_dirs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if include_set and child.name not in include_set:
            continue
        if any(name.endswith(".tar") for name in os.listdir(child)):
            shard_dirs.append(str(child.resolve()))
    return shard_dirs


def build_clip_manifest_records(
    shard_dirs: Iterable[str],
    *,
    source_id: str,
    split: str,
) -> List[ClipManifestRecord]:
    from lib.pipeline.io.video_index import collect_videos_from_factories

    descriptors = collect_videos_from_factories(list(shard_dirs))
    return build_manifest_records_from_descriptors(descriptors, source_id=source_id, split=split)


def build_manifest_records_from_descriptors(
    descriptors,
    *,
    source_id: str,
    split: str,
) -> List[ClipManifestRecord]:
    records = []
    for descriptor in descriptors:
        group_root = descriptor.root_dir or descriptor.seq_folder
        group_id = Path(group_root).name
        records.append(
            ClipManifestRecord(
                clip_id=descriptor.clip_id,
                source_id=source_id,
                split=split,
                descriptor=descriptor,
                group_id=group_id,
            )
        )
    return records


def remap_descriptor_seq_folders(descriptors, seq_folder_root: str | Path):
    """Override descriptor.seq_folder using a new root when rebuilding manifests."""
    root = Path(seq_folder_root).resolve()
    for descriptor in descriptors:
        group_name = Path(descriptor.root_dir).name if descriptor.root_dir else ""
        candidates = [
            root / group_name / "outputs" / descriptor.clip_id,
            root / group_name / descriptor.clip_id,
            root / descriptor.clip_id,
        ]
        for candidate in candidates:
            if candidate.exists():
                descriptor.seq_folder = str(candidate.resolve())
                break
        else:
            descriptor.seq_folder = str(candidates[0])


def write_clip_manifest(records: Iterable[ClipManifestRecord], output_path: str | Path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json())
            handle.write("\n")


def load_clip_manifest(path: str | Path) -> List[ClipManifestRecord]:
    manifest_path = Path(path)
    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(ClipManifestRecord.from_json(line))
    return records


def write_shard_dir_list(shard_dirs: Iterable[str], output_path: str | Path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for shard_dir in shard_dirs:
            handle.write(str(shard_dir))
            handle.write("\n")
