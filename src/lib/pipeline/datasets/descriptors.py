"""Canonical clip descriptor types for dataset adapters and pipeline stages."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


STORAGE_TAR_SHARD = "tar_shard"
STORAGE_IMAGE_SEQUENCE = "image_sequence"


@dataclass
class ClipDescriptor:
    clip_id: str
    clip_name: str
    storage_kind: str
    root_dir: str
    seq_folder: str
    frame_names: list[str]
    frame_offsets: Optional[list[list[int]]] = None
    shard_path: Optional[str] = None
    frame_dir: Optional[str] = None
    media_path: Optional[str] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_count_override: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def video_key(self) -> str:
        return self.clip_id

    @property
    def video_name(self) -> str:
        return self.clip_name

    @property
    def factory_dir(self) -> str:
        return self.root_dir

    @property
    def frame_count(self) -> int:
        if self.frame_count_override is not None:
            return int(self.frame_count_override)
        return len(self.frame_names)

    @property
    def is_tar_shard(self) -> bool:
        return self.storage_kind == STORAGE_TAR_SHARD

    @property
    def is_image_sequence(self) -> bool:
        return self.storage_kind == STORAGE_IMAGE_SEQUENCE

    @property
    def is_lightweight_tar(self) -> bool:
        return self.is_tar_shard and not self.frame_names and self.frame_offsets is None

    @property
    def is_heavyweight_tar(self) -> bool:
        return self.is_tar_shard and (bool(self.frame_names) or self.frame_offsets is not None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "ClipDescriptor":
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipDescriptor":
        if "clip_id" in data:
            return cls(
                clip_id=data["clip_id"],
                clip_name=data.get("clip_name") or data["clip_id"],
                storage_kind=data.get("storage_kind") or STORAGE_TAR_SHARD,
                root_dir=data.get("root_dir") or data.get("factory_dir") or "",
                seq_folder=data["seq_folder"],
                frame_names=list(data.get("frame_names") or []),
                frame_offsets=data.get("frame_offsets"),
                shard_path=data.get("shard_path"),
                frame_dir=data.get("frame_dir"),
                media_path=data.get("media_path"),
                fps=data.get("fps"),
                width=data.get("width"),
                height=data.get("height"),
                frame_count_override=data.get("frame_count_override"),
                extra=data.get("extra") or {},
            )

        # Backward compatibility with legacy VideoDescriptor payloads.
        return cls(
            clip_id=data["video_key"],
            clip_name=data.get("video_name") or data["video_key"],
            storage_kind=STORAGE_TAR_SHARD if data.get("shard_path") else STORAGE_IMAGE_SEQUENCE,
            root_dir=data.get("factory_dir") or "",
            seq_folder=data["seq_folder"],
            frame_names=list(data.get("frame_names") or []),
            frame_offsets=data.get("frame_offsets"),
            shard_path=data.get("shard_path"),
            frame_dir=data.get("frame_dir"),
            media_path=data.get("media_path"),
            fps=data.get("fps"),
            width=data.get("width"),
            height=data.get("height"),
            frame_count_override=data.get("frame_count_override"),
            extra=data.get("extra") or {},
        )

    @classmethod
    def from_tar_shard(
        cls,
        *,
        clip_id: str,
        clip_name: str,
        root_dir: str,
        seq_folder: str,
        shard_path: str,
        frame_names: list[str],
        frame_offsets: Optional[list[list[int]]] = None,
        frame_count_override: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> "ClipDescriptor":
        return cls(
            clip_id=clip_id,
            clip_name=clip_name,
            storage_kind=STORAGE_TAR_SHARD,
            root_dir=root_dir,
            seq_folder=seq_folder,
            frame_names=list(frame_names),
            frame_offsets=frame_offsets,
            shard_path=shard_path,
            frame_count_override=frame_count_override,
            extra=extra or {},
        )

    @classmethod
    def from_image_sequence(
        cls,
        *,
        clip_id: str,
        clip_name: str,
        root_dir: str,
        seq_folder: str,
        frame_dir: str,
        frame_names: list[str],
        media_path: Optional[str] = None,
        fps: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        frame_count_override: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> "ClipDescriptor":
        return cls(
            clip_id=clip_id,
            clip_name=clip_name,
            storage_kind=STORAGE_IMAGE_SEQUENCE,
            root_dir=root_dir,
            seq_folder=seq_folder,
            frame_names=list(frame_names),
            frame_dir=frame_dir,
            media_path=media_path,
            fps=fps,
            width=width,
            height=height,
            frame_count_override=frame_count_override,
            extra=extra or {},
        )
