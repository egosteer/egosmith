"""Helpers for building frame sources and reading raw frame bytes from descriptors."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

from lib.pipeline.datasets.descriptors import (
    ClipDescriptor,
    STORAGE_IMAGE_SEQUENCE,
    STORAGE_TAR_SHARD,
)
from lib.pipeline.io.video_index import load_clip_frame_offsets


def _normalized_frame_ext(ext: str) -> str:
    ext = str(ext or ".jpg").strip()
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext.lower()


def is_light_tar_descriptor(descriptor: ClipDescriptor) -> bool:
    return descriptor.is_lightweight_tar


def classify_descriptor_storage(descriptor: ClipDescriptor) -> str:
    if descriptor.is_lightweight_tar:
        return "light_tar"
    if descriptor.is_heavyweight_tar:
        return "heavy_tar"
    if descriptor.is_image_sequence:
        return "image_sequence"
    return descriptor.storage_kind


def validate_descriptor_for_frame_reads(descriptor: ClipDescriptor) -> None:
    frame_count = int(descriptor.frame_count)
    if frame_count <= 0:
        raise ValueError(f"Descriptor {descriptor.clip_id} has invalid frame_count={frame_count}")

    if descriptor.storage_kind == STORAGE_TAR_SHARD:
        if not descriptor.shard_path:
            raise ValueError(f"Descriptor {descriptor.clip_id} missing shard_path")
        if descriptor.frame_offsets is not None and len(descriptor.frame_offsets) != frame_count:
            raise ValueError(
                f"Descriptor {descriptor.clip_id} has {len(descriptor.frame_offsets)} frame_offsets "
                f"for frame_count={frame_count}"
            )
        if descriptor.frame_names and len(descriptor.frame_names) != frame_count:
            raise ValueError(
                f"Descriptor {descriptor.clip_id} has {len(descriptor.frame_names)} frame_names "
                f"for frame_count={frame_count}"
            )
        if descriptor.is_lightweight_tar:
            extra = descriptor.extra or {}
            for key in ("frame_ext", "frame_start_idx", "frame_index_width"):
                if key not in extra:
                    raise ValueError(f"Descriptor {descriptor.clip_id} missing light-tar field extra.{key}")
            frame_index_width = int(extra["frame_index_width"])
            if frame_index_width < 1:
                raise ValueError(
                    f"Descriptor {descriptor.clip_id} has invalid frame_index_width={frame_index_width}"
                )
            _normalized_frame_ext(str(extra["frame_ext"]))
            int(extra["frame_start_idx"])
        return

    if descriptor.storage_kind == STORAGE_IMAGE_SEQUENCE:
        if not descriptor.frame_dir:
            raise ValueError(f"Descriptor {descriptor.clip_id} missing frame_dir")
        if len(descriptor.frame_names) != frame_count:
            raise ValueError(
                f"Descriptor {descriptor.clip_id} has {len(descriptor.frame_names)} frame_names "
                f"for frame_count={frame_count}"
            )
        return

    raise ValueError(f"Unsupported descriptor storage_kind: {descriptor.storage_kind}")


def _infer_tar_frame_name(descriptor: ClipDescriptor, frame_idx: int) -> str:
    validate_descriptor_for_frame_reads(descriptor)
    frame_count = int(descriptor.frame_count)
    if frame_idx < 0 or frame_idx >= frame_count:
        raise IndexError(
            f"Frame index {frame_idx} out of range [0, {frame_count}) for {descriptor.clip_id}"
        )

    extra = descriptor.extra or {}
    frame_ext = _normalized_frame_ext(extra.get("frame_ext", ".jpg"))
    frame_start_idx = int(extra.get("frame_start_idx", 0))
    frame_index_width = int(extra.get("frame_index_width", 6))
    if frame_index_width < 1:
        raise ValueError(
            f"Descriptor {descriptor.clip_id} has invalid frame_index_width={frame_index_width}"
        )

    frame_number = frame_start_idx + frame_idx
    return f"{descriptor.clip_id}_f{frame_number:0{frame_index_width}d}{frame_ext}"


def _ensure_tar_frame_names(descriptor: ClipDescriptor) -> list[str]:
    validate_descriptor_for_frame_reads(descriptor)
    if descriptor.frame_names:
        return descriptor.frame_names
    if descriptor.storage_kind != STORAGE_TAR_SHARD:
        raise ValueError(f"Descriptor {descriptor.clip_id} is not a tar shard")

    frame_count = int(descriptor.frame_count)
    descriptor.frame_names = [_infer_tar_frame_name(descriptor, frame_idx) for frame_idx in range(frame_count)]
    return descriptor.frame_names


def _ensure_tar_frame_offsets(descriptor: ClipDescriptor) -> list[list[int]] | None:
    validate_descriptor_for_frame_reads(descriptor)
    if descriptor.frame_offsets is not None:
        return descriptor.frame_offsets
    if descriptor.storage_kind != STORAGE_TAR_SHARD:
        return None
    if not descriptor.root_dir:
        return None

    frame_offsets = load_clip_frame_offsets(descriptor.root_dir, descriptor.clip_id)
    if frame_offsets is None:
        return None
    descriptor.frame_offsets = frame_offsets
    return descriptor.frame_offsets


def _descriptor_member_name(descriptor: ClipDescriptor, frame_idx: int) -> str:
    validate_descriptor_for_frame_reads(descriptor)
    if descriptor.frame_names:
        return descriptor.frame_names[frame_idx]
    return _infer_tar_frame_name(descriptor, frame_idx)


def build_frame_source_from_descriptor(descriptor: ClipDescriptor):
    from lib.pipeline.io.frame_source import ImageFolderFrameSource, ShardVideoFrameSource

    validate_descriptor_for_frame_reads(descriptor)

    if descriptor.storage_kind == STORAGE_TAR_SHARD:
        _ensure_tar_frame_offsets(descriptor)
        frame_names = _ensure_tar_frame_names(descriptor)
        return ShardVideoFrameSource(
            descriptor.shard_path,
            frame_names,
            frame_offsets=descriptor.frame_offsets,
        )

    if descriptor.storage_kind == STORAGE_IMAGE_SEQUENCE:
        image_paths = [str((Path(descriptor.frame_dir) / frame_name).resolve()) for frame_name in descriptor.frame_names]
        return ImageFolderFrameSource(image_paths)

    raise ValueError(f"Unsupported descriptor storage_kind: {descriptor.storage_kind}")


def read_frame_bytes_from_descriptor(
    descriptor: ClipDescriptor,
    frame_idx: int,
    *,
    shard_fd_cache: dict | None = None,
    shard_tar_cache: dict | None = None,
) -> bytes:
    reader = build_frame_bytes_reader(
        descriptor,
        shard_fd_cache=shard_fd_cache,
        shard_tar_cache=shard_tar_cache,
    )
    return reader(frame_idx)


def build_frame_bytes_reader(
    descriptor: ClipDescriptor,
    *,
    shard_fd_cache: dict | None = None,
    shard_tar_cache: dict | None = None,
):
    validate_descriptor_for_frame_reads(descriptor)
    frame_count = int(descriptor.frame_count)

    if descriptor.storage_kind == STORAGE_TAR_SHARD:
        shard_path = descriptor.shard_path
        _ensure_tar_frame_offsets(descriptor)
        if descriptor.frame_offsets is not None:
            frame_offsets = descriptor.frame_offsets

            def read_from_offsets(frame_idx: int) -> bytes:
                if frame_idx < 0 or frame_idx >= frame_count:
                    raise IndexError(
                        f"Frame index {frame_idx} out of range [0, {frame_count}) for {descriptor.clip_id}"
                    )
                offset, size = frame_offsets[frame_idx]
                fd = None if shard_fd_cache is None else shard_fd_cache.get(shard_path)
                if fd is None:
                    fd = os.open(shard_path, os.O_RDONLY)
                    if shard_fd_cache is not None:
                        shard_fd_cache[shard_path] = fd
                payload = os.pread(fd, size, offset)
                if len(payload) != size:
                    raise RuntimeError(
                        f"Short read from shard {shard_path} frame {_descriptor_member_name(descriptor, frame_idx)}"
                    )
                return payload

            return read_from_offsets

        if descriptor.frame_names:
            frame_names = descriptor.frame_names
        else:
            frame_names = [_infer_tar_frame_name(descriptor, frame_idx) for frame_idx in range(frame_count)]

        def read_from_tar(frame_idx: int) -> bytes:
            if frame_idx < 0 or frame_idx >= frame_count:
                raise IndexError(
                    f"Frame index {frame_idx} out of range [0, {frame_count}) for {descriptor.clip_id}"
                )
            tar_reader = None if shard_tar_cache is None else shard_tar_cache.get(shard_path)
            if tar_reader is None:
                tar_reader = tarfile.open(shard_path, "r")
                if shard_tar_cache is not None:
                    shard_tar_cache[shard_path] = tar_reader
            member_name = frame_names[frame_idx]
            member = tar_reader.getmember(member_name)
            extracted = tar_reader.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Failed to extract {member_name} from {shard_path}")
            return extracted.read()

        return read_from_tar

    if descriptor.storage_kind == STORAGE_IMAGE_SEQUENCE:
        if descriptor.frame_dir is None:
            raise ValueError(f"Descriptor {descriptor.clip_id} missing frame_dir")
        frame_dir = Path(descriptor.frame_dir)
        frame_names = descriptor.frame_names

        def read_from_image_sequence(frame_idx: int) -> bytes:
            if frame_idx < 0 or frame_idx >= frame_count:
                raise IndexError(
                    f"Frame index {frame_idx} out of range [0, {frame_count}) for {descriptor.clip_id}"
                )
            return (frame_dir / frame_names[frame_idx]).read_bytes()

        return read_from_image_sequence

    raise ValueError(f"Unsupported descriptor storage_kind: {descriptor.storage_kind}")
