"""Helpers for rewriting existing WebDataset tar shards."""

import io
import json
import os
import re
import tarfile


SAMPLE_MEMBER_SUFFIXES = (
    (".image.jpg", "image_bytes"),
    (".lowdim.npy", "lowdim_bytes"),
    (".mano.npy", "mano_bytes"),
    (".depth.npy", "depth_bytes"),
    (".meta.json", "meta_bytes"),
)
REQUIRED_SAMPLE_FIELDS = ("image_bytes", "lowdim_bytes", "meta_bytes")
EPISODE_INDEX_PATTERN = re.compile(r"_ep(\d+)_f\d+$")


def split_sample_member_name(member_name):
    """Split a shard member name into sample key and known suffix."""
    for suffix, field_name in SAMPLE_MEMBER_SUFFIXES:
        if member_name.endswith(suffix):
            return member_name[: -len(suffix)], suffix, field_name
    return None, None, None


def parse_episode_index(sample_key):
    """Extract the episode index from a builder sample key."""
    match = EPISODE_INDEX_PATTERN.search(sample_key)
    if not match:
        raise ValueError(f"Failed to parse episode index from sample key: {sample_key}")
    return int(match.group(1))


def _new_sample_record(sample_key):
    return {
        "key": sample_key,
        "image_bytes": None,
        "lowdim_bytes": None,
        "mano_bytes": None,
        "depth_bytes": None,
        "meta_bytes": None,
    }


def iter_shard_samples(shard_path):
    """Yield grouped sample payloads from a shard in streaming order."""
    current_sample = None

    with tarfile.open(shard_path, "r|") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue

            sample_key, _, field_name = split_sample_member_name(member.name)
            if sample_key is None:
                raise ValueError(f"Unsupported shard member: {member.name}")

            member_file = tar_reader.extractfile(member)
            if member_file is None:
                raise ValueError(f"Failed to extract shard member: {member.name}")
            member_bytes = member_file.read()

            if current_sample is None:
                current_sample = _new_sample_record(sample_key)
            elif current_sample["key"] != sample_key:
                yield current_sample
                current_sample = _new_sample_record(sample_key)

            current_sample[field_name] = member_bytes

    if current_sample is not None:
        yield current_sample


def validate_sample_record(sample):
    """Validate that a grouped sample has all expected payloads."""
    missing = [field_name for field_name in REQUIRED_SAMPLE_FIELDS if sample.get(field_name) is None]
    if missing:
        raise ValueError(f"Incomplete sample {sample['key']}: missing {', '.join(missing)}")


def build_updated_meta(meta_bytes, instruction, language=None):
    """Update a sample meta payload with normalized instruction fields."""
    meta = json.loads(meta_bytes.decode("utf-8"))
    return build_updated_meta_from_meta(meta, instruction, language=language)


def build_updated_meta_from_meta(meta: dict, instruction, language=None):
    """Update an already-decoded sample meta payload with normalized instruction fields."""
    meta = dict(meta)
    meta["instruction"] = list(instruction)
    meta["instruction_num"] = len(instruction)
    if language is not None:
        meta["language"] = language
    return json.dumps(meta, ensure_ascii=False).encode("utf-8")


def _make_tar_info(name, payload):
    tar_info = tarfile.TarInfo(name=name)
    tar_info.size = len(payload)
    return tar_info


def write_sample_to_tar(tar_writer, sample_key, image_bytes, lowdim_bytes, meta_bytes, mano_bytes=None, depth_bytes=None):
    """Write one sample payload into a target tar."""
    tar_writer.addfile(_make_tar_info(f"{sample_key}.image.jpg", image_bytes), io.BytesIO(image_bytes))
    tar_writer.addfile(_make_tar_info(f"{sample_key}.lowdim.npy", lowdim_bytes), io.BytesIO(lowdim_bytes))
    if mano_bytes is not None:
        tar_writer.addfile(_make_tar_info(f"{sample_key}.mano.npy", mano_bytes), io.BytesIO(mano_bytes))
    if depth_bytes is not None:
        tar_writer.addfile(_make_tar_info(f"{sample_key}.depth.npy", depth_bytes), io.BytesIO(depth_bytes))
    tar_writer.addfile(_make_tar_info(f"{sample_key}.meta.json", meta_bytes), io.BytesIO(meta_bytes))


def iter_shard_paths(shard_dir):
    """Yield shard paths in the builder naming order."""
    for name in sorted(os.listdir(shard_dir)):
        if not name.endswith(".tar"):
            continue
        yield os.path.join(shard_dir, name)
