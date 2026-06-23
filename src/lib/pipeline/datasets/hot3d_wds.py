"""Dataset adapter for HOT3D WebDataset frame shards."""

from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

from lib.pipeline.datasets.base import AdapterValidationResult, BaseDatasetAdapter, register_dataset_adapter
from lib.pipeline.datasets.descriptors import ClipDescriptor


_HOT3D_IMAGE_RE = re.compile(r"^(hot3d_ep(?P<episode>\d{6})_f(?P<frame>\d{5}))\.image\.jpg$")
_DEFAULT_REQUIRED_SUFFIXES = (".image.jpg", ".meta.json", ".bbox.npy", ".lowdim.npy", ".mano.npy")


@dataclass(frozen=True)
class _Hot3DFrame:
    frame_idx: int
    sample_key: str
    image_member: str
    offset_size: list[int]


@dataclass(frozen=True)
class _Hot3DSegment:
    episode_id: str
    episode_index: int
    shard_path: Path
    frames: list[_Hot3DFrame]

    @property
    def frame_start_idx(self) -> int:
        return int(self.frames[0].frame_idx)

    @property
    def frame_end_idx(self) -> int:
        return int(self.frames[-1].frame_idx)


def _list_shards(shard_dir: Path) -> list[Path]:
    return sorted(path for path in shard_dir.iterdir() if path.is_file() and path.suffix == ".tar")


def _split_contiguous(frames: list[_Hot3DFrame]) -> list[list[_Hot3DFrame]]:
    if not frames:
        return []
    chunks: list[list[_Hot3DFrame]] = []
    current = [frames[0]]
    for frame in frames[1:]:
        if frame.frame_idx == current[-1].frame_idx + 1:
            current.append(frame)
        else:
            chunks.append(current)
            current = [frame]
    chunks.append(current)
    return chunks


def _collect_shard_segments(
    shard_path: Path,
    *,
    required_suffixes: tuple[str, ...],
    split_discontinuous: bool,
) -> tuple[list[_Hot3DSegment], dict]:
    frames_by_episode: dict[str, list[_Hot3DFrame]] = {}
    sample_suffixes: dict[str, set[str]] = {}
    unsupported_images: list[str] = []

    with tarfile.open(shard_path, "r") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue
            member_base = Path(member.name).name
            suffix = None
            sample_key = None
            for candidate in required_suffixes:
                if member_base.endswith(candidate):
                    suffix = candidate
                    sample_key = member_base[: -len(candidate)]
                    break
            if suffix is not None and sample_key is not None:
                sample_suffixes.setdefault(sample_key, set()).add(suffix)

            image_match = _HOT3D_IMAGE_RE.match(member_base)
            if image_match is None:
                if member_base.endswith(".image.jpg"):
                    unsupported_images.append(member.name)
                continue

            episode_index = int(image_match.group("episode"))
            episode_id = f"hot3d_ep{episode_index:06d}"
            frame_idx = int(image_match.group("frame"))
            sample_key = image_match.group(1)
            frames_by_episode.setdefault(episode_id, []).append(
                _Hot3DFrame(
                    frame_idx=frame_idx,
                    sample_key=sample_key,
                    image_member=member.name,
                    offset_size=[int(member.offset_data), int(member.size)],
                )
            )

    missing_payloads = []
    duplicate_frames = []
    segments: list[_Hot3DSegment] = []
    for episode_id, frames in sorted(frames_by_episode.items()):
        frames = sorted(frames, key=lambda item: item.frame_idx)
        seen = set()
        for frame in frames:
            if frame.frame_idx in seen:
                duplicate_frames.append({"episode_id": episode_id, "frame_idx": frame.frame_idx})
            seen.add(frame.frame_idx)
            missing = sorted(set(required_suffixes) - sample_suffixes.get(frame.sample_key, set()))
            if missing:
                missing_payloads.append(
                    {
                        "sample_key": frame.sample_key,
                        "missing_suffixes": missing,
                        "shard": shard_path.name,
                    }
                )
        if duplicate_frames:
            continue
        if split_discontinuous:
            chunks = _split_contiguous(frames)
        else:
            expected = list(range(frames[0].frame_idx, frames[0].frame_idx + len(frames)))
            actual = [frame.frame_idx for frame in frames]
            if actual != expected:
                raise RuntimeError(
                    f"Non-contiguous HOT3D frames in {shard_path.name} for {episode_id}: "
                    f"start={frames[0].frame_idx} count={len(frames)}"
                )
            chunks = [frames]
        for chunk in chunks:
            segments.append(
                _Hot3DSegment(
                    episode_id=episode_id,
                    episode_index=int(episode_id.rsplit("ep", 1)[1]),
                    shard_path=shard_path,
                    frames=chunk,
                )
            )

    summary = {
        "shard": shard_path.name,
        "episodes": len(frames_by_episode),
        "segments": len(segments),
        "frames": sum(len(frames) for frames in frames_by_episode.values()),
        "missing_payload_examples": missing_payloads[:16],
        "missing_payload_count": len(missing_payloads),
        "duplicate_frame_examples": duplicate_frames[:16],
        "duplicate_frame_count": len(duplicate_frames),
        "unsupported_image_examples": unsupported_images[:16],
        "unsupported_image_count": len(unsupported_images),
    }
    if missing_payloads or duplicate_frames:
        raise RuntimeError(f"Invalid HOT3D shard {shard_path}: {summary}")
    return segments, summary


def _required_suffixes(adapter_cfg: dict) -> tuple[str, ...]:
    raw = adapter_cfg.get("required_suffixes")
    if raw is None:
        return _DEFAULT_REQUIRED_SUFFIXES
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",")]
    else:
        values = [str(item).strip() for item in raw]
    suffixes = tuple(item for item in values if item)
    return suffixes or _DEFAULT_REQUIRED_SUFFIXES


@register_dataset_adapter
class HOT3DWDSDatasetAdapter(BaseDatasetAdapter):
    name = "hot3d_wds"

    def build_descriptors(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context=None,
        prepared=None,
    ):
        shard_dir = Path(adapter_cfg.get("shard_dir") or paths_cfg.get("shard_dir") or paths_cfg.get("shard_root", ""))
        if not shard_dir.is_dir():
            raise FileNotFoundError(f"shard_dir not found: {shard_dir}")

        seq_folder_root = Path(adapter_cfg.get("seq_folder_root") or (shard_dir / "outputs"))
        required_suffixes = _required_suffixes(adapter_cfg)
        split_discontinuous = bool(adapter_cfg.get("split_discontinuous", True))
        shards = _list_shards(shard_dir)
        if not shards:
            raise FileNotFoundError(f"No HOT3D shard tar files found in {shard_dir}")

        segments: list[_Hot3DSegment] = []
        for shard_path in shards:
            shard_segments, _ = _collect_shard_segments(
                shard_path,
                required_suffixes=required_suffixes,
                split_discontinuous=split_discontinuous,
            )
            segments.extend(shard_segments)

        segments.sort(key=lambda item: (item.episode_index, item.frame_start_idx, item.shard_path.name))
        segments_by_episode: dict[str, list[_Hot3DSegment]] = {}
        for segment in segments:
            segments_by_episode.setdefault(segment.episode_id, []).append(segment)

        descriptors = []
        for episode_id, episode_segments in segments_by_episode.items():
            part_count = len(episode_segments)
            for part_idx, segment in enumerate(episode_segments):
                clip_id = episode_id if part_count == 1 else f"{episode_id}_part{part_idx:03d}"
                descriptors.append(
                    ClipDescriptor.from_tar_shard(
                        clip_id=clip_id,
                        clip_name=clip_id,
                        root_dir=str(shard_dir.resolve()),
                        seq_folder=str((seq_folder_root / clip_id).resolve()),
                        shard_path=str(segment.shard_path.resolve()),
                        frame_names=[frame.image_member for frame in segment.frames],
                        frame_offsets=[frame.offset_size for frame in segment.frames],
                        extra={
                            "adapter": self.name,
                            "native_feature_source": "wds_lowdim_mano_v1",
                            "lowdim_schema": "hot3d_wrist_world_v1",
                            "mano_schema": "hot3d_mano_2x55_v1",
                            "dataset_name": dataset_cfg.get("source_id") or "hot3d",
                            "original_episode_id": episode_id,
                            "episode_index": segment.episode_index,
                            "part_index": part_idx,
                            "part_count": part_count,
                            "frame_start_idx": segment.frame_start_idx,
                            "frame_end_idx": segment.frame_end_idx,
                            "source_shard": segment.shard_path.name,
                            "required_suffixes": list(required_suffixes),
                        },
                    )
                )
        return descriptors

    def resolve_annotation_context(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context,
        prepared=None,
    ) -> dict:
        shard_dir = Path(adapter_cfg.get("shard_dir") or paths_cfg.get("shard_dir") or paths_cfg.get("shard_root", ""))
        return {
            "hot3d_shard_dir": str(shard_dir),
        }

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
        shard_dir = Path(adapter_cfg.get("shard_dir") or paths_cfg.get("shard_dir") or paths_cfg.get("shard_root", ""))
        if not shard_dir.is_dir():
            return AdapterValidationResult(ok=False, summary={"adapter": self.name, "error": f"shard_dir not found: {shard_dir}"})

        shards = _list_shards(shard_dir)
        if not shards:
            return AdapterValidationResult(ok=False, summary={"adapter": self.name, "error": f"No tar shards found in {shard_dir}"})

        sample_shards = max(0, int(adapter_cfg.get("validate_sample_shards", 1)))
        required_suffixes = _required_suffixes(adapter_cfg)
        split_discontinuous = bool(adapter_cfg.get("split_discontinuous", True))
        sampled_summaries = []
        try:
            for shard_path in shards[:sample_shards]:
                _, shard_summary = _collect_shard_segments(
                    shard_path,
                    required_suffixes=required_suffixes,
                    split_discontinuous=split_discontinuous,
                )
                sampled_summaries.append(shard_summary)
        except Exception as error:
            return AdapterValidationResult(
                ok=False,
                summary={
                    "adapter": self.name,
                    "shard_dir": str(shard_dir.resolve()),
                    "shard_count": len(shards),
                    "error": str(error),
                },
            )

        return AdapterValidationResult(
            ok=True,
            summary={
                "adapter": self.name,
                "shard_dir": str(shard_dir.resolve()),
                "shard_count": len(shards),
                "required_suffixes": list(required_suffixes),
                "sampled_shards": sampled_summaries,
            },
        )
