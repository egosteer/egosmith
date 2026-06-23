"""Filesystem locations for per-stage outputs: seq folder, tracks dir, done markers.

Leaf module shared by stage_api and stage_validators (avoids an import cycle).
"""

from pathlib import Path

from lib.pipeline.datasets.descriptors import ClipDescriptor
from lib.pipeline.io.workspace import resolve_seq_folder


def get_seq_folder(video_path: str = None, descriptor: ClipDescriptor = None) -> Path:
    # Centralized in lib.pipeline.io.workspace: honors descriptor.seq_folder, then
    # the consolidated <output_root>/stage_outputs/<stem> default (overridable via
    # $HAWOR_OUTPUT_ROOT), with next-to-video only as an explicit legacy opt-in.
    return resolve_seq_folder(descriptor=descriptor, video_path=video_path)


def get_tracks_dir(seq_folder: Path, start_idx: int, end_idx: int) -> Path:
    return seq_folder / f"tracks_{start_idx}_{end_idx}"


def get_stage_done_marker(seq_folder: Path, stage: str) -> Path:
    return seq_folder / f".{stage}.done"
