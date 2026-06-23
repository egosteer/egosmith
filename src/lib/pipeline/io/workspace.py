"""Single source of truth for HaWoR output / scratch path resolution.

Historically, several code paths each built the per-video "seq folder" (where a
stage writes its intermediates) on their own, and they all defaulted to
``parent(video)/stem(video)`` -- i.e. *next to the input video*. For a 1-minute
clip that can mean ~9 GB written beside the source, which fills disks and
scatters artifacts. The stage-3 (SLAM/Any4D) scratch root was resolved even more
ad hoc, deep inside the SLAM stage, so a misconfiguration only surfaced after
stages 1-2 had already run.

This module centralizes all of that so the batch path, the orchestrated dataset
pipeline, and the demo fork resolve paths identically:

* ``resolve_seq_folder`` -- where a video's stage intermediates live. The default
  is a consolidated ``<output_root>/stage_outputs/<stem>`` layout (the same one
  the orchestrated single-video pipeline already uses), redirectable in one place
  via ``--output_root`` / ``$HAWOR_OUTPUT_ROOT``. The legacy next-to-video
  behavior is still available, but only as an explicit opt-in.
* ``resolve_tmp_root`` -- the large-capacity scratch root for stage-3 frame
  materialization. Same precedence the SLAM stage always used, but callable at
  startup (see ``lib.pipeline.proc.preflight``) instead of mid-run.

The module is intentionally dependency-free (stdlib + pathlib only) so it can be
imported and unit-tested without torch / cv2 / GPU.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Environment overrides (documented in README / docs).
ENV_OUTPUT_ROOT = "HAWOR_OUTPUT_ROOT"
ENV_LEGACY_SEQ_FOLDER = "HAWOR_LEGACY_SEQ_FOLDER"
ENV_STAGE3_TMP_ROOT = "HAWOR_STAGE3_TMP_ROOT"
ENV_BATCH_TMPDIR = "HAWOR_BATCH_TMPDIR"

_OUTPUT_ROOT_SUFFIX = ".hawor_pipeline"
_STAGE_OUTPUTS_DIRNAME = "stage_outputs"
_STAGE3_FRAMES_DIRNAME = "hawor_stage3_frames"

_TRUTHY = {"1", "true", "yes", "y", "on"}


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def legacy_seq_folder_enabled(explicit: Optional[bool] = None) -> bool:
    """Whether to write intermediates next to the input video (legacy layout).

    ``explicit`` (a CLI flag / config value) wins when not ``None``; otherwise the
    ``HAWOR_LEGACY_SEQ_FOLDER`` environment variable decides. Default is off, so
    the consolidated layout is the new default everywhere.
    """
    if explicit is not None:
        return bool(explicit)
    return _env_truthy(ENV_LEGACY_SEQ_FOLDER)


def default_output_root(video_path) -> Path:
    """Default consolidated output root for a video.

    With ``$HAWOR_OUTPUT_ROOT`` set, all videos share that root
    (``$HAWOR_OUTPUT_ROOT/<stem>.hawor_pipeline``), which is how you move every
    intermediate off the video's disk in one place. Otherwise it falls back to a
    sibling ``<stem>.hawor_pipeline`` directory next to the video -- consolidated
    into a single, easy-to-delete folder rather than sprayed into the video's
    own directory.
    """
    video_path = Path(video_path)
    stem = video_path.stem
    env_root = os.environ.get(ENV_OUTPUT_ROOT, "").strip()
    if env_root:
        return (Path(env_root).expanduser() / f"{stem}{_OUTPUT_ROOT_SUFFIX}").resolve()
    return (video_path.parent / f"{stem}{_OUTPUT_ROOT_SUFFIX}").resolve()


def _descriptor_seq_folder(descriptor) -> Optional[str]:
    if descriptor is None:
        return None
    return getattr(descriptor, "seq_folder", None)


def resolve_seq_folder(
    *,
    descriptor=None,
    video_path=None,
    output_root=None,
    legacy_next_to_video: Optional[bool] = None,
) -> Path:
    """Resolve the per-video seq folder (stage intermediate root).

    Precedence (highest first):

    1. ``descriptor.seq_folder`` -- the orchestrated pipeline bakes the resolved
       path into the clip descriptor; always honored verbatim.
    2. legacy opt-in (``legacy_next_to_video`` flag or ``$HAWOR_LEGACY_SEQ_FOLDER``)
       -> ``parent(video)/stem`` (the old next-to-video layout).
    3. consolidated default -> ``<output_root>/stage_outputs/<stem>`` where
       ``output_root`` is the explicit argument or :func:`default_output_root`.
    """
    descriptor_folder = _descriptor_seq_folder(descriptor)
    if descriptor_folder:
        return Path(descriptor_folder)

    if video_path is None:
        raise ValueError(
            "resolve_seq_folder requires either a descriptor with seq_folder or a video_path"
        )
    video_path = Path(video_path)

    if legacy_seq_folder_enabled(legacy_next_to_video):
        return video_path.parent / video_path.stem

    root = Path(output_root).expanduser() if output_root else default_output_root(video_path)
    return root / _STAGE_OUTPUTS_DIRNAME / video_path.stem


def _tmp_root_from_args(args) -> Optional[str]:
    if args is None:
        return None
    return getattr(args, "stage3_tmp_root", None)


def resolve_tmp_root(args=None, *, required: bool = True) -> Optional[str]:
    """Resolve the stage-3 (Any4D SLAM) frame-materialization scratch root.

    Precedence: ``--stage3_tmp_root`` (on ``args``) > ``$HAWOR_STAGE3_TMP_ROOT`` >
    ``$HAWOR_BATCH_TMPDIR``. Frame materialization can write many GB, so this is
    never silently defaulted to a small disk such as ``/tmp``.

    Returns the resolved, writable absolute path. When unset:

    * ``required=True`` (default) raises ``ValueError`` with an actionable message
      -- this is what the SLAM stage and preflight use.
    * ``required=False`` returns ``None`` so callers can decide whether the run
      actually needs scratch space (e.g. SLAM stage not selected).
    """
    tmp_root = (
        _tmp_root_from_args(args)
        or os.environ.get(ENV_STAGE3_TMP_ROOT)
        or os.environ.get(ENV_BATCH_TMPDIR)
    )
    if not tmp_root:
        if not required:
            return None
        raise ValueError(
            "Stage3 (Any4D SLAM) temporary workspace root is not configured. "
            "Frame materialization can write many GB, so this is never defaulted "
            "and must point at a large-capacity disk. Set one of (highest priority "
            "first): --stage3_tmp_root, $HAWOR_STAGE3_TMP_ROOT, or $HAWOR_BATCH_TMPDIR, "
            "e.g. `export HAWOR_BATCH_TMPDIR=/efs-exp/<user>/tmp`."
        )
    tmp_root = os.path.abspath(os.path.expanduser(tmp_root))
    os.makedirs(tmp_root, exist_ok=True)
    if not os.access(tmp_root, os.W_OK | os.X_OK):
        raise PermissionError(f"Stage3 tmp root is not writable: {tmp_root}")
    return tmp_root


def stage3_frame_cache_dir(tmp_root: str, seq_folder: str, start_idx: int, end_idx: int) -> str:
    """Deterministic per-clip stage-3 frame cache directory under ``tmp_root``.

    The directory name embeds a short hash of the absolute seq folder so distinct
    videos that happen to share a stem never collide in the shared scratch root.
    """
    seq_hash = hashlib.sha1(os.path.abspath(seq_folder).encode("utf-8")).hexdigest()[:12]
    seq_name = Path(seq_folder).name
    return os.path.join(
        tmp_root, _STAGE3_FRAMES_DIRNAME, f"{seq_name}_{seq_hash}_{start_idx}_{end_idx}"
    )


@dataclass(frozen=True)
class WorkspaceLayout:
    """Resolved set of paths for one video's run.

    Convenience container so callers can resolve once and read named subdirs
    rather than re-deriving string paths. Mirrors the on-disk layout the stages
    already use under the seq folder.
    """

    seq_folder: Path
    tmp_root: Optional[Path] = None

    @classmethod
    def build(
        cls,
        *,
        descriptor=None,
        video_path=None,
        output_root=None,
        legacy_next_to_video: Optional[bool] = None,
        args=None,
        require_tmp_root: bool = False,
    ) -> "WorkspaceLayout":
        seq_folder = resolve_seq_folder(
            descriptor=descriptor,
            video_path=video_path,
            output_root=output_root,
            legacy_next_to_video=legacy_next_to_video,
        )
        tmp_root = resolve_tmp_root(args, required=require_tmp_root)
        return cls(seq_folder=seq_folder, tmp_root=Path(tmp_root) if tmp_root else None)

    @property
    def slam_dir(self) -> Path:
        return self.seq_folder / "SLAM"

    @property
    def cam_space_dir(self) -> Path:
        return self.seq_folder / "cam_space"

    @property
    def frames_dir(self) -> Path:
        return self.seq_folder / "extracted_images"

    def tracks_dir(self, start_idx: int, end_idx: int) -> Path:
        return self.seq_folder / f"tracks_{start_idx}_{end_idx}"

    @property
    def final_result_path(self) -> Path:
        return self.seq_folder / "world_space_res.pth"

    def stage_done_marker(self, stage: str) -> Path:
        return self.seq_folder / f".{stage}.done"
