"""Convert EgoSmith WebDataset shards into a LeRobot v3.0 dataset.

Target layout follows the EgoSteer open-data spec (LeRobot v3, validated against the official
``lerobot`` 0.6.1 loader)::

    <root>/
      meta/info.json
      meta/stats.json
      meta/tasks.parquet
      meta/episodes/chunk-XXX/file-XXX.parquet
      data/chunk-XXX/file-XXX.parquet
      videos/observation.images.head/chunk-XXX/file-XXX.mp4

Only the head RGB stream is exported; depth is dropped by design. ``observation.state`` and
``action`` use the EgoSteer 74-d layout: dims [0:26] (arm joints + hand motor values) do not exist
for human egocentric data and are padded with a constant, dims [26:74] (wrist pose xyz+rot6d and
five fingertips per hand) come straight from the 116-d ``lowdim`` vector. Everything is expressed
in the SLAM world frame by default (``hand_frame="world"``), with the per-frame world->camera matrix
kept as an extra feature; ``hand_frame="camera"`` re-expresses them in the head-camera frame of the
current observation instead (EgoSteer convention: world frame == head camera frame).

Two passes:

1. ``index_shards`` walks tar headers only (plus one ``meta.json`` per episode) and builds an
   offset index, so pass 2 can read any episode by seeking.
2. ``convert_file_group`` (one call per output file pair, parallelisable) streams the episode
   JPEGs through a single ``ffmpeg`` process (decode once, encode h264) and writes the matching
   parquet file, one row group per episode, plus per-episode stats.

The main entry point is :func:`convert_wds_to_lerobot`.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import logging
import math
import multiprocessing
import os
import pickle
import re
import shutil
import socket
import subprocess
import tarfile
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lib.pipeline.exporters.lerobot_source_map import (
    IMAGE_SIZE_KEY,
    NAN_CAMERA,
    SOURCE_CAMERA_KEY,
    SOURCE_FRAME_INDEX_KEY,
    SOURCE_FRAME_OBSERVED_KEY,
    SOURCE_MEDIA_FPS_KEY,
    SOURCE_MEDIA_FRAME_END_KEY,
    SOURCE_MEDIA_FRAME_START_KEY,
    SOURCE_MEDIA_KEY,
    SOURCE_MEDIA_UNDISTORT_KEY,
    SourceMap,
    SourceMapRow,
    check_relative_source_media,
)
from lib.pipeline.exporters.mano_codec import MANO_BETA_DIMS, MANO_PCA_DIMS, MANO_SAMPLE_SHAPE
from lib.pipeline.exporters.webdataset_rewriter import iter_shard_paths, split_sample_member_name
from lib.pipeline.quality.constants import (
    EXTRINSIC_SLICE,
    FRAME_INDEX_PATTERN,
    INTRINSIC_SLICE,
    LEFT_FINGERTIPS_SLICE,
    LEFT_HAND_TRANSLATION_SLICE,
    LEFT_ROOT_ROT6D_SLICE,
    LOWDIM_SIZE,
    RIGHT_FINGERTIPS_SLICE,
    RIGHT_HAND_TRANSLATION_SLICE,
    RIGHT_ROOT_ROT6D_SLICE,
)

logger = logging.getLogger(__name__)

CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNKS_SIZE = 1000
DEFAULT_DATA_FILES_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILES_SIZE_IN_MB = 200
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
TASKS_PATH = "meta/tasks.parquet"
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
WORK_DIR = "_wds_to_lerobot_work"

HEAD_VIDEO_KEY = "observation.images.head"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
PRESENCE_KEY = "observation.hand_presence"
EXTRINSIC_KEY = "observation.camera.head_world2cam"
MANO_KEY = "observation.mano"
INTRINSICS_EPISODE_KEY = "calibration/head_intrinsics"  # EgoSteer episodes use slash-separated calibration columns
HEAD_WORLD2CAM_EPISODE_KEY = "calibration/head_world2cam"

# EgoSteer 74-d layout (spec section 4).
EGOSTEER_STATE_DIM = 74
EGOSTEER_ROBOT_PAD_DIM = 26  # [0:7] left arm, [7:14] right arm, [14:20] left hand, [20:26] right hand
EGOSTEER_LEFT_WRIST = slice(26, 35)
EGOSTEER_RIGHT_WRIST = slice(35, 44)
EGOSTEER_LEFT_TIPS = slice(44, 59)
EGOSTEER_RIGHT_TIPS = slice(59, 74)
HAWOR_STATE_DIM = 48
STATE_LAYOUTS = ("egosteer74", "hawor48")
HAND_FRAMES = ("camera", "world")
TASK_SOURCES = ("language", "first_instruction", "dataset_name", "clip_id")
DEFAULT_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
MANO_FLAT_DIM = 2 * (MANO_PCA_DIMS + MANO_BETA_DIMS)

LOWDIM_ACTION_OFFSET = 48
SHARD_READ_ATTEMPTS = 3
SHARD_READ_BACKOFF_S = 3.0
INDEX_CACHE_DIRNAME = "shard_index"
TAR_WALK_CHUNK = 64 << 10  # one 64 KB positioned read per frame covers the small members' headers; big members are skipped
COALESCE_GAP = 64 << 10  # adjacent wanted members closer than this are fetched with one read
LABEL_COALESCE_GAP = 4 << 10  # without the image stream, do not let the merge swallow the JPEG between two label members
READ_THREADS = 8  # concurrent positioned reads per episode
READ_CHUNK = 8 << 20  # large blocks are fetched as pieces of this size, READ_THREADS at a time
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
AXES = ("x", "y", "z")
ROT6D_NAMES = ("r00", "r10", "r20", "r01", "r11", "r21")


# --------------------------------------------------------------------------------------
# Feature naming
# --------------------------------------------------------------------------------------


def hand_block_names() -> list[str]:
    """Names for the 48 hand dims: left wrist(9) right wrist(9) left tips(15) right tips(15)."""
    names: list[str] = []
    for side in ("left", "right"):
        names.extend(f"{side}_wrist.pos.{axis}" for axis in AXES)
        names.extend(f"{side}_wrist.rot6d.{name}" for name in ROT6D_NAMES)
    for side in ("left", "right"):
        for finger in FINGER_NAMES:
            names.extend(f"{side}_{finger}_tip.{axis}" for axis in AXES)
    return names


def egosteer74_names() -> list[str]:
    names = [f"arm1_joint_link{i + 1}" for i in range(7)]
    names += [f"arm2_joint_link{i + 1}" for i in range(7)]
    hand_dofs = ("1_1", "1_2", "2_1", "3_1", "4_1", "5_1")
    names += [f"hand1_joint_link_{d}" for d in hand_dofs]
    names += [f"hand2_joint_link_{d}" for d in hand_dofs]
    names += hand_block_names()
    if len(names) != EGOSTEER_STATE_DIM:
        raise AssertionError(f"expected {EGOSTEER_STATE_DIM} names, got {len(names)}")
    return names


def state_names(layout: str) -> list[str]:
    if layout == "egosteer74":
        return egosteer74_names()
    if layout == "hawor48":
        return hand_block_names()
    raise ValueError(f"unknown state layout {layout!r}; expected one of {STATE_LAYOUTS}")


def mano_names() -> list[str]:
    names = []
    for side in ("left", "right"):
        names.extend(f"{side}.pca{i:02d}" for i in range(MANO_PCA_DIMS))
        names.extend(f"{side}.beta{i:02d}" for i in range(MANO_BETA_DIMS))
    return names


def extrinsic_names() -> list[str]:
    return [f"w2c.r{r}c{c}" for r in range(4) for c in range(4)]


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class ConvertConfig:
    fps: int = 30
    robot_type: str = "egosmith_human"
    state_layout: str = "egosteer74"
    hand_frame: str = "world"  # world: SLAM world frame + per-row w2c; camera: re-expressed in each frame's head camera
    pad_value: float = 0.0
    task_source: str = "dataset_name"  # tasks[0] is what downstream tools use as the dataset name; the sentences live in `instructions`/`language`
    task_name: str | None = None  # explicit task / dataset name for every episode (overrides task_source)
    split_order: tuple[str, ...] = ("train", "val")
    include_mano: bool = True
    mano_policy: str = "auto"  # auto: no shard has mano.npy -> no MANO feature; some episodes lack it -> skip those; require: fail fast
    include_extrinsic: bool = True
    include_presence: bool = True
    include_video: bool = True  # False = labels-only dataset (no observation.images.head, no ffmpeg)
    descriptor_manifest: str | None = None  # frozen clip manifest -> meta/source_frames.parquet (rehydration index)
    source_map: str | None = None  # parquet from a locator (see lerobot_source_map.py) -> per-frame source_frame_index
    source_map_policy: str = "drop"  # episodes without a usable map row: drop (default) | keep (empty media, index -1)
    ffmpeg: str = "ffmpeg"
    ffmpeg_threads: int = 4
    vcodec: str = "libx264"
    crf: int = 18
    gop: int = 15
    preset: str = "medium"
    pix_fmt: str = "yuv420p"
    chunks_size: int = DEFAULT_CHUNKS_SIZE
    data_files_size_in_mb: int = DEFAULT_DATA_FILES_SIZE_IN_MB
    video_files_size_in_mb: int = DEFAULT_VIDEO_FILES_SIZE_IN_MB
    video_size_ratio: float = 1.0  # expected h264 bytes per source-JPEG byte (planning only)
    max_episodes: int | None = None
    workers: int = 1
    resume: bool = False
    helper: bool = False  # second machine: convert unclaimed file groups only, never write meta (see _try_claim)
    claim_ttl_s: float = 3 * 3600.0  # a claim older than this belongs to a dead worker and may be taken over

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.workers < 0:
            raise ValueError("workers must be >= 0 (0 = auto: cpu_count // ffmpeg_threads)")
        if self.mano_policy not in ("auto", "require"):
            raise ValueError("mano_policy must be 'auto' or 'require'")
        if self.source_map_policy not in ("drop", "keep"):
            raise ValueError("source_map_policy must be 'drop' or 'keep'")
        if self.state_layout not in STATE_LAYOUTS:
            raise ValueError(f"state_layout must be one of {STATE_LAYOUTS}")
        if self.hand_frame not in HAND_FRAMES:
            raise ValueError(f"hand_frame must be one of {HAND_FRAMES}")
        if self.hand_frame == "world" and not self.include_extrinsic:
            raise ValueError(
                "hand_frame='world' with include_extrinsic=False (--no_extrinsic) exports world-frame hands without the "
                "per-frame world->camera matrix, so they cannot be related to the camera; keep the extrinsic or use hand_frame='camera'"
            )
        if self.task_source not in TASK_SOURCES:
            raise ValueError(f"task_source must be one of {TASK_SOURCES}")
        if self.chunks_size <= 0 or self.data_files_size_in_mb <= 0 or self.video_files_size_in_mb <= 0:
            raise ValueError("chunks_size / *_files_size_in_mb must be positive")

    @property
    def state_dim(self) -> int:
        return EGOSTEER_STATE_DIM if self.state_layout == "egosteer74" else HAWOR_STATE_DIM

    @property
    def effective_workers(self) -> int:
        """0 = auto: one file-pair worker per ffmpeg_threads cores (each worker runs its own ffmpeg)."""
        if self.workers > 0:
            return self.workers
        cores = available_cpus()
        per_worker = max(1, self.ffmpeg_threads) if self.include_video else 1
        return max(1, cores // per_worker)


# --------------------------------------------------------------------------------------
# Pass 1: shard index
# --------------------------------------------------------------------------------------

MEMBER_FIELDS = ("image_bytes", "lowdim_bytes", "mano_bytes", "meta_bytes")
_EPISODE_KEY_PATTERN = re.compile(r"^(?P<episode>.+)_f(?P<frame>\d+)$")



# --------------------------------------------------------------------------------------
# Worker processes
#
# Every pool here uses the ``spawn`` start method. Forking the converter's parent (which holds the
# shard index, the plan and whatever the batch driver accumulated) duplicates that heap into every
# worker as soon as the child's garbage collector touches it, and a 70-worker pool then trips the
# memory cgroup. Spawned workers start from a clean interpreter; per-job payloads stay small
# because the shard table and the task vocabulary travel once through the pool initializer.
# --------------------------------------------------------------------------------------

POOL_START_METHOD = "spawn"
HEARTBEAT_SECONDS = 300.0
INFLIGHT_PER_WORKER = 2

_WORKER_SHARED: dict = {}
_WORKER_SOURCE_MAP: dict = {}  # path -> SourceMap, loaded once per worker process


_REPEAT_SUFFIX_PATTERN = re.compile(r"__rep\d+$")


def source_map_row(smap: SourceMap, record) -> SourceMapRow | None:
    """Map row of an episode. Repeated episodes (``<clip>__rep<k>`` keys, see manifest_sample_key) carry
    the same source frames as their clip, so they fall back to the original clip key."""
    row = smap.get(record.key)
    if row is None and _REPEAT_SUFFIX_PATTERN.search(record.key):
        row = smap.get(_REPEAT_SUFFIX_PATTERN.sub("", record.key))
    return row


def _worker_source_map(path: str | None) -> SourceMap | None:
    if not path:
        return None
    if path not in _WORKER_SOURCE_MAP:
        _WORKER_SOURCE_MAP[path] = SourceMap.load(path)
    return _WORKER_SOURCE_MAP[path]


def _pool_context():
    return multiprocessing.get_context(POOL_START_METHOD)


def cgroup_cpu_quota() -> float | None:
    """CPU quota of this container in cores (cgroup v2 ``cpu.max`` or v1 ``cfs_quota_us``); None when unlimited.

    A container can see every host core through the affinity mask and still be throttled to a fraction of
    them. Starting one worker per visible core then only produces throttling: wall-clock time explodes
    while the host looks idle.
    """
    try:
        text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if text and text[0] != "max":
            return float(text[0]) / float(text[1])
    except (OSError, ValueError, IndexError):
        pass
    for base in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
        try:
            quota = float(Path(base, "cpu.cfs_quota_us").read_text())
            period = float(Path(base, "cpu.cfs_period_us").read_text())
            if quota > 0 and period > 0:
                return quota / period
        except (OSError, ValueError):
            continue
    return None


def available_cpus() -> int:
    """CPUs this process may actually use: the affinity mask, capped by the container's cgroup CPU quota."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    quota = cgroup_cpu_quota()
    if quota is not None:
        n = min(n, int(math.floor(quota)) or 1)
    return max(1, n)


def rss_gb() -> float:
    """Resident set size of this process in GB (Linux /proc; falls back to ru_maxrss)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        return float("nan")


def _bind_to_parent_death() -> None:
    """Linux: get SIGTERM when the parent dies so a killed driver never leaves idle workers behind."""
    if sys.platform != "linux":
        return
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, int(signal.SIGTERM))  # PR_SET_PDEATHSIG
    except Exception:
        pass


def _worker_init(shared: dict | None = None) -> None:
    _bind_to_parent_death()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _WORKER_SHARED.clear()
    if shared:
        _WORKER_SHARED.update(shared)


@dataclass
class EpisodeRecord:
    """Everything pass 2 needs to read one episode straight from the shards by seeking."""

    key: str
    shard_paths: list[str] | None  # global shard table (shared across records)
    frame_ids: np.ndarray | None  # (T,) int64, sorted; None for a slim (driver-side) record
    shard_idx: np.ndarray | None  # (T,) int32 into shard_paths
    offsets: dict[str, np.ndarray] | None  # field -> (T,) int64 offset_data; -1 when missing
    sizes: dict[str, np.ndarray] | None  # field -> (T,) int64
    meta: dict
    image_bytes_total: int
    height: int
    width: int
    split: str = "train"
    source_episode_index: int = -1
    episode_index: int = -1  # assigned after ordering
    # slim summary (what the driver keeps for a million-episode dataset; workers re-read the arrays
    # from the per-shard index cache, see ``hydrate_record``)
    frame_count: int = -1
    frame_first: int = -1
    frames_contiguous: bool = True
    mano_complete: bool = True
    parts: tuple[int, ...] = ()  # shard indices that hold this episode, in order

    @property
    def hydrated(self) -> bool:
        return self.offsets is not None

    @property
    def length(self) -> int:
        return int(self.frame_ids.shape[0]) if self.frame_ids is not None else int(self.frame_count)

    @property
    def contiguous(self) -> bool:
        if self.frame_ids is None:
            return bool(self.frames_contiguous)
        return self.length <= 1 or bool(np.all(np.diff(self.frame_ids) == 1))

    @property
    def first_frame_id(self) -> int:
        return int(self.frame_ids[0]) if self.frame_ids is not None else int(self.frame_first)

    @property
    def mano_missing(self) -> bool:
        if self.offsets is None:
            return not self.mano_complete
        return bool((self.offsets["mano_bytes"] < 0).any())


def parse_episode_key(sample_key: str) -> tuple[str, int]:
    match = _EPISODE_KEY_PATTERN.match(sample_key)
    if not match:
        raise ValueError(f"cannot parse frame index from sample key {sample_key!r}")
    return match.group("episode"), int(match.group("frame"))


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Return (height, width) from JPEG SOF markers without decoding the image."""
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise ValueError("not a JPEG stream")
    pos = 2
    n = len(data)
    while pos + 4 <= n:
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        if marker == 0xFF:
            pos += 1
            continue
        seg_len = (data[pos + 2] << 8) | data[pos + 3]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if pos + 9 > n:
                break
            height = (data[pos + 5] << 8) | data[pos + 6]
            width = (data[pos + 7] << 8) | data[pos + 8]
            return int(height), int(width)
        if marker == 0xD9 or marker == 0xDA:
            break
        pos += 2 + seg_len
    raise ValueError("JPEG SOF marker not found")


class EpisodeReader:
    """Fetches the wanted members of one episode with as few positioned reads as possible.

    The wanted (offset, size) ranges are sorted per shard and neighbours closer than
    ``COALESCE_GAP`` are merged into one read, so a plain image/lowdim/meta sequence costs one read
    per run of frames while a large unwanted member in between (a depth map) is never read. Kernel
    readahead is disabled on the shard descriptors for the same reason.
    """

    def __init__(self, record: EpisodeRecord, handles: dict[int, object], *, fields: tuple[str, ...] = MEMBER_FIELDS, gap: int = COALESCE_GAP, max_block: int = 512 << 20, read_threads: int = READ_THREADS):
        self.record = record
        self.handles = handles
        self.blocks: dict[int, list[tuple[int, int, bytes]]] = {}  # shard -> sorted (start, end, data)
        for shard_idx in np.unique(record.shard_idx).tolist():
            rows = np.nonzero(record.shard_idx == shard_idx)[0]
            ranges = []
            for name in fields:
                offs = record.offsets[name][rows]
                sizes = record.sizes[name][rows]
                valid = offs >= 0
                ranges.extend(zip(offs[valid].tolist(), (offs[valid] + sizes[valid]).tolist()))
            if not ranges:
                continue
            ranges.sort()
            merged: list[list[int]] = []
            for start, end in ranges:
                if merged and start - merged[-1][1] <= gap and end - merged[-1][0] <= max_block:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            fd = self._handle(int(shard_idx)).fileno()
            # One stream to the network mount delivers ~50-100 MB/s however large the request, while the
            # mount as a whole has far more to give, so every block is cut into READ_CHUNK pieces and the
            # pieces (or, for label-only reads, the many tiny ranges) are fetched concurrently.
            # os.pread releases the GIL and is safe to call concurrently on one descriptor.
            pieces: list[tuple[int, int, int]] = []  # (block index, start, end)
            for b, (start, end) in enumerate(merged):
                pos = start
                while pos < end:
                    nxt = min(end, pos + READ_CHUNK)
                    pieces.append((b, pos, nxt))
                    pos = nxt
            if read_threads > 1 and len(pieces) > 1:
                with ThreadPoolExecutor(max_workers=min(read_threads, len(pieces))) as pool:
                    parts = list(pool.map(lambda q: _pread_all(fd, q[1], q[2] - q[1]), pieces))
            else:
                parts = [_pread_all(fd, q[1], q[2] - q[1]) for q in pieces]
            joined: list[list[bytes]] = [[] for _ in merged]
            for (b, _s, _e), data in zip(pieces, parts):
                joined[b].append(data)
            datas = [chunks[0] if len(chunks) == 1 else b"".join(chunks) for chunks in joined]
            self.blocks[int(shard_idx)] = [(start, end, data) for (start, end), data in zip(merged, datas) if len(data) == end - start]

    def _handle(self, shard_idx: int):
        handle = self.handles.get(shard_idx)
        if handle is None:
            handle = open(self.record.shard_paths[shard_idx], "rb", buffering=0)
            _fadvise_random(handle.fileno())
            self.handles[shard_idx] = handle
        return handle

    def read(self, row: int, field_name: str) -> bytes:
        off = int(self.record.offsets[field_name][row])
        if off < 0:
            raise KeyError(f"{self.record.key} row {row} has no {field_name}")
        size = int(self.record.sizes[field_name][row])
        shard_idx = int(self.record.shard_idx[row])
        blocks = self.blocks.get(shard_idx) or []
        import bisect

        i = bisect.bisect_right(blocks, (off, float("inf"), b"")) - 1
        if i >= 0:
            start, end, data = blocks[i]
            if off >= start and off + size <= end:
                return data[off - start : off - start + size]
        return _pread_all(self._handle(shard_idx).fileno(), off, size)


def _read_member(handle, offset: int, size: int) -> bytes:
    data = _pread_all(handle.fileno(), offset, size)
    if len(data) != size:
        raise IOError(f"short read at offset {offset}: wanted {size}, got {len(data)}")
    return data




class _TarWalkFallback(Exception):
    """Header layout the fast walker does not handle (pax/gnu long names, base-256 sizes, ...)."""


def _fadvise_random(fd: int) -> None:
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)  # no kernel readahead: skipped members must stay unread
    except (AttributeError, OSError):
        pass


def _pread_all(fd: int, offset: int, size: int) -> bytes:
    """Positioned read of exactly ``size`` bytes (or fewer at EOF), retried on transient I/O errors.

    Network mounts occasionally answer a pread with ENODATA/EIO/ESTALE for a stripe that is
    momentarily unavailable; a short back-off and a second attempt is usually all it takes.
    """
    last_error = None
    for attempt in range(SHARD_READ_ATTEMPTS):
        try:
            chunks = []
            pos, left = offset, size
            while left > 0:
                part = os.pread(fd, left, pos)
                if not part:
                    break
                chunks.append(part)
                pos += len(part)
                left -= len(part)
            return b"".join(chunks)
        except OSError as error:
            last_error = error
            if attempt + 1 < SHARD_READ_ATTEMPTS:
                time.sleep(SHARD_READ_BACKOFF_S * (attempt + 1))
    raise last_error  # type: ignore[misc]


def walk_tar_headers(path: str, *, chunk: int = TAR_WALK_CHUNK):
    """Yield (name, offset_data, size) of every regular file in a ustar/pax tar by reading headers only.

    Reads ``chunk`` bytes at each position it lands on, so a run of small members costs one read and a
    large member (depth maps) is skipped without touching its bytes. Raises ``_TarWalkFallback`` on
    anything beyond plain regular-file headers; the caller then uses ``tarfile``.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        _fadvise_random(fd)
        file_size = os.fstat(fd).st_size
        pos = 0
        buf = b""
        buf_start = 0
        while pos + 512 <= file_size:
            if pos < buf_start or pos + 512 > buf_start + len(buf):
                buf = os.pread(fd, chunk, pos)
                buf_start = pos
                if len(buf) < 512:
                    break
            rel = pos - buf_start
            header = buf[rel : rel + 512]
            if header == b"\0" * 512:
                break  # end-of-archive marker
            if header[257:262] != b"ustar":
                raise _TarWalkFallback("not a ustar header")
            typeflag = header[156:157]
            size_field = header[124:136]
            if size_field[0] & 0x80:
                raise _TarWalkFallback("base-256 size")
            size = int(size_field.split(b"\0", 1)[0].strip() or b"0", 8)
            if typeflag in (b"0", b"\0"):
                name = header[0:100].split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
                prefix = header[345:500].split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
                if prefix:
                    name = prefix + "/" + name
                yield name, pos + 512, size
            elif typeflag != b"5":  # directories carry no data; anything else (x, g, L, K, ...) -> tarfile
                raise _TarWalkFallback(f"typeflag {typeflag!r}")
            pos += 512 + ((size + 511) // 512) * 512
    finally:
        os.close(fd)


def iter_tar_headers_light(path: str):
    """(name, offset_data, size) of regular files, reading 512-byte headers only.

    For archives of large members (mp4 clips, jpg frames) where ``walk_tar_headers``' 64 KB look-ahead
    buys nothing. Understands pax (``x``/``g``) and GNU long-name (``L``) extension entries, so archives
    written by Python's tarfile or GNU tar never fall back to a buffered ``tarfile`` scan.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        _fadvise_random(fd)
        file_size = os.fstat(fd).st_size
        pos = 0
        pending_name: str | None = None
        while pos + 512 <= file_size:
            header = os.pread(fd, 512, pos)
            if len(header) < 512 or header == b"\0" * 512:
                break
            size_field = header[124:136]
            if size_field[0] & 0x80:
                size = int.from_bytes(size_field[1:], "big")
            else:
                size = int(size_field.split(b"\0", 1)[0].strip() or b"0", 8)
            typeflag = header[156:157]
            data_pos = pos + 512
            if typeflag in (b"x", b"L"):
                ext = _pread_all(fd, data_pos, size)
                if typeflag == b"L":
                    pending_name = ext.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
                else:
                    i = 0
                    while i < len(ext):  # pax records: "<len> key=value\n"
                        sp = ext.find(b" ", i)
                        if sp < 0:
                            break
                        try:
                            rec_len = int(ext[i:sp])
                        except ValueError:
                            break
                        rec = ext[sp + 1 : i + rec_len - 1]
                        if rec.startswith(b"path="):
                            pending_name = rec[5:].decode("utf-8", "surrogateescape")
                        i += rec_len
            elif typeflag in (b"0", b"\0"):
                name = header[0:100].split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
                prefix = header[345:500].split(b"\0", 1)[0].decode("utf-8", "surrogateescape") if header[257:262] == b"ustar" and header[262:263] != b" " else ""
                if prefix:
                    name = prefix + "/" + name
                yield (pending_name or name), data_pos, size
                pending_name = None
            elif typeflag not in (b"g", b"5", b"K"):
                pending_name = None
            pos = data_pos + ((size + 511) // 512) * 512
    finally:
        os.close(fd)


def _iter_tar_members(path: str):
    """(name, offset_data, size) for regular files; fast walker with a tarfile fallback.

    The fallback resumes after the last member the fast walker already yielded (data offsets grow
    monotonically through the archive), so no member is ever produced twice.
    """
    last_offset = -1
    try:
        for name, offset_data, size in walk_tar_headers(path):
            yield name, offset_data, size
            last_offset = offset_data
        return
    except _TarWalkFallback:
        pass
    with open(path, "rb", buffering=1 << 20) as raw, tarfile.open(fileobj=raw, mode="r:") as tar:
        for member in tar:
            if member.isfile() and int(member.offset_data) > last_offset:
                yield member.name, int(member.offset_data), int(member.size)


def _shard_cache_path(cache_dir: Path, shard_path: str) -> Path:
    st = os.stat(shard_path)
    digest = hashlib.sha1(os.path.abspath(shard_path).encode("utf-8")).hexdigest()[:20]
    return Path(cache_dir) / f"{digest}-{st.st_size}-{st.st_mtime_ns}.pkl"


def _scan_shard(args) -> list[dict]:
    """Header-only scan of one shard -> per-episode partial records (arrays, first meta, jpeg size).

    Runs in a worker process; returns compact numpy payloads so millions of frames pickle cheaply.
    ``cache_dir`` (optional) memoises the result per shard, keyed by path + size + mtime, so a
    verification pass or a resumed run never re-walks the tar.
    """
    shard_idx, shard_path, cache_dir = args
    last_error = None
    for attempt in range(SHARD_READ_ATTEMPTS):
        try:
            return _scan_shard_once(shard_idx, shard_path, cache_dir)
        except (OSError, EOFError, tarfile.TarError) as error:  # network hiccup, missing stripe, truncated tar ...
            last_error = error
            if attempt + 1 < SHARD_READ_ATTEMPTS:
                time.sleep(SHARD_READ_BACKOFF_S * (attempt + 1))
    logger.warning("shard unreadable after %d attempts, skipping it: %s (%r)", SHARD_READ_ATTEMPTS, shard_path, last_error)
    return [{"key": None, "unreadable": {"shard_idx": shard_idx, "path": shard_path, "error": repr(last_error)}}]


def _scan_shard_once(shard_idx: int, shard_path: str, cache_dir) -> list[dict]:
    cache_path = _shard_cache_path(cache_dir, shard_path) if cache_dir else None
    if cache_path is not None and cache_path.exists():
        try:
            with open(cache_path, "rb") as handle:
                cached = pickle.load(handle)
            for item in cached:
                if item["key"] is not None:
                    item["shard_idx"] = shard_idx
            return cached
        except Exception:  # corrupt / partial cache file: rescan
            pass
    pending: dict[str, dict] = {}
    ignored: dict[str, int] = {}
    for name, offset_data, size in _iter_tar_members(shard_path):
        sample_key, _suffix, field_name = split_sample_member_name(name)
        if sample_key is None:
            # e.g. breast_image.jpg / breast_depth.npy on robot shards: not exported, just counted
            ext = "." + ".".join(os.path.basename(name).split(".")[-2:])
            ignored[ext] = ignored.get(ext, 0) + 1
            continue
        if field_name == "depth_bytes":
            continue  # depth is dropped by design
        episode_key, frame_idx = parse_episode_key(sample_key)
        entry = pending.get(episode_key)
        if entry is None:
            entry = {"frames": {}, "first_meta": None, "first_image": None}
            pending[episode_key] = entry
        frame = entry["frames"].get(frame_idx)
        if frame is None:
            frame = {}
            entry["frames"][frame_idx] = frame
        if field_name in frame:
            raise ValueError(f"duplicate member {name!r} in {shard_path}")
        frame[field_name] = (int(offset_data), int(size))
        if field_name == "meta_bytes" and entry["first_meta"] is None:
            entry["first_meta"] = (int(offset_data), int(size))
        if field_name == "image_bytes" and entry["first_image"] is None:
            entry["first_image"] = (int(offset_data), int(size))

    out = []
    with open(shard_path, "rb", buffering=0) as handle:
        _fadvise_random(handle.fileno())
        for episode_key in sorted(pending):
            entry = pending[episode_key]
            frames = entry["frames"]
            frame_ids = np.array(sorted(frames), dtype=np.int64)
            length = int(frame_ids.shape[0])
            offsets = {name: np.full((length,), -1, dtype=np.int64) for name in MEMBER_FIELDS}
            sizes = {name: np.zeros((length,), dtype=np.int64) for name in MEMBER_FIELDS}
            for row, frame_idx in enumerate(frame_ids.tolist()):
                frame = frames[frame_idx]
                for name in MEMBER_FIELDS:
                    if name in frame:
                        offsets[name][row], sizes[name][row] = frame[name]
            meta = None
            if entry["first_meta"] is not None:
                offset, size = entry["first_meta"]
                meta = json.loads(_read_member(handle, offset, size).decode("utf-8"))
            dims = None
            if entry["first_image"] is not None:
                offset, size = entry["first_image"]
                dims = jpeg_dimensions(_read_member(handle, offset, min(size, 65536)))
            out.append(
                {
                    "key": episode_key,
                    "shard_idx": shard_idx,
                    "frame_ids": frame_ids,
                    "offsets": offsets,
                    "sizes": sizes,
                    "meta": meta,
                    "dims": dims,
                }
            )
    if ignored:
        out.append({"key": None, "ignored": ignored})
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "wb") as handle:
                pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_path)
        except Exception as error:  # cache is an optimisation only
            logger.warning("could not write shard index cache %s: %r", cache_path, error)
    return out




def _part_frame_set(part: dict) -> set[int] | None:
    if "frame_ids" in part:
        return set(part["frame_ids"].tolist())
    if part["contiguous"]:
        return set(range(part["fmin"], part["fmax"] + 1)) if part["n"] else set()
    return None


def _part_range_overlaps(a: dict, b: dict) -> bool:
    amin, amax = (int(a["frame_ids"][0]), int(a["frame_ids"][-1])) if "frame_ids" in a else (a["fmin"], a["fmax"])
    bmin, bmax = (int(b["frame_ids"][0]), int(b["frame_ids"][-1])) if "frame_ids" in b else (b["fmin"], b["fmax"])
    return not (amax < bmin or bmax < amin)


def _reconcile_index(partials: dict[str, list[dict]], unreadable: list[dict], report: dict) -> None:
    """Apply the drop-only-what-is-broken rules before records are built.

    * An episode written twice with the same frame ids (e.g. the same clip in two input
      directories) keeps the copy from the lowest shard index; the duplicate is discarded.
    * An episode whose copies overlap only partially is ambiguous and is dropped whole.
    * An unreadable shard loses its own episodes silently (we never saw them); the episodes that
      may straddle its boundaries (last key of the previous shard, first key of the next) are
      dropped so no truncated episode is exported.
    Every rule records counts and example keys in ``report``.
    """
    deduped: list[str] = []
    ambiguous: list[str] = []
    for key, parts in list(partials.items()):
        if len(parts) < 2:
            continue
        parts.sort(key=lambda p: p["shard_idx"])
        seen: set[int] = set()
        kept: list[dict] = []
        bad = False
        for part in parts:
            ids = _part_frame_set(part)
            if ids is None:  # slim, non-contiguous part: cannot prove anything about overlaps
                bad = any(_part_range_overlaps(part, k) for k in kept)
                if bad:
                    break
                kept.append(part)
                continue
            overlap = len(ids & seen)
            if overlap == 0:
                kept.append(part)
                seen |= ids
            elif overlap == len(ids):
                deduped.append(key)  # identical copy (or subset) already covered
            else:
                bad = True
                break
        if bad:
            ambiguous.append(key)
            del partials[key]
        else:
            partials[key] = kept

    adjacent: list[str] = []
    if unreadable:
        first_key: dict[int, str] = {}
        last_key: dict[int, str] = {}
        for key, parts in partials.items():
            for part in parts:
                idx = part["shard_idx"]
                if idx not in first_key or key < first_key[idx]:
                    first_key[idx] = key
                if idx not in last_key or key > last_key[idx]:
                    last_key[idx] = key
        for item in unreadable:
            idx = item["shard_idx"]
            for key in (last_key.get(idx - 1), first_key.get(idx + 1)):
                if key is not None and key in partials:
                    adjacent.append(key)
                    del partials[key]

    if unreadable:
        report["unreadable_shards"] = unreadable
        logger.warning("%d shard(s) unreadable, e.g. %s", len(unreadable), [u["path"] for u in unreadable[:3]])
    if deduped:
        report["deduplicated_episodes"] = {"count": len(deduped), "keys": sorted(set(deduped))[:50]}
        logger.warning("%d episode(s) appeared in more than one shard with identical frames; kept the first copy, e.g. %s", len(set(deduped)), sorted(set(deduped))[:3])
    if ambiguous:
        report["dropped_ambiguous_duplicates"] = {"count": len(ambiguous), "keys": sorted(ambiguous)[:50]}
        logger.warning("%d episode(s) dropped: copies in several shards with partially overlapping frames, e.g. %s", len(ambiguous), sorted(ambiguous)[:3])
    if adjacent:
        report["dropped_adjacent_to_unreadable"] = {"count": len(adjacent), "keys": sorted(adjacent)[:50]}
        logger.warning("%d episode(s) dropped because they border an unreadable shard, e.g. %s", len(adjacent), sorted(adjacent)[:3])


def index_shards(shard_paths: list[str], *, max_episodes: int | None = None, workers: int = 1, cache_dir: str | Path | None = None, report: dict | None = None, slim: bool = False) -> list[EpisodeRecord]:
    """Walk tar headers and build per-episode seek tables (one meta.json + one JPEG header read per episode).

    ``workers > 1`` scans shards in parallel processes; episodes that span shards are merged.
    ``cache_dir`` stores one pickle per shard so later passes (verification, resume) skip the walk.
    """
    shard_paths = [str(p) for p in shard_paths]
    jobs = [(i, p, str(cache_dir) if cache_dir else None) for i, p in enumerate(shard_paths)]
    partials: dict[str, list[dict]] = {}
    done = 0

    ignored_total: dict[str, int] = {}

    unreadable: list[dict] = []
    report = report if report is not None else {}

    def absorb(items: list[dict]) -> None:
        for item in items:
            if item["key"] is None:
                if "unreadable" in item:
                    unreadable.append(item["unreadable"])
                    continue
                for ext, n in item["ignored"].items():
                    ignored_total[ext] = ignored_total.get(ext, 0) + n
                continue
            partials.setdefault(item["key"], []).append(_summarize_part(item) if slim else item)

    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=_pool_context(), initializer=_worker_init) as pool:
            for items in pool.map(_scan_shard, jobs, chunksize=1):
                absorb(items)
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    logger.info("indexed %d/%d shards (%d episodes so far)", done, len(jobs), len(partials))
    else:
        for job in jobs:
            absorb(_scan_shard(job))
            done += 1
            if done % 50 == 0 or done == len(jobs):
                logger.info("indexed %d/%d shards (%d episodes so far)", done, len(jobs), len(partials))

    if ignored_total:
        logger.info("ignored shard members not used by the LeRobot export: %s", ignored_total)
    _reconcile_index(partials, unreadable, report)
    records: list[EpisodeRecord] = []
    for episode_key in sorted(partials):
        parts = sorted(partials[episode_key], key=lambda p: p["shard_idx"])
        records.append(_slim_record(episode_key, parts) if slim else _merge_parts(episode_key, parts, shard_paths))
        if max_episodes is not None and len(records) >= max_episodes:
            break
    return records


def _summarize_part(item: dict) -> dict:
    """Replace a scanned part's per-frame arrays by a few numbers (what the driver needs to plan)."""
    frame_ids = item["frame_ids"]
    n = int(frame_ids.shape[0])
    fmin = int(frame_ids[0]) if n else -1
    fmax = int(frame_ids[-1]) if n else -1
    return {
        "key": item["key"],
        "shard_idx": item["shard_idx"],
        "n": n,
        "fmin": fmin,
        "fmax": fmax,
        "contiguous": n <= 1 or (fmax - fmin + 1 == n),
        "missing": {name: int((item["offsets"][name] < 0).sum()) for name in MEMBER_FIELDS},
        "image_bytes": int(item["sizes"]["image_bytes"].sum()),
        "meta": item["meta"],
        "dims": item["dims"],
    }


def _slim_record(episode_key: str, parts: list[dict]) -> EpisodeRecord:
    for name in ("image_bytes", "lowdim_bytes", "meta_bytes"):
        missing = sum(p["missing"][name] for p in parts)
        if missing:
            raise ValueError(f"episode {episode_key!r}: {missing} frames missing {name}")
    meta = next((p["meta"] for p in parts if p["meta"] is not None), None)
    dims = next((p["dims"] for p in parts if p["dims"] is not None), None)
    if meta is None or dims is None:
        raise ValueError(f"episode {episode_key!r} has no meta/image member")
    contiguous = all(p["contiguous"] for p in parts) and all(parts[i + 1]["fmin"] == parts[i]["fmax"] + 1 for i in range(len(parts) - 1))
    return EpisodeRecord(
        key=episode_key,
        shard_paths=None,
        frame_ids=None,
        shard_idx=None,
        offsets=None,
        sizes=None,
        meta=meta,
        image_bytes_total=sum(p["image_bytes"] for p in parts),
        height=dims[0],
        width=dims[1],
        split=str(meta.get("split") or "train"),
        source_episode_index=int(meta.get("episode_index", -1)),
        frame_count=sum(p["n"] for p in parts),
        frame_first=min(p["fmin"] for p in parts),
        frames_contiguous=contiguous,
        mano_complete=all(p["missing"]["mano_bytes"] == 0 for p in parts),
        parts=tuple(p["shard_idx"] for p in parts),
    )


def _merge_parts(episode_key: str, parts: list[dict], shard_paths: list[str]) -> EpisodeRecord:
    """Full record from the scanned parts of one episode (already sorted by shard index)."""
    frame_ids = np.concatenate([p["frame_ids"] for p in parts])
    shard_idx_arr = np.concatenate([np.full(p["frame_ids"].shape, p["shard_idx"], dtype=np.int32) for p in parts])
    offsets = {name: np.concatenate([p["offsets"][name] for p in parts]) for name in MEMBER_FIELDS}
    sizes = {name: np.concatenate([p["sizes"][name] for p in parts]) for name in MEMBER_FIELDS}
    order = np.argsort(frame_ids, kind="stable")
    frame_ids = frame_ids[order]
    if frame_ids.shape[0] > 1 and np.any(np.diff(frame_ids) == 0):
        raise ValueError(f"episode {episode_key!r}: duplicate frame ids across shards")
    shard_idx_arr = shard_idx_arr[order]
    offsets = {k: v[order] for k, v in offsets.items()}
    sizes = {k: v[order] for k, v in sizes.items()}
    for name in ("image_bytes", "lowdim_bytes", "meta_bytes"):
        missing = int((offsets[name] < 0).sum())
        if missing:
            raise ValueError(f"episode {episode_key!r}: {missing} frames missing {name}")
    meta = next((p["meta"] for p in parts if p["meta"] is not None), None)
    dims = next((p["dims"] for p in parts if p["dims"] is not None), None)
    if meta is None or dims is None:
        raise ValueError(f"episode {episode_key!r} has no meta/image member")
    return EpisodeRecord(
        key=episode_key,
        shard_paths=shard_paths,
        frame_ids=frame_ids,
        shard_idx=shard_idx_arr,
        offsets=offsets,
        sizes=sizes,
        meta=meta,
        image_bytes_total=int(sizes["image_bytes"].sum()),
        height=dims[0],
        width=dims[1],
        split=str(meta.get("split") or "train"),
        source_episode_index=int(meta.get("episode_index", -1)),
        frame_count=int(frame_ids.shape[0]),
        frame_first=int(frame_ids[0]) if frame_ids.shape[0] else -1,
        frames_contiguous=frame_ids.shape[0] <= 1 or bool(np.all(np.diff(frame_ids) == 1)),
        mano_complete=not bool((offsets["mano_bytes"] < 0).any()),
        parts=tuple(p["shard_idx"] for p in parts),
    )


_HYDRATE_CACHE: dict[str, list[dict]] = {}  # shard path -> scanned parts (keyed by path: shard indices repeat across datasets)
HYDRATE_CACHE_SHARDS = 16


def hydrate_record(record: EpisodeRecord, shard_paths: list[str], cache_dir: str | Path | None) -> EpisodeRecord:
    """Turn a slim record back into a full one by re-reading its parts from the per-shard index cache.

    The cache pickles were written during indexing; a missing one is rebuilt by rescanning that
    shard's headers. A small per-process LRU keeps the last shards' parts around, since the
    episodes of one file group usually sit in one or two consecutive shards.
    """
    if record.hydrated:
        return record
    parts: list[dict] = []
    for shard_idx in record.parts:
        shard_path = shard_paths[shard_idx]
        items = _HYDRATE_CACHE.get(shard_path)
        if items is None:
            items = _scan_shard_once(shard_idx, shard_path, str(cache_dir) if cache_dir else None)
            if len(_HYDRATE_CACHE) >= HYDRATE_CACHE_SHARDS:
                _HYDRATE_CACHE.pop(next(iter(_HYDRATE_CACHE)))
            _HYDRATE_CACHE[shard_path] = items
        for item in items:
            if item.get("key") == record.key:
                item["shard_idx"] = shard_idx
                parts.append(item)
    if len(parts) != len(record.parts):
        raise ValueError(f"episode {record.key!r}: expected parts in shards {list(record.parts)}, found {sorted(p['shard_idx'] for p in parts)}")
    parts.sort(key=lambda p: p["shard_idx"])
    full = _merge_parts(record.key, parts, shard_paths)
    if full.length != record.length:
        raise ValueError(f"episode {record.key!r}: index cache has {full.length} frames, driver planned {record.length}")
    full.split = record.split
    full.source_episode_index = record.source_episode_index
    full.episode_index = record.episode_index
    return full


# --------------------------------------------------------------------------------------
# Ordering, tasks, file planning
# --------------------------------------------------------------------------------------


def episode_task(record: EpisodeRecord, task_source: str, task_name: str | None = None) -> str:
    if task_name:
        return str(task_name)
    return _episode_task_from_meta(record, task_source)


def _episode_task_from_meta(record: EpisodeRecord, task_source: str) -> str:
    meta = record.meta
    instructions = [str(x) for x in (meta.get("instruction") or []) if str(x).strip()]
    language = meta.get("language")
    language = str(language).strip() if language else ""
    dataset_name = str(meta.get("dataset_name") or "")
    if task_source == "language":
        candidates = [language, instructions[0] if instructions else "", dataset_name]
    elif task_source == "first_instruction":
        candidates = [instructions[0] if instructions else "", language, dataset_name]
    elif task_source == "dataset_name":
        candidates = [dataset_name, language]
    elif task_source == "clip_id":
        candidates = [str(meta.get("clip_id") or record.key)]
    else:
        raise ValueError(f"unknown task_source {task_source!r}")
    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def order_episodes(records: list[EpisodeRecord], split_order: tuple[str, ...]) -> list[EpisodeRecord]:
    """Group by split (requested order first, unknown splits after, alphabetically), then by source index."""
    rank = {name: i for i, name in enumerate(split_order)}
    extra = sorted({r.split for r in records if r.split not in rank})
    for name in extra:
        rank[name] = len(rank)
    ordered = sorted(records, key=lambda r: (rank[r.split], r.source_episode_index, r.key))
    for idx, record in enumerate(ordered):
        record.episode_index = idx
    return ordered


def build_splits(records: list[EpisodeRecord]) -> dict[str, str]:
    splits: dict[str, str] = {}
    start = 0
    current = None
    for idx, record in enumerate(records):
        if record.split != current:
            if current is not None:
                splits[current] = f"{start}:{idx}"
            current = record.split
            start = idx
    if current is not None:
        splits[current] = f"{start}:{len(records)}"
    return splits


def estimated_row_bytes(cfg: ConvertConfig) -> int:
    per_row = 4 * (2 * cfg.state_dim) + 4 + 8 * 4
    if cfg.include_extrinsic:
        per_row += 4 * 16
    if cfg.include_mano:
        per_row += 4 * MANO_FLAT_DIM
    if cfg.include_presence:
        per_row += 8
    return per_row


@dataclass
class FileGroup:
    file_seq: int  # linear file index; chunk/file derived from it via chunk_file()
    episodes: list[int]  # positions into the ordered record list
    frame_start: int  # global frame index of the first frame


def plan_file_groups(records: list[EpisodeRecord], cfg: ConvertConfig) -> list[FileGroup]:
    """Greedy split into file pairs (parquet + mp4 share chunk/file indices, like the EgoSteer converter)."""
    data_limit = cfg.data_files_size_in_mb * 1024 * 1024
    video_limit = cfg.video_files_size_in_mb * 1024 * 1024
    row_bytes = estimated_row_bytes(cfg)
    groups: list[FileGroup] = []
    current: list[int] = []
    data_est = 0
    video_est = 0.0
    frame_cursor = 0
    group_frame_start = 0
    for pos, record in enumerate(records):
        ep_data = record.length * row_bytes
        ep_video = record.image_bytes_total * cfg.video_size_ratio if cfg.include_video else 0.0
        if current and (data_est + ep_data > data_limit or video_est + ep_video > video_limit):
            groups.append(FileGroup(len(groups), current, group_frame_start))
            current = []
            data_est = 0
            video_est = 0.0
            group_frame_start = frame_cursor
        current.append(pos)
        data_est += ep_data
        video_est += ep_video
        frame_cursor += record.length
    if current:
        groups.append(FileGroup(len(groups), current, group_frame_start))
    return groups


def chunk_file(file_seq: int, chunks_size: int) -> tuple[int, int]:
    return file_seq // chunks_size, file_seq % chunks_size


# --------------------------------------------------------------------------------------
# Feature schema
# --------------------------------------------------------------------------------------


def build_features(cfg: ConvertConfig, height: int, width: int) -> dict[str, dict]:
    names = state_names(cfg.state_layout)
    features: dict[str, dict] = {}
    if cfg.include_video:
        features[HEAD_VIDEO_KEY] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": cfg.pix_fmt,
                "video.is_depth_map": False,
                "video.fps": int(cfg.fps),
                "video.channels": 3,
                "video.g": int(cfg.gop),
                "video.crf": int(cfg.crf),
                "video.preset": cfg.preset,
                "video.fast_decode": 0,
                "video.extra_options": {},
                "video.video_backend": "pyav",  # lerobot's only accepted value; we encode with the ffmpeg CLI (see provenance)
                "has_audio": False,
                "is_depth_map": False,
            },
        }
    features[STATE_KEY] = {"dtype": "float32", "shape": [cfg.state_dim], "names": names}
    features[ACTION_KEY] = {"dtype": "float32", "shape": [cfg.state_dim], "names": names}
    if cfg.include_presence:
        features[PRESENCE_KEY] = {"dtype": "int64", "shape": [1], "names": None}
    if cfg.include_extrinsic:
        features[EXTRINSIC_KEY] = {"dtype": "float32", "shape": [16], "names": extrinsic_names()}
    if cfg.include_mano:
        features[MANO_KEY] = {"dtype": "float32", "shape": [MANO_FLAT_DIM], "names": mano_names()}
    if cfg.source_map:
        features[SOURCE_FRAME_INDEX_KEY] = {"dtype": "int64", "shape": [1], "names": None}
        features[SOURCE_FRAME_OBSERVED_KEY] = {"dtype": "bool", "shape": [1], "names": None}
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return features


def source_frame_columns(record: EpisodeRecord, row: SourceMapRow | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame source frame numbers and observed flags for one episode (all -1 / False without a map row)."""
    T = record.length
    if row is None or not row.usable:
        return np.full((T,), -1, dtype=np.int64), np.zeros((T,), dtype=bool)
    rel = (record.frame_ids - int(record.frame_ids[0])).astype(np.int64)
    idx = int(row.source_frame_start) + rel * int(max(1, row.frame_stride))
    observed = (rel % int(max(1, row.observed_stride))) == 0
    return idx, observed


def _pa_type(dtype: str):
    return {"float32": pa.float32(), "float64": pa.float64(), "int64": pa.int64(), "int32": pa.int32(), "bool": pa.bool_()}[dtype]


def _hf_value(dtype: str) -> dict:
    return {"dtype": dtype, "_type": "Value"}


def build_arrow_schema(features: dict[str, dict]) -> pa.Schema:
    """Arrow schema + huggingface metadata that ``datasets.Dataset.from_parquet`` understands."""
    fields = []
    hf_features: dict[str, dict] = {}
    for key, ft in features.items():
        if ft["dtype"] in ("video", "image"):
            continue
        shape = tuple(ft["shape"])
        if shape == (1,):
            fields.append(pa.field(key, _pa_type(ft["dtype"])))
            hf_features[key] = _hf_value(ft["dtype"])
        elif len(shape) == 1:
            # datasets stores Sequence(length=N) as a plain list<T> (what the EgoSteer reference carries), not fixed_size_list
            fields.append(pa.field(key, pa.list_(_pa_type(ft["dtype"]))))
            hf_features[key] = {"feature": _hf_value(ft["dtype"]), "length": int(shape[0]), "_type": "Sequence"}
        else:
            raise ValueError(f"feature {key!r}: only 1-d shapes are exported ({shape})")
    metadata = {b"huggingface": json.dumps({"info": {"features": hf_features}}).encode("utf-8")}
    return pa.schema(fields, metadata=metadata)


# --------------------------------------------------------------------------------------
# Per-frame decoding + geometry
# --------------------------------------------------------------------------------------


def decode_npy(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """(...,6) column-major [R00,R10,R20,R01,R11,R21] -> (...,3,3) via Gram-Schmidt."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / (np.linalg.norm(a2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def rotmat_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,6) first two columns, column-major (EgoSteer / EgoSmith convention)."""
    return np.concatenate([rotmat[..., :, 0], rotmat[..., :, 1]], axis=-1)


def hand_block_from_lowdim(block48: np.ndarray, w2c: np.ndarray | None) -> np.ndarray:
    """Re-pack a 48-d EgoSmith state/action block into the EgoSteer hand order.

    Input order: left_pos(3) right_pos(3) left_rot6d(6) right_rot6d(6) left_tips(15) right_tips(15).
    Output order: left_wrist(pos3+rot6) right_wrist(pos3+rot6) left_tips(15) right_tips(15).
    When ``w2c`` (T,4,4) is given, points and rotations are moved from the world frame into that camera frame.
    """
    block48 = np.asarray(block48, dtype=np.float32)
    T = block48.shape[0]
    left_pos = block48[:, LEFT_HAND_TRANSLATION_SLICE]
    right_pos = block48[:, RIGHT_HAND_TRANSLATION_SLICE]
    left_rot = block48[:, LEFT_ROOT_ROT6D_SLICE]
    right_rot = block48[:, RIGHT_ROOT_ROT6D_SLICE]
    left_tips = block48[:, LEFT_FINGERTIPS_SLICE].reshape(T, 5, 3)
    right_tips = block48[:, RIGHT_FINGERTIPS_SLICE].reshape(T, 5, 3)
    if w2c is not None:
        R = w2c[:, :3, :3].astype(np.float32)
        t = w2c[:, :3, 3].astype(np.float32)
        left_pos = np.einsum("tij,tj->ti", R, left_pos) + t
        right_pos = np.einsum("tij,tj->ti", R, right_pos) + t
        left_rot = rotmat_to_rot6d(R @ rot6d_to_rotmat(left_rot))
        right_rot = rotmat_to_rot6d(R @ rot6d_to_rotmat(right_rot))
        left_tips = np.einsum("tij,tkj->tki", R, left_tips) + t[:, None, :]
        right_tips = np.einsum("tij,tkj->tki", R, right_tips) + t[:, None, :]
    return np.concatenate(
        [left_pos, left_rot, right_pos, right_rot, left_tips.reshape(T, 15), right_tips.reshape(T, 15)],
        axis=-1,
    ).astype(np.float32)


def lowdim_to_state_action(lowdim: np.ndarray, cfg: ConvertConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (state, action, w2c_flat16, intrinsic4) for one episode from its (T,116) lowdim matrix."""
    lowdim = np.asarray(lowdim, dtype=np.float32)
    if lowdim.ndim != 2 or lowdim.shape[1] != LOWDIM_SIZE:
        raise ValueError(f"expected lowdim (T,{LOWDIM_SIZE}), got {lowdim.shape}")
    T = lowdim.shape[0]
    w2c = lowdim[:, EXTRINSIC_SLICE].reshape(T, 4, 4)
    intrinsic = lowdim[:, INTRINSIC_SLICE]
    frame_w2c = w2c if cfg.hand_frame == "camera" else None
    state48 = hand_block_from_lowdim(lowdim[:, :HAWOR_STATE_DIM], frame_w2c)
    # action_t = state_{t+1} expressed in the observation frame of t (same camera as state_t)
    action48 = hand_block_from_lowdim(lowdim[:, LOWDIM_ACTION_OFFSET : LOWDIM_ACTION_OFFSET + HAWOR_STATE_DIM], frame_w2c)
    if cfg.state_layout == "egosteer74":
        pad = np.full((T, EGOSTEER_ROBOT_PAD_DIM), cfg.pad_value, dtype=np.float32)
        state = np.concatenate([pad, state48], axis=-1)
        action = np.concatenate([pad, action48], axis=-1)
    else:
        state, action = state48, action48
    return state, action, w2c.reshape(T, 16), intrinsic


def intrinsic_matrix9(intrinsic4: np.ndarray) -> list[float]:
    fx, fy, cx, cy = (float(v) for v in intrinsic4[:4])
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


# --------------------------------------------------------------------------------------
# Stats (mirrors lerobot.datasets.compute_stats conventions)
# --------------------------------------------------------------------------------------


def estimate_num_samples(dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75) -> int:
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    if data_len <= 0:
        return []
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_hw(img_hwc: np.ndarray, target_size: int = 150, max_size_threshold: int = 300) -> np.ndarray:
    height, width = img_hwc.shape[:2]
    if max(width, height) < max_size_threshold:
        return img_hwc
    factor = int(width / target_size) if width > height else int(height / target_size)
    return img_hwc[::factor, ::factor]


def vector_stats(array: np.ndarray, count: int | None = None) -> dict[str, np.ndarray]:
    """Per-dimension stats over axis 0 for (N,D) (or (N,) -> keepdims (1,)) arrays."""
    arr = np.asarray(array)
    keepdims_1d = arr.ndim == 1
    if keepdims_1d:
        arr = arr.reshape(-1, 1)
    arr64 = arr.astype(np.float64)
    n = int(arr.shape[0])
    stats = {
        "min": arr64.min(axis=0),
        "max": arr64.max(axis=0),
        "mean": arr64.mean(axis=0),
        "std": arr64.std(axis=0),
        "count": np.array([n if count is None else count], dtype=np.int64),
    }
    if n >= 2:
        qs = np.quantile(arr64, DEFAULT_QUANTILES, axis=0)
        for q, row in zip(DEFAULT_QUANTILES, qs):
            stats[f"q{int(q * 100):02d}"] = row
    else:
        for q in DEFAULT_QUANTILES:
            stats[f"q{int(q * 100):02d}"] = stats["mean"].copy()
    return stats


def image_stats_from_samples(samples_hwc: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Per-channel stats (3,1,1) in [0,1] from a list of uint8 HxWx3 images; count = number of images."""
    pixels = np.concatenate([img.reshape(-1, 3) for img in samples_hwc], axis=0).astype(np.float32) / 255.0
    stats = vector_stats(pixels, count=len(samples_hwc))
    return {k: (v if k == "count" else v.reshape(3, 1, 1)) for k, v in stats.items()}


def aggregate_feature_stats(stats_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    means = np.stack([s["mean"] for s in stats_list])
    variances = np.stack([s["std"] ** 2 for s in stats_list])
    counts = np.stack([s["count"] for s in stats_list]).astype(np.float64)
    total_count = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)
    total_mean = (means * counts).sum(axis=0) / total_count
    delta = means - total_mean
    total_var = ((variances + delta**2) * counts).sum(axis=0) / total_count
    out = {
        "min": np.min(np.stack([s["min"] for s in stats_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_var),
        "count": total_count.astype(np.int64),
    }
    for key in stats_list[0]:
        if key.startswith("q") and key[1:].isdigit() and all(key in s for s in stats_list):
            out[key] = (np.stack([s[key] for s in stats_list]) * counts).sum(axis=0) / total_count
    return out


def aggregate_stats(per_episode: list[dict[str, dict[str, np.ndarray]]]) -> dict[str, dict[str, np.ndarray]]:
    keys = sorted({k for stats in per_episode for k in stats})
    return {k: aggregate_feature_stats([s[k] for s in per_episode if k in s]) for k in keys}


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _stats_from_lists(stats: dict) -> dict[str, dict[str, np.ndarray]]:
    return {k: {sk: np.asarray(sv) for sk, sv in v.items()} for k, v in stats.items()}


# --------------------------------------------------------------------------------------
# Video encoding
# --------------------------------------------------------------------------------------


def force_keyframe_args(frames: list[int] | None, fps: int) -> list[str]:
    """``-force_key_frames`` times for the given frame numbers. Half a frame early, so float rounding can never
    push a time past its frame (ffmpeg forces the first frame whose pts is >= the given time)."""
    starts = sorted({int(f) for f in (frames or []) if int(f) > 0})
    if not starts:
        return []
    return ["-force_key_frames", ",".join(f"{(f - 0.5) / float(fps):.6f}" for f in starts)]


def probe_keyframe_times(path: Path, *, ffmpeg: str = "ffmpeg") -> list[float] | None:
    """pts (seconds) of every keyframe; decodes keyframes only, so it is cheap. None when ffmpeg cannot tell."""
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-skip_frame", "nokey", "-i", str(path), "-an", "-vf", "showinfo", "-fps_mode", "passthrough", "-f", "null", "-"],
            capture_output=True, text=True, timeout=3600,
        )
        if out.returncode != 0:  # older ffmpeg: no -fps_mode
            out = subprocess.run(
                [ffmpeg, "-hide_banner", "-skip_frame", "nokey", "-i", str(path), "-an", "-vf", "showinfo", "-vsync", "0", "-f", "null", "-"],
                capture_output=True, text=True, timeout=3600,
            )
        if out.returncode != 0:
            return None
    except Exception:  # noqa: BLE001
        return None
    times = [float(m.group(1)) for m in re.finditer(r"pts_time:\s*([0-9.]+)", out.stderr)]
    return times or None


class FfmpegJpegEncoder:
    """Pipe raw JPEG frames into one ffmpeg process (decode once -> h264), constant fps."""

    def __init__(self, cfg: ConvertConfig, out_path: Path, width: int, height: int, *, keyframe_frames: list[int] | None = None):
        """``keyframe_frames``: frame numbers (within this file) that must be IDR frames - the first frame of
        every episode, so a reader can seek to ``from_timestamp`` and decode exactly the episode's frames."""
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_path.with_suffix(".ffmpeg.log")
        self.width = width
        self.height = height
        self.frames = 0
        cmd = [
            cfg.ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "image2pipe",
            "-framerate", str(cfg.fps),
            "-vcodec", "mjpeg",
            "-i", "pipe:0",
            "-an",
            # mjpeg decodes to full-range yuvj420p; go through RGB and force TV range so the stream is a plain
            # limited-range yuv420p like a PIL/pyav RGB encode (some ffmpeg builds otherwise tag the output yuvj420p)
            "-vf", f"format=rgb24,scale=out_range=tv,format={cfg.pix_fmt}",
            "-color_range", "tv",
            "-c:v", cfg.vcodec,
            "-preset", cfg.preset,
            "-crf", str(cfg.crf),
            "-g", str(cfg.gop),
            *force_keyframe_args(keyframe_frames, cfg.fps),
            "-pix_fmt", cfg.pix_fmt,
            "-threads", str(cfg.ffmpeg_threads),
            "-r", str(cfg.fps),
            "-movflags", "+faststart",
            str(self.out_path),
        ]
        self._log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log)

    def write(self, jpeg_bytes: bytes) -> None:
        height, width = jpeg_dimensions(jpeg_bytes)
        if (height, width) != (self.height, self.width):
            raise ValueError(f"frame size {(height, width)} != video size {(self.height, self.width)}")
        try:
            self.proc.stdin.write(jpeg_bytes)
        except BrokenPipeError as error:
            raise RuntimeError(f"ffmpeg died early, see {self.log_path}") from error
        self.frames += 1

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        code = self.proc.wait()
        self._log.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with {code}, see {self.log_path}")
        if self.log_path.exists() and self.log_path.stat().st_size == 0:
            self.log_path.unlink()

    def abort(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        finally:
            self.proc.kill()
            self.proc.wait()
            self._log.close()


def probe_video_pix_fmt(path: Path, *, ffprobe: str = "ffprobe", ffmpeg: str = "ffmpeg") -> str | None:
    """Pixel format of the first video stream (ffprobe, else parsed from the ffmpeg banner)."""
    if shutil.which(ffprobe) is not None or Path(ffprobe).exists():
        result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=False)
        text = result.stdout.strip().rstrip(",")
        if text:
            return text
    if shutil.which(ffmpeg) is None and not Path(ffmpeg).exists():
        return None
    banner = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True, check=False).stderr
    match = re.search(r"Video: [^\n]*?,\s*([a-z0-9]+)(?:\([^)]*\))?,\s*\d+x\d+", banner)
    return match.group(1) if match else None


def probe_video_frames(path: Path, *, ffprobe: str = "ffprobe", ffmpeg: str = "ffmpeg") -> int | None:
    """Frame count of a video. Uses ffprobe container metadata when available, else ffmpeg stream-copy to null.

    Returns None only when neither tool can be run.
    """
    if shutil.which(ffprobe) is not None or Path(ffprobe).exists():
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        text = result.stdout.strip()
        if text.isdigit():
            return int(text)
    if shutil.which(ffmpeg) is None and not Path(ffmpeg).exists():
        return None
    # Full decode to the null muxer: stream-copy runs do not report frame counts on ffmpeg >= 6.
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", os.devnull, "-progress", "pipe:1"],
        capture_output=True,
        text=True,
        check=False,
    )
    frames = [int(m.group(1)) for m in re.finditer(r"^frame=(\d+)", result.stdout, flags=re.MULTILINE)]
    return frames[-1] if frames else None


# --------------------------------------------------------------------------------------
# Pass 2: one file group
# --------------------------------------------------------------------------------------



MAX_REPAIRED_FRAMES_PER_EPISODE = 30


def jpeg_looks_complete(data: bytes) -> bool:
    """Cheap structural check: SOI marker at the start, EOI marker at the end (a truncated or
    zero-length member fails it; cv2/ffmpeg would reject or drop such a frame)."""
    return len(data) >= 4 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


def repair_episode_images(key: str, images: list[bytes]) -> list[int]:
    """Replace structurally broken JPEGs in place by the nearest intact frame (previous, else next).

    Returns the repaired rows. One bad frame must not cost the whole episode (or, mid-conversion,
    the whole dataset: dropping an episode here would shift every later global index); more than
    ``MAX_REPAIRED_FRAMES_PER_EPISODE`` bad frames means the episode is broken and is reported.
    """
    bad = [i for i, data in enumerate(images) if not jpeg_looks_complete(data)]
    if not bad:
        return []
    if len(bad) == len(images):
        raise ValueError(f"{key}: every JPEG frame is truncated or empty")
    if len(bad) > MAX_REPAIRED_FRAMES_PER_EPISODE:
        raise ValueError(f"{key}: {len(bad)} of {len(images)} JPEG frames are broken (limit {MAX_REPAIRED_FRAMES_PER_EPISODE}); rows {bad[:10]}...")
    bad_set = set(bad)
    last_good = None
    for i in range(len(images)):
        if i in bad_set:
            continue
        last_good = i
        break
    for i in bad:
        prev = max((j for j in range(i - 1, -1, -1) if j not in bad_set), default=None)
        src = prev if prev is not None else next(j for j in range(i + 1, len(images)) if j not in bad_set)
        images[i] = images[src]
    return bad


def _decode_jpeg_rgb(data: bytes) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2 failed to decode a JPEG frame")
    return img[:, :, ::-1]


@dataclass
class EpisodeResult:
    episode_index: int
    length: int
    task: str
    instructions: list[str]
    language: str
    split: str
    clip_id: str
    dataset_name: str
    source_episode_index: int
    source_frame_start: int
    source_frames_contiguous: bool
    intrinsics9: list[float]
    intrinsics_constant: bool
    presence_counts: dict[str, int]
    video_from_timestamp: float | None  # None when the dataset has no video stream
    video_to_timestamp: float | None
    stats: dict  # feature -> stat -> list
    repaired_frames: list[int] = field(default_factory=list)  # rows whose JPEG was replaced by a neighbour
    image_size: tuple[int, int] = (0, 0)  # (width, height) of the exported frames
    source_media: str = ""
    source_media_fps: float = 0.0
    source_media_frame_start: int = -1
    source_media_frame_end: int = -1
    source_media_undistort: str = ""
    source_camera: list[float] = field(default_factory=lambda: list(NAN_CAMERA))


@dataclass
class GroupResult:
    file_seq: int
    chunk_index: int
    file_index: int
    frames: int
    height: int
    width: int
    video_frames_probed: int | None
    episodes: list[EpisodeResult] = field(default_factory=list)
    keyframes_at_episode_starts: bool = False  # False for file groups encoded before keyframes were forced at episode starts
    task_policy: str = ""  # digest of the task vocabulary this group's task_index column was written against


def convert_file_group(
    file_seq: int,
    frame_start: int,
    group_records: list[EpisodeRecord],
    cfg: ConvertConfig,
    root: Path,
    task_index_of: dict[str, int],
    source_map: SourceMap | None = None,
) -> GroupResult:
    """Write one parquet + one mp4 for the given (already ordered) episodes."""
    root = Path(root)
    chunk_index, file_index = chunk_file(file_seq, cfg.chunks_size)
    data_path = root / DATA_PATH.format(chunk_index=chunk_index, file_index=file_index)
    video_path = root / VIDEO_PATH.format(video_key=HEAD_VIDEO_KEY, chunk_index=chunk_index, file_index=file_index)
    first = group_records[0]
    height, width = first.height, first.width
    features = build_features(cfg, height, width)
    schema = build_arrow_schema(features)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_data = data_path.with_name(data_path.name + ".tmp")
    tmp_video = video_path.with_name(video_path.name + ".tmp.mp4")
    episode_starts = list(np.cumsum([0] + [r.length for r in group_records[:-1]]))
    encoder = FfmpegJpegEncoder(cfg, tmp_video, width, height, keyframe_frames=episode_starts) if cfg.include_video else None
    writer = pq.ParquetWriter(str(tmp_data), schema=schema, compression="snappy", use_dictionary=True)
    handles: dict[int, object] = {}
    results: list[EpisodeResult] = []
    global_index = frame_start
    video_cursor = 0.0
    try:
        for record in group_records:
            T = record.length
            lowdim = np.empty((T, LOWDIM_SIZE), dtype=np.float32)
            mano = np.empty((T, MANO_FLAT_DIM), dtype=np.float32) if cfg.include_mano else None
            presence = np.zeros((T,), dtype=np.int64)
            sample_rows = set(sample_indices(T))
            image_samples: list[np.ndarray] = []
            wanted = tuple(
                name
                for name in MEMBER_FIELDS
                if name in ("lowdim_bytes", "meta_bytes") or (name == "image_bytes" and encoder is not None) or (name == "mano_bytes" and cfg.include_mano)
            )
            reader = EpisodeReader(record, handles, fields=wanted, gap=COALESCE_GAP if encoder is not None else LABEL_COALESCE_GAP)
            images: list[bytes] = []
            repaired: list[int] = []
            if encoder is not None:
                images = [reader.read(row, "image_bytes") for row in range(T)]
                repaired = repair_episode_images(record.key, images)
                if repaired:
                    logger.warning("%s: %d broken JPEG frame(s) replaced by the nearest intact frame (rows %s)", record.key, len(repaired), repaired[:10])
            for row in range(T):
                image_bytes = images[row] if encoder is not None else None
                lowdim_bytes = reader.read(row, "lowdim_bytes")
                meta_bytes = reader.read(row, "meta_bytes")
                vec = decode_npy(lowdim_bytes).reshape(-1)
                if vec.shape[0] < LOWDIM_SIZE:
                    raise ValueError(f"{record.key} frame {int(record.frame_ids[row])}: lowdim size {vec.shape[0]} < {LOWDIM_SIZE}")
                lowdim[row] = vec[:LOWDIM_SIZE]  # robot shards append breast-camera extrinsic/intrinsic (136-d); head block is the first 116
                presence[row] = int(json.loads(meta_bytes.decode("utf-8")).get("presence", 0))
                if mano is not None:
                    if int(record.offsets["mano_bytes"][row]) < 0:
                        raise ValueError(f"{record.key}: mano.npy missing at frame {int(record.frame_ids[row])} (use --no_mano)")
                    arr = decode_npy(reader.read(row, "mano_bytes"))
                    if arr.shape != MANO_SAMPLE_SHAPE:
                        raise ValueError(f"{record.key}: mano shape {arr.shape} != {MANO_SAMPLE_SHAPE}")
                    mano[row] = arr.reshape(-1)
                if encoder is not None:
                    encoder.write(image_bytes)
                    if row in sample_rows:
                        try:
                            image_samples.append(auto_downsample_hw(_decode_jpeg_rgb(image_bytes)))
                        except ValueError:  # intact markers but undecodable: stats just lose one sample
                            logger.warning("%s row %d: JPEG not decodable by cv2, skipped for image stats", record.key, row)

            state, action, w2c16, intrinsic = lowdim_to_state_action(lowdim, cfg)
            task = episode_task(record, cfg.task_source, cfg.task_name)
            task_index = task_index_of[task]
            timestamps = (np.arange(T, dtype=np.float64) / float(cfg.fps)).astype(np.float32)
            frame_index = np.arange(T, dtype=np.int64)
            index = np.arange(global_index, global_index + T, dtype=np.int64)
            episode_col = np.full((T,), record.episode_index, dtype=np.int64)
            task_col = np.full((T,), task_index, dtype=np.int64)

            columns: dict[str, np.ndarray] = {STATE_KEY: state, ACTION_KEY: action}
            if cfg.include_presence:
                columns[PRESENCE_KEY] = presence
            if cfg.include_extrinsic:
                columns[EXTRINSIC_KEY] = w2c16.astype(np.float32)
            if cfg.include_mano:
                columns[MANO_KEY] = mano
            map_row = source_map_row(source_map, record) if source_map is not None else None
            if cfg.source_map:
                src_idx, src_observed = source_frame_columns(record, map_row)
                columns[SOURCE_FRAME_INDEX_KEY] = src_idx
                columns[SOURCE_FRAME_OBSERVED_KEY] = src_observed
            columns.update(
                {
                    "timestamp": timestamps,
                    "frame_index": frame_index,
                    "episode_index": episode_col,
                    "index": index,
                    "task_index": task_col,
                }
            )
            arrays = {}
            for name in schema.names:
                col = columns[name]
                pa_field = schema.field(name)
                if pa.types.is_list(pa_field.type):
                    col2 = np.ascontiguousarray(col).reshape(len(col), -1)
                    flat = pa.array(col2.reshape(-1), type=pa_field.type.value_type)
                    offsets = pa.array(np.arange(0, (len(col2) + 1) * col2.shape[1], col2.shape[1], dtype=np.int32))
                    arrays[name] = pa.ListArray.from_arrays(offsets, flat)
                else:
                    arrays[name] = pa.array(col, type=pa_field.type)
            writer.write_table(pa.Table.from_arrays([arrays[n] for n in schema.names], schema=schema))

            ep_stats: dict[str, dict[str, np.ndarray]] = {}
            for name, col in columns.items():
                if col.dtype == np.bool_:
                    continue  # observed flags carry no useful statistics
                ep_stats[name] = vector_stats(col)
            if image_samples:
                ep_stats[HEAD_VIDEO_KEY] = image_stats_from_samples(image_samples)

            duration = T / float(cfg.fps)
            intr_const = bool(np.allclose(intrinsic, intrinsic[0:1], atol=1e-4))
            results.append(
                EpisodeResult(
                    episode_index=record.episode_index,
                    length=T,
                    task=task,
                    instructions=[str(x) for x in (record.meta.get("instruction") or []) if str(x).strip()],
                    language=str(record.meta.get("language") or ""),
                    split=record.split,
                    clip_id=str(record.meta.get("clip_id") or record.key),
                    dataset_name=str(record.meta.get("dataset_name") or ""),
                    source_episode_index=record.source_episode_index,
                    source_frame_start=int(record.frame_ids[0]),
                    source_frames_contiguous=record.contiguous,
                    intrinsics9=intrinsic_matrix9(intrinsic[0]),
                    intrinsics_constant=intr_const,
                    presence_counts={str(k): int(v) for k, v in zip(*np.unique(presence, return_counts=True))},
                    video_from_timestamp=video_cursor if encoder is not None else None,
                    video_to_timestamp=(video_cursor + duration) if encoder is not None else None,
                    stats=_to_jsonable(ep_stats),
                    repaired_frames=[int(r) for r in repaired],
                    image_size=(int(width), int(height)),
                    source_media=(map_row.source_media if map_row is not None and map_row.usable else ""),
                    source_media_fps=(float(map_row.source_media_fps) if map_row is not None and map_row.usable else 0.0),
                    source_media_frame_start=(int(columns[SOURCE_FRAME_INDEX_KEY][0]) if cfg.source_map and map_row is not None and map_row.usable else -1),
                    source_media_frame_end=(int(columns[SOURCE_FRAME_INDEX_KEY][-1]) + 1 if cfg.source_map and map_row is not None and map_row.usable else -1),
                    source_media_undistort=(map_row.undistort if map_row is not None and map_row.usable else ""),
                    source_camera=(list(map_row.source_camera) if map_row is not None and map_row.usable else list(NAN_CAMERA)),
                )
            )
            video_cursor += duration
            global_index += T
        writer.close()
        if encoder is not None:
            encoder.close()
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        if encoder is not None:
            encoder.abort()
        for p in (tmp_data, tmp_video):
            if p.exists():
                p.unlink()
        raise
    finally:
        for handle in handles.values():
            handle.close()

    os.replace(tmp_data, data_path)
    frames = global_index - frame_start
    probed = None
    if encoder is not None:
        os.replace(tmp_video, video_path)
        probed = probe_video_frames(video_path, ffprobe=_ffprobe_for(cfg.ffmpeg), ffmpeg=cfg.ffmpeg)
        if probed is not None and probed != frames:
            raise RuntimeError(f"{video_path}: encoded {probed} frames, expected {frames}")
    return GroupResult(
        file_seq=file_seq,
        chunk_index=chunk_index,
        file_index=file_index,
        frames=frames,
        height=height,
        width=width,
        video_frames_probed=probed,
        episodes=results,
        keyframes_at_episode_starts=encoder is not None,
    )


def _ffprobe_for(ffmpeg: str) -> str:
    """Guess the ffprobe binary that ships next to the given ffmpeg."""
    path = Path(ffmpeg)
    candidate = path.with_name(path.name.replace("ffmpeg", "ffprobe")) if "ffmpeg" in path.name else None
    if candidate is not None and candidate.exists():
        return str(candidate)
    return "ffprobe"


def _claim_path(result_path: str | Path) -> Path:
    return Path(str(result_path)[: -len(".json")] + ".claim")


def _claim_is_stale(claim: Path, ttl_s: float) -> bool:
    """A claim is stale when it is older than ``ttl_s`` or was made on this host by a process that no longer exists."""
    try:
        text = claim.read_text(encoding="utf-8")
        age = time.time() - claim.stat().st_mtime
    except OSError:
        return True
    if age >= ttl_s:
        return True
    host, _, rest = text.partition(":")
    pid = rest.partition(":")[0]
    if host == socket.gethostname() and pid.isdigit():
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return True
        except OSError:
            pass
    return False


def _try_claim(result_path: str | Path, ttl_s: float) -> bool:
    """Claim one file group across machines through an exclusive create on the shared output directory.

    Several hosts may run the same conversion into one output directory (a primary plus ``--helper``
    runs); whoever creates ``group-N.claim`` first converts group N, the others move on.
    """
    claim = _claim_path(result_path)
    payload = f"{socket.gethostname()}:{os.getpid()}:{time.time():.0f}"
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        if not _claim_is_stale(claim, ttl_s):
            return False
        tmp = claim.with_name(claim.name + f".{socket.gethostname()}.{os.getpid()}")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, claim)
        time.sleep(1.0)  # let a concurrent taker's replace land, then see whose payload survived
        try:
            return claim.read_text(encoding="utf-8") == payload
        except OSError:
            return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return True


def _worker_entry(args):
    file_seq, frame_start, group_records, cfg, root, result_path = args
    if Path(result_path).exists():
        return "done elsewhere"
    if not _try_claim(result_path, cfg.claim_ttl_s):
        return "claimed elsewhere"
    if Path(result_path).exists():
        # another host finished the group and released its claim between the check above and our claim
        try:
            _claim_path(result_path).unlink()
        except OSError:
            pass
        return "done elsewhere"
    shard_paths = _WORKER_SHARED["shard_paths"]
    task_index_of = _WORKER_SHARED["task_index_of"]
    cache_dir = _WORKER_SHARED.get("cache_dir")
    for i, record in enumerate(group_records):
        record.shard_paths = shard_paths
        if not record.hydrated:
            group_records[i] = hydrate_record(record, shard_paths, cache_dir)
    result = convert_file_group(file_seq, frame_start, group_records, cfg, root, task_index_of, source_map=_worker_source_map(cfg.source_map))
    result.task_policy = str(_WORKER_SHARED.get("task_policy") or "")
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_result = f"{result_path}.{os.getpid()}.tmp"
    with open(tmp_result, "w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle)
    os.replace(tmp_result, result_path)
    try:
        _claim_path(result_path).unlink()
    except OSError:
        pass
    return "done"


_EPISODE_RESULT_FIELDS = None
_GROUP_RESULT_FIELDS = None


def _load_group_result(path: Path) -> GroupResult:
    """Read one file group's result, tolerating results written by an older converter.

    A long conversion outlives code changes: fields added later are absent from earlier results
    (they get a neutral value here and are back-filled from the index by ``_reconcile_group_results``),
    and fields this version no longer knows are ignored.
    """
    global _EPISODE_RESULT_FIELDS, _GROUP_RESULT_FIELDS
    if _EPISODE_RESULT_FIELDS is None:
        from dataclasses import fields as dc_fields

        _EPISODE_RESULT_FIELDS = {f.name for f in dc_fields(EpisodeResult)}
        _GROUP_RESULT_FIELDS = {f.name for f in dc_fields(GroupResult)}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    episodes = []
    for ep in raw.pop("episodes"):
        ep = {k: v for k, v in ep.items() if k in _EPISODE_RESULT_FIELDS}
        ep.setdefault("language", "")
        episodes.append(EpisodeResult(**ep))
    return GroupResult(episodes=episodes, **{k: v for k, v in raw.items() if k in _GROUP_RESULT_FIELDS})


def _rewrite_task_index(args) -> int:
    """Rewrite the ``task_index`` column of one data file (row groups and schema preserved). Returns rows touched."""
    data_path, mapping = args  # mapping: episode_index -> task_index
    # cheap check first: two tiny columns, not the whole file (a 100 MB data file stays unread when it is already right)
    probe = pq.read_table(str(data_path), columns=["episode_index", "task_index"])
    wanted_all = np.array([mapping[int(e)] for e in np.asarray(probe.column("episode_index").to_pylist()).reshape(-1)], dtype=np.int64)
    if np.array_equal(np.asarray(probe.column("task_index").to_pylist(), dtype=np.int64).reshape(-1), wanted_all):
        return 0
    source = pq.ParquetFile(str(data_path))
    schema = source.schema_arrow
    tmp = f"{data_path}.taskfix.{os.getpid()}.tmp"
    touched = 0
    writer = pq.ParquetWriter(tmp, schema=schema, compression="snappy", use_dictionary=True)
    try:
        for i in range(source.num_row_groups):
            table = source.read_row_group(i)
            episode_col = table.column("episode_index").to_numpy()
            wanted = np.array([mapping[int(e)] for e in episode_col.reshape(-1)], dtype=np.int64)
            current = table.column("task_index").to_numpy().reshape(-1)
            if not np.array_equal(current, wanted):
                touched += int(wanted.shape[0])
                idx = table.schema.get_field_index("task_index")
                table = table.set_column(idx, schema.field("task_index"), pa.array(wanted, type=schema.field("task_index").type))
            writer.write_table(table.cast(schema))
    finally:
        writer.close()
    if touched:
        os.replace(tmp, str(data_path))
    else:
        os.unlink(tmp)
    return touched


def _reconcile_group_results(group_results: list[GroupResult], result_paths: dict, records: list[EpisodeRecord], cfg: ConvertConfig, root: Path, task_index_of: dict[str, int], workers: int, task_policy: str) -> dict:
    """Bring file groups written under an earlier task policy / result schema in line with this run.

    The pixels and the state/action columns do not depend on the task policy, so such a group is
    repaired rather than reconverted: its ``task_index`` column is rewritten to the current
    vocabulary and its result record is updated. Returns a small report for provenance.
    """
    jobs = []
    stale_groups = []
    for gr in group_results:
        mapping = {}
        changed = gr.task_policy != task_policy  # same task names can still sit at other indices of another vocabulary
        for ep in gr.episodes:
            record = records[ep.episode_index]
            task = episode_task(record, cfg.task_source, cfg.task_name)
            language = str(record.meta.get("language") or "")
            mapping[int(ep.episode_index)] = int(task_index_of[task])
            if ep.task != task:
                ep.task = task
                changed = True
            if not ep.language and language:
                ep.language = language
            if tuple(ep.image_size) == (0, 0):
                ep.image_size = (int(record.width), int(record.height))
        if changed:
            data_path = root / DATA_PATH.format(chunk_index=gr.chunk_index, file_index=gr.file_index)
            jobs.append((str(data_path), mapping))
            stale_groups.append(gr)
    rows = 0
    if jobs:
        logger.info("%d file group(s) were written against another task vocabulary (or by an older converter); checking their task_index column", len(jobs))
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=_pool_context(), initializer=_worker_init) as pool:
                for n, touched in enumerate(pool.map(_rewrite_task_index, jobs, chunksize=8), 1):
                    rows += touched
                    if n % 5000 == 0:
                        logger.info("task_index rewritten in %d/%d file groups", n, len(jobs))
        else:
            rows = sum(_rewrite_task_index(job) for job in jobs)
        for gr in stale_groups:  # persist, so a later resume does not repeat the repair
            gr.task_policy = task_policy
            path = result_paths[gr.file_seq]
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(asdict(gr), handle)
            os.replace(tmp, path)
    unaligned = sum(1 for gr in group_results if not gr.keyframes_at_episode_starts)
    return {"task_index_checked_groups": len(jobs), "task_index_rewritten_rows": rows, "groups_without_forced_keyframes": unaligned}


# --------------------------------------------------------------------------------------
# Meta writers
# --------------------------------------------------------------------------------------


def write_tasks_parquet(tasks: list[str], root: Path, *, use_pandas: bool = True) -> None:
    """meta/tasks.parquet: pandas frame indexed by task string with a ``task_index`` column (lerobot layout)."""
    path = root / TASKS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if use_pandas:
        try:
            import pandas as pd

            frame = pd.DataFrame({"task_index": np.arange(len(tasks), dtype=np.int64)}, index=pd.Index(tasks, name="task"))
            frame.to_parquet(path)
            return
        except ImportError:
            pass
    # pandas-free fallback: same on-disk shape (index column "task" declared in pandas metadata)
    pandas_meta = {
        "index_columns": ["task"],
        "column_indexes": [{"name": None, "field_name": None, "pandas_type": "unicode", "numpy_type": "object", "metadata": {"encoding": "UTF-8"}}],
        "columns": [
            {"name": "task", "field_name": "task", "pandas_type": "unicode", "numpy_type": "object", "metadata": None},
            {"name": "task_index", "field_name": "task_index", "pandas_type": "int64", "numpy_type": "int64", "metadata": None},
        ],
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.2.0",
    }
    schema = pa.schema(
        [pa.field("task_index", pa.int64()), pa.field("task", pa.string())],
        metadata={b"pandas": json.dumps(pandas_meta).encode("utf-8")},
    )
    table = pa.Table.from_arrays([pa.array(np.arange(len(tasks), dtype=np.int64)), pa.array(tasks, type=pa.string())], schema=schema)
    pq.write_table(table, str(path))


def episode_row(ep: EpisodeResult, chunk_index: int, file_index: int, dataset_from: int, *, head_world2cam_identity: bool = True) -> dict:
    row = {
        "episode_index": ep.episode_index,
        "tasks": [ep.task],
        "length": ep.length,
        "data/chunk_index": chunk_index,
        "data/file_index": file_index,
        "dataset_from_index": dataset_from,
        "dataset_to_index": dataset_from + ep.length,
    }
    if ep.video_from_timestamp is not None:
        row.update(
            {
                f"videos/{HEAD_VIDEO_KEY}/chunk_index": chunk_index,
                f"videos/{HEAD_VIDEO_KEY}/file_index": file_index,
                f"videos/{HEAD_VIDEO_KEY}/from_timestamp": float(ep.video_from_timestamp),
                f"videos/{HEAD_VIDEO_KEY}/to_timestamp": float(ep.video_to_timestamp),
            }
        )
    row.update({
        "instructions": [x for x in ep.instructions if str(x).strip()],  # also cleans group results written before the filter
        "language": ep.language,
        "split": ep.split,
        "clip_id": ep.clip_id,
        "source_dataset": ep.dataset_name,
        "source_episode_index": ep.source_episode_index,
        "source_frame_start": ep.source_frame_start,
        SOURCE_MEDIA_KEY: ep.source_media,
        SOURCE_MEDIA_FPS_KEY: float(ep.source_media_fps),
        SOURCE_MEDIA_FRAME_START_KEY: int(ep.source_media_frame_start),
        SOURCE_MEDIA_FRAME_END_KEY: int(ep.source_media_frame_end),
        SOURCE_MEDIA_UNDISTORT_KEY: ep.source_media_undistort,
        INTRINSICS_EPISODE_KEY: [float(v) for v in ep.intrinsics9],
        IMAGE_SIZE_KEY: [int(ep.image_size[0]), int(ep.image_size[1])],
        SOURCE_CAMERA_KEY: [float(v) for v in ep.source_camera],
    })
    if head_world2cam_identity:
        row[HEAD_WORLD2CAM_EPISODE_KEY] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    for feature, stats in ep.stats.items():
        for stat_name, value in stats.items():
            row[f"stats/{feature}/{stat_name}"] = value
    return row


def write_episodes_parquet(rows: list[dict], root: Path, cfg: ConvertConfig) -> int:
    """Write meta/episodes files, rolling over by (uncompressed) size like the lerobot writer. Returns file count."""
    if not rows:
        raise ValueError("no episodes to write")
    limit = cfg.data_files_size_in_mb * 1024 * 1024
    file_seq = 0
    writer = None
    schema = None
    batch: list[dict] = []
    written_bytes = 0
    files = 0

    def flush():
        nonlocal writer, schema, batch, written_bytes, files
        if not batch:
            return
        for row in batch:
            chunk_index, file_index = chunk_file(file_seq, cfg.chunks_size)
            row["meta/episodes/chunk_index"] = chunk_index
            row["meta/episodes/file_index"] = file_index
        columns = {key: [row[key] for row in batch] for key in batch[0]}
        table = pa.Table.from_pydict(columns)
        if schema is None:
            # Pin string-list columns so an all-empty first batch cannot freeze them as list<null>.
            fields = []
            for pa_field in table.schema:
                if pa_field.name in ("tasks", "instructions"):
                    pa_field = pa.field(pa_field.name, pa.list_(pa.string()))
                fields.append(pa_field)
            schema = pa.schema(fields)
        table = table.select(schema.names).cast(schema)
        if writer is None:
            chunk_index, file_index = chunk_file(file_seq, cfg.chunks_size)
            path = root / EPISODES_PATH.format(chunk_index=chunk_index, file_index=file_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(str(path), schema=schema, compression="snappy", use_dictionary=True)
            files += 1
        writer.write_table(table)
        written_bytes += table.nbytes
        batch = []

    for row in rows:
        batch.append(row)
        if len(batch) >= 256:
            flush()
            if written_bytes >= limit:
                writer.close()
                writer = None
                file_seq += 1
                written_bytes = 0
    flush()
    if writer is not None:
        writer.close()
    return files


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)
        handle.write("\n")


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def collect_shards(inputs: list[str]) -> list[str]:
    paths: list[str] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found = list(iter_shard_paths(str(p)))
            if not found:  # nested layouts (e.g. BuildAI factory folders): search recursively
                found = sorted(str(q) for q in p.rglob("*.tar") if q.is_file())
            paths.extend(found)
        elif p.is_file():
            paths.append(str(p))
        else:
            raise FileNotFoundError(item)
    if not paths:
        raise FileNotFoundError(f"no .tar shards found under {inputs}")
    return paths


def _check_tools(cfg: ConvertConfig) -> None:
    if shutil.which(cfg.ffmpeg) is None and not Path(cfg.ffmpeg).exists():
        raise FileNotFoundError(f"ffmpeg not found: {cfg.ffmpeg!r} (pass --ffmpeg /path/to/ffmpeg)")
    probe = subprocess.run([cfg.ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    if cfg.vcodec not in probe.stdout:
        raise RuntimeError(f"ffmpeg at {cfg.ffmpeg!r} has no encoder {cfg.vcodec!r}")



def _run_file_groups(jobs: list, workers: int, shared: dict, *, heartbeat_s: float = HEARTBEAT_SECONDS) -> None:
    """Run file-group jobs on a spawned pool with a bounded number in flight and a stall heartbeat.

    Jobs are handed to the pool ``INFLIGHT_PER_WORKER`` deep rather than all at once, so the driver
    never holds thousands of pickled payloads, and a ``heartbeat`` line is logged whenever no group
    has finished for ``heartbeat_s`` seconds (the silence that previously looked like a hang).
    """
    total = len(jobs)
    if total == 0:
        return
    if workers <= 1 or total == 1:
        _worker_init(shared)
        for i, job in enumerate(jobs):
            status = _worker_entry(job)
            logger.info("file group %d %s (%d/%d)", job[0], status, i + 1, total)
        return

    max_workers = min(workers, total)
    started = time.monotonic()
    last_progress = started
    done = 0
    pending: dict = {}
    queue = iter(jobs)

    def fill(pool):
        while len(pending) < max_workers * INFLIGHT_PER_WORKER:
            job = next(queue, None)
            if job is None:
                return
            pending[pool.submit(_worker_entry, job)] = job[0]

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_pool_context(), initializer=_worker_init, initargs=(shared,)) as pool:
        fill(pool)
        while pending:
            finished, _ = wait(list(pending), timeout=heartbeat_s, return_when=FIRST_COMPLETED)
            for future in finished:
                seq = pending.pop(future)
                try:
                    status = future.result()
                except Exception as error:
                    logger.error("file group %d failed: %r", seq, error)
                    raise
                done += 1
                last_progress = time.monotonic()
                logger.info("file group %d %s (%d/%d)", seq, status, done, total)
            fill(pool)
            now = time.monotonic()
            if not finished or now - last_progress >= heartbeat_s:
                logger.info(
                    "heartbeat: %d/%d groups done, %d in flight, %.0f s since last completion, %.1f min elapsed, driver rss %.1f GB",
                    done, total, len(pending), now - last_progress, (now - started) / 60.0, rss_gb(),
                )


# ConvertConfig fields that change what an already-written file group contains (numbers, columns,
# timestamps, per-frame source indices). Task naming is deliberately absent: a task-policy change is
# repaired in place by _reconcile_group_results. Encoder settings only change video quality.
SEMANTIC_CONFIG_FIELDS = (
    "fps", "state_layout", "hand_frame", "pad_value",
    "include_mano", "include_extrinsic", "include_presence", "include_video",
    "source_map", "source_map_policy",
)


def _file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_set_digest(shard_paths: list[str]) -> str:
    """Identity of the input shard set: sorted (file name, size). Paths and mtimes are left out on
    purpose so a helper on another machine (different mount point, copied files) still matches."""
    entries = []
    for path in shard_paths:
        try:
            size = int(os.stat(path).st_size)
        except OSError:
            size = -1
        entries.append([Path(path).name, size])
    return hashlib.sha1(json.dumps(sorted(entries)).encode("utf-8")).hexdigest()


_SAMPLE_BLOCK = 64 << 10
_SAMPLE_BLOCKS = 16


def _sampled_content_sha1(path: str) -> str:
    """Cheap content identity of one shard: the whole file when small, otherwise _SAMPLE_BLOCKS blocks of
    64 KiB at evenly spaced offsets (always including the first and the last block). Machine-independent:
    only bytes, never the path or mtime."""
    size = int(os.stat(path).st_size)
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        if size <= _SAMPLE_BLOCK * _SAMPLE_BLOCKS:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        else:
            last = size - _SAMPLE_BLOCK
            for i in range(_SAMPLE_BLOCKS):
                fh.seek(last * i // (_SAMPLE_BLOCKS - 1))
                digest.update(fh.read(_SAMPLE_BLOCK))
    return digest.hexdigest()


def shard_content_digest(shard_paths: list[str]) -> str:
    """Identity of the input shard set by content: sorted (file name, size, sampled content hash). Like
    shard_set_digest it leaves paths out, so a copy on another machine still matches; unlike it, a shard
    rewritten with the same name and size (e.g. a changed copy under a new input root) no longer does."""
    entries = []
    for path in shard_paths:
        try:
            entries.append([Path(path).name, int(os.stat(path).st_size), _sampled_content_sha1(path)])
        except OSError:
            entries.append([Path(path).name, -1, None])
    return hashlib.sha1(json.dumps(sorted(entries, key=json.dumps)).encode("utf-8")).hexdigest()


def shard_mtimes(shard_paths: list[str]) -> dict[str, int]:
    """Per-path st_mtime_ns of the input shards. A shard rebuilt in place usually keeps its size, so
    (name, size) alone misses it; mtimes are compared only for paths both runs saw (the same file on
    the same or shared storage), so a copy under another path still matches by (name, size)."""
    mtimes = {}
    for path in shard_paths:
        try:
            mtimes[str(path)] = int(os.stat(path).st_mtime_ns)
        except OSError:
            mtimes[str(path)] = -1
    return mtimes


def _semantic_fields(values: dict) -> dict:
    """SEMANTIC_CONFIG_FIELDS normalised for comparison (shared by fingerprints and legacy plan configs)."""
    fields = {name: values[name] for name in SEMANTIC_CONFIG_FIELDS if name in values}
    if "pad_value" in fields:
        fields["pad_value"] = repr(float(fields["pad_value"]))  # NaN-safe comparison after a JSON round trip
    if "source_map" in fields:
        fields["source_map"] = bool(fields["source_map"])  # the file's identity below, not its (machine-specific) path
    return fields


def conversion_fingerprint(cfg: ConvertConfig, plan_digest: str, shard_paths: list[str] | None = None) -> dict:
    """Everything a resumed run must share with the run that wrote the existing file groups."""
    fields = _semantic_fields({name: getattr(cfg, name) for name in SEMANTIC_CONFIG_FIELDS})
    if cfg.source_map:
        try:
            # content, not mtime: a copy on another machine is the same map
            fields["source_map_file"] = {"size": int(os.stat(cfg.source_map).st_size), "sha1": _file_sha1(cfg.source_map)}
        except OSError:
            fields["source_map_file"] = None
    if shard_paths is not None:
        fields["input_shards"] = shard_set_digest(shard_paths)
        fields["input_shard_mtimes"] = shard_mtimes(shard_paths)
        fields["input_shard_content"] = shard_content_digest(shard_paths)
    fields["plan_digest"] = plan_digest
    return fields


def check_resume_fingerprint(existing_plan: dict, fingerprint: dict, *, plan_path: Path) -> bool:
    """Refuse to resume into groups written under different conversion semantics.

    Returns True when the existing plan carried a full fingerprint, False when only part of it could
    be checked (a plan written before fingerprints existed): the caller must then keep that plan
    instead of overwriting it with a fingerprint that was never verified against the file groups.
    """
    theirs = existing_plan.get("semantic_fingerprint")
    verified = isinstance(theirs, dict)
    if verified:
        ours = fingerprint
    else:
        # Legacy plan: compare what it did record (its config and group-layout digest) and nothing else.
        config = existing_plan.get("config")
        theirs = _semantic_fields(config) if isinstance(config, dict) else {}
        if existing_plan.get("digest"):
            theirs["plan_digest"] = existing_plan["digest"]
        ours = {k: fingerprint.get(k) for k in theirs}
        if not theirs:
            logger.warning("%s predates conversion fingerprints and records no config; cannot verify that resumed file groups match the current options", plan_path)
            return False
        logger.warning(
            "%s predates conversion fingerprints; checked only its recorded options (%s), not the input shard or source-map identity",
            plan_path, ", ".join(sorted(theirs)),
        )
    theirs, ours = dict(theirs), dict(ours)
    their_mtimes, our_mtimes = theirs.pop("input_shard_mtimes", None), ours.pop("input_shard_mtimes", None)
    their_content, our_content = theirs.pop("input_shard_content", None), ours.pop("input_shard_content", None)
    differing = sorted(k for k in set(theirs) | set(ours) if theirs.get(k) != ours.get(k))
    if their_content is not None and our_content is not None and their_content != our_content:
        # a fingerprint written before content digests were recorded has none: (name, size) and mtimes are all it can check
        differing.append("input_shard_content")
    detail_extra = []
    if isinstance(their_mtimes, dict) and isinstance(our_mtimes, dict):
        # a fingerprint written before mtimes were recorded has none: (name, size) is all it can check
        changed = sorted(p for p in set(their_mtimes) & set(our_mtimes) if their_mtimes[p] != our_mtimes[p])
        if changed:
            differing.append("input_shard_mtimes")
            detail_extra.append(f"input_shard_mtimes: {len(changed)} shard(s) modified since the existing groups were converted (e.g. {changed[:3]})")
    if differing:
        if "input_shard_content" in differing:
            detail_extra.append("input_shard_content: the input shards' contents differ from the ones the existing groups were converted from")
        detail = ", ".join([f"{k}: {theirs.get(k)!r} -> {ours.get(k)!r}" for k in differing if k not in ("input_shard_mtimes", "input_shard_content")] + detail_extra)
        raise RuntimeError(
            f"cannot resume: existing file groups under {plan_path.parent.parent} were converted with different options ({detail}); "
            "rerun with the original options, or use --overwrite / delete or change the output directory "
            "(the orchestrator's lerobot stage cannot pass --overwrite)"
        )
    return verified


def convert_wds_to_lerobot(inputs: list[str], output_dir: str, cfg: ConvertConfig) -> dict:
    cfg.validate()
    if cfg.include_video:
        _check_tools(cfg)
    root = Path(output_dir)
    work = root / WORK_DIR
    if cfg.helper and not (work / "plan.json").exists():
        # checked before indexing so an early helper fails fast instead of converting under its own options
        raise RuntimeError(f"helper started before the primary wrote {work / 'plan.json'}; wait for the primary to write plan.json, then start the helper")
    if root.exists() and any(root.iterdir()) and not cfg.resume:
        raise FileExistsError(f"{root} is not empty (pass --resume to continue a previous run or --overwrite)")
    root.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    shard_paths = collect_shards(inputs)
    workers = cfg.effective_workers
    logger.info("indexing %d shards with %d workers", len(shard_paths), min(workers, len(shard_paths)))
    index_issues: dict = {}
    records = index_shards(shard_paths, max_episodes=cfg.max_episodes, workers=workers, cache_dir=work / INDEX_CACHE_DIRNAME, report=index_issues, slim=True)
    if not records:
        raise ValueError("no episodes found")
    logger.info("index ready: %d episodes; driver rss %.1f GB", len(records), rss_gb())
    sizes = {(r.height, r.width) for r in records}
    if len(sizes) != 1:
        raise ValueError(f"mixed frame sizes across episodes are not supported: {sorted(sizes)}")
    height, width = records[0].height, records[0].width

    # MANO is optional in the shared wds spec (only this repo's own builder writes mano.npy). Decided once,
    # before any file is written: none at all -> no MANO feature; some episodes lack it -> skip those.
    dropped_episodes = None
    if cfg.include_mano:
        missing = [r for r in records if r.mano_missing]
        if len(missing) == len(records):
            if cfg.mano_policy == "require":
                raise ValueError("no episode has mano.npy; rerun with --no_mano or --mano_policy auto")
            logger.info("shards carry no mano.npy; exporting without the %s feature", MANO_KEY)
            cfg.include_mano = False
        elif missing:
            keys = [r.key for r in missing]
            msg = f"{len(missing)}/{len(records)} episodes have no mano.npy for some frames (e.g. {keys[:3]})"
            if cfg.mano_policy == "require":
                raise ValueError(msg + "; rerun with --no_mano or --mano_policy auto")
            logger.warning("%s; skipping those episodes (mano_policy=auto)", msg)
            dropped_episodes = {"reason": "missing_mano", "count": len(missing), "frames": int(sum(r.length for r in missing)), "keys": keys}
            records = [r for r in records if r.key not in set(keys)]
    source_mapping = None
    if cfg.source_map:
        smap = SourceMap.load(cfg.source_map)
        check_relative_source_media(smap)  # the map and its media paths ship with the dataset
        usable = {r.key for r in records if (row := source_map_row(smap, r)) is not None and row.usable}
        unmapped = [r for r in records if r.key not in usable]
        source_mapping = {
            "map": str(cfg.source_map),
            "policy": cfg.source_map_policy,
            "map_rows": len(smap),
            "map_status_counts": smap.status_counts(),
            "episodes_unmapped": len(unmapped),
            "episodes_unmapped_frames": int(sum(r.length for r in unmapped)),
            "unmapped_keys_sample": [r.key for r in unmapped[:20]],
        }
        if unmapped and cfg.source_map_policy == "drop":
            logger.warning("%d/%d episodes have no usable source-map row; dropping them (source_map_policy=drop)", len(unmapped), len(records))
            drop = {r.key for r in unmapped}
            records = [r for r in records if r.key not in drop]
            if not records:
                raise ValueError("every episode was dropped: no usable source-map rows")
        elif unmapped:
            logger.warning("%d/%d episodes have no usable source-map row; kept with source_frame_index=-1 (source_map_policy=keep)", len(unmapped), len(records))
        source_mapping["episodes_mapped"] = len([r for r in records if r.key in usable])
        source_mapping["coverage"] = source_mapping["episodes_mapped"] / max(1, len(records))
    records = order_episodes(records, cfg.split_order)

    tasks: list[str] = []
    task_index_of: dict[str, int] = {}
    for record in records:
        task = episode_task(record, cfg.task_source, cfg.task_name)
        if task not in task_index_of:
            task_index_of[task] = len(tasks)
            tasks.append(task)

    groups = plan_file_groups(records, cfg)
    total_frames = sum(r.length for r in records)
    logger.info("%d episodes, %d frames, %d tasks -> %d file pairs", len(records), total_frames, len(tasks), len(groups))

    plan = {
        "config": asdict(cfg),
        "shards": shard_paths,
        "episodes": [
            {"episode_index": r.episode_index, "key": r.key, "split": r.split, "length": r.length, "source_episode_index": r.source_episode_index}
            for r in records
        ],
        "groups": [{"file_seq": g.file_seq, "episodes": g.episodes, "frame_start": g.frame_start} for g in groups],
        "tasks": tasks,
    }
    plan["digest"] = hashlib.sha1(json.dumps([[g.file_seq, g.frame_start, [records[p].key for p in g.episodes]] for g in groups]).encode("utf-8")).hexdigest()
    plan["semantic_fingerprint"] = conversion_fingerprint(cfg, plan["digest"], shard_paths)
    plan_path = work / "plan.json"
    if cfg.helper:
        if not cfg.resume:
            raise ValueError("helper runs join an existing conversion: resume must be on")
        if plan_path.exists():
            existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            theirs = existing_plan.get("digest")
            if theirs and theirs != plan["digest"]:
                raise RuntimeError("helper plan differs from the primary's plan.json (different code version, shard list or options); refusing to write into its output")
            check_resume_fingerprint(existing_plan, plan["semantic_fingerprint"], plan_path=plan_path)
        else:
            raise RuntimeError(f"helper started before the primary wrote {plan_path}; wait for the primary to write plan.json, then start the helper")
    else:
        keep_existing_plan = False
        if cfg.resume and plan_path.exists() and any((work / "groups").glob("group-*.json")):
            # an unverifiable legacy plan stays in place: overwriting it would bless the existing groups
            keep_existing_plan = not check_resume_fingerprint(json.loads(plan_path.read_text(encoding="utf-8")), plan["semantic_fingerprint"], plan_path=plan_path)
        if not keep_existing_plan:
            write_json(plan, plan_path)

    result_paths = {g.file_seq: work / "groups" / f"group-{g.file_seq:06d}.json" for g in groups}
    todo = [g for g in groups if not (cfg.resume and result_paths[g.file_seq].exists())]
    if cfg.helper:
        todo = todo[::-1]  # work from the far end so primary and helper rarely reach for the same group
    logger.info("converting %d/%d file groups (%d already done); driver rss %.1f GB", len(todo), len(groups), len(groups) - len(todo), rss_gb())

    def _job(g):
        light = []
        for p in g.episodes:
            r = records[p]
            light.append(EpisodeRecord(**{**r.__dict__, "shard_paths": None}))  # shard table travels once per worker
        return (g.file_seq, g.frame_start, light, cfg, root, str(result_paths[g.file_seq]))

    task_policy = hashlib.sha1(json.dumps([cfg.task_source, cfg.task_name, tasks], ensure_ascii=False).encode("utf-8")).hexdigest()
    shared = {"shard_paths": shard_paths, "task_index_of": task_index_of, "cache_dir": str(work / INDEX_CACHE_DIRNAME), "task_policy": task_policy}
    _run_file_groups([_job(g) for g in todo], workers, shared)

    def _missing():
        return [g for g in groups if not result_paths[g.file_seq].exists()]

    missing = _missing()
    if cfg.helper:
        logger.info("helper finished its pass; %d file group(s) still open on other machines. Meta is written by the primary.", len(missing))
        return {"output_dir": str(root), "helper": True, "file_pairs": len(groups), "groups_open": len(missing)}
    waited = 0
    while missing:  # groups claimed by a helper (or by a worker that died): wait, and take over whatever went stale
        takeover = [g for g in missing if _claim_is_stale(_claim_path(result_paths[g.file_seq]), cfg.claim_ttl_s)]
        if takeover:
            logger.info("taking over %d file group(s) with no live claim", len(takeover))
            _run_file_groups([_job(g) for g in takeover], workers, shared)
        else:
            if waited % 10 == 0:
                logger.info("waiting for %d file group(s) claimed by another machine", len(missing))
            time.sleep(60)
            waited += 1
        missing = _missing()

    logger.info("loading %d file-group results", len(groups))
    with ThreadPoolExecutor(max_workers=32) as pool:  # tens of thousands of small files on a network mount: latency-bound
        group_results = list(pool.map(lambda g: _load_group_result(result_paths[g.file_seq]), groups))
    logger.info("file-group results loaded; driver rss %.1f GB", rss_gb())
    reconciled = _reconcile_group_results(group_results, result_paths, records, cfg, root, task_index_of, workers, task_policy)
    if reconciled["groups_without_forced_keyframes"] and cfg.include_video:
        logger.warning(
            "%d of %d file groups were encoded before keyframes were forced at episode starts; provenance will not promise aligned keyframes",
            reconciled["groups_without_forced_keyframes"], len(group_results),
        )

    # ---- meta ----
    rows: list[dict] = []
    per_episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
    cursor = 0
    non_constant_intrinsics = 0
    gapped_episodes = 0
    repaired_frames: dict[str, list[int]] = {}
    for gr in group_results:
        for ep in gr.episodes:
            if ep.episode_index != len(rows):
                raise RuntimeError(f"episode order mismatch: got {ep.episode_index}, expected {len(rows)}")
            if ep.repaired_frames:
                # repeated episodes share a clip_id; later ones are keyed by their own episode key
                name = ep.clip_id if ep.clip_id not in repaired_frames else records[ep.episode_index].key
                repaired_frames[name] = list(ep.repaired_frames)
            rows.append(episode_row(ep, gr.chunk_index, gr.file_index, cursor, head_world2cam_identity=(cfg.hand_frame == "camera")))
            per_episode_stats.append(_stats_from_lists(ep.stats))
            cursor += ep.length
            non_constant_intrinsics += 0 if ep.intrinsics_constant else 1
            gapped_episodes += 0 if ep.source_frames_contiguous else 1
    if cursor != total_frames:
        raise RuntimeError(f"frame accounting mismatch: wrote {cursor}, planned {total_frames}")
    if non_constant_intrinsics:
        logger.warning("%d episodes have non-constant intrinsics; %s stores the first frame's values", non_constant_intrinsics, INTRINSICS_EPISODE_KEY)
    if repaired_frames:
        logger.warning("%d episode(s) had broken JPEG frames replaced by a neighbouring frame (%d frames in total); listed in provenance", len(repaired_frames), sum(len(v) for v in repaired_frames.values()))
    if gapped_episodes:
        logger.warning("%d episodes have gaps in their source frame ids; timestamps assume consecutive frames at %d fps", gapped_episodes, cfg.fps)

    write_tasks_parquet(tasks, root)
    episodes_files = write_episodes_parquet(rows, root, cfg)
    stats = aggregate_stats(per_episode_stats)
    write_json(_to_jsonable(stats), root / STATS_PATH)

    features = build_features(cfg, height, width)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": cfg.robot_type,
        "total_episodes": len(records),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": cfg.chunks_size,
        "data_files_size_in_mb": cfg.data_files_size_in_mb,
        "video_files_size_in_mb": cfg.video_files_size_in_mb,
        "fps": int(cfg.fps),
        "splits": build_splits(records),
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH if cfg.include_video else None,
        "features": features,
    }
    write_json(info, root / INFO_PATH)
    source_frames = None
    if cfg.descriptor_manifest:
        source_frames = write_source_frame_index(records, cfg.descriptor_manifest, root, shard_paths=shard_paths, cache_dir=work / INDEX_CACHE_DIRNAME)
    # EgoSmith provenance lives next to info.json so the lerobot loader sees only known keys.
    provenance = {
        "source": "egosmith_webdataset",
        "state_layout": cfg.state_layout,
        "hand_frame": cfg.hand_frame,
        "robot_pad_dims": [0, EGOSTEER_ROBOT_PAD_DIM] if cfg.state_layout == "egosteer74" else None,
        "pad_value": cfg.pad_value,
        "rot6d": "first two rotation-matrix columns, column-major [R00,R10,R20,R01,R11,R21]",
        "wrist_translation_semantics": "mano_joint_0",
        "camera_extrinsic_convention": "w2c (row-major 4x4, points as column vectors)",
        "action_semantics": (
            "next-frame state expressed in the current frame's reference camera"
            if cfg.hand_frame == "camera"
            else "next-frame state expressed in the episode's SLAM world frame (same frame as state)"
        ),
        "mano_schema": (records[0].meta.get("mano_schema") if cfg.include_mano else None),
        "lowdim_schema": records[0].meta.get("lowdim_schema"),
        "task_source": cfg.task_source,
        "task_name": cfg.task_name,
        "video_encoder": {"tool": "ffmpeg-cli", "vcodec": cfg.vcodec, "crf": cfg.crf, "g": cfg.gop, "preset": cfg.preset, "pix_fmt": cfg.pix_fmt, "color_range": "tv", "keyframes_at_episode_starts": reconciled["groups_without_forced_keyframes"] == 0} if cfg.include_video else None,
        "reconciled_groups": reconciled,
        "include_video": cfg.include_video,
        "include_mano": cfg.include_mano,
        "dropped_episodes": dropped_episodes,
        "index_issues": _shareable_index_issues(index_issues),
        "repaired_frames": repaired_frames,
        "source_frames_index": source_frames,
        "source_mapping": source_mapping,
        "shards": shard_paths,
    }
    # the provenance ships with the dataset: no local absolute paths (shards, source map, unreadable shards)
    write_json(shareable_paths(provenance), root / "meta" / "egosmith_provenance.json")
    if cfg.source_map:
        shutil.copyfile(cfg.source_map, root / "meta" / "source_map.parquet")

    summary = {
        "output_dir": str(root),
        "episodes": len(records),
        "frames": total_frames,
        "tasks": len(tasks),
        "file_pairs": len(groups),
        "episodes_meta_files": episodes_files,
        "splits": info["splits"],
        "frame_size": [height, width],
        "include_video": cfg.include_video,
        "include_mano": cfg.include_mano,
        "dropped_episodes": None if dropped_episodes is None else {k: v for k, v in dropped_episodes.items() if k != "keys"},
        "index_issues": {k: (v if k == "unreadable_shards" else {kk: vv for kk, vv in v.items() if kk != "keys"}) for k, v in index_issues.items()},
        "repaired_frames": {"episodes": len(repaired_frames), "frames": sum(len(v) for v in repaired_frames.values())},
        "source_frames_index": source_frames,
        "source_mapping": None if source_mapping is None else {k: v for k, v in source_mapping.items() if k != "unmapped_keys_sample"},
        "workers": workers,
        "non_constant_intrinsics_episodes": non_constant_intrinsics,
    }
    write_json(summary, work / "summary.json")
    return summary


# --------------------------------------------------------------------------------------
# Rehydration index: episode -> source clip / frame names (from the frozen clip manifest)
# --------------------------------------------------------------------------------------

SOURCE_FRAMES_PATH = "meta/source_frames.parquet"


def _shareable_index_issues(index_issues: dict) -> dict:
    """``index_issues`` for the shipped provenance: an unreadable shard's ``error`` (an exception
    repr that may embed local paths) is reduced to the exception class name; the full text stays in
    the local log and work/summary.json."""
    out = dict(index_issues)
    if out.get("unreadable_shards"):
        out["unreadable_shards"] = [
            {**item, "error": str(item.get("error") or "").split("(", 1)[0].strip() or "error"}
            for item in out["unreadable_shards"]
        ]
    return out


def shareable_paths(value):
    """Return ``value`` with every absolute-path string (recursively inside dicts/lists) reduced to its
    base name, so files that ship with a dataset do not leak local directory layouts."""
    if isinstance(value, dict):
        return {k: shareable_paths(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [shareable_paths(v) for v in value]
    if isinstance(value, str) and (
        os.path.isabs(value) or value.startswith(("~", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        return os.path.basename(value.rstrip("/\\").replace("\\", "/"))
    return value


def write_source_frame_index(records: list[EpisodeRecord], manifest_path: str, root: Path, *, shard_paths: list[str] | None = None, cache_dir: str | Path | None = None) -> str:
    """Write meta/source_frames.parquet: per episode, the clip descriptor fields needed to re-obtain the
    frames from the original data (clip name, media file name, fps, frame names). Local absolute
    paths are reduced to base names so the file is shareable. ``frame_index_in_clip`` lists, per
    episode row, the clip frame id (an index into ``frame_names``), so episodes with gaps in their
    source frame ids are rehydrated frame-exact. Slim, gapped records are re-read from the shard index
    cache (``shard_paths``/``cache_dir``)."""
    from lib.pipeline.clips.clip_manifest import load_clip_manifest

    by_clip = {rec.clip_id: rec for rec in load_clip_manifest(manifest_path)}
    columns = ("episode_index", "clip_id", "source_id", "clip_name", "media_name", "storage_kind", "fps", "width", "height", "frame_count", "frame_names", "frame_start_name", "frame_index_in_clip", "descriptor_extra")
    rows = {k: [] for k in columns}
    missing = []
    for rec in records:
        clip_id = str(rec.meta.get("clip_id") or rec.key)
        m = by_clip.get(clip_id)
        if m is None:
            missing.append(clip_id)
            continue
        d = m.descriptor
        names = [os.path.basename(n) for n in (d.frame_names or [])]
        first = rec.first_frame_id
        frame_ids = getattr(rec, "frame_ids", None)
        if frame_ids is None and hasattr(rec, "length"):  # a record without length info leaves the entry null (rehydrate falls back)
            if rec.contiguous:
                frame_ids = np.arange(first, first + rec.length, dtype=np.int64)
            elif shard_paths is not None:
                frame_ids = hydrate_record(rec, shard_paths, cache_dir).frame_ids
            else:
                raise ValueError(f"episode {rec.key!r} has gaps in its frame ids but is a slim record: pass shard_paths to recover them")
        rows["episode_index"].append(int(rec.episode_index))
        rows["clip_id"].append(clip_id)
        rows["source_id"].append(str(m.source_id))
        rows["clip_name"].append(shareable_paths(str(d.clip_name)))
        rows["media_name"].append(os.path.basename(d.media_path) if d.media_path else "")
        rows["storage_kind"].append(str(d.storage_kind))
        rows["fps"].append(float(d.fps) if d.fps else float("nan"))
        rows["width"].append(int(d.width) if d.width else -1)
        rows["height"].append(int(d.height) if d.height else -1)
        rows["frame_count"].append(int(d.frame_count))
        rows["frame_names"].append(names)
        rows["frame_start_name"].append(names[first] if first < len(names) else "")
        rows["frame_index_in_clip"].append(None if frame_ids is None else [int(i) for i in frame_ids])
        rows["descriptor_extra"].append(json.dumps(shareable_paths(d.extra or {}), ensure_ascii=False))
    if missing:
        raise ValueError(f"{len(missing)} episodes not found in {manifest_path} (e.g. {missing[:3]})")
    schema = pa.schema(
        [
            ("episode_index", pa.int64()), ("clip_id", pa.string()), ("source_id", pa.string()), ("clip_name", pa.string()),
            ("media_name", pa.string()), ("storage_kind", pa.string()), ("fps", pa.float64()), ("width", pa.int64()),
            ("height", pa.int64()), ("frame_count", pa.int64()), ("frame_names", pa.list_(pa.string())),
            ("frame_start_name", pa.string()), ("frame_index_in_clip", pa.list_(pa.int64())), ("descriptor_extra", pa.string()),
        ]
    )
    table = pa.Table.from_pydict(rows).cast(schema)
    path = root / SOURCE_FRAMES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
    return SOURCE_FRAMES_PATH


# --------------------------------------------------------------------------------------
# Validation (pyarrow-level; optional lerobot loader)
# --------------------------------------------------------------------------------------


def validate_lerobot_dataset(
    root: str, *, ffprobe: str = "ffprobe", ffmpeg: str = "ffmpeg", try_lerobot: bool = True, num_items: int = 8
) -> dict:
    root_path = Path(root)
    info = json.loads((root_path / INFO_PATH).read_text(encoding="utf-8"))
    problems: list[str] = []

    episodes_tables = [pq.read_table(p) for p in sorted((root_path / "meta" / "episodes").glob("*/*.parquet"))]
    if not episodes_tables:
        raise FileNotFoundError("no meta/episodes parquet files")
    episodes = pa.concat_tables(episodes_tables).to_pydict()
    n_eps = len(episodes["episode_index"])
    if n_eps != info["total_episodes"]:
        problems.append(f"episodes rows {n_eps} != info.total_episodes {info['total_episodes']}")
    if episodes["episode_index"] != list(range(n_eps)):
        problems.append("episode_index is not 0..N-1 in order")
    expected_from = 0
    for i in range(n_eps):
        if episodes["dataset_from_index"][i] != expected_from:
            problems.append(f"episode {i}: dataset_from_index {episodes['dataset_from_index'][i]} != {expected_from}")
            break
        expected_from = episodes["dataset_to_index"][i]
    if expected_from != info["total_frames"]:
        problems.append(f"dataset_to_index end {expected_from} != info.total_frames {info['total_frames']}")

    frames_per_file: dict[tuple[int, int], int] = {}
    for i in range(n_eps):
        key = (episodes["data/chunk_index"][i], episodes["data/file_index"][i])
        frames_per_file[key] = frames_per_file.get(key, 0) + episodes["length"][i]
    video_keys = [k for k, ft in info["features"].items() if ft["dtype"] == "video" and info.get("video_path")]
    # every episode must start on a keyframe (readers seek to from_timestamp); enforced for datasets encoded
    # with that guarantee, reported as a note for older ones
    prov_path = root_path / "meta" / "egosmith_provenance.json"
    keyframes_promised = False
    if prov_path.exists():
        enc = (json.loads(prov_path.read_text(encoding="utf-8")).get("video_encoder") or {})
        keyframes_promised = bool(enc.get("keyframes_at_episode_starts"))
    starts_per_file: dict[tuple[int, int], list[float]] = {}
    for video_key in video_keys[:1]:
        col = f"videos/{video_key}/from_timestamp"
        if col in episodes:
            for i in range(n_eps):
                starts_per_file.setdefault((episodes["data/chunk_index"][i], episodes["data/file_index"][i]), []).append(float(episodes[col][i]))
    fps = float(info["fps"])

    def _check_file(item):
        (chunk_index, file_index), frames = item
        out = {"problems": [], "rows": 0, "first": None, "last": None}
        path = root_path / info["data_path"].format(chunk_index=chunk_index, file_index=file_index)
        if not path.exists():
            out["problems"].append(f"missing {path}")
            return out
        table = pq.read_table(path, columns=["index", "episode_index"])
        if table.num_rows != frames:
            out["problems"].append(f"{path}: {table.num_rows} rows, episodes say {frames}")
        idx = table.column("index").to_numpy()
        if idx.shape[0]:
            if not np.all(np.diff(idx) == 1):
                out["problems"].append(f"{path}: global index not contiguous")
            out["first"], out["last"] = int(idx[0]), int(idx[-1])
        out["rows"] = table.num_rows
        for video_key in video_keys:
            vpath = root_path / info["video_path"].format(video_key=video_key, chunk_index=chunk_index, file_index=file_index)
            if not vpath.exists():
                out["problems"].append(f"missing {vpath}")
                continue
            probed = probe_video_frames(vpath, ffprobe=ffprobe, ffmpeg=ffmpeg)
            if probed is None:
                out["problems"].append(f"{vpath}: could not count frames (no ffprobe/ffmpeg)")
            elif probed != frames:
                out["problems"].append(f"{vpath}: {probed} frames, expected {frames}")
            declared = (info["features"][video_key].get("info") or {}).get("video.pix_fmt")
            actual = probe_video_pix_fmt(vpath, ffprobe=ffprobe, ffmpeg=ffmpeg)
            if declared and actual and actual != declared:
                out["problems"].append(f"{vpath}: pixel format {actual!r} != declared {declared!r} (full-range mjpeg leaked through?)")
            starts = starts_per_file.get((chunk_index, file_index)) or []
            if starts:
                key_times = probe_keyframe_times(vpath, ffmpeg=ffmpeg)
                if key_times is not None:
                    key_frames = {int(round(t * fps)) for t in key_times}
                    off = [t for t in starts if int(round(t * fps)) not in key_frames]
                    if off:
                        out["keyframe_misses"] = len(off)
                        if keyframes_promised:
                            out["problems"].append(f"{vpath}: {len(off)}/{len(starts)} episodes do not start on a keyframe (first at {off[0]:.3f}s)")
        return out

    items = sorted(frames_per_file.items())
    with ThreadPoolExecutor(max_workers=min(32, max(1, len(items)))) as pool:
        results = list(pool.map(_check_file, items))
    total_rows = 0
    prev_last = -1
    keyframe_misses = sum(int(out.get("keyframe_misses", 0)) for out in results)
    for out in results:
        problems.extend(out["problems"])
        if out["first"] is not None:
            if out["first"] != prev_last + 1:
                problems.append(f"global index not contiguous across files at index {out['first']} (expected {prev_last + 1})")
            prev_last = out["last"]
        total_rows += out["rows"]
    if total_rows != info["total_frames"]:
        problems.append(f"parquet rows {total_rows} != info.total_frames {info['total_frames']}")

    # hard requirement: no depth of any kind leaves the converter
    for key, ft in info["features"].items():
        ft_info = ft.get("info") or {}
        if "depth" in key.lower() or ft_info.get("is_depth_map") or ft_info.get("video.is_depth_map"):
            problems.append(f"depth feature {key!r} present; LeRobot export must not carry depth")
    data_cols = pq.read_schema(root_path / info["data_path"].format(chunk_index=0, file_index=0)).names
    depth_cols = [c for c in data_cols if "depth" in c.lower()]
    if depth_cols:
        problems.append(f"depth columns in data parquet: {depth_cols}")
    depth_dirs = [str(p) for p in (root_path / "videos").glob("*depth*")] if (root_path / "videos").exists() else []
    if depth_dirs:
        problems.append(f"depth video directories present: {depth_dirs}")

    tasks = pq.read_table(root_path / TASKS_PATH).to_pydict()
    if len(tasks.get("task_index", [])) != info["total_tasks"]:
        problems.append(f"tasks rows {len(tasks.get('task_index', []))} != info.total_tasks {info['total_tasks']}")

    lerobot_check = None
    if try_lerobot:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

            ds = LeRobotDataset(repo_id=f"local/{root_path.name}", root=str(root_path))
            picks = np.linspace(0, len(ds) - 1, num=min(num_items, len(ds))).astype(int).tolist()
            shapes = {}
            for i in picks:
                item = ds[i]
                for k, v in item.items():
                    if hasattr(v, "shape"):
                        shapes.setdefault(k, tuple(v.shape))
            lerobot_check = {"num_frames": len(ds), "num_episodes": ds.num_episodes, "sampled": picks, "shapes": {k: list(v) for k, v in shapes.items()}}
        except ImportError:
            lerobot_check = "lerobot not installed; skipped loader check"
        except Exception as error:  # surface loader failures as problems, not crashes
            problems.append(f"lerobot loader failed: {error!r}")

    notes = []
    if keyframe_misses and not keyframes_promised:
        notes.append(f"{keyframe_misses} episodes do not start on a keyframe (dataset encoded before keyframes were aligned to episode starts; re-encode to fix)")
    return {"ok": not problems, "problems": problems, "notes": notes, "lerobot": lerobot_check, "episodes": n_eps, "frames": total_rows}


__all__ = [
    "CODEBASE_VERSION",
    "SOURCE_FRAMES_PATH",
    "HEAD_WORLD2CAM_EPISODE_KEY",
    "probe_video_pix_fmt",
    "write_source_frame_index",
    "ConvertConfig",
    "EpisodeReader",
    "EpisodeRecord",
    "INDEX_CACHE_DIRNAME",
    "FileGroup",
    "build_features",
    "convert_file_group",
    "convert_wds_to_lerobot",
    "egosteer74_names",
    "hand_block_from_lowdim",
    "index_shards",
    "jpeg_dimensions",
    "lowdim_to_state_action",
    "order_episodes",
    "plan_file_groups",
    "validate_lerobot_dataset",
]
