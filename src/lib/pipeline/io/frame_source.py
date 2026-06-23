"""Frame source abstraction: lazy, cached frame decoding (OpenCV / optional TurboJPEG) that the pipeline stages read instead of touching image files directly."""

import cv2
import numpy as np
import os
import threading
from collections import OrderedDict
from pathlib import Path

import torch
import torch.utils.data

# Try to import turbojpeg for faster JPEG decoding
try:
    from turbojpeg import TurboJPEG
    TURBOJPEG_AVAILABLE = True
except ImportError:
    TURBOJPEG_AVAILABLE = False

from lib.pipeline.proc.logging_setup import QUIET_MODE  # noqa: F401
SHARD_MEMBER_INDEX_CACHE_SIZE = max(1, int(os.environ.get("HAWOR_SHARD_MEMBER_INDEX_CACHE_SIZE", "8")))
_SHARD_MEMBER_INDEX_LOCK = threading.Lock()
_SHARD_MEMBER_INDEX_CACHE = OrderedDict()


def _load_shard_member_index(tar_path: str):
    with _SHARD_MEMBER_INDEX_LOCK:
        cached = _SHARD_MEMBER_INDEX_CACHE.get(tar_path)
        if cached is not None:
            _SHARD_MEMBER_INDEX_CACHE.move_to_end(tar_path)
            return cached

    member_index = {}
    with __import__("tarfile").open(tar_path, "r") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue
            member_index[member.name] = (int(member.offset_data), int(member.size))

    with _SHARD_MEMBER_INDEX_LOCK:
        _SHARD_MEMBER_INDEX_CACHE[tar_path] = member_index
        _SHARD_MEMBER_INDEX_CACHE.move_to_end(tar_path)
        while len(_SHARD_MEMBER_INDEX_CACHE) > SHARD_MEMBER_INDEX_CACHE_SIZE:
            _SHARD_MEMBER_INDEX_CACHE.popitem(last=False)
    return member_index


class BaseFrameSource:
    def __len__(self):
        raise NotImplementedError

    def get_frame(self, index: int, rgb: bool = False):
        raise NotImplementedError

    def get_frame_bytes(self, index: int):
        raise NotImplementedError

    def iter_frames(self, rgb: bool = False):
        for idx in range(len(self)):
            yield idx, self.get_frame(idx, rgb=rgb)

    def get_size(self):
        frame = self.get_frame(0, rgb=False)
        h, w = frame.shape[:2]
        return h, w


class ImageFolderFrameSource(BaseFrameSource):
    def __init__(self, image_paths, use_turbojpeg=True):
        self.image_paths = list(image_paths)

        if not QUIET_MODE:
            print(f"ImageFolderFrameSource: {len(self.image_paths)} frames")

        if len(self.image_paths) == 0:
            raise RuntimeError("ImageFolderFrameSource requires non-empty image_paths")

        self.use_turbojpeg = use_turbojpeg and TURBOJPEG_AVAILABLE
        self._thread_local = threading.local()

    def _get_jpeg_decoder(self):
        if not self.use_turbojpeg:
            return None
        decoder = getattr(self._thread_local, 'jpeg_decoder', None)
        if decoder is None:
            decoder = TurboJPEG()
            self._thread_local.jpeg_decoder = decoder
        return decoder

    def __len__(self):
        return len(self.image_paths)

    def get_frame(self, index: int, rgb: bool = False):
        if index < 0 or index >= len(self.image_paths):
            raise IndexError(
                f"Frame index {index} out of range [0, {len(self.image_paths)}). "
                f"Total frames available: {len(self.image_paths)}"
            )

        path = self.image_paths[index]

        # Use turbojpeg for JPEG files if available (2-3x faster than cv2.imread)
        if self.use_turbojpeg and path.lower().endswith(('.jpg', '.jpeg')):
            try:
                with open(path, 'rb') as f:
                    jpeg_data = f.read()
                decoder = self._get_jpeg_decoder()
                if rgb:
                    frame = decoder.decode(jpeg_data, pixel_format=0)  # RGB
                else:
                    frame = decoder.decode(jpeg_data, pixel_format=1)  # BGR
                return frame
            except Exception:
                pass

        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {path}")
        if rgb:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_frame_bytes(self, index: int):
        if index < 0 or index >= len(self.image_paths):
            raise IndexError(
                f"Frame index {index} out of range [0, {len(self.image_paths)}). "
                f"Total frames available: {len(self.image_paths)}"
            )
        return Path(self.image_paths[index]).read_bytes()


class ShardVideoFrameSource(BaseFrameSource):
    """Frame source that reads a single video's frames from a WebDataset tar shard.

    Uses pre-computed byte offsets to read frames via direct seek+read,
    bypassing tarfile's expensive member index scanning entirely.
    """

    def __init__(self, tar_path, frame_names, frame_offsets=None, use_turbojpeg=True):
        """
        Args:
            tar_path: Path to the tar shard containing this video's frames.
            frame_names: Sorted list of JPEG filenames within the tar for this video.
            frame_offsets: List of [offset, size] pairs parallel to frame_names.
                          If provided, uses direct seek+read (fast path).
                          If None, falls back to tarfile (legacy path).
            use_turbojpeg: Use TurboJPEG for faster decoding if available.
        """
        self.tar_path = tar_path
        self.frame_names = list(frame_names)
        self.frame_offsets = frame_offsets  # [[offset, size], ...]

        if len(self.frame_names) == 0:
            raise RuntimeError(f"ShardVideoFrameSource requires non-empty frame_names for {tar_path}")

        if not QUIET_MODE:
            mode = "direct-seek" if frame_offsets else "indexed-pread"
            print(f"ShardVideoFrameSource: {len(self.frame_names)} frames from {os.path.basename(tar_path)} ({mode})")

        self.use_turbojpeg = use_turbojpeg and TURBOJPEG_AVAILABLE
        self._thread_local = threading.local()

        # Shared fd is safe with os.pread() because it does not mutate file offset.
        self._fd = None

    def _get_fd(self):
        if self._fd is None:
            self._fd = os.open(self.tar_path, os.O_RDONLY)
        return self._fd

    def _get_jpeg_decoder(self):
        if not self.use_turbojpeg:
            return None
        decoder = getattr(self._thread_local, 'jpeg_decoder', None)
        if decoder is None:
            decoder = TurboJPEG()
            self._thread_local.jpeg_decoder = decoder
        return decoder

    def _get_tar(self):
        tar = getattr(self._thread_local, 'tar', None)
        if tar is None:
            import tarfile
            tar = tarfile.open(self.tar_path, 'r')
            self._thread_local.tar = tar
        return tar

    def __len__(self):
        return len(self.frame_names)

    def get_frame(self, index: int, rgb: bool = False):
        if index < 0 or index >= len(self.frame_names):
            raise IndexError(
                f"Frame index {index} out of range [0, {len(self.frame_names)}). "
                f"Total frames available: {len(self.frame_names)}"
            )

        member_name, jpeg_data = self._read_member_bytes(index)

        if self.use_turbojpeg and member_name.lower().endswith(('.jpg', '.jpeg')):
            try:
                pixel_format = 0 if rgb else 1  # RGB=0, BGR=1
                decoder = self._get_jpeg_decoder()
                frame = decoder.decode(jpeg_data, pixel_format=pixel_format)
                return frame
            except Exception:
                pass

        frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to decode image from tar: {self.tar_path}/{member_name}")
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _read_member_bytes(self, index: int):
        if index < 0 or index >= len(self.frame_names):
            raise IndexError(
                f"Frame index {index} out of range [0, {len(self.frame_names)}). "
                f"Total frames available: {len(self.frame_names)}"
            )

        member_name = self.frame_names[index]

        if self.frame_offsets is not None:
            offset, size = self.frame_offsets[index]
            payload = os.pread(self._get_fd(), size, offset)
            if len(payload) != size:
                raise RuntimeError(
                    f"Short read from tar: {self.tar_path}/{member_name} "
                    f"(expected {size} bytes, got {len(payload)})"
                )
            return member_name, payload

        member_index = _load_shard_member_index(self.tar_path)
        offset_size = member_index.get(member_name)
        if offset_size is not None:
            offset, size = offset_size
            payload = os.pread(self._get_fd(), size, offset)
            if len(payload) != size:
                raise RuntimeError(
                    f"Short read from tar: {self.tar_path}/{member_name} "
                    f"(expected {size} bytes, got {len(payload)})"
                )
            return member_name, payload

        tar = self._get_tar()
        member = tar.getmember(member_name)
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Failed to extract image from tar: {self.tar_path}/{member_name}")
        return member_name, extracted.read()

    def get_frame_bytes(self, index: int):
        _, payload = self._read_member_bytes(index)
        return payload

    def __del__(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
        tar = getattr(self._thread_local, 'tar', None)
        if tar is not None:
            try:
                tar.close()
            except Exception:
                pass


class CachedFrameSource(BaseFrameSource):
    """Thread-safe LRU cache wrapper around any BaseFrameSource.

    Motion inference revisits the same frames across overlapping chunks, so an
    in-memory cache of recently decoded frames avoids repeated JPEG decode / tar
    reads. ``get_frame`` may be called concurrently by the inference prefetch
    thread pool, so the cache is guarded by a lock. Cache keys include the
    ``rgb`` flag since BGR and RGB decodes differ.
    """

    def __init__(self, inner: BaseFrameSource, max_items: int = 128):
        self.inner = inner
        self.max_items = max(0, int(max_items))
        self._cache = OrderedDict()  # (index, rgb) -> frame
        self._lock = threading.Lock()

    def __len__(self):
        return len(self.inner)

    @property
    def image_paths(self):
        # Preserve the fast paths in any4d (_direct_frame_path) and FrameDataset
        # that look up image_paths via getattr on the source.
        return getattr(self.inner, "image_paths", None)

    def get_frame(self, index: int, rgb: bool = False):
        if self.max_items <= 0:
            return self.inner.get_frame(index, rgb=rgb)
        key = (int(index), bool(rgb))
        with self._lock:
            frame = self._cache.get(key)
            if frame is not None:
                self._cache.move_to_end(key)
                return frame
        # Decode outside the lock so concurrent callers do not serialize on IO.
        frame = self.inner.get_frame(index, rgb=rgb)
        with self._lock:
            self._cache[key] = frame
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)
        return frame

    def get_frame_bytes(self, index: int):
        return self.inner.get_frame_bytes(index)

    def get_size(self):
        return self.inner.get_size()


class FrameDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for parallel frame loading via DataLoader.

    Works with any BaseFrameSource (ImageFolderFrameSource, ShardVideoFrameSource, etc.).
    """

    def __init__(self, frame_source: BaseFrameSource):
        self.frame_source = frame_source
        self.use_turbojpeg = getattr(frame_source, 'use_turbojpeg', False)
        if self.use_turbojpeg:
            self.jpeg_decoder = TurboJPEG()
        else:
            self.jpeg_decoder = None
        # Cache image_paths for ImageFolderFrameSource fast path
        self._image_paths = getattr(frame_source, 'image_paths', None)

    def __len__(self):
        return len(self.frame_source)

    def __getitem__(self, idx):
        # Fast path: ImageFolderFrameSource with TurboJPEG (avoids get_frame overhead)
        if self._image_paths is not None:
            path = self._image_paths[idx]
            if self.use_turbojpeg and path.lower().endswith(('.jpg', '.jpeg')):
                try:
                    with open(path, 'rb') as f:
                        jpeg_data = f.read()
                    frame = self.jpeg_decoder.decode(jpeg_data, pixel_format=1)  # BGR
                    return idx, frame
                except Exception:
                    pass
            frame = cv2.imread(path)
            if frame is None:
                raise RuntimeError(f"Failed to read image: {path}")
            return idx, frame

        # Generic path: any BaseFrameSource (ShardVideoFrameSource etc.)
        frame = self.frame_source.get_frame(idx, rgb=False)
        return idx, frame


def _numpy_collate(batch):
    """Collate (idx, np_array) pairs without stacking (frames may vary in content)."""
    indices = [b[0] for b in batch]
    frames = [b[1] for b in batch]
    return indices, frames


def _frame_dataset_worker_init(worker_id):
    """Each DataLoader worker needs its own TurboJPEG instance (C library not fork-safe)."""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if dataset.use_turbojpeg and TURBOJPEG_AVAILABLE:
        dataset.jpeg_decoder = TurboJPEG()
    # Re-open low-level handles for ShardVideoFrameSource after fork.
    fs = dataset.frame_source
    if hasattr(fs, '_fd') and fs._fd is not None:
        os.close(fs._fd)
        fs._fd = None
    thread_local = getattr(fs, '_thread_local', None)
    if thread_local is not None:
        tar = getattr(thread_local, 'tar', None)
        if tar is not None:
            tar.close()
        fs._thread_local = threading.local()


def build_frame_source(video_path: str):
    """Build an ImageFolderFrameSource from pre-extracted frames.

    For WebDataset format, use ShardVideoFrameSource directly instead.
    """
    from pathlib import Path
    import glob
    from natsort import natsorted

    video_path_obj = Path(video_path)
    video_dir = video_path_obj.parent
    video_stem = video_path_obj.stem

    extracted_dir = video_dir / video_stem / "extracted_images"
    if extracted_dir.exists():
        image_files = natsorted(glob.glob(str(extracted_dir / "*.jpg")))
        if not image_files:
            image_files = natsorted(glob.glob(str(extracted_dir / "*.png")))

        if image_files:
            if not QUIET_MODE:
                print(f"Using extracted frames: {extracted_dir} ({len(image_files)} frames)")
            return ImageFolderFrameSource(image_files)

    raise FileNotFoundError(
        f"No frames found for {video_path}. Expected:\n"
        f"  - JPEG folder: {extracted_dir}/*.jpg\n"
        f"  - Or use ShardVideoFrameSource for WebDataset format"
    )
