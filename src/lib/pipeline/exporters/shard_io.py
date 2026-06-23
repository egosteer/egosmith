"""Shared .npy encoding for WebDataset shard samples (lowdim + MANO arrays).

Both shard writers (webdataset_writer and manifest_build.writer) encode the per-frame lowdim
vector and MANO array the same way. The precomputed npy headers let the common fixed-shape,
C-contiguous case skip np.save and just concatenate header + raw bytes; the output is byte-for-byte
identical to a plain np.save, so existing shards are unaffected.
"""

import io

import numpy as np

from .mano_codec import MANO_SAMPLE_SHAPE

LOWDIM_SIZE = 116
LOWDIM_DTYPE = np.dtype(np.float32)
_LOWDIM_SAMPLE = np.zeros((LOWDIM_SIZE,), dtype=LOWDIM_DTYPE)
_LOWDIM_BUF = io.BytesIO()
np.save(_LOWDIM_BUF, _LOWDIM_SAMPLE, allow_pickle=False)
_LOWDIM_NPY_HEADER = _LOWDIM_BUF.getvalue()[: -_LOWDIM_SAMPLE.nbytes]
MANO_DTYPE = np.dtype(np.float32)
_MANO_SAMPLE = np.zeros(MANO_SAMPLE_SHAPE, dtype=MANO_DTYPE)
_MANO_BUF = io.BytesIO()
np.save(_MANO_BUF, _MANO_SAMPLE, allow_pickle=False)
_MANO_NPY_HEADER = _MANO_BUF.getvalue()[: -_MANO_SAMPLE.nbytes]


def encode_lowdim_npy(lowdim) -> bytes:
    array = np.asarray(lowdim, dtype=LOWDIM_DTYPE)
    if array.shape == (LOWDIM_SIZE,):
        return _LOWDIM_NPY_HEADER + np.ascontiguousarray(array).tobytes()

    lowdim_buf = io.BytesIO()
    np.save(lowdim_buf, array, allow_pickle=False)
    return lowdim_buf.getvalue()


def encode_array_npy(array) -> bytes:
    encoded = np.asarray(array, dtype=MANO_DTYPE)
    if encoded.shape == MANO_SAMPLE_SHAPE and encoded.flags.c_contiguous:
        return _MANO_NPY_HEADER + encoded.tobytes()

    buf = io.BytesIO()
    np.save(buf, encoded, allow_pickle=False)
    return buf.getvalue()
