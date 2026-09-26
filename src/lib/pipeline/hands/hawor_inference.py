"""Batched HaWoR motion inference (EgoSmith first-party).

`batched_hawor_inference` is EgoSmith's throughput-oriented replacement for the
stock per-window HAWOR forward loop. It takes an already-constructed HAWOR model
and runs inference over a video's tracked hand boxes, with two optimizations that
give most of EgoSmith's motion-stage speedup:

  * **Window batching** — HaWoR processes one ``seq_len``-frame temporal window at
    a time; we pack ``chunk_batch_size`` windows into a single forward pass to
    keep the GPU busy.
  * **Overlapped CPU decode / GPU compute** — a background thread pool decodes,
    crops and resizes the next batch's frames (all GIL-releasing C extensions)
    while the current batch runs on the GPU.

Kept first-party (not inside the obtained HaWoR base ``lib/models/hawor.py``) so it
ships as EgoSmith code. It only touches the model's public surface — ``model.seq_len``
and ``model.forward`` — so it works against an unmodified HAWOR instance.
"""

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from lib.pipeline.hands.track_dataset import TrackDatasetEval


def batched_hawor_inference(
    model,
    frame_source,
    frame_indices,
    boxes,
    img_focal,
    img_center,
    device='cuda',
    do_flip=False,
    chunk_batch_size=32,
    num_workers=16,
    output_device='cpu',
    return_perf=False,
    chunk_boundaries=None,
):
    """Run HAWOR over ``frame_indices`` and return per-frame predictions.

    Args mirror the original ``HAWOR.inference`` (``model`` replaces ``self``).
    ``chunk_boundaries`` (optional) lists the cumulative start offsets of the
    concatenated temporal chunks, e.g. ``[0, n0, n0 + n1, ..., len(frame_indices)]``.
    When given, each chunk is padded to a multiple of ``seq_len`` on its own (by
    repeating its last frame, as upstream ``HAWOR.inference`` does per chunk), so
    no temporal window mixes frames from different chunks; padded positions are
    dropped and outputs keep the input order and length. When omitted, the whole
    sequence is windowed as one chunk (legacy behavior).
    Returns a dict with pred_cam / pred_pose / pred_shape / pred_rotmat /
    pred_trans (+ img_focal / img_center, and ``_perf`` timings if requested).
    """
    db = TrackDatasetEval(frame_source, frame_indices, boxes, img_focal=img_focal,
                    img_center=img_center, normalization=True, dilate=1.2, do_flip=do_flip)

    seq_len = model.seq_len
    total_frames = len(db)

    if total_frames == 0:
        empty = torch.empty(0)
        return {
            'pred_cam': empty,
            'pred_pose': empty,
            'pred_shape': empty,
            'pred_rotmat': empty,
            'pred_trans': empty,
            'img_focal': img_focal,
            'img_center': img_center,
        }

    # Pad to a multiple of seq_len so every window is full. With chunk boundaries,
    # each chunk is padded separately so windows never straddle a chunk gap.
    if chunk_boundaries is None:
        segments = [(0, total_frames)]
    else:
        bounds = [int(b) for b in chunk_boundaries]
        if bounds[0] != 0 or bounds[-1] != total_frames or any(b1 <= b0 for b0, b1 in zip(bounds, bounds[1:])):
            raise ValueError(
                f"chunk_boundaries must increase strictly from 0 to {total_frames}, got {chunk_boundaries}"
            )
        segments = list(zip(bounds[:-1], bounds[1:]))
    padded_indices = []
    keep_positions = []  # position in padded order of each original frame, in input order
    for seg_start, seg_end in segments:
        keep_positions.extend(range(len(padded_indices), len(padded_indices) + seg_end - seg_start))
        padded_indices.extend(range(seg_start, seg_end))
        remainder = (seg_end - seg_start) % seq_len
        if remainder != 0:
            padded_indices.extend([seg_end - 1] * (seq_len - remainder))
    keep_is_prefix = keep_positions == list(range(total_frames))
    total_padded = len(padded_indices)
    dataloader_batch_size = chunk_batch_size * seq_len

    # Frame ranges per batched forward pass.
    batch_ranges = []
    for start in range(0, total_padded, dataloader_batch_size):
        batch_ranges.append((start, min(start + dataloader_batch_size, total_padded)))

    # --- Multi-threaded batch loading + GPU pipeline ---
    # Background thread uses a thread pool for parallel frame loading.
    # JPEG decode, crop, and resize are C extensions that release the GIL,
    # so honoring the requested worker count materially improves throughput
    # on high-core hosts backed by slower shared storage.
    # Main thread runs GPU inference on current batch while next batch loads
    cpu_workers = max(1, int(os.cpu_count() or 1))
    load_workers = max(1, min(int(num_workers), cpu_workers))
    prefetch_q = queue.Queue(maxsize=2)
    target_device = torch.device(device)
    use_non_blocking = target_device.type == 'cuda'

    def _collate(items):
        tensors = {}
        for key in items[0]:
            vals = [item[key] for item in items]
            if isinstance(vals[0], torch.Tensor):
                tensors[key] = torch.stack(vals)
        return tensors

    prefetch_error = []
    stop_event = threading.Event()

    def _put_interruptible(item):
        # Retry in short slices so the producer bails out (instead of blocking
        # forever) once the consumer has stopped, e.g. after a forward error.
        while not stop_event.is_set():
            try:
                prefetch_q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _prefetch_worker():
        pool = ThreadPoolExecutor(max_workers=load_workers)
        try:
            for start, end in batch_ranges:
                if stop_event.is_set():
                    break
                indices = [padded_indices[i] for i in range(start, end)]
                items = list(pool.map(db.__getitem__, indices))
                if not _put_interruptible((_collate(items), end - start)):
                    break
                del items
        except Exception as error:
            prefetch_error.append(error)
        finally:
            pool.shutdown(wait=False)
            # Always wake the consumer, including after a loader exception;
            # the consumer re-raises prefetch_error when it sees the sentinel.
            _put_interruptible(None)

    loader_thread = threading.Thread(target=_prefetch_worker, daemon=True)
    loader_thread.start()

    # --- GPU inference loop ---
    pred_cam = []
    pred_pose = []
    pred_shape = []
    pred_rotmat = []
    pred_trans = []
    perf = {
        'total_frames': int(total_frames),
        'batch_count': 0,
        'load_workers': int(load_workers),
        'wait_prefetch_sec': 0.0,
        'host_to_device_sec': 0.0,
        'forward_sec': 0.0,
        'concat_sec': 0.0,
    }

    try:
        while True:
            t_wait = time.time()
            item = prefetch_q.get()
            perf['wait_prefetch_sec'] += time.time() - t_wait
            if item is None:
                if prefetch_error:
                    raise prefetch_error[0]
                break
            batch_tensors, current_batch_size = item
            current_chunks = current_batch_size // seq_len
            perf['batch_count'] += 1

            batch = {}
            t_h2d = time.time()
            for k, v in batch_tensors.items():
                shaped = v.view(current_chunks, seq_len, *v.shape[1:])
                if use_non_blocking and shaped.device.type == 'cpu':
                    shaped = shaped.pin_memory()
                batch[k] = shaped.to(device, non_blocking=use_non_blocking)
            perf['host_to_device_sec'] += time.time() - t_h2d
            del batch_tensors

            t_forward = time.time()
            with torch.inference_mode():
                output = model.forward(batch)
                out = output['out']
            perf['forward_sec'] += time.time() - t_forward

            expected = current_batch_size
            out = {k: v[:expected] for k, v in out.items()}

            pred_cam.append(out['pred_cam'])
            pred_pose.append(out['pred_pose'])
            pred_shape.append(out['pred_shape'])
            pred_rotmat.append(out['pred_rotmat'])
            pred_trans.append(out['trans_full'])
    finally:
        # Stop the loader if we exit early (e.g. a forward error) so it cannot
        # stay parked on a full queue.
        stop_event.set()

    # Concatenate on GPU, then transfer to CPU once.
    t_concat = time.time()
    if keep_is_prefix:
        keep = slice(0, total_frames)
    else:
        keep = torch.as_tensor(keep_positions, dtype=torch.long, device=pred_cam[0].device)
    pred_cam = torch.cat(pred_cam, dim=0)[keep]
    pred_pose = torch.cat(pred_pose, dim=0)[keep]
    pred_shape = torch.cat(pred_shape, dim=0)[keep]
    pred_rotmat = torch.cat(pred_rotmat, dim=0)[keep]
    pred_trans = torch.cat(pred_trans, dim=0)[keep]
    perf['concat_sec'] += time.time() - t_concat

    if output_device is not None:
        target_device = torch.device(output_device)
        if pred_cam.device != target_device:
            pred_cam = pred_cam.to(target_device)
            pred_pose = pred_pose.to(target_device)
            pred_shape = pred_shape.to(target_device)
            pred_rotmat = pred_rotmat.to(target_device)
            pred_trans = pred_trans.to(target_device)

    result = {
        'pred_cam': pred_cam,
        'pred_pose': pred_pose,
        'pred_shape': pred_shape,
        'pred_rotmat': pred_rotmat,
        'pred_trans': pred_trans,
        'img_focal': img_focal,
        'img_center': img_center,
    }
    if return_perf:
        result['_perf'] = perf
    return result
