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
):
    """Run HAWOR over ``frame_indices`` and return per-frame predictions.

    Args mirror the original ``HAWOR.inference`` (``model`` replaces ``self``).
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

    # Pad dataset to a multiple of seq_len so every window is full.
    remainder = total_frames % seq_len
    if remainder != 0:
        pad_size = seq_len - remainder
        padded_indices = list(range(total_frames)) + [total_frames - 1] * pad_size
    else:
        padded_indices = list(range(total_frames))
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

    def _prefetch_worker():
        pool = ThreadPoolExecutor(max_workers=load_workers)
        try:
            for start, end in batch_ranges:
                indices = [padded_indices[i] for i in range(start, end)]
                items = list(pool.map(db.__getitem__, indices))
                prefetch_q.put((_collate(items), end - start))
                del items
        finally:
            pool.shutdown(wait=False)
        prefetch_q.put(None)

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

    while True:
        t_wait = time.time()
        item = prefetch_q.get()
        perf['wait_prefetch_sec'] += time.time() - t_wait
        if item is None:
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

    # Concatenate on GPU, then transfer to CPU once.
    t_concat = time.time()
    pred_cam = torch.cat(pred_cam, dim=0)[:total_frames]
    pred_pose = torch.cat(pred_pose, dim=0)[:total_frames]
    pred_shape = torch.cat(pred_shape, dim=0)[:total_frames]
    pred_rotmat = torch.cat(pred_rotmat, dim=0)[:total_frames]
    pred_trans = torch.cat(pred_trans, dim=0)[:total_frames]
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
