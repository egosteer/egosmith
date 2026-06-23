"""Batched hand detection + tracking and detection post-processing (EgoSmith first-party).

EgoSmith replaces HaWoR's stock per-frame ``cv2.imread`` + YOLO ``.track()`` loop
with a throughput-oriented two-phase pipeline:

  * **Phase 1 — batched detection** runs ``YOLO.predict`` over large frame batches
    fed by a prefetching thread pool (decode/resize release the GIL), so the GPU
    stays busy while the next batch loads.
  * **Phase 2 — sequential tracking** assigns track IDs with ``supervision``'s
    ByteTrack on CPU, plus edge/velocity gating for robustness.

It only consumes the ``frame_source`` abstraction (``__len__`` / ``get_frame`` /
``get_size``) and the external YOLO detector — none of HaWoR's model code — so it
ships as EgoSmith code. The verbatim HaWoR track-segmentation helpers
(``parse_chunks`` / ``parse_chunks_hand_frame``) stay in the obtained
``lib/pipeline/tools.py``.
"""

import os

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
from scipy.interpolate import interp1d

from lib.pipeline.proc.logging_setup import QUIET_MODE  # noqa: F401


def _iter_detect_batches(frame_source, detect_batch_size: int, num_io_workers: int):
    from concurrent.futures import ThreadPoolExecutor
    import queue
    import threading

    num_frames = len(frame_source)
    if num_frames <= 0:
        return

    # Short clips pay more for spawning loader workers than they gain back.
    if num_io_workers <= 0 or num_frames <= max(detect_batch_size * 2, 256):
        for start_idx in tqdm(
            range(0, num_frames, detect_batch_size),
            disable=QUIET_MODE,
            desc="Detect (batched)",
        ):
            end_idx = min(start_idx + detect_batch_size, num_frames)
            batch_indices = list(range(start_idx, end_idx))
            batch_frames = [frame_source.get_frame(frame_idx, rgb=False) for frame_idx in batch_indices]
            yield batch_indices, batch_frames
        return

    effective_io_workers = min(num_io_workers, max(1, os.cpu_count() or 1))
    batch_ranges = [
        (start_idx, min(start_idx + detect_batch_size, num_frames))
        for start_idx in range(0, num_frames, detect_batch_size)
    ]
    get_frame = frame_source.get_frame

    # Threaded loading avoids per-video DataLoader process startup and IPC overhead.
    # JPEG decode / cv2.imdecode / os.pread all release the GIL in the hot path,
    # so threads still parallelize frame IO well here. A dedicated prefetch thread
    # overlaps loading batch N+1 while YOLO runs on batch N.
    prefetch_q = queue.Queue(maxsize=2)
    prefetch_error = []
    stop_event = threading.Event()

    def _put_interruptible(item) -> bool:
        # Park on a full queue only in short slices so that a consumer which
        # stops early (generator .close() or an exception in the loop body) can
        # signal us to bail. Blocking forever on put() while the consumer has
        # already moved on to join() is the classic producer/consumer deadlock.
        while not stop_event.is_set():
            try:
                prefetch_q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _load_batches():
        with ThreadPoolExecutor(max_workers=effective_io_workers) as pool:
            try:
                for start_idx, end_idx in batch_ranges:
                    if stop_event.is_set():
                        break
                    batch_indices = list(range(start_idx, end_idx))
                    batch_frames = list(
                        pool.map(
                            lambda frame_idx: get_frame(frame_idx, rgb=False),
                            batch_indices,
                            chunksize=4,
                        )
                    )
                    if not _put_interruptible((batch_indices, batch_frames)):
                        break
            except Exception as error:
                prefetch_error.append(error)
            finally:
                # Reliably deliver the end sentinel. A bare put(None, timeout=0.5)
                # silently drops the sentinel when the queue is momentarily full
                # (e.g. small clip: both batches buffered while the consumer is busy
                # with the cold-start YOLO model load / first predict), leaving the
                # consumer's blocking get() to deadlock forever. _put_interruptible
                # retries until a slot frees and only bails once the consumer has
                # itself stopped (stop_event set), so it can never hang.
                _put_interruptible(None)

    loader_thread = threading.Thread(target=_load_batches, daemon=True)
    loader_thread.start()

    progress = tqdm(total=len(batch_ranges), disable=QUIET_MODE, desc="Detect (batched)")
    try:
        while True:
            item = prefetch_q.get()
            if item is None:
                if prefetch_error:
                    raise prefetch_error[0]
                break
            progress.update(1)
            yield item
    finally:
        progress.close()
        # Tell the loader to stop and drain the queue so a loader parked on a
        # full put() unblocks; otherwise join() here deadlocks (and would mask
        # any real exception raised by the consumer, e.g. in YOLO.predict()).
        stop_event.set()
        while loader_thread.is_alive():
            try:
                prefetch_q.get(timeout=0.1)
            except queue.Empty:
                pass
        loader_thread.join()


def detect_track(
    frame_source,
    thresh=0.35,
    edge_margin_ratio=0.1,
    min_edge_conf=0.4,
    hand_det_model=None,
    detect_batch_size=128,
    num_io_workers=8,
    device='cuda:0',
    half_precision=True,
):
    """
    Detect and track hands using batched YOLO inference + post-hoc ByteTrack.

    Phase 1: Batch detection - YOLO.predict() with large batch sizes for high GPU utilization
    Phase 2: Sequential tracking - supervision.ByteTrack on CPU for track assignment

    Args:
        frame_source: ImageFolderFrameSource with pre-extracted frames
        thresh: Base confidence threshold for detection
        edge_margin_ratio: Ratio of image size to define edge region (default 0.1 = 10%)
        min_edge_conf: Minimum confidence required for detections near edges
        hand_det_model: Optional preloaded YOLO detector for reuse
        detect_batch_size: Batch size for YOLO.predict() (default 128)
        num_io_workers: Number of DataLoader workers for parallel frame loading
        device: Device for YOLO detector (e.g., 'cuda:0')
        half_precision: Use FP16 for YOLO inference
    """
    hand_det_model = hand_det_model or YOLO('./weights/external/detector.pt')

    if device:
        hand_det_model.to(device)
    use_half = half_precision and device and 'cuda' in device

    num_frames = len(frame_source)
    img_h, img_w = frame_source.get_size()

    # --- Phase 1: Batch Detection (GPU) ---
    all_detections = [None] * num_frames  # (xyxy, confs, class_ids) per frame
    all_boxes_raw = [np.array([]).reshape(0, 5)] * num_frames  # boxes with conf for output

    for batch_indices, batch_frames in _iter_detect_batches(
        frame_source,
        detect_batch_size=detect_batch_size,
        num_io_workers=num_io_workers,
    ):
        with torch.inference_mode():
            results_list = hand_det_model.predict(
                batch_frames, conf=thresh, verbose=False, half=use_half,
            )

        for frame_idx, result in zip(batch_indices, results_list):
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            all_detections[frame_idx] = (boxes, confs, class_ids)
            if len(boxes) > 0:
                all_boxes_raw[frame_idx] = np.hstack([boxes, confs[:, None]])

    # --- Phase 2: Sequential Tracking (CPU) ---
    tracker = sv.ByteTrack()

    tracks = {}
    fallback_counter = 0
    track_last_seen = {}

    edge_left = img_w * edge_margin_ratio
    edge_right = img_w * (1 - edge_margin_ratio)
    edge_top = img_h * edge_margin_ratio
    edge_bottom = img_h * (1 - edge_margin_ratio)

    for t in tqdm(range(num_frames), disable=QUIET_MODE, desc="Track (sequential)"):
        det = all_detections[t]
        if det is None:
            continue

        boxes_xyxy, confs, class_ids = det

        if len(boxes_xyxy) > 0:
            sv_detections = sv.Detections(
                xyxy=boxes_xyxy,
                confidence=confs,
                class_id=class_ids.astype(int),
            )
            tracked = tracker.update_with_detections(sv_detections)

            t_boxes = tracked.xyxy
            t_confs = tracked.confidence
            t_track_ids = tracked.tracker_id if tracked.tracker_id is not None else np.full(len(t_boxes), -1)
            t_class_ids = tracked.class_id if tracked.class_id is not None else np.zeros(len(t_boxes), dtype=int)
        else:
            t_boxes = np.array([]).reshape(0, 4)
            t_confs = np.array([])
            t_track_ids = np.array([])
            t_class_ids = np.array([])

        find_right = False
        find_left = False

        for idx in range(len(t_boxes)):
            x1, y1, x2, y2 = t_boxes[idx]
            conf = t_confs[idx]
            track_id_val = t_track_ids[idx]
            handedness = t_class_ids[idx]

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            is_near_edge = (cx < edge_left or cx > edge_right or
                           cy < edge_top or cy > edge_bottom)

            if is_near_edge and conf < min_edge_conf:
                continue

            if track_id_val == -1:
                if handedness > 0:
                    id = int(10000 + fallback_counter)
                else:
                    id = int(5000 + fallback_counter)
                fallback_counter += 1
            else:
                id = int(track_id_val)

            if id in track_last_seen:
                frames_since_last = t - track_last_seen[id]
                if frames_since_last > 10 and conf < min_edge_conf:
                    continue

            subj = {
                'frame': t,
                'det': True,
                'det_box': np.array([[x1, y1, x2, y2, conf]]),
                'det_handedness': np.array([handedness]),
                'is_near_edge': is_near_edge,
            }

            if (not find_right and handedness > 0) or (not find_left and handedness == 0):
                if id in tracks:
                    tracks[id].append(subj)
                else:
                    tracks[id] = [subj]

                track_last_seen[id] = t

                if handedness > 0:
                    find_right = True
                elif handedness == 0:
                    find_left = True

    tracks = np.array(tracks, dtype=object)
    boxes_ = np.array(all_boxes_raw, dtype=object)

    return boxes_, tracks


def validate_motion_velocity(bboxes, max_relative_velocity=3.0):
    """
    Validate motion velocity to detect physically implausible movements.
    Uses relative velocity (movement relative to bbox size) instead of absolute pixels.

    Args:
        bboxes: (T, 5) array of [x1, y1, x2, y2, conf]
        max_relative_velocity: Maximum movement as multiple of bbox diagonal per frame

    Returns:
        valid_mask: Boolean array indicating valid frames
    """
    T = bboxes.shape[0]
    if T < 2:
        return np.ones(T, dtype=bool)

    # Calculate bbox centers and sizes
    centers = np.stack([
        (bboxes[:, 0] + bboxes[:, 2]) / 2,
        (bboxes[:, 1] + bboxes[:, 3]) / 2
    ], axis=1)

    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    diagonals = np.sqrt(widths**2 + heights**2)

    # Calculate frame-to-frame displacements
    displacements = np.linalg.norm(centers[1:] - centers[:-1], axis=1)

    # Calculate relative velocities (displacement / average diagonal)
    avg_diagonals = (diagonals[1:] + diagonals[:-1]) / 2
    relative_velocities = np.zeros(T - 1)
    valid_diag_mask = avg_diagonals > 0
    relative_velocities[valid_diag_mask] = displacements[valid_diag_mask] / avg_diagonals[valid_diag_mask]

    # Mark frames with excessive relative velocity as invalid
    valid = np.ones(T, dtype=bool)
    valid[1:] = relative_velocities < max_relative_velocity

    return valid


def interpolate_bboxes(bboxes, max_size_change_ratio=2.5):
    """
    Interpolate missing bboxes with size consistency validation.

    Args:
        bboxes: (T, 5) array of [x1, y1, x2, y2, conf]
        max_size_change_ratio: Maximum allowed size change between adjacent frames
    """
    T = bboxes.shape[0]

    # First pass: filter out bboxes with abnormal size changes
    non_zero_mask = np.any(bboxes != 0, axis=1)
    non_zero_indices = np.where(non_zero_mask)[0]

    if len(non_zero_indices) > 1:
        # Calculate bbox areas
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        areas = widths * heights

        # Check size changes between consecutive valid detections
        for i in range(len(non_zero_indices) - 1):
            curr_idx = non_zero_indices[i]
            next_idx = non_zero_indices[i + 1]

            curr_area = areas[curr_idx]
            next_area = areas[next_idx]

            if curr_area > 0 and next_area > 0:
                size_ratio = max(curr_area, next_area) / min(curr_area, next_area)

                # If size change is too large, mark the detection with lower confidence as invalid
                if size_ratio > max_size_change_ratio:
                    # Keep the one with higher confidence, or the earlier one if confidence is same
                    if bboxes[curr_idx, 4] < bboxes[next_idx, 4]:
                        bboxes[curr_idx] = 0
                        non_zero_mask[curr_idx] = False
                    else:
                        bboxes[next_idx] = 0
                        non_zero_mask[next_idx] = False

        # Update non_zero_indices after filtering
        non_zero_indices = np.where(non_zero_mask)[0]

    zero_indices = np.where(~non_zero_mask)[0]

    if len(zero_indices) == 0 or len(non_zero_indices) == 0:
        return bboxes

    interpolated_bboxes = bboxes.copy()
    for i in range(5):
        interp_func = interp1d(non_zero_indices, bboxes[non_zero_indices, i], kind='linear', fill_value="extrapolate")
        interpolated_bboxes[zero_indices, i] = interp_func(zero_indices)

    return interpolated_bboxes
