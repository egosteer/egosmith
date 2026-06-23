"""Any4D depth inference helpers for stage3."""

import contextlib
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        del args, kwargs
        return iterable if iterable is not None else []


# src/lib/pipeline/slam/any4d_depth.py -> parents[4] is the repo root (parents[3] is src/).
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def ensure_exists(path: str, what: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Any4D] {what} not found: {path}")


def _env_flag_on(name: str, *, default_on: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default_on
    value = str(raw).strip().lower()
    if value == "":
        return default_on
    if value in ("0", "false", "no", "off"):
        return False
    return True


def _existing_path_or_none(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.exists() else None


def _fallback_any4d_repo_root(project_root: Path) -> Path | None:
    return _existing_path_or_none(project_root / "thirdparty" / "Any4D")


def _fallback_any4d_checkpoint(repo_root: Path | None) -> Path | None:
    if repo_root is None:
        return None
    return _existing_path_or_none(repo_root / "checkpoints" / "any4d_4v_combined.pth")


def resolve_any4d_paths(project_root=None, any4d_repo_root=None, checkpoint_path=None, resolution_set=None, use_amp=None):
    project_root = Path(project_root or PROJECT_ROOT).resolve()

    raw_repo_root = any4d_repo_root or os.environ.get("HAWOR_ANY4D_REPO_ROOT")
    if raw_repo_root:
        repo_root = Path(raw_repo_root).expanduser()
        if not repo_root.is_absolute():
            repo_root = (project_root / repo_root).resolve()
    else:
        repo_root = _fallback_any4d_repo_root(project_root)
        if repo_root is None:
            repo_root = (project_root / "thirdparty" / "Any4D").resolve()

    raw_checkpoint = checkpoint_path or os.environ.get("HAWOR_ANY4D_CHECKPOINT_PATH")
    if raw_checkpoint:
        checkpoint = Path(raw_checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (project_root / checkpoint).resolve()
    else:
        checkpoint = _fallback_any4d_checkpoint(repo_root)
        if checkpoint is None:
            checkpoint = (repo_root / "checkpoints" / "any4d_4v_combined.pth").resolve()

    if resolution_set is None:
        resolution_set = int(os.environ.get("HAWOR_ANY4D_RESOLUTION", "518"))
    if use_amp is None:
        use_amp = _env_flag_on("HAWOR_ANY4D_USE_AMP", default_on=True)

    return str(repo_root), str(checkpoint), int(resolution_set), bool(use_amp)


@contextlib.contextmanager
def _suppress_any4d_init_io():
    if os.environ.get("HAWOR_ANY4D_VERBOSE", "0").strip() == "1":
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _prepend_sys_path(path: str):
    if path not in sys.path:
        sys.path.insert(0, path)


def _any4d_config_dict(any4d_root: str, *, task: str = "images_only"):
    use_sdpa = _env_flag_on("HAWOR_ANY4D_USE_PYTORCH_SDPA", default_on=True)
    sdpa_override = (
        "+model.encoder.use_pytorch_sdpa=true"
        if use_sdpa
        else "+model.encoder.use_pytorch_sdpa=false"
    )
    return {
        "path": os.path.join(any4d_root, "configs", "train.yaml"),
        "config_overrides": [
            "machine=local",
            "model=any4d",
            "model.encoder.uses_torch_hub=false",
            sdpa_override,
            f"model/task={task}",
        ],
    }


def _direct_frame_path(frame_source, frame_idx: int):
    image_paths = getattr(frame_source, "image_paths", None)
    if image_paths is None:
        return None
    if frame_idx < 0 or frame_idx >= len(image_paths):
        return None
    path = image_paths[frame_idx]
    return path if os.path.exists(path) else None


def _materialize_frame(frame_source, frame_idx: int, temp_dir: str):
    image = frame_source.get_frame(frame_idx, rgb=False)
    path = os.path.join(temp_dir, f"{int(frame_idx):06d}.png")
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"[Any4D] failed to materialize frame {frame_idx} to {path}")
    return path


def _resolve_image_paths(frame_source, frame_indices: Sequence[int], ref_frame_idx: int, temp_dir: Optional[str] = None):
    image_paths = []
    for frame_idx in [ref_frame_idx, *frame_indices]:
        path = _direct_frame_path(frame_source, frame_idx)
        if path is None:
            if temp_dir is None:
                raise RuntimeError(
                    f"[Any4D] frame_source does not expose direct paths for frame {frame_idx}, "
                    "and no temp_dir was provided for materialization."
                )
            path = _materialize_frame(frame_source, frame_idx, temp_dir)
        image_paths.append(path)
    return image_paths


@contextlib.contextmanager
def _build_image_paths(frame_source, frame_indices: Sequence[int], ref_frame_idx: int):
    with tempfile.TemporaryDirectory(prefix="hawor-any4d-") as temp_dir:
        yield _resolve_image_paths(frame_source, frame_indices, ref_frame_idx, temp_dir=temp_dir)


def _import_any4d_modules(repo_root: str):
    _prepend_sys_path(repo_root)

    # `init_inference_model` / `sample_inference` are upstream Any4D code
    # (Apache-2.0), carried first-party in `lib.pipeline.slam.any4d_inference` so
    # EgoSmith does not depend on Any4D's `scripts/` directory.
    from lib.pipeline.slam import any4d_inference as any4d_inference_test
    from any4d.utils.image import load_images

    return any4d_inference_test, load_images


def _import_any4d_camera_modules(repo_root: str):
    _prepend_sys_path(repo_root)

    import torchvision.transforms as tvf
    from PIL import Image
    from PIL.ImageOps import exif_transpose
    from any4d.utils.cropping import crop_resize_if_necessary
    from any4d.utils.image import find_closest_aspect_ratio
    from any4d.utils.inference import preprocess_input_views_for_inference
    from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

    return {
        "tvf": tvf,
        "Image": Image,
        "exif_transpose": exif_transpose,
        "crop_resize_if_necessary": crop_resize_if_necessary,
        "find_closest_aspect_ratio": find_closest_aspect_ratio,
        "preprocess_input_views_for_inference": preprocess_input_views_for_inference,
        "image_normalization_dict": IMAGE_NORMALIZATION_DICT,
    }


def _load_any4d_views(load_images, image_paths, resolution_set: int):
    return load_images(
        image_paths,
        resize_mode="fixed_mapping",
        resolution_set=resolution_set,
        norm_type="dinov2",
        patch_size=14,
        verbose=False,
        compute_moge_mask=False,
        binary_mask_path=None,
    )


def _validate_camera_conditioning_inputs(image_payloads, intrinsics, camera_poses):
    count = len(image_payloads)
    if count <= 0:
        raise ValueError("[Any4D] image_payloads is empty")
    if len(intrinsics) != count:
        raise ValueError(f"[Any4D] intrinsics count {len(intrinsics)} != image count {count}")
    if len(camera_poses) != count:
        raise ValueError(f"[Any4D] camera_poses count {len(camera_poses)} != image count {count}")


def _as_intrinsics_matrix(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (4,):
        fx, fy, cx, cy = array.tolist()
        array = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    if array.shape != (3, 3):
        raise ValueError(f"[Any4D] expected intrinsics shape (3, 3) or (4,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("[Any4D] intrinsics contain non-finite values")
    if float(array[0, 0]) <= 0.0 or float(array[1, 1]) <= 0.0:
        raise ValueError(f"[Any4D] invalid focal lengths: fx={array[0, 0]}, fy={array[1, 1]}")
    return array


def _as_camera_pose_matrix(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (4, 4):
        raise ValueError(f"[Any4D] expected camera pose shape (4, 4), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("[Any4D] camera pose contains non-finite values")
    if abs(float(np.linalg.det(array[:3, :3]))) < 1e-8:
        raise ValueError("[Any4D] camera pose rotation is singular")
    return array


def build_any4d_camera_views_from_image_bytes(
    image_payloads,
    intrinsics,
    camera_poses,
    runner=None,
    *,
    any4d_repo_root=None,
    checkpoint_path=None,
    resolution_set=None,
    use_amp=None,
    task="mvs",
    norm_type="dinov2",
    is_metric_scale: bool = True,
):
    """Build Any4D views with calibrated intrinsics and OpenCV RDF cam2world poses.

    ``is_metric_scale`` flags whether the input camera poses are metric. Pass True for true-metric
    extrinsics (e.g. HOT3D GT); pass False for scale-free poses (e.g. DPVO), which must be paired
    with a non-metric pose task (model/task=non_metric_poses_metric_depth) so the model normalizes
    the input translations instead of treating them as metres.

    ``camera_poses`` must already be cam2world. For HOT3D lowdim extrinsics this
    means passing ``np.linalg.inv(world2cam)``.
    """
    _validate_camera_conditioning_inputs(image_payloads, intrinsics, camera_poses)

    runner = runner or build_any4d_runner(
        any4d_repo_root=any4d_repo_root,
        checkpoint_path=checkpoint_path,
        resolution_set=resolution_set,
        use_amp=use_amp,
        task=task,
    )
    modules = _import_any4d_camera_modules(runner["repo_root"])
    image_norms = modules["image_normalization_dict"]
    if norm_type not in image_norms:
        raise ValueError(f"[Any4D] unknown image normalization type: {norm_type}")
    img_norm = image_norms[norm_type]
    img_transform = modules["tvf"].Compose(
        [
            modules["tvf"].ToTensor(),
            modules["tvf"].Normalize(mean=img_norm.mean, std=img_norm.std),
        ]
    )

    pil_images = []
    aspect_ratios = []
    for payload in image_payloads:
        image = modules["exif_transpose"](modules["Image"].open(BytesIO(payload))).convert("RGB")
        width, height = image.size
        pil_images.append(image)
        aspect_ratios.append(float(width) / float(height))

    target_size = modules["find_closest_aspect_ratio"](
        sum(aspect_ratios) / len(aspect_ratios),
        int(runner["resolution_set"]),
    )

    views = []
    resized_intrinsics = []
    for view_idx, image in enumerate(pil_images):
        intrinsics_matrix = _as_intrinsics_matrix(intrinsics[view_idx])
        camera_pose = _as_camera_pose_matrix(camera_poses[view_idx])
        resized_image, resized_intrinsics_matrix = modules["crop_resize_if_necessary"](
            image,
            resolution=target_size,
            intrinsics=intrinsics_matrix,
        )
        resized_intrinsics_matrix = np.asarray(resized_intrinsics_matrix, dtype=np.float32)
        if not np.isfinite(resized_intrinsics_matrix).all():
            raise ValueError(
                f"[Any4D] resized intrinsics contain non-finite values for view {view_idx}"
            )
        resized_intrinsics.append(resized_intrinsics_matrix)

        mask = torch.ones((resized_image.size[1], resized_image.size[0]), dtype=torch.bool)
        views.append(
            {
                "img": img_transform(resized_image)[None],
                "intrinsics": torch.from_numpy(resized_intrinsics_matrix)[None].float(),
                "camera_poses": torch.from_numpy(camera_pose)[None].float(),
                "is_metric_scale": torch.full((1,), bool(is_metric_scale), dtype=torch.bool),
                "true_shape": np.int32([resized_image.size[::-1]]),
                "idx": view_idx,
                "instance": str(view_idx),
                "data_norm_type": [norm_type],
                "non_ambiguous_mask": mask,
                "binary_mask": mask,
            }
        )

    processed_views = modules["preprocess_input_views_for_inference"](views)
    return processed_views, np.stack(resized_intrinsics, axis=0)


def _predict_depths_from_views(
    any4d_inference_test,
    runner,
    views,
    frame_count: int,
    *,
    prediction_view_offset: int,
):
    device = str(next(runner["model"].parameters()).device)
    pred_result = any4d_inference_test.sample_inference(
        model=runner["model"],
        views=views,
        device=device,
        use_amp=runner["use_amp"],
    )

    depth_list = []
    for target_i in range(frame_count):
        view_idx = int(prediction_view_offset) + target_i
        key_name = f"pred{view_idx}"
        if key_name not in pred_result:
            available = sorted(pred_result.keys())
            raise KeyError(
                f"[Any4D] missing prediction key {key_name}; available keys: {available[:8]}"
            )
        depth_z = (
            pred_result[key_name]["pts3d_cam"][..., 2:3][0]
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )
        depth_list.append(depth_z.astype(np.float32))

    return np.stack(depth_list, axis=0)


def build_any4d_camera_views_from_paths(
    image_paths,
    intrinsics_per_view,
    camera_poses_c2w_per_view,
    runner,
    *,
    task=None,
    norm_type="dinov2",
    is_metric_scale: bool = True,
):
    """Pose-conditioned variant of build_any4d_views (Goal A.1 of the metric plan).

    Reads each image file as bytes and forwards to ``build_any4d_camera_views_from_image_bytes``
    with per-view intrinsics (3x3 K matrices) and cam2world poses (4x4 OpenCV-RDF).
    Use when the slam stage's selected Any4D task is pose-conditioned (anything other than
    ``images_only``), so the network gets DPVO's scale-free per-frame geometry as input. DPVO poses
    are scale-free, so callers should pass ``is_metric_scale=False`` together with a non-metric pose
    task (e.g. ``non_metric_poses_metric_depth``).
    Returns model-ready processed views (same shape contract as ``build_any4d_views``).
    """
    payloads = []
    for path in image_paths:
        with open(path, "rb") as fh:
            payloads.append(fh.read())
    views, _ = build_any4d_camera_views_from_image_bytes(
        payloads,
        intrinsics_per_view,
        camera_poses_c2w_per_view,
        runner=runner,
        task=task or runner.get("task"),
        norm_type=norm_type,
        is_metric_scale=is_metric_scale,
    )
    return views


def build_any4d_views(frame_source, frame_indices, runner=None, *, any4d_repo_root=None, checkpoint_path=None, resolution_set=None, use_amp=None, image_paths=None):
    frame_indices = [int(frame_idx) for frame_idx in frame_indices]
    if not frame_indices:
        raise ValueError("[Any4D] frame_indices is empty")

    runner = runner or build_any4d_runner(
        any4d_repo_root=any4d_repo_root,
        checkpoint_path=checkpoint_path,
        resolution_set=resolution_set,
        use_amp=use_amp,
    )

    load_images = runner["load_images"]
    if image_paths is None:
        ref_frame_idx = frame_indices[len(frame_indices) // 2]
        with _build_image_paths(frame_source, frame_indices, ref_frame_idx) as resolved_image_paths:
            return _load_any4d_views(load_images, resolved_image_paths, runner["resolution_set"])
    return _load_any4d_views(load_images, image_paths, runner["resolution_set"])


def build_any4d_chunk_specs(frame_indices, any4d_batch_size: int, overlap: int = 0):
    """Split frames into Any4D inference chunks.

    ``overlap`` > 0 makes consecutive chunks share their last/first ``overlap`` frames.
    Those shared frames are the SAME frame predicted by both chunks, so the ratio of the
    two predictions there is purely the per-chunk metric-scale ratio s̃_a/s̃_b (no camera
    motion, no dynamic-object confound) — used downstream to stitch the batches together.
    """
    frame_indices = [int(frame_idx) for frame_idx in frame_indices]
    if not frame_indices:
        raise ValueError("[Any4D] frame_indices is empty")
    if int(any4d_batch_size) < 1:
        raise ValueError(f"[Any4D] any4d_batch_size must be >= 1, got {any4d_batch_size}")
    B = int(any4d_batch_size)
    ov = max(0, int(overlap))
    if ov >= B:
        raise ValueError(f"[Any4D] overlap ({ov}) must be < any4d_batch_size ({B})")
    stride = B - ov
    n = len(frame_indices)

    batch_specs = []
    for batch_start in range(0, n, stride):
        batch_indices = frame_indices[batch_start : batch_start + B]
        if not batch_indices:
            break
        ref_frame_idx = int(batch_indices[len(batch_indices) // 2])
        batch_specs.append(
            {
                "batch_start": int(batch_start),
                "batch_indices": batch_indices,
                "ref_frame_idx": ref_frame_idx,
            }
        )
        if batch_start + B >= n:  # last chunk already reaches the end; avoid a tiny dup tail
            break
    return batch_specs


def iter_any4d_depth_sequence_batches(
    frame_indices,
    *,
    any4d_batch_size: int,
    build_views_for_chunk,
    runner,
    progress_desc: str | None = None,
    progress_disable: bool = False,
    timing_callback=None,
    prediction_view_offset: int = 2,
    overlap: int = 0,
):
    batch_specs = build_any4d_chunk_specs(frame_indices, any4d_batch_size, overlap=overlap)

    def _record(name: str, elapsed: float) -> None:
        if timing_callback is not None:
            timing_callback(str(name), float(elapsed))

    def _prepare(spec):
        prepared = build_views_for_chunk(
            list(spec["batch_indices"]),
            int(spec["ref_frame_idx"]),
        )
        if isinstance(prepared, tuple) and len(prepared) == 2:
            return prepared[0], prepared[1]
        return prepared, None

    with ThreadPoolExecutor(max_workers=1) as prefetcher:
        next_views_future = None
        for batch_idx, spec in enumerate(
            tqdm(batch_specs, total=len(batch_specs), desc=progress_desc, disable=progress_disable)
        ):
            if next_views_future is None:
                t_view_prep = time.time()
                views, batch_meta = _prepare(spec)
                _record("view_prep", time.time() - t_view_prep)
            else:
                t_view_wait = time.time()
                views, batch_meta = next_views_future.result()
                _record("view_prep", time.time() - t_view_wait)

            if batch_idx + 1 < len(batch_specs):
                next_views_future = prefetcher.submit(_prepare, batch_specs[batch_idx + 1])
            else:
                next_views_future = None

            t_forward = time.time()
            batch_depths = predict_any4d_depths_from_views(
                spec["batch_indices"],
                views,
                runner=runner,
                prediction_view_offset=prediction_view_offset,
            )
            _record("forward", time.time() - t_forward)
            batch_depths = np.asarray(batch_depths, dtype=np.float32)
            if batch_depths.shape[0] != len(spec["batch_indices"]):
                raise RuntimeError(
                    f"[Any4D] unexpected depth count {batch_depths.shape[0]} "
                    f"for chunk with {len(spec['batch_indices'])} target frames"
                )
            yield {
                "batch_start": int(spec["batch_start"]),
                "batch_indices": list(spec["batch_indices"]),
                "ref_frame_idx": int(spec["ref_frame_idx"]),
                "depths": batch_depths,
                "meta": batch_meta,
            }


def predict_any4d_depths_from_views(
    frame_indices,
    views,
    runner=None,
    *,
    any4d_repo_root=None,
    checkpoint_path=None,
    resolution_set=None,
    use_amp=None,
    prediction_view_offset: int = 1,
):
    frame_indices = [int(frame_idx) for frame_idx in frame_indices]
    if not frame_indices:
        raise ValueError("[Any4D] frame_indices is empty")
    if int(prediction_view_offset) < 1:
        raise ValueError(
            f"[Any4D] prediction_view_offset must be >= 1, got {prediction_view_offset}"
        )

    runner = runner or build_any4d_runner(
        any4d_repo_root=any4d_repo_root,
        checkpoint_path=checkpoint_path,
        resolution_set=resolution_set,
        use_amp=use_amp,
    )

    any4d_inference_test = runner["inference_module"]
    return _predict_depths_from_views(
        any4d_inference_test,
        runner,
        views,
        len(frame_indices),
        prediction_view_offset=int(prediction_view_offset),
    )


def build_any4d_runner(
    any4d_repo_root=None,
    checkpoint_path=None,
    resolution_set=None,
    use_amp=None,
    *,
    task=None,
    device=None,
):
    repo_root, checkpoint_path, resolution_set, use_amp = resolve_any4d_paths(
        PROJECT_ROOT,
        any4d_repo_root,
        checkpoint_path,
        resolution_set,
        use_amp,
    )
    ensure_exists(repo_root, "Any4D repo root")
    ensure_exists(checkpoint_path, "Any4D checkpoint")

    any4d_inference_test, load_images = _import_any4d_modules(repo_root)

    if task is None:
        task = os.environ.get("HAWOR_ANY4D_TASK", "images_only")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with _suppress_any4d_init_io():
        model = any4d_inference_test.init_inference_model(
            _any4d_config_dict(repo_root, task=str(task)),
            checkpoint_path,
            device,
        )

    return {
        "model": model,
        "inference_module": any4d_inference_test,
        "load_images": load_images,
        "repo_root": repo_root,
        "checkpoint_path": checkpoint_path,
        "resolution_set": resolution_set,
        "use_amp": use_amp,
        "task": str(task),
        "device": str(device),
    }


def predict_any4d_depth_batch(frame_source, frame_indices, runner=None, *, any4d_repo_root=None, checkpoint_path=None, resolution_set=None, use_amp=None, image_paths=None):
    frame_indices = [int(frame_idx) for frame_idx in frame_indices]
    if not frame_indices:
        raise ValueError("[Any4D] frame_indices is empty")

    runner = runner or build_any4d_runner(
        any4d_repo_root=any4d_repo_root,
        checkpoint_path=checkpoint_path,
        resolution_set=resolution_set,
        use_amp=use_amp,
    )

    views = build_any4d_views(
        frame_source,
        frame_indices,
        runner=runner,
        any4d_repo_root=any4d_repo_root,
        checkpoint_path=checkpoint_path,
        resolution_set=resolution_set,
        use_amp=use_amp,
        image_paths=image_paths,
    )
    return predict_any4d_depths_from_views(
        frame_indices,
        views,
        runner=runner,
        any4d_repo_root=any4d_repo_root,
        checkpoint_path=checkpoint_path,
        resolution_set=resolution_set,
        use_amp=use_amp,
        prediction_view_offset=2,
    )
