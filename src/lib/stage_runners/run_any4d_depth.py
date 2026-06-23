"""
Any4D depth inference for EgoSmith SLAM (in-process; also runnable as CLI).

Loaded by ``hawor_slam._predict_depths_batched`` — not a subprocess wrapper.
"""
import argparse
import contextlib
import os
import sys
from typing import Any, Optional, Tuple


def parse_frame_indices(s: str):
    """
    Parse comma-separated frame indices like: "0,5,10"
    """
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def ensure_exists(path: str, what: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Any4D] {what} not found: {path}")


def _env_flag_on(name: str, *, default_on: bool = True) -> bool:
    """
    Truthy env: 1/true/yes/on (case-insensitive) or any other non-empty value
    except 0/false/no/off. Unset uses ``default_on``.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default_on
    v = str(raw).strip().lower()
    if v == "":
        return default_on
    if v in ("0", "false", "no", "off"):
        return False
    return True


@contextlib.contextmanager
def _suppress_any4d_init_io():
    """
    Hide Any4D ``init_model`` / torch.hub / checkpoint ``print`` spam during load.

    Set ``HAWOR_ANY4D_VERBOSE=1`` to show full logs (debug).
    """
    if os.environ.get("HAWOR_ANY4D_VERBOSE", "0").strip() == "1":
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as dn:
        with contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            yield


def _any4d_config_dict(any4d_root: str) -> dict:
    """Hydra-style config for EgoSmith Any4D depth."""
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
            "model/task=images_only",
        ],
    }


def run_any4d_depth_batch(
    video_path: str,
    frame_indices,
    any4d_repo_root: str,
    output_depth_npz: Optional[str] = None,
    checkpoint_path: str = "",
    resolution_set: int = 518,
    use_amp: Optional[bool] = None,
    model=None,
) -> Tuple[Any, Any]:
    """
    Run Any4D depth inference for a batch of frame indices; optionally save depths to .npz.

    Expects EgoSmith `extracted_images` next to the video stem folder (same as CLI).

    Args:
        video_path: Original video path (used to locate extracted_images/).
        frame_indices: Iterable of int frame indices (same order as DPVO keyframes).
        output_depth_npz: If set, write npz with key ``depths`` [N,H,W] float32.
            If ``None``, skip disk write (caller merges full-segment cache).
        any4d_repo_root: Path to Any4D repository root.
        checkpoint_path: Path to .pth checkpoint; if empty, uses
            ``<any4d_repo_root>/checkpoints/any4d_4v_combined.pth``.
        resolution_set: Any4D internal resize (default 518).
        use_amp: AMP for Any4D; ``None`` → ``HAWOR_ANY4D_USE_AMP`` (default on).
        model: Optional already-loaded Any4D module from a previous call in the same
            process. When provided, skips checkpoint load (same numerics as reloading).

    Returns:
        ``(model, depth_stack)`` — model for reuse across batches; ``depth_stack`` float32
        array shaped ``[N, H, W]``.
    """
    import cv2  # noqa: F401 (used implicitly by Any4D image loader)
    import numpy as np
    import torch

    if use_amp is None:
        use_amp = _env_flag_on("HAWOR_ANY4D_USE_AMP", default_on=True)

    if not frame_indices:
        raise ValueError("[Any4D] frame_indices is empty")

    frame_indices = [int(x) for x in frame_indices]

    any4d_root = os.path.abspath(any4d_repo_root)
    ensure_exists(any4d_root, "Any4D repo root")

    if checkpoint_path:
        ckpt_path = checkpoint_path
    else:
        ckpt_path = os.path.join(any4d_root, "checkpoints", "any4d_4v_combined.pth")
    ckpt_path = os.path.abspath(ckpt_path)
    ensure_exists(ckpt_path, "Any4D checkpoint")

    # Add Any4D repo root so its package imports resolve.
    sys.path.insert(0, any4d_root)

    # init_inference_model / sample_inference are upstream Any4D code (Apache-2.0),
    # carried first-party in lib.pipeline.slam.any4d_inference so EgoSmith does not
    # depend on Any4D's scripts/ directory.
    from lib.pipeline.slam import any4d_inference as any4d_inference_test
    from any4d.utils.image import load_images

    if output_depth_npz:
        out_dir = os.path.dirname(os.path.abspath(output_depth_npz))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    ref_frame_idx = frame_indices[len(frame_indices) // 2]

    video_path_obj = os.path.abspath(video_path)
    video_dir = os.path.dirname(video_path_obj)
    video_stem = os.path.splitext(os.path.basename(video_path_obj))[0]
    extracted_dir = os.path.join(video_dir, video_stem, "extracted_images")
    if not os.path.isdir(extracted_dir):
        raise FileNotFoundError(
            f"[Any4D] extracted_images not found: {extracted_dir}\n"
            f"Run (from repo root): python scripts/extract_frames.py --video_path {video_path}"
        )

    def frame_to_img_path(i: int) -> str:
        """Resolve frame file: EgoSmith ``extract_frames`` uses 6-digit stems; BuildAI uses 4-digit."""
        idx = int(i)
        candidates = [
            (f"{idx:06d}.jpg", f"{idx:06d}.png"),
        ]
        if idx < 10000:
            candidates.append((f"{idx:04d}.jpg", f"{idx:04d}.png"))
        tried = []
        for jpg_name, png_name in candidates:
            jpg_path = os.path.join(extracted_dir, jpg_name)
            png_path = os.path.join(extracted_dir, png_name)
            tried.extend([jpg_path, png_path])
            if os.path.exists(jpg_path):
                return jpg_path
            if os.path.exists(png_path):
                return png_path
        raise FileNotFoundError(
            f"[Any4D] frame file missing for idx={idx}: tried " + " / ".join(tried[:4]) + (" ..." if len(tried) > 4 else "")
        )

    image_paths = [frame_to_img_path(ref_frame_idx)] + [
        frame_to_img_path(i) for i in frame_indices
    ]

    views = load_images(
        image_paths,
        resize_mode="fixed_mapping",
        resolution_set=resolution_set,
        norm_type="dinov2",
        patch_size=14,
        verbose=False,
        compute_moge_mask=False,
        binary_mask_path=None,
    )

    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        any4d_config = _any4d_config_dict(any4d_root)
        with _suppress_any4d_init_io():
            model = any4d_inference_test.init_inference_model(
                any4d_config, ckpt_path, device
            )
    else:
        device = str(next(model.parameters()).device)

    pred_result = any4d_inference_test.sample_inference(
        model=model, views=views, device=device, use_amp=use_amp
    )

    depth_list = []
    for target_i in range(len(frame_indices)):
        view_idx = 1 + target_i
        depth_z = (
            pred_result[f"pred{view_idx}"]["pts3d_cam"][..., 2:3][0]
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )
        depth_list.append(depth_z.astype(np.float32))

    depth_stack = np.stack(depth_list, axis=0)
    if output_depth_npz:
        np.savez(output_depth_npz, depths=depth_stack)
    return model, depth_stack


def main():
    parser = argparse.ArgumentParser(
        description="CLI entry: run one Any4D depth batch (same logic as hawor_slam import path)."
    )
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument(
        "--frame_indices", type=str, required=True, help="Comma-separated list"
    )
    parser.add_argument(
        "--output_depth_npz", type=str, required=True, help="Output npz path"
    )
    parser.add_argument("--any4d_repo_root", type=str, required=True)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Any4D checkpoint .pth file path",
    )
    parser.add_argument("--resolution_set", type=int, default=518)
    _amp_default = _env_flag_on("HAWOR_ANY4D_USE_AMP", default_on=True)
    _amp = parser.add_mutually_exclusive_group()
    _amp.add_argument(
        "--use_amp",
        dest="use_amp",
        action="store_true",
        default=None,
        help="Force AMP on (default: env HAWOR_ANY4D_USE_AMP, else on)",
    )
    _amp.add_argument(
        "--no_amp",
        dest="use_amp",
        action="store_false",
        help="Force AMP off",
    )
    args = parser.parse_args()
    use_amp = _amp_default if args.use_amp is None else bool(args.use_amp)

    _, _ = run_any4d_depth_batch(
        video_path=args.video_path,
        frame_indices=parse_frame_indices(args.frame_indices),
        any4d_repo_root=args.any4d_repo_root,
        output_depth_npz=args.output_depth_npz,
        checkpoint_path=args.checkpoint_path,
        resolution_set=args.resolution_set,
        use_amp=use_amp,
    )


if __name__ == "__main__":
    main()
