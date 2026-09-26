"""Camera-space hand overlay — a dependency-light correctness check.

Projects the world-space hand reconstruction (`result.npz`, or the legacy
`world_space_res.pth`) back into
each video frame through the SLAM camera using the recorded pinhole intrinsics
(`img_focal`, `img_center`) and draws the MANO vertices on top. This is a direct
K-projection (no aitviewer / OpenGL), so it faithfully reflects what the pipeline
actually produced — use it to verify reconstruction correctness.

Why this exists: it checks a finished run directly from its pose file
without re-running reconstruction or any renderer, and the projected hand lands
exactly where the pipeline placed it — so it faithfully reflects pipeline output.

Usage:
    python scripts/overlay_hand_cam.py --seq_folder /path/to/stage_outputs/<video>
    # writes <seq_folder>/overlay_hand_cam.mp4
    # orchestrated `video:` run: frames live outside the seq folder
    python scripts/overlay_hand_cam.py --seq_folder <output_root>/stage_outputs/<clip_id> \
        --frames_dir <output_root>/frames/<clip_id>

The seq folder must contain (produced by the infer stage):
    - result.npz (or the legacy world_space_res.pth)
    - SLAM/hawor_slam_w_scale_<start>_<end>.npz
The clip's frames are read as *.jpg from --frames_dir, default <seq_folder>/extracted_images/
(the demo.py / scripts/extract_frames.py layout). An orchestrated run keeps its frames under the
adapter's frames root instead (<output_root>/frames/<clip_id>/ for a single-video config), so pass
that directory. Frame t of the sorted *.jpg list is drawn with pose frame t, so the directory must
hold exactly the frames the poses were estimated on (one per pose frame, from the clip's first frame).

A hand is drawn only on frames where its ``pred_valid`` flag is set; invalid frames (and a
hand that was never observed) are skipped instead of being drawn at the world origin.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

# Run from anywhere: put src/ (first-party packages) and the repo root on sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_PROJECT_ROOT, "src"), _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.io.result_io import final_artifact_exists, load_pose_arrays
from lib.pipeline.slam.slam_cam import load_slam_cam
from lib.pipeline.hands.mano_runtime import run_mano, run_mano_left

# (B, G, R) — right hand green, left hand blue.
_RIGHT_COLOR = (0, 255, 0)
_LEFT_COLOR = (255, 0, 0)
_HAND_IDX = {"right": 1, "left": 0}


def _resolve_slam_npz(seq_folder: str) -> str:
    matches = sorted(glob.glob(os.path.join(seq_folder, "SLAM", "hawor_slam_w_scale_*.npz")))
    if not matches:
        raise FileNotFoundError(
            f"[overlay] no SLAM export under {seq_folder}/SLAM/ "
            "(expected hawor_slam_w_scale_<start>_<end>.npz; run the slam stage first)"
        )
    return matches[0]


def _hand_valid_masks(pred_valid, which, num_frames: int) -> dict:
    """Per-hand boolean mask over the first ``num_frames`` frames (``pred_valid`` is (2, T), 0=left, 1=right)."""
    valid = np.asarray(pred_valid).astype(bool)
    return {hand: valid[_HAND_IDX[hand], :num_frames] for hand in which}


def _as_float_tensor(value) -> torch.Tensor:
    # result.npz yields numpy arrays, the legacy .pth torch tensors
    return value.float() if torch.is_tensor(value) else torch.as_tensor(np.asarray(value)).float()


def _project(verts_t: torch.Tensor, R_w2c_t: torch.Tensor, t_w2c_t: torch.Tensor,
             focal: float, cx: float, cy: float):
    """World verts (N,3) -> pixel (u, v) via the SLAM camera + pinhole K."""
    cam = (R_w2c_t @ verts_t.T).T + t_w2c_t            # (N, 3) camera space
    z = cam[:, 2].clamp(min=1e-3)
    u = (focal * cam[:, 0] / z + cx).numpy()
    v = (focal * cam[:, 1] / z + cy).numpy()
    return u, v


def main(argv=None):
    parser = argparse.ArgumentParser(description="Overlay reconstructed hands on video via direct K-projection.")
    parser.add_argument("--seq_folder", required=True, help="Stage output folder for one video (has result.npz or world_space_res.pth, and SLAM/).")
    parser.add_argument("--frames_dir", default=None, help="Directory of the clip's *.jpg frames (default: <seq_folder>/extracted_images).")
    parser.add_argument("--out", default=None, help="Output mp4 path (default: <seq_folder>/overlay_hand_cam.mp4).")
    parser.add_argument("--fps", type=float, default=15.0, help="Output video FPS (default 15).")
    parser.add_argument("--hands", choices=["both", "left", "right"], default="both")
    parser.add_argument("--radius", type=int, default=1, help="Vertex dot radius in pixels.")
    args = parser.parse_args(argv)

    seq = args.seq_folder
    if not final_artifact_exists(seq):
        raise FileNotFoundError(f"[overlay] no result.npz or world_space_res.pth under {seq}")
    slam_npz = _resolve_slam_npz(seq)
    frames_dir = args.frames_dir or os.path.join(seq, "extracted_images")
    jpgs = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not jpgs:
        where = "directory does not exist" if not os.path.isdir(frames_dir) else "directory has no *.jpg files"
        raise FileNotFoundError(
            f"[overlay] no frames at {frames_dir} ({where}). Pass --frames_dir <dir> holding the clip's "
            "frames as *.jpg, one per pose frame (frame t = pose frame t). An orchestrated run keeps them under the "
            "adapter's frames root, e.g. <output_root>/frames/<clip_id>/ for a single-video config; "
            "<seq_folder>/extracted_images/ is the demo.py / scripts/extract_frames.py layout"
        )

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = (
        _as_float_tensor(x) if i < 4 else x for i, x in enumerate(load_pose_arrays(seq))
    )
    R_w2c, t_w2c, _, _ = load_slam_cam(slam_npz)
    R_w2c = R_w2c.float().cpu()
    t_w2c = t_w2c.float().cpu()

    cam_npz = np.load(slam_npz)
    focal = float(cam_npz["img_focal"])
    cx, cy = (float(x) for x in cam_npz["img_center"])

    num_frames = min(len(jpgs), pred_trans.shape[1], R_w2c.shape[0])
    height, width = cv2.imread(jpgs[0]).shape[:2]

    which = ["right", "left"] if args.hands == "both" else [args.hands]
    valid_by_hand = _hand_valid_masks(pred_valid, which, num_frames)
    verts_by_hand = {}
    for hand in which:
        hi = _HAND_IDX[hand]
        runner = run_mano if hand == "right" else run_mano_left
        with torch.no_grad():
            out = runner(pred_trans[hi:hi + 1], pred_rot[hi:hi + 1], pred_hand_pose[hi:hi + 1], betas=pred_betas[hi:hi + 1])
        verts_by_hand[hand] = out["vertices"][0].detach().cpu().float()  # (T, 778, 3)

    out_path = args.out or os.path.join(seq, "overlay_hand_cam.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"[overlay] could not open VideoWriter for {out_path}")

    try:
        for t in range(num_frames):
            img = cv2.imread(jpgs[t])
            for hand in which:
                if not valid_by_hand[hand][t]:
                    continue  # invalid / never-observed frame: zero params would draw a hand at the origin
                color = _RIGHT_COLOR if hand == "right" else _LEFT_COLOR
                u, v = _project(verts_by_hand[hand][t], R_w2c[t], t_w2c[t], focal, cx, cy)
                for x, y in zip(u, v):
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(img, (int(x), int(y)), args.radius, color, -1)
            writer.write(img)
    finally:
        writer.release()

    print(f"[overlay] wrote {out_path}  ({num_frames} frames, {width}x{height}, focal={focal:.1f})")


if __name__ == "__main__":
    main()
