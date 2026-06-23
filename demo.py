#!/usr/bin/env python3
"""
Single-video reconstruction + visualization for EgoSmith — works on any headless server.

Runs detect -> motion -> SLAM -> infiller on one video, then overlays the reconstructed world-space
hands back onto each frame with OpenCV (a direct pinhole K-projection — no OpenGL / pyrender /
aitviewer) and writes an mp4. This is the same projection as scripts/inspection/overlay_hand_cam.py
(which works from a finished run); demo.py just runs the whole pipeline first.

Reads pre-extracted frames, so run `python scripts/extract_frames.py --video_path <video>` first.

Usage:
    python scripts/extract_frames.py --video_path video.mp4
    python demo.py --video_path video.mp4
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# src-layout: first-party packages live under src/; scripts/ stays importable from root.
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.hands.mano_runtime import get_mano_faces, run_mano, run_mano_left
from lib.pipeline.slam.slam_cam import load_slam_cam
from lib.stage_runners.detect_track_video import detect_track_video
from lib.stage_runners.hawor_slam import hawor_slam
from lib.stage_runners.hawor_video import hawor_infiller, hawor_motion_estimation


def project_to_image(vertices, R_w2c, t_w2c, focal, cx, cy):
    """World verts (N,3) -> (N,3) of (u, v, depth) via the SLAM camera + pinhole K.

    Matches scripts/inspection/overlay_hand_cam.py: a plain pinhole projection using the SLAM-recorded
    focal and principal point (img_focal / img_center) — not an image-center approximation.
    """
    cam = vertices @ R_w2c.T + t_w2c
    z = np.clip(cam[:, 2], 1e-3, None)
    u = focal * cam[:, 0] / z + cx
    v = focal * cam[:, 1] / z + cy
    return np.stack([u, v, cam[:, 2]], axis=1)


def render_frame(vertices_left, vertices_right, faces_left, faces_right, bg_image,
                 R_w2c, t_w2c, focal, cx, cy):
    """Overlay both hand meshes onto one RGB frame via OpenCV fillPoly."""
    result = bg_image.copy()
    overlay = np.zeros_like(result)
    mask = np.zeros(result.shape[:2], dtype=np.uint8)

    for verts, faces, color in (
        (vertices_right, faces_right, (202, 152, 53)),   # right hand — blue (BGR)
        (vertices_left, faces_left, (200, 100, 128)),    # left hand — purple (BGR)
    ):
        if verts is None or len(verts) == 0:
            continue
        pts_2d = project_to_image(verts, R_w2c, t_w2c, focal, cx, cy)
        for face in faces:
            # Only draw faces in front of the camera (positive depth).
            if pts_2d[face, 2].mean() > 0:
                poly = pts_2d[face, :2].astype(np.int32)
                cv2.fillPoly(overlay, [poly], color=color)
                cv2.fillPoly(mask, [poly], color=255)

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    return (overlay * mask_3ch * 0.7 + result * (1 - mask_3ch * 0.7)).astype(np.uint8)


def render_overlay_video(left_verts, right_verts, faces_left, faces_right,
                         frame_source, frame_indices, R_w2c, t_w2c, focal, cx, cy,
                         output_path, fps=30):
    """Overlay the world-space hands onto each video frame through the SLAM camera."""
    print("Rendering hand overlay...")
    first_img = frame_source.get_frame(int(frame_indices[0]), rgb=False)
    height, width = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_idx in tqdm(range(left_verts.shape[0]), desc="Rendering frames"):
        bg_img = frame_source.get_frame(int(frame_indices[frame_idx]), rgb=True)
        frame = render_frame(
            left_verts[frame_idx], right_verts[frame_idx],
            faces_left, faces_right, bg_img,
            R_w2c[frame_idx], t_w2c[frame_idx], focal, cx, cy,
        )
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"✓ Video saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Single-video HaWoR reconstruction + hand overlay (OpenCV).")
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--input_type", type=str, default="file")
    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str, default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--fps", type=int, default=30, help="FPS for the output video.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to render (default: all).")
    args = parser.parse_args()

    # Run inference pipeline (reuses existing per-clip outputs when present).
    print("=== Running HaWoR inference ===")
    start_idx, end_idx, seq_folder, frame_source = detect_track_video(args)
    frame_chunks_all, _img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)

    R_w2c_all, t_w2c_all, _R_c2w_all, _t_c2w_all = load_slam_cam(slam_path)
    R_w2c_all = R_w2c_all.float().cpu().numpy()
    t_w2c_all = t_w2c_all.float().cpu().numpy()

    # Camera intrinsics straight from the SLAM export (same source as overlay_hand_cam.py).
    cam_npz = np.load(slam_path)
    focal = float(cam_npz["img_focal"])
    cx, cy = (float(v) for v in cam_npz["img_center"])

    pred_trans, pred_rot, pred_hand_pose, pred_betas, _pred_valid = hawor_infiller(
        args, start_idx, end_idx, frame_chunks_all
    )

    print("\n=== Preparing hand meshes ===")
    vis_start = 0
    vis_end = pred_trans.shape[1] - 1
    if args.max_frames is not None:
        vis_end = min(vis_start + args.max_frames - 1, vis_end)
        print(f"Limiting visualization to first {args.max_frames} frames (0 to {vis_end})")

    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239],
                          [239, 122, 279], [122, 118, 279], [279, 118, 215],
                          [118, 117, 215], [215, 117, 214], [117, 119, 214],
                          [214, 119, 121], [119, 120, 121], [121, 120, 78],
                          [120, 108, 78], [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]

    # World-space MANO vertices (project directly through the SLAM camera; no extra axis flip).
    pred_glob_r = run_mano(
        pred_trans[1:2, vis_start:vis_end], pred_rot[1:2, vis_start:vis_end],
        pred_hand_pose[1:2, vis_start:vis_end], betas=pred_betas[1:2, vis_start:vis_end],
    )
    right_verts = pred_glob_r["vertices"][0].cpu().numpy()
    pred_glob_l = run_mano_left(
        pred_trans[0:1, vis_start:vis_end], pred_rot[0:1, vis_start:vis_end],
        pred_hand_pose[0:1, vis_start:vis_end], betas=pred_betas[0:1, vis_start:vis_end],
    )
    left_verts = pred_glob_l["vertices"][0].cpu().numpy()

    output_dir = Path(seq_folder) / f"demo_overlay_{vis_start}_{vis_end}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "visualization.mp4"
    frame_indices = np.arange(vis_start, vis_end, dtype=np.int64)

    print("\n=== Rendering hand overlay ===")
    render_overlay_video(
        left_verts, right_verts, faces_left, faces_right,
        frame_source, frame_indices,
        R_w2c_all[vis_start:vis_end], t_w2c_all[vis_start:vis_end],
        focal, cx, cy, output_video, fps=args.fps,
    )

    print(f"\n✓ Done! Video saved to: {output_video}")
    print(f"\nTo view: scp user@server:{output_video} ./")


if __name__ == "__main__":
    main()
