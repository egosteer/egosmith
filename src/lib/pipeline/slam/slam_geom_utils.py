"""Geometry helpers shared by stage3 SLAM backends."""

import cv2
import numpy as np


def est_calib(frame_source):
    """Roughly estimate intrinsics from image dimensions."""
    image = frame_source.get_frame(0, rgb=False)
    h0, w0 = image.shape[:2]
    focal = float(np.max([h0, w0]))
    cx, cy = float(w0) / 2.0, float(h0) / 2.0
    return [focal, focal, cx, cy]


def get_dimention(frame_source):
    """Return the resized (H, W) used by stage-3 frame processing."""
    image = frame_source.get_frame(0, rgb=False)
    h0, w0 = image.shape[:2]
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
    image = cv2.resize(image, (w1, h1))
    image = image[: h1 - h1 % 8, : w1 - w1 % 8]
    height, width = image.shape[:2]
    return height, width
