#!/usr/bin/env python3
"""
Extract frames from videos using Decord for fast parallel processing.

This script extracts all frames from videos and saves them as JPG images,
which can significantly speed up subsequent processing stages by eliminating
video decoding overhead and I/O contention.

Usage:
    # Single video
    python scripts/extract_frames.py --video_path path/to/video.mp4

    # Batch processing
    python scripts/extract_frames.py --video_list videos.txt --num_workers 8

    # Custom output directory and quality
    python scripts/extract_frames.py --video_path video.mp4 --output_dir ./frames --quality 95
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import multiprocessing as mp
from tqdm import tqdm
import cv2
import numpy as np


def extract_frames_decord(
    video_path: str,
    output_dir: str,
    quality: int = 95,
    format: str = "jpg",
    verbose: bool = True,
) -> int:
    """
    Extract all frames from a video using Decord.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        quality: JPEG quality (1-100, higher = better quality but larger files)
        format: Output format ('jpg' or 'png')
        verbose: Print progress information

    Returns:
        Number of frames extracted
    """
    try:
        from decord import VideoReader, cpu
    except ImportError:
        print("ERROR: decord not installed. Install with: pip install decord", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video with Decord
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
    except Exception as e:
        print(f"ERROR: Failed to open video {video_path}: {e}", file=sys.stderr)
        return 0

    if verbose:
        print(f"Extracting {num_frames} frames from {os.path.basename(video_path)}")
        print(f"Output directory: {output_dir}")

    # Set up JPEG encoding parameters
    if format == "jpg":
        # Optimize JPEG encoding for speed
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 0,  # Disable optimization for speed
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0,  # Disable progressive for speed
        ]
        ext = ".jpg"
    elif format == "png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 11)]  # Convert quality to compression
        ext = ".png"
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Extract frames in batches for better performance
    batch_size = 128  # Increased from 64 for better throughput
    num_batches = (num_frames + batch_size - 1) // batch_size

    frames_extracted = 0

    # Use tqdm to show progress in frames (not batches)
    if verbose:
        pbar = tqdm(total=num_frames, desc=f"Extracting {os.path.basename(video_path)}", unit="frames")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_frames)

        # Decord can efficiently load multiple frames at once
        indices = list(range(start_idx, end_idx))
        frames = vr.get_batch(indices).asnumpy()  # Shape: (batch, H, W, C) in RGB

        # Save each frame
        for i, frame_idx in enumerate(indices):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)

            # Save with zero-padded filename (e.g., 000000.jpg)
            frame_filename = f"{frame_idx:06d}{ext}"
            frame_path = os.path.join(output_dir, frame_filename)

            success = cv2.imwrite(frame_path, frame_bgr, encode_params)
            if success:
                frames_extracted += 1
            else:
                print(f"WARNING: Failed to save frame {frame_idx}", file=sys.stderr)

        # Update progress bar
        if verbose:
            pbar.update(len(indices))

    if verbose:
        pbar.close()

    if verbose:
        print(f"✓ Extracted {frames_extracted}/{num_frames} frames")

    return frames_extracted


def process_video_worker(args_tuple):
    """Worker function for multiprocessing."""
    video_path, quality, format, verbose = args_tuple

    # Match official format: <video_dir>/<video_stem>/extracted_images/
    video_dir = Path(video_path).parent
    video_stem = Path(video_path).stem
    output_dir = video_dir / video_stem / "extracted_images"

    # Skip if already extracted
    if os.path.exists(output_dir):
        existing_frames = len([f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png'))])
        if existing_frames > 0:
            if verbose:
                print(f"Skipping {video_stem}: {existing_frames} frames already exist")
            return video_path, existing_frames, True

    try:
        num_frames = extract_frames_decord(str(video_path), str(output_dir), quality, format, verbose=False)
        return video_path, num_frames, True
    except Exception as e:
        print(f"ERROR processing {video_path}: {e}", file=sys.stderr)
        return video_path, 0, False


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos using Decord",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video_path",
        type=str,
        help="Path to a single video file"
    )
    input_group.add_argument(
        "--video_list",
        type=str,
        help="Path to text file with one video path per line"
    )

    # Output options (note: output directory is determined by video path)
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95, higher = better quality but slower)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: use quality=85 for 2-3x speedup with minimal quality loss"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Output image format (default: jpg)"
    )

    # Processing options
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for batch processing (default: 1)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index in video list (for distributed processing across machines)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index in video list (for distributed processing across machines)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos that already have extracted frames"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Collect video paths
    video_paths: List[str] = []
    if args.video_path:
        video_paths = [args.video_path]
    elif args.video_list:
        with open(args.video_list, 'r') as f:
            video_paths = [line.strip() for line in f if line.strip()]

    if not video_paths:
        print("ERROR: No videos to process", file=sys.stderr)
        sys.exit(1)

    # Apply start/end slicing for distributed processing
    total_videos = len(video_paths)
    if args.start is not None or args.end is not None:
        start_idx = args.start if args.start is not None else 0
        end_idx = args.end if args.end is not None else total_videos

        # Validate indices
        if start_idx < 0 or start_idx >= total_videos:
            print(f"ERROR: --start {start_idx} is out of range [0, {total_videos})", file=sys.stderr)
            sys.exit(1)
        if end_idx < start_idx or end_idx > total_videos:
            print(f"ERROR: --end {end_idx} is invalid (must be in range [{start_idx}, {total_videos}])", file=sys.stderr)
            sys.exit(1)

        video_paths = video_paths[start_idx:end_idx]
        print(f"Processing slice [{start_idx}:{end_idx}] of {total_videos} total videos ({len(video_paths)} videos)")
    else:
        print(f"Processing all {total_videos} videos")

    # Validate video paths
    valid_paths = []
    for vp in video_paths:
        if not os.path.exists(vp):
            print(f"WARNING: Video not found: {vp}", file=sys.stderr)
        else:
            valid_paths.append(vp)

    if not valid_paths:
        print("ERROR: No valid video paths found", file=sys.stderr)
        sys.exit(1)

    # Apply fast mode if requested
    quality = args.quality
    if args.fast:
        quality = 85
        print("Fast mode enabled: using quality=85 for faster extraction")

    print(f"Output format: <video_dir>/<video_stem>/extracted_images/")
    print(f"Quality: {quality}, Format: {args.format}")
    print(f"Workers: {args.num_workers}")
    print()

    # Process videos
    if len(valid_paths) == 1 or args.num_workers == 1:
        # Single-threaded processing
        for video_path in valid_paths:
            # Match official format: <video_dir>/<video_stem>/extracted_images/
            video_dir = Path(video_path).parent
            video_stem = Path(video_path).stem
            output_dir = video_dir / video_stem / "extracted_images"

            if args.skip_existing and os.path.exists(output_dir):
                existing = len([f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png'))])
                if existing > 0:
                    print(f"Skipping {video_stem}: {existing} frames already exist")
                    continue

            extract_frames_decord(
                str(video_path),
                str(output_dir),
                quality=quality,
                format=args.format,
                verbose=not args.quiet
            )
    else:
        # Multi-threaded processing
        worker_args = [
            (vp, quality, args.format, not args.quiet)
            for vp in valid_paths
        ]

        with mp.Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_video_worker, worker_args),
                total=len(worker_args),
                desc="Processing videos",
                disable=args.quiet
            ))

        # Print summary
        successful = sum(1 for _, _, success in results if success)
        total_frames = sum(num for _, num, _ in results)
        print(f"\n{'='*60}")
        print(f"Extraction complete:")
        print(f"  Videos processed: {successful}/{len(valid_paths)}")
        print(f"  Total frames: {total_frames}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
