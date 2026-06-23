#!/usr/bin/env python3
"""Generate world_space_res.pth for FPHA using right-hand skeleton annotations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.clips.clip_manifest import load_clip_manifest
from lib.pipeline.hands.fpha_skeleton import build_fpha_right_hand_prediction
from lib.pipeline.proc.stage_api import get_stage_done_marker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FPHA world_space_res.pth from right-hand skeleton.")
    parser.add_argument("--descriptor_manifest", required=True, type=str)
    parser.add_argument("--skeleton_root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_iters", type=int, default=180)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--pose_reg", type=float, default=1e-4)
    parser.add_argument("--shape_reg", type=float, default=1e-3)
    parser.add_argument("--temporal_reg", type=float, default=1e-3)
    parser.add_argument("--shape_iters", type=int, default=120)
    parser.add_argument("--shape_sample_size", type=int, default=96)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve_existing_left", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _resolve_device(device_text: str) -> str:
    requested = torch.device(device_text)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return str(requested)


def _load_existing_left(world_res_path: Path, num_frames: int, preserve_existing_left: bool):
    zeros_trans = np.zeros((num_frames, 3), dtype=np.float32)
    zeros_rot = np.zeros((num_frames, 3), dtype=np.float32)
    zeros_hand_pose = np.zeros((num_frames, 45), dtype=np.float32)
    zeros_betas = np.zeros((num_frames, 10), dtype=np.float32)
    zeros_valid = np.zeros((num_frames,), dtype=np.float32)

    if not preserve_existing_left or not world_res_path.is_file():
        return zeros_trans, zeros_rot, zeros_hand_pose, zeros_betas, zeros_valid

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_res_path)
    arrays = [
        np.asarray(pred_trans),
        np.asarray(pred_rot),
        np.asarray(pred_hand_pose),
        np.asarray(pred_betas),
        np.asarray(pred_valid),
    ]
    expected_shapes = [
        (2, num_frames, 3),
        (2, num_frames, 3),
        (2, num_frames, 45),
        (2, num_frames, 10),
        (2, num_frames),
    ]
    if any(array.shape != expected_shape for array, expected_shape in zip(arrays, expected_shapes)):
        return zeros_trans, zeros_rot, zeros_hand_pose, zeros_betas, zeros_valid
    return (
        arrays[0][0].astype(np.float32),
        arrays[1][0].astype(np.float32),
        arrays[2][0].astype(np.float32),
        arrays[3][0].astype(np.float32),
        arrays[4][0].astype(np.float32),
    )


def _write_world_res(record, args, *, device: str) -> None:
    descriptor = record.descriptor
    seq_folder = Path(descriptor.seq_folder)
    seq_folder.mkdir(parents=True, exist_ok=True)
    world_res_path = seq_folder / "world_space_res.pth"

    num_frames = int(descriptor.frame_count)
    extra = descriptor.extra or {}
    right_prediction = build_fpha_right_hand_prediction(
        clip_id=record.clip_id,
        seq_folder=seq_folder,
        num_frames=num_frames,
        tar_root=descriptor.root_dir,
        skeleton_root=args.skeleton_root,
        subject=extra.get("subject"),
        action=extra.get("action"),
        trial=extra.get("trial"),
        device=device,
        num_iters=args.num_iters,
        lr=args.lr,
        pose_reg=args.pose_reg,
        shape_reg=args.shape_reg,
        temporal_reg=args.temporal_reg,
        shape_iters=args.shape_iters,
        shape_sample_size=args.shape_sample_size,
        chunk_size=args.chunk_size,
    )
    left_trans, left_rot, left_hand_pose, left_betas, left_valid = _load_existing_left(
        world_res_path,
        num_frames,
        bool(args.preserve_existing_left),
    )

    payload = [
        np.stack([left_trans, right_prediction["pred_trans"]], axis=0).astype(np.float32),
        np.stack([left_rot, right_prediction["pred_rot"]], axis=0).astype(np.float32),
        np.stack([left_hand_pose, right_prediction["pred_hand_pose"]], axis=0).astype(np.float32),
        np.stack([left_betas, right_prediction["pred_betas"]], axis=0).astype(np.float32),
        np.stack([left_valid, right_prediction["pred_valid"]], axis=0).astype(np.float32),
    ]
    joblib.dump(payload, world_res_path)
    get_stage_done_marker(seq_folder, "infiller").touch()


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = Path(args.descriptor_manifest).resolve()
    records = load_clip_manifest(manifest_path)
    device = _resolve_device(args.device)

    failures = []
    completed = 0
    skipped = 0
    started_at = time.perf_counter()
    for record in tqdm(records, desc="FPHA skeleton->world_res", unit="clip"):
        seq_folder = Path(record.descriptor.seq_folder)
        world_res_path = seq_folder / "world_space_res.pth"
        done_marker = get_stage_done_marker(seq_folder, "infiller")
        if args.resume and world_res_path.is_file() and done_marker.exists():
            skipped += 1
            continue
        try:
            _write_world_res(record, args, device=device)
            completed += 1
        except Exception as exc:
            failures.append({"clip_id": record.clip_id, "seq_folder": str(seq_folder), "error": str(exc)})

    summary = {
        "manifest": str(manifest_path),
        "device": device,
        "total": len(records),
        "completed": completed,
        "skipped": skipped,
        "failed": len(failures),
        "failed_preview": failures[:16],
        "elapsed_sec": time.perf_counter() - started_at,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
