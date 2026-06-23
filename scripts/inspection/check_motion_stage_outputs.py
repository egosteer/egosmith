#!/usr/bin/env python3
"""Diagnose one clip across motion cam-space, world-space, and optional WDS lowdim export."""

import argparse
import json
import re
import sys
import tarfile
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.exporters.manifest_build.resample import (
    resample_episode_features,
)
from lib.pipeline.exporters.webdataset_features import (
    _build_lowdim_features,
    _compute_joint_states,
    _compute_presence_per_frame,
    _load_episode_camera_features,
    _load_world_space_prediction,
    build_mano_models,
    export_frame_count_with_action,
)
from lib.pipeline.quality.quality_metrics import decode_lowdim
from lib.pipeline.proc.stage_api import get_track_range
from lib.pipeline.quality.wds_sanity import LOWDIM_DIMENSION_NAMES


FRAME_INDEX_RE = re.compile(r"_f(\d+)$")
WRIST_STATE_SLICE = slice(0, 18)
HAND_STATE_SLICE = slice(18, 48)
WRIST_ACTION_SLICE = slice(48, 66)
HAND_ACTION_SLICE = slice(66, 96)
EXTRINSIC_SLICE = slice(96, 112)
INTRINSIC_SLICE = slice(112, 116)
WDS_MEMBER_SUFFIXES = {
    ".image.jpg": "image_bytes",
    ".lowdim.npy": "lowdim_bytes",
    ".meta.json": "meta_bytes",
    ".mano.npy": "mano_bytes",
}
ROT6D_UNIT_NORM_TOL = 0.2
ROT6D_ORTHOGONALITY_TOL = 0.2
ROT6D_MIN_CROSS_NORM = 0.5
EXTRINSIC_BOTTOM_ROW_TOL = 1e-3
EXTRINSIC_ROTATION_ORTHO_FROB_TOL = 0.2
EXTRINSIC_ROTATION_DET_TOL = 0.2

try:
    from lib.pipeline.quality.quality_metrics import validate_lowdim_numeric_sanity  # type: ignore
except ImportError:
    def _rot6d_is_sane(rot6d: np.ndarray) -> bool:
        array = np.asarray(rot6d, dtype=np.float32).reshape(-1)
        if array.shape != (6,) or not np.isfinite(array).all():
            return False
        col_a = array[:3]
        col_b = array[3:]
        norm_a = float(np.linalg.norm(col_a))
        norm_b = float(np.linalg.norm(col_b))
        if norm_a <= 1e-8 or norm_b <= 1e-8:
            return False
        if abs(norm_a - 1.0) > ROT6D_UNIT_NORM_TOL or abs(norm_b - 1.0) > ROT6D_UNIT_NORM_TOL:
            return False
        unit_a = col_a / norm_a
        unit_b = col_b / norm_b
        if abs(float(np.dot(unit_a, unit_b))) > ROT6D_ORTHOGONALITY_TOL:
            return False
        if float(np.linalg.norm(np.cross(unit_a, unit_b))) < ROT6D_MIN_CROSS_NORM:
            return False
        return True

    def _extrinsic_is_sane(extrinsic: np.ndarray) -> bool:
        matrix = np.asarray(extrinsic, dtype=np.float32).reshape(4, 4)
        if not np.isfinite(matrix).all():
            return False
        if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=EXTRINSIC_BOTTOM_ROW_TOL):
            return False
        rotation = matrix[:3, :3].astype(np.float64)
        det = float(np.linalg.det(rotation))
        if not np.isfinite(det) or abs(det - 1.0) > EXTRINSIC_ROTATION_DET_TOL:
            return False
        ortho_err = float(np.linalg.norm(rotation.T @ rotation - np.eye(3, dtype=np.float64), ord="fro"))
        if ortho_err > EXTRINSIC_ROTATION_ORTHO_FROB_TOL:
            return False
        return True

    def _intrinsic_is_sane(intrinsic: np.ndarray) -> bool:
        array = np.asarray(intrinsic, dtype=np.float32).reshape(-1)
        if array.shape != (4,) or not np.isfinite(array).all():
            return False
        return float(array[0]) > 0.0 and float(array[1]) > 0.0

    def validate_lowdim_numeric_sanity(lowdim: np.ndarray) -> dict:
        array = np.asarray(lowdim, dtype=np.float32).reshape(-1)
        invalid_rot6d = any(
            not _rot6d_is_sane(array[rot_slice])
            for rot_slice in (
                slice(6, 12),
                slice(12, 18),
                slice(54, 60),
                slice(60, 66),
            )
        )
        invalid_extrinsic = not _extrinsic_is_sane(array[EXTRINSIC_SLICE].reshape(4, 4))
        invalid_intrinsic = not _intrinsic_is_sane(array[INTRINSIC_SLICE])
        issues = []
        if invalid_rot6d:
            issues.append("invalid_rot6d")
        if invalid_extrinsic:
            issues.append("invalid_extrinsic")
        if invalid_intrinsic:
            issues.append("invalid_intrinsic")
        return {
            "valid": not issues,
            "invalid_rot6d": bool(invalid_rot6d),
            "invalid_extrinsic": bool(invalid_extrinsic),
            "invalid_intrinsic": bool(invalid_intrinsic),
            "issues": issues,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose one clip across motion/world/final-WDS numeric stages")
    parser.add_argument(
        "--seq-folder",
        "--seq_folder",
        dest="seq_folders",
        action="append",
        required=True,
        help="Processed clip output folder, e.g. factory*/outputs/<clip_id>. Repeat for multiple clips.",
    )
    parser.add_argument("--mano-dir", default=None, help="Optional MANO model directory override")
    parser.add_argument("--device", default="cpu", help="Torch device for MANO forward, e.g. cpu or cuda:0")
    parser.add_argument("--chunk-limit", type=int, default=8, help="How many motion chunks to print per hand")
    parser.add_argument("--source-fps", type=float, default=5.0, help="Source label fps used during build")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Target label fps used during build")
    parser.add_argument("--interpolate-labels", action=argparse.BooleanOptionalAction, default=True, help="Use the current build-time label interpolation path")
    parser.add_argument("--wds-shard", default=None, help="Optional final WDS shard path for comparing exported lowdim/image samples")
    parser.add_argument("--wds-clip-id", default=None, help="Optional clip_id override for WDS lookup; default uses seq-folder name")
    parser.add_argument("--report-out", "--report_out", default=None, help="Optional JSON report output path")
    return parser


def _hand_name(hand_idx: int) -> str:
    return "left" if int(hand_idx) == 0 else "right"


def _format_range(array: np.ndarray) -> str:
    arr = np.asarray(array, dtype=np.float64)
    if arr.size == 0:
        return "empty"
    return f"[{arr.min():.6g}, {arr.max():.6g}]"


def _format_norm_range(array: np.ndarray, axis: int = -1) -> str:
    arr = np.asarray(array, dtype=np.float64)
    if arr.size == 0:
        return "empty"
    norms = np.linalg.norm(arr, axis=axis)
    return _format_range(norms)


def _parse_frame_index_from_key(sample_key: str) -> int:
    match = FRAME_INDEX_RE.search(sample_key)
    if match is None:
        raise ValueError(f"Failed to parse frame index from sample key: {sample_key}")
    return int(match.group(1))


def _load_cam_space_chunks(seq_folder: Path) -> dict[int, list[dict]]:
    start_idx, end_idx = get_track_range(seq_folder, fast=True)
    tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"
    frame_chunks_all = joblib.load(tracks_dir / "frame_chunks_all.npy")
    results: dict[int, list[dict]] = {0: [], 1: []}
    for hand_idx in (0, 1):
        for frame_chunk in frame_chunks_all.get(hand_idx, []):
            frame_chunk = np.asarray(frame_chunk, dtype=np.int64)
            if frame_chunk.size == 0:
                continue
            key = f"{int(frame_chunk[0])}_{int(frame_chunk[-1])}"
            path = seq_folder / "cam_space" / str(hand_idx) / f"{key}.json"
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            results[hand_idx].append(
                {
                    "key": key,
                    "frame_chunk": frame_chunk,
                    "init_trans": np.asarray(payload["init_trans"], dtype=np.float32),
                    "init_root_orient": np.asarray(payload["init_root_orient"], dtype=np.float32),
                    "init_hand_pose": np.asarray(payload["init_hand_pose"], dtype=np.float32),
                    "init_betas": np.asarray(payload["init_betas"], dtype=np.float32),
                }
            )
    return results


def _summarize_cam_space(cam_chunks: dict[int, list[dict]], chunk_limit: int) -> dict:
    print("\n=== Motion cam_space summary ===")
    summary = {}
    for hand_idx in (0, 1):
        hand_name = _hand_name(hand_idx)
        chunks = cam_chunks[hand_idx]
        print(f"[{hand_name}] chunks={len(chunks)}")
        hand_summary = {
            "chunks": int(len(chunks)),
        }
        if not chunks:
            summary[hand_name] = hand_summary
            continue

        all_trans = np.concatenate([chunk["init_trans"].reshape(-1, 3) for chunk in chunks], axis=0)
        all_root = np.concatenate([chunk["init_root_orient"].reshape(-1, 3, 3) for chunk in chunks], axis=0)
        all_hand_pose = np.concatenate([chunk["init_hand_pose"].reshape(-1, 15, 3, 3) for chunk in chunks], axis=0)
        root_dets = np.linalg.det(all_root.astype(np.float64))
        root_ortho_err = np.linalg.norm(
            np.matmul(np.transpose(all_root, (0, 2, 1)), all_root) - np.eye(3, dtype=np.float32),
            axis=(1, 2),
        )
        pose_dets = np.linalg.det(all_hand_pose.astype(np.float64).reshape(-1, 3, 3))
        pose_ortho_err = np.linalg.norm(
            np.matmul(
                np.transpose(all_hand_pose.reshape(-1, 3, 3), (0, 2, 1)),
                all_hand_pose.reshape(-1, 3, 3),
            ) - np.eye(3, dtype=np.float32),
            axis=(1, 2),
        )

        print(f"  init_trans xyz range={_format_range(all_trans)}")
        print(f"  init_trans norm range={_format_norm_range(all_trans)}")
        print(f"  root_rot det range={_format_range(root_dets)}")
        print(f"  root_rot ortho_err range={_format_range(root_ortho_err)}")
        print(f"  hand_pose det range={_format_range(pose_dets)}")
        print(f"  hand_pose ortho_err range={_format_range(pose_ortho_err)}")

        chunk_examples = []
        for example in chunks[: max(0, int(chunk_limit))]:
            example_trans = example["init_trans"].reshape(-1, 3)
            trans_norm_range = _format_norm_range(example_trans)
            print(f"  chunk={example['key']} frames={len(example['frame_chunk'])} trans_norm={trans_norm_range}")
            chunk_examples.append(
                {
                    "key": example["key"],
                    "frames": int(len(example["frame_chunk"])),
                    "trans_norm_range": trans_norm_range,
                }
            )

        hand_summary.update(
            {
                "init_trans_range": _format_range(all_trans),
                "init_trans_norm_range": _format_norm_range(all_trans),
                "root_rot_det_range": _format_range(root_dets),
                "root_rot_ortho_err_range": _format_range(root_ortho_err),
                "hand_pose_det_range": _format_range(pose_dets),
                "hand_pose_ortho_err_range": _format_range(pose_ortho_err),
                "chunk_examples": chunk_examples,
            }
        )
        summary[hand_name] = hand_summary
    return summary


def _load_world_prediction(seq_folder: Path) -> dict | None:
    if not (seq_folder / "world_space_res.pth").is_file():
        return None
    return _load_world_space_prediction({"episode_id": seq_folder.name}, str(seq_folder / "world_space_res.pth"))


def _summarize_world_prediction(prediction: dict | None) -> dict | None:
    if prediction is None:
        print("\n=== World summary ===")
        print("world_space_res.pth missing; skip world/build checks")
        return None

    print("\n=== World summary ===")
    pred_trans = prediction["pred_trans"].cpu().numpy()
    pred_rot = prediction["pred_rot"].cpu().numpy()
    pred_hand_pose = prediction["pred_hand_pose"].cpu().numpy()
    pred_valid = np.asarray(prediction["pred_valid"])
    summary = {"num_frames": int(pred_trans.shape[1]), "hands": {}}
    print(f"num_frames={pred_trans.shape[1]}")
    for hand_idx in (0, 1):
        hand_name = _hand_name(hand_idx)
        valid = pred_valid[hand_idx] > 0.5
        valid_count = int(valid.sum())
        print(f"[{hand_name}] valid={valid_count}/{pred_valid.shape[1]}")
        hand_summary = {"valid": valid_count}
        if valid_count > 0:
            hand_trans = pred_trans[hand_idx, valid]
            hand_rot = pred_rot[hand_idx, valid]
            hand_pose = pred_hand_pose[hand_idx, valid]
            print(f"  trans xyz range={_format_range(hand_trans)}")
            print(f"  trans norm range={_format_norm_range(hand_trans)}")
            print(f"  rot aa range={_format_range(hand_rot)}")
            print(f"  hand_pose aa range={_format_range(hand_pose)}")
            hand_summary.update(
                {
                    "trans_range": _format_range(hand_trans),
                    "trans_norm_range": _format_norm_range(hand_trans),
                    "rot_range": _format_range(hand_rot),
                    "hand_pose_range": _format_range(hand_pose),
                }
            )
        summary["hands"][hand_name] = hand_summary
    return summary


def _action_consistency_stats(lowdim_all: np.ndarray) -> dict:
    if lowdim_all.shape[0] <= 1:
        return {"wrist_max_abs_diff": 0.0, "hand_max_abs_diff": 0.0}
    wrist_diff = np.abs(lowdim_all[:-1, WRIST_ACTION_SLICE] - lowdim_all[1:, WRIST_STATE_SLICE])
    hand_diff = np.abs(lowdim_all[:-1, HAND_ACTION_SLICE] - lowdim_all[1:, HAND_STATE_SLICE])
    return {
        "wrist_max_abs_diff": float(wrist_diff.max()),
        "hand_max_abs_diff": float(hand_diff.max()),
    }


def _summarize_lowdim(label: str, lowdim_all: np.ndarray) -> dict:
    print(f"\n=== {label} lowdim summary ===")
    invalid_frames = []
    for frame_idx, lowdim in enumerate(lowdim_all):
        sanity = validate_lowdim_numeric_sanity(lowdim)
        if not sanity["valid"]:
            invalid_frames.append({"frame": int(frame_idx), "issues": list(sanity["issues"])})
    action_stats = _action_consistency_stats(lowdim_all)
    print(f"frames={lowdim_all.shape[0]}")
    print(f"invalid_lowdim_frames={len(invalid_frames)}")
    print(f"wrist_action_next_state_max_abs_diff={action_stats['wrist_max_abs_diff']:.6g}")
    print(f"hand_action_next_state_max_abs_diff={action_stats['hand_max_abs_diff']:.6g}")
    if invalid_frames:
        for item in invalid_frames[:10]:
            print(f"  frame={item['frame']} issues={','.join(item['issues'])}")
    return {
        "frames": int(lowdim_all.shape[0]),
        "invalid_lowdim_frames": invalid_frames,
        "action_consistency": action_stats,
    }


def _build_current_export_lowdim(
    seq_folder: Path,
    prediction: dict,
    *,
    mano_dir: str | None,
    device: str,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
) -> dict:
    import torch

    pred_trans = prediction["pred_trans"].float()
    pred_rot = prediction["pred_rot"].float()
    pred_hand_pose = prediction["pred_hand_pose"].float()
    pred_betas = prediction["pred_betas"].float()
    pred_valid = np.asarray(prediction["pred_valid"])
    source_frame_count = int(pred_trans.shape[1])
    torch_device = torch.device(device)
    mano_right, mano_left = build_mano_models(torch_device, mano_dir=mano_dir)
    wrist_state, hand_state = _compute_joint_states(pred_trans, pred_rot, pred_hand_pose, pred_betas, mano_right, mano_left, torch_device)
    ep = {"crop_dir": str(seq_folder), "episode_id": seq_folder.name}
    extrinsics, intrinsic = _load_episode_camera_features(ep, source_frame_count)
    presence_per_frame = _compute_presence_per_frame(pred_valid, source_frame_count)
    target_count = source_frame_count
    if interpolate_labels and source_fps > 0 and target_fps > 0 and source_frame_count > 1:
        duration = float(source_frame_count - 1) / float(source_fps)
        target_count = int(round(duration * float(target_fps))) + 1

    wrist_state, hand_state, pred_rot_resampled, pred_hand_pose_resampled, pred_betas_resampled, extrinsics_resampled, presence_resampled = resample_episode_features(
        wrist_state[:source_frame_count],
        hand_state[:source_frame_count],
        pred_rot[:, :source_frame_count],
        pred_hand_pose[:, :source_frame_count],
        pred_betas[:, :source_frame_count],
        extrinsics[:source_frame_count],
        presence_per_frame[:source_frame_count],
        target_count,
        source_fps=source_fps,
        target_fps=target_fps,
        interpolate_labels=interpolate_labels,
    )
    lowdim_all = _build_lowdim_features(wrist_state, hand_state, extrinsics_resampled[:target_count], intrinsic)
    export_frame_count = export_frame_count_with_action(target_count)
    return {
        "lowdim_all": lowdim_all[:export_frame_count].astype(np.float32),
        "presence_per_frame": np.asarray(presence_resampled)[:export_frame_count],
        "pred_rot": pred_rot_resampled[:, :export_frame_count],
        "pred_hand_pose": pred_hand_pose_resampled[:, :export_frame_count],
        "pred_betas": pred_betas_resampled[:, :export_frame_count],
        "extrinsics": extrinsics_resampled[:export_frame_count].astype(np.float32),
        "intrinsic": np.asarray(intrinsic, dtype=np.float32),
    }


def _load_wds_clip_samples(shard_path: Path, clip_id: str) -> dict | None:
    if not shard_path.is_file():
        raise FileNotFoundError(f"WDS shard not found: {shard_path}")

    samples: dict[str, dict] = {}
    with tarfile.open(shard_path, "r") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue
            for suffix, field_name in WDS_MEMBER_SUFFIXES.items():
                if not member.name.endswith(suffix):
                    continue
                sample_key = member.name[: -len(suffix)]
                if not sample_key.startswith(f"{clip_id}_f"):
                    break
                member_file = tar_reader.extractfile(member)
                if member_file is None:
                    break
                payload = member_file.read()
                sample = samples.setdefault(sample_key, {"key": sample_key})
                sample[field_name] = payload
                break

    if not samples:
        return None

    ordered = sorted(samples.values(), key=lambda item: _parse_frame_index_from_key(item["key"]))
    lowdim_all = np.stack([decode_lowdim(item["lowdim_bytes"]) for item in ordered], axis=0).astype(np.float32)
    image_bytes = [item.get("image_bytes") for item in ordered]
    metas = []
    for item in ordered:
        meta_bytes = item.get("meta_bytes")
        if meta_bytes is None:
            metas.append(None)
        else:
            metas.append(json.loads(meta_bytes.decode("utf-8")))
    return {
        "clip_id": clip_id,
        "sample_keys": [item["key"] for item in ordered],
        "lowdim_all": lowdim_all,
        "image_bytes": image_bytes,
        "metas": metas,
    }


def _compare_lowdim_arrays(label_a: str, a: np.ndarray, label_b: str, b: np.ndarray) -> dict:
    compare_count = min(int(a.shape[0]), int(b.shape[0]))
    if compare_count <= 0:
        return {"compare_frames": 0}
    diff = np.abs(np.asarray(a[:compare_count], dtype=np.float32) - np.asarray(b[:compare_count], dtype=np.float32))
    max_per_dim = diff.max(axis=0)
    top_indices = np.argsort(max_per_dim)[::-1][:10]
    summary = {
        "compare_frames": compare_count,
        "frame_count_a": int(a.shape[0]),
        "frame_count_b": int(b.shape[0]),
        "overall_max_abs_diff": float(diff.max()),
        "state_max_abs_diff": float(diff[:, :48].max()),
        "action_max_abs_diff": float(diff[:, 48:96].max()),
        "extrinsic_max_abs_diff": float(diff[:, 96:112].max()),
        "intrinsic_max_abs_diff": float(diff[:, 112:116].max()),
        "top_dims": [
            {
                "index": int(idx),
                "name": LOWDIM_DIMENSION_NAMES[int(idx)],
                "max_abs_diff": float(max_per_dim[int(idx)]),
            }
            for idx in top_indices
        ],
    }
    print(f"\n=== Compare {label_a} vs {label_b} ===")
    print(f"compare_frames={summary['compare_frames']} frame_count_a={summary['frame_count_a']} frame_count_b={summary['frame_count_b']}")
    print(f"overall_max_abs_diff={summary['overall_max_abs_diff']:.6g}")
    print(f"state_max_abs_diff={summary['state_max_abs_diff']:.6g}")
    print(f"action_max_abs_diff={summary['action_max_abs_diff']:.6g}")
    print(f"extrinsic_max_abs_diff={summary['extrinsic_max_abs_diff']:.6g}")
    print(f"intrinsic_max_abs_diff={summary['intrinsic_max_abs_diff']:.6g}")
    for item in summary["top_dims"]:
        print(f"  dim[{item['index']}] {item['name']} max_abs_diff={item['max_abs_diff']:.6g}")
    return summary


def _analyze_seq_folder(args, seq_folder: Path) -> dict:
    if not seq_folder.is_dir():
        raise FileNotFoundError(f"seq_folder not found: {seq_folder}")

    print(f"seq_folder: {seq_folder}")
    report = {"seq_folder": str(seq_folder)}

    cam_chunks = _load_cam_space_chunks(seq_folder)
    report["motion_cam_space"] = _summarize_cam_space(cam_chunks, args.chunk_limit)

    prediction = _load_world_prediction(seq_folder)
    world_summary = _summarize_world_prediction(prediction)
    if world_summary is not None:
        report["world_summary"] = world_summary

    current_export = None
    if prediction is not None:
        try:
            current_export = _build_current_export_lowdim(
                seq_folder,
                prediction,
                mano_dir=args.mano_dir,
                device=args.device,
                source_fps=args.source_fps,
                target_fps=args.target_fps,
                interpolate_labels=bool(args.interpolate_labels),
            )
            report["current_export_lowdim"] = _summarize_lowdim("Current export", current_export["lowdim_all"])
        except Exception as error:
            print(f"Warning: failed to build current-export lowdim: {error}")
            report["current_export_lowdim_error"] = str(error)

    wds_clip_id = args.wds_clip_id or seq_folder.name
    if args.wds_shard:
        try:
            wds_payload = _load_wds_clip_samples(Path(args.wds_shard).expanduser().resolve(), wds_clip_id)
            if wds_payload is None:
                print(f"\n=== Final WDS summary ===\nclip_id={wds_clip_id} not found in shard")
                report["wds_error"] = f"clip_id {wds_clip_id} not found"
            else:
                report["wds_lowdim"] = _summarize_lowdim("Final WDS", wds_payload["lowdim_all"])
                if current_export is not None:
                    report["current_vs_wds"] = _compare_lowdim_arrays(
                        "current_export",
                        current_export["lowdim_all"],
                        "final_wds",
                        wds_payload["lowdim_all"],
                    )
        except Exception as error:
            print(f"Warning: failed WDS comparison: {error}")
            report["wds_error"] = str(error)

    return report


def _multi_report_path(base_path: Path, clip_id: str) -> Path:
    suffix = base_path.suffix or ".json"
    stem = base_path.stem if base_path.suffix else base_path.name
    return base_path.with_name(f"{stem}.{clip_id}{suffix}")


def main() -> None:
    args = build_parser().parse_args()
    seq_folders = [Path(item).expanduser().resolve() for item in args.seq_folders]
    multi_clip = len(seq_folders) > 1
    reports = []

    for index, seq_folder in enumerate(seq_folders):
        if multi_clip:
            if index > 0:
                print()
            print("=" * 120)

        report = _analyze_seq_folder(args, seq_folder)
        reports.append(report)

    if args.report_out:
        report_out = Path(args.report_out).expanduser().resolve()
        report_out.parent.mkdir(parents=True, exist_ok=True)
        if multi_clip:
            aggregate = {"clips": reports}
            report_out.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nreport_out: {report_out}")
            for item in reports:
                per_clip_path = _multi_report_path(report_out, Path(item["seq_folder"]).name)
                per_clip_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"per_clip_report: {per_clip_path}")
        else:
            report_out.write_text(json.dumps(reports[0], ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nreport_out: {report_out}")


if __name__ == "__main__":
    main()
