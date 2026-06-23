"""Filter a clip manifest using build-equivalent quality metrics."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from multiprocessing import current_process, get_context
from pathlib import Path

from lib.pipeline.clips.clip_manifest import load_clip_manifest, write_clip_manifest

DEFAULT_STAGES = "detect_track,motion,slam,infiller"
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))
_WORKER_CONFIG = None
_WORKER_MANO_RIGHT = None
_WORKER_MANO_LEFT = None
_WORKER_DEVICE = None


def build_parser():
    parser = argparse.ArgumentParser(description="Filter a manifest using build-equivalent quality thresholds")
    parser.add_argument("--input_manifest", required=True, help="Input manifest JSONL")
    parser.add_argument("--output_manifest", required=True, help="Output manifest JSONL for kept clips")
    parser.add_argument("--report_out", default=None, help="Optional JSON report path")
    parser.add_argument("--stages", default=DEFAULT_STAGES, help="Comma-separated stage outputs that must validate")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=16,
        help="Multiprocessing chunksize for manifest evaluation; higher values reduce IPC overhead",
    )
    parser.add_argument(
        "--feature_cache_dir",
        default=None,
        help="Optional shared feature cache directory for lowdim/MANO episode features",
    )
    parser.add_argument("--min_instruction_num", type=int, default=None, help="Optional minimum instruction_num required to keep a clip")
    parser.add_argument(
        "--outlier_checks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable optional outlier checks; NaN/Inf and missing-language hard filters always stay enabled",
    )
    parser.add_argument("--min_presence_ratio", type=float, default=None, help="Optional minimum fraction of frames with presence > 0")
    # Per-frame "abrupt jump" hard caps -- set above plausible human motion at ~30 fps
    # so they only catch reconstruction glitches (teleports / SLAM jumps), not real motion.
    parser.add_argument("--max_hand_translation_step", type=float, default=0.30, help="Max allowed per-frame wrist translation step in meters (~9 m/s glitch cap)")
    parser.add_argument("--max_finger_translation_step", type=float, default=0.30, help="Max allowed per-frame fingertip translation step in meters (max over 5 fingertips; ~9 m/s glitch cap)")
    parser.add_argument("--max_camera_translation_step", type=float, default=0.20, help="Max allowed per-frame camera translation step in meters (~6 m/s glitch cap)")
    parser.add_argument("--max_camera_rotation_step", type=float, default=0.70, help="Max allowed per-frame camera rotation delta, Frobenius norm of R[t+1]-R[t] (~28 deg/frame glitch cap)")
    parser.add_argument("--max_wrist_rotation_step", type=float, default=0.99, help="Max allowed per-frame wrist (root) rotation delta, Frobenius norm of R[t+1]-R[t] (~41 deg/frame glitch cap; paper Stage-4 frame-level, larger margin than camera since hands rotate faster)")
    parser.add_argument("--episode_camera_iqr_multiplier", type=float, default=2.5, help="IQR multiplier for episode-level camera-motion (mean per-frame translation/rotation) dataset outlier bounds")
    parser.add_argument(
        "--fatal_offscreen_scale",
        type=float,
        default=1.4,
        help="Visible-hand fatal bound multiplier for image size; e.g. 1.4 means wrist/fingertips entirely beyond [-0.4W,1.4W]x[-0.4H,1.4H] are dropped",
    )
    parser.add_argument(
        "--min_visible_hand_any_point_inframe_ratio",
        type=float,
        default=0.2,
        help="Minimum ratio of visible-hand frames where wrist/fingertips have at least one projected point inside the image",
    )
    parser.add_argument(
        "--max_visible_hand_all_points_out_of_frame_streak",
        type=int,
        default=30,
        help="Maximum allowed consecutive visible-hand frames with all projected wrist/fingertips points outside the image",
    )
    parser.add_argument("--max_camera_space_wrist_abs", type=float, default=None, help="Optional max absolute camera-space coordinate allowed for wrist positions in meters")
    parser.add_argument("--max_camera_space_hand_abs", type=float, default=None, help="Optional max absolute camera-space coordinate allowed for stored hand keypoints in meters")
    parser.add_argument("--camera_space_auto_method", type=str, default="iqr_bounds", choices=("iqr_bounds", "percentile_abs"), help="Automatic camera-space filter mode when manual abs thresholds are not provided")
    parser.add_argument("--camera_space_iqr_multiplier", type=float, default=2.5, help="IQR multiplier used for automatic camera-space lower/upper bounds")
    parser.add_argument("--camera_space_axis_abs_cap", type=float, default=1.5, help="Hard absolute cap applied to camera-space x/y/z coordinates for wrist and hand points")
    parser.add_argument("--camera_space_abs_percentile", type=float, default=99.0, help="Percentile used for automatic camera-space absolute-value thresholds")
    parser.add_argument("--camera_space_abs_scale", type=float, default=2.5, help="Scale multiplier applied to the chosen percentile for automatic camera-space thresholds")
    parser.add_argument("--chunk_window_past_seconds", type=float, default=6.0, help="Chunk-level sliding window past context in seconds (paper Stage-4 chunk level: wrist rel camera over the window)")
    parser.add_argument("--chunk_window_future_frames", type=int, default=30, help="Chunk-level sliding window future context in frames (paper Stage-4 chunk level)")
    parser.add_argument("--annotation_root", default=None, help="Clip annotation sidecar directory")
    parser.add_argument("--annotation_suffix", type=str, default=".annotation.json", help="Annotation sidecar suffix, e.g. .annotation.json or _qwen-annotation.json")
    parser.add_argument("--require_annotation", action="store_true", help="Drop clips with missing or invalid annotations")
    parser.add_argument("--mano_device", type=str, default="cuda:0", help="Device for MANO forward pass")
    parser.add_argument("--mano_gpus", type=str, default=None, help="Optional comma-separated GPU list for MANO workers")
    parser.add_argument("--mano_dir", type=str, default=None, help="Optional MANO model directory")
    parser.add_argument("--source_fps", type=float, default=5.0, help="FPS of the stage outputs in seq_folder")
    parser.add_argument("--target_fps", type=float, default=30.0, help="FPS of the RGB frames referenced by the manifest")
    parser.add_argument(
        "--interpolate_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpolate source labels onto descriptor frames instead of truncating to the source sequence length",
    )
    parser.add_argument("--dry_run", action="store_true", help="Analyze only; do not write output manifest")
    return parser


def parse_stage_list(raw: str) -> list[str]:
    stages = [stage.strip() for stage in str(raw).split(",") if stage.strip()]
    if not stages:
        raise ValueError("Expected at least one stage in --stages")
    return stages


def _new_result(record) -> dict:
    return {
        "clip_id": record.clip_id,
        "seq_folder": str(Path(record.descriptor.seq_folder)),
        "keep": False,
        "build_ready": False,
        "drop_category": None,
        "reasons": [],
        "build_reasons": [],
        "quality_reasons": [],
        "metrics": {},
    }


def _append_build_reason(result: dict, reason: str, metric_key: str | None = None, metric_value=None) -> dict:
    result["build_reasons"].append(reason)
    result["reasons"] = list(result["build_reasons"])
    result["drop_category"] = "build_invalid"
    if metric_key is not None:
        result["metrics"][metric_key] = metric_value
    return result


def _validate_build_inputs(record, stages: list[str], result: dict) -> tuple[bool, tuple[int, int, dict | None] | None]:
    from lib.pipeline.exporters.manifest_vla import descriptor_uses_native_features, load_manifest_record_prediction
    from lib.pipeline.slam.native_depth import validate_native_depth_output
    from lib.pipeline.proc.stage_api import get_track_range, validate_stage_output

    if descriptor_uses_native_features(record.descriptor):
        unsupported_stages = [stage for stage in stages if stage not in ("native_features", "native_depth")]
        if unsupported_stages:
            _append_build_reason(result, "native_features_stage_mismatch", "unsupported_stages", unsupported_stages)
            return False, None
        if "native_depth" in stages:
            try:
                native_depth_summary = validate_native_depth_output(
                    record.descriptor.seq_folder,
                    expected_frame_count=int(record.descriptor.frame_count),
                )
                result["metrics"]["native_depth"] = native_depth_summary
            except Exception as error:
                _append_build_reason(result, "invalid_stage_output:native_depth", "native_depth_error", str(error))
                return False, None
        result["metrics"]["track_range"] = [0, int(record.descriptor.frame_count)]
        return True, (0, int(record.descriptor.frame_count), None)

    seq_folder = Path(record.descriptor.seq_folder)
    try:
        start_idx, end_idx = get_track_range(seq_folder, fast=False)
    except Exception as error:
        _append_build_reason(result, "missing_track_range", "track_range_error", str(error))
        return False, None

    result["metrics"]["track_range"] = [int(start_idx), int(end_idx)]
    prediction = None
    for stage in stages:
        if stage == "infiller":
            prediction, error_code = load_manifest_record_prediction(record)
            if prediction is None:
                _append_build_reason(result, f"invalid_stage_output:{stage}", f"{stage}_error", str(error_code))
                return False, None
            continue
        try:
            validate_stage_output(stage, seq_folder, start_idx, end_idx)
        except Exception as error:
            _append_build_reason(result, f"invalid_stage_output:{stage}", f"{stage}_error", str(error))
            return False, None
    return True, (int(start_idx), int(end_idx), prediction)


def evaluate_record(record, config: dict) -> dict:
    from lib.pipeline.exporters.manifest_vla import (
        compute_descriptor_episode_quality_metrics,
        descriptor_uses_native_features,
        load_manifest_record_prediction,
        prepare_manifest_record_for_build,
    )

    result = _new_result(record)
    ok, validate_payload = _validate_build_inputs(record, config["stages"], result)
    if not ok:
        return result
    prediction = validate_payload[2] if validate_payload is not None and len(validate_payload) >= 3 else None
    if prediction is None and not descriptor_uses_native_features(record.descriptor):
        prediction, error_code = load_manifest_record_prediction(record)
        if prediction is None:
            return _append_build_reason(result, str(error_code))

    episode, error_code = prepare_manifest_record_for_build(
        record,
        require_annotation=bool(config["require_annotation"]),
        annotation_root=config["annotation_root"],
        annotation_suffix=config["annotation_suffix"],
        source_fps=float(config["source_fps"]),
        target_fps=float(config["target_fps"]),
        interpolate_labels=bool(config["interpolate_labels"]),
        prediction=prediction,
    )
    if episode is None:
        return _append_build_reason(result, str(error_code))

    metrics = compute_descriptor_episode_quality_metrics(
        episode,
        _WORKER_MANO_RIGHT,
        _WORKER_MANO_LEFT,
        _WORKER_DEVICE,
        feature_cache_dir=config["feature_cache_dir"],
        mano_dir=config["mano_dir"],
        prediction=prediction,
        source_fps=float(config["source_fps"]),
        target_fps=float(config["target_fps"]),
        interpolate_labels=bool(config["interpolate_labels"]),
        fatal_offscreen_scale=float(config["fatal_offscreen_scale"]),
        chunk_window_past_seconds=float(config.get("chunk_window_past_seconds", 6.0)),
        chunk_window_future_frames=int(config.get("chunk_window_future_frames", 30)),
        enable_chunk_window=bool(config.get("outlier_checks", True)),
    )
    if metrics is None:
        return _append_build_reason(result, "invalid_episode_features")

    result["metrics"] = {
        **result["metrics"],
        **metrics,
        "instruction_num": int(episode.get("instruction_num", 0)),
        "num_valid_frames": int(episode.get("num_valid_frames", 0)),
    }
    result["build_ready"] = True
    return result


def worker_init(config: dict):
    import torch

    global _WORKER_CONFIG, _WORKER_MANO_RIGHT, _WORKER_MANO_LEFT, _WORKER_DEVICE
    _WORKER_CONFIG = config

    if bool(config.get("skip_mano_models")):
        _WORKER_DEVICE = torch.device("cpu")
        _WORKER_MANO_RIGHT = None
        _WORKER_MANO_LEFT = None
        return

    from lib.pipeline.exporters.webdataset_features import build_mano_models

    identity = current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    device_str = config["mano_device_specs"][worker_idx % len(config["mano_device_specs"])]
    _WORKER_DEVICE = torch.device(device_str)
    _WORKER_MANO_RIGHT, _WORKER_MANO_LEFT = build_mano_models(_WORKER_DEVICE, mano_dir=config["mano_dir"])
    _WORKER_MANO_RIGHT.eval()
    _WORKER_MANO_LEFT.eval()


def worker_eval(task):
    index, record = task
    result = evaluate_record(record, _WORKER_CONFIG)
    result["index"] = index
    return result


def build_report(
    results: list[dict],
    input_manifest: Path,
    output_manifest: Path,
    criteria: dict,
    threshold_info: dict,
) -> dict:
    kept = 0
    build_invalid_reason_counts = Counter()
    quality_reason_counts = Counter()
    dropped = []
    build_ready = 0

    for item in results:
        if item["build_ready"]:
            build_ready += 1
        if item["keep"]:
            kept += 1
            continue
        dropped.append(
            {
                "clip_id": item["clip_id"],
                "seq_folder": item["seq_folder"],
                "drop_category": item["drop_category"],
                "reasons": item["reasons"],
                "metrics": item["metrics"],
            }
        )
        if item["drop_category"] == "quality":
            quality_reason_counts.update(item["quality_reasons"])
        else:
            build_invalid_reason_counts.update(item["build_reasons"])

    resolved_criteria = {
        "hard_rules": {
            "drop_nonfinite_lowdim": True,
            "require_instruction_every_frame": bool(criteria.get("require_annotation")),
        },
        "min_instruction_num": criteria["min_instruction_num"],
        "outlier_checks": bool(criteria["outlier_checks"]),
        "min_presence_ratio": criteria["min_presence_ratio"],
        "max_hand_translation_step": criteria["max_hand_translation_step"],
        "max_finger_translation_step": criteria["max_finger_translation_step"],
        "max_camera_translation_step": criteria["max_camera_translation_step"],
        "max_camera_rotation_step": criteria["max_camera_rotation_step"],
        "max_wrist_rotation_step": criteria["max_wrist_rotation_step"],
        "episode_camera_iqr_multiplier": criteria.get("episode_camera_iqr_multiplier"),
        "fatal_offscreen_scale": criteria["fatal_offscreen_scale"],
        "min_visible_hand_any_point_inframe_ratio": criteria["min_visible_hand_any_point_inframe_ratio"],
        "max_visible_hand_all_points_out_of_frame_streak": criteria["max_visible_hand_all_points_out_of_frame_streak"],
        "camera_space_auto_method": criteria["camera_space_auto_method"],
        "camera_space_iqr_multiplier": criteria["camera_space_iqr_multiplier"],
        "max_camera_space_wrist_abs": threshold_info["resolved"]["max_camera_space_wrist_abs"],
        "max_camera_space_hand_abs": threshold_info["resolved"]["max_camera_space_hand_abs"],
        "camera_space_wrist_bounds": threshold_info["resolved"]["camera_space_wrist_bounds"],
        "camera_space_hand_bounds": threshold_info["resolved"]["camera_space_hand_bounds"],
        "episode_camera_translation_bounds": threshold_info["resolved"]["episode_camera_translation_bounds"],
        "episode_camera_rotation_bounds": threshold_info["resolved"]["episode_camera_rotation_bounds"],
        "camera_space_axis_abs_cap": criteria["camera_space_axis_abs_cap"],
        "chunk_window_past_seconds": criteria.get("chunk_window_past_seconds"),
        "chunk_window_future_frames": criteria.get("chunk_window_future_frames"),
    }
    return {
        "input_manifest": str(input_manifest.resolve()),
        "output_manifest": str(output_manifest.resolve()),
        "criteria": {
            "stages": list(criteria["stages"]),
            "annotation_root": criteria["annotation_root"],
            "annotation_suffix": criteria["annotation_suffix"],
            "require_annotation": bool(criteria["require_annotation"]),
            "source_fps": float(criteria["source_fps"]),
            "target_fps": float(criteria["target_fps"]),
            "interpolate_labels": bool(criteria["interpolate_labels"]),
            "chunksize": int(criteria["chunksize"]),
            "feature_cache_dir": criteria["feature_cache_dir"],
            **resolved_criteria,
        },
        "auto_thresholds": threshold_info,
        "total_clips": len(results),
        "build_ready_clips": build_ready,
        "kept_clips": kept,
        "dropped_clips": len(results) - kept,
        "dropped_quality_clips": sum(1 for item in results if item["drop_category"] == "quality"),
        "build_invalid_clips": sum(1 for item in results if item["drop_category"] == "build_invalid"),
        "quality_reason_counts": dict(sorted(quality_reason_counts.items())),
        "build_invalid_reason_counts": dict(sorted(build_invalid_reason_counts.items())),
        "dropped": dropped,
    }


def run_filter(args) -> dict:
    import torch
    from tqdm import tqdm
    from lib.pipeline.exporters.manifest_vla import descriptor_uses_native_features
    from lib.pipeline.exporters.webdataset_workers import normalize_mano_devices
    from lib.pipeline.quality.quality_metrics import decide_clip_quality, resolve_auto_quality_thresholds

    records = load_clip_manifest(args.input_manifest)
    skip_mano_models = bool(records) and all(descriptor_uses_native_features(record.descriptor) for record in records)
    mano_device_obj = torch.device(args.mano_device if torch.cuda.is_available() else "cpu")
    mano_device_specs = normalize_mano_devices(
        str(mano_device_obj),
        args.mano_gpus if mano_device_obj.type == "cuda" else None,
    )
    worker_count = int(args.workers)
    if mano_device_obj.type == "cuda":
        worker_count = min(worker_count, len(mano_device_specs))

    config = {
        "stages": parse_stage_list(args.stages),
        "annotation_root": args.annotation_root,
        "annotation_suffix": args.annotation_suffix,
        "require_annotation": bool(args.require_annotation),
        "hard_rules": {
            "drop_nonfinite_lowdim": True,
            "require_instruction_every_frame": bool(args.require_annotation),
        },
        "source_fps": float(args.source_fps),
        "target_fps": float(args.target_fps),
        "interpolate_labels": bool(args.interpolate_labels),
        "min_instruction_num": args.min_instruction_num,
        "outlier_checks": bool(args.outlier_checks),
        "min_presence_ratio": args.min_presence_ratio,
        "max_hand_translation_step": args.max_hand_translation_step,
        "max_finger_translation_step": args.max_finger_translation_step,
        "max_camera_translation_step": args.max_camera_translation_step,
        "max_camera_rotation_step": args.max_camera_rotation_step,
        "max_wrist_rotation_step": args.max_wrist_rotation_step,
        "episode_camera_iqr_multiplier": args.episode_camera_iqr_multiplier,
        "fatal_offscreen_scale": float(args.fatal_offscreen_scale),
        "min_visible_hand_any_point_inframe_ratio": args.min_visible_hand_any_point_inframe_ratio,
        "max_visible_hand_all_points_out_of_frame_streak": args.max_visible_hand_all_points_out_of_frame_streak,
        "max_camera_space_wrist_abs": args.max_camera_space_wrist_abs,
        "max_camera_space_hand_abs": args.max_camera_space_hand_abs,
        "camera_space_auto_method": args.camera_space_auto_method,
        "camera_space_iqr_multiplier": args.camera_space_iqr_multiplier,
        "camera_space_axis_abs_cap": args.camera_space_axis_abs_cap,
        "camera_space_abs_percentile": args.camera_space_abs_percentile,
        "camera_space_abs_scale": args.camera_space_abs_scale,
        "chunk_window_past_seconds": float(args.chunk_window_past_seconds),
        "chunk_window_future_frames": int(args.chunk_window_future_frames),
        "chunksize": int(args.chunksize),
        "feature_cache_dir": args.feature_cache_dir,
        "mano_dir": args.mano_dir,
        "mano_device_specs": mano_device_specs,
        "skip_mano_models": skip_mano_models,
    }
    if not config["outlier_checks"]:
        for key in (
            "min_presence_ratio",
            "max_hand_translation_step",
            "max_finger_translation_step",
            "max_camera_translation_step",
            "max_camera_rotation_step",
            "max_wrist_rotation_step",
            "max_camera_space_wrist_abs",
            "max_camera_space_hand_abs",
            "camera_space_axis_abs_cap",
        ):
            config[key] = None
        config["use_auto_camera_space_thresholds"] = False
    else:
        config["use_auto_camera_space_thresholds"] = (
            config["max_camera_space_wrist_abs"] is None or config["max_camera_space_hand_abs"] is None
        )

    if worker_count <= 1:
        worker_init(config)
        results = [
            dict(evaluate_record(record, config), index=index)
            for index, record in tqdm(enumerate(records), total=len(records), desc="Filter manifest")
        ]
    else:
        mp_context = get_context("spawn") if mano_device_obj.type == "cuda" else get_context()
        with mp_context.Pool(worker_count, initializer=worker_init, initargs=(config,)) as pool:
            results = list(
                tqdm(
                    pool.imap(worker_eval, enumerate(records), chunksize=int(args.chunksize)),
                    total=len(records),
                    desc="Filter manifest",
                )
            )

    results.sort(key=lambda item: item["index"])
    clip_metrics = [item["metrics"] for item in results if item["build_ready"]]
    threshold_info = resolve_auto_quality_thresholds(clip_metrics, config)
    resolved_criteria = dict(config)
    resolved_criteria.update(threshold_info["resolved"])

    for item in results:
        if not item["build_ready"]:
            continue
        keep, reasons = decide_clip_quality(
            item["metrics"],
            resolved_criteria,
            include_incomplete_sample_reason=False,
            include_invalid_meta_reason=False,
        )
        item["keep"] = bool(keep)
        item["quality_reasons"] = list(reasons)
        item["reasons"] = list(reasons)
        item["drop_category"] = None if keep else "quality"

    kept_records = [record for record, result in zip(records, results) if result["keep"]]

    if not args.dry_run:
        write_clip_manifest(kept_records, args.output_manifest)

    report = build_report(results, Path(args.input_manifest), Path(args.output_manifest), config, threshold_info)
    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be >= 1")
    if args.dry_run and Path(args.output_manifest).exists():
        print(f"Dry run: not writing {args.output_manifest}")
    report = run_filter(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
