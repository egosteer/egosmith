"""Helpers for normalizing dataset-pipeline configuration formats."""

from __future__ import annotations

import os
import re
import subprocess
from copy import deepcopy
from pathlib import Path

from lib.pipeline.io.workspace import default_output_root


_OFFICIAL_TOP_LEVEL_KEYS = {
    "run_tag",
    "resume",
    "dataset",
    "paths",
    "runtimes",
    "adapter_config",
    "infer",
    "annotation",
    "clip",
    "build",
    "filter",
    "validation",
}

_SINGLE_VIDEO_TOP_LEVEL_KEYS = {
    "video",
    "output_root",
    "run_tag",
    "resume",
    "annotation",
    "clip",
    "build",
    "filter",
    "validation",
    "infer",
    "adapter_config",
}

_LEGACY_TOP_LEVEL_KEYS = {
    "adapter",
    "source_id",
    "split",
    "factory_range",
    "start_factory_id",
    "end_factory_id",
    "buildai_repo_root",
    "buildai_config",
    "shard_root",
    "processed_root",
    "seq_folder_root",
    "annotation_root",
    "final_dataset_root",
    "log_root",
    "buildai_shell",
    "hawor_python",
    "slam_python",
    "batch_infer",
    "annotation_command",
    "require_annotation",
    "gpus",
    "workers_per_gpu",
    "resume",
    "checkpoint",
    "infiller_weight",
    "img_focal",
    "detect_motion",
    "slam",
    "native_depth",
    "infiller",
    "buildai",
}


def _as_dict(value):
    return dict(value) if isinstance(value, dict) else {}


def _ensure_mapping(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Pipeline config `{key}` must be a mapping.")
    return dict(value)


def _resolve_path(value, *, base_dir: Path | None) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _slug_from_stem(stem: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stem)).strip("._-")
    return slug or "video"


def _default_visible_gpus() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        usable = [token for token in tokens if token not in {"-1", "none", "None", "void", "NoDevFiles"}]
        if usable:
            return ",".join(usable)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return "0"
    gpu_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ",".join(gpu_ids) if gpu_ids else "0"


def _reject_legacy_layout(raw: dict) -> None:
    invalid = sorted(key for key in raw if key not in _OFFICIAL_TOP_LEVEL_KEYS)
    if not invalid:
        return
    legacy_hits = [key for key in invalid if key in _LEGACY_TOP_LEVEL_KEYS]
    if legacy_hits:
        raise ValueError(
            "Compact/legacy dataset-pipeline configs are no longer supported. "
            "Move top-level dataset/path/runtime/infer keys into the official nested sections. "
            f"Legacy keys found: {legacy_hits}"
        )
    raise ValueError(f"Unsupported top-level pipeline config keys: {invalid}")


def _apply_shared_nested_defaults(
    *,
    raw: dict,
    dataset_cfg: dict,
    paths_cfg: dict,
    runtimes_cfg: dict,
    build_cfg: dict,
    filter_cfg: dict,
    validation_cfg: dict,
    annotation_cfg: dict,
    clip_cfg: dict,
    adapter_cfg: dict,
    infer_cfg: dict,
    schema: str,
    migration_warnings: list[str] | None = None,
) -> dict:
    adapter_name = dataset_cfg.get("adapter") or "buildai"
    dataset_cfg.setdefault("adapter", adapter_name)
    dataset_cfg.setdefault("source_id", adapter_name)
    dataset_cfg.setdefault("split", "train")

    if adapter_name == "buildai":
        adapter_cfg.setdefault("stages", "1,2,3")
        adapter_cfg.setdefault("setup_decord", False)
        adapter_cfg.setdefault("clean_stage3_output", False)

    for key, default in (
        ("require_annotation", False),
        ("preprocess_workers", 8),
        ("writer_workers", 4),
        ("frames_per_shard", 10000),
        ("repeat_episodes", 1),
        ("mano_device", "cuda:0"),
        ("annotation_suffix", ".annotation.json"),
        ("source_fps", 5.0),
        ("target_fps", 30.0),
        ("interpolate_labels", True),
    ):
        build_cfg.setdefault(key, default)

    for key, default in (
        ("stages", "detect_track,motion,slam,infiller"),
        ("workers", 8),
        ("chunksize", 16),
        ("outlier_checks", True),
        ("camera_space_auto_method", "iqr_bounds"),
        ("camera_space_iqr_multiplier", 2.5),
        ("camera_space_axis_abs_cap", 1.5),
        ("camera_space_abs_percentile", 99.0),
        ("camera_space_abs_scale", 2.5),
    ):
        filter_cfg.setdefault(key, default)

    for key, default in (
        ("max_clips", 200),
        ("dataset_sample_checks", 20),
    ):
        validation_cfg.setdefault(key, default)

    clip_cfg.setdefault("mode", "none")

    normalized_infer_cfg = {
        "common": _as_dict(infer_cfg.get("common")),
        "detect_motion": _as_dict(infer_cfg.get("detect_motion")),
        "slam": _as_dict(infer_cfg.get("slam")),
        "native_depth": _as_dict(infer_cfg.get("native_depth")),
        "infiller": _as_dict(infer_cfg.get("infiller")),
        "multihost": _as_dict(infer_cfg.get("multihost")),
    }

    return {
        "run_tag": raw.get("run_tag"),
        "resume": bool(raw.get("resume", True)),
        "dataset": dataset_cfg,
        "paths": paths_cfg,
        "runtimes": runtimes_cfg,
        "adapter_config": adapter_cfg,
        "infer": normalized_infer_cfg,
        "annotation": annotation_cfg,
        "clip": clip_cfg,
        "build": build_cfg,
        "filter": filter_cfg,
        "validation": validation_cfg,
        "_meta": {
            "schema": schema,
            "migration_warnings": list(migration_warnings or []),
        },
    }


def _normalize_nested_pipeline_config(raw: dict) -> dict:
    _reject_legacy_layout(raw)

    dataset_cfg = _ensure_mapping(raw, "dataset")
    paths_cfg = _ensure_mapping(raw, "paths")
    runtimes_cfg = _ensure_mapping(raw, "runtimes")
    build_cfg = _ensure_mapping(raw, "build")
    filter_cfg = _ensure_mapping(raw, "filter")
    validation_cfg = _ensure_mapping(raw, "validation")
    annotation_cfg = _ensure_mapping(raw, "annotation")
    clip_cfg = _ensure_mapping(raw, "clip")
    adapter_cfg = _ensure_mapping(raw, "adapter_config")
    infer_cfg = _ensure_mapping(raw, "infer")

    return _apply_shared_nested_defaults(
        raw=raw,
        dataset_cfg=dataset_cfg,
        paths_cfg=paths_cfg,
        runtimes_cfg=runtimes_cfg,
        build_cfg=build_cfg,
        filter_cfg=filter_cfg,
        validation_cfg=validation_cfg,
        annotation_cfg=annotation_cfg,
        clip_cfg=clip_cfg,
        adapter_cfg=adapter_cfg,
        infer_cfg=infer_cfg,
        schema="nested",
        migration_warnings=[
            "Nested dataset-pipeline configs remain supported, but the preferred first-party schema is now `video: ...` plus optional `output_root:`."
        ],
    )


def _normalize_single_video_pipeline_config(raw: dict, *, base_dir: Path | None) -> dict:
    invalid = sorted(key for key in raw if key not in _SINGLE_VIDEO_TOP_LEVEL_KEYS)
    if invalid:
        raise ValueError(f"Unsupported single-video pipeline config keys: {invalid}")
    if not raw.get("video"):
        raise ValueError("Single-video pipeline config requires `video`.")

    video_path = _resolve_path(raw["video"], base_dir=base_dir)
    video_stem = _slug_from_stem(video_path.stem)
    output_root = (
        _resolve_path(raw["output_root"], base_dir=base_dir)
        if raw.get("output_root")
        else default_output_root(video_path)
    )

    annotation_cfg = _ensure_mapping(raw, "annotation")
    clip_cfg = _ensure_mapping(raw, "clip")
    adapter_cfg = _ensure_mapping(raw, "adapter_config")
    build_cfg = _ensure_mapping(raw, "build")
    filter_cfg = _ensure_mapping(raw, "filter")
    validation_cfg = _ensure_mapping(raw, "validation")
    infer_cfg = _ensure_mapping(raw, "infer")

    frames_root = output_root / "frames"
    stage_outputs_root = output_root / "stage_outputs"
    annotation_root = output_root / "annotations"
    resolved_annotation_root = annotation_cfg.get("root") or (str(annotation_root) if annotation_cfg.get("command") else None)
    paths_cfg = {
        "output_root": str(output_root),
        "frames_root": str(frames_root),
        "seq_folder_root": str(stage_outputs_root),
        "log_root": str(output_root / "runs"),
        "final_dataset_root": str(output_root / "webdataset"),
    }
    if resolved_annotation_root:
        paths_cfg["annotation_root"] = str(_resolve_path(resolved_annotation_root, base_dir=base_dir))
    dataset_cfg = {
        "adapter": "single_video",
        "source_id": video_stem,
        "split": "train",
    }
    adapter_cfg.setdefault("video", str(video_path))
    adapter_cfg.setdefault("clip_id", video_stem)
    adapter_cfg.setdefault("clip_name", video_stem)
    adapter_cfg.setdefault("frames_root", str(frames_root))
    adapter_cfg.setdefault("seq_folder_root", str(stage_outputs_root))
    adapter_cfg.setdefault("frame_ext", ".jpg")
    adapter_cfg.setdefault("jpeg_quality", 95)

    common_infer = _as_dict(infer_cfg.get("common"))
    common_infer.setdefault("gpus", _default_visible_gpus())
    common_infer.setdefault("resume", True)
    common_infer.setdefault("depth_predict_all_frames", True)
    infer_cfg["common"] = common_infer
    slam_infer = _as_dict(infer_cfg.get("slam"))
    slam_infer.setdefault("depth_backend", "any4d")
    slam_infer.setdefault("slam_backend", "dpvo")
    infer_cfg["slam"] = slam_infer

    build_cfg.setdefault("require_annotation", False)
    build_cfg.setdefault("export_depth", True)
    build_cfg.setdefault("interpolate_labels", False)
    build_cfg.setdefault("source_fps", None)
    build_cfg.setdefault("target_fps", None)

    validation_cfg.setdefault("allow_empty_instruction", True)
    validation_cfg.setdefault("require_depth", True)
    validation_cfg.setdefault("depth_action_consistency", True)

    normalized = _apply_shared_nested_defaults(
        raw={
            "run_tag": raw.get("run_tag") or "run",
            "resume": raw.get("resume", True),
        },
        dataset_cfg=dataset_cfg,
        paths_cfg=paths_cfg,
        runtimes_cfg={},
        build_cfg=build_cfg,
        filter_cfg=filter_cfg,
        validation_cfg=validation_cfg,
        annotation_cfg=annotation_cfg,
        clip_cfg=clip_cfg,
        adapter_cfg=adapter_cfg,
        infer_cfg=infer_cfg,
        schema="single_video",
    )
    normalized["_meta"].update(
        {
            "video": str(video_path),
            "output_root": str(output_root),
            "default_build_fps_from_video": True,
            "default_stages": (
                "prepare,annotate,infer,filter,build,validate"
                if annotation_cfg.get("command")
                else "prepare,infer,filter,build,validate"
            ),
        }
    )
    normalized["run_tag"] = raw.get("run_tag") or "run"
    return normalized


def normalize_pipeline_config(raw_config: dict | None, *, config_path: str | Path | None = None) -> dict:
    raw = deepcopy(raw_config or {})
    if not isinstance(raw, dict):
        raise ValueError("Pipeline config must be a mapping.")

    base_dir = Path(config_path).resolve().parent if config_path is not None else None
    if "video" in raw:
        return _normalize_single_video_pipeline_config(raw, base_dir=base_dir)
    return _normalize_nested_pipeline_config(raw)
