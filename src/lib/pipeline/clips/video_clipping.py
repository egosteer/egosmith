"""Optional raw-video temporal clipping integration for the dataset pipeline."""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from lib.annotation.api_annotation import load_api_keys
from lib.annotation.api_annotation_with_clip import run_api_video_clipping
from lib.clip.heuristic_video_clipper import discover_videos, load_clip_config, run_heuristic_clipping


def _resolve_path(value, *, base_dir: Path | None = None) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _factory_group_names(dataset_cfg: dict) -> list[str]:
    start_factory_id = dataset_cfg.get("start_factory_id")
    end_factory_id = dataset_cfg.get("end_factory_id")
    if start_factory_id is None or end_factory_id is None:
        return []
    return [f"factory{factory_id:03d}" for factory_id in range(int(start_factory_id), int(end_factory_id) + 1)]


def _discover_videos_in_factory_range(video_root: Path, dataset_cfg: dict) -> list[Path]:
    factory_names = _factory_group_names(dataset_cfg)
    if not factory_names:
        return discover_videos(video_root)

    videos: list[Path] = []
    for factory_name in factory_names:
        factory_dir = video_root / factory_name
        if factory_dir.is_dir():
            videos.extend(discover_videos(factory_dir))
    return sorted(videos)


def _source_videos_for_config(
    dataset_cfg: dict,
    adapter_cfg: dict,
    paths_cfg: dict,
    clip_cfg: dict,
) -> tuple[list[Path], Path | None]:
    adapter_name = dataset_cfg.get("adapter")
    if adapter_name == "single_video":
        video_path = _resolve_path(adapter_cfg["video"])
        return [video_path], video_path.parent
    if adapter_name == "video_folder":
        video_root = _resolve_path(adapter_cfg.get("video_root") or paths_cfg.get("video_root"))
        return discover_videos(video_root), video_root
    if adapter_name == "buildai":
        video_root_value = (
            clip_cfg.get("video_root")
            or adapter_cfg.get("video_root")
            or paths_cfg.get("video_root")
            or paths_cfg.get("data_root")
        )
        if not video_root_value:
            raise ValueError(
                "clip.mode requires raw BuildAI videos via "
                "`clip.video_root`, `adapter_config.video_root`, `paths.video_root`, or `paths.data_root`."
            )
        video_root = _resolve_path(video_root_value)
        return _discover_videos_in_factory_range(video_root, dataset_cfg), video_root
    raise ValueError(
        "clip.mode supports raw-video adapters single_video, video_folder, and buildai, "
        f"got adapter={adapter_name!r}"
    )


def _clip_config_overrides(clip_cfg: dict) -> dict | None:
    if clip_cfg.get("config_overrides"):
        return dict(clip_cfg["config_overrides"])
    if clip_cfg.get("heuristic_overrides"):
        return {"heuristic": dict(clip_cfg["heuristic_overrides"])}
    if clip_cfg.get("stage1_overrides"):
        return dict(clip_cfg["stage1_overrides"])
    return None


def _api_keys_from_clip_config(clip_cfg: dict, *, dry_run: bool) -> list[str]:
    if clip_cfg.get("api_keys"):
        return [str(key).strip() for key in clip_cfg["api_keys"] if str(key).strip()]
    args = SimpleNamespace(
        api_key=clip_cfg.get("api_key"),
        api_keys_file=clip_cfg.get("api_keys_file"),
        dry_run=dry_run,
    )
    return load_api_keys(args)


def _redirect_to_clipped_video_folder(
    *,
    config: dict,
    clip_cfg: dict,
    dataset_cfg: dict,
    adapter_cfg: dict,
    paths_cfg: dict,
    clip_root: Path,
    clip_video_root: Path,
    clip_frames_root: Path,
    clip_stage_outputs_root: Path,
) -> dict:
    original_dataset = dict(dataset_cfg)
    original_adapter = dict(adapter_cfg)
    dataset_cfg["adapter"] = "video_folder"
    dataset_cfg.setdefault("source_id", original_dataset.get("source_id") or "clipped_videos")
    adapter_cfg.clear()
    adapter_cfg.update(
        {
            "video_root": str(clip_video_root),
            "frames_root": str(clip_frames_root),
            "seq_folder_root": str(clip_stage_outputs_root),
            "frame_subdir": "extracted_images",
            "extract_frames": True,
            "resume": bool(config.get("resume", True)),
            "frame_ext": str(clip_cfg.get("frame_ext", ".jpg")),
            "jpeg_quality": int(clip_cfg.get("jpeg_quality", 95)),
            "_original_dataset": original_dataset,
            "_original_adapter_config": original_adapter,
        }
    )
    paths_cfg["clip_root"] = str(clip_root)
    paths_cfg["video_root"] = str(clip_video_root)
    paths_cfg["frames_root"] = str(clip_frames_root)
    paths_cfg["seq_folder_root"] = str(clip_stage_outputs_root)
    return original_dataset


def apply_video_clipping_if_configured(
    *,
    config: dict,
    run_dir: Path,
    project_root: Path,
) -> dict | None:
    """Optionally clip raw videos and redirect later stages to clipped videos."""
    clip_cfg = config.get("clip") or {}
    mode = str(clip_cfg.get("mode", "none")).strip().lower()
    if mode in {"", "none", "off", "false"}:
        return None
    if mode not in {"heuristic", "api", "api_annotation", "semantic_api"}:
        raise ValueError(
            f"Unsupported clip.mode={mode!r}; supported values are none, heuristic, api, api_annotation, semantic_api"
        )

    dataset_cfg = config.setdefault("dataset", {})
    adapter_cfg = config.setdefault("adapter_config", {})
    paths_cfg = config.setdefault("paths", {})
    source_videos, source_root = _source_videos_for_config(dataset_cfg, adapter_cfg, paths_cfg, clip_cfg)
    if not source_videos:
        raise RuntimeError(f"clip.mode={mode} found no input videos")

    clip_root = _resolve_path(clip_cfg.get("output_root") or paths_cfg.get("clip_root") or (run_dir / "clips"))
    clip_video_root = clip_root / "videos"
    clip_frames_root = clip_root / "frames"
    clip_stage_outputs_root = clip_root / "stage_outputs"
    report_out = _resolve_path(clip_cfg.get("report_out") or (run_dir / "clip_report.json"))

    if mode == "heuristic":
        clip_config_path = clip_cfg.get("config")
        if clip_config_path:
            clip_config_path = str(_resolve_path(clip_config_path, base_dir=project_root))
        clip_config = load_clip_config(clip_config_path, _clip_config_overrides(clip_cfg))
        heuristic_cfg = clip_config.setdefault("heuristic", clip_config.get("stage1", {}))
        for key in (
            "skip_frames",
            "decode_width",
            "decode_height",
            "fallback_full_video",
            "output_size",
            "model_path",
        ):
            if key in clip_cfg:
                heuristic_cfg[key] = clip_cfg[key]

        report = run_heuristic_clipping(
            video_paths=source_videos,
            output_root=clip_video_root,
            config=clip_config,
            source_root=source_root,
            report_out=report_out,
        )
    else:
        annotation_cfg = config.setdefault("annotation", {})
        annotation_root = _resolve_path(
            clip_cfg.get("annotation_root")
            or annotation_cfg.get("root")
            or paths_cfg.get("annotation_root")
            or (clip_root / "annotations")
        )
        paths_cfg["annotation_root"] = str(annotation_root)
        report = run_api_video_clipping(
            video_paths=source_videos,
            output_root=clip_video_root,
            annotation_root=annotation_root,
            prompt_file=(
                str(_resolve_path(clip_cfg["prompt_file"], base_dir=project_root))
                if clip_cfg.get("prompt_file")
                else None
            ),
            api_keys=_api_keys_from_clip_config(clip_cfg, dry_run=bool(clip_cfg.get("dry_run", False))),
            source_root=source_root,
            annotation_suffix=str(clip_cfg.get("annotation_suffix") or config.get("build", {}).get("annotation_suffix", ".annotation.json")),
            model=str(clip_cfg.get("model", "qwen3.5-plus")),
            target_fps=float(clip_cfg.get("target_fps", 5.0)),
            workers=int(clip_cfg.get("workers", 4)),
            min_segment_sec=float(clip_cfg.get("min_segment_sec", 0.2)),
            max_segment_sec=float(clip_cfg.get("max_segment_sec", 0.0)),
            keep_low_quality=bool(clip_cfg.get("keep_low_quality", False)),
            resume=bool(config.get("resume", True)),
            dry_run=bool(clip_cfg.get("dry_run", False)),
            report_out=report_out,
        )
        annotation_cfg["_api_clip_completed"] = True

    if report["summary"]["kept_clips"] <= 0:
        raise RuntimeError(
            f"clip.mode={mode} produced no clips. Check {report_out}."
        )

    original_dataset = _redirect_to_clipped_video_folder(
        config=config,
        clip_cfg=clip_cfg,
        dataset_cfg=dataset_cfg,
        adapter_cfg=adapter_cfg,
        paths_cfg=paths_cfg,
        clip_root=clip_root,
        clip_video_root=clip_video_root,
        clip_frames_root=clip_frames_root,
        clip_stage_outputs_root=clip_stage_outputs_root,
    )

    summary = {
        "mode": mode,
        "source_videos": len(source_videos),
        "kept_clips": report["summary"]["kept_clips"],
        "clip_video_root": str(clip_video_root),
        "clip_frames_root": str(clip_frames_root),
        "clip_stage_outputs_root": str(clip_stage_outputs_root),
        "report_out": str(report_out),
        "original_dataset": original_dataset,
    }
    (run_dir / "video_clipping_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


apply_heuristic_clipping_if_configured = apply_video_clipping_if_configured
