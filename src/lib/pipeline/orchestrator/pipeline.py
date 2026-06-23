"""Official dataset pipeline orchestration."""

from __future__ import annotations

import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.clips.clip_manifest import (
    build_manifest_records_from_descriptors,
    load_clip_manifest,
    write_clip_manifest,
    write_shard_dir_list,
)
from lib.pipeline.batch.cli import SHARED_PROFILE_CACHE_OPTION_DESTS
from lib.pipeline.batch.state import load_status_payload_with_fallback
from lib.pipeline.clips.video_clipping import apply_video_clipping_if_configured
from lib.pipeline.datasets import DatasetAdapterContext, get_dataset_adapter
from lib.pipeline.io.frame_sources import classify_descriptor_storage
from lib.pipeline.proc.multihost import (
    MultihostStageQueueRunner,
    MultihostStageSpec,
    parse_multihost_config,
    sanitize_infer_args_for_multihost,
)
from lib.pipeline.proc.pipeline_config import normalize_pipeline_config
from lib.pipeline.proc.runtime_resolver import resolve_pipeline_runtimes
from lib.pipeline.proc.stage_api import get_stage_done_marker

from .cli import get_parser
from .constants import BATCH_INFER_NEGATIVE_BOOL_FLAGS, MULTIHOST_DISALLOWED_INFER_KEYS, OFFICIAL_STAGE_ORDER
from .helpers import cli_args_from_mapping, format_annotation_command, load_yaml, stream_command
from .stage_selection import selected_stages
from .validation import (
    infer_stage_worker_count_per_gpu,
    validate_multihost_infer_alignment,
    validate_pipeline_cli_alignment,
)


def _hostname_slug() -> str:
    raw = socket.gethostname().strip().lower()
    cleaned = "".join(char if (char.isalnum() or char in {"-", "_"}) else "-" for char in raw).strip("-_")
    return cleaned or "host"


def _resolve_run_tag(*, cli_run_tag: str | None, config_run_tag: str | None) -> str:
    if cli_run_tag:
        return str(cli_run_tag)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hostname = _hostname_slug()
    template = str(config_run_tag).strip() if config_run_tag is not None else ""
    if not template:
        template = "{hostname}_{timestamp}"
    return template.format(
        hostname=hostname,
        timestamp=timestamp,
        date=timestamp.split("_", 1)[0],
        time=timestamp.split("_", 1)[1],
    )


def _default_stage_string(config: dict) -> str:
    meta_default = (config.get("_meta") or {}).get("default_stages")
    if meta_default:
        return str(meta_default)
    if (config.get("annotation") or {}).get("command"):
        return "prepare,annotate,infer,filter,build,validate"
    return "prepare,infer,filter,build,validate"


def _resolve_effective_resume(cli_resume, config: dict) -> bool:
    """Resolve the effective resume flag.

    Precedence: explicit CLI value (``--resume`` / ``--no-resume``) wins;
    otherwise the config ``resume`` value; otherwise disabled. Resume is
    opt-in by default so a fresh run never silently skips work.
    """
    if cli_resume is None:
        return bool(config.get("resume", False))
    return bool(cli_resume)


def _native_build_fps_enabled(config: dict) -> bool:
    return bool((config.get("_meta") or {}).get("default_build_fps_from_video"))


def _first_positive_fps(values) -> float | None:
    for value in values:
        if value is None:
            continue
        fps = float(value)
        if fps > 0.0:
            return fps
    return None


def _apply_single_video_native_fps_defaults(
    config: dict,
    *,
    prepared=None,
    descriptors=None,
    manifest_path: Path | None = None,
) -> None:
    """Default build source/target fps from the video when ``_meta`` opts in.

    The fps is taken from the first available source, in order: the prepared
    payload (or its descriptor), an explicit descriptor list, or descriptors
    loaded from ``manifest_path``. Missing values stay unset so the build stage
    can fall back to its own defaults.
    """
    if not _native_build_fps_enabled(config):
        return
    # setdefault (not config.get(...) or {}) so the mutation persists even when
    # the config has no "build" section yet.
    build_cfg = config.setdefault("build", {})
    if build_cfg.get("source_fps") is not None and build_cfg.get("target_fps") is not None:
        return

    fps = None
    if prepared is not None:
        descriptor = prepared.payload.get("descriptor")
        fps = _first_positive_fps([prepared.payload.get("fps"), getattr(descriptor, "fps", None)])
    if fps is None and descriptors is not None:
        fps = _first_positive_fps(getattr(d, "fps", None) for d in descriptors)
    if fps is None and manifest_path is not None and manifest_path.exists():
        records = load_clip_manifest(manifest_path)
        fps = _first_positive_fps(getattr(r.descriptor, "fps", None) for r in records)
    if fps is None:
        return

    if build_cfg.get("source_fps") is None:
        build_cfg["source_fps"] = fps
    if build_cfg.get("target_fps") is None:
        build_cfg["target_fps"] = fps


def _resolved_path_string(path: Path) -> str:
    return str(path.resolve())


def _read_stage_failure_errors(events_path: Path, stage_name: str, *, limit: int = 5) -> list[str]:
    """Pull human-readable failure reasons for a stage from ``events.jsonl``.

    Returns up to ``limit`` ``"<clip>: <error>"`` lines drawn from ``stage_failure``
    events so the orchestrator can surface the real cause at the top level instead
    of forcing the user to open the per-worker logs.
    """
    if not events_path.exists():
        return []
    latest_per_clip: dict[str, str] = {}
    try:
        with events_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") != "stage_failure":
                    continue
                if stage_name and event.get("stage") not in (stage_name, None):
                    continue
                error = str(event.get("error") or "").strip()
                if not error:
                    continue
                clip = str(event.get("video") or event.get("clip_id") or "?")
                latest_per_clip[clip] = error
    except OSError:
        return []
    return [f"{clip}: {error}" for clip, error in list(latest_per_clip.items())[:limit]]


def _set_active_prepared_state(run_summary: dict, state_path: Path) -> None:
    resolved = _resolved_path_string(state_path)
    run_summary["active_manifest_path"] = resolved
    run_summary["active_prepared_state_path"] = resolved


def _print_prepare_summary(
    *,
    adapter_name: str,
    source_id: str,
    split: str,
    records,
    run_dir: Path,
    paths_cfg: dict,
    final_dataset_root: Path,
) -> None:
    descriptor_counts = {
        kind: sum(1 for record in records if classify_descriptor_storage(record.descriptor) == kind)
        for kind in sorted({classify_descriptor_storage(record.descriptor) for record in records})
    }
    print(
        json.dumps(
            {
                "adapter": adapter_name,
                "source_id": source_id,
                "split": split,
                "clip_count": len(records),
                "descriptor_paths": descriptor_counts,
                "output_root": paths_cfg.get("output_root"),
                "run_dir": _resolved_path_string(run_dir),
                "final_dataset_root": _resolved_path_string(final_dataset_root),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def run_pipeline(args) -> None:
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"--config file not found: {config_path}. "
            "Pass a valid pipeline YAML (see configs/dataset_pipeline_single_video.example.yaml)."
        )
    config = normalize_pipeline_config(load_yaml(config_path), config_path=config_path)
    for warning in (config.get("_meta") or {}).get("migration_warnings", []):
        print(f"Warning: {warning}", flush=True)
    raw_stages = args.stages or _default_stage_string(config)
    stage_selection = selected_stages(raw_stages)
    requested_stage_tokens = stage_selection["requested_tokens"]
    stages = stage_selection["internal"]
    public_stages = stage_selection["requested_public"]
    deprecated_stages = stage_selection["deprecated"]

    dataset_cfg = config.get("dataset", {})
    paths_cfg = config.get("paths", {})
    runtimes_cfg = config.get("runtimes", {})
    infer_cfg = config.get("infer", {})
    build_cfg = config.get("build", {})
    filter_cfg = config.get("filter", {})
    adapter_cfg = config.get("adapter_config", {})
    annotation_cfg = config.get("annotation", {})
    validation_cfg = config.get("validation", {})
    cli_resume = getattr(args, "resume", None)
    effective_resume = _resolve_effective_resume(cli_resume, config)
    adapter_cfg.setdefault("resume", effective_resume)
    if cli_resume is not None:
        infer_cfg.setdefault("common", {})["resume"] = effective_resume

    validate_pipeline_cli_alignment(
        stages=stages,
        infer_cfg=infer_cfg,
        build_cfg=build_cfg,
        filter_cfg=filter_cfg,
        validation_cfg=validation_cfg,
    )

    run_root = Path(paths_cfg.get("log_root", PROJECT_ROOT / "pipeline_runs"))
    run_tag = _resolve_run_tag(cli_run_tag=getattr(args, "run_tag", None), config_run_tag=config.get("run_tag"))
    run_dir = run_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "clip_manifest.jsonl"
    filtered_manifest_path = run_dir / "clip_manifest.filtered.jsonl"
    shard_dirs_list_path = run_dir / "shard_dirs.txt"
    summary_path = run_dir / "run_summary.json"
    filter_report_path = run_dir / "filter_report.json"
    shared_feature_cache_dir = run_dir / "_episode_feature_cache"
    external_manifest_path = Path(args.descriptor_manifest).resolve() if getattr(args, "descriptor_manifest", None) else None

    video_clipping_summary = None
    if "preprocess" in stages:
        video_clipping_summary = apply_video_clipping_if_configured(
            config=config,
            run_dir=run_dir,
            project_root=PROJECT_ROOT,
        )
        dataset_cfg = config.get("dataset", dataset_cfg)
        paths_cfg = config.get("paths", paths_cfg)
        adapter_cfg = config.get("adapter_config", adapter_cfg)
        annotation_cfg = config.get("annotation", annotation_cfg)

    adapter_name = dataset_cfg.get("adapter") or dataset_cfg.get("source_type", "buildai")
    source_type = adapter_name
    source_id = dataset_cfg.get("source_id", adapter_name)
    split = dataset_cfg.get("split", "train")
    annotation_root = paths_cfg.get("annotation_root")
    final_dataset_root = Path(paths_cfg["final_dataset_root"])
    resolved_runtimes = resolve_pipeline_runtimes(runtimes_cfg)
    hawor_python = resolved_runtimes.hawor_python or sys.executable
    slam_python = resolved_runtimes.slam_python or hawor_python
    infer_multihost_cfg = parse_multihost_config(
        infer_cfg.get("multihost"),
        default_project_root=PROJECT_ROOT,
        default_hawor_python=hawor_python,
        default_slam_python=slam_python,
    )
    # Preflight (orchestrated path): validate conda runtimes, weights/MANO, and the
    # stage-3 scratch root BEFORE the expensive prepare/frame-extraction stage. The
    # GPU check is intentionally skipped here -- it runs in the correct env inside
    # the batch_infer subprocess, which carries its own preflight.
    if os.environ.get("HAWOR_SKIP_PREFLIGHT", "").strip().lower() not in ("1", "true", "yes", "on"):
        from lib.pipeline.proc import preflight as _preflight

        _pf = _preflight.PreflightReport()
        _preflight.check_runtimes(
            _pf, {"hawor_python": hawor_python, "slam_python": slam_python}
        )
        if _preflight._needs(stages, {"detect_motion", "motion", "infiller"}):
            _preflight.check_weights(
                _pf,
                {
                    "detector": PROJECT_ROOT / "weights" / "external" / "detector.pt",
                    "hawor checkpoint": PROJECT_ROOT / "weights" / "hawor" / "checkpoints" / "hawor.ckpt",
                    "hawor model_config": PROJECT_ROOT / "weights" / "hawor" / "model_config.yaml",
                    "infiller weight": PROJECT_ROOT / "weights" / "hawor" / "checkpoints" / "infiller.pt",
                },
            )
            _preflight.check_mano(_pf, PROJECT_ROOT)
        if _preflight._needs(stages, _preflight._TMP_STAGES):
            _preflight.check_tmp_root(_pf, args=None)
        if not _pf.ok:
            print(_pf.render(), file=sys.stderr)
            print(
                "\nAborting before prepare. Set HAWOR_SKIP_PREFLIGHT=1 to bypass (not recommended).",
                file=sys.stderr,
            )
            raise SystemExit(2)

    adapter = get_dataset_adapter(adapter_name)
    adapter_context = DatasetAdapterContext(
        project_root=PROJECT_ROOT,
        run_dir=run_dir,
        manifest_path=manifest_path,
        shard_dirs_list_path=shard_dirs_list_path,
        summary_path=summary_path,
    )
    prepared = None

    run_summary = {
        "config": str(config_path),
        "run_dir": _resolved_path_string(run_dir),
        "resume": bool(effective_resume),
        "config_schema": (config.get("_meta") or {}).get("schema"),
        "source_type": source_type,
        "source_id": source_id,
        "split": split,
        "manifest_path": _resolved_path_string(manifest_path),
        "active_manifest_path": _resolved_path_string(manifest_path),
        "prepared_state_path": _resolved_path_string(manifest_path),
        "active_prepared_state_path": _resolved_path_string(manifest_path),
        "annotation_root": annotation_root,
        "final_dataset_root": _resolved_path_string(final_dataset_root),
        "feature_cache_dir": _resolved_path_string(shared_feature_cache_dir),
        "stages": public_stages,
        "requested_stage_tokens": requested_stage_tokens,
        "expanded_internal_stages": stages,
        "runtime_source": resolved_runtimes.source,
    }
    if external_manifest_path is not None:
        run_summary["descriptor_manifest_override"] = str(external_manifest_path)
        run_summary["prepared_state_override_path"] = str(external_manifest_path)
    if infer_multihost_cfg.enabled:
        run_summary["infer_multihost"] = infer_multihost_cfg.to_summary()
    if video_clipping_summary is not None:
        run_summary["video_clipping"] = video_clipping_summary
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    use_external_manifest = external_manifest_path is not None and "manifest" not in stages
    if external_manifest_path is not None and "manifest" in stages:
        print(
            "Warning: --descriptor_manifest is ignored because prepare is selected; "
            "later stages will use the newly prepared run state.",
            flush=True,
        )
    active_manifest_path = external_manifest_path if use_external_manifest else manifest_path
    annotation_manifest_path = external_manifest_path if use_external_manifest else manifest_path
    _set_active_prepared_state(run_summary, active_manifest_path)
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if deprecated_stages:
        print(
            "Warning: legacy stage names are deprecated. "
            f"Use official stages from {OFFICIAL_STAGE_ORDER}. "
            f"Received legacy names: {sorted(set(deprecated_stages))}"
        )

    def run_logged(
        name: str,
        cmd: list[str],
        *,
        cwd: str | Path | None = None,
        raise_on_error: bool = True,
    ) -> int:
        log_path = run_dir / f"{name}.log"
        print(f"\n[{name}] running. Log: {log_path}\n", flush=True)
        return stream_command(name, cmd, log_path, cwd=cwd, raise_on_error=raise_on_error)

    def ensure_manifest_exists(stage_label: str, manifest_to_check: Path) -> None:
        if manifest_to_check.exists():
            return
        if external_manifest_path is not None and manifest_to_check.resolve() == external_manifest_path:
            raise RuntimeError(
                f"{stage_label} requires prepared clip state, but the supplied state path does not exist: "
                f"{manifest_to_check}"
            )
        raise RuntimeError(
            f"{stage_label} requires prepared clip state.\n"
            "Run `--stages prepare` first, or rerun with the same `output_root`/`--run_tag` used for preparation.\n"
            f"Looked in run directory: {manifest_to_check.parent}"
        )

    def manifest_uses_only_native_features(manifest_to_check: Path) -> bool:
        records = load_clip_manifest(manifest_to_check)
        return bool(records) and all(
            (record.descriptor.extra or {}).get("native_feature_source") == "wds_lowdim_mano_v1"
            for record in records
        )

    def build_completed_stage_manifest(stage_name: str, source_manifest: Path) -> tuple[Path, dict]:
        status_path = run_dir / "status.json"
        events_path = run_dir / "events.jsonl"
        if not status_path.exists() and not events_path.exists():
            raise RuntimeError(f"Missing status.json after {stage_name}: {status_path}")
        source_records = load_clip_manifest(source_manifest)
        status_payload, status_meta = load_status_payload_with_fallback(
            status_path,
            events_path=events_path,
            video_paths=[record.descriptor.video_key for record in source_records],
            stages=[stage_name],
        )
        if status_payload is None:
            raise RuntimeError(f"Missing recoverable batch status after {stage_name}: {status_path}")
        if status_meta.get("source") != "status":
            print(
                f"[{stage_name}] recovered completed-stage run state from {status_meta.get('source')}",
                flush=True,
            )
        tasks = status_payload.get("tasks", {})
        completed_records = []
        failed_clip_ids = []
        incomplete_clip_ids = []

        for record in source_records:
            task = tasks.get(record.descriptor.video_key) or tasks.get(record.clip_id) or {}
            stage_status = (task.get("stage_status") or {}).get(stage_name)
            if stage_status != "completed" and get_stage_done_marker(Path(record.descriptor.seq_folder), stage_name).exists():
                stage_status = "completed"

            if stage_status == "completed":
                completed_records.append(record)
            elif stage_status == "failed":
                failed_clip_ids.append(record.clip_id)
            else:
                incomplete_clip_ids.append(record.clip_id)

        subset_path = run_dir / f"{source_manifest.stem}.{stage_name}.completed.jsonl"
        write_clip_manifest(completed_records, subset_path)
        summary = {
            "source_manifest": _resolved_path_string(source_manifest),
            "completed_manifest": _resolved_path_string(subset_path),
            "source_prepared_state_path": _resolved_path_string(source_manifest),
            "completed_prepared_state_path": _resolved_path_string(subset_path),
            "total": len(source_records),
            "completed": len(completed_records),
            "failed": len(failed_clip_ids),
            "incomplete": len(incomplete_clip_ids),
            "failed_clip_ids_preview": failed_clip_ids[:16],
            "incomplete_clip_ids_preview": incomplete_clip_ids[:16],
        }
        return subset_path, summary

    def handle_partial_infer_stage(
        *,
        stage_label: str,
        completed_stage_name: str,
        source_manifest: Path,
        return_code: int,
    ) -> Path:
        status_path = run_dir / "status.json"
        if not status_path.exists():
            raise RuntimeError(
                f"{stage_label} failed with exit code {return_code} before status.json was created. "
                f"Check {run_dir / f'{stage_label}.log'}."
            )
        subset_manifest, subset_summary = build_completed_stage_manifest(completed_stage_name, source_manifest)
        run_summary.setdefault("infer_stage_manifests", {})[stage_label] = subset_summary
        _set_active_prepared_state(run_summary, subset_manifest)
        summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        if subset_summary["completed"] <= 0:
            events_path = run_dir / "events.jsonl"
            failure_lines = _read_stage_failure_errors(events_path, stage_label)
            detail = ""
            if failure_lines:
                detail = "\nFirst failure(s):\n" + "\n".join(f"  - {line}" for line in failure_lines)
            raise RuntimeError(
                f"{stage_label} failed with exit code {return_code} and produced no successful clips."
                f"{detail}\n"
                f"Per-clip events: {events_path}\n"
                f"Summary: {json.dumps(subset_summary, ensure_ascii=False)}"
            )
        if return_code != 0:
            print(
                f"[{stage_label}] partial failure tolerated: exit_code={return_code}, "
                f"continuing with completed subset {subset_summary['completed']}/{subset_summary['total']}",
                flush=True,
            )
        return subset_manifest

    def handle_partial_external_stage(
        *,
        stage_label: str,
        source_manifest: Path,
        return_code: int,
        output_exists,
    ) -> Path:
        source_records = load_clip_manifest(source_manifest)
        completed_records = []
        incomplete_clip_ids = []

        for record in source_records:
            seq_folder = Path(record.descriptor.seq_folder)
            if output_exists(seq_folder):
                completed_records.append(record)
            else:
                incomplete_clip_ids.append(record.clip_id)

        subset_path = run_dir / f"{source_manifest.stem}.{stage_label}.completed.jsonl"
        write_clip_manifest(completed_records, subset_path)
        summary = {
            "source_manifest": _resolved_path_string(source_manifest),
            "completed_manifest": _resolved_path_string(subset_path),
            "source_prepared_state_path": _resolved_path_string(source_manifest),
            "completed_prepared_state_path": _resolved_path_string(subset_path),
            "total": len(source_records),
            "completed": len(completed_records),
            "incomplete": len(incomplete_clip_ids),
            "incomplete_clip_ids_preview": incomplete_clip_ids[:16],
        }
        run_summary.setdefault("infer_stage_manifests", {})[stage_label] = summary
        _set_active_prepared_state(run_summary, subset_path)
        summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        if summary["completed"] <= 0:
            raise RuntimeError(
                f"{stage_label} failed with exit code {return_code} and produced no successful clips. "
                f"Summary: {json.dumps(summary, ensure_ascii=False)}"
            )
        print(
            f"[{stage_label}] partial failure tolerated: exit_code={return_code}, "
            f"continuing with completed subset {summary['completed']}/{summary['total']}",
            flush=True,
        )
        return subset_path

    if "preprocess" in stages:
        prepared = adapter.prepare(
            dataset_cfg=dataset_cfg,
            adapter_cfg=adapter_cfg,
            paths_cfg=paths_cfg,
            runtimes_cfg=runtimes_cfg,
            context=adapter_context,
            run_logged=run_logged,
        )
        _apply_single_video_native_fps_defaults(config, prepared=prepared)
        build_cfg = config.get("build", build_cfg)
        filter_cfg = config.get("filter", filter_cfg)

    if "manifest" in stages:
        descriptors = list(
            adapter.build_descriptors(
                dataset_cfg=dataset_cfg,
                adapter_cfg=adapter_cfg,
                paths_cfg=paths_cfg,
                context=adapter_context,
                prepared=prepared,
            )
        )
        _apply_single_video_native_fps_defaults(config, descriptors=descriptors)
        build_cfg = config.get("build", build_cfg)
        filter_cfg = config.get("filter", filter_cfg)
        records = build_manifest_records_from_descriptors(
            descriptors,
            source_id=source_id,
            split=split,
        )
        if not records:
            raise RuntimeError(f"No clips found during prepare for adapter={adapter_name}")
        write_clip_manifest(records, manifest_path)

        shard_root = paths_cfg.get("shard_root")
        if shard_root and source_type == "buildai":
            from lib.pipeline.clips.clip_manifest import discover_shard_dirs

            include_dirs = None
            if prepared is not None:
                include_dirs = prepared.payload.get("include_dirs")
            if include_dirs is None:
                include_dirs = adapter_cfg.get("include_dirs") or dataset_cfg.get("include_dirs")
            shard_dirs = discover_shard_dirs(shard_root, include_dirs=include_dirs)
            write_shard_dir_list(shard_dirs, shard_dirs_list_path)
        _print_prepare_summary(
            adapter_name=adapter_name,
            source_id=source_id,
            split=split,
            records=records,
            run_dir=run_dir,
            paths_cfg=paths_cfg,
            final_dataset_root=final_dataset_root,
        )

    if "annotate" in stages:
        ensure_manifest_exists("annotate", annotation_manifest_path)
        annotation_command = annotation_cfg.get("command")
        if not annotation_command and annotation_cfg.get("_api_clip_completed"):
            run_summary["annotation_stage"] = {
                "status": "skipped",
                "reason": "clip.mode API already wrote annotation sidecars during preprocessing",
            }
            summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            annotation_command = None
        if annotation_command is None and annotation_cfg.get("_api_clip_completed"):
            pass
        elif not annotation_command:
            raise RuntimeError("annotate stage selected but annotation.command is missing in config")
        if annotation_command:
            annotation_context = adapter.resolve_annotation_context(
                dataset_cfg=dataset_cfg,
                adapter_cfg=adapter_cfg,
                paths_cfg=paths_cfg,
                context=adapter_context,
                prepared=prepared,
            )
            context = {
                "manifest": str(annotation_manifest_path),
                "active_manifest": str(active_manifest_path),
                "prepared_state": str(annotation_manifest_path),
                "active_prepared_state": str(active_manifest_path),
                "annotation_root": str(annotation_root or ""),
                "run_dir": str(run_dir),
                "hawor_python": hawor_python,
                "slam_python": slam_python,
                "project_root": str(PROJECT_ROOT),
            }
            context.update(annotation_context)
            run_logged("annotate", format_annotation_command(annotation_command, context))

    common_batch_args = cli_args_from_mapping(
        infer_cfg.get("common"),
        negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
    )
    # The dataset pipeline's build stage still consumes per-clip intermediates
    # (frames, cam-space, depth), so the infer subprocess must NOT mid-clean them.
    # batch_infer defaults to --keep_intermediates none for standalone runs; force
    # 'all' here. Post-build retention is handled by the orchestrator separately.
    if "--keep_intermediates" not in common_batch_args:
        common_batch_args = (*common_batch_args, "--keep_intermediates", "all")
    native_depth_cfg = infer_cfg.get("native_depth") or {}
    native_infer_stages = [stage for stage in ("detect_motion", "slam", "infiller") if stage in stages]
    if native_infer_stages:
        ensure_manifest_exists("infer", active_manifest_path)
        if manifest_uses_only_native_features(active_manifest_path):
            print(
                "[infer] native feature source detected; skipping ordinary "
                f"infer sub-stages {native_infer_stages}. "
                "HOT3D final build reads lowdim/mano/cameras directly from raw WDS.",
                flush=True,
            )
            stages = [stage for stage in stages if stage not in set(native_infer_stages)]
            run_summary["native_feature_infer_skip"] = {
                "skipped_internal_stages": native_infer_stages,
                "reason": "native lowdim/mano/camera features are provided by source WDS",
            }
            if bool(native_depth_cfg.get("enabled")) and "native_depth" not in stages:
                slam_idx = stages.index("slam") if "slam" in stages else None
                if slam_idx is not None:
                    stages.insert(slam_idx + 1, "native_depth")
                else:
                    stages.append("native_depth")
                run_summary["native_feature_infer_skip"]["appended_internal_stages"] = ["native_depth"]
            run_summary["expanded_internal_stages_after_native_skip"] = stages
            summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    multihost_runner = None
    multihost_common_batch_args = ()
    if infer_multihost_cfg.enabled and any(stage in stages for stage in ("detect_motion", "slam", "infiller")):
        validate_multihost_infer_alignment(infer_cfg)
        multihost_common_batch_args = tuple(
            cli_args_from_mapping(
                sanitize_infer_args_for_multihost(
                    infer_cfg.get("common"),
                    reserved_keys=MULTIHOST_DISALLOWED_INFER_KEYS,
                ),
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            )
        )
        multihost_runner = MultihostStageQueueRunner(
            config=infer_multihost_cfg,
            manifest_path=active_manifest_path,
            run_dir=run_dir,
            infer_resume=bool((infer_cfg.get("common") or {}).get("resume", True)),
        )

    def run_batch_infer_stage(
        *,
        pipeline_stage: str,
        batch_stages: str,
        python_exe: str,
        runtime_key: str,
        args_key: str,
        completed_stage_name: str,
    ) -> Path:
        """Run one batch_infer.py infer stage (detect_motion / slam / infiller-standard).

        These three share the same shape: dispatch via the multihost runner when enabled, else
        spawn batch_infer.py locally and fold any partial failure into the active manifest. Returns
        the (possibly narrowed) active manifest path. Callers run ensure_manifest_exists first.
        """
        if multihost_runner is not None:
            result = multihost_runner.run_stage(
                MultihostStageSpec(
                    pipeline_stage=pipeline_stage,
                    batch_stages=batch_stages,
                    runtime_key=runtime_key,
                    extra_args=multihost_common_batch_args
                    + tuple(
                        cli_args_from_mapping(
                            sanitize_infer_args_for_multihost(
                                infer_cfg.get(args_key),
                                reserved_keys=MULTIHOST_DISALLOWED_INFER_KEYS,
                            ),
                            negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
                        )
                    ),
                    worker_count_per_gpu=infer_stage_worker_count_per_gpu(
                        pipeline_stage=pipeline_stage,
                        infer_cfg=infer_cfg,
                    ),
                )
            )
            run_summary.setdefault("multihost_dispatch", {})[pipeline_stage] = result["dispatch_path"]
            summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            if not result["success"]:
                raise RuntimeError(
                    f"{pipeline_stage} multihost stage failed: "
                    + json.dumps(result["failed_shards"], ensure_ascii=False)
                )
            return active_manifest_path

        stage_args = tuple(
            cli_args_from_mapping(
                infer_cfg.get(args_key),
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            )
        )
        return_code = run_logged(
            pipeline_stage,
            [
                python_exe,
                str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
                "--descriptor_manifest",
                str(active_manifest_path),
                "--run_dir",
                str(run_dir),
                "--stages",
                batch_stages,
                *common_batch_args,
                *stage_args,
            ],
            raise_on_error=False,
        )
        return handle_partial_infer_stage(
            stage_label=pipeline_stage,
            completed_stage_name=completed_stage_name,
            source_manifest=active_manifest_path,
            return_code=return_code,
        )

    if "detect_motion" in stages:
        ensure_manifest_exists("detect_motion", active_manifest_path)
        active_manifest_path = run_batch_infer_stage(
            pipeline_stage="detect_motion",
            batch_stages="detect_track,motion",
            python_exe=hawor_python,
            runtime_key="hawor",
            args_key="detect_motion",
            completed_stage_name="motion",
        )

    if "slam" in stages:
        ensure_manifest_exists("slam", active_manifest_path)
        active_manifest_path = run_batch_infer_stage(
            pipeline_stage="slam",
            batch_stages="slam",
            python_exe=slam_python,
            runtime_key="slam",
            args_key="slam",
            completed_stage_name="slam",
        )

    if "native_depth" in stages:
        ensure_manifest_exists("native_depth", active_manifest_path)
        common_infer_cfg = infer_cfg.get("common") or {}
        gpus = native_depth_cfg.get("gpus", common_infer_cfg.get("gpus", "0"))
        if isinstance(gpus, (list, tuple)):
            gpus = ",".join(str(item).strip() for item in gpus if str(item).strip())
        native_depth_common_cfg = {
            key: common_infer_cfg.get(key)
            for key in SHARED_PROFILE_CACHE_OPTION_DESTS
            if common_infer_cfg.get(key) is not None
        }
        native_depth_args = tuple(
            cli_args_from_mapping(
                {
                    key: value
                    for key, value in {**native_depth_common_cfg, **native_depth_cfg}.items()
                    if key not in {"enabled", "gpus"}
                },
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            )
        )
        native_depth_cmd = [
            slam_python,
            str(PROJECT_ROOT / "scripts" / "build" / "run_hot3d_native_depth.py"),
            "--descriptor_manifest",
            str(active_manifest_path),
            "--run_dir",
            str(run_dir),
            "--gpus",
            str(gpus),
            *native_depth_args,
        ]
        if bool(common_infer_cfg.get("resume", effective_resume)):
            native_depth_cmd.append("--resume")
        native_depth_return_code = run_logged(
            "native_depth",
            native_depth_cmd,
            raise_on_error=False,
        )
        if native_depth_return_code != 0:
            active_manifest_path = handle_partial_external_stage(
                stage_label="native_depth",
                source_manifest=active_manifest_path,
                return_code=native_depth_return_code,
                output_exists=lambda seq_folder: (
                    get_stage_done_marker(seq_folder, "native_depth").exists()
                    and (seq_folder / "NATIVE_DEPTH" / "any4d_depth.npz").is_file()
                ),
            )

    if "infiller" in stages:
        ensure_manifest_exists("infiller", active_manifest_path)
        fpha_skeleton_cfg = (
            adapter_cfg.get("fpha_skeleton")
            if adapter_name == "fpha_tar"
            else None
        ) or {}
        use_fpha_skeleton_infiller = bool(fpha_skeleton_cfg.get("enabled"))
        if use_fpha_skeleton_infiller:
            if multihost_runner is not None:
                raise ValueError("FPHA skeleton infiller path does not support infer.multihost")
            common_infer_cfg = infer_cfg.get("common") or {}
            device = str(fpha_skeleton_cfg.get("device") or "cuda:0")
            raw_gpus = common_infer_cfg.get("gpus")
            if fpha_skeleton_cfg.get("device") is None and raw_gpus is not None:
                if isinstance(raw_gpus, list):
                    first_gpu = str(raw_gpus[0]).strip() if raw_gpus else ""
                else:
                    first_gpu = str(raw_gpus).split(",")[0].strip()
                if first_gpu:
                    device = first_gpu if first_gpu.startswith("cuda:") else f"cuda:{first_gpu}"
            fpha_cmd = [
                hawor_python,
                str(PROJECT_ROOT / "scripts" / "build" / "generate_fpha_world_res.py"),
                "--descriptor_manifest",
                str(active_manifest_path),
                "--device",
                device,
                "--num_iters",
                str(int(fpha_skeleton_cfg.get("num_iters", 180))),
                "--lr",
                str(float(fpha_skeleton_cfg.get("lr", 1e-2))),
                "--pose_reg",
                str(float(fpha_skeleton_cfg.get("pose_reg", 1e-4))),
                "--shape_reg",
                str(float(fpha_skeleton_cfg.get("shape_reg", 1e-3))),
                "--temporal_reg",
                str(float(fpha_skeleton_cfg.get("temporal_reg", 1e-3))),
            ]
            if fpha_skeleton_cfg.get("shape_iters") is not None:
                fpha_cmd.extend(["--shape_iters", str(int(fpha_skeleton_cfg["shape_iters"]))])
            if fpha_skeleton_cfg.get("shape_sample_size") is not None:
                fpha_cmd.extend(["--shape_sample_size", str(int(fpha_skeleton_cfg["shape_sample_size"]))])
            if fpha_skeleton_cfg.get("chunk_size") is not None:
                fpha_cmd.extend(["--chunk_size", str(int(fpha_skeleton_cfg["chunk_size"]))])
            if fpha_skeleton_cfg.get("skeleton_root"):
                fpha_cmd.extend(["--skeleton_root", str(fpha_skeleton_cfg["skeleton_root"])])
            if bool(common_infer_cfg.get("resume", effective_resume)):
                fpha_cmd.append("--resume")
            if not bool(fpha_skeleton_cfg.get("preserve_existing_left", True)):
                fpha_cmd.append("--no-preserve_existing_left")

            infiller_return_code = run_logged(
                "infiller",
                fpha_cmd,
                raise_on_error=False,
            )
            if infiller_return_code != 0:
                active_manifest_path = handle_partial_external_stage(
                    stage_label="infiller",
                    source_manifest=active_manifest_path,
                    return_code=infiller_return_code,
                    output_exists=lambda seq_folder: (
                        (seq_folder / "world_space_res.pth").is_file()
                        and get_stage_done_marker(seq_folder, "infiller").exists()
                    ),
                )
        else:
            active_manifest_path = run_batch_infer_stage(
                pipeline_stage="infiller",
                batch_stages="infiller",
                python_exe=hawor_python,
                runtime_key="hawor",
                args_key="infiller",
                completed_stage_name="infiller",
            )

    if "filter" in stages:
        ensure_manifest_exists("filter", active_manifest_path)
        _apply_single_video_native_fps_defaults(config, manifest_path=active_manifest_path)
        build_cfg = config.get("build", build_cfg)
        filter_runtime_cfg = dict(filter_cfg)
        filter_runtime_cfg.setdefault("annotation_root", annotation_root)
        filter_runtime_cfg.setdefault("annotation_suffix", build_cfg.get("annotation_suffix"))
        filter_runtime_cfg.setdefault("require_annotation", build_cfg.get("require_annotation"))
        filter_runtime_cfg.setdefault("source_fps", build_cfg.get("source_fps"))
        filter_runtime_cfg.setdefault("target_fps", build_cfg.get("target_fps"))
        filter_runtime_cfg.setdefault("interpolate_labels", build_cfg.get("interpolate_labels"))
        filter_runtime_cfg.setdefault("mano_device", build_cfg.get("mano_device"))
        if build_cfg.get("mano_gpus") is not None:
            filter_runtime_cfg.setdefault("mano_gpus", build_cfg.get("mano_gpus"))
        if build_cfg.get("mano_dir") is not None:
            filter_runtime_cfg.setdefault("mano_dir", build_cfg.get("mano_dir"))
        filter_runtime_cfg.setdefault("feature_cache_dir", str(shared_feature_cache_dir))
        run_logged(
            "filter",
            [
                hawor_python,
                str(PROJECT_ROOT / "scripts" / "build" / "filter_manifest_by_quality.py"),
                "--input_manifest",
                str(active_manifest_path),
                "--output_manifest",
                str(filtered_manifest_path),
                "--report_out",
                str(filter_report_path),
                *cli_args_from_mapping(filter_runtime_cfg),
            ],
        )
        active_manifest_path = filtered_manifest_path
        _set_active_prepared_state(run_summary, active_manifest_path)
        run_summary["filter_report_path"] = _resolved_path_string(filter_report_path)
        summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if "build" in stages:
        ensure_manifest_exists("build", active_manifest_path)
        _apply_single_video_native_fps_defaults(config, manifest_path=active_manifest_path)
        build_cfg = config.get("build", build_cfg)
        build_runtime_cfg = dict(build_cfg)
        build_runtime_cfg.setdefault("feature_cache_dir", str(shared_feature_cache_dir))
        build_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "build" / "build_vla_from_manifest.py"),
            "--descriptor_manifest",
            str(active_manifest_path),
            "--output_dir",
            str(final_dataset_root),
            *cli_args_from_mapping(build_runtime_cfg),
        ]
        if effective_resume:
            build_cmd.append("--resume")
        if annotation_root:
            build_cmd.extend(["--annotation_root", str(annotation_root)])
        run_logged("build", build_cmd)

    if "validate" in stages:
        ensure_manifest_exists("validate", active_manifest_path)
        source_validation = adapter.validate_source(
            dataset_cfg=dataset_cfg,
            adapter_cfg=adapter_cfg,
            paths_cfg=paths_cfg,
            context=adapter_context,
            prepared=prepared,
        )
        if source_validation.summary:
            print(json.dumps({"source_validation": source_validation.summary}, ensure_ascii=False, indent=2))
        if not source_validation.ok:
            raise RuntimeError(f"Source validation failed for adapter={adapter_name}: {source_validation.summary}")
        validate_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "inspection" / "validate_pipeline_run.py"),
            "--descriptor_manifest",
            str(active_manifest_path),
            "--dataset_dir",
            str(final_dataset_root),
            *cli_args_from_mapping(validation_cfg),
        ]
        if annotation_root:
            validate_cmd.extend(["--annotation_root", str(annotation_root)])
            if build_cfg.get("annotation_suffix"):
                validate_cmd.extend(["--annotation_suffix", str(build_cfg["annotation_suffix"])])
        if bool(validation_cfg.get("depth_action_consistency", False)) and not validation_cfg.get("depth_action_report_out"):
            validate_cmd.extend(["--depth_action_report_out", str(run_dir / "depth_action_consistency.json")])
        run_logged("validate", validate_cmd)

    print(f"\nRun complete: {run_dir}")


def main(argv: list[str] | None = None) -> None:
    from lib.pipeline.proc.logging_setup import configure_logging

    configure_logging()
    args = get_parser().parse_args(argv)
    run_pipeline(args)
