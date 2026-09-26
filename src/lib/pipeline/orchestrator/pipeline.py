"""Official dataset pipeline orchestration."""

from __future__ import annotations

import json
import os
import shlex
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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
from .constants import (
    BATCH_INFER_NEGATIVE_BOOL_FLAGS,
    BUILD_NEGATIVE_BOOL_FLAGS,
    FILTER_NEGATIVE_BOOL_FLAGS,
    MULTIHOST_DISALLOWED_INFER_KEYS,
    OFFICIAL_STAGE_ORDER,
)
from .helpers import cli_args_from_mapping, format_annotation_command, load_yaml, stream_command
from .lerobot_stage import build_lerobot_stage_command
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
    otherwise the config ``resume`` value; otherwise disabled. Config
    normalization defaults ``resume`` to true for both config schemas, so
    resume is on by default unless the CLI or the config turns it off.
    """
    if cli_resume is None:
        return bool(config.get("resume", False))
    return bool(cli_resume)


def _physical_to_logical_gpu_ids(gpus, config_key: str = "infer.common.gpus") -> list[str]:
    """Map physical GPU ids (``infer.common.gpus`` semantics) to torch device indices.

    Batch infer workers export ``CUDA_VISIBLE_DEVICES=<id>`` themselves, so ``gpus``
    holds physical ids. Stages that instead pass ``cuda:<n>`` to a child which
    inherits this process's environment need the index of that id inside an
    integer ``CUDA_VISIBLE_DEVICES`` list. Without one (unset, empty, or UUID/MIG
    handles) ids are returned unchanged; ``cpu`` and explicit ``cuda:<n>`` device
    strings are passed through as-is.
    """
    if isinstance(gpus, (list, tuple)):
        tokens = [str(item).strip() for item in gpus if str(item).strip()]
    else:
        tokens = [item.strip() for item in str(gpus).split(",") if item.strip()]
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_tokens = [item.strip() for item in (raw_visible or "").split(",") if item.strip()]
    try:
        visible = [int(item) for item in visible_tokens] if visible_tokens else None
    except ValueError:
        visible = None
    if visible is None:
        return tokens
    mapped = []
    for token in tokens:
        if token.lower() == "cpu" or token.startswith("cuda:"):
            mapped.append(token)
            continue
        try:
            physical = int(token)
        except ValueError as error:
            raise ValueError(f"{config_key}: GPU id {token!r} is not an integer physical GPU id") from error
        if physical not in visible:
            raise ValueError(
                f"{config_key}: GPU id {physical} is not in CUDA_VISIBLE_DEVICES={raw_visible}; "
                f"{config_key} takes physical GPU ids, so pick ids from CUDA_VISIBLE_DEVICES"
            )
        mapped.append(str(visible.index(physical)))
    return mapped


def _resolve_effective_infer_resume(cli_resume, infer_common_cfg: dict | None, effective_resume: bool) -> bool:
    """Resolve the resume flag passed to every infer sub-stage.

    Precedence: explicit CLI value > explicit ``infer.common.resume`` > the
    top-level effective resume, so ``resume: false`` also disables infer resume.
    """
    if cli_resume is not None:
        return bool(cli_resume)
    explicit = (infer_common_cfg or {}).get("resume")
    if explicit is not None:
        return bool(explicit)
    return bool(effective_resume)


def _resolve_infer_section_resume(cli_resume, section_cfg: dict | None, infer_common_resume: bool) -> bool:
    """Resolve the resume flag of one infer sub-stage (detect_motion/slam/infiller/native_depth).

    Precedence: explicit CLI value > explicit ``infer.<section>.resume`` >
    ``infer_common_resume`` (already resolved by ``_resolve_effective_infer_resume``).
    """
    if cli_resume is not None:
        return bool(cli_resume)
    explicit = (section_cfg or {}).get("resume")
    if explicit is not None:
        return bool(explicit)
    return bool(infer_common_resume)


def _effective_infer_option(infer_cfg: dict, section: str, key: str):
    """Value of ``key`` the ``section`` infer child will see.

    The child CLI is ``infer.common`` args followed by ``infer.<section>`` args,
    so a non-null stage-specific value overrides the common one.
    """
    value = (infer_cfg.get(section) or {}).get(key)
    if value is None:
        value = (infer_cfg.get("common") or {}).get(key)
    return value


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


NATIVE_FEATURE_SOURCE = "wds_lowdim_mano_v1"


def _manifest_is_native_only(manifest_path: Path) -> bool:
    """True when every clip of the manifest reads its hand features natively from the source WDS."""
    records = load_clip_manifest(manifest_path)
    return bool(records) and all(
        (record.descriptor.extra or {}).get("native_feature_source") == NATIVE_FEATURE_SOURCE
        for record in records
    )


CLIP_REDIRECT_FILENAME = "clip_redirect.json"
# paths.* keys the clip redirect rewrites (see video_clipping._redirect_to_clipped_video_folder)
_CLIP_REDIRECT_PATH_KEYS = ("annotation_root", "clip_root", "video_root", "frames_root", "seq_folder_root")
_CLIP_SECRET_KEYS = {"api_key", "api_keys"}
# clip.* keys that only change how clipping executes (concurrency, retries, where the report goes,
# key source), not which clips it produces; they must not invalidate an existing clip redirect.
_CLIP_EXECUTION_ONLY_KEYS = {"workers", "max_api_retries", "report_out", "dry_run", "api_keys_file"}
# paths.* keys that select the raw clip inputs (video_clipping._source_videos_for_config)
_CLIP_SOURCE_PATH_KEYS = ("video_root", "data_root")


def _clip_mode_enabled(config: dict) -> bool:
    mode = str((config.get("clip") or {}).get("mode", "none")).strip().lower()
    return mode not in {"", "none", "off", "false"}


def _clip_redirect_fingerprint(config: dict) -> dict:
    """Summary of the (pre-redirect) clip + source configuration a clip redirect belongs to.

    ``resume`` is excluded (it changes per invocation), and so are API keys (never written to disk)
    and execution-only clip keys (``_CLIP_EXECUTION_ONLY_KEYS``: concurrency / retries / report path).
    """
    clip_cfg = {
        key: value
        for key, value in (config.get("clip") or {}).items()
        if key not in _CLIP_SECRET_KEYS and key not in _CLIP_EXECUTION_ONLY_KEYS
    }
    adapter_cfg = {key: value for key, value in (config.get("adapter_config") or {}).items() if key != "resume"}
    dataset_cfg = config.get("dataset") or {}
    paths_cfg = config.get("paths") or {}
    fingerprint = {
        "clip": clip_cfg,
        "adapter": dataset_cfg.get("adapter"),
        "adapter_config": adapter_cfg,
        # Which raw videos get clipped (video_clipping._source_videos_for_config): the source roots
        # under paths.* and the dataset selection (e.g. factory range), not only the adapter name.
        "source": {
            "dataset": {key: value for key, value in dataset_cfg.items() if key != "adapter"},
            "paths": {key: paths_cfg[key] for key in _CLIP_SOURCE_PATH_KEYS if paths_cfg.get(key) is not None},
        },
    }
    return json.loads(json.dumps(fingerprint, sort_keys=True, default=str))


def _write_clip_redirect(path: Path, config: dict, *, fingerprint: dict, video_clipping_summary: dict) -> None:
    """Persist the in-memory clip redirect so later invocations without preprocess can restore it."""
    paths_cfg = config.get("paths") or {}
    payload = {
        "clip_config": fingerprint,
        "dataset": config.get("dataset") or {},
        "adapter_config": {key: value for key, value in (config.get("adapter_config") or {}).items() if key != "resume"},
        "paths": {key: paths_cfg[key] for key in _CLIP_REDIRECT_PATH_KEYS if paths_cfg.get(key) is not None},
        "api_clip_completed": bool((config.get("annotation") or {}).get("_api_clip_completed")),
        "video_clipping": video_clipping_summary,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _restore_clip_redirect(path: Path, config: dict, *, fingerprint: dict) -> dict | None:
    """Re-apply a clip redirect written by an earlier ``preprocess`` run; None when there is none."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = payload.get("clip_config")
    if isinstance(stored, dict) and isinstance(stored.get("clip"), dict):
        # redirects written before execution-only keys were excluded still carry them
        stored = {
            **stored,
            "clip": {key: value for key, value in stored["clip"].items() if key not in _CLIP_EXECUTION_ONLY_KEYS},
        }
    if stored != fingerprint:
        raise RuntimeError(
            f"{path} was written by a run with a different clip/source configuration than the current config "
            "(clip block, dataset, adapter_config or source paths such as paths.video_root changed). Later stages would mix clipped outputs "
            "with the new configuration. Rerun with `--stages prepare,...` to clip again, or restore the "
            "clip configuration used for that run."
        )
    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg.update(payload.get("dataset") or {})
    adapter_cfg = config.setdefault("adapter_config", {})
    resume = adapter_cfg.get("resume")
    adapter_cfg.clear()
    adapter_cfg.update(payload.get("adapter_config") or {})
    if resume is not None:
        adapter_cfg["resume"] = resume
    config.setdefault("paths", {}).update(payload.get("paths") or {})
    if payload.get("api_clip_completed"):
        config.setdefault("annotation", {})["_api_clip_completed"] = True
    return payload


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


def _annotation_report_path(command: str, annotation_root) -> Path | None:
    """Report path ``api_annotation.py`` writes for ``command`` (``--report_out`` or the default)."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = []
    report_path = None
    for index, token in enumerate(tokens):
        if token == "--report_out" and index + 1 < len(tokens):
            report_path = Path(tokens[index + 1])
        elif token.startswith("--report_out="):
            report_path = Path(token.split("=", 1)[1])
    if report_path is None:
        if not annotation_root:
            return None
        report_path = Path(str(annotation_root)) / "_annotation_report.json"
    return report_path


def _file_signature(path: Path | None):
    """(mtime_ns, size, inode) of ``path``, or None when it does not exist."""
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _annotation_partial_failure(command: str, annotation_root, *, report_before) -> dict | None:
    """Return the report summary of an annotation run that failed only for some clips.

    ``api_annotation.py`` exits 1 when any clip failed, after writing its JSON report
    (``--report_out`` or ``<annotation_root>/_annotation_report.json``). The run counts as a
    partial failure when that report was written by this run — it differs from
    ``report_before``, the ``_file_signature`` taken right before the command was launched — and
    at least one clip has a usable sidecar (written, invalid_written or skipped-existing).
    Anything else (no report, a report left unchanged from an earlier run, every clip failed)
    returns None and aborts as before.
    """
    report_path = _annotation_report_path(command, annotation_root)
    if report_path is None:
        return None
    try:
        report_now = _file_signature(report_path)
        if report_now is None or report_now == report_before:
            return None
        summary = json.loads(report_path.read_text(encoding="utf-8")).get("summary") or {}
        failed = int(summary.get("failed", 0))
        usable = sum(int(summary.get(key, 0)) for key in ("written", "invalid_written", "skipped"))
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    if failed <= 0 or usable <= 0:
        return None
    return {**summary, "report_path": str(report_path)}


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
    lerobot_cfg = config.get("lerobot", {})
    cli_resume = getattr(args, "resume", None)
    effective_resume = _resolve_effective_resume(cli_resume, config)
    # Publish the resolved value so every later reader of config["resume"]
    # (e.g. video clipping) sees the same decision as prepare/build.
    config["resume"] = effective_resume
    if cli_resume is not None:
        adapter_cfg["resume"] = effective_resume
    else:
        adapter_cfg.setdefault("resume", effective_resume)
    infer_common_cfg = infer_cfg.setdefault("common", {})
    infer_common_cfg["resume"] = _resolve_effective_infer_resume(cli_resume, infer_common_cfg, effective_resume)

    validate_pipeline_cli_alignment(
        stages=stages,
        infer_cfg=infer_cfg,
        build_cfg=build_cfg,
        filter_cfg=filter_cfg,
        validation_cfg=validation_cfg,
        lerobot_cfg=lerobot_cfg,
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
    clip_redirect_path = run_dir / CLIP_REDIRECT_FILENAME
    clip_redirect_restored = None
    if "preprocess" in stages:
        clip_fingerprint = _clip_redirect_fingerprint(config)
        video_clipping_summary = apply_video_clipping_if_configured(
            config=config,
            run_dir=run_dir,
            project_root=PROJECT_ROOT,
        )
        if video_clipping_summary is not None:
            # The redirect only lives in this process: persist it for later downstream-only runs.
            _write_clip_redirect(
                clip_redirect_path, config, fingerprint=clip_fingerprint, video_clipping_summary=video_clipping_summary
            )
        dataset_cfg = config.get("dataset", dataset_cfg)
        paths_cfg = config.get("paths", paths_cfg)
        adapter_cfg = config.get("adapter_config", adapter_cfg)
        annotation_cfg = config.get("annotation", annotation_cfg)
    elif _clip_mode_enabled(config):
        clip_redirect_restored = _restore_clip_redirect(
            clip_redirect_path, config, fingerprint=_clip_redirect_fingerprint(config)
        )
        if clip_redirect_restored is not None:
            print(f"[clip] restored clip redirect from {clip_redirect_path}", flush=True)
            dataset_cfg = config.get("dataset", dataset_cfg)
            paths_cfg = config.get("paths", paths_cfg)
            adapter_cfg = config.get("adapter_config", adapter_cfg)
            annotation_cfg = config.get("annotation", annotation_cfg)
        else:
            print(
                f"Warning: clip.mode is configured but {clip_redirect_path} is missing (prepare has not run "
                "for this run directory, or ran before clip redirects were persisted); later stages use the "
                "unclipped source configuration. Rerun with `--stages prepare,...` to restore it.",
                flush=True,
            )

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
        _preflight_stages = stages
        _native_candidate = external_manifest_path or manifest_path
        if (adapter_name == "hot3d_wds") if "manifest" in stages else (
            _native_candidate.exists() and _manifest_is_native_only(_native_candidate)
        ):
            # Native-feature manifests skip the ordinary infer sub-stages (see the native skip below), and
            # native_depth does not use the stage-3 scratch root: no HaWoR weights / MANO / scratch checks.
            _preflight_stages = [
                stage for stage in stages if stage not in {"detect_motion", "slam", "infiller", "native_depth"}
            ]
        if _preflight._needs(_preflight_stages, {"detect_motion", "motion", "infiller"}):
            # Check the weights the selected stages will actually load, honoring
            # infer.<stage>/infer.common checkpoint overrides (same precedence as
            # the child CLI, where stage args follow common args).
            _weights = {}
            if _preflight._needs(_preflight_stages, {"detect_motion", "motion"}):
                _checkpoint = Path(
                    _effective_infer_option(infer_cfg, "detect_motion", "checkpoint")
                    or PROJECT_ROOT / "weights" / "hawor" / "checkpoints" / "hawor.ckpt"
                )
                _weights["detector"] = PROJECT_ROOT / "weights" / "external" / "detector.pt"
                _weights["hawor checkpoint"] = _checkpoint
                # hawor_runtime loads model_config.yaml from <checkpoint>/../../
                _weights["hawor model_config"] = _checkpoint.parent.parent / "model_config.yaml"
            if "infiller" in _preflight_stages:
                _weights["infiller weight"] = (
                    _effective_infer_option(infer_cfg, "infiller", "infiller_weight")
                    or PROJECT_ROOT / "weights" / "hawor" / "checkpoints" / "infiller.pt"
                )
            _preflight.check_weights(_pf, _weights)
            _preflight.check_mano(_pf, PROJECT_ROOT)
        if _preflight._needs(_preflight_stages, _preflight._TMP_STAGES):
            _tmp_args = None
            if "slam" in _preflight_stages:
                _stage3_tmp_root = _effective_infer_option(infer_cfg, "slam", "stage3_tmp_root")
                if _stage3_tmp_root:
                    _tmp_args = SimpleNamespace(stage3_tmp_root=str(_stage3_tmp_root))
            _preflight.check_tmp_root(_pf, args=_tmp_args)
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
    if clip_redirect_restored is not None:
        run_summary["clip_redirect_restored_from"] = _resolved_path_string(clip_redirect_path)
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
        return _manifest_is_native_only(manifest_to_check)

    def native_check_stages() -> str:
        # Same stage set the native infer path runs (see the native skip below).
        return "native_features,native_depth" if bool(native_depth_cfg.get("enabled")) else "native_features"

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
            # An explicit batch verdict wins: a leftover .done from an earlier run must not turn a
            # "failed" clip into "completed". The marker is only consulted when the batch status has
            # no verdict for this clip (absent, or a "pending" placeholder synthesized by the
            # events.jsonl recovery for clips that emitted no event for this stage).
            no_batch_verdict = stage_status is None or (
                stage_status == "pending" and status_meta.get("source") == "events"
            )
            if no_batch_verdict and get_stage_done_marker(Path(record.descriptor.seq_folder), stage_name).exists():
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
        if annotation_cfg.get("_api_clip_completed"):
            # The API clipping step already wrote the sidecars; running annotation.command
            # too would pay for a second VLM pass over the same clips.
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
            annotation_cmd = format_annotation_command(annotation_command, context)
            # Snapshot the report before launching: only a report this run (re)wrote may turn a
            # non-zero exit into a tolerated partial failure.
            annotation_report_before = _file_signature(_annotation_report_path(annotation_cmd[-1], annotation_root))
            annotation_return_code = run_logged("annotate", annotation_cmd, raise_on_error=False)
            if annotation_return_code != 0:
                # Like infer: tolerate per-clip failures and continue; clips without a sidecar are
                # then dropped or kept unannotated downstream according to build.require_annotation.
                partial = _annotation_partial_failure(
                    annotation_cmd[-1], annotation_root, report_before=annotation_report_before
                )
                if partial is None:
                    raise RuntimeError(
                        f"annotate failed with exit code {annotation_return_code}. Check {run_dir / 'annotate.log'}."
                    )
                print(
                    f"[annotate] partial failure tolerated: exit_code={annotation_return_code}, "
                    f"failed clips={partial.get('failed')}/{partial.get('total')}, report: {partial['report_path']}",
                    flush=True,
                )
                run_summary["annotation_stage"] = {
                    "status": "partial",
                    "exit_code": annotation_return_code,
                    **partial,
                }
                summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # resume is emitted once per infer sub-stage (see _resolve_infer_section_resume), not here.
    common_batch_args = cli_args_from_mapping(
        {key: value for key, value in (infer_cfg.get("common") or {}).items() if key != "resume"},
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
                    reserved_keys=MULTIHOST_DISALLOWED_INFER_KEYS | {"resume"},
                ),
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            )
        )
        # Same retention guard as the local path above: remote infer workers must
        # keep the intermediates filter/build still read, unless set explicitly.
        if "--keep_intermediates" not in multihost_common_batch_args:
            multihost_common_batch_args = (*multihost_common_batch_args, "--keep_intermediates", "all")
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
        stage_resume_flag = (
            "--resume"
            if _resolve_infer_section_resume(cli_resume, infer_cfg.get(args_key), infer_common_cfg["resume"])
            else "--no-resume"
        )
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
                                reserved_keys=MULTIHOST_DISALLOWED_INFER_KEYS | {"resume"},
                            ),
                            negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
                        )
                    )
                    + (stage_resume_flag,),
                    worker_count_per_gpu=infer_stage_worker_count_per_gpu(
                        pipeline_stage=pipeline_stage,
                        infer_cfg=infer_cfg,
                    ),
                    resume=stage_resume_flag == "--resume",
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
                {key: value for key, value in (infer_cfg.get(args_key) or {}).items() if key != "resume"},
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            )
        ) + (stage_resume_flag,)
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
        gpus = native_depth_cfg.get("gpus", common_infer_cfg.get("gpus"))
        if gpus is None:
            # Unconfigured: first visible device (logical 0), as before; nothing to map.
            gpus = "0"
        else:
            # run_hot3d_native_depth.py uses cuda:<id> under the inherited visibility.
            gpus_key = "native_depth.gpus" if native_depth_cfg.get("gpus") is not None else "infer.common.gpus"
            gpus = ",".join(_physical_to_logical_gpu_ids(gpus, gpus_key))
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
                    if key not in {"enabled", "gpus", "resume"}
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
        # run_hot3d_native_depth.py's --resume is store_true (no --no-resume): emit it only when
        # resuming. native_depth.resume overrides the infer-wide value unless the CLI set one.
        native_depth_resume = common_infer_cfg.get("resume", effective_resume)
        if cli_resume is None and native_depth_cfg.get("resume") is not None:
            native_depth_resume = native_depth_cfg["resume"]
        if bool(native_depth_resume):
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
                    # cuda:<n> is resolved under the inherited visibility (logical index).
                    first_gpu = _physical_to_logical_gpu_ids([first_gpu])[0]
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
            # generate_fpha_world_res.py defaults --resume to True: pass the negative form explicitly.
            fpha_resume = _resolve_infer_section_resume(
                cli_resume, infer_cfg.get("infiller"), common_infer_cfg.get("resume", effective_resume)
            )
            fpha_cmd.append("--resume" if fpha_resume else "--no-resume")
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

    # Downstream-only reruns (e.g. `--stages build,validate,lerobot`) reuse the
    # previous filter result instead of re-exporting clips it dropped. Only when
    # nothing upstream of filter ran now (infer narrows the manifest itself) and
    # the filtered manifest is not older than the prepared one (or than any
    # infer completed-subset manifest written by an earlier invocation).
    if (
        "filter" not in stages
        and any(stage in stages for stage in ("build", "validate", "lerobot"))
        and not any(
            stage in stages
            for stage in ("preprocess", "manifest", "annotate", "detect_motion", "slam", "native_depth", "infiller")
        )
        and active_manifest_path == manifest_path
        and manifest_path.exists()
        and filtered_manifest_path.exists()
    ):
        newer_than_filter = [
            path.name
            for path in (manifest_path, *run_dir.glob(f"{manifest_path.stem}.*.completed.jsonl"))
            if path.stat().st_mtime > filtered_manifest_path.stat().st_mtime
        ]
        if not newer_than_filter:
            print(
                f"[filter] not selected; using previous filter result {filtered_manifest_path} "
                f"instead of {manifest_path}",
                flush=True,
            )
            active_manifest_path = filtered_manifest_path
            _set_active_prepared_state(run_summary, active_manifest_path)
            summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print(
                f"Warning: [filter] not selected and the previous filter result {filtered_manifest_path} is older "
                f"than {', '.join(sorted(newer_than_filter))} (prepare or infer was rerun after the last filter), "
                f"so it is NOT reused: later stages read the unfiltered manifest {manifest_path} and include clips "
                "the filter may reject. Add `filter` to --stages to re-filter the new outputs.",
                flush=True,
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
        if filter_runtime_cfg.get("mano_gpus") is not None:
            # physical GPU ids -> cuda:<n> indices under the inherited CUDA_VISIBLE_DEVICES
            mano_key = "filter.mano_gpus" if filter_cfg.get("mano_gpus") is not None else "build.mano_gpus"
            filter_runtime_cfg["mano_gpus"] = ",".join(
                _physical_to_logical_gpu_ids(filter_runtime_cfg["mano_gpus"], mano_key)
            )
        if (config.get("_meta") or {}).get("filter_stages_defaulted") and manifest_uses_only_native_features(
            active_manifest_path
        ):
            filter_runtime_cfg["stages"] = native_check_stages()
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
                *cli_args_from_mapping(filter_runtime_cfg, negative_bool_flags=FILTER_NEGATIVE_BOOL_FLAGS),
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
        # Same precedence as the infer stages: CLI > build.resume > the run's resolved resume.
        build_section_resume = build_runtime_cfg.pop("resume", None)
        build_resume = effective_resume if cli_resume is not None or build_section_resume is None else bool(build_section_resume)
        build_runtime_cfg.setdefault("feature_cache_dir", str(shared_feature_cache_dir))
        if build_runtime_cfg.get("mano_gpus") is not None:
            # physical GPU ids -> cuda:<n> indices under the inherited CUDA_VISIBLE_DEVICES
            build_runtime_cfg["mano_gpus"] = ",".join(
                _physical_to_logical_gpu_ids(build_runtime_cfg["mano_gpus"], "build.mano_gpus")
            )
        build_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "build" / "build_vla_from_manifest.py"),
            "--descriptor_manifest",
            str(active_manifest_path),
            "--output_dir",
            str(final_dataset_root),
            *cli_args_from_mapping(build_runtime_cfg, negative_bool_flags=BUILD_NEGATIVE_BOOL_FLAGS),
        ]
        if build_resume:
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
        validation_runtime_cfg = dict(validation_cfg)
        # validation.require_annotation (explicit) > build.require_annotation: missing annotations
        # only fail validate when build would also have dropped those clips.
        validate_require_annotation = validation_runtime_cfg.pop("require_annotation", None)
        if validate_require_annotation is None:
            validate_require_annotation = build_cfg.get("require_annotation", False)
        if validation_runtime_cfg.get("stages") is None and manifest_uses_only_native_features(active_manifest_path):
            validation_runtime_cfg["stages"] = native_check_stages()
        validate_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "validate_pipeline_run.py"),
            "--descriptor_manifest",
            str(active_manifest_path),
            "--dataset_dir",
            str(final_dataset_root),
            *cli_args_from_mapping(validation_runtime_cfg),
        ]
        if annotation_root:
            validate_cmd.extend(["--annotation_root", str(annotation_root)])
            if build_cfg.get("annotation_suffix"):
                validate_cmd.extend(["--annotation_suffix", str(build_cfg["annotation_suffix"])])
            validate_cmd.append("--require_annotation" if validate_require_annotation else "--no-require_annotation")
        if bool(validation_cfg.get("depth_action_consistency", False)) and not validation_cfg.get("depth_action_report_out"):
            validate_cmd.extend(["--depth_action_report_out", str(run_dir / "depth_action_consistency.json")])
        run_logged("validate", validate_cmd)

    if "lerobot" in stages:
        # Optional adapter stage: WebDataset shards -> LeRobot v3.0 (scripts/build/wds_to_lerobot.py).
        ensure_manifest_exists("lerobot", active_manifest_path)
        if not final_dataset_root.exists():
            raise RuntimeError(f"lerobot requires built WebDataset shards, none found at {final_dataset_root}. Run `--stages build` first.")
        # Same native-fps default as build, for runs where build is not selected (e.g. `--stages lerobot`).
        _apply_single_video_native_fps_defaults(config, manifest_path=active_manifest_path)
        lerobot_cmd, lerobot_root = build_lerobot_stage_command(
            python=hawor_python,
            project_root=PROJECT_ROOT,
            final_dataset_root=final_dataset_root,
            active_manifest_path=active_manifest_path,
            paths_cfg=paths_cfg,
            build_cfg=config.get("build", build_cfg),
            # CLI > lerobot.resume > top-level: an explicit --resume/--no-resume drops the section value
            lerobot_cfg=lerobot_cfg if cli_resume is None else {k: v for k, v in lerobot_cfg.items() if k != "resume"},
            resume=effective_resume,
        )
        run_logged("lerobot", lerobot_cmd)
        run_summary["lerobot_root"] = _resolved_path_string(lerobot_root)
        summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nRun complete: {run_dir}")


def main(argv: list[str] | None = None) -> None:
    from lib.pipeline.proc.logging_setup import configure_logging

    configure_logging()
    args = get_parser().parse_args(argv)
    run_pipeline(args)
