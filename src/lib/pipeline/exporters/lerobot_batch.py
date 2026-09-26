"""Batch driver: convert many WebDataset roots to LeRobot and verify each one.

A YAML file lists ``defaults`` (converter / verification settings shared by every job) and
``jobs`` (one per output dataset). For each job the driver runs, in order:

    convert -> release -> validate -> verify_wds -> compare_reference -> loader

(``release`` = licence bundle + README + tier consistency check, see lerobot_release.py) and writes machine-readable + markdown reports under ``<output_dir>/_verification/`` plus a batch
summary under ``report_dir``. Steps are idempotent: ``skip_done`` skips jobs whose last report was
green, ``--steps`` reruns a subset, and the converter itself honours ``resume``.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

from lib.pipeline.exporters.lerobot_export import ConvertConfig, _ffprobe_for, _pool_context, _worker_init, convert_wds_to_lerobot, rss_gb, validate_lerobot_dataset
from lib.pipeline.exporters.lerobot_release import check_release_consistency, release_spec_from_dict, write_release_bundle
from lib.pipeline.exporters.lerobot_verify import compare_with_reference, loader_check, verify_against_wds

logger = logging.getLogger(__name__)

STEPS = ("convert", "release", "validate", "verify_wds", "compare_reference", "loader")
VERIFICATION_DIR = "_verification"

DEFAULT_VERIFY = {
    "workers": 0,  # 0 = auto (cpu count)
    "random_frames": 200,
    "full_files": 1,
    "full_stride": 1,
    "numeric_episodes": None,  # None = every episode
    "overlay_episodes": 2,
    "min_psnr": 30.0,
    "seed": 0,
}
DEFAULT_LOADER = {"num_items": 64, "window_state": 16, "window_action": 32}


def load_batch_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg.get("jobs"), list) or not cfg["jobs"]:
        raise ValueError(f"{path}: 'jobs' must be a non-empty list")
    cfg.setdefault("defaults", {})
    return cfg


def _merge(base: dict, override: dict | None) -> dict:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def resolve_job(defaults: dict, job: dict) -> dict:
    """Merge defaults into one job and normalise paths/lists."""
    merged = _merge(defaults, {k: v for k, v in job.items() if k not in ("name",)})
    merged["name"] = job.get("name") or Path(str(job.get("output_dir", "job"))).name
    if "wds_dir" not in merged or "output_dir" not in merged:
        raise ValueError(f"job {merged['name']!r}: needs wds_dir and output_dir")
    wds = merged["wds_dir"]
    merged["wds_dir"] = [str(w) for w in (wds if isinstance(wds, list) else [wds])]
    merged["output_dir"] = str(merged["output_dir"])
    merged["converter"] = merged.get("converter") or {}
    merged["verify"] = _merge(DEFAULT_VERIFY, merged.get("verify"))
    merged["loader"] = _merge(DEFAULT_LOADER, merged.get("loader"))
    merged.setdefault("reference_root", None)
    merged.setdefault("steps", list(STEPS))
    merged.setdefault("descriptor_manifest", None)
    merged.setdefault("source_map", None)
    merged.setdefault("isolate_steps", True)
    merged["release"] = release_spec_from_dict(merged.get("release"))
    # the tier decides whether pixels leave: labels_only never exports video
    if merged["release"].tier == "labels_only":
        merged["converter"]["include_video"] = False
    # tasks[0] is the dataset name downstream (consumers read it back as dataset_name): default to the job name,
    # the shards' own dataset_name is often an internal build label
    conv = merged["converter"]
    if "task_name" not in conv and conv.get("task_source", "dataset_name") == "dataset_name":
        conv["task_name"] = merged["name"]
    return merged


def build_convert_config(job: dict) -> ConvertConfig:
    allowed = {f.name for f in fields(ConvertConfig)}
    kwargs = {}
    for key in ("fps", "workers", "ffmpeg", "resume", "descriptor_manifest", "source_map"):
        if key in job and job[key] is not None:
            kwargs[key] = job[key]
    if str(kwargs.get("workers", "")).lower() == "auto":
        kwargs["workers"] = 0
    for key, value in job["converter"].items():
        if key not in allowed:
            raise ValueError(f"job {job['name']!r}: unknown converter option {key!r} (allowed: {sorted(allowed)})")
        kwargs[key] = value
    if "split_order" in kwargs and isinstance(kwargs["split_order"], str):
        kwargs["split_order"] = tuple(s.strip() for s in kwargs["split_order"].split(",") if s.strip())
    elif "split_order" in kwargs:
        kwargs["split_order"] = tuple(kwargs["split_order"])
    cfg = ConvertConfig(**kwargs)
    cfg.validate()
    return cfg


def _write_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")


def _step_line(name: str, result: dict | None) -> str:
    if result is None:
        return f"| {name} | skipped | |"
    if result.get("skipped"):
        return f"| {name} | skipped | {result['skipped']} |"
    status = "PASS" if result.get("ok") else "FAIL"
    detail = "; ".join(result.get("failures") or result.get("problems") or [])[:400]
    if not detail and result.get("warnings"):
        detail = f"{len(result['warnings'])} warning(s)"
    return f"| {name} | {status} | {detail} |"


def render_job_report_md(report: dict) -> str:
    lines = [f"# {report['name']}", "", f"- output: `{report['output_dir']}`", f"- wds: `{', '.join(report['wds_dir'])}`", f"- started: {report['started']}  elapsed: {report['elapsed_s']:.0f}s", f"- overall: **{'PASS' if report['ok'] else 'FAIL'}**", "", "| step | status | detail |", "|---|---|---|"]
    for step in STEPS:
        lines.append(_step_line(step, report["steps"].get(step)))
    rel = (report["steps"].get("release") or {}).get("release")
    if rel:
        lines += ["", f"- release tier: **{rel.get('tier')}**, source licence: {rel.get('license')}, this dataset: {rel.get('derived_license')}, video: {'yes' if rel.get('has_video') else 'no'}"]
    conv = (report["steps"].get("convert") or {}).get("summary")
    if conv:
        lines += ["", "## conversion", "", "```", json.dumps(conv, indent=2, ensure_ascii=False), "```"]
    vw = (report["steps"].get("verify_wds") or {}).get("notes")
    if vw:
        lines += ["", "## verify_wds notes", "", "```", json.dumps(vw, indent=2, ensure_ascii=False, default=str), "```"]
    cr = report["steps"].get("compare_reference")
    if cr and cr.get("warnings"):
        lines += ["", "## compare_reference warnings", ""] + [f"- {w}" for w in cr["warnings"]]
    for step in STEPS:
        res = report["steps"].get(step)
        if res and (res.get("failures") or res.get("problems")):
            lines += ["", f"## {step} failures", ""] + [f"- {f}" for f in (res.get("failures") or res.get("problems"))]
    return "\n".join(lines) + "\n"




# ---- step bodies -------------------------------------------------------------------------
# Each step is a module-level function of (job, prior) so it can run in a fresh spawned process
# (``isolate_steps``, default on). One batch driver used to run every dataset's convert, verify
# and lerobot-loader pass in the same interpreter; after a few datasets the driver had grown to
# ~70 GB, and forking a worker pool from it tripped the memory cgroup. A step now leaves nothing
# behind in the driver except its JSON result.


def _step_convert(job: dict, prior: dict) -> dict:
    out_dir = Path(job["output_dir"])
    cfg = build_convert_config(job)
    if job.get("overwrite") and out_dir.exists():
        shutil.rmtree(out_dir)
    summary = convert_wds_to_lerobot(job["wds_dir"], str(out_dir), cfg)
    return {"ok": True, "summary": summary}


def _step_release(job: dict, prior: dict) -> dict:
    out_dir = Path(job["output_dir"])
    cfg = build_convert_config(job)
    spec = job["release"]
    has_sf = (out_dir / "meta" / "source_frames.parquet").exists()
    problems = check_release_consistency(spec, include_video=cfg.include_video, has_source_frames=has_sf)
    summary = (prior.get("convert") or {}).get("summary")
    payload = write_release_bundle(str(out_dir), spec, conversion_summary=summary)
    return {"ok": not problems, "failures": problems, "warnings": [], "release": payload}


def _step_validate(job: dict, prior: dict) -> dict:
    ffmpeg = build_convert_config(job).ffmpeg
    return validate_lerobot_dataset(job["output_dir"], ffprobe=_ffprobe_for(ffmpeg), ffmpeg=ffmpeg, try_lerobot=False)


def _step_verify_wds(job: dict, prior: dict) -> dict:
    v = job["verify"]
    vdir = Path(job["output_dir"]) / VERIFICATION_DIR
    return verify_against_wds(
        job["output_dir"],
        job["wds_dir"],
        ffmpeg=build_convert_config(job).ffmpeg,
        random_frames=int(v["random_frames"]),
        full_files=int(v["full_files"]),
        full_stride=int(v["full_stride"]),
        numeric_episodes=None if v["numeric_episodes"] in (None, "all") else int(v["numeric_episodes"]),
        overlay_episodes=int(v["overlay_episodes"]),
        overlay_dir=str(vdir / "overlays"),
        min_psnr=float(v["min_psnr"]),
        seed=int(v["seed"]),
        workers=0 if str(v.get("workers", 0)).lower() == "auto" else int(v.get("workers", 0)),
    )


def _step_compare_reference(job: dict, prior: dict) -> dict:
    ffmpeg = build_convert_config(job).ffmpeg
    return compare_with_reference(job["output_dir"], str(job["reference_root"]), ffprobe=_ffprobe_for(ffmpeg))


def _step_loader(job: dict, prior: dict) -> dict:
    ld = job["loader"]
    return loader_check(job["output_dir"], num_items=int(ld["num_items"]), window_state=int(ld["window_state"]), window_action=int(ld["window_action"]))


STEP_FUNCS = {
    "convert": _step_convert,
    "release": _step_release,
    "validate": _step_validate,
    "verify_wds": _step_verify_wds,
    "compare_reference": _step_compare_reference,
    "loader": _step_loader,
}


def _step_child(step: str, job: dict, prior: dict) -> dict:
    """Runs inside the spawned step process: never raises, so the driver only ever receives a dict."""
    try:
        return STEP_FUNCS[step](job, prior)
    except Exception as error:
        return {"ok": False, "failures": [f"{step} crashed: {error!r}"], "traceback": traceback.format_exc()}


def run_step(step: str, job: dict, prior: dict, *, isolate: bool = True) -> dict:
    """Run one step, in a fresh spawned process when ``isolate`` (the default) so memory never
    accumulates in the driver. A child killed by the kernel (OOM) surfaces as a failed step, not a
    dead batch."""
    if not isolate:
        return _step_child(step, job, prior)
    with ProcessPoolExecutor(max_workers=1, mp_context=_pool_context(), initializer=_worker_init) as pool:
        return pool.submit(_step_child, step, job, prior).result()


def run_job(job: dict, *, steps: list[str] | None = None, dry_run: bool = False) -> dict:
    steps = list(steps or job["steps"])
    out_dir = Path(job["output_dir"])
    vdir = out_dir / VERIFICATION_DIR
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()
    report = {"name": job["name"], "output_dir": str(out_dir), "wds_dir": job["wds_dir"], "started": started, "steps": {}, "ok": True}
    if dry_run:
        report["steps"] = {s: {"ok": True, "skipped": "dry run"} for s in steps}
        report["convert_config"] = asdict(build_convert_config(job))
        report["release"] = asdict(job["release"])
        report["release_problems"] = check_release_consistency(job["release"], include_video=report["convert_config"]["include_video"], has_source_frames=bool(report["convert_config"].get("descriptor_manifest")))
        report["ok"] = not report["release_problems"]
        report["elapsed_s"] = 0.0
        return report

    helper = bool(job.get("helper"))
    if helper:
        steps = ["convert"]  # a helper only converts file groups; the primary owns meta, reports and every later step

    def record(step: str, result: dict) -> bool:
        report["steps"][step] = result
        if not helper:
            _write_json(result, vdir / f"{step}.json")
        ok = bool(result.get("ok", False)) or bool(result.get("skipped"))
        if not ok:
            report["ok"] = False
        return ok

    isolate = bool(job.get("isolate_steps", True))

    def guarded(step: str) -> bool:
        try:
            result = run_step(step, job, report["steps"], isolate=isolate)
        except Exception as error:  # the step process itself died (e.g. OOM-killed) or could not start
            result = {"ok": False, "failures": [f"{step} process failed: {error!r}"], "traceback": traceback.format_exc()}
        logger.info("[%s] %s %s; driver rss %.1f GB", job["name"], step, "ok" if result.get("ok") or result.get("skipped") else "FAIL", rss_gb())
        return record(step, result)

    cfg = build_convert_config(job)
    proceed = True
    if "convert" in steps:
        proceed = guarded("convert")
    if proceed and "release" in steps:
        proceed = guarded("release")
    if proceed and "validate" in steps:
        proceed = guarded("validate")
    if proceed and "verify_wds" in steps:
        proceed = guarded("verify_wds")
    if proceed and "compare_reference" in steps:
        if job.get("reference_root"):
            guarded("compare_reference")
        else:
            record("compare_reference", {"ok": True, "skipped": "no reference_root configured"})
    if proceed and "loader" in steps:
        guarded("loader")

    report["elapsed_s"] = time.time() - t0
    report["convert_config"] = asdict(cfg)
    report["release"] = asdict(job["release"])
    if not helper:
        _write_json(report, vdir / "report.json")
        (vdir / "report.md").write_text(render_job_report_md(report), encoding="utf-8")
    return report


def previous_report_ok(job: dict) -> bool:
    path = Path(job["output_dir"]) / VERIFICATION_DIR / "report.json"
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("ok"))
    except json.JSONDecodeError:
        return False


def run_batch(config: dict, *, jobs_filter: list[str] | None = None, steps: list[str] | None = None, dry_run: bool = False, stop_on_fail: bool = False, skip_done: bool = False, report_dir: str | None = None, helper: bool = False) -> dict:
    defaults = config.get("defaults") or {}
    jobs = [resolve_job(defaults, j) for j in config["jobs"]]
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate job names: {names}")
    if jobs_filter:
        unknown = sorted(set(jobs_filter) - set(names))
        if unknown:
            raise ValueError(f"unknown job(s) {unknown}; available: {names}")
        jobs = [j for j in jobs if j["name"] in jobs_filter]
    if steps:
        unknown = sorted(set(steps) - set(STEPS))
        if unknown:
            raise ValueError(f"unknown step(s) {unknown}; available: {STEPS}")

    out_root = Path(report_dir or config.get("report_dir") or ".")
    out_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for job in jobs:
        if skip_done and previous_report_ok(job):
            logger.info("[%s] previous report is green, skipping", job["name"])
            reports.append({"name": job["name"], "output_dir": job["output_dir"], "ok": True, "skipped": "previous report ok", "steps": {}})
            continue
        if helper:
            # a second machine joining a conversion the primary already started: same plan, unclaimed file groups only
            job["helper"] = True
            job["resume"] = True
            if job.get("overwrite"):
                # overwrite would rmtree the directory the primary is writing into: only the primary may wipe it
                logger.warning("[%s] helper: ignoring overwrite=true (only the primary may clear %s)", job["name"], job["output_dir"])
            job["overwrite"] = False
            job["converter"]["helper"] = True
            if not (Path(job["output_dir"]) / "_wds_to_lerobot_work" / "plan.json").exists():
                # the work dir appears before the primary finishes indexing; only its plan.json makes a join safe
                logger.info("[%s] helper: the primary has not written plan.json for this job yet, skipping", job["name"])
                continue
        logger.info("[%s] start%s: %s -> %s", job["name"], " (helper)" if helper else "", job["wds_dir"], job["output_dir"])
        report = run_job(job, steps=steps, dry_run=dry_run)
        reports.append(report)
        logger.info("[%s] %s (%.0fs)", job["name"], "PASS" if report["ok"] else "FAIL", report.get("elapsed_s", 0.0))
        if stop_on_fail and not report["ok"]:
            logger.error("[%s] failed; stopping batch", job["name"])
            break

    summary = {"ok": all(r["ok"] for r in reports), "jobs": reports, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if helper:
        return summary  # the primary owns batch_report.*
    _write_json(summary, out_root / "batch_report.json")
    lines = ["# WebDataset -> LeRobot batch report", "", f"- finished: {summary['finished']}", f"- overall: **{'PASS' if summary['ok'] else 'FAIL'}**", "", "| job | status | " + " | ".join(STEPS) + " | output |", "|---|---|" + "---|" * len(STEPS) + "---|"]
    for r in reports:
        cells = []
        for s in STEPS:
            res = r.get("steps", {}).get(s)
            cells.append("-" if res is None else ("skip" if res.get("skipped") else ("PASS" if res.get("ok") else "FAIL")))
        status = "skip" if r.get("skipped") else ("PASS" if r["ok"] else "FAIL")
        lines.append(f"| {r['name']} | {status} | " + " | ".join(cells) + f" | `{r['output_dir']}` |")
    (out_root / "batch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


__all__ = ["STEPS", "build_convert_config", "load_batch_config", "resolve_job", "run_batch", "run_job"]
