"""Startup preflight validation for HaWoR runs.

The pipeline is expensive: it allocates GPUs and runs stages that each take
minutes. Historically, misconfigurations surfaced *mid-run* -- a missing
checkpoint produced a silently random-initialized backbone, an unset stage-3
scratch root raised only once the SLAM stage started (after stages 1-2 already
ran), a full disk failed deep in frame materialization. This module front-loads
all of those checks so a single startup call reports **every** problem at once
and the run aborts before any GPU work.

Design:

* Each check appends to a :class:`PreflightReport` and never raises -- so one
  missing file doesn't hide the next problem.
* ``torch`` is imported lazily inside the GPU check so this module (and its
  tests) import without a CUDA/torch install.
* The caller decides what to do with a failing report (typically print
  ``report.render()`` and ``sys.exit(2)``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from lib.pipeline.io.workspace import resolve_tmp_root


DEFAULT_MIN_FREE_GB = 15.0

# Stages that need a GPU / the inference runtime.
_GPU_STAGES = {"detect_track", "detect_motion", "motion", "slam", "infiller"}
# Stages that materialize many frames into the stage-3 scratch root.
_TMP_STAGES = {"slam", "native_depth"}


@dataclass
class PreflightProblem:
    category: str
    detail: str
    fix_hint: str = ""


@dataclass
class PreflightReport:
    problems: List[PreflightProblem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def add(self, category: str, detail: str, fix_hint: str = "") -> None:
        self.problems.append(PreflightProblem(category, detail, fix_hint))

    def render(self) -> str:
        if self.ok:
            return "Preflight checks passed."
        lines = ["Preflight failed with the following problem(s):", ""]
        by_category: dict[str, List[PreflightProblem]] = {}
        for problem in self.problems:
            by_category.setdefault(problem.category, []).append(problem)
        for category, problems in by_category.items():
            lines.append(f"[{category}]")
            for problem in problems:
                lines.append(f"  - {problem.detail}")
                if problem.fix_hint:
                    lines.append(f"      fix: {problem.fix_hint}")
            lines.append("")
        return "\n".join(lines).rstrip()


def _needs(stages: Optional[Iterable[str]], wanted: set) -> bool:
    if stages is None:
        return True
    return bool(set(stages) & wanted)


def check_weights(report: PreflightReport, weights: dict) -> None:
    """``weights`` maps a human label -> filesystem path (str/Path)."""
    for label, path in weights.items():
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            report.add(
                "weights",
                f"{label} not found: {p}",
                "download the checkpoint (see README 'Weights') or fix the path",
            )
        elif p.is_file() and p.stat().st_size == 0:
            report.add("weights", f"{label} is empty (0 bytes): {p}", "re-download the checkpoint")


def check_mano(report: PreflightReport, project_root: Path) -> None:
    mano_files = [
        project_root / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl",
        project_root / "_DATA" / "data_left" / "mano_left" / "MANO_LEFT.pkl",
    ]
    for path in mano_files:
        if not path.exists():
            report.add(
                "mano",
                f"MANO asset missing: {path}",
                "download MANO from the official site and place it at this path (see README)",
            )


def check_inputs(report: PreflightReport, video_paths: Sequence[str], *, sample: Optional[int] = None) -> None:
    """Validate that input videos/dirs exist and are readable.

    ``sample`` caps how many paths are checked (for very large multi-clip runs,
    validating roots + a sample is enough); ``None`` checks them all.
    """
    paths = list(video_paths)
    to_check = paths if sample is None else paths[:sample]
    for raw in to_check:
        p = Path(raw)
        if not p.exists():
            report.add("inputs", f"input path does not exist: {p}", "fix the path in the video list / config")
        elif not os.access(p, os.R_OK):
            report.add("inputs", f"input path is not readable: {p}", "fix file permissions")


def check_tmp_root(
    report: PreflightReport,
    *,
    args=None,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
) -> None:
    """Resolve + validate the stage-3 scratch root and its free space."""
    try:
        tmp_root = resolve_tmp_root(args, required=True)
    except (ValueError, PermissionError) as error:
        report.add("tmp_root", str(error), "set --stage3_tmp_root / $HAWOR_STAGE3_TMP_ROOT / $HAWOR_BATCH_TMPDIR")
        return
    try:
        free_gb = shutil.disk_usage(tmp_root).free / (1024 ** 3)
    except OSError as error:
        report.add("tmp_root", f"cannot stat free space at {tmp_root}: {error}")
        return
    if free_gb < min_free_gb:
        report.add(
            "tmp_root",
            f"low free disk at scratch root {tmp_root}: {free_gb:.1f} GB < {min_free_gb:.0f} GB required",
            "point the scratch root at a larger disk or free space",
        )


def _parse_gpu_indices(gpus) -> List[int]:
    if gpus is None:
        return []
    if isinstance(gpus, (list, tuple)):
        tokens = [str(g) for g in gpus]
    else:
        tokens = [t.strip() for t in str(gpus).split(",")]
    indices = []
    for token in tokens:
        if token in ("", "-1", "none", "None", "void", "cpu"):
            continue
        try:
            indices.append(int(token))
        except ValueError:
            continue
    return indices


_GPU_PROBE = (
    "import torch;"
    "a=torch.cuda.is_available();"
    "print(int(a), torch.cuda.device_count() if a else 0)"
)


def _query_cuda(timeout: float = 60.0):
    """Return (available, device_count) by probing torch.cuda in a SUBPROCESS.

    Crucially this never initializes a CUDA context in the current process: doing
    so before the worker pool forks would break the workers with "Cannot
    re-initialize CUDA in forked subprocess". Returns None on probe failure.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-c", _GPU_PROBE],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    parts = out.stdout.split()
    if len(parts) < 2:
        return None
    try:
        return (parts[0] == "1", int(parts[1]))
    except ValueError:
        return None


def check_gpu(report: PreflightReport, gpus) -> None:
    indices = _parse_gpu_indices(gpus)
    probe = _query_cuda()
    if probe is None:
        report.add(
            "gpu",
            "could not verify CUDA via torch (import/query failed)",
            "ensure a CUDA-enabled torch is installed (see README)",
        )
        return
    available, device_count = probe
    if not available:
        report.add(
            "gpu",
            "CUDA is not available (torch.cuda.is_available() is False)",
            "check drivers / CUDA_VISIBLE_DEVICES",
        )
        return
    for idx in indices:
        if idx >= device_count:
            report.add(
                "gpu",
                f"requested GPU index {idx} but only {device_count} CUDA device(s) visible",
                "fix --gpus or CUDA_VISIBLE_DEVICES",
            )


def check_runtimes(report: PreflightReport, runtimes: Optional[dict]) -> None:
    """``runtimes`` maps label -> python executable path (orchestrated path)."""
    if not runtimes:
        return
    for label, exe in runtimes.items():
        if not exe:
            continue
        p = Path(exe)
        if not p.exists():
            report.add("runtime", f"{label} interpreter not found: {p}", "fix runtimes.* in the config or the conda env")
        elif not os.access(p, os.X_OK):
            report.add("runtime", f"{label} interpreter not executable: {p}", "chmod +x or fix the path")


def run_preflight(
    *,
    stages: Optional[Sequence[str]] = None,
    weights: Optional[dict] = None,
    video_paths: Optional[Sequence[str]] = None,
    args=None,
    gpus=None,
    runtimes: Optional[dict] = None,
    require_mano: bool = True,
    project_root: Optional[Path] = None,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    input_sample: Optional[int] = None,
    tmp_root_required: Optional[bool] = None,
) -> PreflightReport:
    """Run all applicable checks and return a single aggregated report.

    Checks are gated on the selected ``stages`` so a stage-less / non-GPU run
    isn't blocked on a GPU or scratch root it never uses.
    """
    report = PreflightReport()
    project_root = Path(project_root) if project_root else Path.cwd()

    if weights:
        check_weights(report, weights)
    if require_mano and _needs(stages, {"motion", "infiller"}):
        check_mano(report, project_root)
    if video_paths:
        check_inputs(report, video_paths, sample=input_sample)

    needs_tmp = tmp_root_required if tmp_root_required is not None else _needs(stages, _TMP_STAGES)
    if needs_tmp:
        check_tmp_root(report, args=args, min_free_gb=min_free_gb)

    if _needs(stages, _GPU_STAGES):
        check_gpu(report, gpus)

    check_runtimes(report, runtimes)
    return report


def collect_batch_weights(project_root: Path, args) -> dict:
    """Assemble the weight->path map for a batch_infer run from its args/stages."""
    stages = []
    raw_stages = getattr(args, "stages", "") or ""
    if isinstance(raw_stages, str):
        stages = [s.strip() for s in raw_stages.split(",") if s.strip()]
    else:
        stages = list(raw_stages)

    weights: dict = {}
    if "detect_track" in stages:
        weights["detector"] = project_root / "weights" / "external" / "detector.pt"
    if {"motion", "infiller"} & set(stages):
        weights["hawor checkpoint"] = getattr(args, "checkpoint", None)
        weights["hawor model_config"] = project_root / "weights" / "hawor" / "model_config.yaml"
    if "infiller" in stages:
        weights["infiller weight"] = getattr(args, "infiller_weight", None)
    if "slam" in stages:
        # DPVO + Any4D are the only slam/depth backends; check both upfront so a
        # missing checkpoint is reported before any GPU work (not deep in slam).
        weights["dpvo checkpoint"] = project_root / "thirdparty" / "DPVO" / "models" / "dpvo.pth"
        weights["any4d checkpoint"] = _resolve_any4d_checkpoint_for_preflight(project_root, args)
    return weights


def _resolve_any4d_checkpoint_for_preflight(project_root: Path, args) -> Path:
    """Resolve the Any4D checkpoint the slam stage will load, for the preflight check.

    Mirrors :func:`lib.pipeline.slam.any4d_depth.resolve_any4d_paths` (lazy-imported so
    this module stays importable without torch/cv2), with a stdlib-only fallback to
    the same default path so preflight checks exactly what the worker will use.
    """
    repo_root = getattr(args, "any4d_repo_root", None)
    checkpoint = getattr(args, "any4d_checkpoint_path", None)
    try:
        from lib.pipeline.slam.any4d_depth import resolve_any4d_paths

        _, resolved_ckpt, _, _ = resolve_any4d_paths(
            project_root=project_root, any4d_repo_root=repo_root, checkpoint_path=checkpoint
        )
        return Path(resolved_ckpt)
    except Exception:
        raw_ckpt = checkpoint or os.environ.get("HAWOR_ANY4D_CHECKPOINT_PATH")
        if raw_ckpt:
            p = Path(raw_ckpt).expanduser()
            return p if p.is_absolute() else (project_root / p)
        raw_root = repo_root or os.environ.get("HAWOR_ANY4D_REPO_ROOT")
        root = Path(raw_root).expanduser() if raw_root else (project_root / "thirdparty" / "Any4D")
        if not root.is_absolute():
            root = project_root / root
        return root / "checkpoints" / "any4d_4v_combined.pth"
