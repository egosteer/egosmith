"""Resolve the Python executable used to launch pipeline stage subprocesses.

EgoSmith runs in a single conda env (`egosmith`), so by default every stage runs with the
orchestrator's own interpreter (`sys.executable`) — activate the env (or `pip install -e .`) before
running. The only override is an explicit per-runtime python path in the config, which the multihost
path uses to point each remote host's stages at that host's env.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineRuntimes:
    hawor_python: str | None
    slam_python: str | None
    source: dict[str, str]


def _compatible_existing_python_path(raw_path: str | None, *, runtime_name: str) -> str | None:
    if raw_path is None:
        return None
    path_text = str(raw_path)
    if Path(path_text).exists():
        return path_text

    # Tolerate the legacy `any4` -> `any4d` env rename in configured paths.
    fixed_text = path_text.replace("/envs/any4/", "/envs/any4d/")
    if fixed_text != path_text and Path(fixed_text).exists():
        print(
            f"[runtime] {runtime_name} python not found at {path_text}; using compatible fallback {fixed_text}",
            flush=True,
        )
        return fixed_text
    return path_text


def resolve_pipeline_runtimes(runtimes_cfg: dict | None) -> PipelineRuntimes:
    """Return explicitly-configured stage Python paths, if any.

    A runtime is None when the config does not set it; the caller then falls back to
    `sys.executable` (the active env). The hawor/slam split is kept so a multihost config can point
    each stage at a different interpreter per host.
    """
    runtimes_cfg = dict(runtimes_cfg or {})
    source: dict[str, str] = {}

    hawor_python = _compatible_existing_python_path(runtimes_cfg.get("hawor_python"), runtime_name="hawor")
    if hawor_python:
        source["hawor"] = "config"

    slam_python = _compatible_existing_python_path(
        runtimes_cfg.get("slam_python") or runtimes_cfg.get("any4d_python"),
        runtime_name="any4d",
    )
    if slam_python:
        source["slam"] = "config"
    elif hawor_python:
        slam_python = hawor_python
        source["slam"] = source.get("hawor", "config")

    return PipelineRuntimes(
        hawor_python=hawor_python,
        slam_python=slam_python,
        source=source,
    )
