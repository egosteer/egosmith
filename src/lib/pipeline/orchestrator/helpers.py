"""Utility helpers for orchestrator config parsing and subprocess execution."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def cli_args_from_mapping(mapping: dict | None, *, negative_bool_flags: set[str] | None = None) -> list[str]:
    args = []
    negative_bool_flags = negative_bool_flags or set()
    for key, value in (mapping or {}).items():
        flag = f"--{key}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(flag)
            elif key in negative_bool_flags:
                args.append(f"--no-{key}")
            continue
        if isinstance(value, list):
            args.append(flag)
            args.extend(str(item) for item in value)
            continue
        args.extend([flag, str(value)])
    return args


def parser_supported_option_dests(parser: argparse.ArgumentParser) -> set[str]:
    return {
        action.dest
        for action in parser._actions
        if action.option_strings and action.dest != "help"
    }


def validate_cli_mapping_keys(
    *,
    label: str,
    mapping: dict | None,
    supported_keys: set[str],
    reserved_keys: set[str] | None = None,
) -> list[str]:
    if not mapping:
        return []

    reserved_keys = reserved_keys or set()
    invalid = sorted(key for key in mapping if key not in supported_keys or key in reserved_keys)
    if not invalid:
        return []
    return [f"{label}: unsupported keys {invalid}"]


def format_annotation_command(template: str, context: dict) -> list[str]:
    return ["/bin/bash", "-lc", template.format(**context)]


def stream_command(
    name: str,
    cmd: list[str],
    log_path: Path,
    *,
    cwd: str | Path | None = None,
    env: dict | None = None,
    raise_on_error: bool = True,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + shlex.join(cmd) + "\n\n")
        log_handle.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        return_code = process.wait()
        if return_code != 0 and raise_on_error:
            raise RuntimeError(f"{name} failed with exit code {return_code}")
        return return_code
