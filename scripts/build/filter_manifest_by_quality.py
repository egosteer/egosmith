#!/usr/bin/env python3
"""Thin CLI entrypoint for manifest quality filtering."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.filtering.manifest_filter import build_parser, main, run_filter

__all__ = ["build_parser", "main", "run_filter"]


if __name__ == "__main__":
    main()
