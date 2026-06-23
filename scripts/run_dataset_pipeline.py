#!/usr/bin/env python3
"""Thin CLI entrypoint for the official dataset pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# src-layout: first-party packages live under src/; scripts/ stays importable from root.
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.orchestrator.pipeline import main


if __name__ == "__main__":
    main()
