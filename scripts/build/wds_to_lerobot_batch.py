#!/usr/bin/env python3
"""Batch-convert WebDataset roots to LeRobot v3.0 and verify every output.

    python scripts/build/wds_to_lerobot_batch.py --config configs/wds_to_lerobot_batch.example.yaml
    python scripts/build/wds_to_lerobot_batch.py --config cfg.yaml --jobs taco_v1 --steps verify_wds,compare_reference
    python scripts/build/wds_to_lerobot_batch.py --config cfg.yaml --dry_run

Per job the driver runs convert -> validate -> verify_wds -> compare_reference -> loader and writes
``<output_dir>/_verification/report.{json,md}``; the batch summary lands in ``report_dir``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# src-layout: first-party packages live under src/; scripts/ stays importable from root.
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def get_parser() -> argparse.ArgumentParser:
    from lib.pipeline.exporters.lerobot_batch import STEPS

    parser = argparse.ArgumentParser(description="Batch WebDataset -> LeRobot conversion with verification")
    parser.add_argument("--config", type=str, required=True, help="YAML with defaults + jobs (see configs/wds_to_lerobot_batch.example.yaml)")
    parser.add_argument("--jobs", type=str, default=None, help="Comma-separated job names to run (default: all)")
    parser.add_argument("--steps", type=str, default=None, help=f"Comma-separated subset of {','.join(STEPS)} (default: job's steps)")
    parser.add_argument("--report_dir", type=str, default=None, help="Where batch_report.{json,md} go (default: config report_dir or cwd)")
    parser.add_argument("--dry_run", action="store_true", help="Resolve jobs and print the effective configs, run nothing")
    parser.add_argument("--stop_on_fail", action="store_true")
    parser.add_argument("--skip_done", action="store_true", help="Skip jobs whose previous report.json is green")
    parser.add_argument("--helper", action="store_true", help="Run on a second machine against the same output dirs: converts file groups the primary has not claimed, never writes meta or reports. Requires the same code version and config as the primary.")
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser


def main() -> int:
    args = get_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    from lib.pipeline.exporters.lerobot_batch import load_batch_config, run_batch

    config = load_batch_config(args.config)
    summary = run_batch(
        config,
        jobs_filter=[s.strip() for s in args.jobs.split(",") if s.strip()] if args.jobs else None,
        steps=[s.strip() for s in args.steps.split(",") if s.strip()] if args.steps else None,
        dry_run=args.dry_run,
        stop_on_fail=args.stop_on_fail,
        skip_done=args.skip_done,
        report_dir=args.report_dir,
        helper=args.helper,
    )
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    else:
        for job in summary["jobs"]:
            status = "skip" if job.get("skipped") else ("PASS" if job["ok"] else "FAIL")
            print(f"{status:5s} {job['name']}: {job['output_dir']}")
        print(f"batch: {'PASS' if summary['ok'] else 'FAIL'}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
