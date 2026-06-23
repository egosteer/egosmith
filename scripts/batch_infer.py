#!/usr/bin/env python3
"""Unified batch inference entrypoint for HaWoR."""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# src-layout: first-party packages live under src/; scripts/ stays importable from root.
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.batch.cli import (
    build_batch_infer_parser,
    load_batch_inputs,
    normalize_batch_infer_args,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

# Redirect process-level temp files only when HAWOR_BATCH_TMPDIR is set. Never
# fall back to the repo dir (fills the project disk / pollutes the worktree); an
# unset value leaves the system default (e.g. /tmp) in place. Stage3's large
# frame writes require an explicit tmp root and fail loudly if it is missing.
_env_tmp = os.environ.get("HAWOR_BATCH_TMPDIR")
if _env_tmp:
    SHARED_TMP_DIR = Path(_env_tmp).expanduser().resolve()
    SHARED_TMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(SHARED_TMP_DIR)
    os.environ["TEMP"] = str(SHARED_TMP_DIR)
    os.environ["TMP"] = str(SHARED_TMP_DIR)
    tempfile.tempdir = str(SHARED_TMP_DIR)


def get_parser():
    return build_batch_infer_parser()


def _resolve_run_dir(run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "batch_runs" / timestamp


def main(argv: list[str] | None = None):
    from lib.pipeline.proc.logging_setup import configure_logging

    configure_logging()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = get_parser()
    args = parser.parse_args(raw_argv)
    compatibility_notes = normalize_batch_infer_args(args, raw_argv=raw_argv)

    # Export workspace controls to the environment so in-process stage workers and
    # the path resolvers in lib.pipeline.io.workspace pick them up consistently.
    if getattr(args, "output_root", None):
        os.environ["HAWOR_OUTPUT_ROOT"] = str(Path(args.output_root).expanduser().resolve())
    if getattr(args, "legacy_seq_folder", False):
        os.environ["HAWOR_LEGACY_SEQ_FOLDER"] = "1"

    inputs = load_batch_inputs(args)

    # Preflight: validate weights / MANO / inputs / scratch root + disk / GPU once,
    # before any run dir is created or GPU work starts. Reports all problems at once.
    if os.environ.get("HAWOR_SKIP_PREFLIGHT", "").strip().lower() not in ("1", "true", "yes", "on"):
        from lib.pipeline.proc.preflight import collect_batch_weights, run_preflight

        stage_list = [s.strip() for s in str(args.stages).split(",") if s.strip()]
        # In descriptor-manifest mode the "video paths" are descriptor clip-ids, not
        # filesystem paths (the real inputs are the descriptors' frame sources,
        # produced by the prepare stage). Only validate raw paths for file/dir modes.
        preflight_video_paths = (
            inputs.video_paths if inputs.input_mode in ("video_list", "video_dir") else None
        )
        report = run_preflight(
            stages=stage_list,
            weights=collect_batch_weights(PROJECT_ROOT, args),
            video_paths=preflight_video_paths,
            args=args,
            gpus=args.gpus,
            project_root=PROJECT_ROOT,
        )
        if not report.ok:
            print(report.render(), file=sys.stderr)
            print(
                "\nAborting before GPU work. Set HAWOR_SKIP_PREFLIGHT=1 to bypass (not recommended).",
                file=sys.stderr,
            )
            return 2

    run_dir = _resolve_run_dir(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== Batch Inference Configuration ===")
    print(f"Input mode: {inputs.input_mode}")
    print(f"Input path: {inputs.input_path}")
    print(f"Total videos in source: {inputs.total_items}")
    print(f"Processing range: [{inputs.start_idx}, {inputs.end_idx})")
    print(f"Videos to process: {len(inputs.video_paths)}")
    print(f"GPUs: {args.gpus}")
    print(f"Stages: {args.stages}")
    print("Scheduler mode: wave")
    print(f"Worker slots per GPU: {args.workers_per_gpu}")
    print(f"Max stage retries: {args.max_stage_retries}")
    print(f"Chunk batch size (motion): {args.chunk_batch_size}")
    print(f"Render batch size (motion): {args.render_batch_size}")
    print(f"Detect batch size (detect_track): {args.detect_batch_size}")
    print(f"Detect I/O workers: {args.detect_io_workers}")
    print(f"Any4D batch size (slam): {args.any4d_batch_size}")
    print(f"SLAM backend: {args.slam_backend}")
    print(f"Depth backend (slam): {args.depth_backend or '(env HAWOR_DEPTH_BACKEND, default any4d)'}")
    print(f"Dense depth all frames (slam): {args.depth_predict_all_frames}")
    print(f"Resume: {args.resume}")
    print(f"Run directory: {run_dir}")
    if inputs.input_mode != "descriptor_manifest":
        print("Compatibility input mode is in use. `--descriptor_manifest` is the preferred infer input.")
    for note in compatibility_notes:
        print(f"Compatibility note: {note}")
    print()

    from lib.pipeline.batch import BatchRunConfig, BatchScheduler

    config = BatchRunConfig.from_args(
        args,
        video_paths=inputs.video_paths,
        descriptors=inputs.descriptors,
        run_dir=run_dir,
    )
    success = BatchScheduler(config).run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
