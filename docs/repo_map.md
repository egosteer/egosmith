# Repo Map

A quick map of the public EgoSmith repository so you can find the official pipeline code,
the batch-inference path, and the visualization code.

## Top-Level Layout

- `src/` — first-party packages (src-layout):
  - `src/lib/` — the maintained library: pipeline stages, prepared clip state, filtering,
    build/export, viewer, and the per-video stage runners (`src/lib/stage_runners/`).
  - `src/infiller/`, `src/hawor/` — obtained HaWoR base, materialized here (see below).
- `scripts/` — runnable CLIs only (no library code). Pipeline entrypoints at the top level
  (`run_dataset_pipeline.py`, `batch_infer.py`, `extract_frames.py`); the rest grouped into
  `scripts/setup/` (env / weights / provisioning), `scripts/build/` (manifest / WebDataset build,
  downstream stages), and `scripts/inspection/` (validation / visualization). See `scripts/README.md`.
- `configs/` — example configs (see `configs/README.md`).
- `docs/` — documentation.
- `thirdparty/` — vendored `DPVO`, plus the `Any4D` and `hawor_upstream` git submodules. The
  HaWoR-authored base is **not** tracked; `scripts/setup/fetch_hawor_base.sh` materializes it into `src/`
  as unmodified symlinks into the `hawor_upstream` submodule (no patches). See
  [docs/hawor_provenance.md](hawor_provenance.md).
- `patches/` — local patches for vendored deps (currently `patches/chumpy/` only).
- `weights/`, `_DATA/` — model checkpoints and MANO assets, obtained on demand (see README).
- `example_video/` — small demo clip used by `demo.py`.

## Official Paths

Dataset pipeline (single video → trainable WebDataset):

- Entry: `scripts/run_dataset_pipeline.py`
- Config normalization: `src/lib/pipeline/proc/pipeline_config.py`
- Orchestration: `src/lib/pipeline/orchestrator/`
- Source adapters: `src/lib/pipeline/datasets/`
- Prepared clip state boundary: `src/lib/pipeline/clips/clip_manifest.py`
- Filtering / quality control: `src/lib/pipeline/filtering/`, `src/lib/pipeline/quality/quality_metrics.py`
- Final build/export: `src/lib/pipeline/exporters/`

Batch inference (run the HaWoR / SLAM / infiller stages over many videos):

- Entry: `scripts/batch_infer.py` (worker subprocess: `scripts/batch_worker.py`)
- CLI / runtime state: `src/lib/pipeline/batch/`
- Stage validation helpers: `src/lib/pipeline/proc/stage_api.py`
- Stage implementations: `src/lib/pipeline/stages/`
- Per-video stage runners (also used by the demos): `src/lib/stage_runners/`

Single-video inference + visualization:

- `demo.py` — end-to-end single-video reconstruction + OpenCV 3D render (headless-friendly).
- Viewer / visualization code: `src/lib/vis/`
- Dependency-light overlay: `scripts/inspection/overlay_hand_cam.py`

## Public Scripts

The maintained entrypoints:

- `scripts/run_dataset_pipeline.py` — the single-video dataset pipeline.
- `scripts/batch_infer.py` — multi-GPU batch inference.
- `scripts/extract_frames.py` — extract frames from a video.
- `scripts/build/build_vla_from_manifest.py` — build the final WebDataset from a frozen manifest.
- `scripts/inspection/validate_pipeline_run.py`, `scripts/inspection/check_motion_stage_outputs.py`,
  `scripts/build/filter_manifest_by_quality.py` — output validation / filtering.
- `scripts/inspection/overlay_hand_cam.py`, `scripts/inspection/analyze_run.py` — visualization / run inspection.
- `scripts/setup/setup_env.sh`, `scripts/setup/validate_setup.sh`, `scripts/setup/download_weights.sh`,
  `scripts/setup/fetch_hawor_base.sh`, `scripts/setup/fetch_chumpy.sh`, `scripts/setup/provision_worktree.sh` — setup.

## Internal Boundaries

Intended module boundaries:

- `src/lib/pipeline/orchestrator/` — config validation, stage selection, command construction, run execution.
- `src/lib/pipeline/filtering/` — manifest filter orchestration and report generation.
- `src/lib/pipeline/exporters/manifest_build/` — manifest preparation, feature loading/resampling,
  shard writing, final build runner.

## Related Docs

- [README.md](../README.md)
- [docs/install.md](install.md) — manual install + troubleshooting
- [docs/inputs.md](inputs.md) — supported inputs & dataset adapters
- [docs/annotation.md](annotation.md) — language annotation (API config)
- [docs/running_at_scale.md](running_at_scale.md) — many videos / multi-GPU / multi-host
- [docs/dataset_pipeline.md](dataset_pipeline.md) — full stage reference
- [docs/dataset_format.md](dataset_format.md) — output WebDataset schema
- [docs/hawor_provenance.md](hawor_provenance.md) — first-party vs obtained HaWoR base
- [configs/README.md](../configs/README.md)
