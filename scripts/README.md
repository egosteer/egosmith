# scripts/

Runnable entrypoints for EgoSmith — CLIs only; first-party library code lives under `src/lib/`.
Run from the repo root (`python scripts/<...>.py`); each bootstraps `src/` onto `sys.path`, or do
`pip install -e .` once. Layout: the main pipeline entrypoints sit at the top level; everything else
is grouped into `setup/`, `build/`, and `inspection/`.

## Top level — pipeline entrypoints

| Script | What it does |
|---|---|
| `run_dataset_pipeline.py` | The official single-video → WebDataset pipeline (`--config`). Start here. |
| `batch_infer.py` | Multi-GPU batch inference over a `--video_list` (detect_track / motion / slam / infiller). |
| `batch_worker.py` | Per-GPU worker subprocess spawned by the batch path (not run directly). |
| `extract_frames.py` | Extract frames from a video (or `--video_list`) to JPGs. |

## `setup/` — environment & provisioning

| Script | What it does |
|---|---|
| `setup/setup_env.sh` | Build the fixed `egosmith` conda env (CUDA 12.8, Torch 2.8, DPVO build). Idempotent. |
| `setup/fetch_hawor_base.sh` | Materialize the HaWoR base from the `hawor_upstream` submodule into `src/` (symlinks). |
| `setup/fetch_chumpy.sh` | Materialize the patched `chumpy` install copy (needed for MANO's legacy pickles). |
| `setup/download_weights.sh` | Fetch every model checkpoint to its default path. Idempotent. |
| `setup/provision_worktree.sh` | Symlink gitignored runtime deps (weights / MANO / checkpoints) from another checkout (`HAWOR_SRC=...`). |
| `setup/validate_setup.sh` | Verify the env, weights, and imports before any GPU work. |

## `build/` — manifest / WebDataset build, filter, downstream stages

| Script | What it does |
|---|---|
| `build/build_vla_from_manifest.py` | Build the final VLA WebDataset from a frozen clip manifest. |
| `build/filter_manifest_by_quality.py` | Quality-filter a clip manifest. |
| `build/run_hot3d_native_depth.py` | HOT3D native Any4D depth-only inference (HOT3D adapter). |
| `build/generate_fpha_world_res.py` | Generate `world_space_res.pth` for FPHA from right-hand skeleton GT (FPHA stage). |

## `inspection/` — validation, visualization, run inspection

| Script | What it does |
|---|---|
| `inspection/validate_pipeline_run.py` | Validate a whole-pipeline run (manifest, stage outputs, annotations, shards). |
| `inspection/check_motion_stage_outputs.py` | Diagnose one clip across cam-space / world-space / WebDataset lowdim. |
| `inspection/overlay_hand_cam.py` | Overlay the reconstructed hands onto the video via direct K-projection (no aitviewer). |
| `inspection/analyze_run.py` | Inspect a batch run directory and print a report. |

See the repo [README](../README.md) and [docs/dataset_pipeline.md](../docs/dataset_pipeline.md) for
end-to-end usage, and [docs/running_at_scale.md](../docs/running_at_scale.md) /
[docs/inputs.md](../docs/inputs.md) / [docs/annotation.md](../docs/annotation.md) for scaling,
inputs, and annotation.
