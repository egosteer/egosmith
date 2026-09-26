# Running over a whole dataset

The single-video quickstart and the full-dataset workflow use the same stages — the difference is
how you point the pipeline at many clips and how you spread the heavy GPU work across devices.

## Two entrypoints

| Tool | What it does | Use it when |
|---|---|---|
| `scripts/run_dataset_pipeline.py` | the **full** config-driven pipeline (prepare → annotate → infer → filter → build → validate) for one config | you want a finished, trainable WebDataset — single video or a whole source via an adapter |
| `scripts/batch_infer.py` | **only the GPU inference stages** (detect_track / motion / slam / infiller) over many videos, multi-GPU | you want to scale just the heavy inference, then build separately |

## Whole dataset via the orchestrator (recommended)

Point an [adapter](inputs.md) at your source and run the full pipeline once. The orchestrator walks
every clip and its internal inference sub-stage already parallelizes across the GPUs you give it:

```yaml
# config_folder.yaml
run_tag: my_dataset_v1      # fixed run directory (<paths.log_root>/my_dataset_v1); needed to re-run later stages
dataset:
  adapter: video_folder
paths:
  video_root: /data/raw/videos
  final_dataset_root: /data/out/webdataset
  annotation_root: /data/out/annotations   # only needed with the annotation block
adapter_config:
  extract_frames: true      # raw .mp4 folder: extract frames during prepare
infer:
  common:
    gpus: "0,1,2,3"        # physical GPU ids for the inference sub-stage
annotation:                 # optional
  command: >
    {hawor_python} {project_root}/src/lib/annotation/api_annotation.py
    --prepared_state {prepared_state} --annotation_root {annotation_root}
    --annotation_suffix _qwen-annotation.json
    --prompt_file {project_root}/src/lib/annotation/prompts/without_clip/annotation_general_egocentric.txt
build:
  annotation_suffix: _qwen-annotation.json   # must match the suffix the annotation command writes
```
```bash
export HAWOR_BATCH_TMPDIR=/large/disk/tmp   # required: stage-3 scratch root (or HAWOR_STAGE3_TMP_ROOT)
export DASHSCOPE_API_KEY=sk-...           # only if annotating
python scripts/run_dataset_pipeline.py --config config_folder.yaml
```

Run a subset of stages with `--stages` (e.g. re-run only the build):
```bash
python scripts/run_dataset_pipeline.py --config config_folder.yaml --stages build,validate
```

Downstream stages read the prepared clip state from the run directory
`<paths.log_root>/<run_tag>/` (default `log_root`: `<repo>/pipeline_runs`). In a nested
(`dataset:` / `paths:`) config without `run_tag`, every invocation gets a new
`<hostname>_<timestamp>` directory, so a `--stages build,validate` re-run finds no prepared state and
aborts. Set a fixed `run_tag` as above, or pass the earlier run's tag with `--run_tag <tag>`.
(Single-video `video:` configs always use `run_tag: run`.)

## Standalone multi-GPU inference (`batch_infer.py`)

When you only want to run the inference stages over many videos (e.g. frames already prepared), use
`batch_infer.py`. Pick exactly one input source:

```bash
# preferred: a frozen clip manifest produced by the prepare stage
python scripts/batch_infer.py --descriptor_manifest run_dir/clip_manifest.jsonl \
  --gpus 0,1,2,3 --stages detect_track,motion,slam,infiller

# compatibility: a text file with one video path per line
python scripts/batch_infer.py --video_list videos.txt --gpus 0,1,2,3

# compatibility: a directory searched recursively for videos
python scripts/batch_infer.py --video_dir /data/raw/videos --gpus 0,1
```

You can also split stages across runs (e.g. give SLAM more GPUs):
```bash
python scripts/batch_infer.py --video_list videos.txt --gpus 0,1   --stages detect_track,motion
python scripts/batch_infer.py --video_list videos.txt --gpus 0,1,2,3 --stages slam
python scripts/batch_infer.py --video_list videos.txt --gpus 0,1   --stages infiller
```

Per-video intermediates go to a sibling `<stem>.hawor_pipeline/stage_outputs/<stem>/` next to each
video, or to `<output_root>/<stem>.hawor_pipeline/stage_outputs/<stem>/` with `--output_root`
(`--legacy-seq-folder` restores the old `<video_dir>/<stem>/` layout). `--output_root` works by
exporting `HAWOR_OUTPUT_ROOT` (and `--legacy-seq-folder` by exporting `HAWOR_LEGACY_SEQ_FOLDER=1`);
setting either variable yourself has the same effect. Under a shared `--output_root` /
`HAWOR_OUTPUT_ROOT` the folder is keyed by the file stem only, so two videos with the same stem in
different directories share one seq folder and the second reuses or overwrites the first's
outputs — give them unique stems first. With `--descriptor_manifest`, a `seq_folder` recorded for a
clip in the manifest is used verbatim and `--output_root` has no effect on that clip.

Only the `--descriptor_manifest` input feeds back into the orchestrator: run `batch_infer.py` on the
run's `clip_manifest.jsonl` (written by `prepare`), then run the orchestrator's `filter,build,validate`
stages with the same config and `run_tag` (or `--descriptor_manifest` pointing at that manifest), so they
read the `seq_folder`s the inference just wrote. Outputs from `--video_list` / `--video_dir` land in the
per-video folders above, which the orchestrator's `prepare` does not record (adapters assign their own
`seq_folder`s), so the orchestrator will not pick them up; use those inputs for standalone inference only.
For the feed-back path run `batch_infer.py` with `--keep_intermediates all`: the default (`none`)
deletes the `tracks_*` folders and stage markers after the infiller stage, and `filter` then drops
every clip with `missing_track_range`. (The orchestrator's own infer stages already pass
`--keep_intermediates all`.)

The `slam` stage caches its expensive intermediates under `SLAM/` (`dpvo_raw_*.npz`,
`any4d_depth_dpvo_*.npz`, `dense_depth_any4d_*.npz`). Each cache records a digest of what produced it:
the DPVO config, overrides and checkpoint, calibration, frame range and hand masks for `dpvo_raw`; the
Any4D checkpoint, resolution, AMP, batch size, overlap, task (and DPVO poses for a pose-conditioned
task) for the depth caches; plus the stitch / hand-anchor switches and their inputs for
`dense_depth_any4d`. A cache is reused only when its digest matches the current run, so changing any of
these recomputes it. A cache that does not match, and every cache under `--no-resume`, is renamed to
`<name>.stale` (never deleted; remove these by hand, or let `--keep_intermediates slam|none` drop them)
before it is recomputed. Caches written before digests were recorded have none, so the first SLAM run
after upgrading recomputes them once. `HAWOR_DPVO_FORCE_RERUN=1` / `HAWOR_ANY4D_FORCE_RERUN=1` still
delete the DPVO / depth caches outright before rerunning.

DPVO samples its patches at random, and on some clips a run diverges to non-finite camera poses. Each
DPVO run is therefore seeded (seed 0, so reruns are reproducible) and a diverged run is retried with the
next seed, up to `HAWOR_DPVO_ATTEMPTS` attempts (default 3). If every attempt diverges the SLAM stage fails
for that clip with `DPVO diverged`, and no `dpvo_raw` cache is written.

## GPU selection

- `batch_infer.py --gpus 0,1,2,3` (default `0`).
- Orchestrator: `infer.common.gpus: "0,1,2,3"` in the config.
- GPU ids are **physical** ids: each worker pins one GPU by exporting `CUDA_VISIBLE_DEVICES=<id>`.
  If the job already runs under a restricted `CUDA_VISIBLE_DEVICES` (e.g. a scheduler assigned
  GPUs `2,3`), pass ids from that list (`--gpus 2,3`), not the logical indices `0,1`; the preflight
  rejects ids outside the visible list. Single-video configs without `infer.common.gpus` default to
  the ids listed in `CUDA_VISIBLE_DEVICES`. Stages that address GPUs as `cuda:<n>` devices instead
  (`native_depth`, the FPHA skeleton infiller) receive the same physical ids translated to their
  index in that list by the orchestrator.
- `build.mano_gpus` (MANO workers of the `build` stage, and of `filter` unless `filter.mano_gpus`
  is set) also takes **physical** GPU ids, e.g. `mano_gpus: "2,3"` under
  `CUDA_VISIBLE_DEVICES=2,3`; the orchestrator translates them to `cuda:<n>` indices. (Calling
  `build_vla_from_manifest.py` / `filter_manifest_by_quality.py --mano_gpus` directly takes the
  `cuda:<n>` indices of the current visibility, as those scripts do no translation.)

## Multiple hosts

The orchestrator supports a stage-queue across machines via an `infer.multihost` block:

```yaml
infer:
  multihost:
    enabled: true
    mode: stage_queue
    hosts:
      - { name: host-01, ssh_target: user@host1, gpus: "0,1,2,3",
          project_root: /path/repo, hawor_python: /env/egosmith/bin/python, slam_python: /env/egosmith/bin/python }
      - { name: host-02, ssh_target: user@host2, gpus: "0,1,2,3",
          project_root: /path/repo, hawor_python: /env/egosmith/bin/python, slam_python: /env/egosmith/bin/python }
```

EgoSmith uses a single conda env (see [install.md](install.md)), so `hawor_python` and `slam_python`
both point at that host's `egosmith` env Python; the two keys exist only so the HaWoR and Any4D
subprocesses could be split across envs. Each host needs the repo, the `egosmith` env, and the obtained
HaWoR base / weights provisioned.

## Stage order & products

User-facing order: `prepare → annotate → infer → filter → build → validate`
(`prepare` = preprocess + manifest; `infer` = detect_track + motion + slam + [native_depth] + infiller).

| Stage | Produces |
|---|---|
| prepare | extracted frames + a prepared clip-state (manifest JSONL) |
| annotate | language sidecars (optional) — see [annotation.md](annotation.md) |
| infer | per-clip world-space hands, camera trajectory, metric depth |
| filter | a quality-filtered manifest |
| build | the final WebDataset shards — see [dataset_format.md](dataset_format.md) |
| validate | source / output integrity checks |

## Related

- [inputs.md](inputs.md) — adapters & supported inputs
- [dataset_pipeline.md](dataset_pipeline.md) — full stage reference
- [dataset_format.md](dataset_format.md) — output schema
