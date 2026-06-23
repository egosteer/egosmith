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
dataset:
  adapter: video_folder
paths:
  video_root: /data/raw/videos
  final_dataset_root: /data/out/webdataset
infer:
  common:
    gpus: "0,1,2,3"        # multi-GPU for the inference sub-stage
annotation:                 # optional
  command: >
    {hawor_python} {project_root}/src/lib/annotation/api_annotation.py
    --prepared_state {prepared_state} --annotation_root {annotation_root}
    --annotation_suffix _qwen-annotation.json
```
```bash
export DASHSCOPE_API_KEY=sk-...           # only if annotating
python scripts/run_dataset_pipeline.py --config config_folder.yaml
```

Run a subset of stages with `--stages` (e.g. re-run only the build):
```bash
python scripts/run_dataset_pipeline.py --config config_folder.yaml --stages build,validate
```

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

Per-video intermediates are written next to each video (or under `--output_root`). Feed the same
clips back through the orchestrator's `filter,build,validate` stages to produce the final dataset.

## GPU selection

- `batch_infer.py --gpus 0,1,2,3` (default `0`).
- Orchestrator: `infer.common.gpus: "0,1,2,3"` in the config.
- `CUDA_VISIBLE_DEVICES` is honored as usual; each worker pins one GPU.

## Multiple hosts

The orchestrator supports a stage-queue across machines via an `infer.multihost` block:

```yaml
infer:
  multihost:
    enabled: true
    mode: stage_queue
    hosts:
      - { name: host-01, ssh_target: user@host1, gpus: "0,1,2,3",
          project_root: /path/repo, hawor_python: /env/hawor/bin/python, slam_python: /env/any4d/bin/python }
      - { name: host-02, ssh_target: user@host2, gpus: "0,1,2,3",
          project_root: /path/repo, hawor_python: /env/hawor/bin/python, slam_python: /env/any4d/bin/python }
```

Each host needs the repo, the conda envs, and the obtained HaWoR base / weights provisioned.

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
