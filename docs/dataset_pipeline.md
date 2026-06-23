# Dataset Pipeline

This release focuses on the **single-video** path: one input video becomes one set of trainable
WebDataset samples (image, lowdim, MANO, meta, and optional depth payloads).

```bash
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_single_video.example.yaml
```

Minimal config:

```yaml
video: /path/to/input.mp4
```

Optional output root:

```yaml
video: /path/to/input.mp4
output_root: /path/to/input.hawor_pipeline
```

If `output_root` is omitted, outputs go under `<video_dir>/<video_stem>.hawor_pipeline/`:

- `frames/`: extracted native-FPS RGB frames
- `stage_outputs/`: HaWoR / SLAM / infiller outputs
- `runs/run/`: logs, run state, reports
- `webdataset/`: final trainable WebDataset shards

Internally the pipeline is adapter-driven: a source dataset is normalized into a prepared clip
state, and then the same annotation, inference, filter, build, and validation logic runs on top of
that shared boundary. The built-in source adapters live in `src/lib/pipeline/datasets/`; this release
ships only the single-video example config (multi-dataset processing configs are not part of it).

## Official Stages

- `prepare`: source preprocessing plus prepared clip state creation
- `annotate`: clip-level language sidecars (optional — only when `annotation.command` is configured)
- `infer`: `detect_track`, `motion`, `slam`, and `infiller`
- `filter`: build-equivalent clip quality control before export
- `build`: final WebDataset export
- `validate`: source and dataset checks

Default stages are `prepare,infer,filter,build,validate`. `annotate` is inserted automatically only
when `annotation.command` is configured. `--stages` can be used for debugging or resume runs.

## Config Shape

First-party configs use the simplified single-video layout:

```yaml
video: /path/to/input.mp4
output_root: /optional/output_root

# Optional annotation hook. Without it, instruction/language fields are allowed to be empty.
annotation:
  command: >
    echo "Read {prepared_state} and write annotations to {annotation_root}"
```

Notes:

- The simplified `video:` config takes no runtime paths; stages run in the active `egosmith` env
  (see "Runtime" below).
- `annotation.command` receives `{prepared_state}`, `{active_prepared_state}`, `{annotation_root}`,
  `{run_dir}`, `{hawor_python}`, `{slam_python}`, and `{project_root}`.

See [configs/README.md](../configs/README.md) for the config inventory.

## Runtime (which Python runs the stages)

EgoSmith runs in a single conda env (`egosmith`). Stage subprocesses use the orchestrator's own
interpreter (`sys.executable`), so **activate the env first** (`conda activate egosmith`, or
`pip install -e .`). You can override the interpreter per runtime in the config
(`runtimes.hawor_python` / `runtimes.slam_python`) — this is mainly how the multihost path points
each remote host's stages at that host's env. Validate the setup from the repo root:

```bash
bash scripts/setup/validate_setup.sh
```

## Quality Control (the `filter` stage)

The `filter` stage applies multi-level quality control (`src/lib/pipeline/quality/quality_metrics.py`). Hard
rules are always enabled: any `NaN/Inf` lowdim frame, or any missing / empty / mismatched
instruction frame, drops the whole episode. On top of that:

- **Frame level** — per-frame motion caps: camera translation `≤ 0.20 m`, wrist/finger translation
  `≤ 0.30 m`, camera rotation `≈ 28°`, and wrist rotation `≈ 41°` (rotations are Frobenius-norm caps
  on consecutive rotation matrices).
- **Chunk level** — over a sliding window (≈ past 6 s + future 30 frames) the wrist is taken
  relative to the camera and finger joints relative to the wrist; coordinates outside the
  dataset-specific IQR fence (`Q1 − 2.5·IQR, Q3 + 2.5·IQR`) or beyond a `1.5 m` physical cap drop
  the episode.
- **Episode level** — per-episode mean camera translation / rotation compared to the dataset
  distribution via the same IQR fence.

In the 4D-motion stage, adjacent Any4D windows share `--any4d_overlap` frames (default `4`) so their
per-window metric scales are stitched into one consistent scale before the trajectory is anchored.

## Final Dataset Schema

Each sample is `*.image.jpg` + `*.lowdim.npy` (`float32[116]`) + `*.mano.npy` + `*.meta.json`
(plus an optional `*.depth.npy`). The full 116-d `lowdim` layout, coordinate conventions, and
`meta.json` fields are documented in **[dataset_format.md](dataset_format.md)**.

If stage outputs were generated at a lower FPS than the descriptor frames, set `build.source_fps`,
`build.target_fps`, and `build.interpolate_labels` to resample the stage outputs onto the descriptor
frame timeline during final build.

## Validation

Validate a completed run:

```bash
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_single_video.example.yaml \
  --stages validate
```

Or directly:

```bash
python scripts/inspection/validate_pipeline_run.py \
  --dataset_dir /path/to/final_dataset \
  --max_clips 200 \
  --dataset_sample_checks 20
```

Recommended smoke pass before a large run:

1. Run a small representative subset.
2. Complete `prepare` through `validate`.
3. Inspect multiple output samples across different shards.
4. Confirm image, lowdim, MANO, camera, and depth stay aligned.

## Inspect & visualize

**From a finished run** (lightest — no extra deps):

```bash
# overlay the reconstructed hands back onto the video, via direct K-projection
python scripts/inspection/overlay_hand_cam.py --seq_folder /path/to/output_root/.../<clip>
# inspect a batch run directory and print a report
python scripts/inspection/analyze_run.py /path/to/run_dir
```

**End-to-end single-video reconstruction + hand overlay** (`demo.py`). It runs detect → motion →
SLAM → infiller on one video and overlays the reconstructed hands back onto each frame with OpenCV
(direct pinhole projection — no OpenGL / viewer extras, works on any headless server; same projection
as `overlay_hand_cam.py`, but demo.py runs the pipeline first). It reads **pre-extracted frames**, so
extract them first:

```bash
python scripts/extract_frames.py --video_path /path/to/video.mp4
#   → writes <video_dir>/<stem>/extracted_images/  (what demo.py reads)

python demo.py --video_path /path/to/video.mp4
```

## Related Docs

- [README.md](../README.md)
- [docs/inputs.md](inputs.md) — supported inputs & adapters
- [docs/annotation.md](annotation.md) — language annotation
- [docs/running_at_scale.md](running_at_scale.md) — many videos / multi-GPU / multi-host
- [docs/dataset_format.md](dataset_format.md) — output schema
- [docs/repo_map.md](repo_map.md)
- [configs/README.md](../configs/README.md)
