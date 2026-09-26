# Dataset Pipeline

This release focuses on the **single-video** path: one input video becomes one set of trainable
WebDataset samples (image, lowdim, MANO, meta, and optional depth payloads).

```bash
export HAWOR_BATCH_TMPDIR=/large/disk/tmp   # required: stage-3 scratch root (or HAWOR_STAGE3_TMP_ROOT)
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_single_video.example.yaml
```

The `slam` stage needs a large-capacity scratch root and never defaults one: set
`HAWOR_STAGE3_TMP_ROOT` or `HAWOR_BATCH_TMPDIR` (or `infer.slam.stage3_tmp_root` /
`infer.common.stage3_tmp_root` in the config). Without it the preflight aborts the run before frame
extraction. The preflight runs after the optional `clip` step of `prepare`, so a configured
`clip.mode` (including paid API clipping) runs before these checks.

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
- `lerobot` (optional, opt-in): convert the built WebDataset into a LeRobot v3.0 dataset
  (`scripts/build/wds_to_lerobot.py`; see [dataset_format.md](dataset_format.md))

Default stages are `prepare,infer,filter,build,validate`. `annotate` is inserted automatically only
when `annotation.command` is configured; `lerobot` never is — add it explicitly
(`--stages build,validate,lerobot`, or `--stages lerobot` on a finished run). `--stages` can be used
for debugging or resume runs. When `filter` is not selected but the run directory already holds a
`clip_manifest.filtered.jsonl` that is not older than the prepared state (and no earlier stage runs
in the same invocation), the downstream stages use that filtered state, so clips the filter dropped
are not exported again. If prepare or infer was rerun after the last filter, the old filtered state
is not reused (a warning says so) and the unfiltered state is exported: add `filter` to `--stages`.

When `prepare` runs a `clip.mode`, the redirect to the clipped videos (adapter, clip/frame/annotation
roots) is saved to `<run_dir>/clip_redirect.json`; later invocations without `prepare` restore it, so
`--stages build,validate` still reads the clip annotations. Changing the `clip` block or the source
adapter settings afterwards is an error until `prepare` runs again.

For native-feature sources (e.g. `hot3d_wds`), `filter.stages` / `validation.stages` default to
`native_features` (plus `native_depth` when `infer.native_depth.enabled`) unless set explicitly.

## Config Shape

First-party configs use the simplified single-video layout:

```yaml
video: /path/to/input.mp4
output_root: /optional/output_root

# Optional annotation hook. Instruction/language fields may be empty with or without it (see Notes).
annotation:
  command: >
    echo "Read {prepared_state} and write annotations to {annotation_root}"
```

Optional LeRobot export (only used when `lerobot` is in `--stages`; every key is forwarded to
`scripts/build/wds_to_lerobot.py`, `output_dir` defaults to `<output_root>/lerobot`, `fps` to
`build.target_fps`, which for the single-video config defaults to the video's fps; a non-integer
rate such as 29.97 is an error, not rounded, so set `lerobot.fps` explicitly):

```yaml
lerobot:
  hand_frame: world         # world | camera
  state_layout: egosteer74  # egosteer74 | hawor48
  task_source: dataset_name  # default: tasks[0] = dataset name; the sentences stay in instructions/language
  validate: true            # structural checks + no-depth guard after conversion
  # source_map: /path/to/source_map.parquet   # optional per-frame original-media mapping
```

`wds_dir`, `descriptor_manifest`, `validate_only` and `overwrite` are reserved and rejected
in the config. Resume precedence is the same as for the infer and build stages: `--resume`/`--no-resume`
on the CLI, else `lerobot.resume`, else the top-level `resume`. With resume off, an existing non-empty output
directory is an error: the orchestrator never overwrites it, so point `lerobot.output_dir` at a
new directory or empty the old one by hand.

Notes:

- The simplified `video:` config takes no runtime paths: its top-level keys are `video`, `output_root`,
  `run_tag`, `resume`, `annotation`, `clip`, `build`, `filter`, `validation`, `infer`,
  `adapter_config` and `lerobot`, and a `paths:` or `runtimes:` block is rejected. Stages run in the
  active `egosmith` env (see "Runtime" below).
- In a `video:` config, instruction/language fields may stay empty even when `annotation.command` is
  set: `build.require_annotation` defaults to `false` and `validation.allow_empty_instruction` to
  `true`, so a clip without a usable sidecar is still exported and validated. Set
  `build.require_annotation: true` and `validation.allow_empty_instruction: false` to enforce annotations.
- `annotation.command` receives `{prepared_state}`, `{active_prepared_state}`, `{annotation_root}`,
  `{run_dir}`, `{hawor_python}`, `{slam_python}`, and `{project_root}`.

See [configs/README.md](../configs/README.md) for the config inventory.

## Runtime (which Python runs the stages)

EgoSmith runs in a single conda env (`egosmith`). Stage subprocesses use the orchestrator's own
interpreter (`sys.executable`), so **activate the env first** (`conda activate egosmith`, or
`pip install -e .`). Nested (`dataset:` / `paths:`) configs can override the interpreter per runtime
(`runtimes.hawor_python` / `runtimes.slam_python`); the single-video `video:` config rejects a
`runtimes` block. Multihost runs set each remote host's interpreters in its
`infer.multihost.hosts[]` entry (`hawor_python` / `slam_python`; `infer.multihost.hawor_python` /
`slam_python` set them for every host, and `runtimes.*` is the fallback). Validate the setup from the repo root:

```bash
bash scripts/setup/validate_setup.sh
```

## Quality Control (the `filter` stage)

The `filter` stage applies multi-level quality control (`src/lib/pipeline/quality/quality_metrics.py`). Hard
rules: any `NaN/Inf` or otherwise invalid lowdim frame (rot6d / extrinsic / intrinsic) always drops the
whole episode; a missing / empty / mismatched instruction frame drops it only when
`build.require_annotation: true` (default `false`). On top of that:

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

Or directly (`--descriptor_manifest` is required: the run's filtered state
`<output_root>/runs/<run_tag>/clip_manifest.filtered.jsonl`, or `clip_manifest.jsonl` if `filter` did not run):

```bash
python scripts/validate_pipeline_run.py \
  --descriptor_manifest /path/to/output_root/runs/run/clip_manifest.filtered.jsonl \
  --dataset_dir /path/to/output_root/webdataset \
  --max_clips 200 \
  --dataset_sample_checks 20 \
  --allow_empty_instruction
```

The script's own defaults are stricter than the orchestrator's: `--allow_empty_instruction` and
`--require_depth` default to off, whereas the orchestrator passes the config's `validation.*`
values (for a `video:` config, empty instructions allowed and depth required). Pass
`--allow_empty_instruction` for an unannotated run as above, and `--require_depth` to check depth.

Recommended smoke pass before a large run:

1. Run a small representative subset.
2. Complete `prepare` through `validate`.
3. Inspect multiple output samples across different shards.
4. Confirm image, lowdim, MANO, camera, and depth stay aligned.

## Inspect & visualize

**From a finished run** (lightest — no extra deps):

```bash
# overlay the reconstructed hands back onto the video, via direct K-projection
python scripts/overlay_hand_cam.py --seq_folder /path/to/output_root/stage_outputs/<clip_id> \
  --frames_dir /path/to/output_root/frames/<clip_id>
# inspect a batch run directory and print a report
python scripts/analyze_run.py /path/to/run_dir
```

`overlay_hand_cam.py` reads the poses (`result.npz`, or a legacy `world_space_res.pth`) and the SLAM
export from `--seq_folder`, and the frames as `*.jpg` from `--frames_dir` (default
`<seq_folder>/extracted_images/`, which only the `demo.py` / `extract_frames.py` layout has; an
orchestrated run keeps its frames under the adapter's frames root, `<output_root>/frames/<clip_id>/`
for a `video:` config). A hand is drawn only on frames where it is marked valid.

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
