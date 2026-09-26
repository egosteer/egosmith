# Dataset output format

The `build` stage writes a **WebDataset**: a set of `.tar` shards, each holding many per-frame
samples. This page documents what EgoSmith actually produces today.

## Shard layout

```
<final_dataset_root>/
  shard-000000.tar
  shard-000001.tar
  ...
```

Each tar contains per-frame samples sharing a key `<clip_id>_f<frame:06d>`, where `<frame>` is the
0-based frame index within the clip. When clips are repeated (`--repeat_episodes k`, k > 1), every
repeat after the first gets a unique episode part, `<clip_id>__rep<k>_f<frame:06d>`:

```
<clip_id>_f000045.image.jpg     # RGB frame (head camera)
<clip_id>_f000045.lowdim.npy    # 116-d state/action/camera vector (see below)
<clip_id>_f000045.mano.npy      # MANO PCA params for both hands
<clip_id>_f000045.meta.json     # per-sample metadata
<clip_id>_f000045.depth.npy     # metric depth (only when export_depth is on)
```

A loader groups consecutive frames into windows at training time (state/action horizons); the files
above are the per-frame building blocks.

## `lowdim.npy` — shape `(116,)`, float32

Concatenation of six segments (all hand quantities in the **world** frame):

| Segment | Size | Layout |
|---|---|---|
| `wrist_state` | 18 | left_trans(3) + right_trans(3) + left_root_rot6d(6) + right_root_rot6d(6) |
| `hand_state` | 30 | left fingertips (5×3) + right fingertips (5×3) |
| `wrist_action` | 18 | next-frame `wrist_state` |
| `hand_action` | 30 | next-frame `hand_state` |
| `extrinsic` | 16 | flattened 4×4 world→camera matrix (head camera) |
| `intrinsic` | 4 | `[fx, fy, cx, cy]` (head camera) |

Conventions: wrist translation is MANO joint-0 in world coordinates
(`wrist_translation_semantics = mano_joint_0_world`); rot6d is the first two columns of the
wrist→world rotation; the extrinsic is world→camera (`camera_extrinsic_convention = w2c`). The slice
offsets are defined in `src/lib/pipeline/quality/constants.py`; the vector is assembled in
`src/lib/pipeline/exporters/lowdim_assembly.py`.

## `meta.json`

Fields written per sample (`src/lib/pipeline/exporters/manifest_build/writer.py`):

```json
{
  "dataset_name": "taco",
  "clip_id": "...",
  "episode_index": 123,
  "split": "train",
  "instruction": ["pick up the cup", "grasp the cup"],
  "instruction_num": 2,
  "language": "grasp the cup",
  "presence": 3,
  "lowdim_schema": "hawor_wrist_world_v2",
  "wrist_translation_semantics": "mano_joint_0_world",
  "camera_extrinsic_convention": "w2c"
}
```

- `presence` — bitmask of which hands are present: `0` none, `1` left, `2` right, `3` both.
- `instruction` / `instruction_num` / `language` — from annotation; empty when annotation is off
  (see [annotation.md](annotation.md)).
- When `export_depth` is enabled, `depth_schema` / `depth_encoding` are added and each sample gets a
  `.depth.npy`.

## LeRobot export (v3.0, no depth)

`scripts/build/wds_to_lerobot.py` converts the shards above into a LeRobot v3.0 dataset laid out
like the EgoSteer release (loadable by `lerobot` 0.6.x):

```
python scripts/build/wds_to_lerobot.py --wds_dir <final_dataset_root> --output_dir <lerobot_root> \
    --fps 30 --workers 8 --validate
```

```
<lerobot_root>/
  meta/info.json  meta/stats.json  meta/tasks.parquet  meta/episodes/chunk-XXX/file-XXX.parquet
  data/chunk-XXX/file-XXX.parquet                       # one row group per episode
  videos/observation.images.head/chunk-XXX/file-XXX.mp4 # h264 crf18 g15 yuv420p, 1:1 with the parquet
```

| feature | dtype | shape | source |
|---|---|---|---|
| `observation.images.head` | video | (H, W, 3) | `image.jpg`, decoded once and piped into ffmpeg |
| `observation.state` | float32 | (74,) | EgoSteer layout: `[0:26]` robot arm/hand dims padded (`--pad_value`, default 0); `[26:35]` left wrist xyz+rot6d, `[35:44]` right wrist, `[44:59]` left fingertips, `[59:74]` right fingertips, from `lowdim[0:48]` |
| `action` | float32 | (74,) | same layout from `lowdim[48:96]` (next-frame state) |
| `observation.hand_presence` | int64 | (1,) | `meta.presence` bitmask |
| `observation.camera.head_world2cam` | float32 | (16,) | `lowdim[96:112]`, row-major w2c |
| `observation.mano` | float32 | (110,) | `mano.npy` flattened: left pca45+betas10, right pca45+betas10 (`--no_mano` drops it) |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | float32 / int64 | (1,) | LeRobot built-ins; `timestamp = frame_index / fps` |

Coordinate frame: by default (`--hand_frame world`) every hand quantity — state **and** action —
stays in the episode's SLAM world frame, and the per-frame w2c column maps it into the head camera
of frame *t*; this is what a loader that anchors a window to its observation frame (world-frame
hands + per-row w2c) expects. `--hand_frame camera` instead re-expresses every hand quantity of
frame *t* in the head camera of frame *t* (EgoSteer's "world frame ≡ head camera frame"); do not
apply the w2c column again to such data. The per-frame w2c column is exported in both modes
so either can be recovered. `--no_extrinsic` drops it and is therefore only accepted with
`--hand_frame camera`; the converter rejects `--hand_frame world --no_extrinsic`, because world-frame
hands without the per-frame w2c cannot be related to the camera. rot6d is the first two rotation-matrix columns,
column-major, in both datasets.

Episode metadata (`meta/episodes`) carries `tasks` (one task per episode, from `--task_source`,
default `dataset_name`, falling back to `language`; `language` / `first_instruction` / `clip_id`
select the other meta fields, and `--task_name` overrides all of them — the batch driver sets it to the
job name when `task_source` is `dataset_name`), `instructions` (the full
annotation list), `split`, `clip_id`, `source_dataset`, `source_episode_index`,
`calibration/head_intrinsics` (3×3 row-major), `calibration/head_world2cam` (camera mode only: identity, as in
EgoSteer where world ≡ head camera) and the per-episode `stats/*` columns. Episodes are
ordered train first, then val (`--split_order`), and `info.json.splits` records the contiguous
ranges. `--state_layout hawor48` exports the bare 48-d hand block instead of the padded 74-d vector.
Depth is never exported. `meta/egosmith_provenance.json` records the conversion settings.

`--validate` checks row counts, index continuity, video frame counts and (when `lerobot` is
installed) loads a few items through `LeRobotDataset`. `--resume` skips finished file pairs; it refuses
when the conversion options or the input shards differ from the run that wrote them (shards are identified by
name, size and a sampled content hash, not by path, so a copy on another machine still resumes).

### Verifying a conversion

Pass `--validate` to `scripts/build/wds_to_lerobot.py` to check row counts, episode/index
continuity, video frame counts, metadata, and loading through `LeRobotDataset` when installed.

### Release tiers, licences, rehydration

One LeRobot dataset per source dataset. Each batch job carries a `release` block (tier `open` /
`gated` / `labels_only` / `internal`, source licence, attribution); the `release` step writes
`LICENSES/<source>/`, `meta/release.json` and a root `README.md`, and fails on configurations that
would breach the source licence (ShareAlike / NonCommercial propagation, labels-only with video).
`labels_only` datasets carry no pixels (the tier forces `--no_video`). `--descriptor_manifest` writes the
rehydration index `meta/source_frames.parquet` (with or without video; the `labels_only` tier requires it),
and `scripts/build/lerobot_rehydrate_video.py` re-attaches the video from frames the user extracts
from the original data.

### Batch conversion

`scripts/build/wds_to_lerobot_batch.py --config configs/wds_to_lerobot_batch.example.yaml` converts
every job in the YAML and runs convert → release → validate → verify_wds → compare_reference → loader on each,
writing `<output_dir>/_verification/report.{json,md}` and `report_dir/batch_report.{json,md}`.
`--jobs a,b` / `--steps verify_wds,loader` select subsets, `--skip_done` skips jobs whose last
report was green, `--dry_run` prints the resolved configs. Every `defaults` key can be overridden
per job.

## Inspecting output

See [dataset_pipeline.md § Validation](dataset_pipeline.md#validation) for validating a run, and
[dataset_pipeline.md](dataset_pipeline.md) for how the dataset is built and checked.

---

> EgoSmith targets the egocentric-video portion of the data spec. The broader EgoSteer dataset
> conventions (e.g. additional robot cameras, bilingual instructions, teleop/quality flags) are
> defined in the EgoSteer / Robot Stack repositories and are not all produced by this pipeline.
