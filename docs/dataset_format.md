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

Each tar contains per-frame samples sharing a key `…_ep<episode>_f<frame>`:

```
taco_ep000123_f000045.image.jpg     # RGB frame (head camera)
taco_ep000123_f000045.lowdim.npy    # 116-d state/action/camera vector (see below)
taco_ep000123_f000045.mano.npy      # MANO PCA params for both hands
taco_ep000123_f000045.meta.json     # per-sample metadata
taco_ep000123_f000045.depth.npy     # metric depth (only when export_depth is on)
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

## Inspecting output

See [running_at_scale.md](running_at_scale.md) and `scripts/inspection/` for validation, and
[dataset_pipeline.md](dataset_pipeline.md) for how the dataset is built and checked.

---

> EgoSmith targets the egocentric-video portion of the data spec. The broader EgoSteer dataset
> conventions (e.g. additional robot cameras, bilingual instructions, teleop/quality flags) are
> defined in the EgoSteer / Robot Stack repositories and are not all produced by this pipeline.
