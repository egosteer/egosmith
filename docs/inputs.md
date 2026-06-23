# Supported inputs & dataset adapters

EgoSmith reads your source data through a small **adapter** layer that normalizes any source into a
common *prepared clip state* (the rest of the pipeline — inference, filtering, build — is
source-agnostic). This page lists what you can feed in and how to select it.

> The shipped example config is single-video (`configs/dataset_pipeline_single_video.example.yaml`).
> The other adapters are supported in code; for those you author your own config from the keys below
> (ready-made multi-dataset example configs are still being prepared).

## Picking an adapter

- **Single video** — just set the top-level `video:` key; the `single_video` adapter is used
  automatically.
  ```yaml
  video: /path/to/input.mp4
  ```
- **Anything else** — set `dataset.adapter:` and the adapter's input paths under `paths:` /
  `adapter_config:`.
  ```yaml
  dataset:
    adapter: video_folder
  paths:
    video_root: /path/to/videos
  ```

## Adapters

| `dataset.adapter` | Input it expects | Notes |
|---|---|---|
| `single_video` | one `.mp4` file (top-level `video:`) | the blessed quickstart path; frames extracted for you |
| `video_folder` | a folder tree of `.mp4` files | recursively discovered; optional frame extraction |
| `image_sequence` | folders of pre-extracted frames (`.jpg`/`.png`) | one subfolder per clip |
| `buildai` | BuildAI factory-organized raw videos / shards | migration path; uses `start_factory_id`/`end_factory_id` |
| `hot3d_wds` | HOT3D WebDataset `.tar` shards | reads frame/meta/bbox/lowdim/mano payloads directly |
| `fpha_tar` | FPHA `.tar` shards (one tar per sequence) | filename `Subject_<id>_<action>_<trial>.tar` |
| `legacy_buildai`, `flat_shard` | legacy / flat BuildAI shard layouts | advanced migration adapters |

Adapters are registered in `src/lib/pipeline/datasets/__init__.py`; each implementation lives next to
it (e.g. `src/lib/pipeline/datasets/video_folder.py`).

### Common adapters — minimal config

**Single video**
```yaml
video: /path/to/input.mp4
# optional:
# output_root: /path/to/out
# adapter_config: { clip_id: my_clip, frame_ext: .jpg, jpeg_quality: 95 }
```

**Folder of videos**
```yaml
dataset:
  adapter: video_folder
paths:
  video_root: /path/to/videos
adapter_config:
  extract_frames: true        # set false if frames are already extracted
  frame_subdir: extracted_images
  frame_ext: .jpg
```

**Pre-extracted image sequences**
```yaml
dataset:
  adapter: image_sequence
paths:
  sequence_root: /path/with/<clip>/*.jpg
# adapter_config: { include_dirs: [clipA, clipB] }   # optional allowlist
```

## Raw-video clipping (optional, runs before `prepare`)

A `clip:` block trims raw videos into shorter clips before the rest of the pipeline. Modes:

```yaml
clip:
  mode: none            # default — no clipping
```
```yaml
clip:
  mode: heuristic       # optical-flow + hand-gate clipping into MP4s
  config: src/lib/clip/heuristic_clip_config.yaml
```
```yaml
clip:
  mode: api             # one multimodal call per raw video → segments + language
  prompt_file: src/lib/annotation/prompts/with_clip/annotation_general_clip.txt
  annotation_suffix: _qwen-annotation.json
  workers: 4
```

After clipping, later stages operate on the clipped videos. `clip.mode: api` also writes the
language sidecars, so the separate `annotate` stage is skipped — see
[annotation.md](annotation.md).

## Related

- [dataset_pipeline.md](dataset_pipeline.md) — how the stages turn a prepared clip state into a dataset
- [annotation.md](annotation.md) — language annotation
- [running_at_scale.md](running_at_scale.md) — many videos / multi-GPU
