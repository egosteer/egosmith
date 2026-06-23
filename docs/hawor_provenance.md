# HaWoR base provenance

EgoSmith uses [HaWoR](https://github.com/ThunderVVV/HaWoR) (released under
**CC-BY-NC-ND** — non-commercial, no-derivatives) for hand reconstruction. To respect that license,
this repository **does not track or redistribute any HaWoR-authored source**. The HaWoR
base is pinned as a git submodule (`thirdparty/hawor_upstream`) and materialized
into the `src/` tree by `scripts/setup/fetch_hawor_base.sh`:

```bash
git submodule update --init thirdparty/hawor_upstream
bash scripts/setup/fetch_hawor_base.sh
```

This document records who authored what, so you can tell first-party EgoSmith code
from obtained HaWoR base at a glance.

## Two mechanisms

EgoSmith ships **no HaWoR code and no patches against HaWoR** — every base file is an
unmodified symlink, and all EgoSmith behavior lives in first-party modules / config overrides.

| Mechanism | What it means | Tracked in this repo? |
|---|---|---|
| **Symlink** | Pure-upstream file — a relative symlink into `thirdparty/hawor_upstream`. | No (gitignored) |
| **First-party** | EgoSmith-authored module. | Yes |

## Obtained from HaWoR (symlinked, not tracked)

Pure-upstream base — the license-sensitive core and supporting utilities:

- **Model architecture** — `src/lib/models/{backbones,components,mano_wrapper,modules}.py`, `src/lib/core/constants.py`
- **In-betweening network** — `src/infiller/lib/**`
- **`hawor` utils package** — `src/hawor/` (rotation / geometry / render / config / process), symlinked whole
- **Helpers / utils we call as-is** — `src/lib/eval_utils/{custom_utils,filling_utils,video_utils}.py`,
  `src/lib/pipeline/{tools,est_scale}.py`, `src/lib/utils/{geometry,imutils}.py`,
  `src/lib/datasets/track_dataset.py`, `src/lib/models/hawor.py`,
  `src/lib/vis/{renderer,renderer_world,run_vis2,viewer,tools,wham_tools/tools}.py`

> Notes on a few of these:
> - `custom_utils.py` / `tools.py` are obtained because they still contain verbatim HaWoR functions
>   (`cam2world_convert`, `quaternion_to_matrix`, `parse_chunks`, `parse_chunks_hand_frame`);
>   EgoSmith's additions that once lived alongside them are first-party modules (see below).
> - `hawor.py` is unmodified — `torch.compile` is disabled via a config override in the first-party
>   loader (`src/lib/pipeline/stages/hawor_runtime.py`), not by editing the file.
> - `track_dataset.py` / `imutils.py` are unmodified — EgoSmith's frame-source dataset is first-party
>   at `src/lib/pipeline/hands/track_dataset.py` and reuses the upstream `crop`/`boxes_2_cs`.
> - the `src/lib/vis/*` files are obtained as-is (aitviewer renderers); they are not imported by the
>   pipeline or by `demo.py` (which renders with OpenCV), and are kept only for parity with the base.

## First-party EgoSmith code (tracked)

EgoSmith's own modules — including the performance work decoupled out of the HaWoR base:

- `src/lib/pipeline/hands/hawor_inference.py` — batched HAWOR inference (window batching + overlapped CPU decode / GPU compute).
- `src/lib/pipeline/hands/est_scale_batch.py` — batched scale estimation.
- `src/lib/pipeline/hands/detect_track_batched.py` — batched YOLO detection + ByteTrack + bbox cleanup.
- `src/lib/pipeline/slam/slam_cam.py` — SLAM camera-trajectory loading / dense-export validation.
- `src/lib/pipeline/hands/mano_runtime.py` — cached MANO runtime helpers.
- `src/lib/pipeline/hands/track_dataset.py` — frame-source-backed track dataset (replaces HaWoR's imgfiles dataset).
- `src/lib/stage_runners/` — per-video stage runners (the demo-fork detect/slam/motion/infiller drivers).
- everything else under `src/lib/`, `scripts/`, `configs/`, `docs/` — the EgoSmith pipeline, orchestration, exporters, filtering, and entrypoints.

The HaWoR model checkpoints and `model_config.yaml` (see README "Weights") are likewise
CC-BY-NC-ND and obtained separately.
