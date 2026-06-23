# Configs

This directory holds the example config for the official single-video pipeline.

## Single-video config

```yaml
video: /path/to/input.mp4
output_root: /optional/output_root
```

If `output_root` is omitted, it defaults to `<video_dir>/<video_stem>.hawor_pipeline/`.
The orchestrator derives extracted frames, stage outputs, logs, run state, reports, and the
final WebDataset shards from that root.

`annotation.command` is optional. When omitted, the default stages are
`prepare,infer,filter,build,validate` and empty instruction/language fields are allowed.

See `dataset_pipeline_single_video.example.yaml` for the full template (it documents the
optional `clip` and `annotation` blocks inline).

## Other inputs & multi-dataset

The pipeline supports more input types through dataset adapters (folders of videos, image
sequences, BuildAI / HOT3D / FPHA, …) — see **[../docs/inputs.md](../docs/inputs.md)** for the
adapter list and config keys. Ready-made multi-dataset example configs are still being prepared;
for now you author those configs yourself from the documented keys.
