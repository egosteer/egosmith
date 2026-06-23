# Language annotation

EgoSmith can attach natural-language instructions to each clip with a multimodal LLM. Annotation is
**optional** — without it the pipeline still produces a full dataset, just with empty instruction
fields. This page covers how to turn it on and configure the API.

## Backend & API key

Annotation calls the **DashScope (Qwen)** multimodal API (`src/lib/annotation/api_annotation.py`,
default model `qwen3.5-plus`). Provide the key via environment variable (preferred — never commit
keys into configs):

```bash
export DASHSCOPE_API_KEY=sk-...
```

Alternatives: `--api_key sk-...` or `--api_keys_file keys.txt` (one key per line, for rotation).

## Turning it on: the `annotate` stage

Annotation runs as the pipeline's `annotate` stage, driven by an `annotation.command` template in
your config. When `annotation.command` is present, the default stage order becomes
`prepare,annotate,infer,filter,build,validate`.

```yaml
annotation:
  command: >
    {hawor_python} {project_root}/src/lib/annotation/api_annotation.py
    --prepared_state {prepared_state}
    --annotation_root {annotation_root}
    --annotation_suffix _qwen-annotation.json
    --prompt_file {project_root}/src/lib/annotation/prompts/without_clip/annotation_general_egocentric.txt
    --workers 4
    --target_fps 5.0
```
```bash
export DASHSCOPE_API_KEY=sk-...
python scripts/run_dataset_pipeline.py --config configs/my_video.yaml
```

The command is a template; the orchestrator substitutes these variables before running it:

| Variable | Meaning |
|---|---|
| `{prepared_state}` / `{manifest}` | the prepared clip-state JSONL the stage annotates |
| `{active_prepared_state}` | active manifest (may differ after filtering) |
| `{annotation_root}` | directory where sidecars are written |
| `{run_dir}` | this run's directory |
| `{hawor_python}` / `{slam_python}` | resolved conda-env Python executables |
| `{project_root}` | repo root |

Useful `api_annotation.py` flags: `--prompt_file`, `--model`, `--workers`, `--target_fps`,
`--max_clips` / `--clip_ids` (testing subset), `--resume` (skip existing sidecars), `--dry_run`
(skip API calls).

## Two ways to annotate

- **Standalone `annotate` stage** (above) — annotates already-prepared clips, one annotation per clip.
- **`clip.mode: api`** (see [inputs.md](inputs.md)) — segments each *raw* video into clips **and**
  annotates every segment in a single API call during preprocessing. It writes the sidecars itself,
  so the separate `annotate` stage is automatically skipped.

## Prompts

Built-in prompts under `src/lib/annotation/prompts/`:

- `without_clip/annotation_general_egocentric.txt` — annotate already-prepared egocentric clips
- `without_clip/annotation_industrial_egocentric.txt` — industrial/workshop variant
- `with_clip/annotation_general_clip.txt` — segment a raw video **and** annotate (for `clip.mode: api`)

Point `--prompt_file` (or `clip.prompt_file`) at any of these or your own. Always pass an explicit
prompt path. To emit other languages (e.g. Chinese instructions), write a custom prompt that asks the
model for them — the system is language-agnostic; only the prompt controls the output language.

## Output: annotation sidecars

Sidecars land at `{annotation_root}/<clip_id><suffix>` (default suffix `.annotation.json`). Schema
(`src/lib/pipeline/clips/annotation_protocol.py`):

```json
{
  "clip_id": "...",
  "status": "Valid",
  "instruction": ["level1", "level2", "level3", "level4", "level5"],
  "instruction_num": 5,
  "language": "single-string summary",
  "hierarchy": { "level1": "...", "...": "..." }
}
```

`instruction` is a coarse-to-fine list (up to 5 levels); `instruction_num` is its length; `language`
is a single representative string. The `build` stage copies `instruction` / `instruction_num` /
`language` into each sample's `meta.json` (see [dataset_format.md](dataset_format.md)).

### Attaching pre-made annotations (no API call)

Place sidecars under `paths.annotation_root` before `build` and set `build.annotation_suffix` to
match `<clip_id><suffix>`. With `build.require_annotation: false`, clips without a sidecar are still
exported (empty instruction fields).

## Related

- [inputs.md](inputs.md) — `clip.mode: api` semantic clipping + annotation
- [dataset_format.md](dataset_format.md) — where instruction fields land in the output
- [running_at_scale.md](running_at_scale.md) — annotating a whole dataset
