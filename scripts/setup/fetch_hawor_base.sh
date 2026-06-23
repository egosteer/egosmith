#!/bin/bash
# Obtain the HaWoR base source from the pinned HaWoR submodule into the src/ tree.
# EgoSmith uses HaWoR as a component; the HaWoR-authored base (model architecture,
# infiller, hawor utils, ...) is NOT redistributed here — it is gitignored and
# materialized from the upstream HaWoR repo (a git submodule, i.e. a pointer) by
# this script. Run it once after `git clone` / `git pull` (and after
# `git submodule update --init thirdparty/hawor_upstream`).
#
# Single mechanism: every base file is a relative SYMLINK into the submodule, so the
# HaWoR source physically lives ONLY in the submodule and EgoSmith ships no HaWoR
# code or patches. EgoSmith's own behavior changes live in first-party modules /
# config overrides (e.g. lib/pipeline/hands/track_dataset.py, the TORCH_COMPILE override
# in lib/pipeline/stages/hawor_runtime.py), not in edits to these files.
#
# Idempotent: symlinks are recreated each run. The pinned upstream commit is
# recorded by the submodule, so obtained files match the versions EgoSmith expects.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SRC="thirdparty/hawor_upstream"

if [ ! -d "$SRC/lib" ]; then
    echo "[fetch_hawor_base] submodule not populated: $SRC" >&2
    echo "  run: git submodule update --init $SRC" >&2
    exit 1
fi

# Pure-upstream HaWoR base, symlinked into src/<upstream-path> (never redistributed).
SYMLINK_FILES=(
    # infiller (CMIB-derived motion in-betweening)
    infiller/lib/misc/sampler.py
    infiller/lib/model/network.py
    infiller/lib/model/positional_encoding.py
    infiller/lib/model/preprocess.py
    infiller/lib/model/skeleton.py
    infiller/lib/vis/pose.py
    # model architecture (HaMeR / 4D-Humans derived) + HAWOR (EgoSmith's batched
    # inference is first-party in lib/pipeline/hands/hawor_inference.py; torch.compile is
    # disabled via a config override in lib/pipeline/stages/hawor_runtime.py)
    lib/core/constants.py
    lib/models/backbones/__init__.py
    lib/models/backbones/vit.py
    lib/models/components/__init__.py
    lib/models/components/pose_transformer.py
    lib/models/components/t_cond_mlp.py
    lib/models/mano_wrapper.py
    lib/models/modules.py
    lib/models/hawor.py
    # eval / pipeline / data utils (EgoSmith additions live in first-party modules;
    # the frame-source dataset is first-party in lib/pipeline/hands/track_dataset.py and
    # reuses the upstream crop here)
    lib/datasets/track_dataset.py
    lib/eval_utils/custom_utils.py
    lib/eval_utils/filling_utils.py
    lib/eval_utils/video_utils.py
    lib/pipeline/est_scale.py
    lib/pipeline/tools.py
    lib/utils/geometry.py
    lib/utils/imutils.py
    # aitviewer visualization helpers (obtained for base parity; no shipped entrypoint imports them)
    lib/vis/renderer.py
    lib/vis/renderer_world.py
    lib/vis/run_vis2.py
    lib/vis/tools.py
    lib/vis/viewer.py
    lib/vis/wham_tools/tools.py
)

# Pre-flight: every source file must exist before we touch anything.
missing=0
for f in "${SYMLINK_FILES[@]}"; do
    [ -f "$SRC/$f" ] || { echo "[fetch_hawor_base] missing in upstream: $SRC/$f" >&2; missing=1; }
done
[ -d "$SRC/hawor" ] || { echo "[fetch_hawor_base] missing in upstream: $SRC/hawor" >&2; missing=1; }
[ -f "$SRC/_DATA/data/mano_mean_params.npz" ] || { echo "[fetch_hawor_base] missing in upstream: $SRC/_DATA/data/mano_mean_params.npz" >&2; missing=1; }
[ "$missing" -eq 0 ] || exit 1

# The first-party package tree lives under src/ (src/lib, src/infiller, src/hawor),
# so base files are materialized at src/<upstream-path>. Array entries stay as the
# upstream-relative paths (read from $SRC/$f); only the destination is under src/.
# Relative symlink src/$f -> $SRC/$f (relative so the checkout stays relocatable).
link_file() {
    local f="$1" dest="src/$1" target
    target="$(realpath -m --relative-to="$ROOT/$(dirname "$dest")" "$ROOT/$SRC/$f")"
    mkdir -p "$(dirname "$dest")"
    rm -rf "$dest"
    ln -s "$target" "$dest"
}

for f in "${SYMLINK_FILES[@]}"; do
    link_file "$f"
done

# Top-level `hawor` package (HaWoR's utils/configs): whole-directory symlink under
# src/ so both upstream base files and EgoSmith first-party code resolve `hawor.*`.
mkdir -p src
rm -rf src/hawor
ln -s "../$SRC/hawor" src/hawor

mkdir -p _DATA/data
cp "$SRC/_DATA/data/mano_mean_params.npz" _DATA/data/mano_mean_params.npz

echo "[fetch_hawor_base] linked ${#SYMLINK_FILES[@]} HaWoR base files (+ hawor/ + MANO mean params) from $SRC ($(git -C "$SRC" rev-parse --short HEAD))."
