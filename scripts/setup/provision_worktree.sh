#!/bin/bash
# Provision a fresh checkout / worktree with the gitignored runtime dependencies
# (model weights, MANO assets, Any4D/DPVO checkpoints, the upstream Any4D script)
# by symlinking them from an already-provisioned source tree. Use after `git
# clone` / `git pull` / `git worktree add` so the pipeline preflight passes.
#
# Usage:
#   HAWOR_SRC=/path/to/provisioned/checkout  bash scripts/setup/provision_worktree.sh
#
# Everything linked here is gitignored (never tracked). The script only creates
# symlinks into THIS checkout; it does not modify the source tree.
#
# This complements (does not replace):
#   - scripts/setup/fetch_hawor_base.sh   (HaWoR base source, from the submodule)
#   - scripts/setup/download_weights.sh   (fresh download of weights if no source tree)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SRC="${HAWOR_SRC:-}"
if [ -z "$SRC" ]; then
    echo "[provision] set HAWOR_SRC to an already-provisioned checkout, e.g.:" >&2
    echo "    HAWOR_SRC=/path/to/provisioned/checkout bash scripts/setup/provision_worktree.sh" >&2
    exit 1
fi
SRC="$(cd "$SRC" && pwd)"
if [ "$SRC" = "$ROOT" ]; then
    echo "[provision] HAWOR_SRC must differ from this checkout ($ROOT)" >&2
    exit 1
fi

# link <relative-path> : symlink $ROOT/<path> -> $SRC/<path> (resolving the
# source's own symlink to its final target so we don't chain links).
link() {
    local rel="$1"
    local src_real
    if [ ! -e "$SRC/$rel" ]; then
        echo "[provision] MISSING in source, skipped: $rel" >&2
        return 1
    fi
    src_real="$(readlink -f "$SRC/$rel")"
    mkdir -p "$(dirname "$rel")"
    ln -sfn "$src_real" "$rel"
    echo "[provision] linked $rel -> $src_real"
}

# link_abs <abs-source> <relative-dest> : symlink $ROOT/<dest> -> <abs-source>.
# For deps that live outside HAWOR_SRC (e.g. checkpoints on a shared disk).
link_abs() {
    local src="$1" dest="$2"
    if [ ! -e "$src" ]; then
        echo "[provision] MISSING source, skipped: $dest (looked at $src)" >&2
        return 1
    fi
    src="$(readlink -f "$src")"
    mkdir -p "$(dirname "$dest")"
    ln -sfn "$src" "$dest"
    echo "[provision] linked $dest -> $src"
}

# Checkpoints may live outside HAWOR_SRC; override these if your paths differ.
DPVO_CKPT="${HAWOR_DPVO_CKPT:-$SRC/thirdparty/DPVO/models/dpvo.pth}"
ANY4D_CKPT="${HAWOR_ANY4D_CKPT:-$SRC/thirdparty/Any4D/checkpoints/any4d_4v_combined.pth}"

missing=0

# Model weights (preflight: detector / hawor ckpt+config / infiller).
link "weights" || missing=1
# MANO assets (preflight: _DATA/.../MANO_{RIGHT,LEFT}.pkl).
link "_DATA" || missing=1
# Any4D dense-depth checkpoint (slam stage) — absolute, overridable via HAWOR_ANY4D_CKPT.
link_abs "$ANY4D_CKPT" "thirdparty/Any4D/checkpoints/any4d_4v_combined.pth" || missing=1
# DPVO checkpoint (slam stage) — absolute, overridable via HAWOR_DPVO_CKPT.
link_abs "$DPVO_CKPT" "thirdparty/DPVO/models/dpvo.pth" || missing=1
# Upstream Any4D script that provides init_inference_model / sample_inference.
link "thirdparty/Any4D/scripts/demo_inference.py" || missing=1

echo
if [ "$missing" -ne 0 ]; then
    echo "[provision] some items were missing in $SRC (see above) — provision them there first." >&2
fi

# Stage-3 scratch root is never defaulted (can write many GB). Remind, don't guess.
if [ -z "${HAWOR_BATCH_TMPDIR:-}${HAWOR_STAGE3_TMP_ROOT:-}" ]; then
    echo "[provision] NOTE: set a large-disk scratch root before running the slam stage, e.g.:"
    echo "    export HAWOR_BATCH_TMPDIR=/efs-exp/<user>/tmp"
else
    echo "[provision] scratch root OK: HAWOR_BATCH_TMPDIR=${HAWOR_BATCH_TMPDIR:-} HAWOR_STAGE3_TMP_ROOT=${HAWOR_STAGE3_TMP_ROOT:-}"
fi
echo "[provision] done."
