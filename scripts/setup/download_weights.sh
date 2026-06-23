#!/bin/bash
# One-command fetch for every model checkpoint the pipeline needs.
#
#   bash scripts/setup/download_weights.sh
#
# Files land at the exact paths preflight and the stages resolve by default, so a
# fresh checkout is runnable without setting any HAWOR_*_PATH env vars. Idempotent:
# already-present non-empty files are skipped, so re-run to resume a partial fetch.
#
# NOT downloaded here (obtain separately):
#   - MANO assets (_DATA/...): research license from the official MANO site; see README "Weights".
#
# License note: HaWoR checkpoints + model_config are CC-BY-NC-ND (non-commercial,
# no-derivatives). Any4D (Apache-2.0), DPVO (MIT), WiLoR detector — see each source.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

log() { echo -e "\n=== $* ==="; }

# fetch <url> <dest-path>: download to dest unless it already exists and is non-empty.
fetch() {
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then
        echo "skip (present): $dest"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    log "Downloading $dest"
    wget -q --show-progress -O "$dest" "$url" || { rm -f "$dest"; echo "FAILED: $url" >&2; return 1; }
}

# --- WiLoR hand detector (used by detect_track) ---
fetch "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt" \
      "weights/external/detector.pt"

# --- HaWoR backbone + infiller + config (motion / infiller) [CC-BY-NC-ND] ---
fetch "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt" \
      "weights/hawor/checkpoints/hawor.ckpt"
fetch "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt" \
      "weights/hawor/checkpoints/infiller.pt"
fetch "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml" \
      "weights/hawor/model_config.yaml"

# --- Any4D dense-depth checkpoint (slam stage) ---
# Default resolution path (no HAWOR_ANY4D_CHECKPOINT_PATH needed): <repo>/thirdparty/Any4D/checkpoints/
fetch "https://huggingface.co/airlabshare/any4d-checkpoint/resolve/main/any4d_4v_combined.pth" \
      "thirdparty/Any4D/checkpoints/any4d_4v_combined.pth"

# --- DPVO checkpoint (slam stage) ---
# DPVO ships its weight inside models.zip as top-level dpvo.pth; place it under
# the path DPVO resolves by default.
if [ -s "thirdparty/DPVO/models/dpvo.pth" ]; then
    echo "skip (present): thirdparty/DPVO/models/dpvo.pth"
else
    log "Downloading thirdparty/DPVO/models/dpvo.pth (from DPVO models.zip)"
    ( cd thirdparty/DPVO \
      && wget -q --show-progress -O models.zip "https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip?dl=1" \
      && mkdir -p models \
      && unzip -p models.zip "dpvo.pth" > models/dpvo.pth \
      && rm -f models.zip )
fi

log "All weights present. Verify with: bash scripts/setup/validate_setup.sh"
