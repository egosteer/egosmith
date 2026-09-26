#!/bin/bash
# One-command fetch for every model checkpoint the pipeline needs.
#
#   bash scripts/setup/download_weights.sh
#
# Files land at the exact paths preflight and the stages resolve by default, so a
# fresh checkout is runnable without setting any HAWOR_*_PATH env vars. Idempotent:
# already-present non-empty files are skipped, so re-run to resume a partial fetch.
# HF_ENDPOINT=<mirror> (the huggingface_hub convention) redirects the Hugging Face downloads.
#
# NOT downloaded here (obtain separately):
#   - MANO assets (_DATA/...): research license from the official MANO site; see README "Installation".
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
    url="${url/#https:\/\/huggingface.co/${HF_ENDPOINT:-https://huggingface.co}}"
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
# DPVO ships its weight inside models.zip (Dropbox); that link has become unreliable (it may answer
# with an HTML page), so a Hugging Face copy is the fallback. ASSETS_DIR (a directory holding a
# dpvo.pth) is tried first, for machines without outside access.
dpvo_dest="thirdparty/DPVO/models/dpvo.pth"
if [ -s "$dpvo_dest" ]; then
    echo "skip (present): $dpvo_dest"
else
    mkdir -p "$(dirname "$dpvo_dest")"
    found=""
    [ -n "${ASSETS_DIR:-}" ] && found="$(find "$ASSETS_DIR" -name dpvo.pth -size +10M -print -quit 2>/dev/null || true)"
    if [ -n "$found" ]; then
        cp -L "$found" "$dpvo_dest"; echo "copied: $dpvo_dest <- $found"
    else
        log "Downloading $dpvo_dest (DPVO models.zip, Dropbox)"
        if wget -q --show-progress -O thirdparty/DPVO/models.zip "https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip?dl=1" \
           && unzip -tq thirdparty/DPVO/models.zip >/dev/null 2>&1 \
           && unzip -p thirdparty/DPVO/models.zip "dpvo.pth" > "$dpvo_dest"; then
            rm -f thirdparty/DPVO/models.zip
        else
            rm -f thirdparty/DPVO/models.zip "$dpvo_dest"
            echo "Dropbox did not deliver a usable models.zip; trying the Hugging Face copy" >&2
            fetch "https://huggingface.co/pablovela5620/dpvo/resolve/main/dpvo.pth" "$dpvo_dest" || true
        fi
    fi
    if [ ! -s "$dpvo_dest" ] || [ "$(stat -c %s "$dpvo_dest")" -lt 10000000 ]; then
        rm -f "$dpvo_dest"
        echo "FAILED: could not obtain dpvo.pth. Get it from https://github.com/princeton-vl/DPVO (models.zip) on a" >&2
        echo "        machine with access and place it at $dpvo_dest, or point ASSETS_DIR at a directory holding it." >&2
        exit 1
    fi
fi

log "All weights present. Verify with: bash scripts/setup/validate_setup.sh"
