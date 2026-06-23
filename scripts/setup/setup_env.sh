#!/bin/bash
# One-command setup for the conda environment used by the pipeline.
#
#   scripts/setup/setup_env.sh
#
# This script is idempotent -- re-run it to resume a half-finished install.
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="egosmith"

log()  { echo -e "\n=== $* ==="; }
have_conda() { command -v conda >/dev/null 2>&1; }

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

create_env() {  # create_env <name> <python_version>
    local name="$1" py="$2"
    if env_exists "$name"; then
        log "conda env '$name' already exists -- skipping create"
    else
        log "Creating conda env '$name' (python=$py)"
        conda create -n "$name" "python=$py" -y
    fi
}

ensure_eigen() {  # DPVO's setup.py expects Eigen 3.4.0 here (not vendored)
    if [ -f thirdparty/DPVO/thirdparty/eigen-3.4.0/Eigen/Core ]; then
        log "Eigen 3.4.0 already present -- skipping fetch"
        return
    fi
    log "Fetching Eigen 3.4.0 for DPVO"
    ( cd thirdparty/DPVO && mkdir -p thirdparty \
      && wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip -O thirdparty/eigen-3.4.0.zip \
      && unzip -q -o thirdparty/eigen-3.4.0.zip -d thirdparty/ )
}

ensure_chumpy() {
    if [ ! -f thirdparty/chumpy/setup.py ] || grep -q "parse_requirements" thirdparty/chumpy/setup.py; then
        log "Materializing patched chumpy"
        bash scripts/setup/fetch_chumpy.sh
    fi
}

install_cuda_toolkit() {
    log "Installing CUDA toolkit into '$ENV_NAME'"
    conda install -n "$ENV_NAME" -c nvidia cuda-toolkit=12.8 -y
}

install_egosmith() {
    local R="conda run --no-capture-output -n $ENV_NAME"
    create_env "$ENV_NAME" 3.10
    install_cuda_toolkit
    # CUDA-specific wheels: install from the cu128 index BEFORE the generic deps.
    # (These need a custom --index-url, so they can't live in a plain requirements
    # file alongside PyPI packages.) xformers 0.0.32.post2 matches torch 2.8.0.
    log "Installing CUDA-specific torch stack"
    $R pip install --index-url https://download.pytorch.org/whl/cu128 \
        --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html \
        torch==2.8.0 torchvision==0.23.* torchaudio==2.8.* xformers==0.0.32.post2 torch-scatter
    ensure_chumpy
    log "Installing Python runtime requirements (core + Any4D)"
    $R pip install -r requirements.txt
    # Installed with --no-deps (not expressible in a requirements file): rerun-sdk
    # declares numpy>=2 and UniCeption pulls broad torch/vision/audio — both would
    # disturb the pinned torch 2.8 / numpy 1.26 runtime.
    log "Installing Any4D packages that must not resolve dependencies"
    $R pip install --no-deps \
        rerun-sdk==0.23.4 \
        "uniception @ git+https://github.com/JayKarhade/UniCeption.git@dev/any4d" \
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
        -e ./thirdparty/Any4D
    ensure_eigen
    log "Building DPVO into '$ENV_NAME'"
    $R bash -c 'export CUDA_HOME="$CONDA_PREFIX"; export PATH="$CUDA_HOME/bin:$PATH"; cd thirdparty/DPVO && pip install . --no-build-isolation'
    log "EgoSmith env ready."
}

main() {
    if [ "$#" -ne 0 ]; then
        echo "Usage: $0" >&2
        echo "  Creates/updates the fixed conda env: $ENV_NAME"
        exit 2
    fi
    if ! have_conda; then
        echo "ERROR: conda is not on PATH." >&2
        exit 1
    fi
    install_egosmith
    log "Done. Verify with: bash scripts/setup/validate_setup.sh"
}

main "$@"
