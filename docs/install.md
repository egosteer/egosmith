# Installation (manual + troubleshooting)

The recommended path is `bash scripts/setup/setup_env.sh` (see the README). This page documents the exact
steps it runs, plus compatibility notes and troubleshooting.

EgoSmith uses a single conda env named `egosmith` (the pipeline resolves both the HaWoR and Any4D
subprocesses to it). `scripts/setup/setup_env.sh` is idempotent — re-run it to resume a half-finished
install.

## Manual install

```bash
conda create -n egosmith python=3.10 -y
conda activate egosmith

conda install -n egosmith -c nvidia cuda-toolkit=12.8 -y
pip install torch==2.8.0 torchvision==0.23.* torchaudio==2.8.* --index-url https://download.pytorch.org/whl/cu128
pip install xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
bash scripts/setup/fetch_chumpy.sh
pip install -r requirements.txt              # core + Any4D normal deps

# Any4D packages that must skip dependency resolution (rerun-sdk wants numpy>=2,
# UniCeption pulls broad torch) — install with --no-deps after the runtime is pinned:
pip install --no-deps \
  rerun-sdk==0.23.4 \
  "uniception @ git+https://github.com/JayKarhade/UniCeption.git@dev/any4d" \
  "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
  -e ./thirdparty/Any4D

# DPVO (source build) — point CUDA_HOME at the env toolkit
( cd thirdparty/DPVO && mkdir -p thirdparty && \
  wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip -O thirdparty/eigen-3.4.0.zip && \
  unzip -q -o thirdparty/eigen-3.4.0.zip -d thirdparty/ )
cd thirdparty/DPVO && CUDA_HOME="$CONDA_PREFIX" PATH="$CONDA_PREFIX/bin:$PATH" pip install . --no-build-isolation && cd ../..
```

Then obtain the HaWoR base and weights, and verify (see the README "Installation" section):

```bash
git submodule update --init thirdparty/hawor_upstream && bash scripts/setup/fetch_hawor_base.sh
bash scripts/setup/download_weights.sh
bash scripts/setup/validate_setup.sh
```

`pip install -e .` (optional) only sets up import resolution (src-layout, `where = ["src"]`); the
runtime stack is provisioned by `setup_env.sh` / the requirements files.

## Compatibility notes

- **NumPy** — keep `numpy==1.26.4`. MANO's legacy pickle files still need `chumpy`, so
  `scripts/setup/setup_env.sh` (and the manual `scripts/setup/fetch_chumpy.sh`) materialize a patched install
  copy from `thirdparty/chumpy_upstream` plus `patches/chumpy/setup.py.patch`.
- **CUDA wheels** — override the wheel index per stack if your driver differs, e.g.
  `HAWOR_CUDA=cu126 bash scripts/setup/setup_env.sh`.

## Troubleshooting

- **`libffi` symbol error during clipping / video decoding** — some systems load the wrong `libffi`
  for OpenCV/FFmpeg/decord. Export the right one before running:

  ```bash
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
  ```

  When invoking `src/lib/clip/run_heuristic_video_clipper.sh` directly, set
  `HAWOR_CLIP_LD_PRELOAD=/path/to/libffi.so.7` if your path differs.

## Reusing an already-provisioned checkout

A fresh clone / `git worktree` can borrow the gitignored runtime deps (weights, MANO, Any4D/DPVO
checkpoints, the upstream Any4D `demo_inference.py`) from an existing set-up checkout instead of
re-downloading:

```bash
git submodule update --init thirdparty/hawor_upstream
bash scripts/setup/fetch_hawor_base.sh
HAWOR_SRC=/path/to/provisioned/checkout bash scripts/setup/provision_worktree.sh
export HAWOR_BATCH_TMPDIR=/large/disk/tmp     # stage-3 scratch root (never defaulted)
```

`provision_worktree.sh` only symlinks into this checkout (everything it links is gitignored) and
never writes to the source. The pipeline preflight then reports any remaining missing
weight / MANO / scratch-root before GPU work.
