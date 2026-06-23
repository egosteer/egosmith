#!/bin/bash
# Validate the conda runtime used by the dataset pipeline.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="egosmith"

echo "=== EgoSmith / HaWoR Setup Validation ==="
echo "Project root : $PROJECT_ROOT"
echo "EgoSmith env    : $ENV_NAME"
echo ""

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not on PATH. The pipeline runtime resolver expects conda, mamba, or micromamba env discovery."
    exit 1
fi

echo "Checking conda environment..."
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "ERROR: conda env '$ENV_NAME' was not found."
    echo "Create it with the README Environment Setup instructions."
    exit 1
fi
echo "OK: required conda env exists"
echo ""

echo "Checking patched chumpy source..."
if [ ! -f thirdparty/chumpy/setup.py ]; then
    echo "ERROR: patched chumpy source not materialized."
    echo "Run: bash scripts/setup/fetch_chumpy.sh"
    exit 1
fi
if grep -q "parse_requirements" thirdparty/chumpy/setup.py; then
    echo "ERROR: thirdparty/chumpy still has upstream setup.py."
    echo "Run: bash scripts/setup/fetch_chumpy.sh"
    exit 1
fi
echo "OK: patched chumpy source present"
echo ""

echo "Checking HaWoR Python packages..."
conda run -n "$ENV_NAME" python - <<'PY'
import sys

missing = []
for module in (
    "torch",
    "xformers",
    "numpy",
    "cv2",
    "joblib",
    "webdataset",
    "natsort",
    "ultralytics",
):
    try:
        __import__(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")

if missing:
    print("ERROR: missing HaWoR packages:")
    for item in missing:
        print("  -", item)
    raise SystemExit(1)

import torch
import numpy as np
try:
    import chumpy  # noqa: F401
except Exception as exc:
    raise SystemExit(f"ERROR: chumpy import failed; reinstall from ./thirdparty/chumpy: {exc}")
if np.__version__ != "1.26.4":
    raise SystemExit(f"ERROR: expected numpy 1.26.4, got {np.__version__}")

print("python:", sys.executable)
print("torch:", torch.__version__)
print("numpy:", np.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
PY
echo ""

echo "Checking Any4D Python packages and configured paths..."
conda run -n "$ENV_NAME" python - <<'PY'
import sys
from pathlib import Path

missing = []
for module in (
    "torch",
    "cv2",
    "hydra",
    "numba",
    "natsort",
    "PIL",
    "tqdm",
    "joblib",
    "pycocotools",
    "evo",
    "torchmin",
):
    try:
        __import__(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")

try:
    import any4d  # noqa: F401
except Exception as exc:
    missing.append(f"any4d: {exc}")

if missing:
    print("ERROR: missing Any4D packages:")
    for item in missing:
        print("  -", item)
    raise SystemExit(1)

import torch
from lib.pipeline.slam.any4d_depth import resolve_any4d_paths

repo_root, checkpoint_path, resolution, use_amp = resolve_any4d_paths()
demo_inference = Path(repo_root) / "scripts" / "demo_inference.py"
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
print("any4d_repo_root:", repo_root)
print("any4d_checkpoint:", checkpoint_path)
print("any4d_resolution:", resolution)
print("any4d_use_amp:", use_amp)

if not Path(repo_root).is_dir():
    raise SystemExit(f"ERROR: Any4D repo root not found: {repo_root}")
if not Path(checkpoint_path).is_file():
    raise SystemExit(f"ERROR: Any4D checkpoint not found: {checkpoint_path}")
if not demo_inference.is_file():
    raise SystemExit(f"ERROR: Any4D demo_inference.py not found: {demo_inference}")
PY
echo ""

echo "Checking SLAM CLI imports in Any4D env..."
conda run -n "$ENV_NAME" python scripts/batch_infer.py --help >/dev/null
conda run -n "$ENV_NAME" python - <<'PY'
import lib.pipeline.stages.slam  # noqa: F401
from dpvo.config import cfg  # noqa: F401
print("OK: slam stage imports")
PY
echo ""

echo "Checking model weights..."
missing_weights=()
for weight in \
    "./weights/hawor/checkpoints/hawor.ckpt" \
    "./weights/hawor/checkpoints/infiller.pt" \
    "./weights/hawor/model_config.yaml" \
    "./weights/external/detector.pt"; do
    if [ ! -f "$weight" ]; then
        missing_weights+=("$weight")
    fi
done

if [ "${#missing_weights[@]}" -gt 0 ]; then
    echo "ERROR: missing required HaWoR/WiLoR weights:"
    printf '  - %s\n' "${missing_weights[@]}"
    echo "Download them as described in README.md."
    exit 1
fi
echo "OK: required HaWoR/WiLoR weights found"
echo ""

echo "Checking MANO assets..."
missing_mano=()
for asset in \
    "./_DATA/data/mano/MANO_RIGHT.pkl" \
    "./_DATA/data_left/mano_left/MANO_LEFT.pkl" \
    "./_DATA/data/mano_mean_params.npz"; do
    if [ ! -f "$asset" ]; then
        missing_mano+=("$asset")
    fi
done
if [ "${#missing_mano[@]}" -gt 0 ]; then
    echo "ERROR: missing required MANO assets:"
    printf '  - %s\n' "${missing_mano[@]}"
    exit 1
fi
echo "OK: required MANO assets found"
echo ""

echo "Checking pipeline CLI imports in HaWoR env..."
conda run -n "$ENV_NAME" python scripts/run_dataset_pipeline.py --help >/dev/null
conda run -n "$ENV_NAME" python scripts/batch_infer.py --help >/dev/null
echo "OK: pipeline CLIs import"
echo ""

echo "Checking DPVO backend in HaWoR env..."
if conda run -n "$ENV_NAME" python -c "import dpvo" >/dev/null 2>&1; then
    echo "OK: DPVO backend importable"
else
    echo "WARN: DPVO backend is not importable. Build it: cd thirdparty/DPVO && pip install . --no-build-isolation"
fi
echo ""

echo "=== Setup Validation Complete ==="
echo "Next smoke test:"
echo "  conda run -n $ENV_NAME python scripts/run_dataset_pipeline.py --config configs/my_video.yaml --stages prepare"
echo "  conda run -n $ENV_NAME python scripts/run_dataset_pipeline.py --config configs/my_video.yaml"
