# Local patches to vendored DPVO

This is a vendored copy of [DPVO](https://github.com/princeton-vl/DPVO) (MIT
license; see `LICENSE`). It is **not** a pristine upstream checkout — we carry a
small number of build-compatibility patches that a plain git submodule of
upstream would not include. That is why DPVO is vendored rather than referenced
as a submodule. The patches are confined to build/compatibility plumbing; none
change the SLAM algorithm.

If you ever re-sync against upstream DPVO, re-apply (or re-verify) the following.

## Patches

### 1. lietorch dispatch for newer PyTorch
- **Files:** `dpvo/lietorch/src/lietorch_cpu.cpp`, `dpvo/lietorch/src/lietorch_gpu.cu`
- **What:** The `DISPATCH_GROUP_AND_FLOATING_TYPES` call sites passed
  `Tensor.type()` (`DeprecatedTypeProperties`), relying on an implicit
  `ScalarType` conversion that PyTorch 2.6 removed — this broke the lietorch
  extension build. All 38 CPU/GPU call sites now pass `.scalar_type()`.
- **Why:** Without it the lietorch extension fails to compile on PyTorch ≥ 2.6.
  Also clears the matching deprecation warnings on older versions.

### 2. altcorr CUDA kernel dispatch for newer PyTorch
- **File:** `dpvo/altcorr/correlation_kernel.cu`
- **What:** PyTorch 2.6 removed the implicit `Tensor.type()` →
  `c10::ScalarType` conversion that `AT_DISPATCH_FLOATING_TYPES_AND_HALF`
  relied on; pass `.scalar_type()` instead.
- **Why:** Without it the DPVO build fails to compile in the `any4d` env
  (PyTorch 2.6). The fix is correct on all supported torch versions.

## Notes
- DPVO bundles its **own** lietorch (`dpvo/lietorch/`, built as the
  `lietorch_backends` extension by `setup.py`); it does not depend on any
  external lietorch checkout.
- DPVO needs Eigen 3.4.0 at `thirdparty/DPVO/thirdparty/eigen-3.4.0`. It is not
  vendored — `scripts/setup/setup_env.sh` fetches it (see `ensure_eigen`).
- **Trimmed `DPViewer/` and `DPRetrieval/`** (not vendored here). EgoSmith runs DPVO
  headless (no real-time `DPViewer`) and uses only proximity (DPV-SLAM) loop closure,
  which goes through the main BA and needs no retrieval/DBoW — the classical
  `CLASSIC_LOOP_CLOSURE` that depends on `DPRetrieval` is intentionally not exposed
  (see `lib/pipeline/slam/dpvo_slam.py`). `scripts/setup/setup_env.sh` only `pip install .`s the
  DPVO core, so these were unused. Upstream's `README.md` still mentions installing
  them; ignore those steps.
