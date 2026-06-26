# NumPy 2.5.0 float32 `@` matmul bug on macOS arm64 (scipy-openblas64 PyPI wheel)

## Summary

On **macOS arm64**, the PyPI NumPy **2.5.0** wheel tagged `macosx_11_0_arm64` — which
bundles **`libscipy_openblas64_`** from **`scipy-openblas64`** (OpenBLAS 0.3.33.112.0,
ILP64 `scipy_cblas_sgemm64_`) — can return **wrong results** for a routine float32
matrix multiply:

```python
C = A @ A.T   # A.shape == (300, 672), float32, values in [0, 1]
```

Expected max element: **~185.7**. Observed with `@`: overflow warnings and output that
can be ~0, ~1.6×10³⁸, or NaNs (non-deterministic). The same arrays via
`np.einsum('ik,jk->ij', A, A)` return the correct answer.

**Scope:** macOS arm64 + scipy-openblas64 PyPI wheel only. The sibling
`macosx_14_0_arm64` wheel (Apple **Accelerate**) does not reproduce. Linux x86_64 with
the same scipy-openblas stack passes. conda-forge NumPy 2.5.0 (LP64 `cblas_sgemm`) passes
on all tested platforms.

Full reproduction harness, collected JSON results, and cross-stack comparison:
**https://github.com/ogrisel/repro_pypi_numpy_openblas_float32_matmul**

Detailed root cause analysis: [ROOT_CAUSE.md](ROOT_CAUSE.md)

---

## Minimal reproduction (pip only)

**Requirements:** macOS arm64, Python 3.13.

> **Note:** `pip install numpy==2.5.0` on macOS 14+ selects the `macosx_14_0_arm64`
> Accelerate wheel and does **not** reproduce. Pin the scipy-openblas64 wheel explicitly.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install \
  "https://files.pythonhosted.org/packages/85/4b/953118a730ee3b35e28645e0eb4cf9beec5bdbb954e1ac2f5fcefba6bbc3/numpy-2.5.0-cp313-cp313-macosx_11_0_arm64.whl" \
  threadpoolctl
python repro_minimal.py
```

```python
# repro_minimal.py
import warnings

import numpy as np

rng = np.random.default_rng(42)
A = np.ascontiguousarray(rng.random((300, 672), dtype=np.float32))
A[:, :400] = (A[:, :400] > 0.85).astype(np.float32)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    C = A @ A.T

matmul_max = float(np.nanmax(C))
einsum_max = float(np.nanmax(np.einsum("ik,jk->ij", A, A)))

print(f"matmul max:  {matmul_max:.6f}  (expected ~185.7)")
print(f"einsum max:  {einsum_max:.6f}  (expected ~185.7)")
print(f"matmul NaNs: {int(np.isnan(C).sum())}")
print(f"warnings:    {[str(w.message) for w in caught]}")
```

**Expected (buggy):** `matmul max` far from 185.7, overflow/`invalid value` warnings,
while `einsum max ≈ 185.704`.

### Sample output (macOS 26.5.1 arm64, Python 3.13, pinned `macosx_11_0_arm64` wheel)

Environment (`import sys, numpy; print(numpy.__version__); print(sys.version)`):

```
2.5.0
3.13.7 (v3.13.7:bcee1c32211, Aug 14 2025, 19:10:51) [Clang 16.0.0 (clang-1600.0.26.6)]
```

Runtime (`import numpy; numpy.show_runtime()`; requires `threadpoolctl`):

```
[{'numpy_version': '2.5.0',
  'python': '3.13.7 (v3.13.7:bcee1c32211, Aug 14 2025, 19:10:51) [Clang 16.0.0 '
            '(clang-1600.0.26.6)]',
  'uname': uname_result(system='Darwin', node='fuligule.local', release='25.5.0', version='Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:26 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T8132', machine='arm64')},
 {'simd_extensions': {'baseline': ['NEON', 'NEON_FP16', 'NEON_VFPV4', 'ASIMD'],
                      'found': ['ASIMDHP', 'ASIMDDP'],
                      'not_found': ['ASIMDFHM']}},
 {'ignore_floating_point_errors_in_matmul': False},
 {'architecture': 'vortexm4',
  'filepath': '/private/tmp/numpy-bug-repro-pip/.venv/lib/python3.13/site-packages/numpy/.dylibs/libscipy_openblas64_.dylib',
  'internal_api': 'openblas',
  'num_threads': 10,
  'prefix': 'libscipy_openblas',
  'threading_layer': 'pthreads',
  'user_api': 'blas',
  'version': '0.3.33.112.0'}]
```

`repro_minimal.py`:

```
matmul max:  1.000000  (expected ~185.7)
einsum max:  185.703644  (expected ~185.7)
matmul NaNs: 0
warnings:    ['divide by zero encountered in matmul', 'overflow encountered in matmul', 'invalid value encountered in matmul']
```

Wrong `@` output can vary run-to-run (e.g. max ~1, ~0, ~1.6×10³⁸, or some NaNs); the
`einsum` path is consistently ~185.704.

### Control: sibling `macosx_14_0_arm64` wheel (Accelerate, same host)

```
matmul max:  185.703522  (expected ~185.7)
einsum max:  185.703644  (expected ~185.7)
matmul NaNs: 0
warnings:    []
```

---

## What fails vs what works

| Code path | macOS arm64 (`macosx_11_0` wheel) | Linux x86_64 (PyPI scipy-openblas) |
|---|---|---|
| `A @ A.T` | **FAIL** | OK |
| `np.einsum('ik,jk->ij', A, A)` | OK | OK |
| Direct CBLAS SGEMM (ctypes) | OK | OK |
| conda-forge NumPy + libopenblas | OK | OK |

The failure is in **NumPy's `@` matmul SYRK fast path** (`cblas_ssyrk`, not SGEMM) when
using the **scipy-openblas64** PyPI wheel on **macOS arm64** with OpenBLAS **`vortexm4`**
dispatch. See [ROOT_CAUSE.md](ROOT_CAUSE.md).

---

## Analysis (high level)

1. **Two macOS arm64 wheels, one bug.** NumPy 2.5.0 publishes two cp313 arm64 wheels:
   `macosx_11_0_arm64` (bundled scipy-openblas64) and `macosx_14_0_arm64` (Accelerate).
   Only the former reproduces on the same hardware.

2. **Platform-specific, not universal PyPI breakage.** The scipy-openblas64 stack passes
   on Linux x86_64 at the same NumPy version. Likely interaction between NumPy's matmul
   dispatch, the ILP64 scipy-openblas64 wrapper, and the arm64 OpenBLAS build
   (`DYNAMIC_ARCH`, `neoversen1`).

3. **Not an OpenBLAS SGEMM bug — it is float32 SSYRK on `vortexm4`.** Direct
   `scipy_cblas_sgemm64_` and `einsum` succeed. NumPy `@` for `A @ A.T` calls
   `cblas_ssyrk`, which fails on M4 OpenBLAS dispatch. See [ROOT_CAUSE.md](ROOT_CAUSE.md).

4. **Downstream impact.** Any code using float32 `@` on macOS with the scipy-openblas64
   wheel is at risk (e.g. kernel matrices in scikit-learn's Nystroem / similar pipelines).
   Installers on macOS 14+ often pick the Accelerate wheel by default, which masks the bug.

---

## Collected results (2026-06-26)

| Stack | Platform | `@` matmul |
|---|---|---|
| PyPI scipy-openblas64 (`macosx_11_0_arm64`) | macOS arm64 | **FAIL** |
| PyPI Accelerate (`macosx_14_0_arm64`) | macOS arm64 | OK |
| PyPI scipy-openblas64 | Linux x86_64 | OK |
| conda-forge libopenblas / Accelerate / MKL | macOS arm64, Linux x86_64 | OK |

See the [full results tables and environment matrix](https://github.com/ogrisel/repro_pypi_numpy_openblas_float32_matmul#collected-results) in the repro repository.

---

## Environment details (affected wheel)

- **NumPy:** 2.5.0 (`numpy-2.5.0-cp313-cp313-macosx_11_0_arm64.whl`)
- **BLAS:** bundled `libscipy_openblas64_.dylib` (scipy-openblas 0.3.33.112.0)
- **OpenBLAS dispatch (`show_runtime`):** `vortexm4`, pthreads, 10 threads
- **CBAS symbol:** `scipy_cblas_sgemm64_` (ILP64)
- **OpenBLAS config:** `USE64BITINT DYNAMIC_ARCH NO_AFFINITY neoversen1 MAX_THREADS=64`
- **Tested host:** macOS 26.5.1 arm64 (Apple Silicon)

---

## Suggested filing target

**NumPy:** `@` matmul broken with scipy-openblas64 on **macOS arm64 only**;
`einsum` and direct CBLAS OK; Linux x86_64 OK; Accelerate sibling wheel OK.

Related downstream: [scikit-learn#34191](https://github.com/scikit-learn/scikit-learn/issues/34191).
