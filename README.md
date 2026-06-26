# float32 matmul reproducer (PyPI NumPy + scipy-openblas)

## The bug

On **NumPy installed from PyPI** (linked against bundled `libscipy_openblas64_`),
matrix multiplication with `@` can produce **wrong results** for a float32 GEMM that
should be well within range:

```python
C = A @ A.T   # A.shape == (300, 672), float32, values in [0, 1]
```

Observed on **macOS arm64** with NumPy 2.5 + scipy-openblas 0.3.33.112.0 (not on
Linux x86_64 with the same PyPI wheel — see [Collected results](#collected-results)):

- **Overflow warnings** (`divide by zero encountered in matmul`, etc.)
- **Incorrect output** — max element should be ~185.7, but `@` can return ~0,
  garbage (e.g. ~1.6×10³⁸), or NaNs; results can vary run-to-run
- **Same operation via other paths works** — `np.einsum('ik,jk->ij', A, A)` and a
  direct `cblas_sgemm` / `scipy_cblas_sgemm64_` call on the same array return the
  correct answer (~185.7035)

**NumPy from conda-forge** (linked against `libopenblas` or Apple Accelerate, LP64
`cblas_sgemm`) does not show this: `@` is deterministic and correct — even at the
same NumPy version (2.5.0).

The failure is in **NumPy's `@` matmul path** on the PyPI wheel stack on **macOS
arm64**, not in the underlying OpenBLAS SGEMM kernel when called correctly through CBLAS.

This repo reproduces and compares the three code paths above across PyPI and
conda-forge BLAS/NumPy combinations.

## Quick start

```bash
cd repro_pypi_numpy_openblas_float32_matmul
pixi install

# Single environment (runs matmul, einsum, and ctypes SGEMM checks)
pixi run -e pypi repro
pixi run -e openblas-pthreads repro

# All compatible environments on this host → results/<env-name>-<os>-<arch>.json
pixi run collect
```

## What `repro.py` checks

| Check | What it does | PyPI `pypi` env | conda `openblas-*` env |
|---|---|---|---|
| `matmul` | ``A @ A.T`` | **FAIL on macOS arm64**; OK on Linux x86_64 | OK (~185.7) |
| `einsum` | ``np.einsum('ik,jk->ij', A, A)`` | OK | OK |
| `ctypes_sgemm` | ``cblas_sgemm`` / ``scipy_cblas_sgemm64_`` via ctypes | OK | OK |

Exit code **1** means the **matmul** check reproduces the bug (expected on `pypi`
on macOS arm64; on Linux x86_64 `pypi` should exit 0).

## Collected results

Latest runs (`pixi run collect`, 2026-06-26) — one JSON per environment under
`results/` (e.g. `results/pypi-macos-arm64.json`):

### macOS arm64

Host: `macOS-26.5.1-arm64` (collected 2026-06-26).

| Environment | NumPy | BLAS | `matmul` | `einsum` | `ctypes_sgemm` |
|---|---|---|---|---|---|
| `pypi` | 2.5.0 (PyPI wheel) | scipy-openblas 0.3.33.112.0, `scipy_cblas_sgemm64_` | **FAIL** (4 NaNs, max 1.6×10³⁸, overflow warnings) | OK (185.704) | OK (185.704) |
| `openblas-pthreads` | 2.5.0 (conda) | libopenblas, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |
| `openblas-openmp` | 2.5.0 (conda) | libopenblas, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |
| `newaccelerate` | 2.5.0 (conda) | Accelerate reexport, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |

### Linux x86_64

Host: `Linux-6.17.0-23-generic-x86_64-with-glibc2.39` (collected 2026-06-26).

| Environment | NumPy | BLAS | `matmul` | `einsum` | `ctypes_sgemm` |
|---|---|---|---|---|---|
| `pypi` | 2.5.0 (PyPI wheel) | scipy-openblas 0.3.33.112.0, `scipy_cblas_sgemm64_` | OK (185.704) | OK (185.704) | OK (185.704) |
| `openblas-pthreads` | 2.5.0 (conda) | libopenblas, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |
| `openblas-openmp` | 2.5.0 (conda) | libopenblas, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |
| `mkl` | 2.5.0 (conda) | Intel MKL, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |

On **macOS arm64**, only the PyPI NumPy wheel (bundled scipy-openblas) reproduces
the bug. On **Linux x86_64**, the same PyPI wheel passes all three checks — the
failure is **platform-specific**, not universal to the PyPI scipy-openblas stack.
All conda-forge stacks (openblas, Accelerate, MKL) pass on both platforms at NumPy 2.5.0.

## Pixi environments

| Environment | NumPy source | BLAS | `matmul` expected |
|---|---|---|---|
| `pypi` | PyPI wheel | bundled `libscipy_openblas` | FAIL (macOS arm64); OK (Linux x86_64) |
| `openblas-pthreads` | conda-forge | `libopenblas` (pthreads) | OK |
| `openblas-openmp` | conda-forge | `libopenblas` (openmp) | OK |
| `newaccelerate` | conda-forge | Apple Accelerate (macOS only) | OK |
| `mkl` | conda-forge | Intel MKL (linux-64, osx-64, win-64) | OK |

## Analysis

### Cross-platform summary (collected results 2026-06-26)

| Platform | PyPI wheel `matmul` | PyPI wheel `einsum` / `ctypes_sgemm` | conda-forge (openblas / Accelerate / MKL) |
|---|---|---|---|
| macOS arm64 | **FAIL** | OK | OK |
| Linux x86_64 | OK | OK | OK |

The bug is **macOS arm64-specific** among tested platforms. The same PyPI NumPy
2.5.0 wheel (same scipy-openblas 0.3.33.112.0, same ILP64 `scipy_cblas_sgemm64_`
symbol) passes on Linux x86_64 but fails on macOS arm64.

### Library comparison

| Platform | Build | CBLAS symbol | `matmul` | `ctypes_sgemm` |
|---|---|---|---|---|
| macOS arm64 | PyPI `libscipy_openblas64_.dylib` | `scipy_cblas_sgemm64_` | FAIL | OK |
| macOS arm64 | conda `libopenblas.0.dylib` | `cblas_sgemm` | OK | OK |
| macOS arm64 | conda `libblas_reexport.dylib` (Accelerate) | `cblas_sgemm` | OK | OK |
| Linux x86_64 | PyPI `libscipy_openblas64_.so` | `scipy_cblas_sgemm64_` | OK | OK |
| Linux x86_64 | conda `libcblas.so.3` (openblas) | `cblas_sgemm` | OK | OK |
| Linux x86_64 | conda `libcblas.so.3` (MKL) | `cblas_sgemm` | OK | OK |

PyPI OpenBLAS config (DYNAMIC_ARCH target differs by host CPU):

- macOS arm64: `OpenBLAS 0.3.33.112.0 USE64BITINT DYNAMIC_ARCH NO_AFFINITY neoversen1 MAX_THREADS=64`
- Linux x86_64: `OpenBLAS 0.3.33.112.0 USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64`

### What the results show

1. **Platform-specific, not a universal PyPI wheel bug**
   - On macOS arm64, only the PyPI wheel fails ``@``; on Linux x86_64 the same wheel passes.
   - Both platforms use the same NumPy 2.5.0 PyPI wheel and ILP64 scipy-openblas64 API.
   - Likely interaction between NumPy's matmul path, the scipy-openblas64 wrapper, and
     the **arm64 / neoverse-n1** OpenBLAS dispatch path (needs further isolation).

2. **conda-forge does not reproduce on either platform**
   - All conda stacks use LP64 `cblas_sgemm` (libopenblas, Accelerate, or MKL) and pass all checks.
   - PyPI `_multiarray_umath` imports `scipy_cblas_sgemm64_`; conda imports `cblas_sgemm`.
   - The ctypes SGEMM check tries LP64 `cblas_sgemm` first, then ILP64 symbols — matching each stack.

3. **Only NumPy `@` matmul is broken — and only on macOS arm64 PyPI**
   - ``einsum`` and the ctypes SGEMM call return ~185.704 on PyPI on both platforms.
   - ``A @ A.T`` on macOS arm64 PyPI raises overflow warnings and returns wrong values
     (4 NaNs and max 1.6×10³⁸ in the latest collected run; can also be ~0 or vary run-to-run).
   - The same ``@`` on Linux x86_64 PyPI and on every conda-forge stack is correct.

### Where to file bugs

| Audience | What to file |
|---|---|
| **OpenBLAS upstream** | `ctypes_sgemm` shows SGEMM is correct via CBLAS; likely out of scope unless scipy's fork is maintained upstream. |
| **scipy-openblas** | PyPI fork + ILP64 API; direct call OK; NumPy matmul path broken on macOS arm64 only. |
| **NumPy** | ``@`` matmul broken with scipy-openblas64 on **macOS arm64 only** (Linux x86_64 OK); ``einsum`` OK on both. |
