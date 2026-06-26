# float32 matmul reproducer (PyPI NumPy + scipy-openblas)

## The bug

On **NumPy installed from PyPI** (linked against bundled `libscipy_openblas64_`),
matrix multiplication with `@` can produce **wrong results** for a float32 GEMM that
should be well within range:

```python
C = A @ A.T   # A.shape == (300, 672), float32, values in [0, 1]
```

Observed on macOS arm64 with NumPy 2.5 + scipy-openblas 0.3.33.112.0:

- **Overflow warnings** (`divide by zero encountered in matmul`, etc.)
- **Incorrect output** — max element should be ~185.7, but `@` can return ~0,
  garbage (e.g. ~1.6×10³⁸), or NaNs; results can vary run-to-run
- **Same operation via other paths works** — `np.einsum('ik,jk->ij', A, A)` and a
  direct `cblas_sgemm` / `scipy_cblas_sgemm64_` call on the same array return the
  correct answer (~185.7035)

**NumPy from conda-forge** (linked against `libopenblas` or Apple Accelerate, LP64
`cblas_sgemm`) does not show this: `@` is deterministic and correct — even at the
same NumPy version (2.5.0).

The failure is in **NumPy's `@` matmul path** on the PyPI wheel stack, not in the
underlying OpenBLAS SGEMM kernel when called correctly through CBLAS.

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
| `matmul` | ``A @ A.T`` | **FAIL** (warnings / wrong max) | OK (~185.7) |
| `einsum` | ``np.einsum('ik,jk->ij', A, A)`` | OK | OK |
| `ctypes_sgemm` | ``cblas_sgemm`` / ``scipy_cblas_sgemm64_`` via ctypes | OK | OK |

Exit code **1** means the **matmul** check reproduces the bug (expected on `pypi`).

## Collected results

Latest run on macOS arm64 (`pixi run collect`, 2026-06-26) — one JSON per environment
under `results/` (e.g. `results/pypi-macos-arm64.json`):

| Environment | NumPy | BLAS | `matmul` | `einsum` | `ctypes_sgemm` |
|---|---|---|---|---|---|
| `pypi` | 2.5.0 (PyPI wheel) | scipy-openblas 0.3.33.112.0, `scipy_cblas_sgemm64_` | **FAIL** (4 NaNs, max 1.6×10³⁸) | OK (185.704) | OK (185.704) |
| `openblas-pthreads` | 2.5.0 (conda) | libopenblas, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |
| `openblas-openmp` | 2.5.0 (conda) | libopenblas, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |
| `newaccelerate` | 2.5.0 (conda) | Accelerate reexport, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |

Only the **PyPI NumPy wheel** (bundled scipy-openblas) reproduces the bug. All
conda-forge stacks pass all three checks at the same NumPy version, so the failure
is isolated to **how the PyPI wheel links and calls BLAS**, not NumPy 2.5 itself.

## Pixi environments

| Environment | NumPy source | BLAS | `matmul` expected |
|---|---|---|---|
| `pypi` | PyPI wheel | bundled `libscipy_openblas` | FAIL |
| `openblas-pthreads` | conda-forge | `libopenblas` (pthreads) | OK |
| `openblas-openmp` | conda-forge | `libopenblas` (openmp) | OK |
| `newaccelerate` | conda-forge | Apple Accelerate (macOS only) | OK |
| `mkl` | conda-forge | Intel MKL (linux-64, osx-64, win-64) | OK |

## Analysis

### Library comparison (macOS arm64, from collected results 2026-06-26)

| Build | NumPy | CBLAS symbol | `matmul` | `ctypes_sgemm` |
|---|---|---|---|---|
| PyPI `libscipy_openblas64_.dylib` | 2.5.0 wheel | `scipy_cblas_sgemm64_` | FAIL | OK |
| conda `libopenblas.0.dylib` | 2.5.0 conda-forge | `cblas_sgemm` | OK | OK |
| conda `libblas_reexport.dylib` (Accelerate) | 2.5.0 conda-forge | `cblas_sgemm` | OK | OK |

PyPI OpenBLAS config: `OpenBLAS 0.3.33.112.0 USE64BITINT DYNAMIC_ARCH NO_AFFINITY neoversen1 MAX_THREADS=64`.

### Why conda-forge does not reproduce the Python bug

1. **Same NumPy version, different BLAS packaging**
   - All environments above use **NumPy 2.5.0**; only the PyPI wheel fails.
   - PyPI embeds **scipy-openblas 0.3.33.112.0** (ILP64, `DYNAMIC_ARCH`).
   - conda-forge links **libopenblas 0.3.33** or **Apple Accelerate** (LP64 `cblas_sgemm`).

2. **Different NumPy → BLAS call path**
   - PyPI `_multiarray_umath.so` imports `_scipy_cblas_sgemm64_` (64-bit integer interface).
   - conda `_multiarray_umath.so` imports `_cblas_sgemm` (32-bit integer interface).

3. **Only NumPy `@` matmul is broken on the PyPI stack**
   - ``einsum`` and the ctypes SGEMM call return ~185.704 on PyPI.
   - ``A @ A.T`` on PyPI NumPy raises overflow warnings and can return NaNs or garbage.
   - The same ``@`` on every conda-forge stack is correct.

conda-forge "does not reproduce" because the PyPI wheel pairs NumPy 2.5 with the
**ILP64 scipy-openblas64** wrapper in a way that breaks ``@`` (but not ``einsum`` or
a correct direct SGEMM call).

### Where to file bugs

| Audience | What to file |
|---|---|
| **OpenBLAS upstream** | `ctypes_sgemm` shows SGEMM is correct via CBLAS; likely out of scope unless scipy's fork is maintained upstream. |
| **scipy-openblas** | PyPI fork + ILP64 API; direct call OK, NumPy matmul path broken. |
| **NumPy** | ``@`` matmul non-deterministic with scipy-openblas64 on macOS arm64; ``einsum`` OK. |
