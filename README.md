# float32 matmul reproducer (PyPI NumPy + scipy-openblas)

## The bug

On **macOS arm64**, the PyPI NumPy 2.5.0 wheel tagged `macosx_11_0_arm64` — which
bundles **`libscipy_openblas64_`** from the **`scipy-openblas64`** PyPI package
(OpenBLAS 0.3.33.112.0, ILP64 `scipy_cblas_sgemm64_`) — matrix multiplication with
`@` can produce **wrong results** for a float32 GEMM that should be well within range:

```python
C = A @ A.T   # A.shape == (300, 672), float32, values in [0, 1]
```

The sibling PyPI wheel `macosx_14_0_arm64` (linked against **Apple Accelerate**, not
scipy-openblas) does **not** reproduce this on the same machine. On **Linux x86_64**,
the scipy-openblas64 wheel passes all checks (see [Collected results](#collected-results)).

Symptoms with the affected wheel:

- **Overflow warnings** (`divide by zero encountered in matmul`, etc.)
- **Incorrect output** — max element should be ~185.7, but `@` can return ~0,
  garbage (e.g. ~1.6×10³⁸), or NaNs; results can vary run-to-run
- **Same operation via other paths works** — `np.einsum('ik,jk->ij', A, A)` and a
  direct `scipy_cblas_sgemm64_` call on the same array return the correct answer (~185.7035)

**NumPy from conda-forge** (linked against `libopenblas` or Apple Accelerate, LP64
`cblas_sgemm`) does not show this: `@` is deterministic and correct — even at the
same NumPy version (2.5.0).

The failure is in **NumPy's `@` matmul path** when using the **scipy-openblas64**-bundled
PyPI wheel on **macOS arm64**, not in the underlying OpenBLAS SGEMM kernel when called
correctly through CBLAS.

This repo reproduces and compares the three code paths above across both PyPI macOS
wheel variants, Linux PyPI, and conda-forge BLAS/NumPy combinations.

## Quick start

```bash
cd repro_pypi_numpy_openblas_float32_matmul
pixi install

# macOS arm64: both PyPI wheel variants side-by-side
pixi run -e pypi-openblas repro      # macosx_11_0 wheel → scipy-openblas64
pixi run -e pypi-accelerate repro    # macosx_14_0 wheel → Accelerate
pixi run -e openblas-pthreads repro  # conda-forge baseline (host platform)
pixi run --platform osx-arm64-macos-11-0 -e openblas-pthreads repro  # explicit 11.0 target

# Linux: generic PyPI wheel
pixi run -e pypi repro

# All compatible environments on this host → results/<env-name>-<platform>.json
# (openblas conda envs on macOS arm64 also write -osx-arm64-macos-11-0 / -15-5)
pixi run collect
```

## What `repro.py` checks

| Check | What it does | PyPI `pypi-openblas` (macOS) | PyPI `pypi` (Linux) | conda `openblas-*` env |
|---|---|---|---|---|
| `matmul` | ``A @ A.T`` | **FAIL** | OK (~185.7) | OK (~185.7) |
| `einsum` | ``np.einsum('ik,jk->ij', A, A)`` | OK | OK | OK |
| `ctypes_sgemm` | ``cblas_sgemm`` / ``scipy_cblas_sgemm64_`` via ctypes | OK | OK | OK |

Exit code **1** means the **matmul** check reproduces the bug (expected on
`pypi-openblas` on macOS arm64). The Accelerate control env (`pypi-accelerate`)
and Linux `pypi` should exit 0.

## Collected results

Latest runs (`pixi run collect`, 2026-06-26) — one JSON per run under
`results/` (e.g. `results/pypi-openblas-macos-arm64.json`,
`results/openblas-pthreads-osx-arm64-macos-11-0.json`):

### macOS arm64

Host: `macOS-26.5.1-arm64` (collected 2026-06-26).

| Environment | Pixi platform | NumPy | BLAS | `matmul` | `einsum` | `ctypes_sgemm` |
|---|---|---|---|---|---|---|
| `pypi-openblas` | `osx-arm64-macos-11-0` | 2.5.0 (`macosx_11_0_arm64`) | scipy-openblas 0.3.33.112.0, `scipy_cblas_sgemm64_` | **FAIL** (2257 NaNs, max 1.6×10³⁸, 3 overflow warnings) | OK (185.704) | OK (185.704) |
| `pypi-accelerate` | `osx-arm64-macos-15-5` | 2.5.0 (`macosx_14_0_arm64`) | Accelerate | OK (185.704) | OK (185.704) | N/A (framework not in `.dylibs`) |
| `openblas-pthreads` | `osx-arm64-macos-11-0` | 2.5.0 (conda) | libopenblas, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |
| `openblas-pthreads` | `osx-arm64-macos-15-5` | 2.5.0 (conda) | libopenblas, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |
| `openblas-openmp` | `osx-arm64-macos-11-0` | 2.5.0 (conda) | libopenblas, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |
| `openblas-openmp` | `osx-arm64-macos-15-5` | 2.5.0 (conda) | libopenblas, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |
| `newaccelerate` | `osx-arm64-macos-15-5` | 2.5.0 (conda) | Accelerate reexport, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |

### Linux x86_64

Host: `Linux-6.17.0-23-generic-x86_64-with-glibc2.39` (collected 2026-06-26).

| Environment | NumPy | BLAS | `matmul` | `einsum` | `ctypes_sgemm` |
|---|---|---|---|---|---|
| `pypi` | 2.5.0 (PyPI wheel) | scipy-openblas 0.3.33.112.0, `scipy_cblas_sgemm64_` | OK (185.704) | OK (185.704) | OK (185.704) |
| `openblas-pthreads` | 2.5.0 (conda) | libopenblas, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |
| `openblas-openmp` | 2.5.0 (conda) | libopenblas, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |
| `mkl` | 2.5.0 (conda) | Intel MKL, `cblas_sgemm` | OK (185.704) | OK (185.704) | OK (185.704) |

On **macOS arm64**, only the `macosx_11_0_arm64` PyPI wheel (bundled scipy-openblas64)
reproduces the bug; the sibling `macosx_14_0_arm64` Accelerate wheel does not.
**conda-forge libopenblas** (pthreads and openmp) passes on both declared macOS arm64
platforms (`osx-arm64-macos-11-0` and `osx-arm64-macos-15-5`) — the bug is not tied
to Pixi's macOS deployment-target setting. On **Linux x86_64**, the PyPI scipy-openblas
wheel passes all three checks. All conda-forge stacks (openblas, Accelerate, MKL) pass
on both platforms at NumPy 2.5.0.

Wrong `@` output on macOS PyPI can vary run-to-run (e.g. max ~0, ~1.6×10³⁸, or NaNs);
the latest `pypi-openblas` run returned 2257 NaNs with max 1.6×10³⁸.

## Pixi environments

macOS arm64 uses two workspace platforms: `osx-arm64-macos-11-0` (PyPI
`macosx_11_0_arm64` / scipy-openblas64; also used for conda openblas collection) and
`osx-arm64-macos-15-5` (PyPI `macosx_14_0_arm64` / Accelerate; conda newaccelerate).
`newaccelerate` and `mkl` remain on the 15.5 macOS platforms only.

| Environment | NumPy source | BLAS | Platforms | `matmul` expected |
|---|---|---|---|---|
| `pypi` | PyPI wheel | bundled scipy-openblas | Linux, Windows | OK |
| `pypi-openblas` | PyPI wheel (`macosx_11_0_arm64`) | bundled scipy-openblas64 | macOS arm64 (11.0) | **FAIL** |
| `pypi-accelerate` | PyPI wheel (`macosx_14_0_arm64`) | Accelerate | macOS arm64 (15.5) | OK |
| `openblas-pthreads` | conda-forge | `libopenblas` (pthreads) | all; both macOS arm64 variants | OK |
| `openblas-openmp` | conda-forge | `libopenblas` (openmp) | all; both macOS arm64 variants | OK |
| `newaccelerate` | conda-forge | Apple Accelerate | macOS (15.5) | OK |
| `mkl` | conda-forge | Intel MKL | linux-64, osx-64, win-64 | OK |

On macOS arm64, `pixi run collect` runs openblas conda envs twice with
`--platform osx-arm64-macos-11-0` and `--platform osx-arm64-macos-15-5`.

## Analysis

### Cross-platform summary (collected results 2026-06-26)

| Platform | PyPI scipy-openblas wheel `matmul` | PyPI Accelerate wheel `matmul` | PyPI `einsum` / `ctypes_sgemm` | conda-forge |
|---|---|---|---|---|
| macOS arm64 | **FAIL** (`pypi-openblas`) | OK (`pypi-accelerate`) | OK on openblas wheel | OK |
| Linux x86_64 | OK (`pypi`) | — | OK | OK |

The bug is **macOS arm64-specific** among tested platforms. The scipy-openblas64
PyPI wheel (same 0.3.33.112.0 build, same ILP64 `scipy_cblas_sgemm64_` symbol)
passes on Linux x86_64 but fails on macOS arm64. The Accelerate-linked sibling
wheel on the same host does not reproduce.

### Library comparison

| Platform | Build | CBLAS symbol | `matmul` | `ctypes_sgemm` |
|---|---|---|---|---|
| macOS arm64 | PyPI `libscipy_openblas64_.dylib` (`pypi-openblas`) | `scipy_cblas_sgemm64_` | FAIL | OK |
| macOS arm64 | PyPI Accelerate framework (`pypi-accelerate`) | — | OK | N/A |
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
   - On macOS arm64, only the `macosx_11_0_arm64` scipy-openblas64 wheel fails ``@``;
     the `macosx_14_0_arm64` Accelerate wheel and Linux x86_64 scipy-openblas wheel pass.
   - Both failing and passing macOS wheels are NumPy 2.5.0 cp313; the openblas wheel uses
     ILP64 `scipy_cblas_sgemm64_`.
   - Likely interaction between NumPy's matmul path, the scipy-openblas64 wrapper, and
     the **arm64 / neoverse-n1** OpenBLAS dispatch path (needs further isolation).

2. **conda-forge does not reproduce on either platform or macOS deployment target**
   - All conda stacks use LP64 `cblas_sgemm` (libopenblas, Accelerate, or MKL) and pass all checks.
   - conda `libopenblas` passes on both `osx-arm64-macos-11-0` and `osx-arm64-macos-15-5`
     (same NumPy 2.5.0 conda build in both runs).
   - PyPI `_multiarray_umath` imports `scipy_cblas_sgemm64_`; conda imports `cblas_sgemm`.
   - The ctypes SGEMM check tries LP64 `cblas_sgemm` first, then ILP64 symbols — matching each stack.

3. **Only NumPy `@` matmul is broken — and only on macOS arm64 scipy-openblas64 PyPI**
   - ``einsum`` and the ctypes SGEMM call return ~185.704 on the failing macOS wheel and
     on Linux x86_64 PyPI.
   - ``A @ A.T`` on `pypi-openblas` raises overflow warnings and returns wrong values
     (2257 NaNs and max 1.6×10³⁸ in the latest collected run; can also be ~0 or vary run-to-run).
   - ``@`` on `pypi-accelerate`, Linux x86_64 `pypi`, and every conda-forge stack is correct.

### Where to file bugs

| Audience | What to file |
|---|---|
| **OpenBLAS upstream** | `ctypes_sgemm` shows SGEMM is correct via CBLAS; likely out of scope unless scipy's fork is maintained upstream. |
| **scipy-openblas** | PyPI fork + ILP64 API; direct call OK; NumPy matmul path broken on macOS arm64 only. |
| **NumPy** | ``@`` matmul broken with scipy-openblas64 on **macOS arm64 only** (Linux x86_64 OK); ``einsum`` OK on both. |
