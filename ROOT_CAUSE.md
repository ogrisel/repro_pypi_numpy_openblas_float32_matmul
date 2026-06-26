# Root cause analysis: float32 `A @ A.T` on macOS arm64 (scipy-openblas64)

See also [BUG_REPORT.md](BUG_REPORT.md) for a filing-ready summary and
[README.md](README.md) for the full reproduction harness and collected results.

## Executive summary

The bug is **not** a generic failure of float32 `@` or of OpenBLAS SGEMM on this stack.
It is a **three-way interaction**:

1. **NumPy's `@` fast path** for `A @ A.T` when `A.T` is an **F-contiguous view** of
   C-contiguous `A` (always true for `A.T` in this repro).
2. That path calls **`cblas_ssyrk`** (symmetric rank-k update), **not** `cblas_sgemm`.
3. OpenBLAS **`vortexm4`** (Apple M4 dispatch) **float32 SSYRK** returns wrong results.

Direct SGEMM with correct parameters works. `einsum(..., optimize=False)` works.
Accelerate and conda LP64 OpenBLAS work. Forcing `OPENBLAS_CORETYPE=NEOVERSEN1` fixes it.

**Root cause:** OpenBLAS 0.3.33.112.0 **`vortexm4` float32 `SSYRK`** is broken for the
call pattern NumPy uses in `A @ A.T`.

---

## Symptom recap

```python
C = A @ A.T   # A.shape == (300, 672), float32
```

| Path | macOS arm64 scipy-openblas64 wheel |
|---|---|
| `A @ A.T` / `np.matmul(A, A.T)` | **FAIL** (overflow warnings; max ~0, ~1, ~1.6×10³⁸, or NaNs) |
| `np.einsum('ik,jk->ij', A, A, optimize=False)` | OK (~185.704) |
| `np.einsum('ik,jk->ij', A, A, optimize=True)` | **FAIL** (routes through `matmul`) |
| ctypes `scipy_cblas_sgemm64_` (NT on `A`) | OK (~185.704) |
| `float64` `@` | OK |
| `np.matmul(A, A.T.copy())` | OK |
| conda-forge / Accelerate `@` | OK |

---

## Hypothesis testing

| # | Hypothesis | Experiment | Result |
|---|---|---|---|
| H1 | OpenBLAS SGEMM kernel is broken | ctypes `scipy_cblas_sgemm64_` (NoTrans, Trans on `A`) | **Rejected** — ~185.7 |
| H2 | Threading race | `OPENBLAS_NUM_THREADS=1/4/10` | **Rejected** — still fails |
| H3 | float32-specific | `float64 @` vs `float32 @` | **Partial** — float64 OK |
| H4 | All `@`-like paths broken | `@`, `dot`, `tensordot`, `inner`, `matmul` | **Confirmed** — all BLAS matmul paths fail |
| H5 | `einsum` uses different code | `optimize=False` vs `True` | **Confirmed** — only `optimize=True` fails |
| H6 | F-contiguous RHS (`A.T` view) | `matmul(A, A.T)` vs `matmul(A, A.T.copy())` | **Confirmed trigger** |
| H7 | M4 `vortexm4` kernel | `OPENBLAS_CORETYPE=VORTEXM4` vs `NEOVERSEN1` | **Confirmed** — only vortexm4 fails |
| H8 | NumPy calls SGEMM for `@` | Read NumPy 2.5.0 `cblasfuncs.c` | **Rejected** — uses **SYRK** for `A @ A.T` |
| H9 | SSYRK broken on vortexm4 | Direct ctypes `scipy_cblas_ssyrk64_` | **Confirmed root cause** |

---

## The layout trigger

For C-contiguous `A` with shape `(300, 672)`:

- `A.T` has shape `(672, 300)`, is **F-contiguous**, and shares **the same buffer** as `A`.
- `np.matmul(A, A.T)` → **FAIL**
- `np.matmul(A, A.T.copy())` → **OK** (C-contiguous copy → different buffer → no SYRK shortcut)

Cross-stack check on `matmul(A, A.T)` with the F-contiguous view:

| Stack | Result |
|---|---|
| PyPI scipy-openblas64 (`macosx_11_0_arm64`) | **FAIL** |
| PyPI Accelerate (`macosx_14_0_arm64`) | OK |
| conda-forge libopenblas (LP64) | OK |

Even a `(4, 5)` float32 matrix fails with the F-view on scipy-openblas64; the C-copy fix
works at all tested sizes.

---

## What NumPy actually calls

NumPy 2.5.0 [`cblasfuncs.c`](https://github.com/numpy/numpy/blob/v2.5.0/numpy/_core/src/common/cblasfuncs.c)
detects `matrix @ matrix.T` when both operands share storage with transpose-compatible
strides and takes a **SYRK fast path** instead of GEMM:

```c
if (same buffer && transpose stride pattern && one operand Trans) {
    syrk(..., CblasUpper, Trans1, N, M, ap1, lda, out_buf);
} else {
    gemm(...);
}
```

For our repro (`A` C-contiguous, `A.T` F-contiguous view, same buffer):

```
cblas_ssyrk(RowMajor, Upper, NoTrans, n=300, k=672, A, lda=672, C, ldc=300)
```

via the ILP64 symbol `scipy_cblas_ssyrk64_`.

**Why `einsum` works:** `einsum('ik,jk->ij', ..., optimize=False)` uses the native
`c_einsum` contraction loop, not `matmul`/SYRK.

**Why the repro's ctypes SGEMM check passes:** `repro.py` deliberately calls SGEMM
(NoTrans, Trans) on `A` twice, bypassing SYRK entirely.

**Why `A.T.copy()` works:** the copy has a different buffer, so NumPy falls through to
GEMM instead of SYRK.

---

## Smoking-gun experiment: direct SSYRK

Calling OpenBLAS with NumPy's SYRK parameters (row-major, upper, no-trans, `n=300`,
`k=672`, `lda=672`):

| `OPENBLAS_CORETYPE` | SSYRK max | `@` max | Correct (~185.7)? |
|---|---|---|---|
| **vortexm4** (M4 default) | **0.0** | **0.0 – 1.6×10³⁸** | **No** |
| **neoversen1** (forced) | 185.704 | 185.704 | Yes |

Additional checks on vortexm4:

- **float64 SSYRK** → OK
- **`A @ B.T`** with unrelated `B` (no self-transpose SYRK shortcut) → OK
- **GEMM** (NoTrans, Trans) via ctypes → OK

`numpy.show_runtime()` reports OpenBLAS dispatch architecture **`vortexm4`** on Apple M4
hardware with the scipy-openblas64 wheel.

---

## Final diagnosis

> **OpenBLAS 0.3.33.112.0 `vortexm4` float32 `SSYRK` is broken** for row-major Upper
> NoTrans calls of the form NumPy issues for `A @ A.T` on C-contiguous float32 `A`.
>
> NumPy's `@` detects the symmetric product and calls **`cblas_ssyrk`**, not
> **`cblas_sgemm`**. On Apple M4, OpenBLAS selects the **`vortexm4`** kernel, whose
> float32 SSYRK produces garbage. Float32 SGEMM and float64 SSYRK are unaffected.

This is related to other Apple M4 OpenBLAS SME kernel issues
([OpenBLAS #5414](https://github.com/OpenBLAS/OpenBLAS/issues/5414),
[#5528](https://github.com/OpenBLAS/OpenBLAS/issues/5528),
[#5429](https://github.com/OpenBLAS/OpenBLAS/issues/5429)), but the broken routine here
is specifically **SSYRK**, which explains why direct SGEMM and `einsum` looked fine while
`@` failed.

Downstream: [scikit-learn#34191](https://github.com/scikit-learn/scikit-learn/issues/34191)
(Ridge / Nystroem float32 failures on macOS with scipy-openblas64).

---

## Workarounds

| Workaround | Mechanism |
|---|---|
| `np.einsum('ik,jk->ij', A, A, optimize=False)` | Avoids SYRK / matmul |
| `A @ A.T.copy()` | Breaks same-buffer SYRK detection → GEMM |
| Cast to `float64` | Uses DSYRK (works on vortexm4) |
| `OPENBLAS_CORETYPE=NEOVERSEN1` | Avoids vortexm4 kernel |
| Install Accelerate wheel (`macosx_14_0_arm64`) | Different BLAS |
| conda-forge NumPy + libopenblas | LP64 stack; SYRK works |

---

## Suggested filing targets

| Project | What to file |
|---|---|
| **OpenBLAS** | float32 `SSYRK` broken on `vortexm4` (M4); row-major Upper NoTrans |
| **NumPy** | Consider disabling SYRK fast path for self-transpose products when linked against scipy-openblas64 on macOS arm64, or falling back to GEMM |
| **scipy-openblas64** | Ships the affected OpenBLAS 0.3.33.112.0 build |

---

## Reproduce the RCA experiments

C reproducer for OpenBLAS upstream: [openblas_ssyrk_repro/](openblas_ssyrk_repro/)

From the repo root on macOS arm64 with the `pypi-openblas` pixi environment:

```bash
pixi run -e pypi-openblas repro          # matmul FAIL, einsum/ctypes SGEMM OK

# F-view vs C-copy
pixi run -e pypi-openblas python -c "
import numpy as np, repro
A = repro.make_basis()
ref = np.einsum('ik,jk->ij', A, A, optimize=False)
print('F-view', np.allclose(A @ A.T, ref, rtol=1e-4))
print('C-copy', np.allclose(np.matmul(A, A.T.copy()), ref, rtol=1e-4))
"

# Core override
OPENBLAS_CORETYPE=NEOVERSEN1 pixi run -e pypi-openblas python repro_minimal.py
OPENBLAS_CORETYPE=VORTEXM4 pixi run -e pypi-openblas python repro_minimal.py
```
