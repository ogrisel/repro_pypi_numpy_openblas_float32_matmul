# OpenBLAS float32 SSYRK reproducer (vortexm4 / Apple M4)

Minimal **C** reproducer for the root cause identified in
[ROOT_CAUSE.md](../ROOT_CAUSE.md): OpenBLAS **0.3.33.112.0** **`vortexm4` float32
`SSYRK`** returns wrong results for the call pattern NumPy uses in `A @ A.T`.

Related: [NumPy #31776](https://github.com/numpy/numpy/issues/31776)

## What it does

1. Loads the same `(300, 672)` `float32` matrix `A` as the Python reproducer
2. Calls **`scipy_cblas_ssyrk64_`** (ILP64) with NumPy's parameters:

   ```
   SSYRK(RowMajor, Upper, NoTrans, n=300, k=672, A, lda=672, C, ldc=300)
   ```

3. Calls **`scipy_cblas_sgemm64_`** as a control (equivalent `A @ A.T` via GEMM)
4. Exit code **1** when SSYRK fails but SGEMM OK (expected on M4 + vortexm4)

Expected max element: **~185.7**.

## Build and run

**Requirements:** macOS arm64, Python 3 (to generate `testdata/A.bin`), a
`libscipy_openblas64_.dylib` (bundled in the PyPI NumPy `macosx_11_0_arm64` wheel).

```bash
cd openblas_ssyrk_repro
make

# Point at the OpenBLAS dylib from the PyPI NumPy wheel, e.g.:
export OPENBLAS_DYLIB="$(
  python3 -c "import numpy, pathlib; print(next(pathlib.Path(numpy.__file__).parent.joinpath('.dylibs').glob('libscipy_openblas64_*.dylib')))"
)"

./repro_ssyrk
echo exit_code=$?
```

Or pass paths explicitly:

```bash
./repro_ssyrk testdata/A.bin /path/to/libscipy_openblas64_.dylib
```

### Expected output on Apple M4 (vortexm4 default)

```
SSYRK ...  max = 0.000000 (or garbage)  => FAIL
SGEMM ...  max = 185.703522            => OK

Reproduced: SSYRK broken, SGEMM OK (matches NumPy issue).
exit_code=1
```

### Workaround check

```bash
OPENBLAS_CORETYPE=NEOVERSEN1 ./repro_ssyrk
# Both SSYRK and SGEMM should => OK, exit_code=0
```

## Files

| File | Purpose |
|---|---|
| `repro_ssyrk.c` | C reproducer (dlopen SSYRK/SGEMM from scipy-openblas64) |
| `generate_matrix.py` | Writes `testdata/A.bin` (seed 42, same as `repro.py`) |
| `Makefile` | Build helper |

## Upstream filing

Primary target: **OpenBLAS** — float32 `SSYRK` on **`vortexm4`** (M4 SME kernel).

Suggested title: *float32 SSYRK wrong on ARM64 VORTEXM4 (RowMajor, Upper, NoTrans)*
