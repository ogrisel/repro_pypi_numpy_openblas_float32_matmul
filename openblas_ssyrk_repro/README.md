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

**Requirements:** macOS arm64, Python 3.13, a C compiler (`cc`).

The affected OpenBLAS build is bundled in the PyPI NumPy **`macosx_11_0_arm64`**
wheel (scipy-openblas64 0.3.33.112.0). Plain `pip install numpy==2.5.0` on macOS 14+
installs the Accelerate wheel instead and does **not** ship this dylib.

### 1. Install the OpenBLAS dylib in a venv

```bash
cd openblas_ssyrk_repro

python3.13 -m venv .venv
source .venv/bin/activate
pip install \
  "https://files.pythonhosted.org/packages/85/4b/953118a730ee3b35e28645e0eb4cf9beec5bdbb954e1ac2f5fcefba6bbc3/numpy-2.5.0-cp313-cp313-macosx_11_0_arm64.whl"

export OPENBLAS_DYLIB="$(
  .venv/bin/python -c "import numpy, pathlib; print(next(pathlib.Path(numpy.__file__).parent.joinpath('.dylibs').glob('libscipy_openblas64_*.dylib')))"
)"
echo "OPENBLAS_DYLIB=$OPENBLAS_DYLIB"
```

Or use `make venv` (same steps):

```bash
make venv
source .venv/bin/activate
export OPENBLAS_DYLIB="$(.venv/bin/python -c "import numpy, pathlib; print(next(pathlib.Path(numpy.__file__).parent.joinpath('.dylibs').glob('libscipy_openblas64_*.dylib')))")"
```

### 2. Build and run the C reproducer

```bash
make
./repro_ssyrk
echo exit_code=$?
```

One-liner after the venv is set up:

```bash
make && ./repro_ssyrk testdata/A.bin "$OPENBLAS_DYLIB"
```

If `OPENBLAS_DYLIB` is exported, `./repro_ssyrk` alone is enough.

Or pass paths explicitly (no venv env var):

```bash
./repro_ssyrk testdata/A.bin .venv/lib/python3.13/site-packages/numpy/.dylibs/libscipy_openblas64_.dylib
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
| `Makefile` | Build helper (`make venv` installs the NumPy/OpenBLAS wheel) |

## Upstream filing

Primary target: **OpenBLAS** — float32 `SSYRK` on **`vortexm4`** (M4 SME kernel).

Suggested title: *float32 SSYRK wrong on ARM64 VORTEXM4 (RowMajor, Upper, NoTrans)*
