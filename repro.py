#!/usr/bin/env python
"""Reproducer for float32 matmul issues with PyPI NumPy + scipy-openblas.

Computes ``A @ A.T`` on a (300, 672) float32 matrix — the kernel step inside
Nystroem's polynomial kernel.  Also runs ctypes CBLAS and ``einsum`` checks
to compare NumPy code paths against a direct SGEMM call.

Requires only NumPy (ctypes is stdlib).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

N_COMPONENTS = 300
N_FEATURES = 672
RANDOM_STATE = 42
REFERENCE_MAX = 185.7
REFERENCE_RTOL = 1e-4

CBLAS_ROW_MAJOR = 101
CBLAS_NO_TRANS = 111
CBLAS_TRANS = 112


def make_basis() -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)
    basis = np.ascontiguousarray(
        rng.random((N_COMPONENTS, N_FEATURES), dtype=np.float32)
    )
    basis[:, :400] = (basis[:, :400] > 0.85).astype(np.float32)
    return basis


def _check_record(
    *,
    ok: bool,
    kernel_max: float | None = None,
    nan_count: int | None = None,
    warnings_list: list[str] | None = None,
    error: str | None = None,
    **extra,
) -> dict:
    record = {
        "ok": ok,
        "kernel_max": kernel_max,
        "nan_count": nan_count,
        "warnings": warnings_list or [],
        "error": error,
    }
    record.update(extra)
    return record


def check_matmul(basis: np.ndarray) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kernel = basis @ basis.T

    nan_count = int(np.isnan(kernel).sum())
    matmul_warnings = [
        str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)
    ]
    finite = kernel[np.isfinite(kernel)]
    kernel_max = float(finite.max()) if finite.size else None
    ok = nan_count == 0 and not matmul_warnings and _matches_reference(kernel_max)
    return _check_record(
        ok=ok,
        kernel_max=kernel_max,
        nan_count=nan_count,
        warnings_list=matmul_warnings,
    )


def check_einsum(basis: np.ndarray) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kernel = np.einsum("ik,jk->ij", basis, basis)

    nan_count = int(np.isnan(kernel).sum())
    matmul_warnings = [
        str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)
    ]
    kernel_max = float(kernel.max()) if nan_count == 0 else None
    ok = (
        nan_count == 0
        and not matmul_warnings
        and _matches_reference(kernel_max)
    )
    return _check_record(
        ok=ok,
        kernel_max=kernel_max,
        nan_count=nan_count,
        warnings_list=matmul_warnings,
    )


def _matches_reference(kernel_max: float | None) -> bool:
    if kernel_max is None:
        return False
    return abs(kernel_max - REFERENCE_MAX) <= REFERENCE_RTOL * REFERENCE_MAX


def _umath_extension_path() -> Path:
    import numpy._core._multiarray_umath as umath

    return Path(umath.__file__).resolve()


def _resolve_loader_path(reference: Path, candidate: str) -> Path | None:
    if candidate.startswith("@loader_path/"):
        return (reference.parent / candidate.removeprefix("@loader_path/")).resolve()
    if candidate.startswith("@rpath/"):
        rel = candidate.removeprefix("@rpath/")
        search_roots = []
        if prefix := os.environ.get("CONDA_PREFIX"):
            search_roots.append(Path(prefix) / "lib")
        search_roots.extend(
            Path(p) for p in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":")
            if p
        )
        search_roots.append(reference.parent.parent.parent)  # .../envs/<name>/lib
        for root in search_roots:
            resolved = (root / rel).resolve()
            if resolved.is_file():
                return resolved
        return None
    path = Path(candidate)
    return path if path.is_file() else None


def _linked_blas_library() -> Path | None:
    umath_path = _umath_extension_path()
    try:
        if sys.platform == "darwin":
            output = subprocess.check_output(
                ["otool", "-L", str(umath_path)], text=True, stderr=subprocess.DEVNULL
            )
            for line in output.splitlines()[1:]:
                candidate = line.split()[0]
                if "blas" not in candidate.lower():
                    continue
                resolved = _resolve_loader_path(umath_path, candidate)
                if resolved is not None:
                    return resolved
        if sys.platform.startswith("linux"):
            output = subprocess.check_output(
                ["ldd", str(umath_path)], text=True, stderr=subprocess.DEVNULL
            )
            for line in output.splitlines():
                lowered = line.lower()
                if "blas" not in lowered and "openblas" not in lowered:
                    continue
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "=>":
                    return Path(parts[2])
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None
    return None


def _load_sgemm(lib: ctypes.CDLL) -> tuple[str, object]:
    try:
        fn = lib.cblas_sgemm
    except AttributeError:
        pass
    else:
        fn.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        fn.restype = None
        return "cblas_sgemm", fn

    for name in ("scipy_cblas_sgemm64_", "cblas_sgemm_64"):
        try:
            fn = getattr(lib, name)
        except AttributeError:
            continue
        fn.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_longlong,
        ]
        fn.restype = None
        return name, fn

    raise AttributeError("No supported SGEMM symbol found in BLAS library")


def check_ctypes_sgemm(basis: np.ndarray) -> dict:
    library = _linked_blas_library()
    if library is None or not library.is_file():
        return _check_record(
            ok=False,
            error=f"could not locate BLAS library linked from {_umath_extension_path()}",
        )

    try:
        lib = ctypes.CDLL(str(library))
        symbol, sgemm = _load_sgemm(lib)
    except OSError as exc:
        return _check_record(ok=False, error=f"ctypes.CDLL failed: {exc}")

    m = n = N_COMPONENTS
    k = N_FEATURES
    lda = ldb = k
    ldc = n
    kernel = np.zeros((m, n), dtype=np.float32)

    try:
        if symbol.endswith("64_"):
            sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                m,
                n,
                k,
                1.0,
                basis.ctypes.data_as(ctypes.c_void_p),
                lda,
                basis.ctypes.data_as(ctypes.c_void_p),
                ldb,
                0.0,
                kernel.ctypes.data_as(ctypes.c_void_p),
                ldc,
            )
        else:
            sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                m,
                n,
                k,
                1.0,
                basis.ctypes.data_as(ctypes.c_void_p),
                lda,
                basis.ctypes.data_as(ctypes.c_void_p),
                ldb,
                0.0,
                kernel.ctypes.data_as(ctypes.c_void_p),
                ldc,
            )
    except Exception as exc:
        return _check_record(
            ok=False,
            error=f"{symbol} raised {type(exc).__name__}: {exc}",
            library=str(library),
            symbol=symbol,
        )

    nan_count = int(np.isnan(kernel).sum())
    kernel_max = float(kernel.max()) if nan_count == 0 else None
    ok = nan_count == 0 and _matches_reference(kernel_max)
    return _check_record(
        ok=ok,
        kernel_max=kernel_max,
        nan_count=nan_count,
        library=str(library),
        symbol=symbol,
    )


def _numpy_blas_info() -> dict:
    info = np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
    umath_path = _umath_extension_path()
    linked = _linked_blas_library()
    return {
        "name": info.get("name"),
        "version": info.get("version"),
        "openblas_configuration": info.get("openblas configuration"),
        "umath_extension": str(umath_path),
        "linked_library": str(linked) if linked is not None else None,
    }


def run() -> dict:
    basis = make_basis()
    matmul = check_matmul(basis)
    einsum = check_einsum(basis)
    ctypes_sgemm = check_ctypes_sgemm(basis)

    # User-visible bug: ``@`` matmul fails on the PyPI stack.
    reproduces = not matmul["ok"]

    return {
        "ok": matmul["ok"],
        "reproduces": reproduces,
        "numpy_version": np.__version__,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "reference_kernel_max": REFERENCE_MAX,
        "blas": _numpy_blas_info(),
        "checks": {
            "matmul": matmul,
            "einsum": einsum,
            "ctypes_sgemm": ctypes_sgemm,
        },
    }


def _print_human(result: dict) -> None:
    for name, check in result["checks"].items():
        status = "OK" if check["ok"] else "FAIL"
        line = f"{name}: {status}"
        if check.get("kernel_max") is not None:
            line += f", max={check['kernel_max']:.4f}"
        if check.get("nan_count") is not None:
            line += f", nan_count={check['nan_count']}"
        if check.get("symbol"):
            line += f", symbol={check['symbol']}"
        if check.get("warnings"):
            line += f", warnings={check['warnings'][:2]}"
        if check.get("error"):
            line += f", error={check['error']}"
        print(line)

    blas = result["blas"]
    print()
    print(f"NumPy BLAS: {blas.get('name')} {blas.get('version') or ''}".strip())
    if blas.get("openblas_configuration"):
        print(f"OpenBLAS config: {blas['openblas_configuration']}")
    if blas.get("linked_library"):
        print(f"Linked library: {blas['linked_library']}")
    print()
    print(f"Reproduces matmul bug: {'yes' if result['reproduces'] else 'no'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print result as JSON.")
    args = parser.parse_args()

    result = run()

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_human(result)

    return 1 if result["reproduces"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
