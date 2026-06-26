#!/usr/bin/env python3
"""Write the repro matrix A (300, 672) float32 row-major to testdata/A.bin."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

M = 300
K = 672
SEED = 42
OUT = Path(__file__).resolve().parent / "testdata" / "A.bin"


def make_basis() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    basis = np.ascontiguousarray(rng.random((M, K), dtype=np.float32))
    basis[:, :400] = (basis[:, :400] > 0.85).astype(np.float32)
    return basis


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    a = make_basis()
    assert a.shape == (M, K)
    assert a.dtype == np.float32
    assert a.flags.c_contiguous
    OUT.write_bytes(a.tobytes())
    ref = float(np.einsum("ik,jk->ij", a, a).max())
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"Reference max(A @ A.T) via einsum: {ref:.6f}")


if __name__ == "__main__":
    main()
