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
