"""End-to-end differential tests for `sglang_tpu/rl/true_on_policy/matmul_kernel.py`."""

from __future__ import annotations

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from conftest import load_kernel_module, tpu_interpret_mode

# Helper module from the same repository (at the manifest-pinned commit) providing `row_block`.
HELPER_PATH = "sglang_tpu/rl/true_on_policy/common.py"


@pytest.fixture(scope="module")
def mm_mod():
    return load_kernel_module(
        "sglang_tpu/rl/true_on_policy/matmul_kernel.py",
        "_tkl_e2e_true_on_policy_matmul",
        stub_torch=True,
        stub_sglang=True,
        helper_paths=(HELPER_PATH,),
    )


def _matmul_exact_ref(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    # Formula from TpuKernelLean.Examples.TrueOnPolicyMatmul.matmulElem_exact:
    # out[r, c] = sum_{k=0}^{nk*BK - 1} x[r, k] * w[c, k]  (x [M, K] @ w [N, K]^T).
    return x.astype(np.float64) @ w.astype(np.float64).T


@pytest.mark.parametrize(
    "m, n, k",
    [
        # Exact multiple of tile sizes with 2 K-steps (BK = 512, nk = 2)
        (16, 128, 1024),
        # Edge blocks on both M and N (non-multiples of bm and bn) across 2 K-steps
        (13, 135, 1024),
        # Multi-tile M and N with partial trailing blocks on both axes
        (135, 263, 1024),
    ],
)
def test_matmul_exact_integers_matches_matmulElem_exact(mm_mod, m, n, k):
    rng = np.random.default_rng(400 + m + n)
    x_np = np.zeros((m, k), dtype=np.float32)
    w_np = np.zeros((n, k), dtype=np.float32)
    # At most 16 nonzero entries per row in {-2, -1, 1, 2}: every product lies in
    # [-4, 4] and every partial sum and final dot product lies in [-64, 64], which
    # are all exactly representable in both fp32 and bf16 (integers up to 256 are
    # exact in bf16). Therefore the kernel output must equal the reference exactly.
    for r in range(m):
        cols = rng.choice(k, size=16, replace=False)
        x_np[r, cols] = rng.choice([-2.0, -1.0, 1.0, 2.0], size=16)
    for c in range(n):
        cols = rng.choice(k, size=16, replace=False)
        w_np[c, cols] = rng.choice([-2.0, -1.0, 1.0, 2.0], size=16)

    x_bf16 = x_np.astype(ml_dtypes.bfloat16)
    w_bf16 = w_np.astype(ml_dtypes.bfloat16)

    with tpu_interpret_mode():
        out_jax = mm_mod._jax_matmul(
            jnp.asarray(x_bf16, dtype=jnp.bfloat16),
            jnp.asarray(w_bf16, dtype=jnp.bfloat16),
        )

    out_np = np.asarray(out_jax)
    assert out_np.shape == (m, n)
    assert out_np.dtype == ml_dtypes.bfloat16

    ref = _matmul_exact_ref(x_bf16, w_bf16).astype(ml_dtypes.bfloat16)
    assert np.array_equal(out_np, ref)


def test_matmul_random_bf16_matches_matmulElem_exact(mm_mod):
    # Random bf16 inputs on an edge-block shape (M=19, N=141, K=1024 = 2 * BK).
    rng = np.random.default_rng(450)
    m, n, k = 19, 141, 1024
    x_np = rng.standard_normal((m, k)).astype(np.float32).astype(ml_dtypes.bfloat16)
    w_np = rng.standard_normal((n, k)).astype(np.float32).astype(ml_dtypes.bfloat16)

    with tpu_interpret_mode():
        out_jax = mm_mod._jax_matmul(
            jnp.asarray(x_np, dtype=jnp.bfloat16),
            jnp.asarray(w_np, dtype=jnp.bfloat16),
        )

    out_np = np.asarray(out_jax)
    assert out_np.shape == (m, n)
    assert out_np.dtype == ml_dtypes.bfloat16

    ref = _matmul_exact_ref(x_np, w_np)
    # Accumulation runs in fp32 over K=1024 bf16 products (~1e-4 worst-case fp32
    # summation drift) and rounds once to bf16 (7 mantissa bits, 1 ULP = 2^-7 =
    # 7.8125e-3). rtol=1e-2 covers 1 bf16 ULP; atol=1e-3 covers cancellation near 0.
    np.testing.assert_allclose(out_np.astype(np.float32), ref, rtol=1e-2, atol=1e-3)
