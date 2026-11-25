"""End-to-end differential tests for `sglang_tpu/rl/true_on_policy/norm_kernel.py`."""

from __future__ import annotations

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from conftest import load_kernel_module, tpu_interpret_mode

# Helper module from the same repository (at the manifest-pinned commit) providing `row_block`.
HELPER_PATH = "sglang_tpu/rl/true_on_policy/common.py"


@pytest.fixture(scope="module")
def norm_mod():
    return load_kernel_module(
        "sglang_tpu/rl/true_on_policy/norm_kernel.py",
        "_tkl_e2e_true_on_policy_norm",
        stub_torch=True,
        stub_sglang=True,
        helper_paths=(HELPER_PATH,),
    )


def _norm_row_exact_ref(
    s: np.ndarray, w: np.ndarray, eps: float, *, gemma: bool = False
) -> np.ndarray:
    # Formula from TpuKernelLean.Examples.TrueOnPolicyNorm.normRow_exact
    # (and `normRow` for gemma=True): s * (1 / sqrt(mean(s * s) + eps)) * (1 + w if gemma else w).
    s64 = s.astype(np.float64)
    w64 = w.astype(np.float64)
    mean_sq = np.mean(s64 * s64, axis=-1, keepdims=True)
    scale = 1.0 + w64 if gemma else w64
    return s64 / np.sqrt(mean_sq + eps) * scale


@pytest.mark.parametrize(
    "m, h, gemma",
    [
        # Plain RMSNorm, single partial edge block (m=13 not a multiple of 16)
        (13, 128, False),
        # Gemma RMSNorm, single partial edge block
        (13, 128, True),
        # Multi-block with partial trailing edge block:
        # cap = 1 << ((1 << 20) // 8192).bit_length() - 1 = 128, so m=133 gives
        # grid=(2,) with 5 valid rows and 123 NaN-padded rows in the second block.
        (133, 8192, False),
    ],
)
def test_rms_norm_matches_normRow_exact(norm_mod, m, h, gemma):
    rng = np.random.default_rng(300 + m + int(gemma))
    eps = 1e-6
    x_np = rng.standard_normal((m, h)).astype(np.float32).astype(ml_dtypes.bfloat16)
    w_np = rng.uniform(0.5, 1.5, size=(h,)).astype(np.float32).astype(ml_dtypes.bfloat16)

    with tpu_interpret_mode():
        out_jax = norm_mod._jax_rms_norm(
            jnp.asarray(x_np, dtype=jnp.bfloat16),
            jnp.asarray(w_np, dtype=jnp.bfloat16),
            eps,
            gemma,
        )

    out_np = np.asarray(out_jax)
    assert out_np.shape == (m, h)
    assert out_np.dtype == ml_dtypes.bfloat16

    ref = _norm_row_exact_ref(x_np, w_np, eps, gemma=gemma)
    # All norm math runs in fp32 and only the final output is rounded to bf16
    # (7 mantissa bits, 1 ULP = 2^-7 = 7.8125e-3, half-ULP ~ 3.91e-3).
    # rtol=1e-2 covers 1 bf16 ULP plus fp32 rsqrt/reduction rounding; atol=1e-4
    # covers near-zero entries.
    np.testing.assert_allclose(out_np.astype(np.float32), ref, rtol=1e-2, atol=1e-4)


@pytest.mark.parametrize("gemma", [False, True])
def test_rms_norm_residual_matches_rmsNormResidual_correct(norm_mod, gemma):
    # Formula from TpuKernelLean.Examples.TrueOnPolicyNorm.rmsNormResidual_correct
    # and `normRow_exact`: ro = bf16(x + r); plain normalizes unrounded x + r,
    # Gemma normalizes rounded ro with weight (1 + w).
    rng = np.random.default_rng(350 + int(gemma))
    m, h = 19, 128  # edge block: 19 not a multiple of 16
    eps = 1e-6
    x_np = rng.standard_normal((m, h)).astype(np.float32).astype(ml_dtypes.bfloat16)
    r_np = rng.standard_normal((m, h)).astype(np.float32).astype(ml_dtypes.bfloat16)
    w_np = rng.uniform(0.5, 1.5, size=(h,)).astype(np.float32).astype(ml_dtypes.bfloat16)

    with tpu_interpret_mode():
        out_jax, ro_jax = norm_mod._jax_rms_norm_residual(
            jnp.asarray(x_np, dtype=jnp.bfloat16),
            jnp.asarray(r_np, dtype=jnp.bfloat16),
            jnp.asarray(w_np, dtype=jnp.bfloat16),
            eps,
            gemma,
        )

    out_np = np.asarray(out_jax)
    ro_np = np.asarray(ro_jax)
    assert out_np.dtype == ml_dtypes.bfloat16
    assert ro_np.dtype == ml_dtypes.bfloat16

    sum_f32 = x_np.astype(np.float32) + r_np.astype(np.float32)
    ro_ref = sum_f32.astype(ml_dtypes.bfloat16)
    assert np.array_equal(ro_np, ro_ref)

    norm_in = ro_ref.astype(np.float64) if gemma else sum_f32.astype(np.float64)
    out_ref = _norm_row_exact_ref(norm_in, w_np, eps, gemma=gemma)
    # Same bf16 output rounding justification: 1 ULP = 2^-7 = 7.8125e-3.
    np.testing.assert_allclose(out_np.astype(np.float32), out_ref, rtol=1e-2, atol=1e-4)


def test_gated_rms_norm_matches_gatedRmsNorm_correct(norm_mod):
    # Formula from TpuKernelLean.Examples.TrueOnPolicyNorm.gatedRmsNorm_correct
    # with `FOps.exact`: s * (1 / sqrt(mean(s * s) + eps)) * w * silu(z),
    # where silu(z) = z / (1 + exp(-z)).
    rng = np.random.default_rng(375)
    m, h = 19, 128  # edge block: 19 not a multiple of 16
    eps = 1e-6
    x_np = rng.standard_normal((m, h)).astype(np.float32).astype(ml_dtypes.bfloat16)
    z_np = rng.standard_normal((m, h)).astype(np.float32).astype(ml_dtypes.bfloat16)
    w_np = rng.uniform(0.5, 1.5, size=(h,)).astype(np.float32).astype(ml_dtypes.bfloat16)

    with tpu_interpret_mode():
        out_jax = norm_mod._jax_gated_rms_norm(
            jnp.asarray(x_np, dtype=jnp.bfloat16),
            jnp.asarray(z_np, dtype=jnp.bfloat16),
            jnp.asarray(w_np, dtype=jnp.bfloat16),
            eps,
        )

    out_np = np.asarray(out_jax)
    assert out_np.dtype == ml_dtypes.bfloat16

    z64 = z_np.astype(np.float64)
    silu_ref = z64 / (1.0 + np.exp(-z64))
    out_ref = _norm_row_exact_ref(x_np, w_np, eps, gemma=False) * silu_ref
    # Same bf16 output rounding justification: 1 ULP = 2^-7 = 7.8125e-3.
    np.testing.assert_allclose(out_np.astype(np.float32), out_ref, rtol=1e-2, atol=1e-4)
