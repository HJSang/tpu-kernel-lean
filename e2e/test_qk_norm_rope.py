"""End-to-end differential tests for `sglang_tpu/qk_norm_rope.py` (`_qk_norm_rope_jax`)."""

from __future__ import annotations

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from conftest import load_kernel_module, tpu_interpret_mode


@pytest.fixture(scope="module")
def qk_mod():
    return load_kernel_module(
        "sglang_tpu/qk_norm_rope.py",
        "_tkl_e2e_qk_norm_rope",
        stub_torch=True,
        stub_sglang=True,
    )


def _rms_norm_ref(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    mean_sq = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(mean_sq + eps) * w


def _rope_neox_ref(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _qk_norm_rope_exact_ref(
    qkv: np.ndarray,
    cos_sin: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    eps: float,
    num_q: int,
    num_kv: int,
    hd: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Formula from TpuKernelLean.Examples.QkNormRope.qkNormRope_exact_correct
    # (`qkNormRopeExactMath`): extract Q and K slices from `qkv` (ignoring V),
    # apply per-head `Tensor.rmsNorm` then `Tensor.ropeNeox` with unpacked cos/sin.
    t = qkv.shape[0]
    q_size = num_q * hd
    kv_size = num_kv * hd
    half = hd // 2

    q_in = qkv[:, :q_size].astype(np.float64).reshape(t, num_q, hd)
    k_in = qkv[:, q_size : q_size + kv_size].astype(np.float64).reshape(t, num_kv, hd)
    cs = cos_sin.astype(np.float64)
    cos = cs[:, None, :half]
    sin = cs[:, None, half:]

    q_out = _rope_neox_ref(
        _rms_norm_ref(q_in, q_weight.astype(np.float64), eps), cos, sin
    ).reshape(t, q_size)
    k_out = _rope_neox_ref(
        _rms_norm_ref(k_in, k_weight.astype(np.float64), eps), cos, sin
    ).reshape(t, kv_size)
    return q_out, k_out


@pytest.mark.parametrize(
    "tokens, num_q, num_kv, hd",
    [
        # Production head_dim=128, single block (tokens < _MAX_BLOCK_TOKENS=256)
        (5, 4, 2, 128),
        # Edge block: tokens=260 > _MAX_BLOCK_TOKENS=256, 260 % 256 == 4 (grid=(2,))
        # exercises partial trailing block padded with NaN in interpret mode
        (260, 2, 1, 16),
    ],
)
def test_qk_norm_rope_matches_qkNormRope_exact_correct(
    qk_mod, tokens, num_q, num_kv, hd
):
    rng = np.random.default_rng(200 + tokens + hd)
    eps = 1e-6
    width = (num_q + 2 * num_kv) * hd

    qkv_np = rng.standard_normal((tokens, width)).astype(np.float32).astype(ml_dtypes.bfloat16)
    angles = rng.uniform(-np.pi, np.pi, size=(tokens, hd // 2))
    cs_np = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32).astype(ml_dtypes.bfloat16)
    qw_np = rng.uniform(0.5, 1.5, size=(hd,)).astype(np.float32).astype(ml_dtypes.bfloat16)
    kw_np = rng.uniform(0.5, 1.5, size=(hd,)).astype(np.float32).astype(ml_dtypes.bfloat16)

    with tpu_interpret_mode():
        q_jax, k_jax = qk_mod._qk_norm_rope_jax(
            jnp.asarray(qkv_np, dtype=jnp.bfloat16),
            jnp.asarray(cs_np, dtype=jnp.bfloat16),
            jnp.asarray(qw_np, dtype=jnp.bfloat16),
            jnp.asarray(kw_np, dtype=jnp.bfloat16),
            eps,
            num_q,
            num_kv,
            hd,
        )

    q_out = np.asarray(q_jax)
    k_out = np.asarray(k_jax)
    assert q_out.dtype == ml_dtypes.bfloat16
    assert k_out.dtype == ml_dtypes.bfloat16

    q_ref, k_ref = _qk_norm_rope_exact_ref(
        qkv_np, cs_np, qw_np, kw_np, eps, num_q, num_kv, hd
    )

    # All internal arithmetic runs in fp32 and only the final outputs are rounded to
    # bf16 (7 mantissa bits, 1 ULP = 2^-7 = 7.8125e-3; half-ULP ~ 3.91e-3).
    # rtol=1e-2 covers 1 bf16 ULP plus fp32 rsqrt/reduction rounding; atol=1e-4
    # covers cancellation near zero in `x1*cos - x2*sin`.
    np.testing.assert_allclose(q_out.astype(np.float32), q_ref, rtol=1e-2, atol=1e-4)
    np.testing.assert_allclose(k_out.astype(np.float32), k_ref, rtol=1e-2, atol=1e-4)


def test_qk_norm_rope_v_columns_do_not_affect_outputs(qk_mod):
    # Checks `qkNormRope_loadQkv_ignores_v` / `qkNormRope_batch_invariant`:
    # mutating the V columns (including putting NaNs there) must not change Q or K.
    rng = np.random.default_rng(299)
    tokens, num_q, num_kv, hd = 7, 2, 2, 128
    eps = 1e-6
    qk_width = (num_q + num_kv) * hd
    total_width = (num_q + 2 * num_kv) * hd

    qkv_a = rng.standard_normal((tokens, total_width)).astype(np.float32).astype(ml_dtypes.bfloat16)
    qkv_b = qkv_a.copy()
    qkv_b[:, qk_width:] = np.nan

    angles = rng.uniform(-np.pi, np.pi, size=(tokens, hd // 2))
    cs_np = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32).astype(ml_dtypes.bfloat16)
    qw_np = rng.uniform(0.5, 1.5, size=(hd,)).astype(np.float32).astype(ml_dtypes.bfloat16)
    kw_np = rng.uniform(0.5, 1.5, size=(hd,)).astype(np.float32).astype(ml_dtypes.bfloat16)

    with tpu_interpret_mode():
        q_a, k_a = qk_mod._qk_norm_rope_jax(
            jnp.asarray(qkv_a, dtype=jnp.bfloat16),
            jnp.asarray(cs_np, dtype=jnp.bfloat16),
            jnp.asarray(qw_np, dtype=jnp.bfloat16),
            jnp.asarray(kw_np, dtype=jnp.bfloat16),
            eps,
            num_q,
            num_kv,
            hd,
        )
        q_b, k_b = qk_mod._qk_norm_rope_jax(
            jnp.asarray(qkv_b, dtype=jnp.bfloat16),
            jnp.asarray(cs_np, dtype=jnp.bfloat16),
            jnp.asarray(qw_np, dtype=jnp.bfloat16),
            jnp.asarray(kw_np, dtype=jnp.bfloat16),
            eps,
            num_q,
            num_kv,
            hd,
        )

    assert np.array_equal(np.asarray(q_a), np.asarray(q_b))
    assert np.array_equal(np.asarray(k_a), np.asarray(k_b))
