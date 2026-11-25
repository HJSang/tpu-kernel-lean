"""End-to-end differential tests for `sglang_tpu/kernels/fp8_linear_small_m.py`."""

from __future__ import annotations

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from conftest import load_kernel_module, tpu_interpret_mode


@pytest.fixture(scope="module")
def fp8_mod():
    return load_kernel_module(
        "sglang_tpu/kernels/fp8_linear_small_m.py",
        "_tkl_e2e_fp8_linear_small_m",
    )


def _fp8_linear_exact_ref(
    x: np.ndarray, w: np.ndarray, ws: np.ndarray
) -> np.ndarray:
    # Formula from TpuKernelLean.Examples.Fp8LinearSmallM.fp8LinearMath_exact
    # (`fp8LinearElem_exact`): (x @ w) * ws in exact arithmetic, with `ws`
    # broadcast to [1, n].
    n = w.shape[1]
    ws_b = np.broadcast_to(ws.astype(np.float64).reshape(-1), (n,)).reshape(1, n)
    return (x.astype(np.float64) @ w.astype(np.float64)) * ws_b


def _fp8_linear_quantized_ref(
    x: np.ndarray,
    w: np.ndarray,
    ws: np.ndarray,
    fp8_np_dtype,
    out_np_dtype,
) -> np.ndarray:
    # Quantized reference matching TpuKernelLean.Examples.Fp8LinearSmallM.fp8LinearElem:
    # per-row amax / fp8_max floored at 1e-12, unsaturated cast to fp8, fp32 dot,
    # rescale by x_scale * ws, and cast to output dtype.
    n = w.shape[1]
    x_f32 = x.astype(np.float32)
    fp8_max = np.float32(ml_dtypes.finfo(fp8_np_dtype).max)
    x_scale = np.maximum(
        np.max(np.abs(x_f32), axis=-1, keepdims=True) / fp8_max,
        np.float32(1e-12),
    )
    x_q = (x_f32 / x_scale).astype(fp8_np_dtype)
    acc = x_q.astype(np.float32) @ w.astype(np.float32)
    ws_b = np.broadcast_to(ws.astype(np.float32).reshape(-1), (n,)).reshape(1, n)
    return (acc * x_scale * ws_b).astype(out_np_dtype)


def _rel_l2_and_cos(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    rel_l2 = float(np.linalg.norm(a64 - b64) / np.linalg.norm(b64))
    cos = float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64)))
    return rel_l2, cos


@pytest.mark.parametrize(
    "tokens, k, n, fp8_jnp, fp8_np, scalar_ws, max_rel_l2, min_cos",
    [
        # e4m3fn (3 mantissa bits, max per-element rounding error 2^-4 = 6.25%):
        # non-multiple-of-8 tokens=3 (edge M for TPU 8x128 tiles), 3 column tiles (tile_n=128)
        (3, 128, 384, jnp.float8_e4m3fn, ml_dtypes.float8_e4m3fn, False, 0.08, 0.995),
        # e4m3fn, tokens=5, 3 column tiles of tile_n=256 (n=768), scalar weight_scale
        (5, 256, 768, jnp.float8_e4m3fn, ml_dtypes.float8_e4m3fn, True, 0.08, 0.995),
        # e5m2 (2 mantissa bits, max per-element rounding error 2^-3 = 12.5%):
        # single token (tokens=1), 2 column tiles of tile_n=128
        (1, 128, 256, jnp.float8_e5m2, ml_dtypes.float8_e5m2, False, 0.16, 0.985),
    ],
)
def test_fp8_linear_small_m_matches_fp8LinearMath_exact(
    fp8_mod, tokens, k, n, fp8_jnp, fp8_np, scalar_ws, max_rel_l2, min_cos
):
    assert fp8_mod.supports(tokens, k, n)

    rng = np.random.default_rng(100 + tokens * 17 + k + n)
    x_np = rng.standard_normal((tokens, k)).astype(np.float32).astype(ml_dtypes.bfloat16)
    # Draw weights directly representable in fp8 to avoid unrelated weight-cast noise.
    w_np = (rng.standard_normal((k, n)).astype(np.float32) * 2.0).astype(fp8_np)
    if scalar_ws:
        ws_np = np.asarray(0.75, dtype=np.float32)
    else:
        ws_np = rng.uniform(0.25, 1.5, size=(n,)).astype(np.float32)

    x_jax = jnp.asarray(x_np, dtype=jnp.bfloat16)
    w_jax = jnp.asarray(w_np, dtype=fp8_jnp)
    ws_jax = jnp.asarray(ws_np, dtype=jnp.float32)

    with tpu_interpret_mode():
        out_jax = fp8_mod.fp8_linear_small_m(x_jax, w_jax, ws_jax)

    out_np = np.asarray(out_jax)
    assert out_np.shape == (tokens, n)
    assert out_np.dtype == ml_dtypes.bfloat16

    # 1. Exact-arithmetic reference (`fp8LinearMath_exact`):
    # Tolerance justified by FP8 activation quantization (3 mantissa bits for e4m3fn,
    # 2 mantissa bits for e5m2) plus final bf16 output rounding (7 mantissa bits).
    ref_exact = _fp8_linear_exact_ref(x_np, w_np, ws_np)
    rel_l2, cos = _rel_l2_and_cos(out_np, ref_exact)
    assert rel_l2 <= max_rel_l2, f"rel_l2={rel_l2:.4f} > {max_rel_l2}"
    assert cos >= min_cos, f"cos={cos:.6f} < {min_cos}"

    # 2. Quantized NumPy/ml_dtypes reference (`fp8LinearElem`):
    # Both paths apply the same FP8 quantization; residual differences come only from
    # fp32 summation order and a 1-ULP bf16 tie (bf16 eps = 2^-7 = 7.8125e-3).
    ref_quant = _fp8_linear_quantized_ref(
        x_np, w_np, ws_np, fp8_np, ml_dtypes.bfloat16
    )
    np.testing.assert_allclose(
        out_np.astype(np.float32),
        ref_quant.astype(np.float32),
        rtol=1e-2,
        atol=1e-4,
    )


def test_fp8_linear_small_m_zero_row_scale_floor(fp8_mod):
    # Checks the `1e-12` scale floor proved in `fp8LinearElem_exact`: an all-zero
    # activation row must not divide by zero or produce NaN.
    tokens, k, n = 2, 128, 128
    rng = np.random.default_rng(777)
    x_np = rng.standard_normal((tokens, k)).astype(np.float32).astype(ml_dtypes.bfloat16)
    x_np[0, :] = 0.0
    w_np = rng.standard_normal((k, n)).astype(np.float32).astype(ml_dtypes.float8_e4m3fn)
    ws_np = rng.uniform(0.5, 1.5, size=(n,)).astype(np.float32)

    with tpu_interpret_mode():
        out_jax = fp8_mod.fp8_linear_small_m(
            jnp.asarray(x_np, dtype=jnp.bfloat16),
            jnp.asarray(w_np, dtype=jnp.float8_e4m3fn),
            jnp.asarray(ws_np, dtype=jnp.float32),
        )

    out_np = np.asarray(out_jax)
    assert np.all(np.isfinite(out_np.astype(np.float32)))
    assert np.array_equal(
        out_np[0].astype(np.float32), np.zeros((n,), dtype=np.float32)
    )
