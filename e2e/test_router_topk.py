"""End-to-end differential tests for `sglang_tpu/kernels/router_topk.py`."""

from __future__ import annotations

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from conftest import load_kernel_module, tpu_interpret_mode

EPS = 1e-20
SCALE = 2.5


@pytest.fixture(scope="module")
def router_mod():
    return load_kernel_module(
        "sglang_tpu/kernels/router_topk.py",
        "_tkl_e2e_router_topk",
    )


def _score_and_choice(logits: np.ndarray, bias: np.ndarray):
    # `score` and `choice` of TpuKernelLean.Examples.RouterTopk, with the opaque `RouterOps`
    # instantiated by the XLA CPU ops the interpreted kernel also runs: f32 sigmoid, rounded to
    # the logits dtype, promoted to f32, plus the f32 bias.
    x = jnp.asarray(logits)
    s = (1.0 / (1.0 + jnp.exp(-x.astype(jnp.float32)))).astype(x.dtype).astype(jnp.float32)
    c = s + jnp.asarray(bias, jnp.float32)[None, :]
    return np.asarray(s), np.asarray(c)


def _math_row_ref(s: np.ndarray, c: np.ndarray, k: int, renormalize: bool):
    # `mathRow` (`routerTopk_correct`): rank[e] = #{j : c[j] > c[e] or (c[j] == c[e] and j < e)}
    # (`Before`, `rank`), ids[i] = the expert of rank i (`atRank`), weights = the unbiased scores
    # at those ids (`gatherAt`), then `renormRow`.
    tokens, experts = c.shape
    lane = np.arange(experts)
    ids = np.empty((tokens, k), np.int32)
    weights = np.empty((tokens, k), np.float32)
    for t in range(tokens):
        cj = c[t][:, None]
        ce = c[t][None, :]
        before = (cj > ce) | ((cj == ce) & (lane[:, None] < lane[None, :]))  # [j, e]
        rank = before.sum(axis=0)
        assert np.array_equal(np.sort(rank), lane), "rank is not a permutation"
        by_rank = np.argsort(rank)
        ids[t] = by_rank[:k]
        weights[t] = s[t, by_rank[:k]]
    if renormalize:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + np.float32(EPS))
        weights = weights * np.float32(SCALE)
    return weights, ids


def _tied_inputs(rng, tokens, experts, np_dtype):
    # Logits on a coarse grid that bf16 and f32 hold exactly, and a bias of three levels, so many
    # biased scores tie exactly and the lowest-id rule decides.
    logits = (rng.integers(-4, 5, size=(tokens, experts)) * 0.5).astype(np.float32)
    bias = rng.choice(np.array([-0.5, 0.0, 0.5], np.float32), size=(experts,))
    return logits.astype(np_dtype), bias


def _run(router_mod, logits, bias, k, renormalize, jnp_dtype):
    with tpu_interpret_mode():
        w, ids = router_mod.router_topk_jax(
            jnp.asarray(logits, dtype=jnp_dtype),
            jnp.asarray(bias, dtype=jnp.float32),
            k,
            renormalize,
            EPS,
            SCALE,
        )
    return np.asarray(w), np.asarray(ids)


def _uses_rank_path(tokens, experts, k):
    # router_topk.py:35 (`UseRankPath`).
    return tokens == 1 and experts % 128 == 0 and k <= experts <= 1024


@pytest.mark.parametrize(
    "tokens, experts, k, dtype, renormalize",
    [
        # one-token rank path (`_topk_kernel_1tok`), Kimi-K3 shape and a two-tile shape
        (1, 896, 16, "bf16", True),
        (1, 256, 8, "bf16", False),
        # bf16 packed-key rounds, GLM-5.3-Flash shape and Kimi-K3 at T=16
        (4, 288, 8, "bf16", True),
        (16, 896, 16, "bf16", False),
        # one token that misses the rank path (experts % 128 != 0): packed rounds
        (1, 200, 8, "bf16", True),
        # plain rounds (f32 logits), including one token on the rank path with f32
        (3, 160, 6, "f32", True),
        (2, 128, 4, "f32", False),
        (1, 384, 5, "f32", True),
    ],
)
def test_router_topk_matches_mathRow(router_mod, tokens, experts, k, dtype, renormalize):
    np_dtype = ml_dtypes.bfloat16 if dtype == "bf16" else np.float32
    jnp_dtype = jnp.bfloat16 if dtype == "bf16" else jnp.float32
    rng = np.random.default_rng(1000 + tokens * 31 + experts + k)
    logits, bias = _tied_inputs(rng, tokens, experts, np_dtype)

    w, ids = _run(router_mod, logits, bias, k, renormalize, jnp_dtype)
    assert w.shape == (tokens, k) and w.dtype == np.float32
    assert ids.shape == (tokens, k) and ids.dtype == np.int32

    s, c = _score_and_choice(logits.astype(np_dtype), bias)
    ref_w, ref_ids = _math_row_ref(s, c, k, renormalize)
    # Ids and the gathered scores are exact in the proof (compares, max, integer min, one-term
    # sums), so they must match bit for bit.
    np.testing.assert_array_equal(ids, ref_ids)
    if renormalize:
        # `rowSum` is opaque: NumPy's f32 sum order may differ from XLA's by an ulp.
        np.testing.assert_allclose(w, ref_w, rtol=2e-6, atol=0)
    else:
        np.testing.assert_array_equal(w, ref_w)


@pytest.mark.parametrize("tokens", [1, 2])
def test_router_topk_cutoff_ties_take_lowest_ids(router_mod, tokens):
    # The benchmark's tie case: the first k-1 logits are 1, everything else 0, bias 0. Every other
    # expert ties at the cutoff, so slot k-1 must be expert k-1 (the lowest tied id). T=1 takes
    # the rank path, T=2 the packed rounds.
    experts, k = 896, 16
    logits = np.zeros((tokens, experts), np.float32)
    logits[:, : k - 1] = 1.0
    logits = logits.astype(ml_dtypes.bfloat16)
    bias = np.zeros((experts,), np.float32)
    assert _uses_rank_path(tokens, experts, k) == (tokens == 1)

    w, ids = _run(router_mod, logits, bias, k, True, jnp.bfloat16)
    np.testing.assert_array_equal(ids, np.tile(np.arange(k, dtype=np.int32), (tokens, 1)))
    s, c = _score_and_choice(logits, bias)
    ref_w, ref_ids = _math_row_ref(s, c, k, True)
    np.testing.assert_array_equal(ids, ref_ids)
    np.testing.assert_allclose(w, ref_w, rtol=2e-6, atol=0)


def test_router_topk_weights_are_unbiased_scores(router_mod):
    # Selection follows the biased choice, weights the unbiased score: expert 2 has the lowest
    # score but the largest bias, so it is picked first and its weight stays its own score.
    logits = np.array([[2.0, 1.0, -1.0, 0.0]], np.float32)
    bias = np.array([0.0, 0.0, 3.0, 0.0], np.float32)
    w, ids = _run(router_mod, logits, bias, 2, False, jnp.float32)
    np.testing.assert_array_equal(ids, np.array([[2, 0]], np.int32))
    s, _ = _score_and_choice(logits, bias)
    np.testing.assert_array_equal(w, s[:, [2, 0]])
