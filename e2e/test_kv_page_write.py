"""End-to-end differential tests for `sglang_tpu/kv_page_write.py` (`write_pages`)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import load_kernel_module, tpu_interpret_mode


@pytest.fixture(scope="module")
def kv_mod():
    return load_kernel_module(
        "sglang_tpu/kv_page_write.py",
        "_tkl_e2e_kv_page_write",
        stub_torch=True,
    )


def _write_pages_ref(
    kv: np.ndarray, pages: np.ndarray, block_ids: np.ndarray
) -> np.ndarray:
    # Formula from TpuKernelLean.Examples.KvPageWrite.writePages_correct
    # (`writePagesMut`): page `j` receives `pages[i]` if `blockIds[i] = j`,
    # and stays equal to initial `kv[j]` otherwise.
    out = kv.copy()
    for i, target in enumerate(block_ids):
        out[int(target)] = pages[i]
    return out


@pytest.mark.parametrize(
    "kv_shape, block_ids, dtype",
    [
        # 2D flattened pages [N, P], partial write leaving unwritten slots intact
        ((8, 16), [5, 1, 7], np.float32),
        # Non-power-of-two N, M, and trailing 3D page shape [page_size, heads, dim]
        ((9, 3, 5, 7), [8, 0, 4, 2], np.float32),
        # bf16 pages, full permutation of all N slots (M == N)
        ((5, 4, 13), [3, 0, 4, 1, 2], jnp.bfloat16),
        # Single-page write (M == 1) with non-multiple-of-8/128 page width
        ((6, 19), [3], jnp.bfloat16),
    ],
)
def test_write_pages_matches_writePages_correct(kv_mod, kv_shape, block_ids, dtype):
    rng = np.random.default_rng(42 + len(block_ids) + kv_shape[0])
    m = len(block_ids)
    pages_shape = (m, *kv_shape[1:])

    kv_np = rng.standard_normal(kv_shape).astype(np.float32)
    pages_np = rng.standard_normal(pages_shape).astype(np.float32)
    ids_np = np.asarray(block_ids, dtype=np.int32)

    kv_jax = jnp.asarray(kv_np, dtype=dtype)
    pages_jax = jnp.asarray(pages_np, dtype=dtype)
    ids_jax = jnp.asarray(ids_np, dtype=jnp.int32)

    # Pure copy (`out_ref[...] = pages_ref[...]`); compare bit-for-bit against
    # the exact post-state from `writePages_correct`.
    with tpu_interpret_mode():
        out_jax = kv_mod.write_pages(ids_jax, pages_jax, kv_jax)

    out_np = np.asarray(out_jax)
    ref_np = _write_pages_ref(np.asarray(kv_jax), np.asarray(pages_jax), ids_np)

    assert out_np.dtype == ref_np.dtype
    assert out_np.shape == ref_np.shape
    assert np.array_equal(out_np, ref_np)

    # Explicitly verify unwritten pages are bit-for-bit identical to initial `kv`.
    written = set(block_ids)
    unwritten = [j for j in range(kv_shape[0]) if j not in written]
    if unwritten:
        assert np.array_equal(out_np[unwritten], np.asarray(kv_jax)[unwritten])
