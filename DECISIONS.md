# Modelling and Scope Decisions

Every entry records a choice that JAX Pallas / Mosaic documentation does not settle
or that narrows what the Lean theorems cover. Kinds:

- **stricter**: rejects runs or inputs the real runtime would accept;
- **hardware-dependent**: relies on TPU/Mosaic compiler or hardware behaviour not proved in Lean;
- **launch obligation**: a host/caller requirement carried as a theorem premise (`requires`);
- **scope**: a choice of which functions, paths, or semantics the theorems cover.

## Core model (`Pallas/Core.lean`, `Memory.lean`, `Dtype.lean`, `Spec.lean`)

### D-CORE-SEQ: proofs run the grid sequentially; parallel runs are reduced to that
- Decided: `pallasExec` traverses `gridPoints gs.grid` in row-major order and every kernel proof
  is about that run. `dimension_semantics` is carried as `GridSpec.dimSemantics`; under the three
  `parallel_*` conditions of `WellDefined`, `pallasExecPar_eq_pallasExec`
  (`TpuKernelLean/Pallas/Parallel.lean`) shows any split across cores, interleaving and flush
  order gives the same memory. `parallel` dimensions must precede `arbitrary` ones.
- Basis: `docs/pallas/tpu/details.rst` (sequential, lexicographic order) and
  `docs/pallas/tpu/pipelining.md` (megacore) in JAX at 35066e1. JAX's TPU interpreter assumes
  the same ordering (`mosaic/interpret/interpret_pallas_call.py:2273`); the compiled lowering
  does not check it. The split itself happens in libtpu and has not been probed.
- Alternative: an event-level model of concurrent DMA.
- Kind: hardware-dependent (the parallel model).

### D-CORE-STALE: no input may read a region an earlier output block covers
- Decided: `GridSpec.WellDefined.no_stale_read` forbids an input block at step `t` from
  overlapping an output block of any earlier step `s < t`, whether or not that block has been
  written back. This also forbids an aliased input re-reading a block the kernel is still writing.
- Basis: Mosaic skips the fetch when an input block index repeats and prefetches before the
  write-back lands (`docs/pallas/tpu/details.rst:94-96`). For an aliased pair on a repeated
  index, JAX's TPU interpreter and `interpret=True` disagree (`hlo_interpreter.py:473-490`), and
  the compiled path is closed, so the pattern is undefined here.
- Alternative: model the pipeline depth and probe the aliased case on hardware.
- Kind: stricter.

### D-CORE-GRID-EMPTY: empty grids and zero-size blocks are undefined
- Decided: `WellDefined.grid_pos` and `block_specs_ok` require every grid dimension and every
  block dimension to be positive and every `BlockSpec` to have its array's rank, checked without
  reference to the grid. Each kernel carries the positivity hypotheses it needs (`0 < M`, `0 < T`).
- Basis: JAX rejects a rank mismatch at trace time (`jax/_src/pallas/core.py:719-724`). Its
  interpreters accept a zero-trip grid and return uninitialised outputs; the model does not, so
  no kernel is correct vacuously.
- Alternative: define the empty run as the identity on memory.
- Kind: stricter.

### D-CORE-REAL-MEM: real-valued logical memory without IEEE special values or XLA tiling
- Decided: `Mem := BufId → ℕ → ℝ` (`TpuKernelLean/Memory.lean:8-9`) with logical row-major offsets.
  `-0.0` and `+0.0` are equal in `ℝ`; `±inf`, NaN propagation/payloads, and XLA `(8, 128)`
  HBM tiling are not modelled (`TpuKernelLean/Dtype.lean:6,58-62`, `README.md:113,123`).
- Basis: keeps index/mask/frame proofs lightweight while covering non-byte-reinterpreting kernels.
- Alternative: IEEE bit-level memory and tiled physical addresses.
- Kind: scope (and hardware-dependent where bitwise claims identify `-0.0` with `+0.0`).

### D-CORE-OPAQUE-OPS: opaque float ops for invariance, exact reals for functional math
- Decided: bitwise theorems (`FOps`, `MOps`, `FP8LinearOps`, `CastOps`) keep float ops opaque
  and deterministic (`TpuKernelLean/Dtype.lean:318-323`); functional theorems instantiate them with
  exact real arithmetic and identity casts (`*.exact`, `TpuKernelLean/Dtype.lean:342-345`).
- Basis: TPU ISA/LLO rounding schedules are closed; exact reals verify algorithm structure,
  opaque ops verify data/shape independence.
- Alternative: IEEE-754 rounding error bounds (`withinTol ε`).
- Kind: hardware-dependent (opaque ops) / scope (exact arithmetic).

### D-CORE-SCALAR-ALIAS: scalar prefetch is a checked int32 snapshot of named buffers
- Decided: `GridSpec.scalarPrefetchBufs` lists the scalar-prefetch buffers; the argument count
  is its length, and `scalarPrefetchFn` sees only checked `readInt32` readers of those buffers,
  taken once before the grid runs. `AliasesValid.out_disjoint_scalar` forbids any output from
  using one of them.
- Basis: Python argument numbering (`jax/_src/pallas/pallas_call.py:1304-1324`); the TPU
  interpreter refuses to alias a scalar-prefetch argument.
- Alternative: model SMEM refs that the body may mutate.
- Kind: scope / stricter.

### D-CORE-ALIAS-BS: aliased pairs must use the same block shape and block index
- Decided: `AliasesValid.entries_ok` requires an aliased input and output to share buffer, array
  shape, `blockShape` and block index at every grid step.
- Basis: JAX checks only array shape and dtype (`pallas_call.py:1304-1324`); what libtpu does
  with mismatched windows is unknown, so the mismatch is undefined here.
- Alternative: probe mismatched aliased windows on hardware and model the result.
- Kind: stricter.

### D-CORE-TILE-ALIGN: Mosaic `8×128` tile alignment and VMEM capacity unchecked
- Decided: `blockOk` (`TpuKernelLean/Pallas/Core.lean:167-174`) checks rank, positive block dims,
  non-negative indices, and at least one in-bounds element, not Mosaic's TPU `8×128`
  alignment or VMEM limits.
- Basis: hardware/compiler-version specific constraints.
- Alternative: encode TPU tile alignment and VMEM budgets in `WellDefined`.
- Kind: launch obligation.

### D-CORE-DMA: explicit async DMA and `pl.ANY` HBM refs disconnected from `pallasCall`
- Decided: `TpuKernelLean/Pallas/DMA.lean` checks standalone explicit DMA event traces; `pallasCall`
  models only automatic `BlockSpec` VMEM windows (`README.md:71,114`).
- Basis: none of the verified `pallasCall` kernels use `make_async_copy`.
- Alternative: integrate DMA events and HBM refs into `stepGrid`.
- Kind: scope.

### D-CORE-FOOTPRINT: `KernelSpec.WellFormed` enforces footprint/read separation and exact equality
- Decided: `KernelSpec` (`TpuKernelLean/Spec.lean`) has no caller-chosen `closeOut`/`closeSt`
  predicates (`KernelSpec.Correct` requires exact equality of outputs and post-state), and
  `KernelSpec.Correct` includes `KernelSpec.WellFormed` under `requires`: `readOut` and `readSt`
  depend only on memory inside `footprint` (`Mem.agreeOn`), while `readIn` depends only on memory
  outside `footprint` (`Mem.agreeOutside`).
- Basis: prevents specs from using trivial closeness predicates, over-broad footprints (such as
  `Set.univ`) that swallow inputs and vacate framing, or `readOut`/`readSt` functions that read
  untouched buffers outside `footprint`. Checking `WellFormed` under `requires` keeps specs
  parameterized over arbitrary `BufId`s (with buffer distinctness stated in `requires`) while
  forcing layout well-formedness unconditionally across all memory pairs whenever `requires` is
  satisfiable.
- Alternative: make `KernelSpec` carry explicit `Layout` objects and derive `footprint`
  structurally, or lift buffer distinctness out of `requires` into every `*_correct` theorem's
  hypotheses.
- Kind: stricter.

## `KvPageWrite` (`sglang_tpu/kv_page_write.py`)

### D-KV-SHAPE: trailing page dimensions flattened and `None` dim modelled as `1`
- Decided: `kv : [N, P]` and `pages : [M, P]` flatten `*kv.shape[1:]` (`kv_page_write.py:31`)
  into one dimension `P`, and `block = (None, ...)` is modelled as `[1, P]`
  (`TpuKernelLean/Examples/KvPageWrite.lean:6,58-59`).
- Basis: `write_pages` never indexes inside a page (`out_ref[...] = pages_ref[...]`).
- Alternative: rank-polymorphic `ArrayDesc` with squeezed dims.
- Kind: scope.

### D-KV-WRAPPER: Torch custom-op wrapper not modelled; `blockIds` obligations
- Decided: proves `write_pages` (`kv_page_write.py:25-48`) only, omitting `get_write_pages_op`
  and `write_pages_torch` (`kv_page_write.py:51-94`, including `.to(torch.int32)` at `:92` and
  `donate_argnums=(2,)` at `:69`). Caller must supply distinct in-range natural `blockIds` and
  pairwise-distinct buffers (`TpuKernelLean/Examples/KvPageWrite.lean:49-52`).
- Basis: Pallas-level scope; distinctness avoids undefined write-order races.
- Alternative: model the Torch wrapper and migrate `scalarPrefetch` to checked `int32`.
- Kind: scope / launch obligation.

## `QkNormRope` (`sglang_tpu/qk_norm_rope.py`)

### D-QK-WRAPPER: claims cover `_qk_norm_rope_jax`, not the Torch wrapper `tpu_fused_qk_norm_rope`
- Decided: theorems cover `_qk_norm_rope_jax` (`qk_norm_rope.py:79-119`) and its helpers (`_rms`,
  `_rope_neox`, `_qk_norm_rope_kernel`), not the Torch wrapper `tpu_fused_qk_norm_rope`
  (`qk_norm_rope.py:204-289`), omitting the `positions` gather and lossy
  `cos_sin_cache[...].to(qkv.dtype)` bf16 truncation (`:278`). Buffer disjointness (matching
  `donate_argnums=()` / `mutates_args=()` in `sglang_tpu/pallas_jax_op.py:215,220,259`) and
  `0 < eps` (unchecked in Python) are assumed in `requires`
  (`TpuKernelLean/Examples/QkNormRope.lean:237-241`). Head dim is parameterized as even `2 * h`
  (wrapper pins `head_dim == 128`, `qk_norm_rope.py:178,236`). `requires` includes `0 < T`: with no tokens
  `_qk_norm_rope_jax` raises `ZeroDivisionError` (`qk_norm_rope.py:95-96`), and the Lean run is
  undefined (`Tests/QkNormRope.lean`, `empty_batch_rejected`).
- Basis: Pallas-level scope; `0 < eps` ensures exact RMSNorm denominator positivity.
- Alternative: model the gather + bf16 truncation and require `0 < T` in `requires`.
- Kind: scope / launch obligation / stricter (`0 < eps`).

### D-QK-HW: cross-shape `FOps` hardware assumptions for batch invariance
- Decided: `qkNormRope_batch_invariant` (`TpuKernelLean/Examples/QkNormRope.lean:627`) quantifies
  over one shared `ops : FOps` (`:36-50`) across `(T, bm)` and `(T', bm')` and across `q`/`k`
  heads, assuming Mosaic lowers `jnp.mean(..., axis=-1)` identically across compiled
  `(bm, num_q, num_kv)` shapes, does not vary FMA contraction in `_rope_neox`, uses one `toF32`
  across inputs/weights, and does not underflow `eps` to `0.0f`.
- Basis: no Mosaic guarantee; requires hardware confirmation on TPU (`hd = 128`).
- Alternative: index `FOps` by `(bm, num_heads)` (restricting to fixed-shape independence, as
  in `Fp8LinearSmallM`) and add opaque rotary ops for FMA.
- Kind: hardware-dependent.

## `Fp8LinearSmallM` (`sglang_tpu/kernels/fp8_linear_small_m.py`)

### D-FP8-FIXED-SHAPE: fixed-shape independence only; no cross-batch invariance claim
- Decided: `FP8LinearOps.rowMax` and `dot` take `(tokens, k, tile_n)`
  (`TpuKernelLean/Examples/Fp8LinearSmallM.lean:119,128`), so
  `fp8Linear_fixedShape_row_independent` (`:516`) claims row independence only at one compiled
  `(tokens, k, tile_n)` shape, assuming Mosaic's untiled full-`K` `dot_general`
  (`fp8_linear_small_m.py:38-43`) depends only on its `x` row and `w` column at that shape.
- Basis: changing `tokens` (1..16) or `tile_n` changes VMEM shapes handed to untiled
  `dot_general`, so Mosaic may change K-blocking and fp32 accumulation order.
- Alternative: omit `(tokens, tile_n)` from `dot` (unsound on hardware).
- Kind: hardware-dependent / scope.

### D-FP8-SUPPORTS: stricter `0 < k`, `0 < n`; pre-broadcast `ws`; exact unsaturated bound
- Decided: `Supports` (`TpuKernelLean/Examples/Fp8LinearSmallM.lean:61-77`) adds `k_pos : 0 < k`
  and `n_pos : 0 < n` beyond Python `supports()` (`fp8_linear_small_m.py:30-31`), accepts any
  candidate in `[1024, 512, 256, 128]` rather than only `_tile_n`'s first match (`:24-27`),
  assumes `ws` is already broadcast to `[1, n]` (`:55-57`), and omits the `tokens > 16` XLA
  fallback. `fp8LinearScaled_abs_le_exact` (`:541`) justifies omitting `jnp.clip` (`:36-37`) in
  exact arithmetic only (not fp32 rounding near `fp8Max`).
- Basis: positive dims are required for non-empty blocks/grid; exact arithmetic avoids an IEEE
  rounding model.
- Alternative: tighten Python `supports()` to check `k > 0 and n > 0` and verify rounded scaling.
- Kind: stricter (`k > 0, n > 0`) / scope.

## `RouterTopk` (`sglang_tpu/kernels/router_topk.py`)

### D-ROUTER-TIE: the selection order and its tie rule are the spec
- Decided: `mathRow` (`TpuKernelLean/Examples/RouterTopk.lean:236`) selects by `Before`
  (`:51`): `j` precedes `e` when `c e < c j`, or `c j = c e` and `j < e`. `rank` (`:80`) counts the
  experts before `e`, `atRank` (`:135`) returns the expert of a given rank, and
  `atRank_before_atRank` (`:155`) and `atRank_before_unselected` (`:162`) restate the order and
  that no unselected expert beats a selected one. The kernel's own rank formula
  (`router_topk.py:121-131`) is proved equal to `Before` (`tileBefore_iff`, `:403`), not assumed.
- Basis: the module docstring (`router_topk.py:5-6,97-99`) and the benchmark task contract
  (ties go to the lowest expert id; ids in selection order). `rank` is the position in a stable
  descending sort, the formulation of the benchmark's float64 oracle.
- Alternative: specify only the selected set; that would not pin the returned order or ties.
- Kind: scope.

### D-ROUTER-EXACT: compares, max, integer min, `where` and one-term sums are exact
- Decided: only the float arithmetic is opaque (`RouterOps`, `:177`: casts, `exp`, `+`, `/`, `*`,
  the renormalization `rowSum`). `>`, `>=`, `==`, `jnp.max` of f32 values, every integer op and
  `jnp.where` are exact. `jnp.sum(jnp.where(hit, scores, 0.0))` (`router_topk.py:81,140-144`) is the
  real sum, because at most one term is non-zero and adding exact zeros is exact in IEEE.
- Basis: IEEE comparison and `max` return or compare the operands themselves; `x + 0 = x` for every
  finite `x` (`-0.0` and `+0.0` are one value here, D-CORE-REAL-MEM).
- Alternative: opaque compares and reductions; the selection could then not be stated.
- Kind: hardware-dependent (IEEE behaviour of Mosaic's compare, max and add).

### D-ROUTER-NEGINF: the `-inf` mask is `⊥` of `WithBot ℝ`, inside the body only
- Decided: the round loop keeps `choice` as `Fin E → WithBot ℝ` (`RoundState`, `:254`) and
  masks a picked lane with `⊥` (`router_topk.py:61,84`). Memory never holds `-inf`.
- Basis: `-inf` is below every finite f32 and equal to itself, which is what `max` and `==` use.
- Alternative: an extended-real memory model.
- Kind: scope.

### D-ROUTER-PACKED: bf16 keys through opaque bitcasts, one stated bit fact, `experts < 32768`
- Decided: the packed path (`router_topk.py:63-69,76-78`) is transcribed with opaque
  `f32Bits`/`f32OfBits` (`:177`), `>> 16` as floor division by `65536` and `& 0xFFFF` as the
  non-negative remainder. The proof needs `RouterOps.Bf16BitsExact` (`:201`): a bf16 value
  promoted to f32, cut to its top 16 bits and shifted back, bitcasts to itself. `Supports.packed_fits`
  (`:369`) requires `experts < 32768` with bf16 logits, so `expert * 2^16 + bits` and
  `no_key = experts * 2^16` fit in int32 and the model's unbounded integers agree with the kernel.
- Basis: an IEEE f32 that is a bf16 value has zero low 16 bits. Memory holds reals, so the
  fact is stated, not proved; the e2e test runs the packed path. `RouterOps.exact` (`:208`) does not
  satisfy it, so the end-to-end witness uses f32 logits.
- Alternative: a bit-level float model.
- Kind: hardware-dependent / launch obligation (`experts < 32768`).

### D-ROUTER-ROWS: row-by-row body; rank tiles summed over lanes
- Decided: `body` (`:347`) computes each output row from its logits row with `kernelRow`
  (`:308`); `rowSum tokens k` is parameterised by the `[tokens, k]` shape. The rank path's double
  loop over 128-lane tiles (`router_topk.py:121-133`) is one sum over lanes `j` with the tile of `j`
  and `e` read from `j / 128` and `e / 128` (`tileRank`, `:296`). The transposes and broadcasts of
  `router_topk.py:116-117,145-146` move data only and are not modelled.
- Basis: every op of `_topk_kernel` acts along the expert axis; the dispatch to the rank path
  (`router_topk.py:35`, `UseRankPath`, `:300`) requires `experts % 128 == 0`, so the tiles cover
  every lane once.
- Alternative: transcribe the `[tokens, experts]` arrays and the tile loops literally.
- Kind: scope.

### D-ROUTER-SCOPE: `0 < tokens`, `0 < k <= experts`; no exact-arithmetic theorem
- Decided: `Supports` (`:365`) requires `0 < tokens`, `0 < k`, `k <= experts`. With `k > experts`
  the round loop would pick an already masked lane again; the kernel does not check this. No
  `*_exact` theorem links `score` to the real sigmoid: the claim is the bitwise one over
  `RouterOps`. The bias is the f32 buffer the `pallas_call` receives; the host-side
  `bias.reshape(experts).astype(float32)` (`router_topk.py:190`) is not modelled.
- Basis: top-k of fewer than `k` experts has no meaning for the caller; positive sizes are
  required for non-empty blocks (D-CORE-GRID-EMPTY).
- Alternative: specify the repeated-lane output for `k > experts`.
- Kind: stricter (`k <= experts`) / scope.

## `TrueOnPolicyNorm` (PR 2299 `sglang_tpu/rl/true_on_policy/norm_kernel.py`)

### D-NORM-PR2299: written against the PR 2299 head, repinned to `main` after the merge
- Decided: `TpuKernelLean/Examples/TrueOnPolicyNorm.lean` models `_norm_kernel`,
  `_gated_norm_kernel` and `_norm_call`. The proof was written against PR head `0f138b0`; the PR
  merged on 2026-11-24 and `manifest.json` now pins `main` at `b5391b8`. The only changes in
  between are in the torch backward (`_norm_input_grad` and its callers), which is not modelled,
  as are the torch wrappers `rms_norm` and `gated_rms_norm`.
- Basis: diff of `norm_kernel.py` between the two commits.
- Alternative: model the wrappers and the backward.
- Kind: scope.

### D-NORM-HW: per-row `rowMean` independence across `M` and `rows`
- Decided: `FOps` (`TpuKernelLean/Examples/TrueOnPolicyNorm.lean:27-40`) assumes `astype(float32)`
  on bf16 is exact (`id`) and Mosaic's `jnp.mean(..., axis=1)` over width `H` produces identical
  bits regardless of `M` and row block size `rows`.
- Basis: PR 2299's batch-invariance contract; depends on Mosaic lowering on TPU.
- Alternative: parameterize `rowMean` by `rows` if Mosaic lowers reductions differently by tile.
- Kind: hardware-dependent.

## `TrueOnPolicyMatmul` (PR 2299 `sglang_tpu/rl/true_on_policy/matmul_kernel.py`)

### D-MATMUL-PR2299: written against the PR 2299 head, repinned to `main` after the merge
- Decided: `TpuKernelLean/Examples/TrueOnPolicyMatmul.lean` models `_mm_kernel` and
  `_jax_matmul`. The proof was written against PR head `0f138b0`, where the N block was the
  constant `_MM_BN`; the PR merged on 2026-11-24 with `bn = row_block(n, _MM_BN, 128)`, and
  `manifest.json` now pins `main` at `b5391b8`. The Lean `gridSpec` takes `bn` as a parameter
  (`wSpec bn`, `oSpec bm bn`, scratch `[bm, bn]`), so it covers both. The torch wrapper and the
  backward (`_matmul_grad`) are not modelled.
- Basis: diff of `matmul_kernel.py` between the two commits, compared field by field with the
  Lean `gridSpec`.
- Alternative: model `row_block` and prove which `bn` it picks.
- Kind: scope.

### D-MATMUL-HW: exact `K = nk * BK` and tile-independent MXU `dot BK`
- Decided: `gridSpec` (`TpuKernelLean/Examples/TrueOnPolicyMatmul.lean:66-72`) assumes
  `K = nk * BK`, and `MOps.dot` (`:24-30`) assumes one `BK`-wide MXU `dot_general` step yields
  the same fp32 partial for `(r, c)` regardless of `M`, `N`, `bm`, `bn`, and other rows/columns.
- Basis: PR 2299 fixes `BK = 512` with explicit K-loop accumulation to hold reduction order fixed.
- Alternative: parameterize `dot` by `(bm, bn)` if MXU lowering varies with tile shape.
- Kind: launch obligation (`K = nk * BK`) / hardware-dependent.

## `FlashAttention` (`sglang_tpu/kernels/flash_attention.py`)

### D-FA-ALGO: algorithm-only unnormalized online softmax
- Decided: `TpuKernelLean/Examples/FlashAttention.lean` proves only that an unnormalized
  online-softmax recurrence over `n + 1` KV blocks equals exact attention
  (`runBlocks_normalize_eq_attentionRow` at `:258`). It is not tied to `pallas_call`, does not
  model causal/segment/bias masks or the single-step kernel
  (`flash_attention.py:224-225,376-445`), and differs from `flash_attention.py:357-363`, which
  renormalizes `acc_scratch_ref` on every step instead of once at the end (`store_output` at
  `:365-372`).
- Basis: prototype algorithm proof; full kernel requires `-inf` scratch support in
  `Pallas/Core.lean` (`flash_attention.py:259-265,562-566`).
- Alternative: model the per-step renormalized recurrence and tie it to `pallasCall`.
- Kind: scope.

## `BlockScaledMm` (two block-scaled FP8 matmul kernels; sources not public)

### D-BSM-PADDED: theorems are stated on the padded arrays the `pallas_call` sees
- Decided: `blockwiseScaledMm_correct` and `blockwise128ScaledMm_correct` take `a [gm*tile_m,
  gk*tile_k]`, `b [gk*tile_k, gn*tile_n]`, block-major scales and the padded bias as given. Any
  zero-padding, scale expansion, axis swap or final slice done by a wrapper around the
  `pallas_call` is not modelled, and neither is a transposed `b` layout.
- Basis: the Pallas-level scope of this library. With opaque ops a zero-padded K block need not
  contribute exactly zero, so "padding is harmless" is only an exact-arithmetic fact.
- Alternative: model the wrappers in exact arithmetic and prove the unpadded formula.
- Kind: scope.

### D-BSM-TILEK: `tile_k` is a positive multiple of the block size
- Decided: the model takes `tile_k = steps_k * block_size` with `steps_k > 0`.
- Basis: with a `tile_k` that is not a multiple of the block size, a body that unrolls
  `tile_k // block_size` blocks covers only part of each K tile. The caller must ensure
  divisibility.
- Alternative: none in the model.
- Kind: launch obligation.

### D-BSM-OPS: the per-block update is one opaque op per unrolled iteration
- Decided: `ScaledMmOps.update tile_m tile_n steps_k i` stands for `acc + dot * sa * sb` of
  unrolled iteration `i`, and `dot` carries the tile shape and block size. `*_elem_independent`
  is claimed at one compiled shape only.
- Basis: the compiler may fuse the multiply-add and may lower each unrolled iteration
  differently; the MXU accumulation order may change with the tile shape. `*_exact` is a
  statement about the algorithm, not about what the hardware computes.
- Alternative: separate `mul` and `add` ops (assumes no fusion).
- Kind: hardware-dependent.

## `GroupedScaledMm` (a grouped row-scaled FP8 matmul kernel; source not public)

### D-GSM-SCHEDULE: the visit schedule is a precondition, not proved
- Decided: `rowwiseScaledGroupedMm_correct` assumes `ValidSchedule` on the scalar-prefetched
  tables: group offsets start at 0, are non-decreasing and end at the row count; every visit names
  a group and an M tile; M tiles are non-decreasing across visits; every row of every group is
  covered by a visit of that group to the row's M tile. The host-side code that builds the tables
  is not modelled.
- Basis: the kernel is correct only for such tables. Repeated visits with the same group and M
  tile are harmless, because a visit's tile value does not depend on earlier state.
- Alternative: model the schedule builder and prove it produces a valid schedule.
- Kind: launch obligation.

### D-GSM-FIRST-REAL: the first visit must belong to a real group
- Decided: `ValidSchedule.first_real` requires the group of visit 0 to be a real group, which
  excludes inputs where every real group is empty.
- Basis: a virtual-group visit leaves the scratch accumulator untouched. The model's
  `parallel_scratch_reset` rule requires the first step of each parallel group to ignore incoming
  scratch, so such a first visit is undefined in the model even though it never reads the scratch.
- Alternative: weaken `parallel_scratch_reset` to "scratch is not read before it is written".
- Kind: stricter (a limitation of the model's rule, not of the kernel).

### D-GSM-PERSIST: the proof relies on the output block persisting across visits
- Decided: visits to one M tile are consecutive and share an output block index, so the model
  keeps the VMEM block between them (`PAL-OUT-PERSIST`). Each visit writes its own group's rows and
  passes the others through; on the first visit the passed-through rows are uninitialised and are
  overwritten later or lie in the padding.
- Basis: `docs/pallas/pipelining.md` in JAX at 35066e1: when consecutive iterations use the same
  slice, no copy is issued and the same buffer is passed to the kernel again.
- Alternative: none; the kernel depends on this behaviour too.
- Kind: hardware-dependent.

### D-GSM-ROUNDTRIP: re-storing an output element does not change it
- Decided: `GroupedMmOps.cast_up_cast` assumes `cast (up (cast x)) = cast x`: reading an already
  stored output element as fp32 and casting it back gives the same element.
- Basis: widening bf16 to fp32 is exact, so the round trip is the identity; for an fp32 output
  both casts are the identity.
- Alternative: none without a bit-level value model.
- Kind: hardware-dependent.

### D-GSM-SCOPE: padded arrays, one layout, no claim for padding rows
- Decided: the theorem is stated on the padded arrays the `pallas_call` sees, for `b` stored
  `[G, K, N]`. Rows at or beyond the row count `M` carry no claim (they hold uninitialised data
  that the caller slices off). Int32 arithmetic on rows is modelled in ℤ without wraparound.
- Basis: the Pallas-level scope of this library.
- Alternative: model the wrapper's padding and slice.
- Kind: scope.
