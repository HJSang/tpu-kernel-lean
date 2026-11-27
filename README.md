# TpuKernelLean

A Lean 4 + Mathlib library for specifying TPU Pallas kernels (`pl.pallas_call`) and proving them correct. The kernels it covers come mostly from sglang-torchtpu (inference and true-on-policy RL).

The TPU instruction set is closed, so proofs are at the **Pallas level**: `pallas_call` is modelled in Lean, each kernel body is written as a Lean function, and a theorem states that the kernel meets a tensor-level spec with an explicit memory layout and frame condition. The spec design follows [TensorLean](https://tensorlean.ai). Checking a proof needs only a CPU.

## What is verified

Paths starting `sglang_tpu/` are in `sgl-project/sglang-torchtpu`.

| Kernel | Theorems | Result |
| --- | --- | --- |
| `sglang_tpu/kv_page_write.py` | `writePages_correct` | Defined and correct for `M > 0` when page ids are distinct int32 naturals below `N` (checked decoding: a non-integer or out-of-range id is undefined behaviour) |
| `sglang_tpu/rl/true_on_policy/norm_kernel.py`: `rms_norm` (plain, Gemma, with residual), `gated_rms_norm` | `rmsNorm_correct`, `rmsNormResidual_correct`, `gatedRmsNorm_correct`, `rmsNorm_batch_invariant`, `normRow_exact` | Defined and correct for every positive row count and row block; bit-for-bit batch invariant* |
| `sglang_tpu/rl/true_on_policy/matmul_kernel.py` | `matmul_correct`, `matmul_batch_invariant`, `matmulElem_exact` | Element `(r, c)` is the in-order fp32 accumulation over `K` steps, for every positive `M`, `N`, `bm`, `bn`; bit-for-bit batch invariant* |
| `sglang_tpu/kernels/fp8_linear_small_m.py` | `fp8Linear_correct`, `fp8Linear_fixedShape_row_independent`, `fp8LinearElem_exact` | Per-row dynamic fp8 quantization then fp8 matmul: defined and correct for every shape the Python `supports` accepts (plus `k > 0`, `n > 0`). At one compiled shape `(tokens, k, tile_n)` an output row depends only on its own `x` row. No claim across batch sizes: the untiled dot may change accumulation order with the block shape |
| `sglang_tpu/kernels/router_topk.py` (`router_topk_jax`) | `routerTopk_correct`, `routerTopkSpec_correct`, `kernelRow_eq_mathRow`, `tileBefore_iff`, `atRank_before_*` | Sigmoid MoE routing with a correction bias. Defined and correct on all three code paths (one-token rank, bf16 packed-key rounds, plain rounds) for every `tokens > 0` and `0 < k ≤ experts` (bf16 logits also need `experts < 32768`, so the int32 keys cannot overflow): `ids[t, i]` is the expert at position `i` of the order by biased score descending, then expert id ascending (equal scores go to the lower id), and `weights[t, i]` is its unbiased score, renormalized. Bit for bit*, given the IEEE bf16 bit-pattern fact `Bf16BitsExact` on the bf16 path |
| `sglang_tpu/qk_norm_rope.py` (`_qk_norm_rope_jax`, not the torch wrapper) | `qkNormRope_pallas_correct`, `qkNormRope_v_unchanged`, `qkNormRope_batch_invariant`, `qkNormRope_exact_correct` | Defined and correct for every positive token count and block size, both q and k outputs; `v` columns are never read; bit-for-bit batch invariant*; with exact arithmetic and `eps > 0`, `rmsNorm` then `ropeNeox` |
| `sglang_tpu/kernels/flash_attention.py` | `runBlocks_normalize_eq_attentionRow` | Algorithm only: online softmax over `n` blocks equals attention. Not tied to the kernel, which renormalises every step |
| Two block-scaled FP8 matmul kernels (MXFP8-style with any block size; block 128 with or without bias). Their sources are not public and are not referenced | `blockwiseScaledMm_correct`, `blockwise128ScaledMm_correct`, `*_elem_independent`, `blockwiseScaledMmElem_exact`, `blockwise128ScaledMmElem_exact` | On the padded arrays the `pallas_call` sees: defined for all positive grid and tile sizes; each output element is the in-order accumulation over K tiles and K blocks. At one compiled shape, element `(r, c)` depends only on row `r` of `a`, column `c` of `b` and its scale (and bias) entries*. With exact arithmetic it is `Σ_b Sa[b,r]·Sb[b,c]·Σ_{k∈b} A[r,k]·B[k,c]` plus the bias |
| A grouped row-scaled FP8 matmul kernel (each group owns a row range; the visit schedule comes from scalar-prefetched tables). Its source is not public and is not referenced | `rowwiseScaledGroupedMm_correct`, `groupedElem_exact`, `ValidSchedule.exists_owner` | Given a valid visit schedule (`ValidSchedule`, a stated precondition; the code that builds it is not modelled): defined, every row of a real group is `cast(scale(acc, scale_a[r], scale_b[g,c]))` with `acc` the in-order accumulation over K tiles, rows past the last group are `cast 0`, and nothing outside the output buffer changes. With exact arithmetic the element is `scale_a[r]·scale_b[g,c]·Σ_k a[r,k]·b[g,k,c]`. No claim for padding rows, or when every real group is empty |

A well-defined kernel also gives the same result when its `parallel` grid dimensions are split across cores (`pallasExecPar_eq_pallasExec`), so every theorem above carries over to that case.

### Two kinds of numeric claim

- **Bit-for-bit claims** (`*_batch_invariant`, `*_row_independent`, and the `*_correct` theorems stated over `FOps`, `MOps` or `FP8LinearOps`) keep the float operations opaque, so they hold for whatever rounding the hardware does. \* They rest on assumptions about the compiler and hardware that Lean cannot check: each op is a deterministic function of its arguments; Mosaic lowers an op the same way for every block shape the theorem ranges over; a row mean depends only on its row; one MXU `dot` step for one output element depends only on its two input rows. `DECISIONS.md` lists them per kernel.
- **Exact-arithmetic claims** (`*_exact`, `FlashAttention`) instantiate the ops with real arithmetic. They show the kernel computes the right function. They are not what the hardware computes: rounding, overflow, NaN and signed zero are outside them.

## How it is checked

All declarations depend only on the standard axioms (`propext`, `Classical.choice`, `Quot.sound`). `lakefile.toml` enables `warningAsError = true` so Lean warnings fail the build. CI (`.github/workflows/lean.yml`) runs `lake build`, then `bash scripts/ci.sh`:

- `lake env leanchecker` independently replays every declaration of every module under `TpuKernelLean/` through the kernel before the axiom gate runs.
- `scripts/CheckAxioms.lean` runs in two phases so no library module is loaded with initializers enabled until source scanning passes:
  1. **Source scan (`--scan-only`, trusted parser environment only):** parses every source file with Lean's parser and rejects parse errors, `import Lean*`/`Lake*`, `example`, metaprogramming and notation commands (`elab`, `elab_rules`, `macro`, `macro_rules`, `syntax`, `declare_syntax_cat`, `notation`, `infix*`/`prefix`/`postfix`, `run_cmd`, `run_elab`, `run_meta`, `initialize`, `builtin_initialize`, `#eval`), `set_option` outside the allowlist, non-ASCII characters in declaration names, and forbidden tokens such as `sorry` and `native_decide`.
  2. **Environment audit (`--check-env`):** checks every declaration in the library (public or private) for non-standard axioms, `unsafe`/`partial`, `opaque`, `@[extern]`, `@[implemented_by]`, and non-ASCII names; checks trusted-module imports and that every source file is imported by `TpuKernelLean.lean`; enforces protected model names and namespaces; rejects truncating `ℕ` subtraction or `Nat.pred` in `BlockSpec.indexMap` and `ReductionKernel.outIndexMap`; and audits every `correct`, `other`, and `witness` theorem in `manifest.json` (verifying compiled existence, non-trivial types, `TpuKernelLean.Pallas.pallasCall` reachability, and that each `witness` in `TpuKernelLean.Tests.*` shares a kernel definition with its `correct` theorem).
- `scripts/test-gate.sh` runs that gate on the files under `scripts/gate-fixtures/`: one clean file that must pass and ten that must be rejected for a stated reason.
- `scripts/check-ledger.sh` checks that every `PAL-` rule id in `TpuKernelLean/` has a matching heading in `docs/pallas-model-ledger.md` and vice versa.
- `scripts/check-readme.sh` checks every row of the *What is verified* table above against the `manifest.json` theorem registry (every listed theorem or glob must be in that row's manifest entry, and every `correct` theorem in `manifest.json` must appear in the README).
- `scripts/check-manifest.sh --offline` checks `manifest.json` structure, required fields (including `correct`, `other`, `witness`), SHA formats, `level` values, recursive coverage of `TpuKernelLean/Examples/**/*.lean`, unique `(repository, path)` pins, `e2e/test_<key>.py` existence for public `pallas_call` entries, module file existence, and that `unreferenced` entries are on the explicit allowlist and omit `repository`, `path`, and `commit`.
- `scripts/test-checks.sh` runs positive and negative fixture tests for the three shell checks above.
- `scripts/test-attacks.sh` runs reduced negative fixtures under `scripts/attack-fixtures/` for the red-team attack classes (non-Pallas runners, namespace shadowing, truncating `ℕ` index subtraction, provenance spoofing, kernel-checker bypasses and `initialize`, check-script gaps, `local notation`, and Unicode homoglyphs) and verifies each is rejected with the expected reason.

### End-to-end differential tests

CI (`.github/workflows/e2e.yml`) runs `bash scripts/e2e.sh`, which fetches each public `pallas_call` kernel from `sgl-project/sglang-torchtpu` at the commit pinned in `manifest.json`, checks its git blob SHA, and runs `pytest e2e -q` on CPU using JAX's TPU interpret mode (`pltpu.InterpretParams` with race detection, out-of-bounds read checks, and NaN-filled uninitialized memory). Each test compares the real Python kernel's output on deterministic inputs (including edge blocks where the kernel accepts non-divisible shapes) against a NumPy reference implementing the formula of the kernel's Lean theorem (`writePages_correct`, `normRow_exact`, `matmulElem_exact`, `fp8LinearMath_exact`, `qkNormRope_exact_correct`, `routerTopk_correct`).

- **Covered:** `kv_page_write`, `true_on_policy_norm` (plain, Gemma, residual, and gated), `true_on_policy_matmul`, `fp8_linear_small_m`, `qk_norm_rope` (`_qk_norm_rope_jax`), and `router_topk` (`router_topk_jax`, all three code paths). Host/TPU registration imports are stubbed out so the tests call the JAX-level functions directly on CPU; the shared block-sizing helper `sglang_tpu/rl/true_on_policy/common.py` is fetched at the same pinned commit.
- **Not covered:** `flash_attention` (algorithm-only proof, not tied to `pallas_call`); `block_scaled_mm` and `grouped_scaled_mm` (marked `source: unreferenced` in `manifest.json`, with no public Python source to fetch).
- **What this does and does not establish:** a passing run catches layout, indexing, scaling, or formula mismatches between the real Python kernels and the Lean theorem statements on the tested shapes. It does not prove general correctness: it checks only finite inputs, runs in JAX's CPU TPU interpret mode rather than compiled Mosaic/libtpu on TPU hardware, and compares against a third transcription of the spec in NumPy (it does not execute the noncomputable Lean `pallasCall` model itself).

## Provenance

[manifest.json](manifest.json) pins each example to its Python file: repository, commit, path, git blob SHA, the functions modelled and the ones not modelled. `bash scripts/check-manifest.sh` (run weekly by `.github/workflows/manifest.yml`) fails if a pinned hash is wrong and reports `STALE` when the upstream file has changed since the proof was written. An example whose kernel source is not public has an entry with no repository, path or commit, and the check skips it. [DECISIONS.md](DECISIONS.md) records each modelling choice: what was decided, on what basis, the alternative, and whether it narrows scope, is stricter than Pallas, or depends on hardware.

## Build

Lean `v4.37.0-rc1` and Mathlib are pinned in `lean-toolchain` and `lake-manifest.json`. Build on a Linux devbox, not a laptop:

```bash
curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
lake exe cache get   # prebuilt Mathlib, about 9k files
lake build
bash scripts/ci.sh   # the gate and its fixture tests, as CI runs them
```

## Specs

A `KernelSpec I S O` (`Spec.lean`) has four parts:

1. `math`: outputs as a function of state and inputs.
2. `mutate`: the new value of in-place state (for example a KV cache).
3. Layout: `readIn`, `readSt`, `readOut` read tensors from memory; `footprint` is what the kernel may write.
4. `requires`: the caller's preconditions.

`KernelSpec.WellFormed` ties the layout to `footprint`: `readOut` and `readSt` depend only on addresses inside `footprint` (`Mem.agreeOn`), and `readIn` depends only on addresses outside `footprint` (`Mem.agreeOutside`).

`KernelSpec.Correct run` says: for every junk oracle and every memory meeting `requires`, `spec.WellFormed` holds, the run is defined, outputs and post-state equal `math` and `mutate` exactly, and every address outside `footprint` is unchanged. Approximate tolerance (`withinTol ε`) is not available in `KernelSpec` yet; a tolerance form can return once a floating-point rounding model is in place.

## The Pallas model

`Pallas/Core.lean`. Each rule carries a `PAL-` id and is compared with the JAX source at commit `35066e1` in [docs/pallas-model-ledger.md](docs/pallas-model-ledger.md). `pallasCall` returns `none` (undefined behaviour) unless the run is `GridSpec.WellDefined`, otherwise `some` of the sequential execution `pallasExec`:

- `grid_pos`, `block_specs_ok`: every grid dimension is positive, and every `BlockSpec` has its array's rank and positive block dimensions. An empty grid is undefined behaviour, so no kernel is correct vacuously.
- `blocks_ok`: every block index at every grid step has the array's rank, is non-negative and has at least one element in bounds. Block indices and scalar prefetch are integers, so `-1` is out of bounds rather than clamped.
- `outputs_consecutive`: writes to an output block happen in consecutive grid steps.
- `body_arity`: the body returns one block per output and one value per scratch buffer.
- `aliases_ok`: `GridSpec.aliases` uses Python's argument numbering (the scalar-prefetch buffers `scalarPrefetchBufs` come first; their count cannot drift from the buffers read). Each entry pairs an input and an output with the same buffer, array shape, block shape and block index at every grid step, and does not target a scalar-prefetch argument; no output is aliased twice; outputs use distinct buffers, none of them a scalar-prefetch buffer; and an input and an output share a buffer only if they are aliased.
- `no_stale_read`: no input reads an HBM region that an output block of any earlier step covers, whether or not that block has been written back yet. Mosaic skips the fetch when an input's block index repeats and prefetches before the write-back lands, and JAX's two interpreters disagree on what an aliased input then sees, so such reads are forbidden. Under this rule every input read sees the initial memory.
- `dim_semantics_len`, `dim_semantics_prefix`: `dimSemantics` (`parallel` or `arbitrary`, default all `arbitrary`) has one entry per grid dimension, `parallel` ones first (JAX's TPU interpreter assumes this ordering; the compiled lowering does not check it).
- `parallel_outputs_disjoint`, `parallel_read_write_disjoint`, `parallel_scratch_reset`: distinct parallel coordinates write disjoint output regions, never read a region another one writes, and do not depend on incoming scratch at their first step.

Within a defined run:

- The grid runs sequentially in row-major order. `Pallas/Parallel.lean` proves that any split of the parallel dimensions across cores, any interleaving and any flush order gives the same memory.
- Input blocks are loaded at `indexMap(grid point, scalars)`. Edge padding and coordinates outside the block read junk.
- Output blocks are never loaded from HBM. A new block index starts from junk; a repeated index keeps the VMEM block (the accumulator pattern). Blocks are written back, masked to the array, when the index changes and after the last step.
- Scratch buffers (`GridSpec.scratch`, Pallas `scratch_shapes`) start as junk once per launch, persist across steps and are never written to HBM. The body sees `prev = outputs ++ scratch` and returns new outputs, then new scratch.
- VMEM keeps only coordinates inside a block or scratch shape.
- Unaliased outputs start as junk; aliased outputs start as the aliased input.
- Every junk read draws from its own stream, keyed by grid step, role and coordinate, so no proof can rely on two garbage values being equal.

`Pallas/DMA.lean` checks explicit async-copy traces (`make_async_copy(...).start()/.wait()`): `check t = true ↔ HazardFree t`. It is not yet connected to `pallasCall`.

**Trusted, not proved:** that each Lean body matches its Python kernel; that the model matches Pallas (compared rule by rule with the JAX source at commit `35066e1` in the ledger; the compiled path through libtpu is closed); Mosaic lowering, libtpu and the hardware; and, for exact-arithmetic theorems, floating-point rounding.

## Layout

| Path | Contents |
| --- | --- |
| `TpuKernelLean/Tensor.lean`, `Memory.lean`, `Layout.lean` | Tensors over `Idx` shapes, element-addressed memory, row-major layouts |
| `TpuKernelLean/Dtype.lean` | Checked int32/int8 decoding (no flooring or clamping), float dtypes (`f32`, `bf16`, fp8 e4m3/e5m2) with fp8 range limits, and opaque float casts (`CastOps`) |
| `TpuKernelLean/Spec.lean` | `KernelSpec`, `KernelSpec.WellFormed`, `KernelSpec.Correct`, `withinTol` |
| `TpuKernelLean/Pallas/Core.lean` | `GridSpec`, `BlockSpec`, `Body`, `WellDefined`, `pallasExec`, `pallasCall` |
| `TpuKernelLean/Pallas/Lemmas.lean` | `pallasExec_read_out`: for a kernel that rewrites its output blocks from scratch, each output buffer is the in-order sequence of its block stores |
| `TpuKernelLean/Pallas/RowKernel.lean` | Any row-wise kernel (`[rows, H]` blocks, `[1, H]` weight): `RowKernel.correct` and `RowKernel.batch_invariant` |
| `TpuKernelLean/Pallas/ReductionKernel.lean` | Any reduction kernel (outer dimensions tile the output, last grid dimension is the reduction axis, scratch accumulators reset at step 0, output written at the last step): `ReductionKernel.wellDefined` and `ReductionKernel.correct` |
| `TpuKernelLean/Pallas/Parallel.lean` | Multi-core execution `pallasExecPar` (each core has its own VMEM and scratch) and `pallasExecPar_eq_pallasExec` |
| `TpuKernelLean/Pallas/DMA.lean` | DMA hazard traces and checker |
| `TpuKernelLean/Examples/` | The kernels in the table above, one file each |
| `TpuKernelLean/Tests/` | Specs the model must reject or accept: one per soundness bug found in review (`Regression`), scratch, `qk_norm_rope` edge cases, typed values, two more reduction kernels, the parallel rules, and concrete end-to-end runs of every `pallas_call` kernel (`EndToEnd`) |
| `scripts/` | The CI gate, its fixtures, `ci.sh`, the ledger, README, and manifest checks, and their fixture tests (`test-checks.sh`) |
| `manifest.json`, `DECISIONS.md`, `docs/pallas-model-ledger.md` | Source pins, modelling decisions, and the model rules with their JAX sources |

`TpuKernelLean.lean` imports every module.

## Adding a kernel

1. Read the Pallas kernel: grid, `BlockSpec`s, scratch, scalar prefetch, aliases, `dimension_semantics`, body.
2. Write the spec: `math` and `mutate` over tensors, the layout, `requires`. Use opaque ops if you want a bitwise result, and put anything that can vary with the compiled block shape into the op's type.
3. Embed the kernel as a `GridSpec` and a `Body`, citing the Python lines.
4. Prove `GridSpec.WellDefined` from `requires`, rewrite with `pallasCall_of_wellDefined`, then reason about `pallasExec`:
   - row-wise kernel: instantiate `RowKernel` and use `RowKernel.correct`;
   - kernel that rewrites every output block from scratch: `pallasExec_read_out` and `foldl_storeBlock_read_last`;
   - reduction kernel with scratch accumulators: instantiate `ReductionKernel` and use `ReductionKernel.correct`, as `TrueOnPolicyMatmul.lean` does.
5. Import the module in `TpuKernelLean.lean`, add the kernel to `manifest.json` and its decisions to `DECISIONS.md`, and add a row to the table above.
6. Run `lake build` and `bash scripts/ci.sh`. Tests are named theorems; the gate rejects `example`.

## Known gaps

- **Model** (rule by rule in [docs/pallas-model-ledger.md](docs/pallas-model-ledger.md), including the readings no source settles and the probe that would settle each):
  - Stricter than Pallas in places: empty grids, aliased pairs with different block specs, and any input read of a region an earlier output block covers are all undefined behaviour here, so a kernel relying on them cannot be verified.
  - Scalar prefetch is a read-only int32 snapshot taken before the grid runs; mutating a scalar-prefetch ref in the body is not modelled.
  - Not modelled: Mosaic's block-shape alignment rules, VMEM capacity and double buffering, squeezed (`None`) block dimensions as a change of rank, memory spaces other than VMEM windows, `core_axis` in `emit_pipeline` and manual cross-core DMA.
  - The parallel execution model follows the Pallas documentation; the split itself happens in libtpu and has not been probed on hardware.
- **Specs:**
  - Precondition satisfiability is shown for one concrete instance per `pallas_call` kernel (`Tests/EndToEnd.lean`), not characterised in general.
  - Memory holds reals, so `-0.0` and `+0.0` are one value, and `±inf` and NaN are not modelled.
- **Examples:**
  - `FlashAttention` proves an unnormalised-accumulator recurrence; the sglang kernel renormalises every step.
  - The true-on-policy `logprob` kernel (scratch plus `-inf` masking) is not done. The attention, GDN and MoE ops wrap engine kernels that use manual DMA, which the model cannot express.
- **Tooling:** an RC toolchain; the Lean `pallasCall` model itself is noncomputable (differential testing in `e2e/` checks the Python kernels against NumPy transcriptions of the proved formulas rather than executing the Lean model directly); automated checks cannot verify that a spec's `requires` or `math` matches the intended semantics or that a Lean body matches its pinned Python source (see [CONTRIBUTING.md](CONTRIBUTING.md#what-the-checks-cannot-catch)).

## Future work

- Probes on a TPU for the model rules the JAX source does not settle (aliasing, edge-block write-back, the megacore split).
- A floating-point rounding model, to prove `withinTol ε` for concrete `ε`, and a value type with infinities for masking.
- XLA tiled HBM layouts and byte addresses.
- Remote DMA and collective kernels, parameterised by interconnect topology (2D torus on v5e/v6e, 3D on v5p/v7x, slice shape, device-id order).
- SparseCore kernels, which need their own execution model.
- A cross-check of each Lean `GridSpec` against the traced `pallas_call` at concrete shapes, to shrink the trusted embedding.

## License

Apache License 2.0; see [LICENSE](LICENSE).
