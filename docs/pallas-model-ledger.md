# Pallas TPU Model Semantic Ledger

Every semantic rule in `TpuKernelLean/Pallas/Core.lean`, `TpuKernelLean/Pallas/Parallel.lean`, and
`TpuKernelLean/Pallas/DMA.lean` is listed below with its Lean definitions, its source reference in
JAX (`jax-ml/jax` at commit `35066e1`), its classification, and any caveats.

Each rule ID appears in the docstring of the Lean definitions it covers (`[PAL-...]`) and is checked
bidirectionally against this file by `scripts/check-ledger.sh`.

## Classification Legend

- **`equal`** — The Lean model matches JAX/Pallas documentation and lowering/interpreter code at
  commit `35066e1`.
- **`stricter`** — The Lean model rejects more runs (`pallasCall = none`) or assumes more
  nondeterminism (independent junk) than JAX requires. Safe for soundness: any theorem proved in Lean
  still holds for every run the model accepts.
- **`reading`** — Behaviour depends on closed-source Mosaic/libtpu code or is not completely
  pinned down by open-source JAX alone.
- **`interpreter-only`** — Observed in JAX's CPU HLO interpreter (`interpret=True`) or TPU interpret
  mode (`pltpu.InterpretParams`), while the compiled TPU path delegates to closed-source libtpu.

## Summary Table

| ID | Lean definitions | Class | Summary |
| :--- | :--- | :--- | :--- |
| `PAL-GRID-ORDER` | `gridPoints`, `pallasExec` | `equal` | Sequential row-major (lexicographic) grid traversal for `arbitrary` dims |
| `PAL-GRID-EMPTY` | `GridSpec.WellDefined.grid_pos`, `block_specs_ok` | `equal` / `stricter` | Static rejection of empty grids, zero block dims, and rank mismatches |
| `PAL-PAR-SEMANTICS` | `DimSemantics`, `parallelPrefix`, `GridSpec.dimSemantics`, `numParallel`, `parGrid`, `arbGrid`, `WellDefined.dim_semantics_len`, `dim_semantics_prefix` | `stricter` | Per-dim `parallel`/`arbitrary` annotations; `parallel` dims must lead |
| `PAL-PAR-EXEC` | `GridSpec.groupPoints`, `CoreState`, `stepCore`, `pallasExecPar`, `pallasExecPar_eq_pallasExec` | `reading` | Multi-core execution with private VMEM/scratch matches sequential `pallasExec` |
| `PAL-PAR-DISJOINT` | `WellDefined.parallel_outputs_disjoint`, `parallel_read_write_disjoint` | `stricter` | Distinct parallel coords have disjoint outputs and no read-write overlap |
| `PAL-PAR-SCRATCH` | `WellDefined.parallel_scratch_reset` | `stricter` | First arbitrary step of each parallel group ignores prior scratch VMEM |
| `PAL-IN-LOAD` | `loadBlock`, `stepGrid` | `equal` | Input blocks loaded into VMEM at `indexMap(g, scalars)` |
| `PAL-IN-READONLY` | `Body`, `stepGrid` | `stricter` | Kernel body cannot mutate input refs |
| `PAL-OUT-NO-LOAD` | `flushAndPrep` | `equal` / `reading` | Output VMEM is never loaded from HBM; fresh junk on first visit and index change |
| `PAL-OUT-PERSIST` | `OutSlot`, `flushAndPrep`, `stepGrid` | `equal` | Output VMEM persists across consecutive steps with the same block index |
| `PAL-OUT-FLUSH` | `flushAndPrep`, `flushAll` | `equal` | Output VMEM flushes to HBM on block-index change and after the last step |
| `PAL-EDGE-LEGAL` | `blockOk`, `WellDefined.blocks_ok` | `equal` | Every block index has array rank, positive block dims, `0 ≤ i`, and `i * b < d` |
| `PAL-EDGE-READ` | `loadBlock` | `equal` | Out-of-bounds edge elements and coordinates outside `blockShape` read junk |
| `PAL-EDGE-WRITE` | `storeBlock`, `addrInBlock` | `reading` | Output write-back masks to in-bounds array elements and drops OOB padding |
| `PAL-SCRATCH-LIFE` | `GridSpec.scratch`, `stepGrid`, `pallasExec` | `equal` | Scratch VMEM starts as junk once per launch, persists across steps, never flushes |
| `PAL-SCALAR-PREFETCH` | `GridSpec.scalarPrefetchBufs`, `scalarPrefetchFn`, `scalarPrefetch`, `pallasExec` | `reading` | Scalar prefetch reads only declared buffers via checked int32 once before grid |
| `PAL-SCALAR-NUM` | `GridSpec.numScalarPrefetch`, `AliasesValid.entries_ok`, `buf_eq_aliased` | `equal` | Alias input indices count leading scalar-prefetch args (`scalarPrefetchBufs.length`) |
| `PAL-ALIAS-ENTRY` | `AliasesValid.entries_ok` | `stricter` | Aliased `(in, out)` pair has equal buffer, shape, `blockShape`, and `indexMap` |
| `PAL-ALIAS-NO-SCALAR` | `AliasesValid.entries_ok`, `out_disjoint_scalar` | `interpreter-only` / `stricter` | Scalar-prefetch arguments cannot be aliased or targeted by any output |
| `PAL-ALIAS-INJ` | `AliasesValid.outs_nodup`, `AliasesValid.ins_nodup` | `stricter` | Alias relation is one-to-one (`ins_nodup` derived from `outs_nodup` + `out_bufs_nodup`) |
| `PAL-ALIAS-DISJOINT` | `AliasesValid.out_bufs_nodup`, `buf_eq_aliased`, `out_disjoint_scalar` | `equal` | Outputs use distinct buffers; shared in/out buffers must be aliased; disjoint from scalars |
| `PAL-ALIAS-INIT` | `initOutMem`, `initOutMemAux`, `initOutMem_read_aliased`, `flushAndPrep` | `reading` / `interpreter-only` | Aliased output HBM starts from `m0`; unaliased HBM and initial output VMEM start as junk |
| `PAL-OUT-CONSEC` | `noRevisit`, `WellDefined.outputs_consecutive` | `stricter` | Writes to any output block index occur in one consecutive run of grid steps |
| `PAL-BODY-ARITY` | `WellDefined.body_arity`, `makeSlots` | `equal` | Body returns exactly `outSpecs.length + scratch.length` blocks |
| `PAL-NO-STALE-READ` | `blocksOverlap`, `WellDefined.no_stale_read` | `stricter` | No input block at step `t` overlaps an output block written at any step `s < t` |
| `PAL-ALIAS-REPEAT` | `WellDefined.no_stale_read` | `stricter` | Repeated block indices on an aliased pair are also forbidden by `no_stale_read` |
| `PAL-JUNK-INDEP` | `junkAt`, `loadBlock`, `initOutMem`, `flushAndPrep`, `stepGrid` | `stricter` | Distinct uninitialized/OOB read sites draw independent values from `junk` |
| `PAL-UB-NONE` | `pallasCall` | `stricter` | Any run violating `GridSpec.WellDefined` evaluates to `none` (UB) |
| `PAL-FRAME` | `pallasExec_read_non_out_buf`, `pallasExec_agreeOutside_outBufs` | `equal` | Non-output HBM buffers are completely unchanged by `pallasExec` |
| `PAL-DMA-HAZARD` | `DMA.HazardFreeFrom`, `HazardFree`, `checkFrom`, `check` | `stricter` / `interpreter-only` | Standalone async-DMA trace checker forbidding overlapping in-flight accesses and semaphore reuse |

---

## Rules

### PAL-GRID-ORDER
- **Lean definitions:** `TpuKernelLean.Pallas.gridPoints`, `TpuKernelLean.Pallas.pallasExec`
- **JAX source (`35066e1`):**
  - `docs/pallas/tpu/details.rst:90-92`: "Ergo, the grid is generally not processed in parallel, but sequentially, in lexicographic order (though see the `Multicore TPU configurations`_ section for exceptions)."
  - `jax/_src/pallas/hlo_interpreter.py:366-368,490`: sequential `while_loop` incrementing `loop_idx` via `_get_next_indices`.
- **Classification:** `equal`
- **Note:** Applies to `arbitrary` grid dimensions (the default). When leading dimensions are marked `parallel`, `PAL-PAR-EXEC` proves that any multi-core partition, interleaving, and flush order equals the sequential row-major `pallasExec`.

### PAL-GRID-EMPTY
- **Lean definitions:** `TpuKernelLean.Pallas.GridSpec.WellDefined.grid_pos`, `TpuKernelLean.Pallas.GridSpec.WellDefined.block_specs_ok`
- **JAX source (`35066e1`):**
  - `jax/_src/pallas/core.py:719-724`: `"Block shape for {origin} (= {block_shape}) must have the same number of dimensions as the array shape {array_aval.shape}."`
  - `jax/_src/pallas/mosaic/lowering.py:1005-1008`: `"The Pallas TPU lowering currently supports only blocks of rank >= 1."`
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2230-2234`: computes `num_iterations = functools.reduce(jnp.multiply, grid)` (`0` when `0 ∈ grid`, returning uninitialized outputs without invoking the body).
- **Classification:** `equal` (rank check) / `stricter` (empty grid, zero block dimension)
- **Note:** Previously `blocks_ok` was vacuous when `0 ∈ grid`, allowing empty grids with zero-sized blocks or rank mismatches to pass `WellDefined`. `grid_pos` now rejects empty grids unconditionally and `block_specs_ok` checks array/block rank equality and strictly positive block dimensions independently of grid execution. Note that JAX interpreters accept a 0-trip grid with valid block specs and return uninitialized output buffers; rejecting `0 ∈ grid` is slightly stricter on that corner case and matches `grid_pos`.

### PAL-PAR-SEMANTICS
- **Lean definitions:** `TpuKernelLean.Pallas.DimSemantics`, `TpuKernelLean.Pallas.DimSemantics.isParallel`, `TpuKernelLean.Pallas.parallelPrefix`, `TpuKernelLean.Pallas.GridSpec.dimSemantics`, `TpuKernelLean.Pallas.GridSpec.numParallel`, `TpuKernelLean.Pallas.GridSpec.parGrid`, `TpuKernelLean.Pallas.GridSpec.arbGrid`, `TpuKernelLean.Pallas.GridSpec.WellDefined.dim_semantics_len`, `TpuKernelLean.Pallas.GridSpec.WellDefined.dim_semantics_prefix`
- **JAX source (`35066e1`):**
  - `docs/pallas/tpu/pipelining.md:315-316`: `` `dimension_semantics` should be a tuple of same length as `grid` where each entry is either `"parallel"` or `"arbitrary"`. ``
  - `jax/_src/pallas/mosaic/lowering.py:801-811`: defaults to `("arbitrary",) * len(user_grid)` and checks `len(user_grid) != len(dimension_semantics)`.
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2273-2275`: `# NOTE: We assume here that all parallel dimensions appear before all arbitrary dimensions in the grid. (We will have raised an error earlier if this is not the case.)`
- **Classification:** `stricter`
- **Note:** `lowering.py` itself only checks length and allowed names; requiring all `parallel` dimensions to form a leading prefix (`parallelPrefix`) matches TPU interpret mode and is stricter than compiled Mosaic lowering.

### PAL-PAR-EXEC
- **Lean definitions:** `TpuKernelLean.Pallas.GridSpec.groupPoints`, `TpuKernelLean.Pallas.CoreState`, `TpuKernelLean.Pallas.stepCore`, `TpuKernelLean.Pallas.pallasExecPar`, `TpuKernelLean.Pallas.pallasExecPar_eq_pallasExec`
- **JAX source (`35066e1`):**
  - `docs/pallas/tpu/pipelining.md:261-262`: "The separate TensorCores have their own separate VMEM, VREGs, SMEM, SREGs and compute units but *share HBM*."
  - `docs/pallas/tpu/pipelining.md:316`: `` `"parallel"` indicates to Pallas that the iterations of the for loop corresponding to that dimension can be executed independently without affecting the correctness of the program. ``
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2241-2261,2272-2292`: randomizes parallel coordinates and partitions the parallel subgrid across `num_cores_per_device`, running each core's assigned arbitrary-dimension loops with private VMEM.
- **Classification:** `reading` (follows the documentation and the TPU interpreter; the split on hardware happens in libtpu and has not been probed)
- **Note:** `pallasExecPar_eq_pallasExec` proves that under `WellDefined`, any core count `C`, any partition of parallel groups across cores, any initial per-core scratch VMEM, any step-level interleaving trace, and any final flush order produces the exact same HBM memory as sequential `pallasExec`. `core_axis` in `pltpu.emit_pipeline` and cross-core manual DMA are not modelled.

### PAL-PAR-DISJOINT
- **Lean definitions:** `TpuKernelLean.Pallas.GridSpec.WellDefined.parallel_outputs_disjoint`, `TpuKernelLean.Pallas.GridSpec.WellDefined.parallel_read_write_disjoint`
- **JAX source (`35066e1`):**
  - `docs/pallas/tpu/pipelining.md:316`: iterations of a `"parallel"` dimension "can be executed independently without affecting the correctness of the program."
  - `docs/pallas/grid_blockspec.md:213-214`: "When multiple invocations write to the same elements of the output array the result is platform dependent."
  - `jax/_src/pallas/mosaic/interpret/race_detection_state.py:92-130`: detects concurrent read/write and write/write overlaps across cores/DMAs when `detect_races=True`.
- **Classification:** `stricter`
- **Note:** Forbids any HBM output-output or input-output block overlap between distinct parallel coordinates `pg₁ ≠ pg₂`, including benign identical-value writes. This guarantees freedom from write-write, read-write, and write-read races under every core split and interleaving.

### PAL-PAR-SCRATCH
- **Lean definitions:** `TpuKernelLean.Pallas.GridSpec.WellDefined.parallel_scratch_reset`
- **JAX source (`35066e1`):**
  - `docs/pallas/tpu/pipelining.md:261-262`: separate TensorCores have private VMEM, so scratch state cannot be relied upon across parallel iterations scheduled on different cores or in reordered groups.
- **Classification:** `stricter`
- **Note:** Requires that whenever `gs.parGrid ≠ []`, the body at the first arbitrary step `ag₀` of every parallel group is invariant to all scratch inputs. This is a conservative shortcut: it forbids lazy initialization of a scratch buffer on a later arbitrary step even if `ag₀` does not read it until it is initialized.

### PAL-IN-LOAD
- **Lean definitions:** `TpuKernelLean.Pallas.loadBlock`, `TpuKernelLean.Pallas.stepGrid`
- **JAX source (`35066e1`):**
  - `docs/pallas/grid_blockspec.md:80-82`: "Each block index is then multiplied by the corresponding axis size from `block_shape` to get the actual element index on the corresponding array axis."
  - `docs/pallas/tpu/details.rst:94-96`: "When two (lexicographically) consecutive grid indices use the same slice of an input, the HBM transfer for the second iteration is skipped, as the data is already available."
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2440-2453`: reloads input VMEM only on the first step or when `cur_start_indices[j] != prev_start_indices[j]`.
- **Classification:** `equal`
- **Note:** `stepGrid` re-loads every input from HBM on every step; under `PAL-NO-STALE-READ` and `PAL-ALIAS-REPEAT`, HBM at the loaded region still equals `m0`, so re-loading and Mosaic's skip-on-repeat caching agree.

### PAL-IN-READONLY
- **Lean definitions:** `TpuKernelLean.Pallas.Body`, `TpuKernelLean.Pallas.stepGrid`
- **JAX source (`35066e1`):**
  - `docs/pallas/pipelining.md:458-476`: input refs are read-only in pipelined `pallas_call`; writing to an unaliased input ref is not written back to HBM.
  - `jax/_src/pallas/hlo_interpreter.py:482-489`: tracks `alias_input_written` in case a body mutates an aliased input ref.
- **Classification:** `stricter`
- **Note:** `Body` takes `ins` as immutable functions and returns only output and scratch blocks, so mutating an input ref (even an aliased one) cannot be expressed. Kernels must write through their output ref.

### PAL-OUT-NO-LOAD
- **Lean definitions:** `TpuKernelLean.Pallas.flushAndPrep`
- **JAX source (`35066e1`):**
  - `docs/pallas/tpu/pipelining.md:148-158`: outputs are pipelined from VMEM to HBM (double-buffered by default); unaliased outputs are not pre-loaded from HBM.
  - `jax/_src/pallas/hlo_interpreter.py:155,392`: fills unaliased output buffer with uninitialized values (`NaN`).
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2202-2214,2523-2539`: allocates output VMEM as uninitialized (`NaN`) once; does not re-poison VMEM on block-index change.
- **Classification:** `equal` (for first visit of unaliased output) / `reading` (on block-index change VMEM reuse)
- **Note:** On both the first step (`none`) and every block-index change (`oldIdx ≠ newIdx`), `flushAndPrep` gives the body a fresh `junk` block. This is strictly safer than assuming VMEM retains either the previous block's values or HBM values across a block-index change (under double buffering, the next slot holds older or uninitialized contents).

### PAL-OUT-PERSIST
- **Lean definitions:** `TpuKernelLean.Pallas.OutSlot`, `TpuKernelLean.Pallas.flushAndPrep`, `TpuKernelLean.Pallas.stepGrid`
- **JAX source (`35066e1`):**
  - `docs/pallas/tpu/details.rst:98-100,112`: "Multiple invocations of the kernel body can write to the same slice of the output, without any risk of race conditions. ... The output reference can be then used as an accumulator for partial results."
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2523-2539`: keeps `kernel_output_ids[j]` in VMEM when `cur_start_indices == next_start_indices`.
- **Classification:** `equal`
- **Note:** Only coordinates inside `outSpec.blockShape` persist across repeated indices; coordinates outside `blockShape` read fresh junk.

### PAL-OUT-FLUSH
- **Lean definitions:** `TpuKernelLean.Pallas.flushAndPrep`, `TpuKernelLean.Pallas.flushAll`
- **JAX source (`35066e1`):**
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2523-2539`: flushes `kernel_output_ids[j]` to HBM when `(iteration_idx + 1 == loop_bound) | reduce_or(cur_start_indices != next_start_indices)`.
- **Classification:** `equal`
- **Note:** Matches Mosaic's store-on-index-change and end-of-grid flush schedule.

### PAL-EDGE-LEGAL
- **Lean definitions:** `TpuKernelLean.Pallas.blockOk`, `TpuKernelLean.Pallas.GridSpec.WellDefined.blocks_ok`
- **JAX source (`35066e1`):**
  - `docs/pallas/grid_blockspec.md:106-107`: "Note that at least one of the elements in each block must be within bounds."
  - `docs/pallas/grid_blockspec.md:125-128`: `start_idx = block_idx * block_size; # At least one element of the block must be within bounds; assert start_idx < x_size`.
- **Classification:** `equal`
- **Note:** Block indices are `ℤ` (matching Pallas `int32`/`int64` index-map outputs, `jax/_src/pallas/core.py:801`), so negative indices such as `-1` are rejected by `0 ≤ i` instead of being clamped to `0`.

### PAL-EDGE-READ
- **Lean definitions:** `TpuKernelLean.Pallas.loadBlock`
- **JAX source (`35066e1`):**
  - `docs/pallas/grid_blockspec.md:99-105`: "If the block shape does not divide evenly the overall shape then the last iteration on each axis will still receive references to blocks of `block_shape` but the elements that are out-of-bounds are padded on input and discarded on output. The values of the padding are unspecified, and you should assume they are garbage."
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2091-2095` and `jax/_src/pallas/hlo_interpreter.py:392`: pad edge elements with uninitialized values (`NaN`).
- **Classification:** `equal`
- **Note:** Both out-of-array edge elements and local coordinates outside `blockShape` return independent `junk` values.

### PAL-EDGE-WRITE
- **Lean definitions:** `TpuKernelLean.Pallas.storeBlock`, `TpuKernelLean.Pallas.addrInBlock`
- **JAX source (`35066e1`):**
  - `docs/pallas/grid_blockspec.md:101-102`: "the elements that are out-of-bounds are padded on input and discarded on output."
  - `jax/_src/pallas/hlo_interpreter.py:508-510`: slices padded output back to `bm.array_aval.shape`.
  - `jax/_src/pallas/mosaic/pipeline.py:483-501`: "When we have an out of bounds block like this, we need to truncate it to a tile boundary (tiles are (8, 128) along the two minormost dimensions)."
- **Classification:** `reading`
- **Note:** Logically, `storeBlock` masks writes to `a < arr.shape.prod` and `coordInBlock`. On hardware, XLA pads physical HBM tensors to `(8, 128)` tiles and Mosaic truncates edge DMAs to the next tile multiple (`pipeline.py:499-501`), assuming out-of-bounds padding within the physical tile is invisible to JAX consumers.

### PAL-SCRATCH-LIFE
- **Lean definitions:** `TpuKernelLean.Pallas.GridSpec.scratch`, `TpuKernelLean.Pallas.stepGrid`, `TpuKernelLean.Pallas.pallasExec`
- **JAX source (`35066e1`):**
  - `docs/pallas/tpu/pipelining.md:110`: "Scratch buffers are persistent across kernel iterations and are useful for storing intermediate results such as partial accumulations and reductions. A scratch buffer must reside in `VMEM`, `SMEM`, or `SEMAPHORE`."
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2202-2214`: allocates scratch buffers once per call filled with uninitialized values (`NaN`) and never writes them back to HBM.
- **Classification:** `equal`
- **Note:** Coordinates outside the declared scratch shape do not persist across steps (read fresh junk). Only VMEM-style array scratch buffers are modelled (`SMEM` and `SEMAPHORE` scratch types are omitted).

### PAL-SCALAR-PREFETCH
- **Lean definitions:** `TpuKernelLean.Pallas.GridSpec.scalarPrefetchBufs`, `TpuKernelLean.Pallas.GridSpec.scalarPrefetchFn`, `TpuKernelLean.Pallas.GridSpec.scalarPrefetch`, `TpuKernelLean.Pallas.pallasExec`
- **JAX source (`35066e1`):**
  - `jax/_src/pallas/core.py:1094-1101`: "The pallas_call is invoked with: `*dynamic_grid_sizes, *index, *inputs`. The `index` operands are for the scalar prefetch. ... The index map functions are invoked with: `*program_ids, *index`."
  - `jax/_src/pallas/hlo_interpreter.py:446,470-472`: evaluates `index_map` with initial `scalars` and drops `scalars` from the loop carry.
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2152-2166,2365-2367`: allocates `scalar_buffer_ids` in SMEM and passes the same buffer IDs to both the body and `_compute_start_indices`.
- **Classification:** `reading`
- **Note:** `GridSpec.scalarPrefetch` reads only the declared `scalarPrefetchBufs` through `readInt32` once before the grid runs (read-only snapshot). This prevents `scalarPrefetch` from secretly reading arbitrary HBM buffers, and `out_disjoint_scalar` ensures no output writes to a scalar-prefetch buffer. Mutable SMEM scalar-prefetch refs visible to later `index_map` evaluations are not modelled.

### PAL-SCALAR-NUM
- **Lean definitions:** `TpuKernelLean.Pallas.GridSpec.numScalarPrefetch`, `TpuKernelLean.Pallas.GridSpec.AliasesValid.entries_ok`, `TpuKernelLean.Pallas.GridSpec.AliasesValid.buf_eq_aliased`
- **JAX source (`35066e1`):**
  - `jax/_src/pallas/pallas_call.py:1304-1315,1325`: `input_output_aliases` keys index `flat_in_avals` (all `pallas_call` inputs, with the `num_index_operands` scalar-prefetch arguments first).
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2116`: `oi_alias_map = {v: k - len(scalars) for k, v in input_output_aliases}`.
  - `jax/_src/pallas/hlo_interpreter.py:480`: `in_idx_adj = in_idx - len(scalars)`.
- **Classification:** `equal`
- **Note:** `numScalarPrefetch` is derived as `scalarPrefetchBufs.length`, so the alias offset cannot drift from the declared scalar-prefetch buffers.

### PAL-ALIAS-ENTRY
- **Lean definitions:** `TpuKernelLean.Pallas.GridSpec.AliasesValid.entries_ok`
- **JAX source (`35066e1`):**
  - `jax/_src/pallas/pallas_call.py:1304-1324`: validates `i_idx in range(len(flat_in_avals))`, `o_idx in range(len(flat_out_avals))`, and `in_aval.shape == out_aval.shape` and `in_aval.dtype == out_aval.dtype`.
  - `jax/_src/pallas/mosaic/pallas_call_registration.py:500-515`: passes `input_output_aliases` straight to XLA custom-call lowering without checking `BlockSpec` equality.
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2126-2130`: reuses `input_args[oi_alias_map[i]].shape` (input-padded shape) for the aliased output buffer, which can mismatch when input and output `block_shape`s differ.
- **Classification:** `stricter`
- **Note:** In addition to matching buffer and array shape, `entries_ok` requires every aliased `(in, out)` pair to have equal `blockShape` and equal `indexMap g sc` at every grid point `g` and scalar table `sc`. This is stricter than `pallas_call.py`, closing the closed-source libtpu uncertainty over mismatched aliased `BlockSpec`s.

### PAL-ALIAS-NO-SCALAR
- **Lean definitions:** `TpuKernelLean.Pallas.GridSpec.AliasesValid.entries_ok`, `TpuKernelLean.Pallas.GridSpec.AliasesValid.out_disjoint_scalar`
- **JAX source (`35066e1`):**
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2116-2119`: `'Aliasing of scalar prefetch arguments is not currently supported in TPU interpret mode.'` (Note: commit `35066e1` checks `oi_alias_map.keys()`, which are output indices `v`, instead of `oi_alias_map.values()`.)
  - `jax/_src/pallas/pallas_call.py:1304-1324`: does not itself reject `i_idx < num_index_operands`.
- **Classification:** `interpreter-only` / `stricter`
- **Note:** `entries_ok` requires `gs.numScalarPrefetch ≤ p.1` and `out_disjoint_scalar` forbids any output buffer from coinciding with a buffer in `scalarPrefetchBufs`. Safe for soundness even if compiled libtpu accepts aliasing a scalar-prefetch SMEM argument.

### PAL-ALIAS-INJ
- **Lean definitions:** `TpuKernelLean.Pallas.GridSpec.AliasesValid.outs_nodup`, `TpuKernelLean.Pallas.GridSpec.AliasesValid.ins_nodup`
- **JAX source (`35066e1`):**
  - `jax/_src/pallas/pallas_call.py:1304,1341`: `input_output_aliases` is a Python `dict[int, int]`, so input keys are syntactically unique (`ins_nodup`).
  - XLA custom-call output aliasing requires each output buffer to alias at most one input (`outs_nodup`).
- **Classification:** `stricter`
- **Note:** `outs_nodup` is a field of `AliasesValid`; `ins_nodup` is now a derived theorem proved from `entries_ok`, `outs_nodup`, and `out_bufs_nodup`.

### PAL-ALIAS-DISJOINT
- **Lean definitions:** `TpuKernelLean.Pallas.GridSpec.AliasesValid.out_bufs_nodup`, `TpuKernelLean.Pallas.GridSpec.AliasesValid.buf_eq_aliased`, `TpuKernelLean.Pallas.GridSpec.AliasesValid.out_disjoint_scalar`
- **JAX source (`35066e1`):**
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2125-2148`: allocates a fresh HBM buffer for every unaliased output and reuses the input's HBM buffer if and only if the output appears in `oi_alias_map`.
- **Classification:** `equal`
- **Note:** Captures XLA's buffer-aliasing contract in element-addressed `Mem`: distinct outputs occupy distinct buffers, an input and output share a buffer iff paired in `aliases`, and no output overwrites a scalar-prefetch buffer.

### PAL-ALIAS-INIT
- **Lean definitions:** `TpuKernelLean.Pallas.initOutMem`, `TpuKernelLean.Pallas.initOutMemAux`, `TpuKernelLean.Pallas.initOutMem_read_aliased`, `TpuKernelLean.Pallas.flushAndPrep`
- **JAX source (`35066e1`):**
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2126-2148`: in HBM, an aliased output reuses the input's HBM buffer (preserving unwritten regions of `m0`), while an unaliased output starts uninitialized (`NaN`).
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2202-2214`: in VMEM, every pipelined output ref (aliased or not) starts uninitialized (`NaN`).
  - `jax/_src/pallas/hlo_interpreter.py:145-157,449-455`: in HLO interpret mode, slices the aliased output carry from `m0`, so the aliased output ref starts with `m0`'s contents rather than `NaN`.
- **Classification:** `reading` / `interpreter-only`
- **Note:** HBM preservation of untouched regions of an aliased buffer matches both interpreters (`equal`). Giving the body `junk` in VMEM on the first visit of an aliased output (`flushAndPrep`) matches TPU interpret mode and is conservatively safe even if compiled libtpu preloads `m0`.

### PAL-OUT-CONSEC
- **Lean definitions:** `TpuKernelLean.Pallas.noRevisit`, `TpuKernelLean.Pallas.GridSpec.WellDefined.outputs_consecutive`
- **JAX source (`35066e1`):**
  - `docs/pallas/tpu/details.rst:99-100`: "However, we do require that all invocations that write to a particular slice are consecutive."
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2517-2522,2540-2545`: `_check_for_revisiting` enforces non-revisiting for unaliased outputs and suppresses the check for aliased outputs.
- **Classification:** `stricter`
- **Note:** Enforced on both unaliased and aliased outputs (matching `details.rst:99-100` and stricter than TPU interpret mode, which skips `_check_for_revisiting` on aliased outputs).

### PAL-BODY-ARITY
- **Lean definitions:** `TpuKernelLean.Pallas.GridSpec.WellDefined.body_arity`, `TpuKernelLean.Pallas.makeSlots`
- **JAX source (`35066e1`):**
  - `jax/_src/pallas/core.py:1097-1098`: "The kernel function is invoked with: `*index, *inputs, *scratch`." (where `*inputs` holds block inputs followed by block outputs).
  - `jax/_src/pallas/hlo_interpreter.py:457-464`: asserts exact arity of discharged jaxpr invars/outvars.
- **Classification:** `equal`
- **Note:** Prevents the shallow-embedded Lean `Body` from returning too few or too many output/scratch blocks (which would otherwise trigger `makeSlots`'s default zero-fill).

### PAL-NO-STALE-READ
- **Lean definitions:** `TpuKernelLean.Pallas.blocksOverlap`, `TpuKernelLean.Pallas.GridSpec.WellDefined.no_stale_read`
- **JAX source (`35066e1`):**
  - `docs/pallas/tpu/details.rst:94-96`: "When two (lexicographically) consecutive grid indices use the same slice of an input, the HBM transfer for the second iteration is skipped, as the data is already available."
  - `docs/pallas/tpu/pipelining.md:148-158`: double-buffered HBM-VMEM pipelining prefetches upcoming input blocks while earlier output write-backs may still be in flight.
- **Classification:** `stricter`
- **Note:** Forbids any input block at step `t` from overlapping any output block at any earlier step `s < t`. This is a conservative shortcut: it guarantees every input read observes `m0` (so `stepGrid`'s per-step HBM load agrees with Mosaic's input caching and double-buffered prefetch), at the cost of rejecting pipelines that read back an output block written many steps earlier.

### PAL-ALIAS-REPEAT
- **Lean definitions:** `TpuKernelLean.Pallas.GridSpec.WellDefined.no_stale_read`
- **JAX source (`35066e1`):**
  - `jax/_src/pallas/hlo_interpreter.py:473-490`: synchronizes aliased input/output carry buffers after every grid step, so step `s + 1` sees step `s`'s write to an aliased output even when the block index did not change.
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:2440-2453,2523-2539`: skips both the input reload and the output flush when the block index repeats, so step `s + 1`'s input still sees `m0`.
- **Classification:** `stricter`
- **Note:** Previously `no_stale_read` had an exception guard `gs.idxAt o.2 m0 s ≠ gs.idxAt o.2 m0 (s + 1)`, which allowed an aliased `(in, out)` pair to repeat a block index while assuming the input still read `m0` — contradicting `hlo_interpreter.py:473-490` and unverified against closed-source libtpu. Dropping that guard makes `no_stale_read` reject repeated overlapping aliased steps as well.

### PAL-JUNK-INDEP
- **Lean definitions:** `TpuKernelLean.Pallas.junkAt`, `TpuKernelLean.Pallas.loadBlock`, `TpuKernelLean.Pallas.initOutMem`, `TpuKernelLean.Pallas.flushAndPrep`, `TpuKernelLean.Pallas.stepGrid`
- **JAX source (`35066e1`):**
  - `docs/pallas/grid_blockspec.md:102-103`: "The values of the padding are unspecified, and you should assume they are garbage."
  - `jax/_src/pallas/mosaic/interpret/params.py:167-179`: `uninitialized_memory` supports `"nan"` or `"zero"` to expose reads of uninitialized memory.
- **Classification:** `stricter`
- **Note:** `KernelSpec.Correct` universally quantifies over `junk : ℕ → ℝ`, and `junkAt` / `Encodable.encode` key every uninitialized or out-of-bounds read by `(step, role, buffer/slot index, local coordinate)`. No proof can exploit coincidence between two garbage reads.

### PAL-UB-NONE
- **Lean definitions:** `TpuKernelLean.Pallas.pallasCall`
- **JAX source (`35066e1`):**
  - `docs/pallas/grid_blockspec.md:106-107,213-214` and `docs/pallas/tpu/details.rst:98-100`: preconditions violated at runtime are undefined or platform-dependent on TPU because Mosaic does not emit runtime checks.
- **Classification:** `stricter`
- **Note:** Returning `none` whenever `¬ gs.WellDefined body m0` forces every `KernelSpec.Correct` proof to establish `WellDefined` from `spec.requires` first.

### PAL-FRAME
- **Lean definitions:** `TpuKernelLean.Pallas.pallasExec_read_non_out_buf`, `TpuKernelLean.Pallas.pallasExec_agreeOutside_outBufs`
- **JAX source (`35066e1`):**
  - `jax/_src/pallas/pallas_call.py:1330-1348`: `pallas_call` is a pure JAX primitive returning only `out_flat` (with in-place HBM reuse restricted to `input_output_aliases`).
- **Classification:** `equal`
- **Note:** Proves that `pallasExec` never modifies any buffer outside `gs.outSpecs`, matching functional JAX semantics and enabling `KernelSpec.Correct` frame proofs.

### PAL-DMA-HAZARD
- **Lean definitions:** `TpuKernelLean.Pallas.DMA.HazardFreeFrom`, `TpuKernelLean.Pallas.DMA.HazardFree`, `TpuKernelLean.Pallas.DMA.checkFrom`, `TpuKernelLean.Pallas.DMA.check`
- **JAX source (`35066e1`):**
  - `docs/pallas/tpu/pipelining.md:143-145`: "`pltpu.async_copy` starts a copy and returns a descriptor to await later. Its DMA semaphore follows two rules: If you pass a semaphore, you must await it."
  - `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:983-1082`: tracks in-flight async DMAs and checks buffer read/write races via `race_detection_state.py:92-205`.
  - `jax/_src/pallas/mosaic/interpret/shared_memory.py:125,184-185`: TPU DMA semaphores are counting semaphores, allowing multiple in-flight DMAs on the same semaphore.
- **Classification:** `stricter` / `interpreter-only`
- **Note:** Standalone trace model not yet connected to `pallasCall`. `HazardFreeFrom.start`'s `hFresh` condition forbids reusing a semaphore ID while a copy on it is still in flight, which is stricter than TPU hardware's counting DMA semaphores.

---

## Rules JAX has that the model lacks

These rules are enforced by JAX/Mosaic at trace or lowering time (at commit `35066e1`) and are **not** checked by `GridSpec.WellDefined`:

1. **TPU block-shape alignment and rank >= 1** (`jax/_src/pallas/mosaic/lowering.py:1005-1084`, `docs/pallas/grid_blockspec.md:85-91`):
   - Mosaic rejects rank-0 blocks (`rank < 1`).
   - For `rank >= 2`, the last two block dimensions `(bs1, bs0)` must satisfy `(bs0 == as0 or bs0 % 128 == 0)` and `(bs1 == as1 or bs1 % 8 == 0 or retiled)` (where `retiled` allows `bs1 ∈ {1, 2, 4}` when `bs1 >= 32 // bitwidth` on VMEM with `libtpu >= 0.0.54`).
   - For `rank == 1`, `bs0` must equal `as0`, be a multiple of `num_sublanes * num_lanes` (1024), or be a power of 2 at least `num_lanes * (32 // bitwidth)`.
   - `TpuKernelLean` accepts arbitrary positive block shapes, so a Lean-verified kernel with unaligned block dims can fail Mosaic compilation on TPU.
2. **Double buffering, `pipeline_mode`, and VMEM capacity** (`docs/pallas/tpu/details.rst:115-120`, `docs/pallas/tpu/pipelining.md:148-158`, `jax/_src/pallas/core.py:197-255`, `jax/_src/pallas/mosaic/lowering.py:1259-1339`):
   - `BlockSpec.pipeline_mode = pl.Buffered(buffer_count=..., use_lookahead=...)` (default `buffer_count=2`) multiplies VMEM usage for pipelined inputs/outputs.
   - Exceeding VMEM capacity (16 MB+ per TensorCore, including spilled VREGs and scratch) causes a low-level libtpu OOM compile error.
   - `TpuKernelLean` models unbounded VMEM and single-slot persistence without lookahead or buffer counts.
3. **`None` / `Squeezed`, `Element`, `BoundedSlice`, and `Indirect` block dimensions** (`docs/pallas/grid_blockspec.md:239-244`, `docs/pallas/tpu/pipelining.md:203-214`, `jax/_src/pallas/core.py:421-542,726`):
   - In JAX, `None` / `pl.Squeezed()` removes a length-1 dimension from the ref seen by the kernel body (`_get_ref_block_shape`). In `TpuKernelLean`, `None` dims are represented as unsqueezed size-`1` dims.
   - `pl.Element(block_size, padding)` (element-indexed blocks), `pl.BoundedSlice(max_size)` (dynamic block sizes via `pl.ds(start, size)`), and indirect/indexed block dims are not modelled.
4. **Memory spaces (`VMEM`, `SMEM`, `ANY`/HBM, `SEMAPHORE`) and trivial-window restrictions** (`docs/pallas/tpu/pipelining.md:96-111`, `jax/_src/pallas/mosaic/core.py:298-304`, `jax/_src/pallas/mosaic/lowering.py:989-995,1010-1016`):
   - `ANY` / `HBM` operands require a trivial window (`has_trivial_window()`: `block_shape == array_shape` and constant-zero `index_map`) and cannot be dereferenced directly without `sync_copy`/`async_copy`.
   - `SMEM` allows only scalar loads/stores; `SEMAPHORE` holds DMA/barrier semaphores.
   - `TpuKernelLean` treats all `inSpecs`/`outSpecs` uniformly as pipelined VMEM refs and all `scratch` entries as VMEM arrays; `Pallas/DMA.lean` is not yet wired into `pallasCall`.
5. **Dtypes, dtype matching, and sub-32-bit physicalization** (`jax/_src/pallas/pallas_call.py:1317`, `jax/_src/pallas/mosaic/lowering.py:969-978,1025-1029`):
   - JAX requires aliased inputs and outputs to match in `dtype` as well as `shape`, and physicalizes packed/sub-byte/bool dtypes before alignment checks (`packing = 32 // bitwidth`).
   - `TpuKernelLean`'s HBM `Mem` stores mathematical reals `ℝ`; `Dtype.lean` provides checked int32/int8 decoding and opaque float/cast ops, but does not model `-0.0` vs `+0.0` bitcasts, `NaN`/`±inf` propagation, or per-array dtype tags inside `ArrayDesc`.

---

## Open readings and the probe that would settle each

The rules below involve closed-source Mosaic/libtpu behaviour or divergence between JAX's HLO interpreter (`interpret=True`) and TPU interpret mode (`pltpu.InterpretParams`). All probes use legal TPU `(8, 128)` `float32` blocks so compilation errors isolate the rule under test. No Python probe harness is committed in this CPU-only Lean repo; when TPU time is booked, run these probes in all three modes (compiled TPU, `interpret=True`, and `pltpu.InterpretParams(detect_races=True, on_race="raise", out_of_bounds_reads="raise", uninitialized_memory="nan")`) and record the observed hardware/version outcome here.

| Rule ID | Current Status | Minimal Distinguishing Kernel | What Real-TPU Outcomes Settle |
| :--- | :--- | :--- | :--- |
| `PAL-ALIAS-REPEAT` | Now conservatively `stricter` (guard dropped from `no_stale_read`) | `grid=(2,)`, `x` `(8, 128)` filled with `10.0`, `in_spec = out_spec = (8, 128)` at `(0, 0)`, `input_output_aliases={0: 0}`, plus diagnostic output `obs` `(2, 8, 128)`. Step 0 records `x` into `obs[0]` and writes `42.0` to `o`; step 1 records `x` into `obs[1]` without touching `o`. | `obs[1] == 10.0` ⇒ separate VMEM slots with no mid-repeat flush (matches TPU interpret; proves relaxing `no_stale_read` could be considered if ever needed). `obs[1] == 42.0` ⇒ shared VMEM or per-step flush (matches `interpret=True`; confirms dropping the guard was required for soundness). |
| `PAL-ALIAS-ENTRY` | Now conservatively `stricter` (matching `blockShape` and `indexMap` required) | `x` `(16, 128)` with rows `0..7 = 10.0`, `8..15 = 20.0`; `grid=(2,)`, `in_spec=((8, 128), i → (i, 0))`, `out_spec=((16, 128), i → (0, 0))`, `input_output_aliases={0: 0}`. Step 0 writes `99.0` to `o`; both steps record `x` into diagnostic `obs`. Also test non-divisible shape `(10, 128)`. | Compile error ⇒ confirms `AliasesValid.entries_ok` must require matching `BlockSpec`s. Succeeds with `obs[1] == 20.0` vs `99.0` ⇒ determines whether libtpu keeps separate VMEM or exposes step 0's write. |
| `PAL-ALIAS-INIT` | `reading` / `interpreter-only` (model gives VMEM `junk`, conservatively safe) | `grid=(1,)`, `x` `(8, 128)` filled with `10.0`, `in_spec = out_spec = (8, 128)` at `(0, 0)`, `input_output_aliases={0: 0}`, diagnostic `obs` `(8, 128)`. Body copies `o` to `obs` before writing `42.0` to `o`. | `obs == 10.0` ⇒ libtpu preloads aliased `o_ref` from HBM (matches `interpret=True`; model's `junk` is strictly conservative). `obs` is `NaN`/garbage/`0.0` ⇒ uninitialized VMEM (matches model and TPU interpret). |
| `PAL-OUT-NO-LOAD` | `reading` on block-index change (model gives fresh `junk`, conservatively safe) | `grid=(2,)`, `out` `(16, 128)`, `out_spec=((8, 128), i → (i, 0))`, diagnostic `obs` `(2, 8, 128)`. Step 0 reads `o` into `obs[0]` then writes `77.0` to `o`; step 1 reads `o` into `obs[1]` then writes `88.0` to `o`. | `obs[1] == 77.0` ⇒ single VMEM slot reused (matches TPU interpret); `obs[1]` garbage/`NaN`/`0.0` ⇒ double-buffer slot 1 uninitialized (matches `interpret=True`). Both are safely subsumed by fresh `junk`. |
| `PAL-EDGE-READ`, `PAL-EDGE-WRITE` | `reading` for hardware tile-padded HBM writes | `x` `(10, 128)` filled with `5.0`, `grid=(2,)`, `block_shape=(8, 128)` at `(i, 0)`. Body writes `x + 1.0` to `o` and copies full `x` block into `(16, 128)` diagnostic `seen`; repeat with `input_output_aliases={0: 0}` on `(10, 128)`. | Confirms `seen[10:16]` is unspecified padding and `out.shape == (10, 128)` equals `6.0` with no fault or HBM corruption on masked write-back. |
| `PAL-SCALAR-PREFETCH` | `reading` (SMEM mutation visibility) | `num_scalar_prefetch=1`, `idx` `int32[2]` initialized `[0, 0]`, `x` `(16, 128)` with block 0 = `10.0` and block 1 = `20.0`, `grid=(2,)`, `in_spec=((8, 128), (i, s) → (s[i], 0))`. Step 0 writes `s[1] = 1`; both steps copy `x` to `obs[i]`. | `obs[1] == 20.0` ⇒ SMEM mutation is visible to later `index_map` calls (matches TPU interpret; confirms mutable SMEM stays out of scope for the read-only snapshot model). `obs[1] == 10.0` or compile error ⇒ `index_map` sees initial SMEM only (matches `interpret=True` and model). |
| `PAL-ALIAS-NO-SCALAR` | `interpreter-only` / `stricter` | `PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(1,), in_specs=[], out_specs=BlockSpec(memory_space=pltpu.SMEM))`, `input_output_aliases={0: 0}`, `s = jnp.zeros((1,), jnp.int32)`. | Compile error ⇒ confirms `entries_ok`/`out_disjoint_scalar` match Mosaic. Succeeds ⇒ model is safely `stricter`. |
| `PAL-GRID-EMPTY` | `stricter` (statically rejected by `grid_pos` and `block_specs_ok`) | `grid=(0,)`, `x` `(0, 128)` with `block_shape=(8, 128)` vs `block_shape=(0, 128)`. | Confirms compile-time rejection of `(0, 128)` and checks whether compiled Mosaic lowers a 0-trip grid `(0,)` or rejects it too. |
| `PAL-PAR-EXEC` | `reading` (megacore v4/v5p) | `grid=(4,)`, `dimension_semantics=("parallel",)`, `scratch=VMEM((8, 128))`, step writes `program_id(0)` to scratch then copies scratch to `obs[i]`. | On Megacore (v4/v5p), confirms non-sequential / core-local scratch splitting across TensorCores; on single-core v6e/v7x, runs sequentially (`docs/pallas/tpu/pipelining.md:322`). |
| `PAL-DMA-HAZARD` | `stricter` / `interpreter-only` (counting semaphore) | Two `pltpu.async_copy` calls started back-to-back on the same `pltpu.SemaphoreType.DMA` semaphore to disjoint VMEM slots, then awaited. | Succeeds on TPU (counting semaphore), confirming `DMA.HazardFreeFrom.start`'s `hFresh` is safely `stricter`. |
