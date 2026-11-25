import TpuKernelLean.Examples.KvPageWrite
import TpuKernelLean.Pallas.DMA

/-!
Regression tests for the soundness bugs S1–S8 found in review. Each was once a machine-checked
"proof" of behaviour a real TPU does not have; each test now proves the model rejects it. The DMA
cases (S5) live next to the checker in `Pallas/DMA.lean`.
-/

namespace TpuKernelLean.Tests

open TpuKernelLean.Pallas TpuKernelLean.Examples.KvPageWrite

private theorem not_wd_of_bad_out_block {gs : GridSpec} {body : Body} {m0 : Mem} {k : ℕ}
    (hk : k < (gridPoints gs.grid).length) {p : ArrayDesc × BlockSpec} (hp : p ∈ gs.outSpecs)
    (hbad : blockOk p.1.shape p.2.blockShape (gs.idxAt p.2 m0 k) = false) :
    ¬ gs.WellDefined body m0 := fun h => by
  have := h.blocks_ok k hk p (List.mem_append_right _ hp)
  simp_all

private theorem not_wd_of_revisit {gs : GridSpec} {body : Body} {m0 : Mem}
    {p : ArrayDesc × BlockSpec} (hp : p ∈ gs.outSpecs)
    (hbad : noRevisit ((List.range (gridPoints gs.grid).length).map (gs.idxAt p.2 m0)) = false) :
    ¬ gs.WellDefined body m0 := fun h => by
  have := h.outputs_consecutive p hp
  simp_all

/-- `write_pages` on `kv : [2, 1]` with one page; the id buffer (0) holds `id`. -/
private noncomputable def kvMem (ids : ℕ → ℝ) : Mem := fun b o => if b = 0 then ids o else 0

private theorem kvOut_mem :
    (writePagesGridSpec 2 1 1 0 1 2).outSpecs.head (by simp [writePagesGridSpec]) ∈
      (writePagesGridSpec 2 1 1 0 1 2).outSpecs :=
  List.head_mem _

/-- S4: a `-1` padding id used to decode to page 0 through `Nat.floor`; it is now out of bounds. -/
theorem reg_s4_neg_one_id (junk : ℕ → ℝ) :
    writePagesRun 2 1 1 0 1 2 junk (kvMem fun _ => ((-1 : ℤ) : ℝ)) = none := by
  have hd : decodeInt32 (-1 : ℝ) = some (-1) := by
    simpa using decodeInt32_intCast (z := -1) (by decide)
  apply pallasCall_of_not_wellDefined
  apply not_wd_of_bad_out_block (k := 0) (by simp [writePagesGridSpec, gridPoints_length])
    kvOut_mem
  simp [blockOk, GridSpec.idxAt, GridSpec.scalarPrefetch, writePagesGridSpec, gridPoints, kvMem,
    readInt32, Mem.read, hd]

/-- S3: an id equal to `N` used to be a silent no-op; the block is now out of bounds. -/
theorem reg_s3_id_eq_n (junk : ℕ → ℝ) :
    writePagesRun 2 1 1 0 1 2 junk (kvMem fun _ => ((2 : ℕ) : ℝ)) = none := by
  have hd : decodeInt32 (2 : ℝ) = some 2 := by
    simpa using decodeInt32_intCast (z := 2) (by decide)
  apply pallasCall_of_not_wellDefined
  apply not_wd_of_bad_out_block (k := 0) (by simp [writePagesGridSpec, gridPoints_length])
    kvOut_mem
  simp [blockOk, GridSpec.idxAt, GridSpec.scalarPrefetch, writePagesGridSpec, gridPoints, kvMem,
    readInt32, Mem.read, hd]

/-- S2: ids `[0, 1, 0]` revisit output block 0 non-consecutively; the run is now undefined. -/
theorem reg_s2_revisit_block (junk : ℕ → ℝ) :
    writePagesRun 2 3 1 0 1 2 junk
      (kvMem fun o => ((if o = 1 then 1 else 0 : ℕ) : ℝ)) = none := by
  have hd0 : decodeInt32 (0 : ℝ) = some 0 := by
    simpa using decodeInt32_intCast (z := 0) (by decide)
  have hd1 : decodeInt32 (1 : ℝ) = some 1 := by
    simpa using decodeInt32_intCast (z := 1) (by decide)
  apply pallasCall_of_not_wellDefined
  apply not_wd_of_revisit (List.head_mem (by simp [writePagesGridSpec]))
  simp [noRevisit, GridSpec.idxAt, GridSpec.scalarPrefetch, writePagesGridSpec, gridPoints,
    List.range_succ, kvMem, readInt32, Mem.read, hd0, hd1]

/-- Checked `readInt32` decoding rejects a non-integer page id (`3 / 2`) rather than flooring it. -/
theorem writePages_non_int_id_none (junk : ℕ → ℝ) :
    writePagesRun 2 1 1 0 1 2 junk (kvMem fun _ => (3 / 2 : ℝ)) = none := by
  have hd : decodeInt32 (3 / 2 : ℝ) = none := by
    apply decodeInt32_of_not_int
    intro z hz
    have h1 : ((2 * z : ℤ) : ℝ) = ((3 : ℤ) : ℝ) := by push_cast; linarith
    have h2 : 2 * z = 3 := Int.cast_injective h1
    omega
  apply pallasCall_of_not_wellDefined
  apply not_wd_of_bad_out_block (k := 0) (by simp [writePagesGridSpec, gridPoints_length])
    kvOut_mem
  simp [blockOk, GridSpec.idxAt, GridSpec.scalarPrefetch, writePagesGridSpec, gridPoints, kvMem,
    readInt32, Mem.read, hd]

/-- Checked `readInt32` decoding rejects an int32-overflow page id (`2 ^ 31`) rather than clamping. -/
theorem writePages_overflow_id_none (junk : ℕ → ℝ) :
    writePagesRun 2 1 1 0 1 2 junk (kvMem fun _ => ((2 ^ 31 : ℤ) : ℝ)) = none := by
  have hd : decodeInt32 ((2 ^ 31 : ℤ) : ℝ) = none := decodeInt32_of_not_inRange (by decide)
  have hsc : (writePagesGridSpec 2 1 1 0 1 2).scalarPrefetch
      (kvMem fun _ => ((2 ^ 31 : ℤ) : ℝ)) 0 = -1 := by
    change (decodeInt32 ((2 ^ 31 : ℤ) : ℝ)).getD (-1) = -1
    rw [hd]
    rfl
  apply pallasCall_of_not_wellDefined
  apply not_wd_of_bad_out_block (k := 0) (by simp [writePagesGridSpec, gridPoints_length])
    kvOut_mem
  have hidx : (writePagesGridSpec 2 1 1 0 1 2).idxAt
      ((writePagesGridSpec 2 1 1 0 1 2).outSpecs.head (by simp [writePagesGridSpec])).2
      (kvMem fun _ => ((2 ^ 31 : ℤ) : ℝ)) 0 = [-1, 0] := by
    change [(writePagesGridSpec 2 1 1 0 1 2).scalarPrefetch
      (kvMem fun _ => ((2 ^ 31 : ℤ) : ℝ)) 0, 0] = [-1, 0]
    rw [hsc]
  rw [hidx]
  decide

/-- S1: an input with a constant block index aliased to an output that writes that block in step 0.
Mosaic does not re-fetch the input in step 1, so the old model's re-read was wrong; the run is now
undefined. -/
def staleArr : ArrayDesc := { buf := 0, shape := [2] }

def staleGridSpec : GridSpec :=
  { grid := [2]
    inSpecs := [(staleArr, { blockShape := [1], indexMap := fun _ _ => [0] })]
    outSpecs := [(staleArr, { blockShape := [1], indexMap := fun g _ => g.map (↑) })]
    aliases := [(0, 0)] }

theorem reg_s1_stale_read (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall staleGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  apply h.no_stale_read 0 1 (by decide) (by decide) _ (List.head_mem (by simp [staleGridSpec]))
    _ (List.head_mem (by simp [staleGridSpec]))
  refine ⟨rfl, 0, ?_, ?_⟩ <;>
    simp [GridSpec.idxAt, staleGridSpec, staleArr, gridPoints, addrInBlock, coordInBlock,
      unflatCoord, List.range_succ]

/-- PAL-ALIAS-REPEAT: an aliased `(in, out)` pair staying on the same block index across
consecutive steps used to bypass `no_stale_read`; it is now rejected. -/
def aliasRepeatGridSpec : GridSpec :=
  { grid := [2]
    inSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    outSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    aliases := [(0, 0)] }

theorem alias_repeat_stale_read_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall aliasRepeatGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  apply h.no_stale_read 0 1 (by decide) (by decide)
    _ (List.head_mem (by simp [aliasRepeatGridSpec]))
    _ (List.head_mem (by simp [aliasRepeatGridSpec]))
  refine ⟨rfl, 0, ?_, ?_⟩ <;>
    simp [GridSpec.idxAt, aliasRepeatGridSpec, gridPoints, addrInBlock, coordInBlock,
      unflatCoord, List.range_succ]

/-- S6: a body that returns one block for two outputs used to zero-fill the second; it is now
undefined. -/
def twoOutGridSpec : GridSpec :=
  { grid := [1]
    inSpecs := []
    outSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] }),
                 ({ buf := 1, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })] }

theorem reg_s6_short_body (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall twoOutGridSpec (fun _ _ _ _ => [fun _ => 1]) junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have := h.body_arity [] (fun _ => 0) [] []
  simp [twoOutGridSpec] at this

/-- S7: a coordinate outside the block and an edge-padding coordinate used to read the same
junk value; they now read independent junk. -/
theorem reg_s7_independent_junk (m : Mem) :
    ∃ junk : ℕ → ℝ, loadBlock { buf := 0, shape := [1, 2] } [2, 2] [0, 0] junk m [0, 2] ≠
      loadBlock { buf := 0, shape := [1, 2] } [2, 2] [0, 0] junk m [1, 0] := by
  refine ⟨fun n => n, ?_⟩
  have hne : Encodable.encode ([0, 2] : List ℕ) ≠ Encodable.encode ([1, 0] : List ℕ) :=
    Encodable.encode_injective.ne (by decide)
  simp [loadBlock, inBounds, globalCoord]

/-- S8: a `GridSpec` aliasing input buffer 0 to a distinct output buffer 1. -/
def diffBufAliasGridSpec : GridSpec :=
  { grid := [1]
    inSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    outSpecs := [({ buf := 1, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    aliases := [(0, 0)] }

/-- S8: aliasing two different buffers used to copy the input into a separate output buffer so a
proof could claim the input was unchanged; the run is now undefined. -/
theorem alias_diff_buf_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall diffBufAliasGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have ⟨_, inp, outp, hin, hout, hbuf, _⟩ :=
    h.aliases_ok.entries_ok (0, 0) (by simp [diffBufAliasGridSpec])
  simp [diffBufAliasGridSpec] at hin hout
  subst hin hout
  simp at hbuf

/-- S8: a `GridSpec` aliasing an input and output with the same buffer but different shapes. -/
def diffShapeAliasGridSpec : GridSpec :=
  { grid := [1]
    inSpecs := [({ buf := 0, shape := [2] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    outSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    aliases := [(0, 0)] }

/-- S8: an alias whose input and output differ in shape is rejected. -/
theorem alias_diff_shape_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall diffShapeAliasGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have ⟨_, inp, outp, hin, hout, _, hshape, _⟩ :=
    h.aliases_ok.entries_ok (0, 0) (by simp [diffShapeAliasGridSpec])
  simp [diffShapeAliasGridSpec] at hin hout
  subst hin hout
  simp at hshape

/-- S8: a `GridSpec` aliasing an input and output with the same buffer and shape, but different
`blockShape` (violates only `blockShape` matching in `entries_ok`). -/
def diffBlockShapeAliasGridSpec : GridSpec :=
  { grid := [1]
    inSpecs := [({ buf := 0, shape := [2] }, { blockShape := [2], indexMap := fun _ _ => [0] })]
    outSpecs := [({ buf := 0, shape := [2] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    aliases := [(0, 0)] }

/-- S8: an alias whose input and output differ in `blockShape` is rejected. -/
theorem alias_diff_blockShape_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall diffBlockShapeAliasGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have ⟨_, inp, outp, hin, hout, _, _, hbs, _⟩ :=
    h.aliases_ok.entries_ok (0, 0) (by simp [diffBlockShapeAliasGridSpec])
  simp [diffBlockShapeAliasGridSpec] at hin hout
  subst hin hout
  simp at hbs

/-- S8: a `GridSpec` aliasing an input and output with the same buffer, shape, and `blockShape`,
but different block indices at grid step `[0]` (violates only `indexMap` matching in `entries_ok`). -/
def diffIndexMapAliasGridSpec : GridSpec :=
  { grid := [1]
    inSpecs := [({ buf := 0, shape := [2] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    outSpecs := [({ buf := 0, shape := [2] }, { blockShape := [1], indexMap := fun _ _ => [1] })]
    aliases := [(0, 0)] }

/-- S8: an alias whose input and output differ in block index at a grid step is rejected. -/
theorem alias_diff_indexMap_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall diffIndexMapAliasGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have ⟨_, inp, outp, hin, hout, _, _, _, himap⟩ :=
    h.aliases_ok.entries_ok (0, 0) (by simp [diffIndexMapAliasGridSpec])
  simp [diffIndexMapAliasGridSpec] at hin hout
  subst hin hout
  have h0 := himap [0] (by simp [diffIndexMapAliasGridSpec, gridPoints]) (fun _ => 0)
  simp at h0

/-- S8: the old `[(1, 0)]` alias numbering for `write_pages` (which omitted the scalar-prefetch
`block_ids` argument) now refers to `pages` instead of `kv`, making the run undefined. -/
theorem writePages_off_by_one_alias_none (N M P : ℕ) (idsBuf pagesBuf kvBuf : BufId)
    (hbuf : kvBuf ≠ pagesBuf) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall { writePagesGridSpec N M P idsBuf pagesBuf kvBuf with aliases := [(1, 0)] }
      writePagesBody junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have ⟨_, inp, outp, hin, hout, heq, _⟩ := h.aliases_ok.entries_ok (1, 0) (by simp)
  simp [writePagesGridSpec, GridSpec.numScalarPrefetch] at hin hout
  subst hin hout
  exact hbuf heq.symm

/-- S8: dedicated fixture whose alias targets scalar-prefetch arg 0 while all buffers are
disjoint (violates only `gs.numScalarPrefetch ≤ p.1` in `entries_ok`). -/
def scalarAliasGridSpec : GridSpec :=
  { grid := [1]
    scalarPrefetchBufs := [0]
    inSpecs := [({ buf := 1, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    outSpecs := [({ buf := 2, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    aliases := [(0, 0)] }

/-- S8: aliasing a scalar-prefetch argument is rejected (isolated to `entries_ok`). -/
theorem alias_scalar_prefetch_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall scalarAliasGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have ⟨hge, _⟩ := h.aliases_ok.entries_ok (0, 0) (by simp [scalarAliasGridSpec])
  simp [scalarAliasGridSpec, GridSpec.numScalarPrefetch] at hge

/-- S8: dedicated fixture with an out-of-range alias input index and disjoint buffers (violates
only `inSpecs[p.1 - numScalarPrefetch]? = some inp` in `entries_ok`). -/
def inOobAliasGridSpec : GridSpec :=
  { grid := [1]
    inSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    outSpecs := [({ buf := 1, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    aliases := [(1, 0)] }

/-- S8: an out-of-range alias input index is rejected (isolated to `entries_ok`). -/
theorem alias_in_oob_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall inOobAliasGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have ⟨_, _, _, hin, _⟩ := h.aliases_ok.entries_ok (1, 0) (by simp [inOobAliasGridSpec])
  simp [inOobAliasGridSpec, GridSpec.numScalarPrefetch] at hin

/-- S8: dedicated fixture with an out-of-range alias output index and disjoint buffers (violates
only `outSpecs[p.2]? = some outp` in `entries_ok`). -/
def outOobAliasGridSpec : GridSpec :=
  { grid := [1]
    inSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    outSpecs := [({ buf := 1, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    aliases := [(0, 1)] }

/-- S8: an out-of-range alias output index is rejected (isolated to `entries_ok`). -/
theorem alias_out_oob_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall outOobAliasGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have ⟨_, _, _, _, hout, _⟩ := h.aliases_ok.entries_ok (0, 1) (by simp [outOobAliasGridSpec])
  simp [outOobAliasGridSpec] at hout

/-- S8: dedicated fixture aliasing one input to two outputs on the same buffer (`entries_ok`,
`outs_nodup`, and `buf_eq_aliased` all hold; rejected via derived `AliasesValid.ins_nodup`). -/
def dupInAliasGridSpec : GridSpec :=
  { grid := [1]
    inSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    outSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] }),
                 ({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    aliases := [(0, 0), (0, 1)] }

/-- S8: aliasing the same input to two outputs is rejected. -/
theorem alias_dup_in_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall dupInAliasGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have hnd := h.aliases_ok.ins_nodup
  simp [dupInAliasGridSpec] at hnd

/-- S8: dedicated fixture aliasing two inputs on buffer `0` to the same output on buffer `0`
(satisfies every other `WellDefined` and `AliasesValid` field; violates only `outs_nodup`). -/
def dupOutAliasGridSpec : GridSpec :=
  { grid := [1]
    inSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] }),
                ({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    outSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    aliases := [(0, 0), (1, 0)] }

/-- S8: aliasing two inputs to the same output is rejected (isolated to `outs_nodup`). -/
theorem alias_dup_out_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall dupOutAliasGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have hnd := h.aliases_ok.outs_nodup
  simp [dupOutAliasGridSpec] at hnd

/-- S8: a `GridSpec` with an unaliased output whose buffer equals an input's buffer. -/
def unaliasedOutEqInGridSpec : GridSpec :=
  { grid := [1]
    inSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    outSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    aliases := [] }

/-- S8: an unaliased output sharing an input's buffer used to silently junk-initialize that input;
it is now rejected. -/
theorem unaliased_out_eq_in_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall unaliasedOutEqInGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have hmem := h.aliases_ok.buf_eq_aliased 0 _ rfl 0 _ rfl rfl
  simp [unaliasedOutEqInGridSpec] at hmem

/-- S8: a `GridSpec` with two unaliased outputs sharing the same buffer. -/
def dupOutBufGridSpec : GridSpec :=
  { grid := [1]
    inSpecs := []
    outSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] }),
                 ({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    aliases := [] }

/-- S8: an output sharing a buffer with another output is rejected. -/
theorem unaliased_out_eq_out_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall dupOutBufGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have hnd := h.aliases_ok.out_bufs_nodup
  simp [dupOutBufGridSpec] at hnd

/-- S8: a `GridSpec` whose output targets a scalar-prefetch buffer (violates only
`out_disjoint_scalar`). -/
def outEqScalarGridSpec : GridSpec :=
  { grid := [1]
    scalarPrefetchBufs := [0]
    inSpecs := []
    outSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    aliases := [] }

/-- S8: an output targeting a scalar-prefetch buffer is rejected. -/
theorem out_aliases_scalar_prefetch_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall outEqScalarGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  exact h.aliases_ok.out_disjoint_scalar 0 (by simp [outEqScalarGridSpec])
    _ (List.head_mem (by simp [outEqScalarGridSpec])) rfl

/-- S8: `write_pages` with `kvBuf = idsBuf` is rejected because the output targets the
scalar-prefetch buffer. -/
theorem writePages_kv_eq_ids_none (N M P : ℕ) (idsBuf pagesBuf : BufId)
    (junk : ℕ → ℝ) (m0 : Mem) :
    writePagesRun N M P idsBuf pagesBuf idsBuf junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  exact h.aliases_ok.out_disjoint_scalar idsBuf (by simp [writePagesGridSpec])
    _ (List.head_mem (by simp [writePagesGridSpec])) rfl

/-- PAL-GRID-EMPTY: an empty grid (`0 ∈ grid`) with otherwise valid block specs is rejected by
`grid_pos`. -/
def emptyGridSpec : GridSpec :=
  { grid := [0]
    inSpecs := [({ buf := 0, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
    outSpecs := [({ buf := 1, shape := [1] }, { blockShape := [1], indexMap := fun _ _ => [0] })] }

theorem empty_grid_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall emptyGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have hpos := h.grid_pos 0 (by simp [emptyGridSpec])
  omega

/-- PAL-GRID-EMPTY: a zero-sized block dimension is rejected statically by `block_specs_ok`, even
when the grid is empty. -/
theorem empty_grid_zero_block_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall
      { grid := [0]
        inSpecs := [({ buf := 0, shape := [1] }, { blockShape := [0], indexMap := fun _ _ => [0] })]
        outSpecs := [] }
      body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have hok := (h.block_specs_ok ({ buf := 0, shape := [1] },
    { blockShape := [0], indexMap := fun _ _ => [0] }) (by simp)).2 0 (by simp)
  omega

/-- PAL-GRID-EMPTY: a block/array rank mismatch is rejected statically by `block_specs_ok`, even
when the grid is empty. -/
theorem empty_grid_rank_mismatch_none (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall
      { grid := [0]
        inSpecs := [({ buf := 0, shape := [1, 1] }, { blockShape := [1], indexMap := fun _ _ => [0] })]
        outSpecs := [] }
      body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have hlen := (h.block_specs_ok ({ buf := 0, shape := [1, 1] },
    { blockShape := [1], indexMap := fun _ _ => [0] }) (by simp)).1
  simp at hlen

end TpuKernelLean.Tests
