import TpuKernelLean.Memory
import TpuKernelLean.Dtype

namespace TpuKernelLean.Pallas

/-- Descriptor of a logical HBM array passed to `pallas_call`. -/
structure ArrayDesc where
  /-- Backing HBM buffer identifier. -/
  buf : BufId
  /-- Static array shape; `None` block dims correspond to size `1` in `BlockSpec`. -/
  shape : List ℕ

/-- Pallas `BlockSpec`: block shape and index map from `(gridIdx, scalars)` to block coordinates.
Block coordinates are integers, as in Pallas, so a negative or out-of-range index is representable
and is rejected by `GridSpec.WellDefined` instead of being clamped. -/
structure BlockSpec where
  /-- Local VMEM block shape (with Pallas `None` dims represented as `1`). -/
  blockShape : List ℕ
  /-- Map from grid coordinates and the int32 scalar-prefetch table to block indices. -/
  indexMap : List ℕ → (ℕ → ℤ) → List ℤ

/-- Execution semantics of a single grid dimension (`pltpu.CompilerParams.dimension_semantics`).
[PAL-PAR-SEMANTICS] -/
inductive DimSemantics where
  | parallel
  | arbitrary
  deriving DecidableEq, Repr

/-- Return `true` iff a grid dimension is marked `parallel`. [PAL-PAR-SEMANTICS] -/
def DimSemantics.isParallel : DimSemantics → Bool
  | .parallel => true
  | .arbitrary => false

/-- Every `parallel` dimension precedes every `arbitrary` dimension. JAX's TPU interpreter assumes
this and says an error is raised otherwise (`mosaic/interpret/interpret_pallas_call.py:2273` at
35066e1); the compiled lowering (`mosaic/lowering.py`) checks only the length and the names.
[PAL-PAR-SEMANTICS] -/
def parallelPrefix : List DimSemantics → Bool
  | [] => true
  | .parallel :: rest => parallelPrefix rest
  | .arbitrary :: rest => rest.all fun d => !d.isParallel

/-- Grid, scalar prefetch extractor, and input/output block specs for a `pallas_call`. -/
structure GridSpec where
  /-- Iteration grid dimensions, traversed in row-major order (last dim fastest). -/
  grid : List ℕ
  /-- Backing HBM buffers of the leading scalar-prefetch arguments
  (`PrefetchScalarGridSpec.num_scalar_prefetch = scalarPrefetchBufs.length`).
  Input `inSpecs[i]` has Python argument index `scalarPrefetchBufs.length + i`.
  [PAL-SCALAR-PREFETCH] -/
  scalarPrefetchBufs : List BufId := []
  /-- Extract int32 scalar-prefetch values from the checked `readInt32` readers of
  `scalarPrefetchBufs` alone (never arbitrary `Mem`), evaluated once before the grid runs.
  [PAL-SCALAR-PREFETCH] -/
  scalarPrefetchFn : List (ℕ → Option ℤ) → ℕ → ℤ := fun _ _ => 0
  /-- Input array descriptors paired with their block specs. -/
  inSpecs : List (ArrayDesc × BlockSpec)
  /-- Output array descriptors paired with their block specs. -/
  outSpecs : List (ArrayDesc × BlockSpec)
  /-- Shapes of the VMEM scratch buffers (`scratch_shapes`). Each starts as junk once per launch,
  persists across every grid step, and is never written to HBM. [PAL-SCRATCH-LIFE] -/
  scratch : List (List ℕ) := []
  /-- Input-output alias pairs `(inIdx, outIdx)` (`input_output_aliases`), where `inIdx` uses
  Python's argument numbering (including the `scalarPrefetchBufs.length` leading scalar-prefetch
  args) and `outIdx` indexes `outSpecs`. -/
  aliases : List (ℕ × ℕ) := []
  /-- Per-dimension execution semantics (`dimension_semantics`); defaults to all `arbitrary`.
  [PAL-PAR-SEMANTICS] -/
  dimSemantics : List DimSemantics := List.replicate grid.length .arbitrary

/-- Number of leading scalar-prefetch arguments (`PrefetchScalarGridSpec.num_scalar_prefetch`),
derived from `scalarPrefetchBufs.length` so it cannot drift from the buffers read.
[PAL-SCALAR-NUM] -/
def GridSpec.numScalarPrefetch (gs : GridSpec) : ℕ :=
  gs.scalarPrefetchBufs.length

/-- Extract the scalar-prefetch table from initial HBM memory `m0`, reading only buffers in
`gs.scalarPrefetchBufs` through the checked `readInt32` decoder. [PAL-SCALAR-PREFETCH] -/
noncomputable def GridSpec.scalarPrefetch (gs : GridSpec) (m0 : Mem) : ℕ → ℤ :=
  gs.scalarPrefetchFn (gs.scalarPrefetchBufs.map fun b => readInt32 m0 b)

/-- Number of leading `parallel` dimensions in `gs.dimSemantics`. [PAL-PAR-SEMANTICS] -/
def GridSpec.numParallel (gs : GridSpec) : ℕ :=
  (gs.dimSemantics.takeWhile DimSemantics.isParallel).length

/-- Leading grid dimensions marked `parallel`. [PAL-PAR-SEMANTICS] -/
def GridSpec.parGrid (gs : GridSpec) : List ℕ :=
  gs.grid.take gs.numParallel

/-- Trailing grid dimensions marked `arbitrary`. [PAL-PAR-SEMANTICS] -/
def GridSpec.arbGrid (gs : GridSpec) : List ℕ :=
  gs.grid.drop gs.numParallel

/-- Shallow-embedded Pallas kernel body receiving `(gridIdx, scalars, ins, prev)` and returning new
blocks. `prev` and the result list the output blocks, then the scratch buffers, in order. Inputs
cannot be mutated by `Body`. [PAL-IN-READONLY] -/
abbrev Body :=
  List ℕ → (ℕ → ℤ) → List (List ℕ → ℝ) → List (List ℕ → ℝ) → List (List ℕ → ℝ)

/-- Per-output VMEM cache state holding the currently buffered block index and block contents,
if any. [PAL-OUT-PERSIST] -/
abbrev OutSlot := Option (List ℕ × (List ℕ → ℝ))

/-- Block index of `spec` at grid point `g` as naturals; exact whenever `GridSpec.WellDefined`
holds, since that requires every coordinate to be non-negative. -/
def BlockSpec.blockIdx (spec : BlockSpec) (g : List ℕ) (scalars : ℕ → ℤ) : List ℕ :=
  (spec.indexMap g scalars).map Int.toNat

/-- Row-major linear offset of coordinate list `coord` within `shape`. -/
def flatCoord : List ℕ → List ℕ → ℕ
  | [], _ => 0
  | _ :: _, [] => 0
  | _ :: ds, c :: cs => c * ds.prod + flatCoord ds cs

/-- Inverse of row-major flattening: decode linear offset `a` into coordinates for `shape`. -/
def unflatCoord : List ℕ → ℕ → List ℕ
  | [], _ => []
  | _ :: ds, a =>
    let stride := ds.prod
    if stride = 0 then 0 :: unflatCoord ds 0
    else (a / stride) :: unflatCoord ds (a % stride)

/-- Check whether `coord` has the same rank as `shape` and every coordinate is in bounds. -/
def inBounds : List ℕ → List ℕ → Bool
  | [], [] => true
  | d :: ds, c :: cs => decide (c < d) && inBounds ds cs
  | _, _ => false

/-- Check whether global coordinate `coord` lies within the block `(blockShape, blockIdx)`. -/
def coordInBlock : List ℕ → List ℕ → List ℕ → Bool
  | [], [], [] => true
  | bs :: bss, bi :: bis, c :: cs =>
    decide (bi * bs ≤ c ∧ c < (bi + 1) * bs) && coordInBlock bss bis cs
  | _, _, _ => false

/-- Convert global coordinate `coord` to local block coordinate by subtracting `blockIdx * blockShape`. -/
def localCoord : List ℕ → List ℕ → List ℕ → List ℕ
  | bs :: bss, bi :: bis, c :: cs => (c - bi * bs) :: localCoord bss bis cs
  | _, _, _ => []

/-- Convert local block coordinate `loc` to global coordinate `blockIdx * blockShape + loc`. -/
def globalCoord : List ℕ → List ℕ → List ℕ → List ℕ
  | bs :: bss, bi :: bis, l :: ls => (bi * bs + l) :: globalCoord bss bis ls
  | _, _, _ => []

/-- Predicate checking whether linear offset `a` is an in-bounds element of `arr` covered by
`(blockShape, blockIdx)`. [PAL-EDGE-WRITE] -/
def addrInBlock (arr : ArrayDesc) (blockShape blockIdx : List ℕ) (a : ℕ) : Bool :=
  decide (a < arr.shape.prod) && coordInBlock blockShape blockIdx (unflatCoord arr.shape a)

/-- Enumerate all grid points in row-major (lexicographic, last dim fastest) order.
[PAL-GRID-ORDER] -/
def gridPoints : List ℕ → List (List ℕ)
  | [] => [[]]
  | d :: ds => (List.range d).flatMap fun i => (gridPoints ds).map fun rest => i :: rest

/-- Grid points of the arbitrary-dimension group for parallel coordinate `pg`, in row-major order.
[PAL-PAR-EXEC] -/
def GridSpec.groupPoints (gs : GridSpec) (pg : List ℕ) : List (List ℕ) :=
  (gridPoints gs.arbGrid).map (pg ++ ·)

/-- `p` appears in `gridPoints grid` if and only if `p` is in bounds for `grid`. -/
theorem mem_gridPoints : ∀ (grid p : List ℕ),
    p ∈ gridPoints grid ↔ inBounds grid p = true
  | [], [] => by simp [gridPoints, inBounds]
  | [], _ :: _ => by simp [gridPoints, inBounds]
  | _ :: _, [] => by simp [gridPoints, inBounds]
  | d :: ds, i :: ps => by
    simp only [gridPoints, List.mem_flatMap, List.mem_range, List.mem_map, inBounds,
      Bool.and_eq_true, decide_eq_true_eq]
    constructor
    · rintro ⟨i', hi', rest, hrest, hcons⟩
      injection hcons with hi hps
      subst hi
      subst hps
      exact ⟨hi', (mem_gridPoints ds rest).mp hrest⟩
    · rintro ⟨hi, hps⟩
      exact ⟨i, hi, ps, (mem_gridPoints ds ps).mpr hps, rfl⟩

private theorem list_sum_map_const {α : Type*} (l : List α) (c : ℕ) :
    (l.map fun _ => c).sum = l.length * c := by
  induction l with
  | nil => simp
  | cons _ _ _ => simp [Nat.succ_mul, Nat.add_comm]

/-- `gridPoints grid` enumerates exactly `grid.prod` grid points. -/
theorem gridPoints_length : ∀ (grid : List ℕ), (gridPoints grid).length = grid.prod
  | [] => by simp [gridPoints]
  | d :: ds => by
    simp only [gridPoints, List.length_flatMap, List.length_map, gridPoints_length ds,
      list_sum_map_const, List.length_range, List.prod_cons]

/-- A grid whose dimensions are all strictly positive contains the all-zero grid point. -/
theorem replicate_zero_mem_gridPoints : ∀ {grid : List ℕ},
    (∀ d ∈ grid, 0 < d) → List.replicate grid.length 0 ∈ gridPoints grid
  | [], _ => by simp [gridPoints]
  | d :: ds, h => by
    have hd : 0 < d := h d (by simp)
    have hds : ∀ x ∈ ds, 0 < x := fun x hx => h x (by simp [hx])
    rw [mem_gridPoints]
    simp only [List.length_cons, List.replicate_succ, inBounds, Bool.and_eq_true,
      decide_eq_true_eq]
    exact ⟨hd, (mem_gridPoints ds _).mp (replicate_zero_mem_gridPoints hds)⟩

/-- Unflattening a 2-D row-major offset `j * P + p` with `p < P` recovers `[j, p]`. -/
theorem unflatCoord_two (N P j p : ℕ) (hp : p < P) :
    unflatCoord [N, P] (j * P + p) = [j, p] := by
  have hP : 0 < P := Nat.lt_of_le_of_lt (Nat.zero_le _) hp
  have hPne : P ≠ 0 := Nat.ne_of_gt hP
  have hdiv : (j * P + p) / P = j := by
    rw [Nat.mul_comm, Nat.mul_add_div hP, Nat.div_eq_of_lt hp, Nat.add_zero]
  have hmod : (j * P + p) % P = p := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hp]
  simp [unflatCoord, List.prod_cons, List.prod_nil, hPne, hdiv, hmod]

/-- Pair each output's new block index with the corresponding block returned by the body.
Missing blocks never occur in a well-defined run (`GridSpec.WellDefined.body_arity`).
[PAL-BODY-ARITY] -/
def makeSlots : List (List ℕ) → List (List ℕ → ℝ) → List OutSlot
  | [], _ => []
  | idx :: idxs, [] => some (idx, fun _ => 0) :: makeSlots idxs []
  | idx :: idxs, val :: vals => some (idx, val) :: makeSlots idxs vals

/-! ### Undefined behaviour

`pallas_call` has preconditions that Mosaic does not check at run time. A run that violates one
is undefined behaviour, and `pallasCall` returns `none` for it, so no correctness theorem can be
proved about it. -/

/-- A block index is legal for its array: ranks agree, block dims are positive, every coordinate is
non-negative and the block has at least one in-bounds element ("at least one of the elements in
each block must be within bounds", docs.jax.dev/en/latest/pallas/grid_blockspec.html).
[PAL-EDGE-LEGAL] -/
def blockOk : List ℕ → List ℕ → List ℤ → Bool
  | [], [], [] => true
  | d :: ds, b :: bs, i :: is =>
    decide (0 < b ∧ 0 ≤ i ∧ i * (b : ℤ) < (d : ℤ)) && blockOk ds bs is
  | _, _, _ => false

/-- Every legal (`blockOk`) block has matching array/block rank and strictly positive block
dimensions. -/
theorem blockOk_spec : ∀ {ds bs : List ℕ} {is : List ℤ},
    blockOk ds bs is = true → ds.length = bs.length ∧ ∀ b ∈ bs, 0 < b
  | [], [], [], _ => ⟨rfl, fun _ h => by simp at h⟩
  | _ :: ds, b :: bs, _ :: is, h => by
    simp only [blockOk, Bool.and_eq_true, decide_eq_true_eq] at h
    have ih := blockOk_spec h.2
    refine ⟨by simp [ih.1], ?_⟩
    intro x hx
    simp only [List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact h.1.1
    · exact ih.2 x hx
  | [], _ :: _, _, h | [], [], _ :: _, h | _ :: _, [], _, h | _ :: _, _ :: _, [], h => by
    simp [blockOk] at h

/-- No value reappears after a different one, i.e. every visit to a value is consecutive.
[PAL-OUT-CONSEC] -/
def noRevisit {α : Type*} [DecidableEq α] : List α → Bool
  | [] => true
  | x :: xs => decide (x ∉ xs.dropWhile fun y => decide (y = x)) && noRevisit xs

/-- A duplicate-free list trivially has no revisits. -/
theorem noRevisit_of_nodup {α : Type*} [DecidableEq α] :
    ∀ {l : List α}, l.Nodup → noRevisit l = true
  | [], _ => rfl
  | x :: xs, h => by
    rw [List.nodup_cons] at h
    simp only [noRevisit, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨fun hx => h.1 ((List.dropWhile_sublist _).subset hx), noRevisit_of_nodup h.2⟩

/-- Two blocks of the same HBM buffer share an in-bounds element. [PAL-NO-STALE-READ] -/
def blocksOverlap (a₁ : ArrayDesc) (bs₁ i₁ : List ℕ) (a₂ : ArrayDesc) (bs₂ i₂ : List ℕ) : Prop :=
  a₁.buf = a₂.buf ∧ ∃ a, addrInBlock a₁ bs₁ i₁ a = true ∧ addrInBlock a₂ bs₂ i₂ a = true

/-- Integer block index of `spec` at the `k`-th grid point (row-major order). -/
noncomputable def GridSpec.idxAt (gs : GridSpec) (spec : BlockSpec) (m0 : Mem) (k : ℕ) : List ℤ :=
  spec.indexMap ((gridPoints gs.grid).getD k []) (gs.scalarPrefetch m0)

/-- Alias table and output-buffer disjointness rules for a `pallas_call`. -/
structure GridSpec.AliasesValid (gs : GridSpec) : Prop where
  /-- Every alias entry names a non-scalar-prefetch input and an output in range, with the same
  HBM buffer, array shape, `blockShape`, and block index at every grid step.
  [PAL-ALIAS-ENTRY] [PAL-ALIAS-NO-SCALAR] [PAL-SCALAR-NUM] -/
  entries_ok : ∀ p ∈ gs.aliases,
    gs.numScalarPrefetch ≤ p.1 ∧
    ∃ inp outp,
      gs.inSpecs[p.1 - gs.numScalarPrefetch]? = some inp ∧
      gs.outSpecs[p.2]? = some outp ∧
      inp.1.buf = outp.1.buf ∧ inp.1.shape = outp.1.shape ∧
      inp.2.blockShape = outp.2.blockShape ∧
      ∀ g ∈ gridPoints gs.grid, ∀ sc, inp.2.indexMap g sc = outp.2.indexMap g sc
  /-- No output is aliased more than once. [PAL-ALIAS-INJ] -/
  outs_nodup : (gs.aliases.map Prod.snd).Nodup
  /-- No two outputs share an HBM buffer. [PAL-ALIAS-DISJOINT] -/
  out_bufs_nodup : (gs.outSpecs.map fun p => p.1.buf).Nodup
  /-- Any input and output sharing an HBM buffer must be paired in `aliases` (in particular, no
  unaliased output shares a buffer with any input). [PAL-ALIAS-DISJOINT] [PAL-SCALAR-NUM] -/
  buf_eq_aliased : ∀ i inp, gs.inSpecs[i]? = some inp →
    ∀ q outp, gs.outSpecs[q]? = some outp →
    inp.1.buf = outp.1.buf → (gs.numScalarPrefetch + i, q) ∈ gs.aliases
  /-- No output buffer may target or alias a scalar-prefetch buffer.
  [PAL-ALIAS-NO-SCALAR] [PAL-ALIAS-DISJOINT] -/
  out_disjoint_scalar : ∀ b ∈ gs.scalarPrefetchBufs, ∀ outp ∈ gs.outSpecs, outp.1.buf ≠ b

/-- No input argument is aliased more than once (derived from `entries_ok`, `outs_nodup`, and
`out_bufs_nodup`). [PAL-ALIAS-INJ] -/
theorem GridSpec.AliasesValid.ins_nodup {gs : GridSpec} (h : gs.AliasesValid) :
    (gs.aliases.map Prod.fst).Nodup := by
  rw [List.nodup_iff_injective_getElem]
  intro i j hij
  simp only [List.getElem_map] at hij
  have hi : i.val < gs.aliases.length := by simpa using i.isLt
  have hj : j.val < gs.aliases.length := by simpa using j.isLt
  obtain ⟨_, inp₁, outp₁, hin₁, hout₁, hbuf₁, _⟩ :=
    h.entries_ok gs.aliases[i.val] (List.getElem_mem hi)
  obtain ⟨_, inp₂, outp₂, hin₂, hout₂, hbuf₂, _⟩ :=
    h.entries_ok gs.aliases[j.val] (List.getElem_mem hj)
  rw [hij] at hin₁
  rw [hin₁, Option.some.injEq] at hin₂
  subst hin₂
  have hbuf_eq : outp₁.1.buf = outp₂.1.buf := hbuf₁.symm.trans hbuf₂
  have hlt₁ : gs.aliases[i.val].2 < gs.outSpecs.length := by
    by_contra hlt; rw [List.getElem?_eq_none (by omega)] at hout₁; cases hout₁
  have hlt₂ : gs.aliases[j.val].2 < gs.outSpecs.length := by
    by_contra hlt; rw [List.getElem?_eq_none (by omega)] at hout₂; cases hout₂
  have h1 : (gs.outSpecs.map fun p => p.1.buf)[gs.aliases[i.val].2]'(by simpa using hlt₁) =
      outp₁.1.buf := by
    simp [List.getElem?_eq_getElem hlt₁] at hout₁; simp [hout₁]
  have h2 : (gs.outSpecs.map fun p => p.1.buf)[gs.aliases[j.val].2]'(by simpa using hlt₂) =
      outp₂.1.buf := by
    simp [List.getElem?_eq_getElem hlt₂] at hout₂; simp [hout₂]
  have hout_eq : gs.aliases[i.val].2 = gs.aliases[j.val].2 := by
    have hnd := List.nodup_iff_injective_getElem.mp h.out_bufs_nodup
    have := @hnd ⟨gs.aliases[i.val].2, by simpa using hlt₁⟩
      ⟨gs.aliases[j.val].2, by simpa using hlt₂⟩ (by dsimp only; rw [h1, h2, hbuf_eq])
    exact congrArg Fin.val this
  have hnd_outs := List.nodup_iff_injective_getElem.mp h.outs_nodup
  have hij' := @hnd_outs ⟨i.val, by simpa using hi⟩ ⟨j.val, by simpa using hj⟩
    (by simp [hout_eq])
  exact Fin.ext (by simpa [Fin.mk.injEq] using hij')

/-- For a kernel with no aliases, `AliasesValid` reduces to distinct output buffers disjoint from
all input and scalar-prefetch buffers. -/
theorem GridSpec.AliasesValid.of_empty {gs : GridSpec}
    (hempty : gs.aliases = [])
    (hout : (gs.outSpecs.map fun p => p.1.buf).Nodup)
    (hdisj : ∀ inp ∈ gs.inSpecs, ∀ outp ∈ gs.outSpecs, inp.1.buf ≠ outp.1.buf)
    (hscalar : ∀ b ∈ gs.scalarPrefetchBufs, ∀ outp ∈ gs.outSpecs, outp.1.buf ≠ b) :
    gs.AliasesValid where
  entries_ok := by simp [hempty]
  outs_nodup := by simp [hempty]
  out_bufs_nodup := hout
  buf_eq_aliased := fun _ inp hi _ outp ho heq =>
    absurd heq (hdisj inp (List.mem_of_getElem? hi) outp (List.mem_of_getElem? ho))
  out_disjoint_scalar := hscalar

/-- Preconditions of a defined `pallas_call` run (docs.jax.dev/en/latest/pallas/tpu/details.html). -/
structure GridSpec.WellDefined (gs : GridSpec) (body : Body) (m0 : Mem) : Prop where
  /-- Every grid dimension is strictly positive (no empty grid). [PAL-GRID-EMPTY] -/
  grid_pos : ∀ d ∈ gs.grid, 0 < d
  /-- Static legality of every input and output `BlockSpec`: ranks agree with the array shape
  and every block dimension is strictly positive, checked independently of grid execution.
  [PAL-GRID-EMPTY] -/
  block_specs_ok : ∀ p ∈ gs.inSpecs ++ gs.outSpecs,
    p.1.shape.length = p.2.blockShape.length ∧ ∀ b ∈ p.2.blockShape, 0 < b
  /-- Every block index of every input and output is legal (`blockOk`). [PAL-EDGE-LEGAL] -/
  blocks_ok : ∀ k < (gridPoints gs.grid).length, ∀ p ∈ gs.inSpecs ++ gs.outSpecs,
    blockOk p.1.shape p.2.blockShape (gs.idxAt p.2 m0 k) = true
  /-- "All invocations that write to a particular slice are consecutive": an output block is
  never revisited after the pipeline has moved to another block and written it back.
  [PAL-OUT-CONSEC] -/
  outputs_consecutive : ∀ p ∈ gs.outSpecs,
    noRevisit ((List.range (gridPoints gs.grid).length).map (gs.idxAt p.2 m0)) = true
  /-- The body returns exactly one block per output and one value per scratch buffer.
  [PAL-BODY-ARITY] -/
  body_arity : ∀ g s ins outs,
    (body g s ins outs).length = gs.outSpecs.length + gs.scratch.length
  /-- No input reads an HBM region that an earlier grid step has written to an output block,
  whether or not the output's block index changed in between. Mosaic skips the HBM fetch when an
  input's block index repeats and prefetches the next input before the current output's write-back
  lands, and `interpret=True` syncs aliased buffers every step; forbidding all earlier-output
  overlaps ensures every input read sees the initial HBM contents `m0` so re-fetching (as
  `pallasExec` does), Mosaic's caching, and HLO interpretation agree.
  shortcut: forbids all such reads, including ones a fully modelled pipeline would order safely.
  [PAL-NO-STALE-READ] [PAL-ALIAS-REPEAT] -/
  no_stale_read : ∀ s t, s < t → t < (gridPoints gs.grid).length →
    ∀ i ∈ gs.inSpecs, ∀ o ∈ gs.outSpecs,
      ¬ blocksOverlap i.1 i.2.blockShape ((gs.idxAt i.2 m0 t).map Int.toNat)
          o.1 o.2.blockShape ((gs.idxAt o.2 m0 s).map Int.toNat)
  /-- Alias table and output-buffer disjointness rules hold (`GridSpec.AliasesValid`). -/
  aliases_ok : gs.AliasesValid
  /-- `dimSemantics` has one entry per grid dimension. [PAL-PAR-SEMANTICS] -/
  dim_semantics_len : gs.dimSemantics.length = gs.grid.length
  /-- Every `parallel` dimension precedes every `arbitrary` dimension (a model restriction).
  [PAL-PAR-SEMANTICS] -/
  dim_semantics_prefix : parallelPrefix gs.dimSemantics = true
  /-- Distinct parallel coordinates write disjoint HBM output regions (no write-write race or
  cross-core output VMEM carry). [PAL-PAR-DISJOINT] -/
  parallel_outputs_disjoint :
    ∀ pg₁ ∈ gridPoints gs.parGrid, ∀ pg₂ ∈ gridPoints gs.parGrid, pg₁ ≠ pg₂ →
      ∀ ag₁ ∈ gridPoints gs.arbGrid, ∀ ag₂ ∈ gridPoints gs.arbGrid,
        ∀ o₁ ∈ gs.outSpecs, ∀ o₂ ∈ gs.outSpecs,
          ¬ blocksOverlap o₁.1 o₁.2.blockShape (o₁.2.blockIdx (pg₁ ++ ag₁) (gs.scalarPrefetch m0))
              o₂.1 o₂.2.blockShape (o₂.2.blockIdx (pg₂ ++ ag₂) (gs.scalarPrefetch m0))
  /-- No parallel coordinate reads an HBM region written by a different parallel coordinate
  (no read-write or write-read race across cores or reorderings). [PAL-PAR-DISJOINT] -/
  parallel_read_write_disjoint :
    ∀ pg₁ ∈ gridPoints gs.parGrid, ∀ pg₂ ∈ gridPoints gs.parGrid, pg₁ ≠ pg₂ →
      ∀ ag₁ ∈ gridPoints gs.arbGrid, ∀ ag₂ ∈ gridPoints gs.arbGrid,
        ∀ i ∈ gs.inSpecs, ∀ o ∈ gs.outSpecs,
          ¬ blocksOverlap i.1 i.2.blockShape (i.2.blockIdx (pg₂ ++ ag₂) (gs.scalarPrefetch m0))
              o.1 o.2.blockShape (o.2.blockIdx (pg₁ ++ ag₁) (gs.scalarPrefetch m0))
  /-- When any dimension is `parallel`, the first step of every parallel iteration group must not
  depend on prior scratch VMEM contents (each core starts from junk scratch and groups may run in
  any order). shortcut: requires all scratch buffers to be independent at the first arbitrary step,
  rather than allowing lazy initialization of a scratch buffer on a later step.
  [PAL-PAR-SCRATCH] -/
  parallel_scratch_reset : gs.parGrid ≠ [] →
    ∀ pg ∈ gridPoints gs.parGrid, ∀ ag₀ ∈ (gridPoints gs.arbGrid).head?,
      ∀ ins outs sc₁ sc₂,
        outs.length = gs.outSpecs.length →
        sc₁.length = gs.scratch.length →
        sc₂.length = gs.scratch.length →
        body (pg ++ ag₀) (gs.scalarPrefetch m0) ins (outs ++ sc₁) =
        body (pg ++ ag₀) (gs.scalarPrefetch m0) ins (outs ++ sc₂)

noncomputable section

/-- Junk oracle for one read site: `tag` names the site (grid step and role), so garbage read at
different sites is independent instead of one shared value. [PAL-JUNK-INDEP] -/
def junkAt (junk : ℕ → ℝ) (tag : ℕ) : ℕ → ℝ :=
  fun l => junk (Nat.pair tag l)

/-- Load a block from `arr` at `blockIdx`. Coordinates outside the block (VMEM holds only the
block) or outside the array (edge-block padding) read values from `junk`, keyed injectively by the
local coordinate. [PAL-IN-LOAD] [PAL-EDGE-READ] [PAL-JUNK-INDEP] -/
def loadBlock (arr : ArrayDesc) (blockShape blockIdx : List ℕ)
    (junk : ℕ → ℝ) (m : Mem) : List ℕ → ℝ :=
  fun loc =>
    let gc := globalCoord blockShape blockIdx loc
    if inBounds blockShape loc && inBounds arr.shape gc then
      m.read arr.buf (flatCoord arr.shape gc)
    else
      junk (Encodable.encode loc)

/-- `loadBlock` depends only on the contents of `arr.buf`. -/
theorem loadBlock_congr_mem (arr : ArrayDesc) (blockShape blockIdx : List ℕ)
    (junk : ℕ → ℝ) {m₁ m₂ : Mem} (h : ∀ a, m₁.read arr.buf a = m₂.read arr.buf a) :
    loadBlock arr blockShape blockIdx junk m₁ = loadBlock arr blockShape blockIdx junk m₂ := by
  funext loc
  simp only [loadBlock]
  split_ifs
  · exact h _
  · rfl

/-- At any coordinate inside both `blockShape` and `arr.shape`, `loadBlock` reads HBM independently of `junk`. -/
theorem loadBlock_eq_of_inBounds (arr : ArrayDesc) (blockShape blockIdx loc : List ℕ)
    (junk : ℕ → ℝ) (m : Mem)
    (hloc : inBounds blockShape loc = true)
    (hgc : inBounds arr.shape (globalCoord blockShape blockIdx loc) = true) :
    loadBlock arr blockShape blockIdx junk m loc =
      m.read arr.buf (flatCoord arr.shape (globalCoord blockShape blockIdx loc)) := by
  simp [loadBlock, hloc, hgc]

/-- Masked write-back of block `val` to `arr` at `blockIdx`; elements outside `arr` are dropped.
[PAL-EDGE-WRITE] -/
def storeBlock (arr : ArrayDesc) (blockShape blockIdx : List ℕ)
    (val : List ℕ → ℝ) (m : Mem) : Mem :=
  fun b a =>
    if b = arr.buf ∧ addrInBlock arr blockShape blockIdx a = true then
      val (localCoord blockShape blockIdx (unflatCoord arr.shape a))
    else
      m b a

/-- `storeBlock` leaves every address outside the target buffer and block unchanged. -/
theorem storeBlock_read_outside (arr : ArrayDesc) (blockShape blockIdx : List ℕ)
    (val : List ℕ → ℝ) (m : Mem) (b : BufId) (a : ℕ)
    (h : b ≠ arr.buf ∨ addrInBlock arr blockShape blockIdx a = false) :
    (storeBlock arr blockShape blockIdx val m).read b a = m.read b a := by
  simp only [storeBlock, Mem.read]
  split_ifs with hcond
  · rcases h with hbuf | hout
    · exact absurd hcond.1 hbuf
    · rw [hcond.2] at hout
      cases hout
  · rfl

/-- `storeBlock` writes the local block value at every address inside the target buffer and block. -/
theorem storeBlock_read_inside (arr : ArrayDesc) (blockShape blockIdx : List ℕ)
    (val : List ℕ → ℝ) (m : Mem) (a : ℕ)
    (hin : addrInBlock arr blockShape blockIdx a = true) :
    (storeBlock arr blockShape blockIdx val m).read arr.buf a =
      val (localCoord blockShape blockIdx (unflatCoord arr.shape a)) := by
  simp [storeBlock, Mem.read, hin]

/-- `storeBlock` leaves every buffer other than `arr.buf` unchanged. -/
theorem storeBlock_read_other_buf (arr : ArrayDesc) (blockShape blockIdx : List ℕ)
    (val : List ℕ → ℝ) (m : Mem) (b : BufId) (a : ℕ) (h : b ≠ arr.buf) :
    (storeBlock arr blockShape blockIdx val m).read b a = m.read b a :=
  storeBlock_read_outside arr blockShape blockIdx val m b a (Or.inl h)

/-- Writing the same block twice completely overwrites the first write. -/
theorem storeBlock_storeBlock_same (arr : ArrayDesc) (blockShape blockIdx : List ℕ)
    (v₁ v₂ : List ℕ → ℝ) (m : Mem) :
    storeBlock arr blockShape blockIdx v₂ (storeBlock arr blockShape blockIdx v₁ m) =
    storeBlock arr blockShape blockIdx v₂ m := by
  funext b a
  simp only [storeBlock]
  split_ifs <;> rfl

/-- `storeBlock` depends on `val` only at local coordinates of in-block addresses. -/
theorem storeBlock_congr_val (arr : ArrayDesc) (blockShape blockIdx : List ℕ)
    {v₁ v₂ : List ℕ → ℝ} (m : Mem)
    (h : ∀ a, addrInBlock arr blockShape blockIdx a = true →
      v₁ (localCoord blockShape blockIdx (unflatCoord arr.shape a)) =
      v₂ (localCoord blockShape blockIdx (unflatCoord arr.shape a))) :
    storeBlock arr blockShape blockIdx v₁ m = storeBlock arr blockShape blockIdx v₂ m := by
  funext b a
  simp only [storeBlock]
  split_ifs with hcond
  · exact h a hcond.2
  · rfl

/-- Auxiliary pass initializing each unaliased output buffer from its own junk stream, leaving
aliased output buffers untouched (they share their aliased input's HBM buffer under `AliasesValid`).
[PAL-ALIAS-INIT] -/
def initOutMemAux (aliases : List (ℕ × ℕ)) (junk : ℕ → ℝ) :
    ℕ → List (ArrayDesc × BlockSpec) → Mem → Mem
  | _, [], m => m
  | idx, (outArr, _) :: rest, m =>
    let m' : Mem :=
      if aliases.any (fun p => p.2 == idx) then m
      else fun b a => if b = outArr.buf then junkAt junk idx a else m b a
    initOutMemAux aliases junk (idx + 1) rest m'

/-- Initialize output HBM buffers: aliased outputs keep their HBM contents, others start as junk.
[PAL-ALIAS-INIT] [PAL-JUNK-INDEP] -/
def initOutMem (gs : GridSpec) (junk : ℕ → ℝ) (m0 : Mem) : Mem :=
  initOutMemAux gs.aliases junk 0 gs.outSpecs m0

/-- Flush changed output VMEM slots to HBM and prepare initial VMEM blocks for grid point `g`;
`idx` numbers the outputs so each fresh block draws its own junk stream.
[PAL-OUT-NO-LOAD] [PAL-OUT-PERSIST] [PAL-OUT-FLUSH] [PAL-ALIAS-INIT] [PAL-JUNK-INDEP] -/
def flushAndPrep (g : List ℕ) (scalars : ℕ → ℤ) (junk : ℕ → ℝ) :
    ℕ → List (ArrayDesc × BlockSpec) → List OutSlot → Mem →
    Mem × List (List ℕ) × List (List ℕ → ℝ)
  | _, [], _, m => (m, [], [])
  | idx, (outArr, outSpec) :: restSpecs, slots, m =>
    let newIdx := outSpec.blockIdx g scalars
    let prevSlot := match slots with | s :: _ => s | [] => none
    let restSlots := match slots with | _ :: ss => ss | [] => []
    let m1 :=
      match prevSlot with
      | some (oldIdx, oldVal) =>
        if oldIdx = newIdx then m else storeBlock outArr outSpec.blockShape oldIdx oldVal m
      | none => m
    let fresh : List ℕ → ℝ := fun loc => junkAt junk idx (Encodable.encode loc)
    let prevBlock :=
      match prevSlot with
      | some (oldIdx, oldVal) =>
        -- VMEM holds only the block: coordinates outside it do not persist.
        if oldIdx = newIdx then
          (fun loc => if inBounds outSpec.blockShape loc then oldVal loc else fresh loc)
        else fresh
      | none => fresh
    let res := flushAndPrep g scalars junk (idx + 1) restSpecs restSlots m1
    (res.1, newIdx :: res.2.1, prevBlock :: res.2.2)

/-- Flush all remaining VMEM output slots to HBM after the final grid step. [PAL-OUT-FLUSH] -/
def flushAll : List (ArrayDesc × BlockSpec) → List OutSlot → Mem → Mem
  | [], _, m => m
  | (outArr, outSpec) :: restSpecs, slots, m =>
    let prevSlot := match slots with | s :: _ => s | [] => none
    let restSlots := match slots with | _ :: ss => ss | [] => []
    let m1 :=
      match prevSlot with
      | some (oldIdx, oldVal) => storeBlock outArr outSpec.blockShape oldIdx oldVal m
      | none => m
    flushAll restSpecs restSlots m1

/-- Execute one grid point `g`, updating HBM and the persistent VMEM output and scratch slots.
Junk tags: init uses step tag 0 and grid step `g` uses `flatCoord grid g + 1`; within a step
outputs and scratch use role 0 (output `q` gets stream `q`, scratch `j` stream `n + j`) and input
`i` uses role `i + 1`.
[PAL-IN-LOAD] [PAL-IN-READONLY] [PAL-OUT-PERSIST] [PAL-SCRATCH-LIFE] [PAL-JUNK-INDEP] -/
def stepGrid (gs : GridSpec) (body : Body) (scalars : ℕ → ℤ) (junk : ℕ → ℝ)
    (state : Mem × List OutSlot) (g : List ℕ) : Mem × List OutSlot :=
  let step := flatCoord gs.grid g + 1
  let res := flushAndPrep g scalars (junkAt junk (Nat.pair step 0)) 0 gs.outSpecs state.2 state.1
  let m1 := res.1
  let newIdxs := res.2.1
  let outsPrev := res.2.2
  let ins := gs.inSpecs.mapIdx fun i p =>
    loadBlock p.1 p.2.blockShape (p.2.blockIdx g scalars) (junkAt junk (Nat.pair step (i + 1))) m1
  let n := gs.outSpecs.length
  -- Scratch slots follow the output slots. Scratch `j` draws junk as output number `n + j`: at
  -- launch from the init stream, later outside its shape (VMEM holds only the shape).
  let scratchPrev := ((state.2.drop n).zip gs.scratch).mapIdx fun j p => fun loc =>
      match p.1 with
      | some (_, v) =>
        if inBounds p.2 loc then v loc
        else junkAt (junkAt junk (Nat.pair step 0)) (n + j) (Encodable.encode loc)
      | none => junkAt (junkAt junk (Nat.pair 0 0)) (n + j) (Encodable.encode loc)
  let newVals := body g scalars ins (outsPrev ++ scratchPrev)
  (m1, makeSlots newIdxs newVals ++ (newVals.drop n).map fun v => some ([], v))

/-- Sequential execution of a `pallas_call` over a row-major grid, ignoring undefined behaviour.
Use `pallasCall`, which returns `none` for runs that are not `GridSpec.WellDefined`.
[PAL-GRID-ORDER] [PAL-SCRATCH-LIFE] [PAL-SCALAR-PREFETCH] -/
def pallasExec (gs : GridSpec) (body : Body) (junk : ℕ → ℝ) (m0 : Mem) : Mem :=
  let scalars := gs.scalarPrefetch m0
  let mInit := initOutMem gs (junkAt junk (Nat.pair 0 0)) m0
  let initSlots : List OutSlot :=
    (gs.outSpecs.map fun _ => none) ++ gs.scratch.map fun _ => none
  let finalState :=
    (gridPoints gs.grid).foldl (stepGrid gs body scalars junk) (mInit, initSlots)
  flushAll gs.outSpecs finalState.2 finalState.1

open Classical in
/-- TPU `pallas_call` semantics: the sequential execution when the run is well defined, `none`
(undefined behaviour) otherwise. [PAL-UB-NONE] -/
def pallasCall (gs : GridSpec) (body : Body) (junk : ℕ → ℝ) (m0 : Mem) : Option Mem :=
  if gs.WellDefined body m0 then some (pallasExec gs body junk m0) else none

theorem pallasCall_of_wellDefined {gs : GridSpec} {body : Body} {m0 : Mem}
    (junk : ℕ → ℝ) (h : gs.WellDefined body m0) :
    pallasCall gs body junk m0 = some (pallasExec gs body junk m0) := by
  simp [pallasCall, h]

theorem pallasCall_of_not_wellDefined {gs : GridSpec} {body : Body} {m0 : Mem}
    (junk : ℕ → ℝ) (h : ¬ gs.WellDefined body m0) :
    pallasCall gs body junk m0 = none := by
  simp [pallasCall, h]

theorem initOutMemAux_read_other_buf (aliases : List (ℕ × ℕ)) (junk : ℕ → ℝ) (b : BufId) (a : ℕ) :
    ∀ (idx : ℕ) (outSpecs : List (ArrayDesc × BlockSpec)) (m : Mem),
      (∀ x ∈ outSpecs, x.1.buf ≠ b) →
      (initOutMemAux aliases junk idx outSpecs m).read b a = m.read b a
  | _, [], _, _ => rfl
  | idx, (outArr, outSpec) :: rest, m, hne => by
    have hhead : outArr.buf ≠ b := hne (outArr, outSpec) (by simp)
    have htail : ∀ x ∈ rest, x.1.buf ≠ b := fun x hx => hne x (by simp [hx])
    simp only [initOutMemAux]
    rw [initOutMemAux_read_other_buf aliases junk b a (idx + 1) rest _ htail]
    split_ifs
    · rfl
    · simp [Mem.read, Ne.symm hhead]

private theorem initOutMemAux_read_aliased (aliases : List (ℕ × ℕ)) (junk : ℕ → ℝ) (b : BufId)
    (a : ℕ) :
    ∀ (idx : ℕ) (outSpecs : List (ArrayDesc × BlockSpec)) (m : Mem),
      (∀ q outp, outSpecs[q]? = some outp → outp.1.buf = b →
        aliases.any (fun p => p.2 == idx + q) = true) →
      (initOutMemAux aliases junk idx outSpecs m).read b a = m.read b a
  | _, [], _, _ => rfl
  | idx, (outArr, outSpec) :: rest, m, hal => by
    have htail : ∀ q outp, rest[q]? = some outp → outp.1.buf = b →
        aliases.any (fun p => p.2 == idx + 1 + q) = true := by
      intro q outp hq hb
      have := hal (q + 1) outp hq hb
      rwa [show idx + (q + 1) = idx + 1 + q by omega] at this
    simp only [initOutMemAux]
    rw [initOutMemAux_read_aliased aliases junk b a (idx + 1) rest _ htail]
    by_cases hbuf : outArr.buf = b
    · have h0 := hal 0 (outArr, outSpec) rfl hbuf
      simp only [Nat.add_zero] at h0
      simp [h0]
    · split_ifs
      · rfl
      · simp [Mem.read, Ne.symm hbuf]

/-- Under `AliasesValid`, an aliased output buffer keeps its initial HBM contents `m0` after
`initOutMem`. [PAL-ALIAS-INIT] -/
theorem initOutMem_read_aliased {gs : GridSpec} (ha : gs.AliasesValid)
    {q : ℕ} {outp : ArrayDesc × BlockSpec} (hq : gs.outSpecs[q]? = some outp)
    (haliased : gs.aliases.any (fun p => p.2 == q) = true)
    (junk : ℕ → ℝ) (m0 : Mem) (a : ℕ) :
    (initOutMem gs junk m0).read outp.1.buf a = m0.read outp.1.buf a := by
  apply initOutMemAux_read_aliased gs.aliases junk outp.1.buf a 0 gs.outSpecs m0
  intro q' outp' hq' hbuf
  have hqeq : q' = q := by
    have hnd := List.nodup_iff_injective_getElem.mp ha.out_bufs_nodup
    have hlt1 : q' < gs.outSpecs.length := by
      by_contra h; rw [List.getElem?_eq_none (by omega)] at hq'; cases hq'
    have hlt2 : q < gs.outSpecs.length := by
      by_contra h; rw [List.getElem?_eq_none (by omega)] at hq; cases hq
    have h1 : (gs.outSpecs.map fun p => p.1.buf)[q']'(by simpa using hlt1) = outp'.1.buf := by
      simp [List.getElem?_eq_getElem hlt1] at hq'
      simp [hq']
    have h2 : (gs.outSpecs.map fun p => p.1.buf)[q]'(by simpa using hlt2) = outp.1.buf := by
      simp [List.getElem?_eq_getElem hlt2] at hq
      simp [hq]
    have heq := @hnd ⟨q', by simpa using hlt1⟩ ⟨q, by simpa using hlt2⟩
      (by dsimp only; rw [h1, h2, hbuf])
    exact congrArg Fin.val heq
  subst hqeq
  simpa using haliased

theorem flushAndPrep_read_other_buf (g : List ℕ) (scalars : ℕ → ℤ) (junk : ℕ → ℝ)
    (b : BufId) (a : ℕ) :
    ∀ (idx : ℕ) (outSpecs : List (ArrayDesc × BlockSpec)) (slots : List OutSlot) (m : Mem),
      (∀ x ∈ outSpecs, x.1.buf ≠ b) →
      (flushAndPrep g scalars junk idx outSpecs slots m).1.read b a = m.read b a
  | _, [], _, _, _ => rfl
  | idx, (outArr, outSpec) :: rest, slots, m, hne => by
    have hhead : outArr.buf ≠ b := hne (outArr, outSpec) (by simp)
    have htail : ∀ x ∈ rest, x.1.buf ≠ b := fun x hx => hne x (by simp [hx])
    simp only [flushAndPrep]
    rw [flushAndPrep_read_other_buf g scalars junk b a (idx + 1) rest _ _ htail]
    cases slots with
    | nil => rfl
    | cons s ss =>
      cases s with
      | none => rfl
      | some p =>
        simp only
        split_ifs
        · rfl
        · exact storeBlock_read_other_buf outArr outSpec.blockShape p.1 p.2 m b a (Ne.symm hhead)

theorem stepGrid_read_other_buf (gs : GridSpec) (body : Body) (scalars : ℕ → ℤ) (junk : ℕ → ℝ)
    (b : BufId) (a : ℕ) (hne : ∀ x ∈ gs.outSpecs, x.1.buf ≠ b)
    (state : Mem × List OutSlot) (g : List ℕ) :
    (stepGrid gs body scalars junk state g).1.read b a = state.1.read b a :=
  flushAndPrep_read_other_buf g scalars _ b a 0 gs.outSpecs state.2 state.1 hne

theorem foldl_stepGrid_read_other_buf (gs : GridSpec) (body : Body) (scalars : ℕ → ℤ)
    (junk : ℕ → ℝ) (b : BufId) (a : ℕ) (hne : ∀ x ∈ gs.outSpecs, x.1.buf ≠ b) :
    ∀ (pts : List (List ℕ)) (state : Mem × List OutSlot),
      (pts.foldl (stepGrid gs body scalars junk) state).1.read b a = state.1.read b a
  | [], _ => rfl
  | g :: rest, state => by
    simp only [List.foldl_cons]
    rw [foldl_stepGrid_read_other_buf gs body scalars junk b a hne rest _]
    exact stepGrid_read_other_buf gs body scalars junk b a hne state g

theorem flushAll_read_other_buf (b : BufId) (a : ℕ) :
    ∀ (outSpecs : List (ArrayDesc × BlockSpec)) (slots : List OutSlot) (m : Mem),
      (∀ x ∈ outSpecs, x.1.buf ≠ b) →
      (flushAll outSpecs slots m).read b a = m.read b a
  | [], _, _, _ => rfl
  | (outArr, outSpec) :: rest, slots, m, hne => by
    have hhead : outArr.buf ≠ b := hne (outArr, outSpec) (by simp)
    have htail : ∀ x ∈ rest, x.1.buf ≠ b := fun x hx => hne x (by simp [hx])
    simp only [flushAll]
    rw [flushAll_read_other_buf b a rest _ _ htail]
    cases slots with
    | nil => rfl
    | cons s ss =>
      cases s with
      | none => rfl
      | some p =>
        exact storeBlock_read_other_buf outArr outSpec.blockShape p.1 p.2 m b a (Ne.symm hhead)

/-- `pallasExec` leaves every buffer that is not an output buffer completely unchanged.
[PAL-FRAME] -/
theorem pallasExec_read_non_out_buf (gs : GridSpec) (body : Body)
    (junk : ℕ → ℝ) (m0 : Mem) (b : BufId) (a : ℕ)
    (hne : ∀ x ∈ gs.outSpecs, x.1.buf ≠ b) :
    (pallasExec gs body junk m0).read b a = m0.read b a := by
  simp only [pallasExec]
  rw [flushAll_read_other_buf b a gs.outSpecs _ _ hne]
  rw [foldl_stepGrid_read_other_buf gs body _ junk b a hne]
  exact initOutMemAux_read_other_buf gs.aliases _ b a 0 gs.outSpecs m0 hne

/-- Frame corollary: `pallasExec` agrees with `m0` outside the set of its output buffers.
[PAL-FRAME] -/
theorem pallasExec_agreeOutside_outBufs (gs : GridSpec) (body : Body)
    (junk : ℕ → ℝ) (m0 : Mem) :
    Mem.agreeOutside { a : Addr | ∃ x ∈ gs.outSpecs, x.1.buf = a.1 } m0
      (pallasExec gs body junk m0) := by
  intro ⟨b, a⟩ hnot
  have hne : ∀ x ∈ gs.outSpecs, x.1.buf ≠ b := fun x hx heq => hnot ⟨x, hx, heq⟩
  exact (pallasExec_read_non_out_buf gs body junk m0 b a hne).symm

end

end TpuKernelLean.Pallas
