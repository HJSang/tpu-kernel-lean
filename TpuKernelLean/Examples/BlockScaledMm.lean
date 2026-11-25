import TpuKernelLean.Pallas.ReductionKernel

/-!
Verification of two block-scaled FP8 matmul Pallas kernels (the kernel sources are not public
and are not referenced here):

- K1: any `block_size`, a rank-3 block-major `scale_a`, `b` in `[K, N]` layout;
- K2: `block_size = 128`, a rank-2 block-major `scale_a`, and an optional `bias` input added on
  the last grid step.

Both compute `out[r, c] = ∑ b, scale_a[b, r] * scale_b[b, c] * ∑ k in block b, a[r, k] * b[k, c]`
(plus `bias[c]`) on a grid `(M tiles, N tiles, K tiles)`: each grid step unrolls `steps_k`
`block_size`-wide `dot_general`s into a local fp32 `acc`, and a VMEM scratch accumulator adds the
`acc` of successive K tiles. The theorems are stated on the zero-padded arrays the `pallas_call`
sees: `a [PM, PK]`, `b [PK, PN]`, `scale_a [NB, PM, 1]` (K1) or `[NB, PM]` (K2), `scale_b [NB, PN]`,
`bias [PN]`, with `PM = gm * tile_m`, `PN = gn * tile_n`, `PK = gk * (steps_k * block_size)` and
`NB = gk * steps_k`. The padding, the scale expansion and the final slice of the wrappers are
outside the `pallas_call` and are not modelled.

The float operations are opaque (`ScaledMmOps`), so the theorems hold for every implementation of
them:
- `blockwiseScaledMm_correct`, `blockwise128ScaledMm_correct` instantiate
  `ReductionKernel.correct`: the call is defined for all positive grid and tile sizes, output
  element `(r, c)` is the in-order fold over the K tiles of the in-order fold over the unrolled
  iterations (`blockwiseElem`, `blockwise128Elem`), and nothing outside the output buffer changes;
- `blockwiseScaledMm_elem_independent`, `blockwise128ScaledMm_elem_independent`: at one compiled
  shape, output element `(r, c)` depends only on row `r` of `a`, column `c` of `b`, and the `r` /
  `c` entries of the scales (and of the bias). No claim is made across tile shapes;
- `blockwiseScaledMmElem_exact`, `blockwise128ScaledMmElem_exact`: with exact arithmetic
  (`ScaledMmOps.exact`) the element is the block-scaled sum above.

`dimension_semantics=("parallel", "parallel", "arbitrary")` is modelled on the `GridSpec`s: every
`(mi, ni)` writes its own output block, reads no output buffer, and overwrites the scratch
accumulator at `ki == 0`, so any multi-core split or reordering agrees with sequential execution
(`Pallas.pallasExecPar_eq_pallasExec`).
-/

namespace TpuKernelLean.Examples.BlockScaledMm

open TpuKernelLean.Pallas

/-- Opaque TPU float operations of the kernel body (both kernels).

Hardware assumptions for the bit-for-bit theorems (`*_correct`, `*_elem_independent`):
- Each op is a deterministic function of its arguments, lowered the same way at every element of
  the tile and at every grid step.
- For a fixed compiled tile shape `(tile_m, tile_n)`, `dot tile_m tile_n bs` (the MXU's fp8-in,
  fp32-accumulated result for one output element of one `bs`-wide `dot_general` step) depends only
  on the element's `bs`-vector of `a_blk` and `bs`-vector of `b_blk`, not on the other rows or
  columns of the tile, nor on which unrolled iteration issues it.
- For a fixed `(tile_m, tile_n, steps_k)`, `update tile_m tile_n steps_k i` (`acc + dot * sa * sb`
  of unrolled iteration `i`) depends only on its four scalars. It is one op because the compiler
  may fuse it into an FMA, and it carries `i` because each unrolled iteration may be lowered
  differently (for example folding `0 + x` at `i = 0`).

Nothing is assumed across different `tile_m`, `tile_n`, `steps_k` or `bs`: changing the compiled
block shape can change the MXU accumulation order and the fusion. -/
structure ScaledMmOps where
  /-- One output element of one `dot_general` step over a `bs`-wide K block, at compiled tile
  shape `(tileM, tileN)`. -/
  dot : (tileM tileN bs : ℕ) → (Fin bs → ℝ) → (Fin bs → ℝ) → ℝ
  /-- `acc + dot * sa * sb` of unrolled iteration `i`, as one op: arguments `acc dot sa sb`. -/
  update : (tileM tileN stepsK : ℕ) → (i : ℕ) → ℝ → ℝ → ℝ → ℝ → ℝ
  /-- fp32 `acc_ref + acc`. -/
  add : ℝ → ℝ → ℝ
  /-- fp32 `acc_ref + bias` (K2 only). -/
  addBias : ℝ → ℝ → ℝ
  /-- `astype(out_ref.dtype)`. -/
  cast : ℝ → ℝ

/-- Exact real arithmetic, for functional correctness. -/
noncomputable def ScaledMmOps.exact : ScaledMmOps where
  dot := fun _ _ _ a b => ∑ u, a u * b u
  update := fun _ _ _ _ acc d sa sb => acc + d * sa * sb
  add := (· + ·)
  addBias := (· + ·)
  cast := id

/-! ### Helpers -/

/-- Element `l` of tile `i` lies inside `g` tiles of size `t`. -/
private theorem tile_lt {g t i l : ℕ} (hi : i < g) (hl : l < t) : i * t + l < g * t := by
  have h := Nat.mul_le_mul_right t hi
  rw [Nat.succ_mul] at h
  omega

private theorem mul_tile_lt {g t i : ℕ} (hi : i < g) (ht : 0 < t) : i * t < g * t := by
  have h := Nat.mul_le_mul_right t hi
  rw [Nat.succ_mul] at h
  omega

private theorem blockOk_nil : blockOk [] [] [] = true := by simp [blockOk]

private theorem blockOk_cons {d b i : ℕ} {ds bl : List ℕ} {is : List ℤ} (hb : 0 < b)
    (h : i * b < d) (hrest : blockOk ds bl is = true) :
    blockOk (d :: ds) (b :: bl) ((i : ℤ) :: is) = true := by
  have h' : (i : ℤ) * b < d := by exact_mod_cast h
  simp [blockOk, hb, h', hrest]

private theorem inBounds_nil : inBounds [] [] = true := by simp [inBounds]

private theorem inBounds_cons {d x : ℕ} {ds xs : List ℕ} (h : x < d)
    (hrest : inBounds ds xs = true) : inBounds (d :: ds) (x :: xs) = true := by
  simp [inBounds, h, hrest]

private theorem mem_outerGrid_iff {a b : ℕ} {og : List ℕ} :
    og ∈ gridPoints [a, b] ↔ ∃ i j, og = [i, j] ∧ i < a ∧ j < b := by
  rw [mem_gridPoints]
  rcases og with _ | ⟨i, _ | ⟨j, _ | ⟨u, g⟩⟩⟩ <;> simp [inBounds]

/-- Reading an in-bounds local coordinate `loc` of input `i` at grid point `g` fetches HBM. -/
private theorem loadIns_read (gs : GridSpec) (junk : ℕ → ℝ) (m0 : Mem) (g : List ℕ) (i : ℕ)
    (arr : ArrayDesc) (spec : BlockSpec) (hp : gs.inSpecs[i]? = some (arr, spec))
    (idx loc : List ℕ) (hidx : spec.blockIdx g (gs.scalarPrefetch m0) = idx)
    (hloc : inBounds spec.blockShape loc = true)
    (hgc : inBounds arr.shape (globalCoord spec.blockShape idx loc) = true) :
    (loadIns gs junk m0 g).getD i (fun _ => 0) loc =
      m0.read arr.buf (flatCoord arr.shape (globalCoord spec.blockShape idx loc)) := by
  subst hidx
  rw [loadIns_getD gs junk m0 g i (arr, spec) hp]
  exact loadBlock_eq_of_inBounds _ _ _ _ _ _ hloc hgc

private theorem foldl_scaled_exact (bs : ℕ) (ar bc sa sb : ℕ → ℝ) (n : ℕ) :
    (List.range n).foldl (fun acc i =>
        acc + (∑ u : Fin bs, ar (i * bs + u.val) * bc (i * bs + u.val)) * sa i * sb i) 0 =
      ∑ i ∈ Finset.range n,
        sa i * sb i * ∑ k ∈ Finset.range bs, ar (i * bs + k) * bc (i * bs + k) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.foldl_append, ih, Finset.sum_range_succ]
    simp only [List.foldl_cons, List.foldl_nil,
      Fin.sum_univ_eq_sum_range (fun u => ar (n * bs + u) * bc (n * bs + u))]
    ring

variable (ops : ScaledMmOps) (gm gn gk tileM tileN stepsK bs : ℕ)

/-! ### Element spec -/

/-- The local `acc` of one grid step for one output element: starting
from `0`, unrolled iteration `i` adds `dot_i * sa[i] * sb[i]`, where `dot_i` is the `bs`-wide dot
product of columns `i * bs ..< (i + 1) * bs`. `ar`, `bc` are the element's row of the `a` block and
column of the `b` block (length `stepsK * bs`); `sa`, `sb` its scales (length `stepsK`). -/
def tileAcc (ar bc sa sb : ℕ → ℝ) : ℝ :=
  (List.range stepsK).foldl (fun acc i =>
    ops.update tileM tileN stepsK i acc
      (ops.dot tileM tileN bs (fun u => ar (i * bs + u.val)) (fun u => bc (i * bs + u.val)))
      (sa i) (sb i)) 0

/-- `tileAcc` of K tile `t`, on the whole row `ar` of `a` and column `bc` of `b` (length
`gk * (stepsK * bs)`) and the element's scales `sa`, `sb` over all K blocks (length
`gk * stepsK`). -/
def tileElem (t : ℕ) (ar bc sa sb : ℕ → ℝ) : ℝ :=
  tileAcc ops tileM tileN stepsK bs (fun k => ar (t * (stepsK * bs) + k))
    (fun k => bc (t * (stepsK * bs) + k)) (fun i => sa (t * stepsK + i))
    (fun i => sb (t * stepsK + i))

/-- The scratch accumulator after grid step `ki = t`, from its value `s` before: `acc` at `t = 0`,
else `acc_ref + acc`. -/
def accStep (ar bc sa sb : ℕ → ℝ) (s : ℝ) (t : ℕ) : ℝ :=
  if t = 0 then tileElem ops tileM tileN stepsK bs t ar bc sa sb
  else ops.add s (tileElem ops tileM tileN stepsK bs t ar bc sa sb)

/-- The scratch accumulator after the last K tile: the in-order fold of `accStep` over the `gk`
K tiles. -/
def accElem (ar bc sa sb : ℕ → ℝ) : ℝ :=
  (List.range gk).foldl (accStep ops tileM tileN stepsK bs ar bc sa sb) 0

/-- K1's contract for one output element: `cast` of the accumulator after the last K tile. `ar` is
row `r` of `a`, `bc` column `c` of `b`, `sa = scale_a[·, r, 0]`, `sb = scale_b[·, c]`. -/
def blockwiseElem (ar bc sa sb : ℕ → ℝ) : ℝ :=
  ops.cast (accElem ops gk tileM tileN stepsK bs ar bc sa sb)

/-- K2's contract for one output element, with `block_size = 128`: as `blockwiseElem`, with
`bias = bias[c]` added before the cast when `hasBias`. `sa = scale_a[·, r]`. -/
def blockwise128Elem (hasBias : Bool) (ar bc sa sb : ℕ → ℝ) (bias : ℝ) : ℝ :=
  ops.cast (if hasBias then ops.addBias (accElem ops gk tileM tileN stepsK 128 ar bc sa sb) bias
    else accElem ops gk tileM tileN stepsK 128 ar bc sa sb)

/-! ### Block specs -/

/-- `a`: `pl.BlockSpec((tile_m, tile_k), lambda mi, ni, ki: (mi, ki))`, `tile_k = steps_k * bs`. -/
def aSpec : BlockSpec :=
  { blockShape := [tileM, stepsK * bs]
    indexMap := fun g _ => [((g.getD 0 0 : ℕ) : ℤ), ((g.getD 2 0 : ℕ) : ℤ)] }

/-- `b`: `pl.BlockSpec((tile_k, tile_n), lambda mi, ni, ki: (ki, ni))`. -/
def bSpec : BlockSpec :=
  { blockShape := [stepsK * bs, tileN]
    indexMap := fun g _ => [((g.getD 2 0 : ℕ) : ℤ), ((g.getD 1 0 : ℕ) : ℤ)] }

/-- K1 `scale_a`: `pl.BlockSpec((steps_k, tile_m, 1), lambda mi, ni, ki: (ki, mi, 0))`. -/
def saSpec3 : BlockSpec :=
  { blockShape := [stepsK, tileM, 1]
    indexMap := fun g _ => [((g.getD 2 0 : ℕ) : ℤ), ((g.getD 0 0 : ℕ) : ℤ), 0] }

/-- K2 `scale_a`: `pl.BlockSpec((steps_k, tile_m), lambda mi, ni, ki: (ki, mi))`. -/
def saSpec2 : BlockSpec :=
  { blockShape := [stepsK, tileM]
    indexMap := fun g _ => [((g.getD 2 0 : ℕ) : ℤ), ((g.getD 0 0 : ℕ) : ℤ)] }

/-- `scale_b`: `pl.BlockSpec((steps_k, tile_n), lambda mi, ni, ki: (ki, ni))`. -/
def sbSpec : BlockSpec :=
  { blockShape := [stepsK, tileN]
    indexMap := fun g _ => [((g.getD 2 0 : ℕ) : ℤ), ((g.getD 1 0 : ℕ) : ℤ)] }

/-- K2 `bias`: `pl.BlockSpec((tile_n,), lambda mi, ni, ki: (ni,))`. -/
def biasSpec : BlockSpec :=
  { blockShape := [tileN]
    indexMap := fun g _ => [((g.getD 1 0 : ℕ) : ℤ)] }

/-- Output: `pl.BlockSpec((tile_m, tile_n), lambda mi, ni, ki: (mi, ni))`. -/
def oSpec : BlockSpec :=
  { blockShape := [tileM, tileN]
    indexMap := fun g _ => [((g.getD 0 0 : ℕ) : ℤ), ((g.getD 1 0 : ℕ) : ℤ)] }

/-! ### The `pallas_call`s -/

/-- K1's `pallas_call` on the
padded arrays, with `num_scalar_prefetch=0` and no aliases. -/
def blockwiseGridSpec (aBuf bBuf saBuf sbBuf oBuf : BufId) : GridSpec :=
  { grid := [gm, gn, gk]
    inSpecs := [(⟨aBuf, [gm * tileM, gk * (stepsK * bs)]⟩, aSpec tileM stepsK bs),
      (⟨bBuf, [gk * (stepsK * bs), gn * tileN]⟩, bSpec tileN stepsK bs),
      (⟨saBuf, [gk * stepsK, gm * tileM, 1]⟩, saSpec3 tileM stepsK),
      (⟨sbBuf, [gk * stepsK, gn * tileN]⟩, sbSpec tileN stepsK)]
    outSpecs := [(⟨oBuf, [gm * tileM, gn * tileN]⟩, oSpec tileM tileN)]
    scratch := [[tileM, tileN]]
    dimSemantics := [.parallel, .parallel, .arbitrary] }

/-- K2's `pallas_call` on the padded arrays, with
`_BLOCK_SIZE = 128`; the `bias` input is present iff `hasBias`. -/
def blockwise128GridSpec (hasBias : Bool) (aBuf bBuf saBuf sbBuf biasBuf oBuf : BufId) :
    GridSpec :=
  { grid := [gm, gn, gk]
    inSpecs := [(⟨aBuf, [gm * tileM, gk * (stepsK * 128)]⟩, aSpec tileM stepsK 128),
        (⟨bBuf, [gk * (stepsK * 128), gn * tileN]⟩, bSpec tileN stepsK 128),
        (⟨saBuf, [gk * stepsK, gm * tileM]⟩, saSpec2 tileM stepsK),
        (⟨sbBuf, [gk * stepsK, gn * tileN]⟩, sbSpec tileN stepsK)] ++
      if hasBias then [(⟨biasBuf, [gn * tileN]⟩, biasSpec tileN)] else []
    outSpecs := [(⟨oBuf, [gm * tileM, gn * tileN]⟩, oSpec tileM tileN)]
    scratch := [[tileM, tileN]]
    dimSemantics := [.parallel, .parallel, .arbitrary] }

/-- The local `acc` of one grid step at local element `loc = [r, c]`, read from the VMEM blocks
`ins = [a_ref, b_ref, sa_ref, sb_ref, ..]`: `a_ref[r, k]`, `b_ref[k, c]`, `sb_ref[i, c]`, and
`sa_ref[i, r, 0]` when `sa3` (K1) or `sa_ref[i, r]` otherwise
(K2). -/
def stepAcc (sa3 : Bool) (ins : List (List ℕ → ℝ)) (loc : List ℕ) : ℝ :=
  tileAcc ops tileM tileN stepsK bs
    (fun k => (ins.getD 0 fun _ => 0) [loc.getD 0 0, k])
    (fun k => (ins.getD 1 fun _ => 0) [k, loc.getD 1 0])
    (fun i => (ins.getD 2 fun _ => 0) (if sa3 then [i, loc.getD 0 0, 0] else [i, loc.getD 0 0]))
    (fun i => (ins.getD 3 fun _ => 0) [i, loc.getD 1 0])

/-- The kernel body shared by both kernels: compute the local `acc`, set `acc_ref` to `acc` at
`ki == 0` and to `acc_ref + acc` otherwise, and at `ki == gk - 1` write `acc_ref` (plus
`bias_ref[c]` when `hasBias`) cast to the output dtype. `prev` is `[out_ref, acc_ref]`; the result
is `[out_ref, acc_ref]` after the step. -/
noncomputable def body (sa3 hasBias : Bool) : Body := fun g _ ins prev =>
  let ki := g.getD 2 0
  let accRef : List ℕ → ℝ := fun loc =>
    if ki = 0 then stepAcc ops tileM tileN stepsK bs sa3 ins loc
    else ops.add ((prev.getD 1 fun _ => 0) loc) (stepAcc ops tileM tileN stepsK bs sa3 ins loc)
  [fun loc =>
      if ki = gk - 1 then
        ops.cast (if hasBias then
          ops.addBias (accRef loc) ((ins.getD 4 fun _ => 0) [loc.getD 1 0])
        else accRef loc)
      else (prev.getD 0 fun _ => 0) loc,
    accRef]

/-- K1's kernel body: rank-3 `sa_ref`, no bias. -/
noncomputable def blockwiseBody : Body := body ops gk tileM tileN stepsK bs true false

/-- K2's kernel body: `block_size = 128`, rank-2 `sa_ref`, `bias_ref` iff `hasBias`. -/
noncomputable def blockwise128Body (hasBias : Bool) : Body :=
  body ops gk tileM tileN stepsK 128 false hasBias

/-! ### Both kernels as one `ReductionKernel` -/

section Kernel

variable (sa3 hasBias : Bool) (aBuf bBuf saBuf sbBuf biasBuf oBuf : BufId)

/-- Both kernels packaged as a `ReductionKernel` over the 2-D outer tile grid and the K-tile
reduction axis: `sa3` selects K1's rank-3 `scale_a`, `hasBias` adds K2's `bias` input. -/
noncomputable def kernel : ReductionKernel 1 where
  outerGrid := [gm, gn]
  nk := gk
  inSpecs := [(⟨aBuf, [gm * tileM, gk * (stepsK * bs)]⟩, aSpec tileM stepsK bs),
      (⟨bBuf, [gk * (stepsK * bs), gn * tileN]⟩, bSpec tileN stepsK bs),
      if sa3 then (⟨saBuf, [gk * stepsK, gm * tileM, 1]⟩, saSpec3 tileM stepsK)
      else (⟨saBuf, [gk * stepsK, gm * tileM]⟩, saSpec2 tileM stepsK),
      (⟨sbBuf, [gk * stepsK, gn * tileN]⟩, sbSpec tileN stepsK)] ++
    if hasBias then [(⟨biasBuf, [gn * tileN]⟩, biasSpec tileN)] else []
  outArr := ⟨oBuf, [gm * tileM, gn * tileN]⟩
  outBlockShape := [tileM, tileN]
  outIndexMap := fun og _ => [((og.getD 0 0 : ℕ) : ℤ), ((og.getD 1 0 : ℕ) : ℤ)]
  init := fun _ => 0
  step := fun _ t _ ins loc prevAcc _ =>
    if t = 0 then stepAcc ops tileM tileN stepsK bs sa3 ins loc
    else ops.add (prevAcc 0) (stepAcc ops tileM tileN stepsK bs sa3 ins loc)
  finalize := fun _ _ ins loc acc =>
    ops.cast (if hasBias then ops.addBias (acc 0) ((ins.getD 4 fun _ => 0) [loc.getD 1 0])
      else acc 0)
  outerDimSemantics := [.parallel, .parallel]

private theorem kernel_outSpec :
    (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
      oBuf).outSpec = oSpec tileM tileN := by
  simp only [ReductionKernel.outSpec, kernel, oSpec]
  congr 1
  funext g _
  rcases g with _ | ⟨a, _ | ⟨b, g'⟩⟩ <;> rfl

private theorem kernel_gridSpec_blockwise :
    (kernel ops gm gn gk tileM tileN stepsK bs true false aBuf bBuf saBuf sbBuf biasBuf
      oBuf).gridSpec = blockwiseGridSpec gm gn gk tileM tileN stepsK bs aBuf bBuf saBuf sbBuf oBuf := by
  rw [ReductionKernel.gridSpec, ReductionKernel.outPair, kernel_outSpec]
  rfl

private theorem kernel_gridSpec_blockwise128 :
    (kernel ops gm gn gk tileM tileN stepsK 128 false hasBias aBuf bBuf saBuf sbBuf biasBuf
      oBuf).gridSpec =
      blockwise128GridSpec gm gn gk tileM tileN stepsK hasBias aBuf bBuf saBuf sbBuf biasBuf oBuf := by
  rw [ReductionKernel.gridSpec, ReductionKernel.outPair, kernel_outSpec]
  rfl

private theorem kernel_body :
    (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
      oBuf).body = body ops gk tileM tileN stepsK bs sa3 hasBias := by
  funext g sc ins prev
  by_cases h : g[2]?.getD 0 = 0 <;>
    simp [ReductionKernel.body, kernel, body, List.ofFn_succ, h]

private theorem mem_inSpecs {p : ArrayDesc × BlockSpec}
    (hp : p ∈ (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf
      biasBuf oBuf).inSpecs) :
    p = (⟨aBuf, [gm * tileM, gk * (stepsK * bs)]⟩, aSpec tileM stepsK bs) ∨
    p = (⟨bBuf, [gk * (stepsK * bs), gn * tileN]⟩, bSpec tileN stepsK bs) ∨
    p = (⟨saBuf, [gk * stepsK, gm * tileM, 1]⟩, saSpec3 tileM stepsK) ∨
    p = (⟨saBuf, [gk * stepsK, gm * tileM]⟩, saSpec2 tileM stepsK) ∨
    p = (⟨sbBuf, [gk * stepsK, gn * tileN]⟩, sbSpec tileN stepsK) ∨
    (hasBias = true ∧ p = (⟨biasBuf, [gn * tileN]⟩, biasSpec tileN)) := by
  cases sa3 <;> cases hasBias <;> simp [kernel] at hp ⊢ <;> tauto

private theorem kernel_valid (hgm : 0 < gm) (hgn : 0 < gn) (hgk : 0 < gk) (htm : 0 < tileM)
    (htn : 0 < tileN) (hsk : 0 < stepsK) (hbs : 0 < bs)
    (ha : oBuf ≠ aBuf) (hb : oBuf ≠ bBuf) (hsa : oBuf ≠ saBuf) (hsb : oBuf ≠ sbBuf)
    (hbias : hasBias = true → oBuf ≠ biasBuf) (m0 : Mem) :
    (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
      oBuf).Valid m0 where
  outer_grid_pos := by
    intro d hd
    simp only [kernel, List.mem_cons, List.not_mem_nil, or_false] at hd
    rcases hd with rfl | rfl
    · exact hgm
    · exact hgn
  nk_pos := hgk
  in_blocks_ok := by
    intro og hog t ht p hp
    obtain ⟨i, j, rfl, hi, hj⟩ := mem_outerGrid_iff.mp hog
    have ht' : t < gk := ht
    have hk : 0 < stepsK * bs := Nat.mul_pos hsk hbs
    rcases mem_inSpecs ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf
      biasBuf oBuf hp with rfl | rfl | rfl | rfl | rfl | ⟨-, rfl⟩
    · show blockOk [gm * tileM, gk * (stepsK * bs)] [tileM, stepsK * bs]
        [((i : ℕ) : ℤ), ((t : ℕ) : ℤ)] = true
      exact blockOk_cons htm (mul_tile_lt hi htm) (blockOk_cons hk (mul_tile_lt ht' hk) blockOk_nil)
    · show blockOk [gk * (stepsK * bs), gn * tileN] [stepsK * bs, tileN]
        [((t : ℕ) : ℤ), ((j : ℕ) : ℤ)] = true
      exact blockOk_cons hk (mul_tile_lt ht' hk) (blockOk_cons htn (mul_tile_lt hj htn) blockOk_nil)
    · show blockOk [gk * stepsK, gm * tileM, 1] [stepsK, tileM, 1]
        [((t : ℕ) : ℤ), ((i : ℕ) : ℤ), ((0 : ℕ) : ℤ)] = true
      exact blockOk_cons hsk (mul_tile_lt ht' hsk) (blockOk_cons htm (mul_tile_lt hi htm)
        (blockOk_cons Nat.one_pos (by decide) blockOk_nil))
    · show blockOk [gk * stepsK, gm * tileM] [stepsK, tileM] [((t : ℕ) : ℤ), ((i : ℕ) : ℤ)] = true
      exact blockOk_cons hsk (mul_tile_lt ht' hsk) (blockOk_cons htm (mul_tile_lt hi htm) blockOk_nil)
    · show blockOk [gk * stepsK, gn * tileN] [stepsK, tileN] [((t : ℕ) : ℤ), ((j : ℕ) : ℤ)] = true
      exact blockOk_cons hsk (mul_tile_lt ht' hsk) (blockOk_cons htn (mul_tile_lt hj htn) blockOk_nil)
    · show blockOk [gn * tileN] [tileN] [((j : ℕ) : ℤ)] = true
      exact blockOk_cons htn (mul_tile_lt hj htn) blockOk_nil
  out_blocks_ok := by
    intro og hog
    obtain ⟨i, j, rfl, hi, hj⟩ := mem_outerGrid_iff.mp hog
    show blockOk [gm * tileM, gn * tileN] [tileM, tileN] [((i : ℕ) : ℤ), ((j : ℕ) : ℤ)] = true
    exact blockOk_cons htm (mul_tile_lt hi htm) (blockOk_cons htn (mul_tile_lt hj htn) blockOk_nil)
  out_inj := by
    intro og₁ h₁ og₂ h₂ heq
    obtain ⟨i₁, j₁, rfl, _, _⟩ := mem_outerGrid_iff.mp h₁
    obtain ⟨i₂, j₂, rfl, _, _⟩ := mem_outerGrid_iff.mp h₂
    simp only [kernel, List.getD_cons_zero, List.getD_cons_succ, List.cons.injEq,
      Nat.cast_inj, and_true] at heq
    simp [heq]
  ins_disjoint := by
    intro p hp
    rcases mem_inSpecs ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf
      biasBuf oBuf hp with rfl | rfl | rfl | rfl | rfl | ⟨hB, rfl⟩
    · exact ha.symm
    · exact hb.symm
    · exact hsa.symm
    · exact hsa.symm
    · exact hsb.symm
    · exact (hbias hB).symm

/-- At grid step `(i0, j0, n)`, the in-bounds elements of every VMEM input block are the HBM
elements of the padded arrays, whatever the junk. -/
private theorem kernel_reads (junk : ℕ → ℝ) (m0 : Mem) {i0 j0 n rl cl : ℕ} (hi0 : i0 < gm)
    (hj0 : j0 < gn) (hn : n < gk) (hrl : rl < tileM) (hcl : cl < tileN) :
    (∀ k < stepsK * bs,
      (loadIns (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
        oBuf).gridSpec junk m0 ([i0, j0] ++ [n])).getD 0 (fun _ => 0) [rl, k] =
        m0.read aBuf ((i0 * tileM + rl) * (gk * (stepsK * bs)) + (n * (stepsK * bs) + k))) ∧
    (∀ k < stepsK * bs,
      (loadIns (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
        oBuf).gridSpec junk m0 ([i0, j0] ++ [n])).getD 1 (fun _ => 0) [k, cl] =
        m0.read bBuf ((n * (stepsK * bs) + k) * (gn * tileN) + (j0 * tileN + cl))) ∧
    (∀ i < stepsK,
      (loadIns (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
        oBuf).gridSpec junk m0 ([i0, j0] ++ [n])).getD 2 (fun _ => 0)
          (if sa3 then [i, rl, 0] else [i, rl]) =
        m0.read saBuf ((n * stepsK + i) * (gm * tileM) + (i0 * tileM + rl))) ∧
    (∀ i < stepsK,
      (loadIns (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
        oBuf).gridSpec junk m0 ([i0, j0] ++ [n])).getD 3 (fun _ => 0) [i, cl] =
        m0.read sbBuf ((n * stepsK + i) * (gn * tileN) + (j0 * tileN + cl))) ∧
    (hasBias = true →
      (loadIns (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
        oBuf).gridSpec junk m0 ([i0, j0] ++ [n])).getD 4 (fun _ => 0) [cl] =
        m0.read biasBuf (j0 * tileN + cl)) := by
  have hr : i0 * tileM + rl < gm * tileM := tile_lt hi0 hrl
  have hc : j0 * tileN + cl < gn * tileN := tile_lt hj0 hcl
  have hkk : ∀ k < stepsK * bs, n * (stepsK * bs) + k < gk * (stepsK * bs) :=
    fun k hk => tile_lt hn hk
  have hii : ∀ i < stepsK, n * stepsK + i < gk * stepsK := fun i hi => tile_lt hn hi
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro k hk
    refine (loadIns_read _ junk m0 _ 0 ⟨aBuf, [gm * tileM, gk * (stepsK * bs)]⟩
      (aSpec tileM stepsK bs) ?_ [i0, n] [rl, k] ?_ ?_ ?_).trans ?_
    · simp [ReductionKernel.gridSpec, kernel]
    · simp only [aSpec, BlockSpec.blockIdx, List.cons_append, List.nil_append,
        List.getD_cons_zero, List.getD_cons_succ, List.map_cons, List.map_nil, Int.toNat_natCast]
    · exact inBounds_cons hrl (inBounds_cons hk inBounds_nil)
    · exact inBounds_cons hr (inBounds_cons (hkk k hk) inBounds_nil)
    · simp [aSpec, globalCoord, flatCoord]
  · intro k hk
    refine (loadIns_read _ junk m0 _ 1 ⟨bBuf, [gk * (stepsK * bs), gn * tileN]⟩
      (bSpec tileN stepsK bs) ?_ [n, j0] [k, cl] ?_ ?_ ?_).trans ?_
    · simp [ReductionKernel.gridSpec, kernel]
    · simp only [bSpec, BlockSpec.blockIdx, List.cons_append, List.nil_append,
        List.getD_cons_zero, List.getD_cons_succ, List.map_cons, List.map_nil, Int.toNat_natCast]
    · exact inBounds_cons hk (inBounds_cons hcl inBounds_nil)
    · exact inBounds_cons (hkk k hk) (inBounds_cons hc inBounds_nil)
    · simp [bSpec, globalCoord, flatCoord]
  · intro i hi
    cases sa3
    · show (loadIns _ junk m0 _).getD 2 (fun _ => 0) [i, rl] = _
      refine (loadIns_read _ junk m0 _ 2 ⟨saBuf, [gk * stepsK, gm * tileM]⟩
        (saSpec2 tileM stepsK) ?_ [n, i0] [i, rl] ?_ ?_ ?_).trans ?_
      · simp [ReductionKernel.gridSpec, kernel]
      · simp only [saSpec2, BlockSpec.blockIdx, List.cons_append, List.nil_append,
          List.getD_cons_zero, List.getD_cons_succ, List.map_cons, List.map_nil, Int.toNat_natCast]
      · exact inBounds_cons hi (inBounds_cons hrl inBounds_nil)
      · exact inBounds_cons (hii i hi) (inBounds_cons hr inBounds_nil)
      · simp [saSpec2, globalCoord, flatCoord]
    · show (loadIns _ junk m0 _).getD 2 (fun _ => 0) [i, rl, 0] = _
      refine (loadIns_read _ junk m0 _ 2 ⟨saBuf, [gk * stepsK, gm * tileM, 1]⟩
        (saSpec3 tileM stepsK) ?_ [n, i0, 0] [i, rl, 0] ?_ ?_ ?_).trans ?_
      · simp [ReductionKernel.gridSpec, kernel]
      · simp only [saSpec3, BlockSpec.blockIdx, List.cons_append, List.nil_append,
          List.getD_cons_zero, List.getD_cons_succ, List.map_cons, List.map_nil, Int.toNat_natCast,
          Int.toNat_zero]
      · exact inBounds_cons hi (inBounds_cons hrl (inBounds_cons Nat.one_pos inBounds_nil))
      · exact inBounds_cons (hii i hi) (inBounds_cons hr (inBounds_cons (by decide) inBounds_nil))
      · simp [saSpec3, globalCoord, flatCoord]
  · intro i hi
    refine (loadIns_read _ junk m0 _ 3 ⟨sbBuf, [gk * stepsK, gn * tileN]⟩
      (sbSpec tileN stepsK) ?_ [n, j0] [i, cl] ?_ ?_ ?_).trans ?_
    · simp [ReductionKernel.gridSpec, kernel]
    · simp only [sbSpec, BlockSpec.blockIdx, List.cons_append, List.nil_append,
        List.getD_cons_zero, List.getD_cons_succ, List.map_cons, List.map_nil, Int.toNat_natCast]
    · exact inBounds_cons hi (inBounds_cons hcl inBounds_nil)
    · exact inBounds_cons (hii i hi) (inBounds_cons hc inBounds_nil)
    · simp [sbSpec, globalCoord, flatCoord]
  · intro hB
    subst hB
    refine (loadIns_read _ junk m0 _ 4 ⟨biasBuf, [gn * tileN]⟩ (biasSpec tileN) ?_ [j0] [cl]
      ?_ ?_ ?_).trans ?_
    · simp [ReductionKernel.gridSpec, kernel]
    · simp only [biasSpec, BlockSpec.blockIdx, List.cons_append, List.nil_append,
        List.getD_cons_zero, List.getD_cons_succ, List.map_cons, List.map_nil, Int.toNat_natCast]
    · exact inBounds_cons hcl inBounds_nil
    · exact inBounds_cons hc inBounds_nil
    · simp [biasSpec, globalCoord, flatCoord]

/-- `tileAcc` depends only on the first `stepsK * bs` entries of `ar`, `bc` and the first `stepsK`
entries of `sa`, `sb`. -/
private theorem tileAcc_congr {ar ar' bc bc' sa sa' sb sb' : ℕ → ℝ}
    (ha : ∀ k < stepsK * bs, ar k = ar' k) (hb : ∀ k < stepsK * bs, bc k = bc' k)
    (hsa : ∀ i < stepsK, sa i = sa' i) (hsb : ∀ i < stepsK, sb i = sb' i) :
    tileAcc ops tileM tileN stepsK bs ar bc sa sb =
      tileAcc ops tileM tileN stepsK bs ar' bc' sa' sb' := by
  unfold tileAcc
  apply List.foldl_ext
  intro acc i hi
  rw [List.mem_range] at hi
  have hk : ∀ u : Fin bs, i * bs + u.val < stepsK * bs := fun u => tile_lt hi u.isLt
  have e1 : (fun u : Fin bs => ar (i * bs + u.val)) = fun u => ar' (i * bs + u.val) :=
    funext fun u => ha _ (hk u)
  have e2 : (fun u : Fin bs => bc (i * bs + u.val)) = fun u => bc' (i * bs + u.val) :=
    funext fun u => hb _ (hk u)
  show ops.update tileM tileN stepsK i acc (ops.dot tileM tileN bs _ _) (sa i) (sb i) =
    ops.update tileM tileN stepsK i acc (ops.dot tileM tileN bs _ _) (sa' i) (sb' i)
  rw [e1, e2, hsa i hi, hsb i hi]

/-- `accElem` depends only on the entries of the row, column and scales that exist. -/
private theorem accElem_congr {ar ar' bc bc' sa sa' sb sb' : ℕ → ℝ}
    (ha : ∀ k < gk * (stepsK * bs), ar k = ar' k) (hb : ∀ k < gk * (stepsK * bs), bc k = bc' k)
    (hsa : ∀ j < gk * stepsK, sa j = sa' j) (hsb : ∀ j < gk * stepsK, sb j = sb' j) :
    accElem ops gk tileM tileN stepsK bs ar bc sa sb =
      accElem ops gk tileM tileN stepsK bs ar' bc' sa' sb' := by
  unfold accElem
  apply List.foldl_ext
  intro s t ht
  rw [List.mem_range] at ht
  have e : tileElem ops tileM tileN stepsK bs t ar bc sa sb =
      tileElem ops tileM tileN stepsK bs t ar' bc' sa' sb' := by
    unfold tileElem
    exact tileAcc_congr ops tileM tileN stepsK bs (fun k hk => ha _ (tile_lt ht hk))
      (fun k hk => hb _ (tile_lt ht hk)) (fun i hi => hsa _ (tile_lt ht hi))
      (fun i hi => hsb _ (tile_lt ht hi))
  unfold accStep
  rw [e]

/-- The scratch accumulator of tile `(i0, j0)` at local element `(rl, cl)` after `n` K tiles is
the `n`-tile prefix of `accElem` on the element's row, column and scales in initial memory. -/
private theorem kernel_accFold (junk : ℕ → ℝ) (m0 : Mem) {i0 j0 rl cl : ℕ} (hi0 : i0 < gm)
    (hj0 : j0 < gn) (hrl : rl < tileM) (hcl : cl < tileN) :
    ∀ n ≤ gk,
      ((kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
        oBuf).accFold junk m0 [i0, j0] [rl, cl] n) 0 =
      (List.range n).foldl (accStep ops tileM tileN stepsK bs
        (fun k => m0.read aBuf ((i0 * tileM + rl) * (gk * (stepsK * bs)) + k))
        (fun k => m0.read bBuf (k * (gn * tileN) + (j0 * tileN + cl)))
        (fun j => m0.read saBuf (j * (gm * tileM) + (i0 * tileM + rl)))
        (fun j => m0.read sbBuf (j * (gn * tileN) + (j0 * tileN + cl)))) 0 := by
  intro n hn
  induction n with
  | zero => rfl
  | succ n ih =>
    obtain ⟨hA, hB, hSA, hSB, -⟩ := kernel_reads ops gm gn gk tileM tileN stepsK bs sa3 hasBias
      aBuf bBuf saBuf sbBuf biasBuf oBuf junk m0 hi0 hj0 (show n < gk by omega) hrl hcl
    have htile : stepAcc ops tileM tileN stepsK bs sa3
        (loadIns (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf
          biasBuf oBuf).gridSpec junk m0 ([i0, j0] ++ [n])) [rl, cl] =
        tileElem ops tileM tileN stepsK bs n
          (fun k => m0.read aBuf ((i0 * tileM + rl) * (gk * (stepsK * bs)) + k))
          (fun k => m0.read bBuf (k * (gn * tileN) + (j0 * tileN + cl)))
          (fun j => m0.read saBuf (j * (gm * tileM) + (i0 * tileM + rl)))
          (fun j => m0.read sbBuf (j * (gn * tileN) + (j0 * tileN + cl))) := by
      unfold stepAcc tileElem
      exact tileAcc_congr ops tileM tileN stepsK bs hA hB hSA hSB
    rw [ReductionKernel.accFold_succ, List.range_succ, List.foldl_append, List.foldl_cons,
      List.foldl_nil, ← ih (by omega), accStep, ← htile]
    rfl

/-- Both kernels at once: the run is defined, output element `(r, c)` is `cast` of `accElem`
(plus the bias when `hasBias`), and nothing outside the output buffer changes. -/
private theorem kernel_correct (hgm : 0 < gm) (hgn : 0 < gn) (hgk : 0 < gk) (htm : 0 < tileM)
    (htn : 0 < tileN) (hsk : 0 < stepsK) (hbs : 0 < bs)
    (ha : oBuf ≠ aBuf) (hb : oBuf ≠ bBuf) (hsa : oBuf ≠ saBuf) (hsb : oBuf ≠ sbBuf)
    (hbias : hasBias = true → oBuf ≠ biasBuf) (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', pallasCall
        (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
          oBuf).gridSpec
        (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
          oBuf).body junk m0 = some m' ∧
      (∀ (r : Fin (gm * tileM)) (c : Fin (gn * tileN)),
        m'.read oBuf (r.val * (gn * tileN) + c.val) =
          ops.cast (if hasBias then
            ops.addBias (accElem ops gk tileM tileN stepsK bs
              (fun k => m0.read aBuf (r.val * (gk * (stepsK * bs)) + k))
              (fun k => m0.read bBuf (k * (gn * tileN) + c.val))
              (fun j => m0.read saBuf (j * (gm * tileM) + r.val))
              (fun j => m0.read sbBuf (j * (gn * tileN) + c.val))) (m0.read biasBuf c.val)
          else accElem ops gk tileM tileN stepsK bs
            (fun k => m0.read aBuf (r.val * (gk * (stepsK * bs)) + k))
            (fun k => m0.read bBuf (k * (gn * tileN) + c.val))
            (fun j => m0.read saBuf (j * (gm * tileM) + r.val))
            (fun j => m0.read sbBuf (j * (gn * tileN) + c.val)))) ∧
      Mem.agreeOutside (Mem.bufFootprint oBuf) m0 m' := by
  have hv := kernel_valid ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf
    biasBuf oBuf hgm hgn hgk htm htn hsk hbs ha hb hsa hsb hbias m0
  obtain ⟨m', hcall, hval, hframe⟩ := ReductionKernel.correct _ hv junk
  refine ⟨m', hcall, ?_, hframe⟩
  intro r c
  have hi0 : r.val / tileM < gm :=
    Nat.div_lt_of_lt_mul (lt_of_lt_of_eq r.isLt (Nat.mul_comm gm tileM))
  have hj0 : c.val / tileN < gn :=
    Nat.div_lt_of_lt_mul (lt_of_lt_of_eq c.isLt (Nat.mul_comm gn tileN))
  have hog : [r.val / tileM, c.val / tileN] ∈ gridPoints
      (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
        oBuf).outerGrid :=
    mem_outerGrid_iff.mpr ⟨_, _, rfl, hi0, hj0⟩
  have hin : addrInBlock
      (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
        oBuf).outArr
      (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
        oBuf).outBlockShape
      ((kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
        oBuf).outBlockIdx [r.val / tileM, c.val / tileN]
        ((kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
          oBuf).scalarPrefetch m0))
      (r.val * (gn * tileN) + c.val) = true := by
    simp only [kernel, ReductionKernel.outBlockIdx, List.getD_cons_zero, List.getD_cons_succ,
      List.map_cons, List.map_nil, Int.toNat_natCast]
    rw [addrInBlock_two htm htn r.isLt c.isLt]
    simp
  refine (hval [r.val / tileM, c.val / tileN] hog (r.val * (gn * tileN) + c.val) hin).trans ?_
  have hr' : r.val / tileM * tileM + (r.val - r.val / tileM * tileM) = r.val := by
    have := Nat.div_mul_le_self r.val tileM; omega
  have hc' : c.val / tileN * tileN + (c.val - c.val / tileN * tileN) = c.val := by
    have := Nat.div_mul_le_self c.val tileN; omega
  have hrl : r.val - r.val / tileM * tileM < tileM := by
    have := Nat.lt_div_mul_add (a := r.val) htm; omega
  have hcl : c.val - c.val / tileN * tileN < tileN := by
    have := Nat.lt_div_mul_add (a := c.val) htn; omega
  have hloc : localCoord
      (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
        oBuf).outBlockShape
      ((kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
        oBuf).outBlockIdx [r.val / tileM, c.val / tileN]
        ((kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
          oBuf).scalarPrefetch m0))
      (unflatCoord
        (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
          oBuf).outArr.shape (r.val * (gn * tileN) + c.val)) =
      [r.val - r.val / tileM * tileM, c.val - c.val / tileN * tileN] := by
    simp only [kernel, ReductionKernel.outBlockIdx, List.getD_cons_zero, List.getD_cons_succ,
      List.map_cons, List.map_nil, Int.toNat_natCast,
      unflatCoord_two (gm * tileM) (gn * tileN) r.val c.val c.isLt, localCoord]
  rw [hloc]
  have hE : ((kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
      oBuf).accFold junk m0 [r.val / tileM, c.val / tileN]
      [r.val - r.val / tileM * tileM, c.val - c.val / tileN * tileN] gk) 0 =
      accElem ops gk tileM tileN stepsK bs
        (fun k => m0.read aBuf ((r.val / tileM * tileM + (r.val - r.val / tileM * tileM)) *
          (gk * (stepsK * bs)) + k))
        (fun k => m0.read bBuf (k * (gn * tileN) +
          (c.val / tileN * tileN + (c.val - c.val / tileN * tileN))))
        (fun j => m0.read saBuf (j * (gm * tileM) +
          (r.val / tileM * tileM + (r.val - r.val / tileM * tileM))))
        (fun j => m0.read sbBuf (j * (gn * tileN) +
          (c.val / tileN * tileN + (c.val - c.val / tileN * tileN)))) :=
    kernel_accFold ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
      oBuf junk m0 hi0 hj0 hrl hcl gk le_rfl
  obtain ⟨-, -, -, -, hBias⟩ := kernel_reads ops gm gn gk tileM tileN stepsK bs sa3 hasBias
    aBuf bBuf saBuf sbBuf biasBuf oBuf junk m0 hi0 hj0 (show gk - 1 < gk by omega) hrl hcl
  rw [hr', hc'] at hE
  rw [hc'] at hBias
  change ops.cast (if hasBias = true then
      ops.addBias
        (((kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
          oBuf).accFold junk m0 [r.val / tileM, c.val / tileN]
          [r.val - r.val / tileM * tileM, c.val - c.val / tileN * tileN] gk) 0)
        ((loadIns (kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf
          biasBuf oBuf).gridSpec junk m0 ([r.val / tileM, c.val / tileN] ++ [gk - 1])).getD 4
          (fun _ => 0) [c.val - c.val / tileN * tileN])
    else ((kernel ops gm gn gk tileM tileN stepsK bs sa3 hasBias aBuf bBuf saBuf sbBuf biasBuf
      oBuf).accFold junk m0 [r.val / tileM, c.val / tileN]
      [r.val - r.val / tileM * tileM, c.val - c.val / tileN * tileN] gk) 0) = _
  rw [hE]
  by_cases hB : hasBias = true
  · rw [hBias hB]
  · simp only [hB, Bool.false_eq_true, ↓reduceIte]

end Kernel

/-! ### K1: any block size, rank-3 `scale_a` -/

/-- K1's `pallas_call` is defined for all positive grid sizes, tile sizes and block size, every
output element `(r, c)` of the padded output is `blockwiseElem` of row `r` of `a`, column `c` of
`b`, `scale_a[·, r, 0]` and `scale_b[·, c]`, and nothing outside the output buffer changes. -/
theorem blockwiseScaledMm_correct (aBuf bBuf saBuf sbBuf oBuf : BufId)
    (hgm : 0 < gm) (hgn : 0 < gn) (hgk : 0 < gk) (htm : 0 < tileM) (htn : 0 < tileN)
    (hsk : 0 < stepsK) (hbs : 0 < bs)
    (ha : oBuf ≠ aBuf) (hb : oBuf ≠ bBuf) (hsa : oBuf ≠ saBuf) (hsb : oBuf ≠ sbBuf)
    (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', pallasCall (blockwiseGridSpec gm gn gk tileM tileN stepsK bs aBuf bBuf saBuf sbBuf oBuf)
        (blockwiseBody ops gk tileM tileN stepsK bs) junk m0 = some m' ∧
      (∀ (r : Fin (gm * tileM)) (c : Fin (gn * tileN)),
        m'.read oBuf (r.val * (gn * tileN) + c.val) =
          blockwiseElem ops gk tileM tileN stepsK bs
            (fun k => m0.read aBuf (r.val * (gk * (stepsK * bs)) + k))
            (fun k => m0.read bBuf (k * (gn * tileN) + c.val))
            (fun j => m0.read saBuf (j * (gm * tileM) + r.val))
            (fun j => m0.read sbBuf (j * (gn * tileN) + c.val))) ∧
      Mem.agreeOutside (Mem.bufFootprint oBuf) m0 m' := by
  have h := kernel_correct ops gm gn gk tileM tileN stepsK bs true false aBuf bBuf saBuf sbBuf
    aBuf oBuf hgm hgn hgk htm htn hsk hbs ha hb hsa hsb (fun h => Bool.noConfusion h) junk m0
  rw [kernel_gridSpec_blockwise, kernel_body] at h
  exact h

/-- K1, bit for bit under the `ScaledMmOps` assumptions, at one compiled shape: two runs whose
initial memories agree on row `r` of `a`, column `c` of `b`, `scale_a[·, r, 0]` and
`scale_b[·, c]` give the same output element `(r, c)`, whatever the other rows, columns and
junk. -/
theorem blockwiseScaledMm_elem_independent {aBuf bBuf saBuf sbBuf oBuf : BufId}
    (hgm : 0 < gm) (hgn : 0 < gn) (hgk : 0 < gk) (htm : 0 < tileM) (htn : 0 < tileN)
    (hsk : 0 < stepsK) (hbs : 0 < bs)
    (ha : oBuf ≠ aBuf) (hb : oBuf ≠ bBuf) (hsa : oBuf ≠ saBuf) (hsb : oBuf ≠ sbBuf)
    {junk junk' : ℕ → ℝ} {m0 m0' m₁ m₂ : Mem}
    (h₁ : pallasCall (blockwiseGridSpec gm gn gk tileM tileN stepsK bs aBuf bBuf saBuf sbBuf oBuf)
      (blockwiseBody ops gk tileM tileN stepsK bs) junk m0 = some m₁)
    (h₂ : pallasCall (blockwiseGridSpec gm gn gk tileM tileN stepsK bs aBuf bBuf saBuf sbBuf oBuf)
      (blockwiseBody ops gk tileM tileN stepsK bs) junk' m0' = some m₂)
    (r : Fin (gm * tileM)) (c : Fin (gn * tileN))
    (har : ∀ k < gk * (stepsK * bs), m0.read aBuf (r.val * (gk * (stepsK * bs)) + k) =
      m0'.read aBuf (r.val * (gk * (stepsK * bs)) + k))
    (hbc : ∀ k < gk * (stepsK * bs), m0.read bBuf (k * (gn * tileN) + c.val) =
      m0'.read bBuf (k * (gn * tileN) + c.val))
    (hsar : ∀ j < gk * stepsK, m0.read saBuf (j * (gm * tileM) + r.val) =
      m0'.read saBuf (j * (gm * tileM) + r.val))
    (hsbc : ∀ j < gk * stepsK, m0.read sbBuf (j * (gn * tileN) + c.val) =
      m0'.read sbBuf (j * (gn * tileN) + c.val)) :
    m₁.read oBuf (r.val * (gn * tileN) + c.val) = m₂.read oBuf (r.val * (gn * tileN) + c.val) := by
  obtain ⟨n₁, e₁, v₁, -⟩ := blockwiseScaledMm_correct ops gm gn gk tileM tileN stepsK bs
    aBuf bBuf saBuf sbBuf oBuf hgm hgn hgk htm htn hsk hbs ha hb hsa hsb junk m0
  obtain ⟨n₂, e₂, v₂, -⟩ := blockwiseScaledMm_correct ops gm gn gk tileM tileN stepsK bs
    aBuf bBuf saBuf sbBuf oBuf hgm hgn hgk htm htn hsk hbs ha hb hsa hsb junk' m0'
  rw [e₁, Option.some.injEq] at h₁
  rw [e₂, Option.some.injEq] at h₂
  subst h₁ h₂
  rw [v₁, v₂, blockwiseElem, blockwiseElem,
    accElem_congr ops gk tileM tileN stepsK bs har hbc hsar hsbc]

private theorem tileAcc_exact (ar bc sa sb : ℕ → ℝ) :
    tileAcc ScaledMmOps.exact tileM tileN stepsK bs ar bc sa sb =
      ∑ i ∈ Finset.range stepsK,
        sa i * sb i * ∑ k ∈ Finset.range bs, ar (i * bs + k) * bc (i * bs + k) :=
  foldl_scaled_exact bs ar bc sa sb stepsK

private theorem tileElem_exact (t : ℕ) (ar bc sa sb : ℕ → ℝ) :
    tileElem ScaledMmOps.exact tileM tileN stepsK bs t ar bc sa sb =
      ∑ i ∈ Finset.range stepsK, sa (t * stepsK + i) * sb (t * stepsK + i) *
        ∑ k ∈ Finset.range bs,
          ar ((t * stepsK + i) * bs + k) * bc ((t * stepsK + i) * bs + k) := by
  rw [tileElem, tileAcc_exact]
  simp only [show ∀ i k : ℕ, t * (stepsK * bs) + (i * bs + k) = (t * stepsK + i) * bs + k from
    fun i k => by ring]

private theorem accElem_exact (ar bc sa sb : ℕ → ℝ) :
    accElem ScaledMmOps.exact gk tileM tileN stepsK bs ar bc sa sb =
      ∑ j ∈ Finset.range (gk * stepsK),
        sa j * sb j * ∑ k ∈ Finset.range bs, ar (j * bs + k) * bc (j * bs + k) := by
  unfold accElem
  induction gk with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.foldl_append, ih, List.foldl_cons, List.foldl_nil, accStep,
      tileElem_exact, Nat.succ_mul, Finset.sum_range_add]
    split_ifs with h
    · subst h
      simp
    · rfl

/-- With exact arithmetic K1's element is the block-scaled matmul over all `gk * stepsK` K blocks:
`∑ b, scale_a[b, r] * scale_b[b, c] * ∑ k < bs, a[r, b * bs + k] * b[b * bs + k, c]`. -/
theorem blockwiseScaledMmElem_exact (ar bc sa sb : ℕ → ℝ) :
    blockwiseElem ScaledMmOps.exact gk tileM tileN stepsK bs ar bc sa sb =
      ∑ j ∈ Finset.range (gk * stepsK),
        sa j * sb j * ∑ k ∈ Finset.range bs, ar (j * bs + k) * bc (j * bs + k) := by
  show accElem ScaledMmOps.exact gk tileM tileN stepsK bs ar bc sa sb = _
  exact accElem_exact gk tileM tileN stepsK bs ar bc sa sb

/-! ### K2: block size 128, rank-2 `scale_a`, optional bias -/

/-- K2's `pallas_call`, with or without the bias input, is defined for all positive grid and tile
sizes, every output element `(r, c)` of the padded output is `blockwise128Elem` of row `r` of `a`,
column `c` of `b`, `scale_a[·, r]`, `scale_b[·, c]` and `bias[c]`, and nothing outside the output
buffer changes. Without the bias input (`hasBias = false`) `biasBuf` is not read. -/
theorem blockwise128ScaledMm_correct (hasBias : Bool) (aBuf bBuf saBuf sbBuf biasBuf oBuf : BufId)
    (hgm : 0 < gm) (hgn : 0 < gn) (hgk : 0 < gk) (htm : 0 < tileM) (htn : 0 < tileN)
    (hsk : 0 < stepsK)
    (ha : oBuf ≠ aBuf) (hb : oBuf ≠ bBuf) (hsa : oBuf ≠ saBuf) (hsb : oBuf ≠ sbBuf)
    (hbias : hasBias = true → oBuf ≠ biasBuf) (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', pallasCall
        (blockwise128GridSpec gm gn gk tileM tileN stepsK hasBias aBuf bBuf saBuf sbBuf biasBuf
          oBuf)
        (blockwise128Body ops gk tileM tileN stepsK hasBias) junk m0 = some m' ∧
      (∀ (r : Fin (gm * tileM)) (c : Fin (gn * tileN)),
        m'.read oBuf (r.val * (gn * tileN) + c.val) =
          blockwise128Elem ops gk tileM tileN stepsK hasBias
            (fun k => m0.read aBuf (r.val * (gk * (stepsK * 128)) + k))
            (fun k => m0.read bBuf (k * (gn * tileN) + c.val))
            (fun j => m0.read saBuf (j * (gm * tileM) + r.val))
            (fun j => m0.read sbBuf (j * (gn * tileN) + c.val))
            (m0.read biasBuf c.val)) ∧
      Mem.agreeOutside (Mem.bufFootprint oBuf) m0 m' := by
  have h := kernel_correct ops gm gn gk tileM tileN stepsK 128 false hasBias aBuf bBuf saBuf sbBuf
    biasBuf oBuf hgm hgn hgk htm htn hsk (by decide) ha hb hsa hsb hbias junk m0
  rw [kernel_gridSpec_blockwise128, kernel_body] at h
  exact h

/-- K2, bit for bit under the `ScaledMmOps` assumptions, at one compiled shape: two runs whose
initial memories agree on row `r` of `a`, column `c` of `b`, `scale_a[·, r]`, `scale_b[·, c]` and
(with the bias input) `bias[c]` give the same output element `(r, c)`. -/
theorem blockwise128ScaledMm_elem_independent {hasBias : Bool}
    {aBuf bBuf saBuf sbBuf biasBuf oBuf : BufId}
    (hgm : 0 < gm) (hgn : 0 < gn) (hgk : 0 < gk) (htm : 0 < tileM) (htn : 0 < tileN)
    (hsk : 0 < stepsK)
    (ha : oBuf ≠ aBuf) (hb : oBuf ≠ bBuf) (hsa : oBuf ≠ saBuf) (hsb : oBuf ≠ sbBuf)
    (hbias : hasBias = true → oBuf ≠ biasBuf)
    {junk junk' : ℕ → ℝ} {m0 m0' m₁ m₂ : Mem}
    (h₁ : pallasCall
      (blockwise128GridSpec gm gn gk tileM tileN stepsK hasBias aBuf bBuf saBuf sbBuf biasBuf oBuf)
      (blockwise128Body ops gk tileM tileN stepsK hasBias) junk m0 = some m₁)
    (h₂ : pallasCall
      (blockwise128GridSpec gm gn gk tileM tileN stepsK hasBias aBuf bBuf saBuf sbBuf biasBuf oBuf)
      (blockwise128Body ops gk tileM tileN stepsK hasBias) junk' m0' = some m₂)
    (r : Fin (gm * tileM)) (c : Fin (gn * tileN))
    (har : ∀ k < gk * (stepsK * 128), m0.read aBuf (r.val * (gk * (stepsK * 128)) + k) =
      m0'.read aBuf (r.val * (gk * (stepsK * 128)) + k))
    (hbc : ∀ k < gk * (stepsK * 128), m0.read bBuf (k * (gn * tileN) + c.val) =
      m0'.read bBuf (k * (gn * tileN) + c.val))
    (hsar : ∀ j < gk * stepsK, m0.read saBuf (j * (gm * tileM) + r.val) =
      m0'.read saBuf (j * (gm * tileM) + r.val))
    (hsbc : ∀ j < gk * stepsK, m0.read sbBuf (j * (gn * tileN) + c.val) =
      m0'.read sbBuf (j * (gn * tileN) + c.val))
    (hbiasc : hasBias = true → m0.read biasBuf c.val = m0'.read biasBuf c.val) :
    m₁.read oBuf (r.val * (gn * tileN) + c.val) = m₂.read oBuf (r.val * (gn * tileN) + c.val) := by
  obtain ⟨n₁, e₁, v₁, -⟩ := blockwise128ScaledMm_correct ops gm gn gk tileM tileN stepsK hasBias
    aBuf bBuf saBuf sbBuf biasBuf oBuf hgm hgn hgk htm htn hsk ha hb hsa hsb hbias junk m0
  obtain ⟨n₂, e₂, v₂, -⟩ := blockwise128ScaledMm_correct ops gm gn gk tileM tileN stepsK hasBias
    aBuf bBuf saBuf sbBuf biasBuf oBuf hgm hgn hgk htm htn hsk ha hb hsa hsb hbias junk' m0'
  rw [e₁, Option.some.injEq] at h₁
  rw [e₂, Option.some.injEq] at h₂
  subst h₁ h₂
  rw [v₁, v₂, blockwise128Elem, blockwise128Elem,
    accElem_congr ops gk tileM tileN stepsK 128 har hbc hsar hsbc]
  by_cases hB : hasBias = true
  · rw [hbiasc hB]
  · simp only [hB, Bool.false_eq_true, ↓reduceIte]

/-- With exact arithmetic K2's element is the block-scaled matmul over all `gk * stepsK` 128-wide
K blocks, plus `bias[c]` when the bias input is present. -/
theorem blockwise128ScaledMmElem_exact (hasBias : Bool) (ar bc sa sb : ℕ → ℝ) (bias : ℝ) :
    blockwise128Elem ScaledMmOps.exact gk tileM tileN stepsK hasBias ar bc sa sb bias =
      (∑ j ∈ Finset.range (gk * stepsK),
        sa j * sb j * ∑ k ∈ Finset.range 128, ar (j * 128 + k) * bc (j * 128 + k)) +
        (if hasBias then bias else 0) := by
  cases hasBias
  · show accElem ScaledMmOps.exact gk tileM tileN stepsK 128 ar bc sa sb = _
    rw [accElem_exact]
    simp
  · show accElem ScaledMmOps.exact gk tileM tileN stepsK 128 ar bc sa sb + bias = _
    rw [accElem_exact]
    simp

end TpuKernelLean.Examples.BlockScaledMm
