import TpuKernelLean.Pallas.ReductionKernel

/-!
Verification of a grouped row-scaled FP8 matmul Pallas kernel, for the layout with `b` stored
`[G, K, N]`.

Group `g` owns rows `[go g, go (g + 1))` of `a`, where `go` is the group-offset table, and computes
`out[r, c] = scale_a[r] * scale_b[g, c] * ∑ k, a[r, k] * b[g, k, c]`; the virtual group `G` owns
the rows past the last real group's end and writes zeros. The grid is
`(N tiles, visits, K tiles)`. A visit is one (group, M tile) pair, read from scalar-prefetched
tables. Visits to one M tile are consecutive, so the output block index stays the same across
them and the pipeline keeps the VMEM block (`PAL-OUT-PERSIST`): each visit writes its own group's
rows and passes the other rows of the block through. This kernel therefore reads its previous
output block.

The theorems are stated on the arrays the `pallas_call` sees, after the caller's padding and
reshapes: `a [PM, PK]`, `b [G, PK, PN]`, `scale_a [1, PM]`, `scale_b [G, 1, PN]`, output
`[PM, PN]`, with `PM = tilesM * tileM`, `PK = tilesK * tileK`, `PN = gn * tileN`. The padding, the
final slice and the construction of the visit schedule are outside the `pallas_call` and are not
modelled: the visit tables are constrained by `ValidSchedule` instead.

- `rowwiseScaledGroupedMm_correct`: under `Requires`, the call is defined; every element
  `(r, c)` with `r` a row of real group `g` is `groupedElem` of row `r` of `a`, column `c` of
  `b[g]`, `scale_a[r]` and `scale_b[g, c]`; every element of a row of the virtual group is
  `cast 0`; and nothing outside the output buffer changes. `ValidSchedule.exists_owner` shows every
  row `r < M` has such a group. No claim is made for the padding rows `r ≥ M`.
- `groupedElem_exact`: with exact arithmetic the element is
  `scale_a[r] * scale_b[g, c] * ∑ k, a[r, k] * b[g, k, c]`.

The float operations are opaque (`GroupedMmOps`), so the first theorem holds for every
implementation of them that satisfies `cast_up_cast`.
-/

namespace TpuKernelLean.Examples.GroupedScaledMm

open TpuKernelLean.Pallas

/-- Opaque TPU float operations of the kernel body.

Hardware assumptions:
- Each op is a deterministic function of its arguments, lowered the same way at every element of
  the tile and at every grid step.
- For a fixed compiled tile shape, `dot tileM tileN tileK` (the MXU's fp8-in, fp32-accumulated
  result for one output element of the `tileK`-wide `dot_general`) depends only on the element's
  `tileK`-vector of the `a` block and `tileK`-vector of the `b` block. Nothing is assumed across
  tile shapes.
- `cast_up_cast`: reading back an output element that was already stored (a cast of the output
  block to `float32`) and storing it again (a cast to the output dtype) does not change it. A row
  written by one visit goes through this round trip on every later visit to the same M tile. It
  holds for `bfloat16 → float32 → bfloat16` (the widening is exact) and trivially for a `float32`
  output. -/
structure GroupedMmOps where
  /-- One output element of the `tileK`-wide `dot_general` at compiled tile shape
  `(tileM, tileN)`. -/
  dot : (tileM tileN tileK : ℕ) → (Fin tileK → ℝ) → (Fin tileK → ℝ) → ℝ
  /-- fp32 accumulator plus partial product. -/
  add : ℝ → ℝ → ℝ
  /-- `acc * sa * sb` as one op, arguments `acc sa sb`. -/
  scale : ℝ → ℝ → ℝ → ℝ
  /-- Cast of an fp32 tile to the output dtype. -/
  cast : ℝ → ℝ
  /-- Cast of the output block back to `float32`. -/
  up : ℝ → ℝ
  /-- Storing an already-stored output element again does not change it. -/
  cast_up_cast : ∀ x, cast (up (cast x)) = cast x

noncomputable section

/-- Exact real arithmetic, for functional correctness. -/
def GroupedMmOps.exact : GroupedMmOps where
  dot := fun _ _ _ a b => ∑ u, a u * b u
  add := (· + ·)
  scale := fun acc sa sb => acc * sa * sb
  cast := id
  up := id
  cast_up_cast := fun _ => rfl

/-! ### Element spec -/

section Elem

variable (ops : GroupedMmOps) (tilesK tileM tileN tileK : ℕ)

/-- The partial product of K tile `t` for one output element: the `tileK`-wide dot product of
columns `t * tileK ..< (t + 1) * tileK` of the element's row `ar` of `a` and column `bc` of
`b[g]`. -/
def dotTile (t : ℕ) (ar bc : ℕ → ℝ) : ℝ :=
  ops.dot tileM tileN tileK (fun u => ar (t * tileK + u.val)) (fun u => bc (t * tileK + u.val))

/-- The accumulator after K tile `t`, from its value `s` before: the partial product at `t = 0`,
else the accumulator plus the partial product. -/
def accStep (ar bc : ℕ → ℝ) (s : ℝ) (t : ℕ) : ℝ :=
  if t = 0 then dotTile ops tileM tileN tileK t ar bc
  else ops.add s (dotTile ops tileM tileN tileK t ar bc)

/-- The accumulator after the first `n` K tiles: the in-order fold of `accStep`. -/
def accFold (ar bc : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (accStep ops tileM tileN tileK ar bc) 0

/-- The contract for one output element of a real group: `cast (scale acc sa sb)` with `acc` the
accumulator after all `tilesK` K tiles. `ar` is row `r` of `a`, `bc` column `c` of `b[g]`
(both of length `tilesK * tileK`), `sa = scale_a[r]`, `sb = scale_b[g, c]`. -/
def groupedElem (ar bc : ℕ → ℝ) (sa sb : ℝ) : ℝ :=
  ops.cast (ops.scale (accFold ops tileM tileN tileK ar bc tilesK) sa sb)

end Elem

private theorem accFold_exact (tileM tileN tileK : ℕ) (ar bc : ℕ → ℝ) (n : ℕ) :
    accFold GroupedMmOps.exact tileM tileN tileK ar bc n =
      ∑ k ∈ Finset.range (n * tileK), ar k * bc k := by
  induction n with
  | zero => simp [accFold]
  | succ n ih =>
    have hdot : dotTile GroupedMmOps.exact tileM tileN tileK n ar bc =
        ∑ x ∈ Finset.range tileK, ar (n * tileK + x) * bc (n * tileK + x) :=
      Fin.sum_univ_eq_sum_range (fun u => ar (n * tileK + u) * bc (n * tileK + u)) tileK
    unfold accFold at ih ⊢
    rw [List.range_succ, List.foldl_append, ih, List.foldl_cons, List.foldl_nil, accStep, hdot,
      Nat.succ_mul, Finset.sum_range_add]
    split_ifs with h
    · subst h
      simp
    · rfl

/-- With exact arithmetic the element is `sa * sb * ∑ k < tilesK * tileK, a[r, k] * b[g, k, c]`. -/
theorem groupedElem_exact (tilesK tileM tileN tileK : ℕ) (ar bc : ℕ → ℝ) (sa sb : ℝ) :
    groupedElem GroupedMmOps.exact tilesK tileM tileN tileK ar bc sa sb =
      sa * sb * ∑ k ∈ Finset.range (tilesK * tileK), ar k * bc k := by
  show accFold GroupedMmOps.exact tileM tileN tileK ar bc tilesK * sa * sb = _
  rw [accFold_exact]
  ring

/-! ### The visit schedule -/

/-- What the proof needs from the visit schedule, whose construction is not modelled: `go` is the
group-offset table (`G + 2` entries), `gid` the group of each visit and `mt` the M tile of each
visit (`V` entries each), for `G` real groups, `M` rows, `tilesM` M tiles of `tileM` rows and `V`
visits. Group `g ≤ G` owns rows `[go g, go (g + 1))`; `g = G` is the virtual group. -/
structure ValidSchedule (go gid mt : ℕ → ℕ) (G tilesM tileM M V : ℕ) : Prop where
  /-- There is at least one real group. -/
  groups_pos : 1 ≤ G
  /-- The first group starts at row `0`. -/
  go_zero : go 0 = 0
  /-- The group offsets are non-decreasing. -/
  go_mono : ∀ j ≤ G + 1, ∀ i ≤ j, go i ≤ go j
  /-- The virtual group ends at the last row. -/
  go_last : go (G + 1) = M
  /-- Every visit names a real group or the virtual group. -/
  gid_le : ∀ v < V, gid v ≤ G
  /-- Every visit names an M tile. -/
  mt_lt : ∀ v < V, mt v < tilesM
  /-- Visits are ordered by M tile, so the visits to one M tile are consecutive. -/
  mt_mono : ∀ v < V, ∀ u ≤ v, mt u ≤ mt v
  /-- Every row of every group is covered by a visit of that group to the row's M tile. -/
  covered : ∀ r < M, ∀ g ≤ G, go g ≤ r → r < go (g + 1) →
    ∃ v < V, gid v = g ∧ mt v = r / tileM
  /-- The first visit belongs to a real group. A virtual-group visit leaves the scratch
  accumulator untouched, so if it came first the body's result at the first step of a parallel
  N tile would depend on the incoming scratch, which the model's `parallel_scratch_reset` rule
  forbids. This excludes the case where every real group is empty: there the first visit is the
  virtual group's. -/
  first_real : gid 0 < G

/-- Every row `r < M` belongs to a group `g ≤ G`, so `rowwiseScaledGroupedMm_correct` covers every
unpadded row. -/
theorem ValidSchedule.exists_owner {go gid mt : ℕ → ℕ} {G tilesM tileM M V : ℕ}
    (hs : ValidSchedule go gid mt G tilesM tileM M V) {r : ℕ} (hr : r < M) :
    ∃ g ≤ G, go g ≤ r ∧ r < go (g + 1) := by
  have key : ∀ k, r < go k → ∃ g < k, go g ≤ r ∧ r < go (g + 1) := by
    intro k
    induction k with
    | zero =>
      intro h
      rw [hs.go_zero] at h
      omega
    | succ k ih =>
      intro h
      by_cases hk : go k ≤ r
      · exact ⟨k, by omega, hk, h⟩
      · obtain ⟨g, hg, h1, h2⟩ := ih (by omega)
        exact ⟨g, by omega, h1, h2⟩
  obtain ⟨g, hg, h⟩ := key (G + 1) (by rw [hs.go_last]; exact hr)
  exact ⟨g, by omega, h⟩

/-- A row belongs to at most one group. -/
private theorem owner_unique {go gid mt : ℕ → ℕ} {G tilesM tileM M V : ℕ}
    (hs : ValidSchedule go gid mt G tilesM tileM M V) {g₁ g₂ r : ℕ} (h₁ : g₁ ≤ G) (h₂ : g₂ ≤ G)
    (o₁ : go g₁ ≤ r ∧ r < go (g₁ + 1)) (o₂ : go g₂ ≤ r ∧ r < go (g₂ + 1)) : g₁ = g₂ := by
  by_contra hne
  rcases Nat.lt_or_gt_of_ne hne with h | h
  · have := hs.go_mono g₂ (by omega) (g₁ + 1) (by omega)
    omega
  · have := hs.go_mono g₁ (by omega) (g₂ + 1) (by omega)
    omega

/-! ### The `pallas_call` -/

/-- Static shape parameters of the call: `G` real groups, grid `(gn, tilesM + G, tilesK)`, tile
shape `(tileM, tileN, tileK)`. -/
structure Dims where
  /-- Number of real groups (the leading dimension of `b`). -/
  G : ℕ
  /-- Number of N tiles. -/
  gn : ℕ
  /-- Number of M tiles. -/
  tilesM : ℕ
  /-- Number of K tiles. -/
  tilesK : ℕ
  /-- Rows per M tile. -/
  tileM : ℕ
  /-- Columns per N tile. -/
  tileN : ℕ
  /-- Width of a K tile. -/
  tileK : ℕ

/-- HBM buffers of the call's arguments, in argument order, and of its output. -/
structure Bufs where
  /-- The group-offset table, `[G + 2]` int32. -/
  goBuf : BufId
  /-- The group of each visit, `[V]` int32. -/
  gidBuf : BufId
  /-- The M tile of each visit, `[V]` int32. -/
  mtBuf : BufId
  /-- The number of real groups, `[1]` int32. -/
  ngBuf : BufId
  /-- First shape probe, never read. -/
  mProbeBuf : BufId
  /-- Second shape probe, never read. -/
  kProbeBuf : BufId
  /-- Third shape probe, never read. -/
  nProbeBuf : BufId
  /-- `a [PM, PK]`. -/
  aBuf : BufId
  /-- `b [G, PK, PN]`. -/
  bBuf : BufId
  /-- `scale_a [1, PM]`. -/
  saBuf : BufId
  /-- `scale_b [G, 1, PN]`. -/
  sbBuf : BufId
  /-- The output `[PM, PN]`. -/
  oBuf : BufId

/-- The scalar-prefetch table: entry `Nat.pair t i` is element `i` of scalar-prefetch argument
`t`, so `Nat.pair 0 i` is group offset `i`, `Nat.pair 1 v` is the group of visit `v`,
`Nat.pair 2 v` is the M tile of visit `v` and `Nat.pair 3 0` is the number of real groups. An
entry that does not decode as int32 is `-1`. -/
def scalarTable : List (ℕ → Option ℤ) → ℕ → ℤ := fun bufs key =>
  ((bufs.getD (Nat.unpair key).1 fun _ => none) (Nat.unpair key).2).getD (-1)

/-- `a`: block `(tileM, tileK)` at `(M tile of the visit, K tile)`. Grid points are
`[N tile, visit, K tile]`. -/
def aSpec (d : Dims) : BlockSpec :=
  { blockShape := [d.tileM, d.tileK]
    indexMap := fun g sc => [sc (Nat.pair 2 (g.getD 1 0)), ((g.getD 2 0 : ℕ) : ℤ)] }

/-- `b`: block `(None, tileK, tileN)` at `(min(group of the visit, G - 1), K tile, N tile)`, so
the virtual group reads (and ignores) the last real group's block. The `None` block dimension is
size `1`. -/
def bSpec (d : Dims) : BlockSpec :=
  { blockShape := [1, d.tileK, d.tileN]
    indexMap := fun g sc =>
      [min (sc (Nat.pair 1 (g.getD 1 0))) (sc (Nat.pair 3 0) - 1), ((g.getD 2 0 : ℕ) : ℤ),
        ((g.getD 0 0 : ℕ) : ℤ)] }

/-- `scale_a`: block `(1, tileM)` at `(0, M tile of the visit)`. -/
def saSpec (d : Dims) : BlockSpec :=
  { blockShape := [1, d.tileM]
    indexMap := fun g sc => [0, sc (Nat.pair 2 (g.getD 1 0))] }

/-- `scale_b`: block `(None, 1, tileN)` at `(min(group of the visit, G - 1), 0, N tile)`, with the
`None` block dimension as size `1`. -/
def sbSpec (d : Dims) : BlockSpec :=
  { blockShape := [1, 1, d.tileN]
    indexMap := fun g sc =>
      [min (sc (Nat.pair 1 (g.getD 1 0))) (sc (Nat.pair 3 0) - 1), 0, ((g.getD 0 0 : ℕ) : ℤ)] }

/-- Output: block `(tileM, tileN)` at `(M tile of the visit, N tile)`. -/
def oSpec (d : Dims) : BlockSpec :=
  { blockShape := [d.tileM, d.tileN]
    indexMap := fun g sc => [sc (Nat.pair 2 (g.getD 1 0)), ((g.getD 0 0 : ℕ) : ℤ)] }

/-- The output array `[PM, PN]`. -/
def oArr (d : Dims) (bf : Bufs) : ArrayDesc :=
  ⟨bf.oBuf, [d.tilesM * d.tileM, d.gn * d.tileN]⟩

/-- The output array with its block spec. -/
def oPair (d : Dims) (bf : Bufs) : ArrayDesc × BlockSpec :=
  (oArr d bf, oSpec d)

/-- The `pallas_call` on the padded arrays: grid `(N tiles, visits, K tiles)` with
`tilesM + G` visits, seven scalar-prefetch arguments, one `(tileM, tileN)` scratch accumulator and
`dimension_semantics=("parallel", "arbitrary", "arbitrary")`. -/
def gridSpec (d : Dims) (bf : Bufs) : GridSpec :=
  { grid := [d.gn, d.tilesM + d.G, d.tilesK]
    scalarPrefetchBufs :=
      [bf.goBuf, bf.gidBuf, bf.mtBuf, bf.ngBuf, bf.mProbeBuf, bf.kProbeBuf, bf.nProbeBuf]
    scalarPrefetchFn := scalarTable
    inSpecs := [(⟨bf.aBuf, [d.tilesM * d.tileM, d.tilesK * d.tileK]⟩, aSpec d),
      (⟨bf.bBuf, [d.G, d.tilesK * d.tileK, d.gn * d.tileN]⟩, bSpec d),
      (⟨bf.saBuf, [1, d.tilesM * d.tileM]⟩, saSpec d),
      (⟨bf.sbBuf, [d.G, 1, d.gn * d.tileN]⟩, sbSpec d)]
    outSpecs := [oPair d bf]
    scratch := [[d.tileM, d.tileN]]
    dimSemantics := [.parallel, .arbitrary, .arbitrary] }

/-- The partial product at local element `loc = [r, c]`: the `tileK`-wide `dot_general` of row `r`
of the `a` block and column `c` of the `b` block (`ins` lists the `a`, `b`, `scale_a` and
`scale_b` blocks). -/
def dotElem (ops : GroupedMmOps) (d : Dims) (ins : List (List ℕ → ℝ)) (loc : List ℕ) : ℝ :=
  ops.dot d.tileM d.tileN d.tileK (fun u => (ins.getD 0 fun _ => 0) [loc.getD 0 0, u.val])
    (fun u => (ins.getD 1 fun _ => 0) [0, u.val, loc.getD 1 0])

/-- The scratch accumulator after grid step `g = [N tile, visit, K tile]`, from its value
`accPrev` before: for a real group (the visit's group is below the number of real groups), the
partial product at K tile `0` and the accumulator plus the partial product at later K tiles;
unchanged for the virtual group. -/
def accNext (ops : GroupedMmOps) (d : Dims) (g : List ℕ) (sc : ℕ → ℤ)
    (ins : List (List ℕ → ℝ)) (accPrev : List ℕ → ℝ) : List ℕ → ℝ := fun loc =>
  if sc (Nat.pair 1 (g.getD 1 0)) < sc (Nat.pair 3 0) then
    if g.getD 2 0 = 0 then dotElem ops d ins loc
    else ops.add (accPrev loc) (dotElem ops d ins loc)
  else accPrev loc

/-- The fp32 tile stored at the last K tile: `acc * scale_a[row] * scale_b[column]` for a real
group, zeros for the virtual group. `acc` is the accumulator after the step. -/
def tileOut (ops : GroupedMmOps) (g : List ℕ) (sc : ℕ → ℤ)
    (ins : List (List ℕ → ℝ)) (acc : List ℕ → ℝ) : List ℕ → ℝ := fun loc =>
  if sc (Nat.pair 1 (g.getD 1 0)) < sc (Nat.pair 3 0) then
    ops.scale (acc loc) ((ins.getD 2 fun _ => 0) [0, loc.getD 0 0])
      ((ins.getD 3 fun _ => 0) [0, 0, loc.getD 1 0])
  else 0

/-- The output block after grid step `g`, from its value `outPrev` before. The block changes only
at the last K tile. There, with `[groupStart, groupEnd)` the rows of the visit's group and `row0`
the first row of the visit's M tile:
- if the whole M tile lies in the group (`groupStart ≤ row0` and `row0 + tileM ≤ groupEnd`), the
  block is the tile cast to the output dtype;
- otherwise local row `r` gets the tile's row if `groupStart ≤ row0 + r < groupEnd` and the
  previous block's row cast to `float32` if not, and the result is cast to the output dtype. -/
def outNext (ops : GroupedMmOps) (d : Dims) (g : List ℕ) (sc : ℕ → ℤ)
    (ins : List (List ℕ → ℝ)) (acc outPrev : List ℕ → ℝ) : List ℕ → ℝ := fun loc =>
  let group := sc (Nat.pair 1 (g.getD 1 0))
  let groupStart := sc (Nat.pair 0 group.toNat)
  let groupEnd := sc (Nat.pair 0 (group + 1).toNat)
  let row0 := sc (Nat.pair 2 (g.getD 1 0)) * (d.tileM : ℤ)
  if g.getD 2 0 = d.tilesK - 1 then
    if groupStart ≤ row0 ∧ row0 + (d.tileM : ℤ) ≤ groupEnd then
      ops.cast (tileOut ops g sc ins acc loc)
    else
      ops.cast (if groupStart ≤ row0 + ((loc.getD 0 0 : ℕ) : ℤ) ∧
          row0 + ((loc.getD 0 0 : ℕ) : ℤ) < groupEnd then
        tileOut ops g sc ins acc loc
      else ops.up (outPrev loc))
  else outPrev loc

/-- The kernel body. `prev` is the output block and the scratch accumulator before the step; the
result is the output block and the scratch accumulator after it. -/
def body (ops : GroupedMmOps) (d : Dims) : Body := fun g sc ins prev =>
  let acc := accNext ops d g sc ins (prev.getD 1 fun _ => 0)
  [outNext ops d g sc ins acc (prev.getD 0 fun _ => 0), acc]

/-- Hypotheses of `rowwiseScaledGroupedMm_correct` on the initial memory `m0`: the
scalar-prefetch buffers hold the tables `go`, `gid`, `mt` and the number of real groups `G` as
int32, the tables are a `ValidSchedule` for `M` rows and `V = tilesM + G` visits, the grid and
tile sizes are positive, and the output buffer is distinct from every argument buffer. -/
structure Requires (d : Dims) (bf : Bufs) (go gid mt : ℕ → ℕ) (M : ℕ) (m0 : Mem) : Prop where
  /-- The visit tables are a valid schedule. -/
  sched : ValidSchedule go gid mt d.G d.tilesM d.tileM M (d.tilesM + d.G)
  /-- At least one N tile. -/
  gn_pos : 0 < d.gn
  /-- At least one K tile. -/
  tilesK_pos : 0 < d.tilesK
  /-- M tiles are non-empty. -/
  tileM_pos : 0 < d.tileM
  /-- N tiles are non-empty. -/
  tileN_pos : 0 < d.tileN
  /-- K tiles are non-empty. -/
  tileK_pos : 0 < d.tileK
  /-- `goBuf` holds the group offsets. -/
  read_go : ∀ i ≤ d.G + 1, readInt32 m0 bf.goBuf i = some (go i : ℤ)
  /-- `gidBuf` holds the group of each visit. -/
  read_gid : ∀ v < d.tilesM + d.G, readInt32 m0 bf.gidBuf v = some (gid v : ℤ)
  /-- `mtBuf` holds the M tile of each visit. -/
  read_mt : ∀ v < d.tilesM + d.G, readInt32 m0 bf.mtBuf v = some (mt v : ℤ)
  /-- `ngBuf` holds the number of real groups. -/
  read_ng : readInt32 m0 bf.ngBuf 0 = some (d.G : ℤ)
  /-- The output buffer is none of the scalar-prefetch or input buffers. -/
  out_ne : ∀ b ∈ [bf.goBuf, bf.gidBuf, bf.mtBuf, bf.ngBuf, bf.mProbeBuf, bf.kProbeBuf,
    bf.nProbeBuf, bf.aBuf, bf.bBuf, bf.saBuf, bf.sbBuf], bf.oBuf ≠ b

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

private theorem div_tile {i t l : ℕ} (hl : l < t) : (i * t + l) / t = i := by
  rw [Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hl, Nat.add_zero]

private theorem tile_key_inj {t i i' a a' : ℕ} (ha : a < t) (ha' : a' < t)
    (h : i * t + a = i' * t + a') : i = i' ∧ a = a' := by
  have h1 := div_tile (i := i) ha
  have h2 := div_tile (i := i') ha'
  rw [h, h2] at h1
  subst h1
  exact ⟨rfl, by omega⟩

private theorem mem_grid3 {a b c : ℕ} {g : List ℕ} :
    g ∈ gridPoints [a, b, c] ↔ ∃ i j k, g = [i, j, k] ∧ i < a ∧ j < b ∧ k < c := by
  rw [mem_gridPoints]
  rcases g with _ | ⟨i, _ | ⟨j, _ | ⟨k, _ | ⟨u, g⟩⟩⟩⟩ <;> simp [inBounds]

private theorem gridPoints_three (a b c : ℕ) :
    gridPoints [a, b, c] =
      (List.range a).flatMap fun i => (List.range b).flatMap fun j =>
        (List.range c).map fun k => [i, j, k] := by
  have h1 := gridPoints_append_singleton [a, b] c
  have h2 := gridPoints_append_singleton [a] b
  simp only [List.cons_append, List.nil_append] at h1 h2
  rw [h1, h2, gridPoints_one]
  simp only [List.flatMap_map, List.flatMap_assoc, List.cons_append, List.nil_append]

/-- A relation that holds between lexicographically ordered grid points holds pairwise along the
row-major grid order. -/
private theorem pairwise_grid3 {a b c : ℕ} {R : List ℕ → List ℕ → Prop}
    (h : ∀ i j k i' j' k', i < a → j < b → k < c → i' < a → j' < b → k' < c →
      (i < i' ∨ (i = i' ∧ (j < j' ∨ (j = j' ∧ k < k')))) → R [i, j, k] [i', j', k']) :
    (gridPoints [a, b, c]).Pairwise R := by
  rw [gridPoints_three, List.pairwise_flatMap]
  refine ⟨fun i hi => ?_, ?_⟩
  · rw [List.mem_range] at hi
    rw [List.pairwise_flatMap]
    refine ⟨fun j hj => ?_, ?_⟩
    · rw [List.mem_range] at hj
      rw [List.pairwise_map]
      refine (List.pairwise_lt_range (n := c)).imp_of_mem ?_
      intro k k' hk hk' hlt
      rw [List.mem_range] at hk hk'
      exact h i j k i j k' hi hj hk hi hj hk' (Or.inr ⟨rfl, Or.inr ⟨rfl, hlt⟩⟩)
    · refine (List.pairwise_lt_range (n := b)).imp_of_mem ?_
      intro j j' hj hj' hlt x hx y hy
      rw [List.mem_range] at hj hj'
      simp only [List.mem_map, List.mem_range] at hx hy
      obtain ⟨k, hk, rfl⟩ := hx
      obtain ⟨k', hk', rfl⟩ := hy
      exact h i j k i j' k' hi hj hk hi hj' hk' (Or.inr ⟨rfl, Or.inl hlt⟩)
  · refine (List.pairwise_lt_range (n := a)).imp_of_mem ?_
    intro i i' hi hi' hlt x hx y hy
    rw [List.mem_range] at hi hi'
    simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hx hy
    obtain ⟨j, hj, k, hk, rfl⟩ := hx
    obtain ⟨j', hj', k', hk', rfl⟩ := hy
    exact h i j k i' j' k' hi hj hk hi' hj' hk' (Or.inl hlt)

private theorem not_mem_dropWhile_of_sorted {α : Type*} [DecidableEq α] (key : α → ℕ) (x : α) :
    ∀ (xs : List α), (∀ y ∈ xs, key x ≤ key y) → (∀ y ∈ xs, key y = key x → y = x) →
      xs.Pairwise (fun a b => key a ≤ key b) → x ∉ xs.dropWhile fun y => decide (y = x)
  | [], _, _, _ => by simp
  | y :: ys, hle, hinj, hsort => by
    rw [List.pairwise_cons] at hsort
    by_cases hy : y = x
    · rw [List.dropWhile_cons_of_pos (by simpa using hy)]
      exact not_mem_dropWhile_of_sorted key x ys (fun z hz => hle z (List.mem_cons_of_mem _ hz))
        (fun z hz => hinj z (List.mem_cons_of_mem _ hz)) hsort.2
    · rw [List.dropWhile_cons_of_neg (by simpa using hy)]
      intro hmem
      rcases List.mem_cons.mp hmem with h | h
      · exact hy h.symm
      · exact hy (hinj y (by simp) (Nat.le_antisymm (hsort.1 x h) (hle y (by simp))))

/-- A list sorted by a key that is injective on its elements never revisits an element. -/
private theorem noRevisit_of_sorted {α : Type*} [DecidableEq α] (key : α → ℕ) :
    ∀ (l : List α), (∀ a ∈ l, ∀ b ∈ l, key a = key b → a = b) →
      l.Pairwise (fun a b => key a ≤ key b) → noRevisit l = true
  | [], _, _ => rfl
  | x :: xs, hinj, hsort => by
    rw [List.pairwise_cons] at hsort
    simp only [noRevisit, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨not_mem_dropWhile_of_sorted key x xs hsort.1
        (fun y hy h => hinj y (List.mem_cons_of_mem _ hy) x (by simp) h) hsort.2,
      noRevisit_of_sorted key xs
        (fun a ha b hb => hinj a (List.mem_cons_of_mem _ ha) b (List.mem_cons_of_mem _ hb))
        hsort.2⟩

/-- Row `r` of `a`. -/
private def rowA (d : Dims) (bf : Bufs) (m0 : Mem) (r : ℕ) : ℕ → ℝ := fun k =>
  m0.read bf.aBuf (r * (d.tilesK * d.tileK) + k)

/-- Column `c` of `b[g]`. -/
private def colB (d : Dims) (bf : Bufs) (m0 : Mem) (g c : ℕ) : ℕ → ℝ := fun k =>
  m0.read bf.bBuf (g * (d.tilesK * d.tileK * (d.gn * d.tileN)) + (k * (d.gn * d.tileN) + c))

/-- The specified output element `(r, c)` when row `r` belongs to group `g`. -/
private def rowVal (ops : GroupedMmOps) (d : Dims) (bf : Bufs) (m0 : Mem) (g r c : ℕ) : ℝ :=
  if g < d.G then
    groupedElem ops d.tilesK d.tileM d.tileN d.tileK (rowA d bf m0 r) (colB d bf m0 g c)
      (m0.read bf.saBuf r) (m0.read bf.sbBuf (g * (d.gn * d.tileN) + c))
  else ops.cast 0

private theorem rowVal_fix (ops : GroupedMmOps) (d : Dims) (bf : Bufs) (m0 : Mem) (g r c : ℕ) :
    ops.cast (ops.up (rowVal ops d bf m0 g r c)) = rowVal ops d bf m0 g r c := by
  unfold rowVal
  split_ifs
  · exact ops.cast_up_cast _
  · exact ops.cast_up_cast _

/-! ### The run is well defined -/

section Proof

variable {d : Dims} {bf : Bufs} {go gid mt : ℕ → ℕ} {M : ℕ} {m0 : Mem}

private theorem sc_go (hr : Requires d bf go gid mt M m0) {i : ℕ} (hi : i ≤ d.G + 1) :
    (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 0 i) = (go i : ℤ) := by
  simp [GridSpec.scalarPrefetch, gridSpec, scalarTable, hr.read_go i hi]

private theorem sc_gid (hr : Requires d bf go gid mt M m0) {v : ℕ} (hv : v < d.tilesM + d.G) :
    (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 1 v) = (gid v : ℤ) := by
  simp [GridSpec.scalarPrefetch, gridSpec, scalarTable, hr.read_gid v hv]

private theorem sc_mt (hr : Requires d bf go gid mt M m0) {v : ℕ} (hv : v < d.tilesM + d.G) :
    (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 2 v) = (mt v : ℤ) := by
  simp [GridSpec.scalarPrefetch, gridSpec, scalarTable, hr.read_mt v hv]

private theorem sc_ng (hr : Requires d bf go gid mt M m0) :
    (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 3 0) = (d.G : ℤ) := by
  simp [GridSpec.scalarPrefetch, gridSpec, scalarTable, hr.read_ng]

/-- The group whose `b` and `scale_b` blocks a visit loads is a real group, and the visit's own
group when that is real. -/
private theorem weightGroup_eq (hr : Requires d bf go gid mt M m0) {v : ℕ}
    (hv : v < d.tilesM + d.G) :
    ∃ w, w < d.G ∧ (gid v < d.G → w = gid v) ∧
      min ((GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 1 v))
        ((GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 3 0) - 1) = (w : ℤ) := by
  have hG := hr.sched.groups_pos
  by_cases h : gid v < d.G
  · exact ⟨gid v, h, fun _ => rfl, by rw [sc_gid hr hv, sc_ng hr]; exact min_eq_left (by omega)⟩
  · refine ⟨d.G - 1, by omega, fun h' => absurd h' h, ?_⟩
    rw [sc_gid hr hv, sc_ng hr, min_eq_right (by omega)]
    omega

private theorem ins_ne (hr : Requires d bf go gid mt M m0) :
    ∀ p ∈ (gridSpec d bf).inSpecs, p.1.buf ≠ bf.oBuf := by
  intro p hp
  simp only [gridSpec, List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl
  · exact (hr.out_ne bf.aBuf (by simp)).symm
  · exact (hr.out_ne bf.bBuf (by simp)).symm
  · exact (hr.out_ne bf.saBuf (by simp)).symm
  · exact (hr.out_ne bf.sbBuf (by simp)).symm

private theorem blockOk_at (hr : Requires d bf go gid mt M m0) {ni v ki : ℕ} (hni : ni < d.gn)
    (hv : v < d.tilesM + d.G) (hki : ki < d.tilesK) (p : ArrayDesc × BlockSpec)
    (hp : p ∈ (gridSpec d bf).inSpecs ++ (gridSpec d bf).outSpecs) :
    blockOk p.1.shape p.2.blockShape
      (p.2.indexMap [ni, v, ki] (GridSpec.scalarPrefetch (gridSpec d bf) m0)) = true := by
  have hmt := hr.sched.mt_lt v hv
  obtain ⟨w, hw, -, hwg⟩ := weightGroup_eq hr hv
  simp only [gridSpec, List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil,
    or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl
  · show blockOk [d.tilesM * d.tileM, d.tilesK * d.tileK] [d.tileM, d.tileK]
      [(GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 2 v), ((ki : ℕ) : ℤ)] = true
    rw [sc_mt hr hv]
    exact blockOk_cons hr.tileM_pos (mul_tile_lt hmt hr.tileM_pos)
      (blockOk_cons hr.tileK_pos (mul_tile_lt hki hr.tileK_pos) blockOk_nil)
  · show blockOk [d.G, d.tilesK * d.tileK, d.gn * d.tileN] [1, d.tileK, d.tileN]
      [min ((GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 1 v))
          ((GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 3 0) - 1),
        ((ki : ℕ) : ℤ), ((ni : ℕ) : ℤ)] = true
    rw [hwg]
    exact blockOk_cons Nat.one_pos (by omega)
      (blockOk_cons hr.tileK_pos (mul_tile_lt hki hr.tileK_pos)
        (blockOk_cons hr.tileN_pos (mul_tile_lt hni hr.tileN_pos) blockOk_nil))
  · show blockOk [1, d.tilesM * d.tileM] [1, d.tileM]
      [((0 : ℕ) : ℤ), (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 2 v)] = true
    rw [sc_mt hr hv]
    exact blockOk_cons Nat.one_pos (by decide)
      (blockOk_cons hr.tileM_pos (mul_tile_lt hmt hr.tileM_pos) blockOk_nil)
  · show blockOk [d.G, 1, d.gn * d.tileN] [1, 1, d.tileN]
      [min ((GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 1 v))
          ((GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 3 0) - 1),
        ((0 : ℕ) : ℤ), ((ni : ℕ) : ℤ)] = true
    rw [hwg]
    exact blockOk_cons Nat.one_pos (by omega)
      (blockOk_cons Nat.one_pos (by decide)
        (blockOk_cons hr.tileN_pos (mul_tile_lt hni hr.tileN_pos) blockOk_nil))
  · show blockOk [d.tilesM * d.tileM, d.gn * d.tileN] [d.tileM, d.tileN]
      [(GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 2 v), ((ni : ℕ) : ℤ)] = true
    rw [sc_mt hr hv]
    exact blockOk_cons hr.tileM_pos (mul_tile_lt hmt hr.tileM_pos)
      (blockOk_cons hr.tileN_pos (mul_tile_lt hni hr.tileN_pos) blockOk_nil)

/-- The output block index `(mt v, ni)` is never revisited: it is sorted by
`ni * tilesM + mt v` along the grid, because `mt` is monotone. -/
private theorem out_noRevisit (hr : Requires d bf go gid mt M m0) :
    noRevisit ((gridPoints [d.gn, d.tilesM + d.G, d.tilesK]).map
      fun g => (oSpec d).indexMap g (GridSpec.scalarPrefetch (gridSpec d bf) m0)) = true := by
  refine noRevisit_of_sorted
    (fun z : List ℤ => (z.getD 1 0).toNat * d.tilesM + (z.getD 0 0).toNat) _ ?_ ?_
  · intro x hx y hy hxy
    simp only [List.mem_map] at hx hy
    obtain ⟨g, hg, rfl⟩ := hx
    obtain ⟨g', hg', rfl⟩ := hy
    obtain ⟨ni, v, k, rfl, hni, hv, hk⟩ := mem_grid3.mp hg
    obtain ⟨ni', v', k', rfl, hni', hv', hk'⟩ := mem_grid3.mp hg'
    simp only [oSpec, List.getD_cons_zero, List.getD_cons_succ, sc_mt hr hv, sc_mt hr hv',
      Int.toNat_natCast] at hxy ⊢
    obtain ⟨h1, h2⟩ := tile_key_inj (hr.sched.mt_lt v hv) (hr.sched.mt_lt v' hv') hxy
    rw [h1, h2]
  · rw [List.pairwise_map]
    apply pairwise_grid3
    intro ni v k ni' v' k' hni hv hk hni' hv' hk' hlex
    simp only [oSpec, List.getD_cons_zero, List.getD_cons_succ, sc_mt hr hv, sc_mt hr hv',
      Int.toNat_natCast]
    have hm := hr.sched.mt_lt v hv
    rcases hlex with h | ⟨rfl, h | ⟨rfl, _⟩⟩
    · have h' := Nat.mul_le_mul_right d.tilesM h
      rw [Nat.succ_mul] at h'
      omega
    · have := hr.sched.mt_mono v' hv' v (by omega)
      omega
    · exact le_rfl

/-- At the first visit, which belongs to a real group, the accumulator is overwritten at K tile
`0`. -/
private theorem accNext_reset (ops : GroupedMmOps) (hr : Requires d bf go gid mt M m0) (ni : ℕ)
    (ins : List (List ℕ → ℝ)) (a₁ a₂ : List ℕ → ℝ) :
    accNext ops d [ni, 0, 0] (GridSpec.scalarPrefetch (gridSpec d bf) m0) ins a₁ =
      accNext ops d [ni, 0, 0] (GridSpec.scalarPrefetch (gridSpec d bf) m0) ins a₂ := by
  have hV : 0 < d.tilesM + d.G := by
    have := hr.sched.groups_pos
    omega
  have hreal : (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 1 0) <
      (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 3 0) := by
    rw [sc_gid hr hV, sc_ng hr]
    exact_mod_cast hr.sched.first_real
  funext loc
  simp [accNext, hreal]

/-- Under `Requires` the call is a defined `pallas_call` run. -/
theorem gridSpec_wellDefined (ops : GroupedMmOps) (hr : Requires d bf go gid mt M m0) :
    (gridSpec d bf).WellDefined (body ops d) m0 := by
  have hG := hr.sched.groups_pos
  have hpar : (gridSpec d bf).parGrid = [d.gn] := rfl
  have harb : (gridSpec d bf).arbGrid = [d.tilesM + d.G, d.tilesK] := rfl
  constructor
  · intro x hx
    simp only [gridSpec, List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl
    · exact hr.gn_pos
    · omega
    · exact hr.tilesK_pos
  · intro p hp
    simp only [gridSpec, List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil,
      or_false] at hp
    rcases hp with rfl | rfl | rfl | rfl | rfl <;>
      simp [aSpec, bSpec, saSpec, sbSpec, oSpec, oPair, oArr, hr.tileM_pos, hr.tileN_pos,
        hr.tileK_pos]
  · intro k hk p hp
    have hmem : (gridPoints (gridSpec d bf).grid).getD k [] ∈
        gridPoints [d.gn, d.tilesM + d.G, d.tilesK] := by
      rw [List.getD_eq_getElem _ _ hk]
      exact List.getElem_mem hk
    obtain ⟨ni, v, ki, hg, hni, hv, hki⟩ := mem_grid3.mp hmem
    simp only [GridSpec.idxAt, hg]
    exact blockOk_at hr hni hv hki p hp
  · intro p hp
    simp only [gridSpec, List.mem_singleton] at hp
    subst hp
    exact (congrArg noRevisit (range_map_getD (gridPoints (gridSpec d bf).grid) []
      (fun g => (oSpec d).indexMap g ((gridSpec d bf).scalarPrefetch m0)))).trans
      (out_noRevisit hr)
  · intro _ _ _ _
    rfl
  · intro s t _ _ i hi o ho hov
    simp only [gridSpec, List.mem_singleton] at ho
    subst ho
    exact ins_ne hr i hi hov.1
  · refine GridSpec.AliasesValid.of_empty rfl (by simp [gridSpec]) ?_ ?_
    · intro inp hinp outp houtp
      simp only [gridSpec, List.mem_singleton] at houtp
      subst houtp
      exact ins_ne hr inp hinp
    · intro b hb outp houtp
      simp only [gridSpec, List.mem_singleton] at houtp
      subst houtp
      refine hr.out_ne b ?_
      simp only [gridSpec, List.mem_cons, List.not_mem_nil, or_false] at hb ⊢
      tauto
  · rfl
  · rfl
  · intro pg₁ hpg₁ pg₂ hpg₂ hne ag₁ _ ag₂ _ o₁ ho₁ o₂ ho₂ hov
    simp only [gridSpec, List.mem_singleton] at ho₁ ho₂
    subst ho₁ ho₂
    rw [hpar, gridPoints_one] at hpg₁ hpg₂
    simp only [List.mem_map, List.mem_range] at hpg₁ hpg₂
    obtain ⟨n₁, _, rfl⟩ := hpg₁
    obtain ⟨n₂, _, rfl⟩ := hpg₂
    obtain ⟨_, a, ha₁, ha₂⟩ := hov
    have hbi := addrInBlock_unique ha₁ ha₂
    simp only [oPair, oSpec, BlockSpec.blockIdx, List.cons_append, List.nil_append,
      List.getD_cons_zero, List.map_cons, List.map_nil, Int.toNat_natCast, List.cons.injEq,
      and_true] at hbi
    exact hne (by rw [hbi.2])
  · intro _ _ _ _ _ _ _ _ _ i hi o ho hov
    simp only [gridSpec, List.mem_singleton] at ho
    subst ho
    exact ins_ne hr i hi hov.1
  · intro _ pg hpg ag₀ hag₀ ins outs sc₁ sc₂ houts _ _
    rw [hpar, gridPoints_one] at hpg
    simp only [List.mem_map, List.mem_range] at hpg
    obtain ⟨ni, _, rfl⟩ := hpg
    have hag := ReductionKernel.head_gridPoints_eq_replicate_zero (gridSpec d bf).arbGrid ag₀ hag₀
    subst hag
    have houts' : outs.length = 1 := houts
    obtain ⟨o0, rfl⟩ := List.length_eq_one_iff.mp houts'
    show body ops d [ni, 0, 0] _ ins (o0 :: sc₁) = body ops d [ni, 0, 0] _ ins (o0 :: sc₂)
    simp only [body, List.getD_cons_zero, List.getD_cons_succ]
    rw [accNext_reset ops hr ni ins (sc₁.getD 0 fun _ => 0) (sc₂.getD 0 fun _ => 0)]

/-! ### One grid step -/

/-- The input blocks `stepGrid` loads at grid point `g` from memory `m`. -/
private def rawIns (d : Dims) (bf : Bufs) (sc : ℕ → ℤ) (junk : ℕ → ℝ) (m : Mem) (g : List ℕ) :
    List (List ℕ → ℝ) :=
  (gridSpec d bf).inSpecs.mapIdx fun i p =>
    loadBlock p.1 p.2.blockShape (p.2.blockIdx g sc)
      (junkAt junk (Nat.pair (flatCoord (gridSpec d bf).grid g + 1) (i + 1))) m

/-- One grid step of the model on the two-slot state `[out, acc]`: the previous output block `P`
is the slot's block when the block index repeats, and the previous scratch `S` is the scratch
slot's value, both inside the tile. -/
private theorem step_raw (ops : GroupedMmOps) (d : Dims) (bf : Bufs) (sc : ℕ → ℤ) (junk : ℕ → ℝ)
    (m : Mem) (s0 s1 : OutSlot) (g : List ℕ) :
    ∃ P S : List ℕ → ℝ,
      stepGrid (gridSpec d bf) (body ops d) sc junk (m, [s0, s1]) g =
        (flushOne (oPair d bf) ((oSpec d).blockIdx g sc) s0 m,
          [some ((oSpec d).blockIdx g sc,
              outNext ops d g sc
                (rawIns d bf sc junk (flushOne (oPair d bf) ((oSpec d).blockIdx g sc) s0 m) g)
                (accNext ops d g sc
                  (rawIns d bf sc junk (flushOne (oPair d bf) ((oSpec d).blockIdx g sc) s0 m) g)
                  S) P),
            some ([], accNext ops d g sc
              (rawIns d bf sc junk (flushOne (oPair d bf) ((oSpec d).blockIdx g sc) s0 m) g)
              S)]) ∧
      (∀ O0, s0 = some ((oSpec d).blockIdx g sc, O0) →
        ∀ loc, inBounds [d.tileM, d.tileN] loc = true → P loc = O0 loc) ∧
      (∀ i A0, s1 = some (i, A0) →
        ∀ loc, inBounds [d.tileM, d.tileN] loc = true → S loc = A0 loc) := by
  rcases s0 with _ | ⟨oi, ov⟩ <;> rcases s1 with _ | ⟨si, sv⟩
  · exact ⟨_, _, rfl, (fun _ h => by cases h), (fun _ _ h => by cases h)⟩
  · refine ⟨_, _, rfl, (fun _ h => by cases h), ?_⟩
    intro i A0 h loc hloc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    simp [flushAndPrep, gridSpec, oPair, hloc]
  · refine ⟨_, _, rfl, ?_, (fun _ _ h => by cases h)⟩
    intro O0 h loc hloc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    simp [flushAndPrep, gridSpec, oPair, oSpec, hloc]
  · refine ⟨_, _, rfl, ?_, ?_⟩
    · intro O0 h loc hloc
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      simp [flushAndPrep, gridSpec, oPair, oSpec, hloc]
    · intro i A0 h loc hloc
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      simp [flushAndPrep, gridSpec, oPair, hloc]

private theorem inb {tm tn rl cl : ℕ} (hrl : rl < tm) (hcl : cl < tn) :
    inBounds [tm, tn] [rl, cl] = true :=
  inBounds_cons hrl (inBounds_cons hcl inBounds_nil)

/-- `step_raw` at grid point `(ni, v, k)` of a defined run: the output block index is
`(mt v, ni)` and the inputs are read from the initial memory. -/
private theorem step_eq (ops : GroupedMmOps) (hr : Requires d bf go gid mt M m0) (junk : ℕ → ℝ)
    (m : Mem) (hm : ∀ b, b ≠ bf.oBuf → ∀ a, m.read b a = m0.read b a) (s0 s1 : OutSlot)
    {ni v k : ℕ} (hv : v < d.tilesM + d.G) :
    ∃ P S : List ℕ → ℝ,
      stepGrid (gridSpec d bf) (body ops d) (GridSpec.scalarPrefetch (gridSpec d bf) m0) junk
          (m, [s0, s1]) [ni, v, k] =
        (flushOne (oPair d bf) [mt v, ni] s0 m,
          [some ([mt v, ni],
              outNext ops d [ni, v, k] (GridSpec.scalarPrefetch (gridSpec d bf) m0)
                (loadIns (gridSpec d bf) junk m0 [ni, v, k])
                (accNext ops d [ni, v, k] (GridSpec.scalarPrefetch (gridSpec d bf) m0)
                  (loadIns (gridSpec d bf) junk m0 [ni, v, k]) S) P),
            some ([], accNext ops d [ni, v, k] (GridSpec.scalarPrefetch (gridSpec d bf) m0)
              (loadIns (gridSpec d bf) junk m0 [ni, v, k]) S)]) ∧
      (∀ O0, s0 = some ([mt v, ni], O0) →
        ∀ loc, inBounds [d.tileM, d.tileN] loc = true → P loc = O0 loc) ∧
      (∀ i A0, s1 = some (i, A0) →
        ∀ loc, inBounds [d.tileM, d.tileN] loc = true → S loc = A0 loc) := by
  obtain ⟨P, S, hstep, hP, hS⟩ :=
    step_raw ops d bf (GridSpec.scalarPrefetch (gridSpec d bf) m0) junk m s0 s1 [ni, v, k]
  have hidx : (oSpec d).blockIdx [ni, v, k] (GridSpec.scalarPrefetch (gridSpec d bf) m0) = [mt v, ni] := by
    simp only [oSpec, BlockSpec.blockIdx, List.getD_cons_zero, List.getD_cons_succ,
      List.map_cons, List.map_nil, Int.toNat_natCast, sc_mt hr hv]
  have hins : rawIns d bf (GridSpec.scalarPrefetch (gridSpec d bf) m0) junk
      (flushOne (oPair d bf)
        ((oSpec d).blockIdx [ni, v, k] (GridSpec.scalarPrefetch (gridSpec d bf) m0)) s0 m) [ni, v, k] =
      loadIns (gridSpec d bf) junk m0 [ni, v, k] := by
    apply loadIns_congr (gridSpec d bf) junk (GridSpec.scalarPrefetch (gridSpec d bf) m0) _ m0 [ni, v, k]
    intro p hp a
    rw [flushOne_read_other_buf (oPair d bf) _ _ _ _ a (ins_ne hr p hp)]
    exact hm _ (ins_ne hr p hp) a
  rw [hins, hidx] at hstep
  rw [hidx] at hP
  exact ⟨P, S, hstep, hP, hS⟩

/-- At grid step `(ni, v, k)`, the in-bounds elements of the VMEM input blocks are the HBM
elements of the padded arrays: row `mt v * tileM + rl` of `a` and `scale_a`, and, for a real
group, column `ni * tileN + cl` of `b[gid v]` and `scale_b[gid v]`. -/
private theorem reads (hr : Requires d bf go gid mt M m0) (junk : ℕ → ℝ) {ni v k rl cl : ℕ}
    (hni : ni < d.gn) (hv : v < d.tilesM + d.G) (hk : k < d.tilesK) (hrl : rl < d.tileM)
    (hcl : cl < d.tileN) :
    (∀ u < d.tileK, (loadIns (gridSpec d bf) junk m0 [ni, v, k]).getD 0 (fun _ => 0) [rl, u] =
      rowA d bf m0 (mt v * d.tileM + rl) (k * d.tileK + u)) ∧
    (gid v < d.G → ∀ u < d.tileK,
      (loadIns (gridSpec d bf) junk m0 [ni, v, k]).getD 1 (fun _ => 0) [0, u, cl] =
        colB d bf m0 (gid v) (ni * d.tileN + cl) (k * d.tileK + u)) ∧
    (loadIns (gridSpec d bf) junk m0 [ni, v, k]).getD 2 (fun _ => 0) [0, rl] =
      m0.read bf.saBuf (mt v * d.tileM + rl) ∧
    (gid v < d.G → (loadIns (gridSpec d bf) junk m0 [ni, v, k]).getD 3 (fun _ => 0) [0, 0, cl] =
      m0.read bf.sbBuf (gid v * (d.gn * d.tileN) + (ni * d.tileN + cl))) := by
  have hmt := hr.sched.mt_lt v hv
  have hrow : mt v * d.tileM + rl < d.tilesM * d.tileM := tile_lt hmt hrl
  have hcol : ni * d.tileN + cl < d.gn * d.tileN := tile_lt hni hcl
  have hkk : ∀ u < d.tileK, k * d.tileK + u < d.tilesK * d.tileK := fun u hu => tile_lt hk hu
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro u hu
    refine (loadIns_read _ junk m0 _ 0 ⟨bf.aBuf, [d.tilesM * d.tileM, d.tilesK * d.tileK]⟩
      (aSpec d) rfl [mt v, k] [rl, u] ?_ ?_ ?_).trans ?_
    · simp only [aSpec, BlockSpec.blockIdx, List.getD_cons_zero, List.getD_cons_succ,
        List.map_cons, List.map_nil, Int.toNat_natCast, sc_mt hr hv]
    · exact inBounds_cons hrl (inBounds_cons hu inBounds_nil)
    · exact inBounds_cons hrow (inBounds_cons (hkk u hu) inBounds_nil)
    · simp only [aSpec, rowA, globalCoord, flatCoord, List.prod_cons, List.prod_nil]
      congr 1
      ring
  · intro hreal u hu
    obtain ⟨w, hw, hwv, hwg⟩ := weightGroup_eq hr hv
    obtain rfl := hwv hreal
    refine (loadIns_read _ junk m0 _ 1 ⟨bf.bBuf, [d.G, d.tilesK * d.tileK, d.gn * d.tileN]⟩
      (bSpec d) rfl [gid v, k, ni] [0, u, cl] ?_ ?_ ?_).trans ?_
    · simp only [bSpec, BlockSpec.blockIdx, List.getD_cons_zero, List.getD_cons_succ,
        List.map_cons, List.map_nil, Int.toNat_natCast, hwg]
    · exact inBounds_cons Nat.one_pos (inBounds_cons hu (inBounds_cons hcl inBounds_nil))
    · exact inBounds_cons (by simpa using hw)
        (inBounds_cons (hkk u hu) (inBounds_cons hcol inBounds_nil))
    · simp only [bSpec, colB, globalCoord, flatCoord, List.prod_cons, List.prod_nil]
      congr 1
      ring
  · refine (loadIns_read _ junk m0 _ 2 ⟨bf.saBuf, [1, d.tilesM * d.tileM]⟩
      (saSpec d) rfl [0, mt v] [0, rl] ?_ ?_ ?_).trans ?_
    · simp only [saSpec, BlockSpec.blockIdx, List.getD_cons_zero, List.getD_cons_succ,
        List.map_cons, List.map_nil, Int.toNat_natCast, Int.toNat_zero, sc_mt hr hv]
    · exact inBounds_cons Nat.one_pos (inBounds_cons hrl inBounds_nil)
    · exact inBounds_cons (by decide) (inBounds_cons hrow inBounds_nil)
    · simp only [saSpec, globalCoord, flatCoord, List.prod_cons, List.prod_nil]
      congr 1
      ring
  · intro hreal
    obtain ⟨w, hw, hwv, hwg⟩ := weightGroup_eq hr hv
    obtain rfl := hwv hreal
    refine (loadIns_read _ junk m0 _ 3 ⟨bf.sbBuf, [d.G, 1, d.gn * d.tileN]⟩
      (sbSpec d) rfl [gid v, 0, ni] [0, 0, cl] ?_ ?_ ?_).trans ?_
    · simp only [sbSpec, BlockSpec.blockIdx, List.getD_cons_zero, List.getD_cons_succ,
        List.map_cons, List.map_nil, Int.toNat_natCast, Int.toNat_zero, hwg]
    · exact inBounds_cons Nat.one_pos (inBounds_cons Nat.one_pos (inBounds_cons hcl inBounds_nil))
    · exact inBounds_cons (by simpa using hw)
        (inBounds_cons (by decide) (inBounds_cons hcol inBounds_nil))
    · simp only [sbSpec, globalCoord, flatCoord, List.prod_cons, List.prod_nil]
      congr 1
      ring

/-- The accumulator after a step of a real group is `accStep` of the element's row and column. -/
private theorem accNext_real (ops : GroupedMmOps) (hr : Requires d bf go gid mt M m0)
    (junk : ℕ → ℝ) {ni v k rl cl : ℕ} (hni : ni < d.gn) (hv : v < d.tilesM + d.G)
    (hk : k < d.tilesK) (hrl : rl < d.tileM) (hcl : cl < d.tileN) (hreal : gid v < d.G)
    (S : List ℕ → ℝ) :
    accNext ops d [ni, v, k] (GridSpec.scalarPrefetch (gridSpec d bf) m0)
        (loadIns (gridSpec d bf) junk m0 [ni, v, k]) S [rl, cl] =
      accStep ops d.tileM d.tileN d.tileK (rowA d bf m0 (mt v * d.tileM + rl))
        (colB d bf m0 (gid v) (ni * d.tileN + cl)) (S [rl, cl]) k := by
  obtain ⟨hA, hB, -, -⟩ := reads hr junk hni hv hk hrl hcl
  have hre : (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 1 v) <
      (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 3 0) := by
    rw [sc_gid hr hv, sc_ng hr]
    exact_mod_cast hreal
  have hdot : dotElem ops d (loadIns (gridSpec d bf) junk m0 [ni, v, k]) [rl, cl] =
      dotTile ops d.tileM d.tileN d.tileK k (rowA d bf m0 (mt v * d.tileM + rl))
        (colB d bf m0 (gid v) (ni * d.tileN + cl)) :=
    congrArg₂ (ops.dot d.tileM d.tileN d.tileK) (funext fun u => hA u.val u.isLt)
      (funext fun u => hB hreal u.val u.isLt)
  simp only [accNext, List.getD_cons_zero, List.getD_cons_succ, ite_eq_left hre, hdot, accStep]

/-- The stored tile is `cast (scale acc sa sb)` for a real group and `cast 0` for the virtual
group. -/
private theorem tileOut_eq (ops : GroupedMmOps) (hr : Requires d bf go gid mt M m0)
    (junk : ℕ → ℝ) {ni v k rl cl : ℕ} (hni : ni < d.gn) (hv : v < d.tilesM + d.G)
    (hk : k < d.tilesK) (hrl : rl < d.tileM) (hcl : cl < d.tileN) (A : List ℕ → ℝ) :
    ops.cast (tileOut ops [ni, v, k] (GridSpec.scalarPrefetch (gridSpec d bf) m0)
        (loadIns (gridSpec d bf) junk m0 [ni, v, k]) A [rl, cl]) =
      if gid v < d.G then
        ops.cast (ops.scale (A [rl, cl]) (m0.read bf.saBuf (mt v * d.tileM + rl))
          (m0.read bf.sbBuf (gid v * (d.gn * d.tileN) + (ni * d.tileN + cl))))
      else ops.cast 0 := by
  obtain ⟨-, -, hSA, hSB⟩ := reads hr junk hni hv hk hrl hcl
  by_cases hreal : gid v < d.G
  · have hre : (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 1 v) <
        (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 3 0) := by
      rw [sc_gid hr hv, sc_ng hr]
      exact_mod_cast hreal
    simp only [tileOut, List.getD_cons_zero, List.getD_cons_succ, ite_eq_left hre, ite_eq_left hreal, hSA,
      hSB hreal]
  · have hre : ¬ (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 1 v) <
        (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 3 0) := by
      rw [sc_gid hr hv, sc_ng hr]
      exact_mod_cast hreal
    simp only [tileOut, List.getD_cons_zero, List.getD_cons_succ, ite_eq_right hre, ite_eq_right hreal]

/-- Before the last K tile the output block is unchanged. -/
private theorem outNext_mid (ops : GroupedMmOps) {k : ℕ} (hk : k + 1 < d.tilesK) (ni v : ℕ)
    (sc : ℕ → ℤ) (ins : List (List ℕ → ℝ)) (A P : List ℕ → ℝ) (loc : List ℕ) :
    outNext ops d [ni, v, k] sc ins A P loc = P loc := by
  have h : ¬ k = d.tilesK - 1 := by omega
  simp [outNext, h]

/-- The output block at the last K tile: a row of the visit's group gets the cast tile, and any
other row is read back and stored again. The whole-tile and the masked branch agree on this. -/
private theorem outNext_last (ops : GroupedMmOps) (hr : Requires d bf go gid mt M m0)
    {ni v k rl cl : ℕ} (hv : v < d.tilesM + d.G) (hk : k + 1 = d.tilesK) (hrl : rl < d.tileM)
    (ins : List (List ℕ → ℝ)) (A P : List ℕ → ℝ) :
    ((go (gid v) ≤ mt v * d.tileM + rl ∧ mt v * d.tileM + rl < go (gid v + 1)) →
      outNext ops d [ni, v, k] (GridSpec.scalarPrefetch (gridSpec d bf) m0) ins A P [rl, cl] =
        ops.cast (tileOut ops [ni, v, k] (GridSpec.scalarPrefetch (gridSpec d bf) m0) ins A [rl, cl])) ∧
    (¬ (go (gid v) ≤ mt v * d.tileM + rl ∧ mt v * d.tileM + rl < go (gid v + 1)) →
      outNext ops d [ni, v, k] (GridSpec.scalarPrefetch (gridSpec d bf) m0) ins A P [rl, cl] =
        ops.cast (ops.up (P [rl, cl]))) := by
  have hgid := hr.sched.gid_le v hv
  have hk' : k = d.tilesK - 1 := by omega
  have hgs : (GridSpec.scalarPrefetch (gridSpec d bf) m0)
      (Nat.pair 0 ((GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 1 v)).toNat) =
      (go (gid v) : ℤ) := by
    rw [sc_gid hr hv, Int.toNat_natCast, sc_go hr (by omega)]
  have hge : (GridSpec.scalarPrefetch (gridSpec d bf) m0)
      (Nat.pair 0 ((GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 1 v) + 1).toNat) =
      (go (gid v + 1) : ℤ) := by
    rw [sc_gid hr hv, show ((gid v : ℤ) + 1).toNat = gid v + 1 by omega, sc_go hr (by omega)]
  have hrow : (GridSpec.scalarPrefetch (gridSpec d bf) m0) (Nat.pair 2 v) * (d.tileM : ℤ) =
      ((mt v * d.tileM : ℕ) : ℤ) := by
    rw [sc_mt hr hv, Nat.cast_mul]
  constructor
  · intro hown
    simp only [outNext, List.getD_cons_zero, List.getD_cons_succ, hgs, hge, hrow, ite_eq_left hk']
    split_ifs with h1 h2
    · rfl
    · rfl
    · exfalso
      omega
  · intro hnot
    simp only [outNext, List.getD_cons_zero, List.getD_cons_succ, hgs, hge, hrow, ite_eq_left hk']
    split_ifs with h1 h2
    · exfalso
      omega
    · exfalso
      omega
    · rfl

/-- One grid step `(ni, v, k)` from any two-slot state: the new accumulator `A` and output
block `O` in terms of the previous scratch `S` and previous output block `P`. -/
private theorem step_spec (ops : GroupedMmOps) (hr : Requires d bf go gid mt M m0) (junk : ℕ → ℝ)
    (m : Mem) (hm : ∀ b, b ≠ bf.oBuf → ∀ a, m.read b a = m0.read b a) (s0 s1 : OutSlot)
    {ni v k : ℕ} (hni : ni < d.gn) (hv : v < d.tilesM + d.G) (hk : k < d.tilesK) :
    ∃ O A P S : List ℕ → ℝ,
      stepGrid (gridSpec d bf) (body ops d) (GridSpec.scalarPrefetch (gridSpec d bf) m0) junk
          (m, [s0, s1]) [ni, v, k] =
        (flushOne (oPair d bf) [mt v, ni] s0 m, [some ([mt v, ni], O), some ([], A)]) ∧
      (∀ O0, s0 = some ([mt v, ni], O0) →
        ∀ rl < d.tileM, ∀ cl < d.tileN, P [rl, cl] = O0 [rl, cl]) ∧
      (∀ i A0, s1 = some (i, A0) → ∀ rl < d.tileM, ∀ cl < d.tileN, S [rl, cl] = A0 [rl, cl]) ∧
      ∀ rl < d.tileM, ∀ cl < d.tileN,
        (gid v < d.G → A [rl, cl] =
          accStep ops d.tileM d.tileN d.tileK (rowA d bf m0 (mt v * d.tileM + rl))
            (colB d bf m0 (gid v) (ni * d.tileN + cl)) (S [rl, cl]) k) ∧
        (k + 1 < d.tilesK → O [rl, cl] = P [rl, cl]) ∧
        (k + 1 = d.tilesK →
          ((go (gid v) ≤ mt v * d.tileM + rl ∧ mt v * d.tileM + rl < go (gid v + 1)) →
            O [rl, cl] =
              if gid v < d.G then
                ops.cast (ops.scale (A [rl, cl]) (m0.read bf.saBuf (mt v * d.tileM + rl))
                  (m0.read bf.sbBuf (gid v * (d.gn * d.tileN) + (ni * d.tileN + cl))))
              else ops.cast 0) ∧
          (¬ (go (gid v) ≤ mt v * d.tileM + rl ∧ mt v * d.tileM + rl < go (gid v + 1)) →
            O [rl, cl] = ops.cast (ops.up (P [rl, cl])))) := by
  obtain ⟨P, S, hst, hP, hS⟩ := step_eq ops hr junk m hm s0 s1 (ni := ni) (k := k) hv
  refine ⟨_, _, P, S, hst, fun O0 h0 rl hrl cl hcl => hP O0 h0 _ (inb hrl hcl),
    fun i A0 h1 rl hrl cl hcl => hS i A0 h1 _ (inb hrl hcl), ?_⟩
  intro rl hrl cl hcl
  refine ⟨fun hreal => accNext_real ops hr junk hni hv hk hrl hcl hreal S,
    fun hlt => outNext_mid ops hlt _ _ _ _ _ _ _, fun heq => ?_⟩
  obtain ⟨h1, h2⟩ := outNext_last ops hr (ni := ni) (cl := cl) hv heq hrl
    (loadIns (gridSpec d bf) junk m0 [ni, v, k])
    (accNext ops d [ni, v, k] (GridSpec.scalarPrefetch (gridSpec d bf) m0)
      (loadIns (gridSpec d bf) junk m0 [ni, v, k]) S) P
  exact ⟨fun hown => (h1 hown).trans (tileOut_eq ops hr junk hni hv hk hrl hcl _), h2⟩

/-! ### One visit: the fold over K tiles -/

/-- The first `n` K tiles of visit `v` at column tile `ni`, from any two-slot state: memory gets
the previous block's write-back once; a real group's accumulator is `accFold`; before the last
K tile the output block is the previous one; after it, the visit's own rows hold their specified
values and the other rows hold the previous block read back and stored again. -/
private theorem visit_run (ops : GroupedMmOps) (hr : Requires d bf go gid mt M m0) (junk : ℕ → ℝ)
    (m : Mem) (hm : ∀ b, b ≠ bf.oBuf → ∀ a, m.read b a = m0.read b a) (s0 s1 : OutSlot)
    {ni v : ℕ} (hni : ni < d.gn) (hv : v < d.tilesM + d.G) (n : ℕ) (hn : 0 < n)
    (hle : n ≤ d.tilesK) :
    ∃ O A : List ℕ → ℝ,
      ((List.range n).map fun k => [ni, v, k]).foldl
          (stepGrid (gridSpec d bf) (body ops d) (GridSpec.scalarPrefetch (gridSpec d bf) m0) junk)
          (m, [s0, s1]) =
        (flushOne (oPair d bf) [mt v, ni] s0 m, [some ([mt v, ni], O), some ([], A)]) ∧
      (gid v < d.G → ∀ rl < d.tileM, ∀ cl < d.tileN,
        A [rl, cl] = accFold ops d.tileM d.tileN d.tileK (rowA d bf m0 (mt v * d.tileM + rl))
          (colB d bf m0 (gid v) (ni * d.tileN + cl)) n) ∧
      (n < d.tilesK → ∀ O0, s0 = some ([mt v, ni], O0) →
        ∀ rl < d.tileM, ∀ cl < d.tileN, O [rl, cl] = O0 [rl, cl]) ∧
      (n = d.tilesK → ∀ rl < d.tileM, ∀ cl < d.tileN,
        ((go (gid v) ≤ mt v * d.tileM + rl ∧ mt v * d.tileM + rl < go (gid v + 1)) →
          O [rl, cl] = rowVal ops d bf m0 (gid v) (mt v * d.tileM + rl) (ni * d.tileN + cl)) ∧
        (¬ (go (gid v) ≤ mt v * d.tileM + rl ∧ mt v * d.tileM + rl < go (gid v + 1)) →
          ∀ O0, s0 = some ([mt v, ni], O0) → O [rl, cl] = ops.cast (ops.up (O0 [rl, cl])))) := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [List.range_succ, List.map_append, List.foldl_append]
    simp only [List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil]
    rcases Nat.eq_zero_or_pos n with rfl | hpos
    · simp only [List.range_zero, List.map_nil, List.foldl_nil]
      obtain ⟨O, A, P, S, hst, hP, -, hsem⟩ :=
        step_spec ops hr junk m hm s0 s1 hni hv (k := 0) (by omega)
      refine ⟨O, A, hst, ?_, ?_, ?_⟩
      · intro hreal rl hrl cl hcl
        rw [(hsem rl hrl cl hcl).1 hreal]
        simp [accFold, accStep]
      · intro hlt O0 h0 rl hrl cl hcl
        rw [(hsem rl hrl cl hcl).2.1 (by omega)]
        exact hP O0 h0 rl hrl cl hcl
      · intro heq rl hrl cl hcl
        obtain ⟨hA, -, hlast⟩ := hsem rl hrl cl hcl
        obtain ⟨h1, h2⟩ := hlast (by omega)
        refine ⟨fun hown => ?_, fun hnot O0 h0 => ?_⟩
        · rw [h1 hown, rowVal]
          by_cases hreal : gid v < d.G
          · rw [ite_eq_left hreal, ite_eq_left hreal, hA hreal, groupedElem, ← heq]
            simp [accFold, accStep]
          · rw [ite_eq_right hreal, ite_eq_right hreal]
        · rw [h2 hnot, hP O0 h0 rl hrl cl hcl]
    · obtain ⟨O, A, hst, hA, hO, -⟩ := ih hpos (by omega)
      rw [hst]
      have hm1 : ∀ b, b ≠ bf.oBuf → ∀ a,
          (flushOne (oPair d bf) [mt v, ni] s0 m).read b a = m0.read b a := fun b hb a => by
        rw [flushOne_read_other_buf (oPair d bf) _ _ _ _ a hb]
        exact hm b hb a
      obtain ⟨O', A', P, S, hst', hP, hS, hsem⟩ := step_spec ops hr junk _ hm1
        (some ([mt v, ni], O)) (some ([], A)) hni hv (k := n) (by omega)
      have hfl : flushOne (oPair d bf) [mt v, ni] (some ([mt v, ni], O))
          (flushOne (oPair d bf) [mt v, ni] s0 m) = flushOne (oPair d bf) [mt v, ni] s0 m := by
        simp [flushOne]
      rw [hfl] at hst'
      refine ⟨O', A', hst', ?_, ?_, ?_⟩
      · intro hreal rl hrl cl hcl
        rw [(hsem rl hrl cl hcl).1 hreal, hS _ _ rfl rl hrl cl hcl, hA hreal rl hrl cl hcl]
        simp [accFold, List.range_succ]
      · intro hlt O0 h0 rl hrl cl hcl
        rw [(hsem rl hrl cl hcl).2.1 (by omega), hP _ rfl rl hrl cl hcl]
        exact hO (by omega) O0 h0 rl hrl cl hcl
      · intro heq rl hrl cl hcl
        obtain ⟨hA', -, hlast⟩ := hsem rl hrl cl hcl
        obtain ⟨h1, h2⟩ := hlast heq
        refine ⟨fun hown => ?_, fun hnot O0 h0 => ?_⟩
        · rw [h1 hown, rowVal]
          by_cases hreal : gid v < d.G
          · rw [ite_eq_left hreal, ite_eq_left hreal, hA' hreal, hS _ _ rfl rl hrl cl hcl,
              hA hreal rl hrl cl hcl, groupedElem, ← heq]
            simp [accFold, List.range_succ]
          · rw [ite_eq_right hreal, ite_eq_right hreal]
        · rw [h2 hnot, hP _ rfl rl hrl cl hcl, hO (by omega) O0 h0 rl hrl cl hcl]

/-! ### Reading the output buffer -/

private theorem addr_in_tile (hr : Requires d bf go gid mt M m0) {t t' ni ni' rl cl : ℕ}
    (ht : t < d.tilesM) (hni : ni < d.gn) (hrl : rl < d.tileM) (hcl : cl < d.tileN) :
    addrInBlock (oArr d bf) [d.tileM, d.tileN] [t', ni']
      ((t * d.tileM + rl) * (d.gn * d.tileN) + (ni * d.tileN + cl)) =
      decide (t' = t ∧ ni' = ni) := by
  have h := addrInBlock_two (buf := bf.oBuf) (i := t') (j := ni') hr.tileM_pos hr.tileN_pos
    (tile_lt ht hrl) (tile_lt hni hcl)
  rw [div_tile hrl, div_tile hcl] at h
  exact h

private theorem read_tile (hr : Requires d bf go gid mt M m0) {t ni rl cl : ℕ}
    (ht : t < d.tilesM) (hni : ni < d.gn) (hrl : rl < d.tileM) (hcl : cl < d.tileN)
    (O : List ℕ → ℝ) (m : Mem) :
    (storeBlock (oArr d bf) [d.tileM, d.tileN] [t, ni] O m).read bf.oBuf
      ((t * d.tileM + rl) * (d.gn * d.tileN) + (ni * d.tileN + cl)) = O [rl, cl] := by
  have hin : addrInBlock (oArr d bf) [d.tileM, d.tileN] [t, ni]
      ((t * d.tileM + rl) * (d.gn * d.tileN) + (ni * d.tileN + cl)) = true := by
    rw [addr_in_tile hr ht hni hrl hcl, decide_eq_true_eq]
    exact ⟨rfl, rfl⟩
  refine (storeBlock_read_inside (oArr d bf) _ _ O m _ hin).trans ?_
  show O (localCoord [d.tileM, d.tileN] [t, ni]
    (unflatCoord [d.tilesM * d.tileM, d.gn * d.tileN]
      ((t * d.tileM + rl) * (d.gn * d.tileN) + (ni * d.tileN + cl)))) = O [rl, cl]
  rw [unflatCoord_two _ _ _ _ (tile_lt hni hcl)]
  simp [localCoord]

/-- One whole visit, seen through the output buffer as it would be after a flush: addresses
outside the visit's block keep their value, the visit's own rows get their specified values, and
the other rows of the block are read back and stored again. -/
private theorem visit_view (ops : GroupedMmOps) (hr : Requires d bf go gid mt M m0)
    (junk : ℕ → ℝ) {ni v : ℕ} (hni : ni < d.gn) (hv : v < d.tilesM + d.G) (m : Mem)
    (hm : ∀ b, b ≠ bf.oBuf → ∀ a, m.read b a = m0.read b a) (s0 s1 : OutSlot) :
    ∃ O A : List ℕ → ℝ,
      ((List.range d.tilesK).map fun k => [ni, v, k]).foldl
          (stepGrid (gridSpec d bf) (body ops d) (GridSpec.scalarPrefetch (gridSpec d bf) m0) junk)
          (m, [s0, s1]) =
        (flushOne (oPair d bf) [mt v, ni] s0 m, [some ([mt v, ni], O), some ([], A)]) ∧
      (∀ a, addrInBlock (oArr d bf) [d.tileM, d.tileN] [mt v, ni] a = false →
        (storeBlock (oArr d bf) [d.tileM, d.tileN] [mt v, ni] O
          (flushOne (oPair d bf) [mt v, ni] s0 m)).read bf.oBuf a =
          (flushFinal (oPair d bf) s0 m).read bf.oBuf a) ∧
      ∀ rl < d.tileM, ∀ cl < d.tileN,
        ((go (gid v) ≤ mt v * d.tileM + rl ∧ mt v * d.tileM + rl < go (gid v + 1)) →
          (storeBlock (oArr d bf) [d.tileM, d.tileN] [mt v, ni] O
            (flushOne (oPair d bf) [mt v, ni] s0 m)).read bf.oBuf
              ((mt v * d.tileM + rl) * (d.gn * d.tileN) + (ni * d.tileN + cl)) =
            rowVal ops d bf m0 (gid v) (mt v * d.tileM + rl) (ni * d.tileN + cl)) ∧
        (¬ (go (gid v) ≤ mt v * d.tileM + rl ∧ mt v * d.tileM + rl < go (gid v + 1)) →
          ∀ O0, s0 = some ([mt v, ni], O0) →
            (storeBlock (oArr d bf) [d.tileM, d.tileN] [mt v, ni] O
              (flushOne (oPair d bf) [mt v, ni] s0 m)).read bf.oBuf
                ((mt v * d.tileM + rl) * (d.gn * d.tileN) + (ni * d.tileN + cl)) =
              ops.cast (ops.up (O0 [rl, cl]))) := by
  obtain ⟨O, A, hst, -, -, hlast⟩ :=
    visit_run ops hr junk m hm s0 s1 hni hv d.tilesK hr.tilesK_pos le_rfl
  have hmt := hr.sched.mt_lt v hv
  refine ⟨O, A, hst, ?_, ?_⟩
  · intro a ha
    exact (storeBlock_read_outside _ _ _ _ _ _ _ (Or.inr ha)).trans
      (flushOne_read_eq_flushFinal (oPair d bf) _ _ _ _ ha)
  · intro rl hrl cl hcl
    rw [read_tile hr hmt hni hrl hcl]
    exact hlast rfl rl hrl cl hcl

/-! ### One column tile: the fold over visits -/

/-- The first `j` visits of column tile `ni`, from any two-slot state. Seen through the output
buffer as it would be after a flush: addresses outside column tile `ni` keep their value, and
every row of the group of a completed visit holds its specified value in that visit's m-tile. -/
private theorem col_run (ops : GroupedMmOps) (hr : Requires d bf go gid mt M m0) (junk : ℕ → ℝ)
    {ni : ℕ} (hni : ni < d.gn) (m : Mem)
    (hm : ∀ b, b ≠ bf.oBuf → ∀ a, m.read b a = m0.read b a) (s0 s1 : OutSlot) (j : ℕ)
    (hj : 0 < j) (hjV : j ≤ d.tilesM + d.G) :
    ∃ (m' : Mem) (O A : List ℕ → ℝ),
      (List.range j).foldl (fun st v =>
          ((List.range d.tilesK).map fun k => [ni, v, k]).foldl
            (stepGrid (gridSpec d bf) (body ops d) (GridSpec.scalarPrefetch (gridSpec d bf) m0) junk) st) (m, [s0, s1]) =
        (m', [some ([mt (j - 1), ni], O), some ([], A)]) ∧
      (∀ b, b ≠ bf.oBuf → ∀ a, m'.read b a = m0.read b a) ∧
      (∀ a, (∀ t, addrInBlock (oArr d bf) [d.tileM, d.tileN] [t, ni] a = false) →
        (storeBlock (oArr d bf) [d.tileM, d.tileN] [mt (j - 1), ni] O m').read bf.oBuf a =
          (flushFinal (oPair d bf) s0 m).read bf.oBuf a) ∧
      (∀ u < j, ∀ rl < d.tileM, ∀ cl < d.tileN,
        (go (gid u) ≤ mt u * d.tileM + rl ∧ mt u * d.tileM + rl < go (gid u + 1)) →
        (storeBlock (oArr d bf) [d.tileM, d.tileN] [mt (j - 1), ni] O m').read bf.oBuf
            ((mt u * d.tileM + rl) * (d.gn * d.tileN) + (ni * d.tileN + cl)) =
          rowVal ops d bf m0 (gid u) (mt u * d.tileM + rl) (ni * d.tileN + cl)) := by
  induction j with
  | zero => omega
  | succ j ih =>
    rw [List.range_succ, List.foldl_append]
    simp only [List.foldl_cons, List.foldl_nil, Nat.add_sub_cancel]
    rcases Nat.eq_zero_or_pos j with rfl | hpos
    · simp only [List.range_zero, List.foldl_nil]
      obtain ⟨O, A, hst, hout, hin⟩ :=
        visit_view ops hr junk hni (v := 0) (by omega) m hm s0 s1
      refine ⟨_, O, A, hst, ?_, ?_, ?_⟩
      · intro b hb a
        rw [flushOne_read_other_buf (oPair d bf) _ _ _ _ a hb]
        exact hm b hb a
      · intro a ha
        exact hout a (ha _)
      · intro u hu rl hrl cl hcl hown
        obtain rfl : u = 0 := by omega
        exact (hin rl hrl cl hcl).1 hown
    · obtain ⟨m', O, A, hst, hm', hpres, hvals⟩ := ih hpos (by omega)
      rw [hst]
      obtain ⟨O', A', hst', hout, hin⟩ := visit_view ops hr junk hni (v := j) (by omega) m' hm'
        (some ([mt (j - 1), ni], O)) (some ([], A))
      refine ⟨_, O', A', hst', ?_, ?_, ?_⟩
      · intro b hb a
        rw [flushOne_read_other_buf (oPair d bf) _ _ _ _ a hb]
        exact hm' b hb a
      · intro a ha
        exact (hout a (ha _)).trans (hpres a ha)
      · intro u hu rl hrl cl hcl hown
        have hmtj := hr.sched.mt_lt j (by omega)
        have hmtu := hr.sched.mt_lt u (by omega)
        by_cases hsame : mt u = mt j
        · have hown' : go (gid u) ≤ mt j * d.tileM + rl ∧
              mt j * d.tileM + rl < go (gid u + 1) := by
            rw [← hsame]
            exact hown
          by_cases hownj : go (gid j) ≤ mt j * d.tileM + rl ∧
              mt j * d.tileM + rl < go (gid j + 1)
          · have hg : gid u = gid j := owner_unique hr.sched (hr.sched.gid_le u (by omega))
              (hr.sched.gid_le j (by omega)) hown' hownj
            rw [hsame, hg]
            exact (hin rl hrl cl hcl).1 hownj
          · have hu' : u < j := by
              by_contra hge
              have huj : u = j := by omega
              rw [huj] at hown'
              exact hownj hown'
            have hprev : mt (j - 1) = mt j := by
              have h1 := hr.sched.mt_mono (j - 1) (by omega) u (by omega)
              have h2 := hr.sched.mt_mono j (by omega) (j - 1) (by omega)
              omega
            have hold := hvals u hu' rl hrl cl hcl hown
            rw [hprev, hsame, read_tile hr hmtj hni hrl hcl] at hold
            rw [hsame, (hin rl hrl cl hcl).2 hownj O (by rw [hprev]), hold]
            exact rowVal_fix ops d bf m0 _ _ _
        · have hne : addrInBlock (oArr d bf) [d.tileM, d.tileN] [mt j, ni]
              ((mt u * d.tileM + rl) * (d.gn * d.tileN) + (ni * d.tileN + cl)) = false := by
            rw [addr_in_tile hr hmtu hni hrl hcl, decide_eq_false_iff_not]
            rintro ⟨h, -⟩
            exact hsame h.symm
          have hu' : u < j := by
            by_contra hge
            have huj : u = j := by omega
            exact hsame (by rw [huj])
          exact (hout _ hne).trans (hvals u hu' rl hrl cl hcl hown)

/-! ### The whole grid -/

/-- The first `n` column tiles, from any two-slot state: seen through the output buffer as it
would be after a flush, every row of every group holds its specified value in every completed
column tile. -/
private theorem grid_run (ops : GroupedMmOps) (hr : Requires d bf go gid mt M m0) (junk : ℕ → ℝ)
    (m : Mem) (hm : ∀ b, b ≠ bf.oBuf → ∀ a, m.read b a = m0.read b a) (s0 s1 : OutSlot) (n : ℕ)
    (hn : n ≤ d.gn) :
    ∃ (m' : Mem) (t0 t1 : OutSlot),
      (List.range n).foldl (fun st ni =>
          (List.range (d.tilesM + d.G)).foldl (fun st v =>
            ((List.range d.tilesK).map fun k => [ni, v, k]).foldl
              (stepGrid (gridSpec d bf) (body ops d) (GridSpec.scalarPrefetch (gridSpec d bf) m0) junk) st) st) (m, [s0, s1]) = (m', [t0, t1]) ∧
      (∀ b, b ≠ bf.oBuf → ∀ a, m'.read b a = m0.read b a) ∧
      ∀ ni < n, ∀ u < d.tilesM + d.G, ∀ rl < d.tileM, ∀ cl < d.tileN,
        (go (gid u) ≤ mt u * d.tileM + rl ∧ mt u * d.tileM + rl < go (gid u + 1)) →
        (flushFinal (oPair d bf) t0 m').read bf.oBuf
            ((mt u * d.tileM + rl) * (d.gn * d.tileN) + (ni * d.tileN + cl)) =
          rowVal ops d bf m0 (gid u) (mt u * d.tileM + rl) (ni * d.tileN + cl) := by
  induction n with
  | zero => exact ⟨m, s0, s1, rfl, hm, fun ni hni => absurd hni (Nat.not_lt_zero _)⟩
  | succ n ih =>
    obtain ⟨m', t0, t1, hst, hm', hvals⟩ := ih (by omega)
    rw [List.range_succ, List.foldl_append, hst]
    simp only [List.foldl_cons, List.foldl_nil]
    have hV : 0 < d.tilesM + d.G := by
      have := hr.sched.groups_pos
      omega
    obtain ⟨m'', O, A, hst', hm'', hpres, hnew⟩ :=
      col_run ops hr junk (ni := n) (by omega) m' hm' t0 t1 (d.tilesM + d.G) hV le_rfl
    refine ⟨m'', _, _, hst', hm'', ?_⟩
    intro ni hni u hu rl hrl cl hcl hown
    by_cases hlt : ni < n
    · have hout : ∀ t, addrInBlock (oArr d bf) [d.tileM, d.tileN] [t, n]
          ((mt u * d.tileM + rl) * (d.gn * d.tileN) + (ni * d.tileN + cl)) = false := by
        intro t
        rw [addr_in_tile hr (hr.sched.mt_lt u hu) (by omega) hrl hcl, decide_eq_false_iff_not]
        rintro ⟨-, h⟩
        omega
      exact (hpres _ hout).trans (hvals ni hlt u hu rl hrl cl hcl hown)
    · obtain rfl : ni = n := by omega
      exact hnew u hu rl hrl cl hcl hown

/-! ### Correctness -/

/-- The `pallas_call` of the grouped row-scaled matmul on the padded arrays, under `Requires`: the
call is defined; for every row `r < M`, column `c < PN` and group `g ≤ G` that owns row `r`
(`go g ≤ r < go (g + 1)`), output element `(r, c)` is `groupedElem` of row `r` of `a`, column `c`
of `b[g]`, `scale_a[r]` and `scale_b[g, c]` when `g` is a real group, and `cast 0` when `g = G` is
the virtual group; and nothing outside the output buffer changes. No claim is made for the padding
rows `r ≥ M`. -/
theorem rowwiseScaledGroupedMm_correct (ops : GroupedMmOps)
    (hr : Requires d bf go gid mt M m0) (junk : ℕ → ℝ) :
    ∃ m', pallasCall (gridSpec d bf) (body ops d) junk m0 = some m' ∧
      (∀ r < M, ∀ c < d.gn * d.tileN, ∀ g ≤ d.G, go g ≤ r → r < go (g + 1) →
        m'.read bf.oBuf (r * (d.gn * d.tileN) + c) =
          if g < d.G then
            groupedElem ops d.tilesK d.tileM d.tileN d.tileK
              (fun k => m0.read bf.aBuf (r * (d.tilesK * d.tileK) + k))
              (fun k => m0.read bf.bBuf
                (g * (d.tilesK * d.tileK * (d.gn * d.tileN)) + (k * (d.gn * d.tileN) + c)))
              (m0.read bf.saBuf r) (m0.read bf.sbBuf (g * (d.gn * d.tileN) + c))
          else ops.cast 0) ∧
      Mem.agreeOutside (Mem.bufFootprint bf.oBuf) m0 m' := by
  refine ⟨pallasExec (gridSpec d bf) (body ops d) junk m0,
    pallasCall_of_wellDefined _ (gridSpec_wellDefined ops hr), ?_, ?_⟩
  · intro r hrM c hc g hg hlo hhi
    obtain ⟨v, hv, hgv, hmv⟩ := hr.sched.covered r hrM g hg hlo hhi
    have hinit : ∀ b, b ≠ bf.oBuf → ∀ a,
        (initOutMem (gridSpec d bf) (junkAt junk (Nat.pair 0 0)) m0).read b a = m0.read b a := by
      intro b hb a
      refine initOutMemAux_read_other_buf _ _ b a 0 _ m0 ?_
      intro x hx
      simp only [gridSpec, List.mem_singleton] at hx
      subst hx
      exact Ne.symm hb
    obtain ⟨m', t0, t1, hrun, -, hvals⟩ := grid_run ops hr junk _ hinit none none d.gn le_rfl
    have hfold : (gridPoints [d.gn, d.tilesM + d.G, d.tilesK]).foldl
        (stepGrid (gridSpec d bf) (body ops d) ((gridSpec d bf).scalarPrefetch m0) junk)
        (initOutMem (gridSpec d bf) (junkAt junk (Nat.pair 0 0)) m0, [none, none]) =
        (m', [t0, t1]) := by
      rw [← hrun, gridPoints_three]
      simp only [List.foldl_flatMap]
    have hexec : pallasExec (gridSpec d bf) (body ops d) junk m0 =
        flushAll (gridSpec d bf).outSpecs [t0, t1] m' := by
      show flushAll (gridSpec d bf).outSpecs
        ((gridPoints [d.gn, d.tilesM + d.G, d.tilesK]).foldl
          (stepGrid (gridSpec d bf) (body ops d) ((gridSpec d bf).scalarPrefetch m0) junk)
          (initOutMem (gridSpec d bf) (junkAt junk (Nat.pair 0 0)) m0, [none, none])).2
        ((gridPoints [d.gn, d.tilesM + d.G, d.tilesK]).foldl
          (stepGrid (gridSpec d bf) (body ops d) ((gridSpec d bf).scalarPrefetch m0) junk)
          (initOutMem (gridSpec d bf) (junkAt junk (Nat.pair 0 0)) m0, [none, none])).1 = _
      rw [hfold]
    have hread : (pallasExec (gridSpec d bf) (body ops d) junk m0).read bf.oBuf
        (r * (d.gn * d.tileN) + c) =
        (flushFinal (oPair d bf) t0 m').read bf.oBuf (r * (d.gn * d.tileN) + c) := by
      rw [hexec]
      exact flushAll_read_at (gridSpec d bf).outSpecs [t0, t1] m' 0 (oPair d bf) rfl
        (by simp [gridSpec]) _
    have hrl : r % d.tileM < d.tileM := Nat.mod_lt _ hr.tileM_pos
    have hcl : c % d.tileN < d.tileN := Nat.mod_lt _ hr.tileN_pos
    have hni : c / d.tileN < d.gn :=
      Nat.div_lt_of_lt_mul (lt_of_lt_of_eq hc (Nat.mul_comm _ _))
    have hre : mt v * d.tileM + r % d.tileM = r := by
      rw [hmv]
      exact Nat.div_add_mod' r d.tileM
    have hce : c / d.tileN * d.tileN + c % d.tileN = c := Nat.div_add_mod' c d.tileN
    have h := hvals (c / d.tileN) hni v hv (r % d.tileM) hrl (c % d.tileN) hcl
      (by rw [hgv, hre]; exact ⟨hlo, hhi⟩)
    rw [hre, hce, hgv] at h
    rw [hread, h]
    rfl
  · refine Mem.agreeOutside_mono ?_
      (pallasExec_agreeOutside_outBufs (gridSpec d bf) (body ops d) junk m0)
    rintro ⟨b, a⟩ ⟨x, hx, hbuf⟩
    simp only [gridSpec, List.mem_singleton] at hx
    subst hx
    simpa [Mem.bufFootprint, oPair, oArr] using hbuf.symm

end Proof

end

end TpuKernelLean.Examples.GroupedScaledMm
