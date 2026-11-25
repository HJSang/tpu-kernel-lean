import TpuKernelLean.Pallas.RowKernel

/-!
Reduction kernels: `pallas_call`s whose grid is `outerGrid ++ [nk]`, where the outer dimensions
injectively tile one output array, the final dimension of length `nk` reduces into `nAcc` VMEM
scratch accumulators (reset at step `0`, updated elementwise in the tile each step from input
blocks), and the output block is written from the accumulators at step `nk - 1`.

`ReductionKernel.correct` proves, for every outer grid size, reduction length `nk > 0`, block
shape, junk and initial memory satisfying `ReductionKernel.Valid`, that the run is well
defined, every output element covered by an outer grid point equals `finalize` of the in-order
fold of `step` over inputs read from initial memory, and nothing outside the output buffer changes.
-/

namespace TpuKernelLean.Pallas

/-! ### Grid and block helpers -/

/-- Every in-bounds coordinate list has the same length as its shape. -/
theorem inBounds_length : ∀ {s g : List ℕ}, inBounds s g = true → g.length = s.length
  | [], [], _ => rfl
  | [], _ :: _, h => by simp [inBounds] at h
  | _ :: _, [], h => by simp [inBounds] at h
  | _ :: ds, _ :: gs, h => by
    simp only [inBounds, Bool.and_eq_true] at h
    simp [inBounds_length h.2]

/-- Every point in `gridPoints grid` has length `grid.length`. -/
theorem length_of_mem_gridPoints {grid g : List ℕ} (h : g ∈ gridPoints grid) :
    g.length = grid.length :=
  inBounds_length ((mem_gridPoints grid g).mp h)

/-- `gridPoints grid` enumerates every grid point without duplicates. -/
theorem gridPoints_nodup : ∀ (grid : List ℕ), (gridPoints grid).Nodup
  | [] => List.nodup_singleton []
  | d :: ds => by
    simp only [gridPoints]
    rw [List.nodup_flatMap]
    refine ⟨fun i _ => (gridPoints_nodup ds).map (fun a b h => by simpa using h), ?_⟩
    apply List.nodup_range.pairwise_of_forall_ne
    intro i _ i' _ hne x hx hx'
    simp only [List.mem_map] at hx hx'
    obtain ⟨r, _, rfl⟩ := hx
    obtain ⟨r', _, h⟩ := hx'
    exact hne (List.cons.inj h).1.symm

/-- Splitting off the last axis of `gridPoints` groups points by outer coordinates, then last axis. -/
theorem gridPoints_append_singleton : ∀ (outer : List ℕ) (nk : ℕ),
    gridPoints (outer ++ [nk]) =
      (gridPoints outer).flatMap fun og => (List.range nk).map fun t => og ++ [t]
  | [], nk => by
    simp only [List.nil_append, gridPoints, List.map_cons, List.map_nil, List.flatMap_cons,
      List.flatMap_nil, List.append_nil]
    induction (List.range nk) with
    | nil => rfl
    | cons x xs ih => simp [ih]
  | d :: ds, nk => by
    simp only [List.cons_append, gridPoints]
    rw [gridPoints_append_singleton ds nk]
    simp only [List.flatMap_assoc, List.map_flatMap, List.flatMap_map, List.map_map,
      Function.comp_def, List.cons_append]

/-- Mapping `fun k => f (l.getD k d)` over `List.range l.length` equals `l.map f`. -/
theorem range_map_getD {α β : Type*} (l : List α) (d : α) (f : α → β) :
    (List.range l.length).map (fun k => f (l.getD k d)) = l.map f := by
  apply List.ext_getElem (by simp)
  intro k h1 h2
  simp only [List.getElem_map, List.getElem_range]
  rw [List.getD_eq_getElem _ _ (by simpa using h1)]

private theorem dropWhile_replicate_append {α : Type*} [DecidableEq α] (x : α) (L : List α) :
    ∀ n, (List.replicate n x ++ L).dropWhile (fun y => decide (y = x)) =
      L.dropWhile (fun y => decide (y = x))
  | 0 => rfl
  | n + 1 => by
    rw [List.replicate_succ, List.cons_append, List.dropWhile_cons]
    simp [dropWhile_replicate_append x L n]

/-- Prepending a consecutive run of an element absent from `L` preserves `noRevisit`. -/
theorem noRevisit_replicate_append {α : Type*} [DecidableEq α] (x : α) (L : List α) (hx : x ∉ L) :
    ∀ n, noRevisit (List.replicate n x ++ L) = noRevisit L
  | 0 => rfl
  | n + 1 => by
    rw [List.replicate_succ, List.cons_append, noRevisit, noRevisit_replicate_append x L hx n,
      dropWhile_replicate_append]
    have : x ∉ L.dropWhile fun y => decide (y = x) :=
      fun h => hx ((List.dropWhile_sublist _).subset h)
    simp [this]

/-- Repeating each element of a duplicate-free list `n` times consecutively satisfies `noRevisit`. -/
theorem noRevisit_flatMap_replicate {β α : Type*} [DecidableEq α] (f : β → α) (n : ℕ) :
    ∀ (l : List β), (l.map f).Nodup → noRevisit (l.flatMap fun b => List.replicate n (f b)) = true
  | [], _ => rfl
  | b :: rest, h => by
    rw [List.map_cons, List.nodup_cons] at h
    rw [List.flatMap_cons, noRevisit_replicate_append]
    · exact noRevisit_flatMap_replicate f n rest h.2
    · intro hm
      simp only [List.mem_flatMap, List.mem_replicate] at hm
      obtain ⟨b', hb', _, hfb⟩ := hm
      exact h.1 (List.mem_map.mpr ⟨b', hb', hfb.symm⟩)

/-- Distinct block indices with the same block shape cover disjoint coordinate sets. -/
theorem coordInBlock_unique {bs bi₁ bi₂ c : List ℕ}
    (h₁ : coordInBlock bs bi₁ c = true) (h₂ : coordInBlock bs bi₂ c = true) : bi₁ = bi₂ := by
  induction bs generalizing bi₁ bi₂ c with
  | nil =>
    cases bi₁ <;> cases bi₂ <;> cases c <;> simp [coordInBlock] at h₁ h₂ ⊢
  | cons b bss ih =>
    cases bi₁ with
    | nil => simp [coordInBlock] at h₁
    | cons i₁ is₁ =>
      cases bi₂ with
      | nil => simp [coordInBlock] at h₂
      | cons i₂ is₂ =>
        cases c with
        | nil => simp [coordInBlock] at h₁
        | cons x xs =>
          simp only [coordInBlock, Bool.and_eq_true, decide_eq_true_eq] at h₁ h₂
          have hi : i₁ = i₂ := by
            by_contra hne
            rcases Nat.lt_or_gt_of_ne hne with hlt | hgt
            · have : (i₁ + 1) * b ≤ i₂ * b := Nat.mul_le_mul_right b hlt
              omega
            · have : (i₂ + 1) * b ≤ i₁ * b := Nat.mul_le_mul_right b hgt
              omega
          exact congrArg₂ List.cons hi (ih h₁.2 h₂.2)

/-- Distinct block indices with the same block shape cover disjoint addresses of `arr`. -/
theorem addrInBlock_unique {arr : ArrayDesc} {bs bi₁ bi₂ : List ℕ} {a : ℕ}
    (h₁ : addrInBlock arr bs bi₁ a = true) (h₂ : addrInBlock arr bs bi₂ a = true) :
    bi₁ = bi₂ := by
  simp only [addrInBlock, Bool.and_eq_true] at h₁ h₂
  exact coordInBlock_unique h₁.2 h₂.2

/-- The local coordinate of any coordinate inside `(bs, bi)` is in bounds for `bs`. -/
theorem inBounds_localCoord_of_coordInBlock {bs bi c : List ℕ}
    (h : coordInBlock bs bi c = true) : inBounds bs (localCoord bs bi c) = true := by
  induction bs generalizing bi c with
  | nil =>
    cases bi <;> cases c <;> simp [coordInBlock, localCoord, inBounds] at h ⊢
  | cons b bss ih =>
    cases bi with
    | nil => simp [coordInBlock] at h
    | cons i is =>
      cases c with
      | nil => simp [coordInBlock] at h
      | cons x xs =>
        simp only [coordInBlock, Bool.and_eq_true, decide_eq_true_eq] at h
        simp only [localCoord, inBounds, Bool.and_eq_true, decide_eq_true_eq]
        refine ⟨?_, ih h.2⟩
        have hmul : (i + 1) * b = i * b + b := by rw [Nat.succ_mul]
        omega

/-- Every coordinate of a `blockOk` integer block index is non-negative. -/
theorem blockOk_nonneg {ds bs : List ℕ} {is : List ℤ} (h : blockOk ds bs is = true) :
    ∀ z ∈ is, 0 ≤ z := by
  induction ds generalizing bs is with
  | nil =>
    cases bs <;> cases is <;> simp [blockOk] at h ⊢
  | cons d ds ih =>
    cases bs with
    | nil => simp [blockOk] at h
    | cons b bs =>
      cases is with
      | nil => simp [blockOk] at h
      | cons i is =>
        simp only [blockOk, Bool.and_eq_true, decide_eq_true_eq] at h
        intro z hz
        simp only [List.mem_cons] at hz
        rcases hz with rfl | hz
        · exact h.1.2.1
        · exact ih h.2 z hz

/-- `List.map Int.toNat` is injective on lists of non-negative integers. -/
theorem map_toNat_inj {l₁ l₂ : List ℤ} (h₁ : ∀ z ∈ l₁, 0 ≤ z) (h₂ : ∀ z ∈ l₂, 0 ≤ z)
    (heq : l₁.map Int.toNat = l₂.map Int.toNat) : l₁ = l₂ := by
  induction l₁ generalizing l₂ with
  | nil =>
    cases l₂ with
    | nil => rfl
    | cons _ _ => simp at heq
  | cons a as ih =>
    cases l₂ with
    | nil => simp at heq
    | cons b bs =>
      simp only [List.map_cons, List.cons.injEq] at heq
      have ha : 0 ≤ a := h₁ a (by simp)
      have hb : 0 ≤ b := h₂ b (by simp)
      have hab : a = b := by
        have hcast := congrArg Int.ofNat heq.1
        rwa [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast, Int.toNat_of_nonneg ha,
          Int.toNat_of_nonneg hb] at hcast
      exact congrArg₂ List.cons hab
        (ih (fun z hz => h₁ z (by simp [hz])) (fun z hz => h₂ z (by simp [hz])) heq.2)

/-- Every tile index below `cdiv D B` starts inside `D`. -/
theorem lt_of_lt_cdiv {D B t : ℕ} (hB : 0 < B) (ht : t < (D + B - 1) / B) : t * B < D := by
  have h := (Nat.le_div_iff_mul_le hB).mp ht
  rw [Nat.succ_mul] at h
  omega

/-- Unflattening a 1-D offset `r` returns `[r]`. -/
theorem unflatCoord_one (M r : ℕ) : unflatCoord [M] r = [r] := by
  simp [unflatCoord]

/-- An in-bounds element `r` of a 1-D `[M]` array lies in block `r / bm`. -/
theorem addrInBlock_one {buf : BufId} {M bm r i : ℕ} (hbm : 0 < bm) (hr : r < M) :
    addrInBlock ⟨buf, [M]⟩ [bm] [i] r = decide (i = r / bm) := by
  have key : (i * bm ≤ r ∧ r < (i + 1) * bm) ↔ i = r / bm := by
    constructor
    · rintro ⟨h1, h2⟩
      exact (Nat.div_eq_of_lt_le h1 h2).symm
    · rintro rfl
      refine ⟨Nat.div_mul_le_self r bm, ?_⟩
      rw [Nat.succ_mul]
      exact Nat.lt_div_mul_add hbm
  simp [addrInBlock, unflatCoord_one, coordInBlock, hr, key]

/-- An in-bounds element `(r, c)` of a 2-D `[M, N]` array lies in tile `(r / bm, c / bn)`. -/
theorem addrInBlock_two {buf : BufId} {M N bm bn r c i j : ℕ} (hbm : 0 < bm) (hbn : 0 < bn)
    (hr : r < M) (hc : c < N) :
    addrInBlock ⟨buf, [M, N]⟩ [bm, bn] [i, j] (r * N + c) = decide (i = r / bm ∧ j = c / bn) := by
  have hlt : r * N + c < M * N := by nlinarith
  have key : ∀ {b x k : ℕ}, 0 < b → ((k * b ≤ x ∧ x < (k + 1) * b) ↔ k = x / b) := by
    intro b x k hb
    constructor
    · rintro ⟨h1, h2⟩
      exact (Nat.div_eq_of_lt_le h1 h2).symm
    · rintro rfl
      refine ⟨Nat.div_mul_le_self x b, ?_⟩
      rw [Nat.succ_mul]
      exact Nat.lt_div_mul_add hb
  simp [addrInBlock, unflatCoord_two M N r c hc, coordInBlock, hlt, key hbm, key hbn]

/-! ### Reduction kernel abstraction -/

/-- A reduction kernel whose last grid axis of length `nk` reduces into `nAcc` VMEM scratch
accumulators (reset at step `0`, updated elementwise in the tile each step) and writes one output
tile from the accumulators at step `nk - 1`. -/
structure ReductionKernel (nAcc : ℕ) where
  /-- Outer grid dimensions (all axes before the final reduction axis). -/
  outerGrid : List ℕ
  /-- Length of the reduction axis (the final grid dimension). -/
  nk : ℕ
  /-- Backing HBM buffers of the leading scalar-prefetch arguments. -/
  scalarPrefetchBufs : List BufId := []
  /-- Scalar-prefetch extractor fed only the checked `readInt32` readers of `scalarPrefetchBufs`. -/
  scalarPrefetchFn : List (ℕ → Option ℤ) → ℕ → ℤ := fun _ _ => 0
  /-- Input array descriptors paired with their block specs. -/
  inSpecs : List (ArrayDesc × BlockSpec)
  /-- Output array descriptor. -/
  outArr : ArrayDesc
  /-- Local VMEM block shape of the output tile and of each scratch accumulator. -/
  outBlockShape : List ℕ
  /-- Output block index as a function of outer grid coordinates and scalar prefetch. -/
  outIndexMap : List ℕ → (ℕ → ℤ) → List ℤ
  /-- Initial value of each accumulator at reduction step `0`. -/
  init : Fin nAcc → ℝ
  /-- Elementwise-in-the-tile accumulator update at outer coordinate `og`, reduction step `t`,
  scalars `sc`, input blocks `ins`, local coordinate `loc`, and previous accumulator values. -/
  step : List ℕ → ℕ → (ℕ → ℤ) → List (List ℕ → ℝ) → List ℕ → (Fin nAcc → ℝ) → Fin nAcc → ℝ
  /-- Output element written at step `nk - 1` at outer coordinate `og`, scalars `sc`, that step's
  input blocks `ins` and local coordinate `loc`, from the final accumulator values. -/
  finalize : List ℕ → (ℕ → ℤ) → List (List ℕ → ℝ) → List ℕ → (Fin nAcc → ℝ) → ℝ
  /-- Per-dimension execution semantics for `outerGrid` (default: all `arbitrary`). -/
  outerDimSemantics : List DimSemantics := List.replicate outerGrid.length .arbitrary

namespace ReductionKernel

variable {nAcc : ℕ}

/-- Output `BlockSpec`: depends only on the outer grid coordinates `g.take outerGrid.length`. -/
def outSpec (rk : ReductionKernel nAcc) : BlockSpec :=
  { blockShape := rk.outBlockShape
    indexMap := fun g sc => rk.outIndexMap (g.take rk.outerGrid.length) sc }

/-- Output array descriptor paired with its `BlockSpec`. -/
def outPair (rk : ReductionKernel nAcc) : ArrayDesc × BlockSpec :=
  (rk.outArr, rk.outSpec)

/-- Natural-number output block index for outer grid point `og`. -/
def outBlockIdx (rk : ReductionKernel nAcc) (og : List ℕ) (sc : ℕ → ℤ) : List ℕ :=
  (rk.outIndexMap og sc).map Int.toNat

/-- Scalar-prefetch table read from `rk.scalarPrefetchBufs` through `readInt32`. -/
noncomputable def scalarPrefetch (rk : ReductionKernel nAcc) (m0 : Mem) : ℕ → ℤ :=
  rk.scalarPrefetchFn (rk.scalarPrefetchBufs.map fun b => readInt32 m0 b)

/-- The `pallas_call` `GridSpec`: grid `outerGrid ++ [nk]`, one output, and `nAcc` scratch buffers
shaped `outBlockShape`. -/
def gridSpec (rk : ReductionKernel nAcc) : GridSpec :=
  { grid := rk.outerGrid ++ [rk.nk]
    scalarPrefetchBufs := rk.scalarPrefetchBufs
    scalarPrefetchFn := rk.scalarPrefetchFn
    inSpecs := rk.inSpecs
    outSpecs := [rk.outPair]
    scratch := List.replicate nAcc rk.outBlockShape
    dimSemantics := rk.outerDimSemantics ++ [.arbitrary] }

/-- Kernel body: reset the `nAcc` scratch accumulators to `init` at step `0`, apply `step`
elementwise in the tile, and write `finalize` to the output block at step `nk - 1`. -/
def body (rk : ReductionKernel nAcc) : Body := fun g sc ins prev =>
  let og := g.take rk.outerGrid.length
  let kk := g.getD rk.outerGrid.length 0
  let newAcc : List ℕ → Fin nAcc → ℝ := fun loc =>
    rk.step og kk sc ins loc fun j =>
      if kk = 0 then rk.init j else (prev.getD (j.val + 1) fun _ => 0) loc
  (fun loc =>
    if kk = rk.nk - 1 then rk.finalize og sc ins loc (newAcc loc)
    else (prev.getD 0 fun _ => 0) loc) ::
    List.ofFn fun j : Fin nAcc => fun loc => newAcc loc j

/-- Preconditions for a `ReductionKernel` run on initial memory `m0` to be well defined and free of
output/input buffer aliasing. -/
structure Valid (rk : ReductionKernel nAcc) (m0 : Mem) : Prop where
  /-- Every outer grid dimension is strictly positive. -/
  outer_grid_pos : ∀ d ∈ rk.outerGrid, 0 < d
  /-- The reduction axis is non-empty. -/
  nk_pos : 0 < rk.nk
  /-- Every input block index is legal at every grid point `og ++ [t]`. -/
  in_blocks_ok : ∀ og ∈ gridPoints rk.outerGrid, ∀ t < rk.nk, ∀ p ∈ rk.inSpecs,
    blockOk p.1.shape p.2.blockShape (p.2.indexMap (og ++ [t]) (rk.scalarPrefetch m0)) = true
  /-- Every output block index is legal at every outer grid point `og`. -/
  out_blocks_ok : ∀ og ∈ gridPoints rk.outerGrid,
    blockOk rk.outArr.shape rk.outBlockShape (rk.outIndexMap og (rk.scalarPrefetch m0)) = true
  /-- Distinct outer grid points map to distinct output block indices. -/
  out_inj : ∀ og₁ ∈ gridPoints rk.outerGrid, ∀ og₂ ∈ gridPoints rk.outerGrid,
    rk.outIndexMap og₁ (rk.scalarPrefetch m0) = rk.outIndexMap og₂ (rk.scalarPrefetch m0) →
      og₁ = og₂
  /-- No input buffer aliases the output buffer. -/
  ins_disjoint : ∀ p ∈ rk.inSpecs, p.1.buf ≠ rk.outArr.buf
  /-- No scalar-prefetch buffer aliases the output buffer. -/
  out_disjoint_scalar : ∀ b ∈ rk.scalarPrefetchBufs, rk.outArr.buf ≠ b := by
    intro _ h; cases h
  /-- `outerDimSemantics` has one entry per outer grid axis. -/
  outer_dim_semantics_len : rk.outerDimSemantics.length = rk.outerGrid.length := by rfl
  /-- Every `parallel` outer axis precedes every `arbitrary` outer axis. -/
  outer_dim_semantics_prefix : parallelPrefix rk.outerDimSemantics = true := by rfl

private theorem parallelPrefix_append_arbitrary : ∀ (l : List DimSemantics),
    parallelPrefix l = true → parallelPrefix (l ++ [.arbitrary]) = true
  | [], _ => rfl
  | .parallel :: rest, h => parallelPrefix_append_arbitrary rest h
  | .arbitrary :: rest, h => by
    simp only [List.cons_append, parallelPrefix, List.all_append, List.all_cons, List.all_nil,
      DimSemantics.isParallel, Bool.not_false, Bool.and_true] at h ⊢
    exact h

private theorem takeWhile_append_false {α : Type*} (p : α → Bool) (l : List α) (x : α)
    (hx : p x = false) : (l ++ [x]).takeWhile p = l.takeWhile p := by
  induction l with
  | nil => simp [hx]
  | cons a as ih =>
    simp only [List.cons_append, List.takeWhile_cons]
    cases p a <;> simp [ih]

private theorem length_takeWhile_le {α : Type*} (p : α → Bool) :
    ∀ (l : List α), (l.takeWhile p).length ≤ l.length
  | [] => Nat.zero_le _
  | a :: as => by
    simp only [List.takeWhile_cons]
    cases p a
    · simp
    · simp only [List.length_cons]
      exact Nat.succ_le_succ (length_takeWhile_le p as)

private theorem inBounds_append : ∀ {p a x y : List ℕ},
    inBounds p x = true → inBounds a y = true → inBounds (p ++ a) (x ++ y) = true
  | [], _, [], _, _, hy => hy
  | [], _, _ :: _, _, hx, _ => by simp [inBounds] at hx
  | _ :: _, _, [], _, hx, _ => by simp [inBounds] at hx
  | _ :: _, _, _ :: _, _, hx, hy => by
    simp only [List.cons_append, inBounds, Bool.and_eq_true] at hx ⊢
    exact ⟨hx.1, inBounds_append hx.2 hy⟩

private theorem inBounds_take : ∀ {s g : List ℕ} (n : ℕ),
    inBounds s g = true → inBounds (s.take n) (g.take n) = true
  | _, _, 0, _ => rfl
  | [], [], _ + 1, _ => rfl
  | [], _ :: _, _ + 1, h => by simp [inBounds] at h
  | _ :: _, [], _ + 1, h => by simp [inBounds] at h
  | _ :: _, _ :: _, n + 1, h => by
    simp only [List.take_succ_cons, inBounds, Bool.and_eq_true] at h ⊢
    exact ⟨h.1, inBounds_take n h.2⟩

theorem head_gridPoints_eq_replicate_zero : ∀ (s : List ℕ) (g : List ℕ),
    g ∈ (gridPoints s).head? → g = List.replicate s.length 0
  | [], g, h => by
    simp only [gridPoints, List.head?_cons, Option.mem_def, Option.some.injEq] at h
    exact h.symm
  | d :: ds, g, h => by
    cases hd : gridPoints ds with
    | nil =>
      have hnil : gridPoints (d :: ds) = [] := by simp [gridPoints, hd]
      rw [hnil] at h
      simp at h
    | cons g0 gs =>
      cases d with
      | zero => simp [gridPoints] at h
      | succ d' =>
        have ih := head_gridPoints_eq_replicate_zero ds g0 (by simp [hd])
        cases hr : List.range (d' + 1) with
        | nil =>
          have := congrArg List.length hr
          simp at this
        | cons x xs =>
          have hx : (List.range (d' + 1))[0]'(by simp) = x := by simp [hr]
          simp only [List.getElem_range] at hx
          subst hx
          rw [gridPoints, hr, List.flatMap_cons, hd, List.map_cons, List.cons_append,
            List.head?_cons, Option.mem_def, Option.some.injEq] at h
          rw [← h, ih, List.length_cons, List.replicate_succ]

/-- A valid reduction kernel satisfies `GridSpec.WellDefined`. -/
theorem wellDefined (rk : ReductionKernel nAcc) {m0 : Mem} (hv : rk.Valid m0) :
    rk.gridSpec.WellDefined rk.body m0 := by
  have hnumPar : rk.gridSpec.numParallel =
      (rk.outerDimSemantics.takeWhile DimSemantics.isParallel).length := by
    simp only [GridSpec.numParallel, gridSpec,
      takeWhile_append_false DimSemantics.isParallel rk.outerDimSemantics .arbitrary rfl]
  have hnumPar_le : rk.gridSpec.numParallel ≤ rk.outerGrid.length := by
    rw [hnumPar, ← hv.outer_dim_semantics_len]
    exact length_takeWhile_le _ _
  have hparGrid : rk.gridSpec.parGrid = rk.outerGrid.take rk.gridSpec.numParallel := by
    simp only [GridSpec.parGrid, gridSpec]
    exact List.take_append_of_le_length hnumPar_le
  have harbGrid : rk.gridSpec.arbGrid =
      rk.outerGrid.drop rk.gridSpec.numParallel ++ [rk.nk] := by
    simp only [GridSpec.arbGrid, gridSpec]
    exact List.drop_append_of_le_length hnumPar_le
  constructor
  · intro d hd
    simp only [gridSpec, List.mem_append, List.mem_singleton] at hd
    rcases hd with hd | rfl
    · exact hv.outer_grid_pos d hd
    · exact hv.nk_pos
  · intro p hp
    have hog₀ : List.replicate rk.outerGrid.length 0 ∈ gridPoints rk.outerGrid :=
      replicate_zero_mem_gridPoints hv.outer_grid_pos
    simp only [gridSpec, List.mem_append, List.mem_singleton] at hp
    rcases hp with hp | rfl
    · exact blockOk_spec (hv.in_blocks_ok _ hog₀ 0 hv.nk_pos p hp)
    · exact blockOk_spec (hv.out_blocks_ok _ hog₀)
  · intro k hk p hp
    have hk' : k < (gridPoints (rk.outerGrid ++ [rk.nk])).length := hk
    set g := (gridPoints (rk.outerGrid ++ [rk.nk])).getD k [] with hg_def
    have hmem : g ∈ gridPoints (rk.outerGrid ++ [rk.nk]) := by
      rw [hg_def, List.getD_eq_getElem _ _ hk']
      exact List.getElem_mem hk'
    rw [gridPoints_append_singleton, List.mem_flatMap] at hmem
    obtain ⟨og, hog, hmemt⟩ := hmem
    rw [List.mem_map] at hmemt
    obtain ⟨t, ht, hg⟩ := hmemt
    rw [List.mem_range] at ht
    simp only [gridSpec, List.mem_append, List.mem_singleton] at hp
    rcases hp with hp | rfl
    · simp only [GridSpec.idxAt, gridSpec]
      rw [← hg_def, ← hg]
      exact hv.in_blocks_ok og hog t ht p hp
    · have hlen : og.length = rk.outerGrid.length := length_of_mem_gridPoints hog
      simp only [GridSpec.idxAt, gridSpec, outPair, outSpec]
      rw [← hg_def, ← hg, ← hlen, List.take_left]
      exact hv.out_blocks_ok og hog
  · intro p hp
    simp only [gridSpec, List.mem_singleton] at hp
    subst hp
    have hrange : (List.range (gridPoints rk.gridSpec.grid).length).map
        (rk.gridSpec.idxAt rk.outSpec m0) =
        (gridPoints (rk.outerGrid ++ [rk.nk])).map
          fun g => rk.outSpec.indexMap g (rk.scalarPrefetch m0) :=
      range_map_getD (gridPoints (rk.outerGrid ++ [rk.nk])) []
        (fun g => rk.outSpec.indexMap g (rk.scalarPrefetch m0))
    rw [outPair, hrange, gridPoints_append_singleton, List.map_flatMap]
    have hflat : ((gridPoints rk.outerGrid).flatMap fun og =>
        ((List.range rk.nk).map fun t => og ++ [t]).map
          (fun g => rk.outSpec.indexMap g (rk.scalarPrefetch m0))) =
        (gridPoints rk.outerGrid).flatMap fun og =>
          List.replicate rk.nk (rk.outIndexMap og (rk.scalarPrefetch m0)) := by
      apply List.flatMap_congr
      intro og hog
      have hlen : og.length = rk.outerGrid.length := length_of_mem_gridPoints hog
      apply List.ext_getElem (by simp)
      intro n h1 h2
      simp [outSpec, ← hlen]
    rw [hflat]
    apply noRevisit_flatMap_replicate (fun og => rk.outIndexMap og (rk.scalarPrefetch m0))
    exact (gridPoints_nodup rk.outerGrid).map_on (fun a ha b hb h => hv.out_inj a ha b hb h)
  · intro _ _ _ _
    simp [body, gridSpec, Nat.add_comm]
  · intro s t _ _ i hi o ho hov
    simp only [gridSpec, List.mem_singleton] at ho
    subst ho
    exact hv.ins_disjoint i hi hov.1
  · exact GridSpec.AliasesValid.of_empty rfl (by simp [gridSpec]) (by
      intro inp hinp outp houtp
      simp only [gridSpec, List.mem_singleton] at houtp
      subst houtp
      exact hv.ins_disjoint inp hinp) (by
      intro b hb outp houtp
      simp only [gridSpec, List.mem_singleton] at houtp
      subst houtp
      exact hv.out_disjoint_scalar b hb)
  · simp [gridSpec, hv.outer_dim_semantics_len]
  · exact parallelPrefix_append_arbitrary rk.outerDimSemantics hv.outer_dim_semantics_prefix
  · intro pg₁ hpg₁ pg₂ hpg₂ hne ag₁ hag₁ ag₂ hag₂ o₁ ho₁ o₂ ho₂ hov
    simp only [gridSpec, List.mem_singleton] at ho₁ ho₂
    subst ho₁ ho₂
    have hsplit : rk.gridSpec.parGrid ++ rk.gridSpec.arbGrid = rk.outerGrid ++ [rk.nk] := by
      rw [hparGrid, harbGrid, ← List.append_assoc, List.take_append_drop]
    have hmem₁ : inBounds (rk.outerGrid ++ [rk.nk]) (pg₁ ++ ag₁) = true := by
      rw [← hsplit]
      exact inBounds_append ((mem_gridPoints _ _).mp hpg₁) ((mem_gridPoints _ _).mp hag₁)
    have hmem₂ : inBounds (rk.outerGrid ++ [rk.nk]) (pg₂ ++ ag₂) = true := by
      rw [← hsplit]
      exact inBounds_append ((mem_gridPoints _ _).mp hpg₂) ((mem_gridPoints _ _).mp hag₂)
    set og₁ := (pg₁ ++ ag₁).take rk.outerGrid.length
    set og₂ := (pg₂ ++ ag₂).take rk.outerGrid.length
    have hog₁ : og₁ ∈ gridPoints rk.outerGrid := by
      rw [mem_gridPoints]
      have ht := inBounds_take rk.outerGrid.length hmem₁
      rwa [List.take_left] at ht
    have hog₂ : og₂ ∈ gridPoints rk.outerGrid := by
      rw [mem_gridPoints]
      have ht := inBounds_take rk.outerGrid.length hmem₂
      rwa [List.take_left] at ht
    obtain ⟨_, a, ha₁, ha₂⟩ := hov
    have hbi : rk.outBlockIdx og₁ (rk.scalarPrefetch m0) =
        rk.outBlockIdx og₂ (rk.scalarPrefetch m0) := addrInBlock_unique ha₁ ha₂
    have hidx : rk.outIndexMap og₁ (rk.scalarPrefetch m0) =
        rk.outIndexMap og₂ (rk.scalarPrefetch m0) :=
      map_toNat_inj (blockOk_nonneg (hv.out_blocks_ok og₁ hog₁))
        (blockOk_nonneg (hv.out_blocks_ok og₂ hog₂)) hbi
    have hog_eq : og₁ = og₂ := hv.out_inj og₁ hog₁ og₂ hog₂ hidx
    have hlen₁ : pg₁.length = rk.gridSpec.numParallel := by
      rw [length_of_mem_gridPoints hpg₁, hparGrid, List.length_take, Nat.min_eq_left hnumPar_le]
    have hlen₂ : pg₂.length = rk.gridSpec.numParallel := by
      rw [length_of_mem_gridPoints hpg₂, hparGrid, List.length_take, Nat.min_eq_left hnumPar_le]
    have htake₁ : og₁.take rk.gridSpec.numParallel = pg₁ := by
      rw [List.take_take, Nat.min_eq_left hnumPar_le, ← hlen₁, List.take_left]
    have htake₂ : og₂.take rk.gridSpec.numParallel = pg₂ := by
      rw [List.take_take, Nat.min_eq_left hnumPar_le, ← hlen₂, List.take_left]
    exact hne (by rw [← htake₁, ← htake₂, hog_eq])
  · intro _ _ _ _ _ _ _ _ _ i hi o ho hov
    simp only [gridSpec, List.mem_singleton] at ho
    subst ho
    exact hv.ins_disjoint i hi hov.1
  · intro _ pg hpg ag₀ hag₀ ins outs sc₁ sc₂ houts _ _
    have hag₀_eq := head_gridPoints_eq_replicate_zero rk.gridSpec.arbGrid ag₀ hag₀
    have hlen_pg : pg.length = rk.gridSpec.numParallel := by
      rw [length_of_mem_gridPoints hpg, hparGrid, List.length_take, Nat.min_eq_left hnumPar_le]
    have hlen_ag : ag₀.length = rk.outerGrid.length - rk.gridSpec.numParallel + 1 := by
      rw [hag₀_eq, List.length_replicate, harbGrid, List.length_append, List.length_drop,
        List.length_singleton]
    have hkk : (pg ++ ag₀).getD rk.outerGrid.length 0 = 0 := by
      rw [List.getD_eq_getElem _ _ (by rw [List.length_append, hlen_pg, hlen_ag]; omega),
        List.getElem_append_right (by omega)]
      simp [hag₀_eq]
    simp only [gridSpec, List.length_singleton] at houts
    rcases outs with _ | ⟨o0, _ | ⟨_, _⟩⟩ <;> simp at houts
    simp only [body, hkk, ↓reduceIte, List.cons_append, List.nil_append, List.getD_cons_zero]

noncomputable section

/-- Fold `rk.step` over the first `n` reduction steps at outer coordinate `og` and local coordinate
`loc`, reading input blocks from initial memory `m0`. -/
def accFold (rk : ReductionKernel nAcc) (junk : ℕ → ℝ) (m0 : Mem) (og loc : List ℕ) (n : ℕ) :
    Fin nAcc → ℝ :=
  (List.range n).foldl
    (fun acc t =>
      rk.step og t (rk.scalarPrefetch m0) (loadIns rk.gridSpec junk m0 (og ++ [t])) loc acc)
    rk.init

/-- `accFold` over `0` steps returns `rk.init`. -/
theorem accFold_zero (rk : ReductionKernel nAcc) (junk : ℕ → ℝ) (m0 : Mem) (og loc : List ℕ) :
    rk.accFold junk m0 og loc 0 = rk.init := rfl

/-- `accFold` over `n + 1` steps applies `rk.step` at step `n` to the `n`-step accumulator. -/
theorem accFold_succ (rk : ReductionKernel nAcc) (junk : ℕ → ℝ) (m0 : Mem) (og loc : List ℕ) (n : ℕ) :
    rk.accFold junk m0 og loc (n + 1) =
      rk.step og n (rk.scalarPrefetch m0) (loadIns rk.gridSpec junk m0 (og ++ [n])) loc
        (rk.accFold junk m0 og loc n) := by
  simp [accFold, List.range_succ]

private theorem outSpec_blockIdx_append (rk : ReductionKernel nAcc) {og : List ℕ}
    (hlen : og.length = rk.outerGrid.length) (t : ℕ) (sc : ℕ → ℤ) :
    rk.outSpec.blockIdx (og ++ [t]) sc = rk.outBlockIdx og sc := by
  simp [BlockSpec.blockIdx, outSpec, outBlockIdx, ← hlen]

private theorem getD_append_singleton {og : List ℕ} {n t : ℕ} (hlen : og.length = n) :
    (og ++ [t]).getD n 0 = t := by
  rw [List.getD_eq_getElem _ _ (by simp [hlen]), List.getElem_append_right (by omega)]
  simp [hlen]

/-- First reduction step (`t = 0`) at outer coordinate `og`: flushes the previous output slot once,
resets all scratch accumulators to `init`, and applies step `0` to inputs read from `m0`. -/
private theorem step_zero (rk : ReductionKernel nAcc) (junk : ℕ → ℝ) (m0 m : Mem)
    (hm : ∀ b, b ≠ rk.outArr.buf → ∀ a, m.read b a = m0.read b a)
    (hdisj : ∀ p ∈ rk.inSpecs, p.1.buf ≠ rk.outArr.buf)
    (s0 : OutSlot) (ss : List OutSlot) {og : List ℕ} (hlen : og.length = rk.outerGrid.length) :
    ∃ (O : List ℕ → ℝ) (A : List ℕ → Fin nAcc → ℝ),
      stepGrid rk.gridSpec rk.body (rk.scalarPrefetch m0) junk (m, s0 :: ss) (og ++ [0]) =
        (flushOne rk.outPair (rk.outBlockIdx og (rk.scalarPrefetch m0)) s0 m,
          some (rk.outBlockIdx og (rk.scalarPrefetch m0), O) ::
            List.ofFn fun j : Fin nAcc => some ([], fun loc => A loc j)) ∧
      (∀ loc, A loc = rk.accFold junk m0 og loc 1) ∧
      (1 = rk.nk → ∀ loc, O loc = rk.finalize og (rk.scalarPrefetch m0)
        (loadIns rk.gridSpec junk m0 (og ++ [0])) loc (A loc)) := by
  set sc := rk.scalarPrefetch m0
  set idx := rk.outBlockIdx og sc
  set m1 := flushOne rk.outPair idx s0 m with hm1
  set tag := flatCoord (rk.outerGrid ++ [rk.nk]) (og ++ [0]) + 1
  have hflush : (flushAndPrep (og ++ [0]) sc (junkAt junk (Nat.pair tag 0))
      0 [rk.outPair] (s0 :: ss) m).1 = m1 := by
    rcases s0 with _ | ⟨oi, ov⟩ <;>
      simp [flushAndPrep, hm1, flushOne, outSpec_blockIdx_append rk hlen 0 sc, outPair, idx]
  have hins : (rk.inSpecs.mapIdx fun i p =>
      loadBlock p.1 p.2.blockShape (p.2.blockIdx (og ++ [0]) sc)
        (junkAt junk (Nat.pair tag (i + 1))) m1) =
      loadIns rk.gridSpec junk m0 (og ++ [0]) := by
    apply loadIns_congr rk.gridSpec junk sc m1 m0 (og ++ [0])
    intro p hp a
    rw [hm1, flushOne_read_other_buf _ _ _ _ _ a (hdisj p hp)]
    exact hm p.1.buf (hdisj p hp) a
  set outsPrev := (flushAndPrep (og ++ [0]) sc (junkAt junk (Nat.pair tag 0)) 0 [rk.outPair] (s0 :: ss) m).2.2
  set scratchPrev : List (List ℕ → ℝ) := ((ss.zip (List.replicate nAcc rk.outBlockShape)).mapIdx fun j p =>
    fun loc => match p.1 with
      | some (_, v) =>
        if inBounds p.2 loc then v loc
        else junkAt (junkAt junk (Nat.pair tag 0)) (1 + j) (Encodable.encode loc)
      | none => junkAt (junkAt junk (Nat.pair 0 0)) (1 + j) (Encodable.encode loc))
  set newAcc : List ℕ → Fin nAcc → ℝ := fun loc =>
    rk.step og 0 sc (loadIns rk.gridSpec junk m0 (og ++ [0])) loc rk.init
  set O : List ℕ → ℝ := fun loc =>
    if (0 : ℕ) = rk.nk - 1 then
      rk.finalize og sc (loadIns rk.gridSpec junk m0 (og ++ [0])) loc (newAcc loc)
    else ((outsPrev ++ scratchPrev).getD 0 fun _ => 0) loc
  refine ⟨O, newAcc, ?_, ?_, ?_⟩
  · simp only [stepGrid, gridSpec]
    change ((flushAndPrep (og ++ [0]) sc (junkAt junk (Nat.pair tag 0)) 0 [rk.outPair] (s0 :: ss) m).1, _) = _
    rw [hflush, hins]
    simp only [makeSlots, body, ← hlen, List.take_left, getD_append_singleton (og := og) rfl,
      ↓reduceIte, List.length_singleton, List.drop_succ_cons, List.drop_zero, List.map_ofFn,
      Function.comp_def, flushAndPrep_idxs, List.map_cons, List.map_nil, outPair,
      outSpec_blockIdx_append rk hlen 0 sc, List.cons_append, List.nil_append]
    rfl
  · intro loc
    simp [newAcc, accFold, List.range_succ, sc]
  · intro h1 loc
    simp only [O, eq_true (show (0 : ℕ) = rk.nk - 1 by omega), ↓reduceIte]

/-- Subsequent reduction step (`0 < t`) at outer coordinate `og`: memory is unchanged because the
output block index repeats, and each scratch accumulator reads its previous value inside
`outBlockShape`. -/
private theorem step_succ (rk : ReductionKernel nAcc) (junk : ℕ → ℝ) (m0 m : Mem)
    (hm : ∀ b, b ≠ rk.outArr.buf → ∀ a, m.read b a = m0.read b a)
    (hdisj : ∀ p ∈ rk.inSpecs, p.1.buf ≠ rk.outArr.buf)
    (O : List ℕ → ℝ) (A : List ℕ → Fin nAcc → ℝ)
    {og : List ℕ} (hlen : og.length = rk.outerGrid.length) {t : ℕ} (ht : 0 < t) :
    ∃ (O' : List ℕ → ℝ) (A' : List ℕ → Fin nAcc → ℝ),
      stepGrid rk.gridSpec rk.body (rk.scalarPrefetch m0) junk
          (m, some (rk.outBlockIdx og (rk.scalarPrefetch m0), O) ::
            List.ofFn fun j : Fin nAcc => some ([], fun loc => A loc j)) (og ++ [t]) =
        (m, some (rk.outBlockIdx og (rk.scalarPrefetch m0), O') ::
          List.ofFn fun j : Fin nAcc => some ([], fun loc => A' loc j)) ∧
      (∀ loc, inBounds rk.outBlockShape loc = true →
        A' loc = rk.step og t (rk.scalarPrefetch m0)
          (loadIns rk.gridSpec junk m0 (og ++ [t])) loc (A loc)) ∧
      (t + 1 = rk.nk → ∀ loc, O' loc = rk.finalize og (rk.scalarPrefetch m0)
        (loadIns rk.gridSpec junk m0 (og ++ [t])) loc (A' loc)) := by
  set sc := rk.scalarPrefetch m0
  set idx := rk.outBlockIdx og sc
  set ss : List OutSlot := List.ofFn fun j : Fin nAcc => some ([], fun loc => A loc j)
  set tag := flatCoord (rk.outerGrid ++ [rk.nk]) (og ++ [t]) + 1
  set jtag := junkAt junk (Nat.pair tag 0)
  have hflush : (flushAndPrep (og ++ [t]) sc jtag 0 [rk.outPair] (some (idx, O) :: ss) m).1 = m := by
    simp [flushAndPrep, outSpec_blockIdx_append rk hlen t sc, outPair, idx]
  have hins : (rk.inSpecs.mapIdx fun i p =>
      loadBlock p.1 p.2.blockShape (p.2.blockIdx (og ++ [t]) sc)
        (junkAt junk (Nat.pair tag (i + 1))) m) =
      loadIns rk.gridSpec junk m0 (og ++ [t]) := by
    apply loadIns_congr rk.gridSpec junk sc m m0 (og ++ [t])
    intro p hp a
    exact hm p.1.buf (hdisj p hp) a
  set scratchPrev : List (List ℕ → ℝ) := ((ss.zip (List.replicate nAcc rk.outBlockShape)).mapIdx fun j p =>
    fun loc => match p.1 with
      | some (_, v) => if inBounds p.2 loc then v loc else junkAt jtag (1 + j) (Encodable.encode loc)
      | none => junkAt (junkAt junk (Nat.pair 0 0)) (1 + j) (Encodable.encode loc))
  have hscratch : ∀ (j : Fin nAcc) (loc : List ℕ), inBounds rk.outBlockShape loc = true →
      scratchPrev.getD j.val (fun _ => 0) loc = A loc j := by
    intro j loc hloc
    have hlt : j.val < scratchPrev.length := by
      simp [scratchPrev, ss]
    rw [List.getD_eq_getElem _ _ hlt]
    simp [scratchPrev, ss, hloc]
  set outsPrev := (flushAndPrep (og ++ [t]) sc jtag 0 [rk.outPair] (some (idx, O) :: ss) m).2.2
  have hprev_eq : ∀ (j : Fin nAcc) (loc : List ℕ), inBounds rk.outBlockShape loc = true →
      ((outsPrev ++ scratchPrev).getD (j.val + 1) fun _ => 0) loc = A loc j := by
    intro j loc hloc
    have houts : ∃ o0, outsPrev = [o0] := ⟨_, rfl⟩
    obtain ⟨o0, ho0⟩ := houts
    rw [ho0, List.cons_append, List.nil_append, List.getD_cons_succ]
    exact hscratch j loc hloc
  set newAcc : List ℕ → Fin nAcc → ℝ := fun loc =>
    rk.step og t sc (loadIns rk.gridSpec junk m0 (og ++ [t])) loc fun j =>
      if t = 0 then rk.init j else ((outsPrev ++ scratchPrev).getD (j.val + 1) fun _ => 0) loc
  set O' : List ℕ → ℝ := fun loc =>
    if t = rk.nk - 1 then
      rk.finalize og sc (loadIns rk.gridSpec junk m0 (og ++ [t])) loc (newAcc loc)
    else ((outsPrev ++ scratchPrev).getD 0 fun _ => 0) loc
  refine ⟨O', newAcc, ?_, ?_, ?_⟩
  · simp only [stepGrid, gridSpec]
    change ((flushAndPrep (og ++ [t]) sc jtag 0 [rk.outPair] (some (idx, O) :: ss) m).1, _) = _
    rw [hflush, hins]
    simp only [makeSlots, body, ← hlen, List.take_left, getD_append_singleton (og := og) rfl,
      List.length_singleton, List.drop_succ_cons, List.drop_zero, List.map_ofFn,
      Function.comp_def, flushAndPrep_idxs, List.map_cons, List.map_nil, outPair,
      outSpec_blockIdx_append rk hlen t sc, List.cons_append, List.nil_append]
    rfl
  · intro loc hloc
    simp only [newAcc, Nat.pos_iff_ne_zero.mp ht, ↓reduceIte]
    congr 1
    funext j
    exact hprev_eq j loc hloc
  · intro hnk loc
    simp only [O', eq_true (show t = rk.nk - 1 by omega), ↓reduceIte]

/-- Running the first `n` (`0 < n ≤ nk`) reduction steps of outer tile `og`: memory gets the
previous tile's write-back once, each scratch accumulator holds `accFold`, and at `n = nk` the
output block holds `finalize` of `accFold`. -/
private theorem run_spec (rk : ReductionKernel nAcc) (junk : ℕ → ℝ) (m0 m : Mem)
    (hm : ∀ b, b ≠ rk.outArr.buf → ∀ a, m.read b a = m0.read b a)
    (hdisj : ∀ p ∈ rk.inSpecs, p.1.buf ≠ rk.outArr.buf)
    (s0 : OutSlot) (ss : List OutSlot) {og : List ℕ} (hlen : og.length = rk.outerGrid.length)
    (n : ℕ) (hn : 0 < n) (hle : n ≤ rk.nk) :
    ∃ (O : List ℕ → ℝ) (A : List ℕ → Fin nAcc → ℝ),
      ((List.range n).map fun t => og ++ [t]).foldl
          (stepGrid rk.gridSpec rk.body (rk.scalarPrefetch m0) junk) (m, s0 :: ss) =
        (flushOne rk.outPair (rk.outBlockIdx og (rk.scalarPrefetch m0)) s0 m,
          some (rk.outBlockIdx og (rk.scalarPrefetch m0), O) ::
            List.ofFn fun j : Fin nAcc => some ([], fun loc => A loc j)) ∧
      (∀ loc, inBounds rk.outBlockShape loc = true → A loc = rk.accFold junk m0 og loc n) ∧
      (n = rk.nk → ∀ loc, O loc = rk.finalize og (rk.scalarPrefetch m0)
        (loadIns rk.gridSpec junk m0 (og ++ [n - 1])) loc (A loc)) := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [List.range_succ, List.map_append, List.foldl_append]
    rcases Nat.eq_zero_or_pos n with rfl | hpos
    · simp only [List.range_zero, List.map_nil, List.foldl_nil, List.map_cons, List.foldl_cons]
      obtain ⟨O, A, hst, hA, hO⟩ := step_zero rk junk m0 m hm hdisj s0 ss hlen
      exact ⟨O, A, hst, fun loc _ => hA loc, hO⟩
    · obtain ⟨O, A, hst, hA, -⟩ := ih hpos (by omega)
      rw [hst]
      simp only [List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil]
      have hm1 : ∀ b, b ≠ rk.outArr.buf → ∀ a,
          (flushOne rk.outPair (rk.outBlockIdx og (rk.scalarPrefetch m0)) s0 m).read b a =
            m0.read b a := fun b hb a => by
        rw [flushOne_read_other_buf _ _ _ _ _ a hb]
        exact hm b hb a
      obtain ⟨O', A', hst', hA', hO'⟩ := step_succ rk junk m0 _ hm1 hdisj O A hlen hpos
      refine ⟨O', A', hst', ?_, by rw [Nat.add_sub_cancel]; exact hO'⟩
      intro loc hloc
      rw [hA' loc hloc, hA loc hloc]
      simp [accFold, List.range_succ]

/-- The specified value of outer tile `og` at local coordinate `loc`. -/
def tileVal (rk : ReductionKernel nAcc) (junk : ℕ → ℝ) (m0 : Mem) (og loc : List ℕ) : ℝ :=
  rk.finalize og (rk.scalarPrefetch m0) (loadIns rk.gridSpec junk m0 (og ++ [rk.nk - 1])) loc
    (rk.accFold junk m0 og loc rk.nk)

/-- Running outer tiles `ogs` in order: non-output buffers keep `m0`, and the output buffer (after
the pending write-back) equals the in-order sequence of `tileVal` block stores. -/
private theorem outer_fold (rk : ReductionKernel nAcc) (junk : ℕ → ℝ) (m0 : Mem)
    (hnk : 0 < rk.nk) (hdisj : ∀ p ∈ rk.inSpecs, p.1.buf ≠ rk.outArr.buf) :
    ∀ (ogs : List (List ℕ)) (st : Mem × List OutSlot) (S : Mem),
      (∀ og ∈ ogs, og.length = rk.outerGrid.length) →
      (∀ b, b ≠ rk.outArr.buf → ∀ a, st.1.read b a = m0.read b a) →
      (∃ s0 ss, st.2 = s0 :: ss ∧ ∀ a,
        (flushFinal rk.outPair s0 st.1).read rk.outArr.buf a = S.read rk.outArr.buf a) →
      let st' := ogs.foldl (fun st og =>
        ((List.range rk.nk).map fun t => og ++ [t]).foldl
          (stepGrid rk.gridSpec rk.body (rk.scalarPrefetch m0) junk) st) st
      (∀ b, b ≠ rk.outArr.buf → ∀ a, st'.1.read b a = m0.read b a) ∧
      ∃ s0 ss, st'.2 = s0 :: ss ∧ ∀ a,
        (flushFinal rk.outPair s0 st'.1).read rk.outArr.buf a =
          (ogs.foldl (fun S og =>
            storeBlock rk.outArr rk.outBlockShape (rk.outBlockIdx og (rk.scalarPrefetch m0))
              (rk.tileVal junk m0 og) S) S).read rk.outArr.buf a
  | [], st, S, _, hframe, hslots => ⟨hframe, hslots⟩
  | og :: rest, (m, slots), S, hlen, hframe, ⟨s0, ss, hst, hval⟩ => by
    simp only at hst hframe hval
    subst hst
    have hog : og.length = rk.outerGrid.length := hlen og (by simp)
    have hrest : ∀ og' ∈ rest, og'.length = rk.outerGrid.length :=
      fun og' hog' => hlen og' (by simp [hog'])
    obtain ⟨O, A, hrun, hA, hO⟩ := run_spec rk junk m0 m hframe hdisj s0 ss hog rk.nk hnk le_rfl
    simp only [List.foldl_cons]
    rw [hrun]
    apply outer_fold rk junk m0 hnk hdisj rest _ _ hrest
    · intro b hb a
      rw [flushOne_read_other_buf _ _ _ _ _ a hb]
      exact hframe b hb a
    · refine ⟨_, _, rfl, ?_⟩
      intro a
      simp only [flushFinal, outPair, outSpec]
      cases hin : addrInBlock rk.outArr rk.outBlockShape (rk.outBlockIdx og (rk.scalarPrefetch m0)) a
      · rw [storeBlock_read_outside _ _ _ _ _ _ _ (Or.inr hin),
          storeBlock_read_outside _ _ _ _ _ _ _ (Or.inr hin)]
        exact (flushOne_read_eq_flushFinal rk.outPair _ _ _ _ hin).trans (hval a)
      · rw [storeBlock_read_inside _ _ _ _ _ _ hin, storeBlock_read_inside _ _ _ _ _ _ hin]
        have hcoord : coordInBlock rk.outBlockShape (rk.outBlockIdx og (rk.scalarPrefetch m0))
            (unflatCoord rk.outArr.shape a) = true := by
          simp only [addrInBlock, Bool.and_eq_true] at hin
          exact hin.2
        have hloc : inBounds rk.outBlockShape
            (localCoord rk.outBlockShape (rk.outBlockIdx og (rk.scalarPrefetch m0))
              (unflatCoord rk.outArr.shape a)) = true :=
          inBounds_localCoord_of_coordInBlock hcoord
        rw [hO rfl, hA _ hloc]
        rfl

/-- General correctness of a reduction kernel: under `Valid m0`, `pallasCall` succeeds, every
output element covered by an outer grid point `og` equals `finalize` (at the last step's input
blocks) of `accFold` over inputs read from `m0`, and every buffer other than `outArr.buf` is
unchanged. -/
theorem correct (rk : ReductionKernel nAcc) {m0 : Mem} (hv : rk.Valid m0) (junk : ℕ → ℝ) :
    ∃ m', pallasCall rk.gridSpec rk.body junk m0 = some m' ∧
      (∀ og ∈ gridPoints rk.outerGrid, ∀ a : ℕ,
        addrInBlock rk.outArr rk.outBlockShape (rk.outBlockIdx og (rk.scalarPrefetch m0)) a = true →
          m'.read rk.outArr.buf a =
            rk.finalize og (rk.scalarPrefetch m0)
              (loadIns rk.gridSpec junk m0 (og ++ [rk.nk - 1]))
              (localCoord rk.outBlockShape (rk.outBlockIdx og (rk.scalarPrefetch m0))
                (unflatCoord rk.outArr.shape a))
              (rk.accFold junk m0 og
                (localCoord rk.outBlockShape (rk.outBlockIdx og (rk.scalarPrefetch m0))
                  (unflatCoord rk.outArr.shape a))
                rk.nk)) ∧
      Mem.agreeOutside (Mem.bufFootprint rk.outArr.buf) m0 m' := by
  refine ⟨pallasExec rk.gridSpec rk.body junk m0,
    pallasCall_of_wellDefined _ (rk.wellDefined hv), ?_, ?_⟩
  · intro og hog a hin
    set sc := rk.scalarPrefetch m0
    set mInit := initOutMem rk.gridSpec (junkAt junk (Nat.pair 0 0)) m0
    have hinit : ∀ b, b ≠ rk.outArr.buf → ∀ a', mInit.read b a' = m0.read b a' := by
      intro b hb a'
      exact initOutMemAux_read_other_buf _ _ b a' 0 _ m0
        (by intro x hx'; simp only [gridSpec, outPair, List.mem_singleton] at hx'; subst hx'; exact Ne.symm hb)
    obtain ⟨-, s0, ss, hst, hval⟩ := outer_fold rk junk m0 hv.nk_pos hv.ins_disjoint
      (gridPoints rk.outerGrid) (mInit, none :: List.replicate nAcc none) mInit
      (fun og' hog' => length_of_mem_gridPoints hog') hinit
      ⟨none, List.replicate nAcc none, rfl, fun _ => rfl⟩
    have hexec : pallasExec rk.gridSpec rk.body junk m0 =
        flushAll rk.gridSpec.outSpecs
          ((gridPoints rk.outerGrid).foldl (fun st og' =>
            ((List.range rk.nk).map fun t => og' ++ [t]).foldl
              (stepGrid rk.gridSpec rk.body sc junk) st)
            (mInit, none :: List.replicate nAcc none)).2
          ((gridPoints rk.outerGrid).foldl (fun st og' =>
            ((List.range rk.nk).map fun t => og' ++ [t]).foldl
              (stepGrid rk.gridSpec rk.body sc junk) st)
            (mInit, none :: List.replicate nAcc none)).1 := by
      simp only [pallasExec, gridSpec, gridPoints_append_singleton, List.foldl_flatMap,
        List.map_cons, List.map_nil, List.map_replicate, List.cons_append, List.nil_append]
      rfl
    have hflush_at := flushAll_read_at rk.gridSpec.outSpecs
      ((gridPoints rk.outerGrid).foldl (fun st og' =>
        ((List.range rk.nk).map fun t => og' ++ [t]).foldl
          (stepGrid rk.gridSpec rk.body sc junk) st)
        (mInit, none :: List.replicate nAcc none)).2
      ((gridPoints rk.outerGrid).foldl (fun st og' =>
        ((List.range rk.nk).map fun t => og' ++ [t]).foldl
          (stepGrid rk.gridSpec rk.body sc junk) st)
        (mInit, none :: List.replicate nAcc none)).1
      0 rk.outPair (by simp [gridSpec]) (by simp [gridSpec]) a
    have hval_a := hval a
    simp only [outPair] at hflush_at hval_a
    rw [hexec, hflush_at, hst, List.getD_cons_zero]
    dsimp only [sc]
    rw [hval_a]
    obtain ⟨l₁, l₂, hsplit⟩ := List.append_of_mem hog
    have hnd := gridPoints_nodup rk.outerGrid
    rw [hsplit] at hnd
    have hnot : og ∉ l₂ := (List.nodup_cons.mp (List.nodup_append.mp hnd).2.1).1
    have hl₂_mem : ∀ og' ∈ l₂, og' ∈ gridPoints rk.outerGrid := by
      intro og' hog'
      rw [hsplit]
      exact List.mem_append_right _ (List.mem_cons_of_mem _ hog')
    rw [hsplit, foldl_storeBlock_read_last rk.outArr rk.outBlockShape
      (fun og' => rk.outBlockIdx og' sc) (fun og' => rk.tileVal junk m0 og') a l₁ l₂ og mInit hin]
    · rfl
    · intro og' hog'
      cases h : addrInBlock rk.outArr rk.outBlockShape (rk.outBlockIdx og' sc) a with
      | false => rfl
      | true =>
        exfalso
        have hbi : rk.outBlockIdx og' sc = rk.outBlockIdx og sc := addrInBlock_unique h hin
        have hidx : rk.outIndexMap og' sc = rk.outIndexMap og sc :=
          map_toNat_inj (blockOk_nonneg (hv.out_blocks_ok og' (hl₂_mem og' hog')))
            (blockOk_nonneg (hv.out_blocks_ok og hog)) hbi
        have heq : og' = og := hv.out_inj og' (hl₂_mem og' hog') og hog hidx
        exact hnot (heq ▸ hog')
  · refine Mem.agreeOutside_mono ?_
      (pallasExec_agreeOutside_outBufs rk.gridSpec rk.body junk m0)
    rintro ⟨b, a⟩ ⟨x, hx, hbuf⟩
    simp only [gridSpec, outPair, List.mem_singleton] at hx
    subst hx
    simpa [Mem.bufFootprint] using hbuf.symm

end

end ReductionKernel

end TpuKernelLean.Pallas
