import TpuKernelLean.Pallas.ReductionKernel

/-!
Tests of `ReductionKernel` on non-matmul shapes:
1. `rowSumKernel`: 1-D outer grid `[cdiv M bm]`, 1-D output `[M]`, 1 input `[M, nk * BK]`,
   1 scratch accumulator reducing along axis 1.
2. `colStatsKernel`: 1-D outer grid `[cdiv N bn]`, 1-D output `[N]`, 1 input `[nr * BR, N]`,
   2 scratch accumulators (`∑ x` and `∑ x^2`) reducing along axis 0 with nonlinear `finalize`.
-/

namespace TpuKernelLean.Tests.Reduction

open TpuKernelLean.Pallas

private theorem mem_outerGrid_one_iff {a : ℕ} {og : List ℕ} :
    og ∈ gridPoints [a] ↔ ∃ i, og = [i] ∧ i < a := by
  rw [mem_gridPoints]
  rcases og with _ | ⟨i, _ | ⟨u, g⟩⟩ <;> simp [inBounds]

/-! ### 1. Row-sum reduction (`[M, nk * BK]` → `[M]`, 1 accumulator) -/

/-- Row-sum reduction kernel over `x : [M, nk * BK]` tiled by `[bm, BK]`, writing `[M]` in `[bm]`
blocks over grid `[cdiv M bm, nk]`. -/
noncomputable def rowSumKernel (M bm BK nk : ℕ) (xBuf oBuf : BufId) : ReductionKernel 1 where
  outerGrid := [(M + bm - 1) / bm]
  nk := nk
  inSpecs := [(⟨xBuf, [M, nk * BK]⟩,
    { blockShape := [bm, BK]
      indexMap := fun g _ => [((g.getD 0 0 : ℕ) : ℤ), ((g.getD 1 0 : ℕ) : ℤ)] })]
  outArr := ⟨oBuf, [M]⟩
  outBlockShape := [bm]
  outIndexMap := fun og _ => [((og.getD 0 0 : ℕ) : ℤ)]
  init := fun _ => 0
  step := fun _ _ _ ins loc prevAcc _ =>
    prevAcc 0 + ∑ u : Fin BK, (ins.getD 0 fun _ => 0) [loc.getD 0 0, u.val]
  finalize := fun _ _ _ _ acc => acc 0

/-- `rowSumKernel` is valid whenever `M`, block sizes and reduction length are positive and buffers
do not alias. -/
theorem rowSumKernel_valid (M bm BK nk : ℕ) (hM : 0 < M) (hbm : 0 < bm) (hBK : 0 < BK) (hnk : 0 < nk)
    (xBuf oBuf : BufId) (hx : oBuf ≠ xBuf) (m0 : Mem) :
    (rowSumKernel M bm BK nk xBuf oBuf).Valid m0 where
  outer_grid_pos := by
    intro d hd
    simp only [rowSumKernel, List.mem_singleton] at hd
    subst hd
    exact cdiv_pos hM hbm
  nk_pos := hnk
  in_blocks_ok := by
    intro og hog t ht p hp
    obtain ⟨i, rfl, hi⟩ := mem_outerGrid_one_iff.mp hog
    have h1 : (i : ℤ) * bm < M := by exact_mod_cast lt_of_lt_cdiv hbm hi
    have ht' : t < nk := ht
    simp only [rowSumKernel, List.mem_singleton] at hp
    subst hp
    simp [blockOk, hbm, hBK, h1, ht']
  out_blocks_ok := by
    intro og hog
    obtain ⟨i, rfl, hi⟩ := mem_outerGrid_one_iff.mp hog
    have h1 : (i : ℤ) * bm < M := by exact_mod_cast lt_of_lt_cdiv hbm hi
    simp [rowSumKernel, blockOk, hbm, h1]
  out_inj := by
    intro og₁ h₁ og₂ h₂ heq
    obtain ⟨i₁, rfl, _⟩ := mem_outerGrid_one_iff.mp h₁
    obtain ⟨i₂, rfl, _⟩ := mem_outerGrid_one_iff.mp h₂
    simp only [rowSumKernel, List.getD_cons_zero, List.cons.injEq, Nat.cast_inj, and_true] at heq
    simp [heq]
  ins_disjoint := by
    intro p hp
    simp only [rowSumKernel, List.mem_singleton] at hp
    subst hp
    exact hx.symm

/-- Row-sum correctness: every row `r < M` sums all `nk * BK` entries of row `r` of `x`. -/
theorem rowSum_correct (M bm BK nk : ℕ) (hM : 0 < M) (hbm : 0 < bm) (hBK : 0 < BK) (hnk : 0 < nk)
    (xBuf oBuf : BufId) (hx : oBuf ≠ xBuf) (junk : ℕ → ℝ) (m0 : Mem) :
    let rk := rowSumKernel M bm BK nk xBuf oBuf
    ∃ m', pallasCall rk.gridSpec rk.body junk m0 = some m' ∧
      (∀ r : Fin M, m'.read oBuf r.val =
        ∑ k ∈ Finset.range (nk * BK), m0.read xBuf (r.val * (nk * BK) + k)) ∧
      Mem.agreeOutside (Mem.bufFootprint oBuf) m0 m' := by
  dsimp only
  set rk := rowSumKernel M bm BK nk xBuf oBuf
  obtain ⟨m', hcall, hval, hframe⟩ :=
    rk.correct (rowSumKernel_valid M bm BK nk hM hbm hBK hnk xBuf oBuf hx m0) junk
  refine ⟨m', hcall, ?_, hframe⟩
  intro r
  have hog : [r.val / bm] ∈ gridPoints rk.outerGrid :=
    mem_outerGrid_one_iff.mpr ⟨_, rfl, div_lt_cdiv hbm r.isLt⟩
  have hin : addrInBlock rk.outArr rk.outBlockShape
      (rk.outBlockIdx [r.val / bm] (rk.scalarPrefetch m0)) r.val = true := by
    simp only [rk, rowSumKernel, ReductionKernel.outBlockIdx, List.getD_cons_zero, List.map_cons,
      List.map_nil, Int.toNat_natCast]
    rw [addrInBlock_one hbm r.isLt]
    simp
  refine (hval [r.val / bm] hog r.val hin).trans ?_
  have hr' : r.val / bm * bm + (r.val - r.val / bm * bm) = r.val := by
    have := Nat.div_mul_le_self r.val bm; omega
  have hrl : r.val - r.val / bm * bm < bm := by have := Nat.lt_div_mul_add (a := r.val) hbm; omega
  have hloc : localCoord rk.outBlockShape
      (rk.outBlockIdx [r.val / bm] (rk.scalarPrefetch m0))
      (unflatCoord rk.outArr.shape r.val) = [r.val - r.val / bm * bm] := by
    simp only [rk, rowSumKernel, ReductionKernel.outBlockIdx, List.getD_cons_zero, List.map_cons,
      List.map_nil, Int.toNat_natCast, unflatCoord_one, localCoord]
  rw [hloc]
  change ((rowSumKernel M bm BK nk xBuf oBuf).accFold junk m0
    [r.val / bm] [r.val - r.val / bm * bm] nk) 0 = _
  suffices ∀ n ≤ nk,
      ((rowSumKernel M bm BK nk xBuf oBuf).accFold junk m0
        [r.val / bm] [r.val - r.val / bm * bm] n) 0 =
        ∑ k ∈ Finset.range (n * BK), m0.read xBuf (r.val * (nk * BK) + k) from
    this nk le_rfl
  intro n hn
  induction n with
  | zero => simp [ReductionKernel.accFold_zero, rowSumKernel]
  | succ n ih =>
    rw [ReductionKernel.accFold_succ]
    have hstep : (rowSumKernel M bm BK nk xBuf oBuf).step [r.val / bm] n
        ((rowSumKernel M bm BK nk xBuf oBuf).scalarPrefetch m0)
        (loadIns (rowSumKernel M bm BK nk xBuf oBuf).gridSpec junk m0 ([r.val / bm] ++ [n]))
        [r.val - r.val / bm * bm]
        ((rowSumKernel M bm BK nk xBuf oBuf).accFold junk m0 [r.val / bm]
          [r.val - r.val / bm * bm] n) 0 =
        ((rowSumKernel M bm BK nk xBuf oBuf).accFold junk m0 [r.val / bm]
            [r.val - r.val / bm * bm] n 0) +
          ∑ u : Fin BK, ((loadIns (rowSumKernel M bm BK nk xBuf oBuf).gridSpec junk m0
            ([r.val / bm] ++ [n])).getD 0 fun _ => 0) [r.val - r.val / bm * bm, u.val] := rfl
    rw [hstep, ih (by omega), Nat.succ_mul, Finset.sum_range_add]
    congr 1
    rw [← Fin.sum_univ_eq_sum_range (fun u => m0.read xBuf (r.val * (nk * BK) + (n * BK + u)))]
    refine Finset.sum_congr rfl fun u _ => ?_
    have hk : n * BK + u.val < nk * BK := by nlinarith [u.isLt]
    let xBs : BlockSpec :=
      { blockShape := [bm, BK]
        indexMap := fun g _ => [((g.getD 0 0 : ℕ) : ℤ), ((g.getD 1 0 : ℕ) : ℤ)] }
    have hidx : xBs.blockIdx ([r.val / bm] ++ [n])
        ((rowSumKernel M bm BK nk xBuf oBuf).gridSpec.scalarPrefetch m0) = [r.val / bm, n] := by
      simp only [xBs, BlockSpec.blockIdx, List.cons_append, List.nil_append, List.getD_cons_zero,
        List.getD_cons_succ, List.map_cons, List.map_nil, Int.toNat_natCast]
    rw [loadIns_getD (rowSumKernel M bm BK nk xBuf oBuf).gridSpec junk m0 _ 0
        (⟨xBuf, [M, nk * BK]⟩, xBs) (by simp [ReductionKernel.gridSpec, rowSumKernel, xBs]),
      hidx, loadBlock_eq_of_inBounds]
    · simp [xBs, globalCoord, flatCoord, hr']
    · simp [xBs, inBounds, hrl, u.isLt]
    · simp [xBs, globalCoord, inBounds, hr', r.isLt, hk]

/-! ### 2. Two-accumulator column reduction (`[nr * BR, N]` → `[N]`, 2 accumulators) -/

/-- Column reduction kernel with two VMEM scratch accumulators (`∑ x` and `∑ x^2`) over
`x : [nr * BR, N]`, writing `∑ x^2 - (∑ x)^2` to `[N]` over grid `[cdiv N bn, nr]`. -/
noncomputable def colStatsKernel (N bn BR nr : ℕ) (xBuf oBuf : BufId) : ReductionKernel 2 where
  outerGrid := [(N + bn - 1) / bn]
  nk := nr
  inSpecs := [(⟨xBuf, [nr * BR, N]⟩,
    { blockShape := [BR, bn]
      indexMap := fun g _ => [((g.getD 1 0 : ℕ) : ℤ), ((g.getD 0 0 : ℕ) : ℤ)] })]
  outArr := ⟨oBuf, [N]⟩
  outBlockShape := [bn]
  outIndexMap := fun og _ => [((og.getD 0 0 : ℕ) : ℤ)]
  init := fun _ => 0
  step := fun _ _ _ ins loc prevAcc j =>
    if j.val = 0 then
      prevAcc 0 + ∑ u : Fin BR, (ins.getD 0 fun _ => 0) [u.val, loc.getD 0 0]
    else
      prevAcc 1 + ∑ u : Fin BR, ((ins.getD 0 fun _ => 0) [u.val, loc.getD 0 0]) ^ 2
  finalize := fun _ _ _ _ acc => acc 1 - (acc 0) ^ 2

/-- `colStatsKernel` is valid whenever `N`, block sizes and reduction length are positive and buffers
do not alias. -/
theorem colStatsKernel_valid (N bn BR nr : ℕ) (hN : 0 < N) (hbn : 0 < bn) (hBR : 0 < BR) (hnr : 0 < nr)
    (xBuf oBuf : BufId) (hx : oBuf ≠ xBuf) (m0 : Mem) :
    (colStatsKernel N bn BR nr xBuf oBuf).Valid m0 where
  outer_grid_pos := by
    intro d hd
    simp only [colStatsKernel, List.mem_singleton] at hd
    subst hd
    exact cdiv_pos hN hbn
  nk_pos := hnr
  in_blocks_ok := by
    intro og hog t ht p hp
    obtain ⟨j, rfl, hj⟩ := mem_outerGrid_one_iff.mp hog
    have h1 : (j : ℤ) * bn < N := by exact_mod_cast lt_of_lt_cdiv hbn hj
    have ht' : t < nr := ht
    simp only [colStatsKernel, List.mem_singleton] at hp
    subst hp
    simp [blockOk, hbn, hBR, h1, ht']
  out_blocks_ok := by
    intro og hog
    obtain ⟨j, rfl, hj⟩ := mem_outerGrid_one_iff.mp hog
    have h1 : (j : ℤ) * bn < N := by exact_mod_cast lt_of_lt_cdiv hbn hj
    simp [colStatsKernel, blockOk, hbn, h1]
  out_inj := by
    intro og₁ h₁ og₂ h₂ heq
    obtain ⟨j₁, rfl, _⟩ := mem_outerGrid_one_iff.mp h₁
    obtain ⟨j₂, rfl, _⟩ := mem_outerGrid_one_iff.mp h₂
    simp only [colStatsKernel, List.getD_cons_zero, List.cons.injEq, Nat.cast_inj, and_true] at heq
    simp [heq]
  ins_disjoint := by
    intro p hp
    simp only [colStatsKernel, List.mem_singleton] at hp
    subst hp
    exact hx.symm

/-- Column reduction correctness: every column `c < N` outputs `∑_r x[r, c]^2 - (∑_r x[r, c])^2`,
exercising two independent scratch accumulators and axis-0 reduction. -/
theorem colStats_correct (N bn BR nr : ℕ) (hN : 0 < N) (hbn : 0 < bn) (hBR : 0 < BR) (hnr : 0 < nr)
    (xBuf oBuf : BufId) (hx : oBuf ≠ xBuf) (junk : ℕ → ℝ) (m0 : Mem) :
    let rk := colStatsKernel N bn BR nr xBuf oBuf
    ∃ m', pallasCall rk.gridSpec rk.body junk m0 = some m' ∧
      (∀ c : Fin N, m'.read oBuf c.val =
        (∑ r ∈ Finset.range (nr * BR), (m0.read xBuf (r * N + c.val)) ^ 2) -
          (∑ r ∈ Finset.range (nr * BR), m0.read xBuf (r * N + c.val)) ^ 2) ∧
      Mem.agreeOutside (Mem.bufFootprint oBuf) m0 m' := by
  dsimp only
  set rk := colStatsKernel N bn BR nr xBuf oBuf
  obtain ⟨m', hcall, hval, hframe⟩ :=
    rk.correct (colStatsKernel_valid N bn BR nr hN hbn hBR hnr xBuf oBuf hx m0) junk
  refine ⟨m', hcall, ?_, hframe⟩
  intro c
  have hog : [c.val / bn] ∈ gridPoints rk.outerGrid :=
    mem_outerGrid_one_iff.mpr ⟨_, rfl, div_lt_cdiv hbn c.isLt⟩
  have hin : addrInBlock rk.outArr rk.outBlockShape
      (rk.outBlockIdx [c.val / bn] (rk.scalarPrefetch m0)) c.val = true := by
    simp only [rk, colStatsKernel, ReductionKernel.outBlockIdx, List.getD_cons_zero, List.map_cons,
      List.map_nil, Int.toNat_natCast]
    rw [addrInBlock_one hbn c.isLt]
    simp
  refine (hval [c.val / bn] hog c.val hin).trans ?_
  have hc' : c.val / bn * bn + (c.val - c.val / bn * bn) = c.val := by
    have := Nat.div_mul_le_self c.val bn; omega
  have hcl : c.val - c.val / bn * bn < bn := by have := Nat.lt_div_mul_add (a := c.val) hbn; omega
  have hloc : localCoord rk.outBlockShape
      (rk.outBlockIdx [c.val / bn] (rk.scalarPrefetch m0))
      (unflatCoord rk.outArr.shape c.val) = [c.val - c.val / bn * bn] := by
    simp only [rk, colStatsKernel, ReductionKernel.outBlockIdx, List.getD_cons_zero, List.map_cons,
      List.map_nil, Int.toNat_natCast, unflatCoord_one, localCoord]
  rw [hloc]
  suffices ∀ n ≤ nr,
      ((colStatsKernel N bn BR nr xBuf oBuf).accFold junk m0
          [c.val / bn] [c.val - c.val / bn * bn] n 0 =
        ∑ r ∈ Finset.range (n * BR), m0.read xBuf (r * N + c.val)) ∧
      ((colStatsKernel N bn BR nr xBuf oBuf).accFold junk m0
          [c.val / bn] [c.val - c.val / bn * bn] n 1 =
        ∑ r ∈ Finset.range (n * BR), (m0.read xBuf (r * N + c.val)) ^ 2) by
    obtain ⟨h0, h1⟩ := this nr le_rfl
    change ((colStatsKernel N bn BR nr xBuf oBuf).accFold junk m0
        [c.val / bn] [c.val - c.val / bn * bn] nr 1) -
      (((colStatsKernel N bn BR nr xBuf oBuf).accFold junk m0
        [c.val / bn] [c.val - c.val / bn * bn] nr 0) ^ 2) = _
    rw [h0, h1]
  intro n hn
  induction n with
  | zero => simp [ReductionKernel.accFold_zero, colStatsKernel]
  | succ n ih =>
    obtain ⟨ih0, ih1⟩ := ih (by omega)
    have hread : ∀ u : Fin BR,
        (loadIns (colStatsKernel N bn BR nr xBuf oBuf).gridSpec junk m0 ([c.val / bn] ++ [n])).getD
          0 (fun _ => 0) [u.val, c.val - c.val / bn * bn] =
          m0.read xBuf ((n * BR + u.val) * N + c.val) := by
      intro u
      have hr : n * BR + u.val < nr * BR := by nlinarith [u.isLt]
      let xBs : BlockSpec :=
        { blockShape := [BR, bn]
          indexMap := fun g _ => [((g.getD 1 0 : ℕ) : ℤ), ((g.getD 0 0 : ℕ) : ℤ)] }
      have hidx : xBs.blockIdx ([c.val / bn] ++ [n])
          ((colStatsKernel N bn BR nr xBuf oBuf).gridSpec.scalarPrefetch m0) = [n, c.val / bn] := by
        simp only [xBs, BlockSpec.blockIdx, List.cons_append, List.nil_append, List.getD_cons_zero,
          List.getD_cons_succ, List.map_cons, List.map_nil, Int.toNat_natCast]
      rw [loadIns_getD (colStatsKernel N bn BR nr xBuf oBuf).gridSpec junk m0 _ 0
          (⟨xBuf, [nr * BR, N]⟩, xBs) (by simp [ReductionKernel.gridSpec, colStatsKernel, xBs]),
        hidx, loadBlock_eq_of_inBounds]
      · simp [xBs, globalCoord, flatCoord, hc']
      · simp [xBs, inBounds, hcl, u.isLt]
      · simp [xBs, globalCoord, inBounds, hc', c.isLt, hr]
    constructor
    · rw [ReductionKernel.accFold_succ]
      have hstep0 : (colStatsKernel N bn BR nr xBuf oBuf).step [c.val / bn] n
          ((colStatsKernel N bn BR nr xBuf oBuf).scalarPrefetch m0)
          (loadIns (colStatsKernel N bn BR nr xBuf oBuf).gridSpec junk m0 ([c.val / bn] ++ [n]))
          [c.val - c.val / bn * bn]
          ((colStatsKernel N bn BR nr xBuf oBuf).accFold junk m0 [c.val / bn]
            [c.val - c.val / bn * bn] n) 0 =
          ((colStatsKernel N bn BR nr xBuf oBuf).accFold junk m0 [c.val / bn]
              [c.val - c.val / bn * bn] n 0) +
            ∑ u : Fin BR, ((loadIns (colStatsKernel N bn BR nr xBuf oBuf).gridSpec junk m0
              ([c.val / bn] ++ [n])).getD 0 fun _ => 0) [u.val, c.val - c.val / bn * bn] := rfl
      rw [hstep0, ih0, Nat.succ_mul, Finset.sum_range_add]
      congr 1
      rw [← Fin.sum_univ_eq_sum_range (fun u => m0.read xBuf ((n * BR + u) * N + c.val))]
      exact Finset.sum_congr rfl fun u _ => hread u
    · rw [ReductionKernel.accFold_succ]
      have hstep1 : (colStatsKernel N bn BR nr xBuf oBuf).step [c.val / bn] n
          ((colStatsKernel N bn BR nr xBuf oBuf).scalarPrefetch m0)
          (loadIns (colStatsKernel N bn BR nr xBuf oBuf).gridSpec junk m0 ([c.val / bn] ++ [n]))
          [c.val - c.val / bn * bn]
          ((colStatsKernel N bn BR nr xBuf oBuf).accFold junk m0 [c.val / bn]
            [c.val - c.val / bn * bn] n) 1 =
          ((colStatsKernel N bn BR nr xBuf oBuf).accFold junk m0 [c.val / bn]
              [c.val - c.val / bn * bn] n 1) +
            ∑ u : Fin BR, (((loadIns (colStatsKernel N bn BR nr xBuf oBuf).gridSpec junk m0
              ([c.val / bn] ++ [n])).getD 0 fun _ => 0) [u.val, c.val - c.val / bn * bn]) ^ 2 := rfl
      rw [hstep1, ih1, Nat.succ_mul, Finset.sum_range_add]
      congr 1
      rw [← Fin.sum_univ_eq_sum_range (fun u => (m0.read xBuf ((n * BR + u) * N + c.val)) ^ 2)]
      exact Finset.sum_congr rfl fun u _ => by rw [hread u]

end TpuKernelLean.Tests.Reduction
