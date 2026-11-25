import TpuKernelLean.Pallas.ReductionKernel

/-!
Verification of the batch-invariant matmul kernel of sglang-torchtpu PR 2299
(`sglang_tpu/rl/true_on_policy/matmul_kernel.py`): `x [M, K] @ w [N, K]^T`, fp32 accumulation
over `K` in `BK`-element steps (512 in the PR), in order, in a VMEM scratch accumulator.

The float operations are opaque (`MOps`), so the theorems hold for every implementation of them.
`matmul_correct` instantiates `ReductionKernel.correct` to show output element `(r, c)` is `cast`
of `add` folded in order over the `K` chunks of row `r` of `x` and row `c` of `w`, starting from
`0`, for every `M`, `N` and block sizes `bm`, `bn`. That is the PR's contract, and
`matmul_batch_invariant` follows. With exact arithmetic (`MOps.exact`) the element is the dot
product of the two rows.

`dimension_semantics=("parallel", "parallel", "arbitrary")` is modelled on `gridSpec`: every
`(i, j)` writes its own output block, reads no output buffer, and resets the scratch accumulator at
`k == 0`, so any multi-core split or reordering agrees with sequential execution
(`Pallas.pallasExecPar_eq_pallasExec`).
-/

namespace TpuKernelLean.Examples.TrueOnPolicyMatmul

open TpuKernelLean.Pallas

/-- Opaque TPU float operations of `_mm_kernel`. A theorem proved for every `MOps` holds bit for
bit on the TPU provided each op is a deterministic function of its arguments and `dot BK` (the
MXU's fp32 result for one output element of one `BK`-wide `dot_general` step) depends only on the
two `BK`-vectors, not on the block's other rows or columns. -/
structure MOps where
  /-- One output element of one `K` step: the fp32 dot product of two `BK`-vectors. -/
  dot : (BK : ℕ) → (Fin BK → ℝ) → (Fin BK → ℝ) → ℝ
  /-- fp32 `acc + partial`. -/
  add : ℝ → ℝ → ℝ
  /-- `astype(o_ref.dtype)`. -/
  cast : ℝ → ℝ

/-- Exact real arithmetic, for functional correctness. -/
noncomputable def MOps.exact : MOps where
  dot := fun _ a b => ∑ u, a u * b u
  add := (· + ·)
  cast := id

variable (ops : MOps) (BK nk : ℕ)

/-- The PR's contract for one output element: `cast` of the in-order fp32 accumulation over the
`nk` chunks of `BK` elements, starting from `0`. `xr` and `wc` are row `r` of `x` and row `c` of
`w`. -/
def matmulElem (xr wc : ℕ → ℝ) : ℝ :=
  ops.cast ((List.range nk).foldl (fun acc t =>
    ops.add acc (ops.dot BK (fun u => xr (t * BK + u.val)) (fun u => wc (t * BK + u.val)))) 0)

/-- `pl.BlockSpec((bm, BK), lambda i, j, kk: (i, kk))`. -/
def xSpec (bm : ℕ) : BlockSpec :=
  { blockShape := [bm, BK]
    indexMap := fun g _ => [((g.getD 0 0 : ℕ) : ℤ), ((g.getD 2 0 : ℕ) : ℤ)] }

/-- `pl.BlockSpec((bn, BK), lambda i, j, kk: (j, kk))`. -/
def wSpec (bn : ℕ) : BlockSpec :=
  { blockShape := [bn, BK]
    indexMap := fun g _ => [((g.getD 1 0 : ℕ) : ℤ), ((g.getD 2 0 : ℕ) : ℤ)] }

/-- `pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j))`. -/
def oSpec (bm bn : ℕ) : BlockSpec :=
  { blockShape := [bm, bn]
    indexMap := fun g _ => [((g.getD 0 0 : ℕ) : ℤ), ((g.getD 1 0 : ℕ) : ℤ)] }

/-- The `pallas_call` of `_jax_matmul` (`matmul_kernel.py:39-61`), with `K = nk * BK`. -/
def gridSpec (M N bm bn : ℕ) (xBuf wBuf oBuf : BufId) : GridSpec :=
  { grid := [(M + bm - 1) / bm, (N + bn - 1) / bn, nk]
    inSpecs := [(⟨xBuf, [M, nk * BK]⟩, xSpec BK bm), (⟨wBuf, [N, nk * BK]⟩, wSpec BK bn)]
    outSpecs := [(⟨oBuf, [M, N]⟩, oSpec bm bn)]
    scratch := [[bm, bn]]
    dimSemantics := [.parallel, .parallel, .arbitrary] }

/-- `_mm_kernel` (`matmul_kernel.py:20-36`): zero the accumulator at `k == 0`, add this step's
`dot_general`, and write `acc.astype(...)` to the output at `k == nk - 1`. `prev` is
`[o_ref, acc_ref]`; the result is `[o_ref, acc_ref]` after the step. -/
noncomputable def body : Body := fun g _ ins prev =>
  let kk := g.getD 2 0
  let acc : List ℕ → ℝ := fun loc =>
    ops.add (if kk = 0 then 0 else (prev.getD 1 fun _ => 0) loc)
      (ops.dot BK (fun u => (ins.getD 0 fun _ => 0) [loc.getD 0 0, u.val])
        (fun u => (ins.getD 1 fun _ => 0) [loc.getD 1 0, u.val]))
  [fun loc => if kk = nk - 1 then ops.cast (acc loc) else (prev.getD 0 fun _ => 0) loc, acc]

/-- `_mm_kernel` packaged as a `ReductionKernel` over the 2-D outer tile grid and reduction axis
`nk` (`matmul_kernel.py:20-61`). -/
noncomputable def matmulKernel (M N bm bn : ℕ) (xBuf wBuf oBuf : BufId) : ReductionKernel 1 where
  outerGrid := [(M + bm - 1) / bm, (N + bn - 1) / bn]
  nk := nk
  inSpecs := [(⟨xBuf, [M, nk * BK]⟩, xSpec BK bm), (⟨wBuf, [N, nk * BK]⟩, wSpec BK bn)]
  outArr := ⟨oBuf, [M, N]⟩
  outBlockShape := [bm, bn]
  outIndexMap := fun og _ => [((og.getD 0 0 : ℕ) : ℤ), ((og.getD 1 0 : ℕ) : ℤ)]
  init := fun _ => 0
  step := fun _ _ _ ins loc prevAcc _ =>
    ops.add (prevAcc 0)
      (ops.dot BK (fun u => (ins.getD 0 fun _ => 0) [loc.getD 0 0, u.val])
        (fun u => (ins.getD 1 fun _ => 0) [loc.getD 1 0, u.val]))
  finalize := fun _ _ _ _ acc => ops.cast (acc 0)
  outerDimSemantics := [.parallel, .parallel]

private theorem matmulKernel_gridSpec (M N bm bn : ℕ) (xBuf wBuf oBuf : BufId) :
    (matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).gridSpec =
      gridSpec BK nk M N bm bn xBuf wBuf oBuf := by
  simp only [ReductionKernel.gridSpec, ReductionKernel.outPair, ReductionKernel.outSpec,
    matmulKernel, gridSpec, oSpec, List.cons_append, List.nil_append, List.replicate_succ,
    List.replicate_zero]
  congr 1
  simp only [List.cons.injEq, Prod.mk.injEq, and_true, true_and]
  congr 1
  funext g _
  rcases g with _ | ⟨a, _ | ⟨b, g'⟩⟩ <;> rfl

private theorem matmulKernel_body (M N bm bn : ℕ) (xBuf wBuf oBuf : BufId) :
    (matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).body = body ops BK nk := by
  funext g sc ins prev
  simp [ReductionKernel.body, matmulKernel, body, List.ofFn_succ]

private theorem mem_outerGrid_iff {a b : ℕ} {og : List ℕ} :
    og ∈ gridPoints [a, b] ↔ ∃ i j, og = [i, j] ∧ i < a ∧ j < b := by
  rw [mem_gridPoints]
  rcases og with _ | ⟨i, _ | ⟨j, _ | ⟨u, g⟩⟩⟩ <;> simp [inBounds]

private theorem matmulKernel_valid (M N bm bn : ℕ) (hM : 0 < M) (hN : 0 < N)
    (hbm : 0 < bm) (hbn : 0 < bn) (hBK : 0 < BK)
    (hnk : 0 < nk) (xBuf wBuf oBuf : BufId) (hx : oBuf ≠ xBuf) (hw : oBuf ≠ wBuf) (m0 : Mem) :
    (matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).Valid m0 where
  outer_grid_pos := by
    intro d hd
    simp only [matmulKernel, List.mem_cons, List.not_mem_nil, or_false] at hd
    rcases hd with rfl | rfl
    · exact cdiv_pos hM hbm
    · exact cdiv_pos hN hbn
  nk_pos := hnk
  in_blocks_ok := by
    intro og hog t ht p hp
    obtain ⟨i, j, rfl, hi, hj⟩ := mem_outerGrid_iff.mp hog
    have h1 : (i : ℤ) * bm < M := by exact_mod_cast lt_of_lt_cdiv hbm hi
    have h2 : (j : ℤ) * bn < N := by exact_mod_cast lt_of_lt_cdiv hbn hj
    have ht' : t < nk := ht
    simp only [matmulKernel, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;>
      simp [xSpec, wSpec, blockOk, hbm, hbn, hBK, h1, h2, ht']
  out_blocks_ok := by
    intro og hog
    obtain ⟨i, j, rfl, hi, hj⟩ := mem_outerGrid_iff.mp hog
    have h1 : (i : ℤ) * bm < M := by exact_mod_cast lt_of_lt_cdiv hbm hi
    have h2 : (j : ℤ) * bn < N := by exact_mod_cast lt_of_lt_cdiv hbn hj
    simp [matmulKernel, blockOk, hbm, hbn, h1, h2]
  out_inj := by
    intro og₁ h₁ og₂ h₂ heq
    obtain ⟨i₁, j₁, rfl, _, _⟩ := mem_outerGrid_iff.mp h₁
    obtain ⟨i₂, j₂, rfl, _, _⟩ := mem_outerGrid_iff.mp h₂
    simp only [matmulKernel, List.getD_cons_zero, List.getD_cons_succ, List.cons.injEq,
      Nat.cast_inj, and_true] at heq
    simp [heq]
  ins_disjoint := by
    intro p hp
    simp only [matmulKernel, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl
    · exact hx.symm
    · exact hw.symm

/-- The PR's matmul is defined for every positive `M`, `N` and block size, every output element is
the in-order fp32 accumulation of its rows over the `K` steps, and nothing else changes. -/
theorem matmul_correct (M N bm bn : ℕ) (xBuf wBuf oBuf : BufId)
    (hM : 0 < M) (hN : 0 < N) (hbm : 0 < bm) (hbn : 0 < bn) (hBK : 0 < BK) (hnk : 0 < nk)
    (hx : oBuf ≠ xBuf) (hw : oBuf ≠ wBuf) (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', pallasCall (gridSpec BK nk M N bm bn xBuf wBuf oBuf) (body ops BK nk) junk m0 =
        some m' ∧
      (∀ (r : Fin M) (c : Fin N), m'.read oBuf (r.val * N + c.val) =
        matmulElem ops BK nk (fun k => m0.read xBuf (r.val * (nk * BK) + k))
          (fun k => m0.read wBuf (c.val * (nk * BK) + k))) ∧
      Mem.agreeOutside (Mem.bufFootprint oBuf) m0 m' := by
  set rk := matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf
  have hv := matmulKernel_valid ops BK nk M N bm bn hM hN hbm hbn hBK hnk xBuf wBuf oBuf hx hw m0
  obtain ⟨m', hcall, hval, hframe⟩ := rk.correct hv junk
  rw [matmulKernel_gridSpec, matmulKernel_body] at hcall
  refine ⟨m', hcall, ?_, hframe⟩
  intro r c
  have hog : [r.val / bm, c.val / bn] ∈ gridPoints rk.outerGrid :=
    mem_outerGrid_iff.mpr ⟨_, _, rfl, div_lt_cdiv hbm r.isLt, div_lt_cdiv hbn c.isLt⟩
  have hin : addrInBlock rk.outArr rk.outBlockShape
      (rk.outBlockIdx [r.val / bm, c.val / bn] (rk.scalarPrefetch m0)) (r.val * N + c.val) = true := by
    simp only [rk, matmulKernel, ReductionKernel.outBlockIdx, List.getD_cons_zero,
      List.getD_cons_succ, List.map_cons, List.map_nil, Int.toNat_natCast]
    rw [addrInBlock_two hbm hbn r.isLt c.isLt]
    simp
  refine (hval [r.val / bm, c.val / bn] hog (r.val * N + c.val) hin).trans ?_
  have hr' : r.val / bm * bm + (r.val - r.val / bm * bm) = r.val := by
    have := Nat.div_mul_le_self r.val bm; omega
  have hc' : c.val / bn * bn + (c.val - c.val / bn * bn) = c.val := by
    have := Nat.div_mul_le_self c.val bn; omega
  have hrl : r.val - r.val / bm * bm < bm := by have := Nat.lt_div_mul_add (a := r.val) hbm; omega
  have hcl : c.val - c.val / bn * bn < bn := by have := Nat.lt_div_mul_add (a := c.val) hbn; omega
  have hloc : localCoord rk.outBlockShape
      (rk.outBlockIdx [r.val / bm, c.val / bn] (rk.scalarPrefetch m0))
      (unflatCoord rk.outArr.shape (r.val * N + c.val)) =
      [r.val - r.val / bm * bm, c.val - c.val / bn * bn] := by
    simp only [rk, matmulKernel, ReductionKernel.outBlockIdx, List.getD_cons_zero,
      List.getD_cons_succ, List.map_cons, List.map_nil, Int.toNat_natCast,
      unflatCoord_two M N r.val c.val c.isLt, localCoord]
  rw [hloc]
  simp only [matmulElem]
  change ops.cast (((matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).accFold junk m0
    [r.val / bm, c.val / bn] [r.val - r.val / bm * bm, c.val - c.val / bn * bn] nk) 0) = _
  congr 1
  suffices ∀ n ≤ nk,
      ((matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).accFold junk m0
        [r.val / bm, c.val / bn] [r.val - r.val / bm * bm, c.val - c.val / bn * bn] n) 0 =
      (List.range n).foldl (fun acc t =>
        ops.add acc (ops.dot BK
          (fun u => m0.read xBuf (r.val * (nk * BK) + (t * BK + u.val)))
          (fun u => m0.read wBuf (c.val * (nk * BK) + (t * BK + u.val))))) 0 from
    this nk le_rfl
  intro n hn
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [ReductionKernel.accFold_succ, List.range_succ, List.foldl_append, List.foldl_cons,
      List.foldl_nil]
    have hstep : (matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).step
        [r.val / bm, c.val / bn] n ((matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).scalarPrefetch m0)
        (loadIns (matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).gridSpec junk m0
          ([r.val / bm, c.val / bn] ++ [n]))
        [r.val - r.val / bm * bm, c.val - c.val / bn * bn]
        ((matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).accFold junk m0
          [r.val / bm, c.val / bn] [r.val - r.val / bm * bm, c.val - c.val / bn * bn] n) 0 =
        ops.add (((matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).accFold junk m0
            [r.val / bm, c.val / bn] [r.val - r.val / bm * bm, c.val - c.val / bn * bn] n) 0)
          (ops.dot BK
            (fun u => ((loadIns (matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).gridSpec junk m0
              ([r.val / bm, c.val / bn] ++ [n])).getD 0 fun _ => 0) [r.val - r.val / bm * bm, u.val])
            (fun u => ((loadIns (matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).gridSpec junk m0
              ([r.val / bm, c.val / bn] ++ [n])).getD 1 fun _ => 0) [c.val - c.val / bn * bn, u.val])) := rfl
    rw [hstep, ih (by omega)]
    have hnt : n < nk := by omega
    have hk : ∀ u : Fin BK, n * BK + u.val < nk * BK := fun u => by nlinarith [u.isLt]
    congr 2 <;> funext u
    · have hx_idx : (xSpec BK bm).blockIdx ([r.val / bm, c.val / bn] ++ [n])
          ((matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).gridSpec.scalarPrefetch m0) =
          [r.val / bm, n] := by
        simp only [xSpec, BlockSpec.blockIdx, List.cons_append, List.nil_append, List.getD_cons_zero,
          List.getD_cons_succ, List.map_cons, List.map_nil, Int.toNat_natCast]
      rw [loadIns_getD (matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).gridSpec junk m0 _ 0
        (⟨xBuf, [M, nk * BK]⟩, xSpec BK bm) (by simp [ReductionKernel.gridSpec, matmulKernel]),
        hx_idx, loadBlock_eq_of_inBounds]
      · simp [xSpec, globalCoord, flatCoord, hr']
      · simp [xSpec, inBounds, hrl, u.isLt]
      · simp [xSpec, globalCoord, inBounds, hr', r.isLt, hk u]
    · have hw_idx : (wSpec BK bn).blockIdx ([r.val / bm, c.val / bn] ++ [n])
          ((matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).gridSpec.scalarPrefetch m0) =
          [c.val / bn, n] := by
        simp only [wSpec, BlockSpec.blockIdx, List.cons_append, List.nil_append, List.getD_cons_zero,
          List.getD_cons_succ, List.map_cons, List.map_nil, Int.toNat_natCast]
      rw [loadIns_getD (matmulKernel ops BK nk M N bm bn xBuf wBuf oBuf).gridSpec junk m0 _ 1
        (⟨wBuf, [N, nk * BK]⟩, wSpec BK bn) (by simp [ReductionKernel.gridSpec, matmulKernel]),
        hw_idx, loadBlock_eq_of_inBounds]
      · simp [wSpec, globalCoord, flatCoord, hc']
      · simp [wSpec, inBounds, hcl, u.isLt]
      · simp [wSpec, globalCoord, inBounds, hc', c.isLt, hk u]

/-- The PR's contract, bit for bit under the `MOps` assumptions: an output element is the same
whatever the call's `M`, `N`, block sizes, junk and other rows, as long as its `x` row and `w` row
are the same. -/
theorem matmul_batch_invariant {M N M' N' bm bn bm' bn' : ℕ} (hbm : 0 < bm) (hbn : 0 < bn)
    (hbm' : 0 < bm') (hbn' : 0 < bn') (hBK : 0 < BK) (hnk : 0 < nk)
    {xBuf wBuf oBuf xBuf' wBuf' oBuf' : BufId} (hx : oBuf ≠ xBuf) (hw : oBuf ≠ wBuf)
    (hx' : oBuf' ≠ xBuf') (hw' : oBuf' ≠ wBuf')
    {junk junk' : ℕ → ℝ} {m0 m0' m₁ m₂ : Mem}
    (h₁ : pallasCall (gridSpec BK nk M N bm bn xBuf wBuf oBuf) (body ops BK nk) junk m0 =
      some m₁)
    (h₂ : pallasCall (gridSpec BK nk M' N' bm' bn' xBuf' wBuf' oBuf') (body ops BK nk)
      junk' m0' = some m₂)
    (r : Fin M) (c : Fin N) (r' : Fin M') (c' : Fin N')
    (hxr : ∀ k < nk * BK, m0.read xBuf (r.val * (nk * BK) + k) =
      m0'.read xBuf' (r'.val * (nk * BK) + k))
    (hwc : ∀ k < nk * BK, m0.read wBuf (c.val * (nk * BK) + k) =
      m0'.read wBuf' (c'.val * (nk * BK) + k)) :
    m₁.read oBuf (r.val * N + c.val) = m₂.read oBuf' (r'.val * N' + c'.val) := by
  have hM : 0 < M := Nat.lt_of_le_of_lt (Nat.zero_le _) r.isLt
  have hN : 0 < N := Nat.lt_of_le_of_lt (Nat.zero_le _) c.isLt
  have hM' : 0 < M' := Nat.lt_of_le_of_lt (Nat.zero_le _) r'.isLt
  have hN' : 0 < N' := Nat.lt_of_le_of_lt (Nat.zero_le _) c'.isLt
  obtain ⟨n₁, e₁, v₁, -⟩ :=
    matmul_correct ops BK nk M N bm bn xBuf wBuf oBuf hM hN hbm hbn hBK hnk hx hw junk m0
  obtain ⟨n₂, e₂, v₂, -⟩ :=
    matmul_correct ops BK nk M' N' bm' bn' xBuf' wBuf' oBuf' hM' hN' hbm' hbn' hBK hnk hx' hw'
      junk' m0'
  rw [e₁, Option.some.injEq] at h₁
  rw [e₂, Option.some.injEq] at h₂
  subst h₁ h₂
  rw [v₁, v₂, matmulElem, matmulElem]
  congr 1
  apply List.foldl_ext
  intro acc t ht
  rw [List.mem_range] at ht
  have hk : ∀ u : Fin BK, t * BK + u.val < nk * BK := fun u => by nlinarith [u.isLt]
  congr 2 <;> funext u
  · exact hxr _ (hk u)
  · exact hwc _ (hk u)

/-- With exact arithmetic the element is the dot product of the two rows over all `K` columns. -/
theorem matmulElem_exact (xr wc : ℕ → ℝ) :
    matmulElem MOps.exact BK nk xr wc = ∑ k ∈ Finset.range (nk * BK), xr k * wc k := by
  simp only [matmulElem, MOps.exact, id]
  induction nk with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.foldl_append, ih, Nat.succ_mul, Finset.sum_range_add]
    simp [Fin.sum_univ_eq_sum_range (fun u => xr (n * BK + u) * wc (n * BK + u))]

end TpuKernelLean.Examples.TrueOnPolicyMatmul
