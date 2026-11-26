import TpuKernelLean.Examples.KvPageWrite
import TpuKernelLean.Examples.TrueOnPolicyNorm
import TpuKernelLean.Examples.TrueOnPolicyMatmul
import TpuKernelLean.Examples.Fp8LinearSmallM
import TpuKernelLean.Examples.QkNormRope
import TpuKernelLean.Examples.BlockScaledMm
import TpuKernelLean.Examples.GroupedScaledMm
import TpuKernelLean.Tests.GroupedScaledMm

/-!
End-to-end concrete tests for every `pallas_call`-level kernel in `TpuKernelLean/Examples/`:
each theorem supplies concrete small shapes, concrete distinct buffer ids, a concrete initial
memory, and exact-arithmetic ops where applicable, applies the kernel's headline correctness
theorem and its `*_exact` theorem, and computes concrete numeric output values.

This shows that each kernel's precondition is satisfiable and that the chain from `pallasCall` to
concrete numbers is unbroken.
-/

namespace TpuKernelLean.Tests.EndToEnd

open TpuKernelLean.Pallas

noncomputable section

private theorem sqrt_four : Real.sqrt 4 = 2 := by
  rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]

private theorem sqrt_sixteen : Real.sqrt 16 = 4 := by
  rw [show (16 : ℝ) = 4 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]

private theorem sum_fin_two_mul_one (f : Fin (2 * 1) → ℝ) :
    (∑ k : Fin (2 * 1), f k) = f ⟨0, by decide⟩ + f ⟨1, by decide⟩ :=
  Fin.sum_univ_two f

/-! ### 1. `KvPageWrite` -/

/-- Concrete initial memory for `kvPageWrite_endToEnd`:
`idsBuf = 0` maps page 0 to slot 2 and page 1 to slot 0; `pagesBuf = 1` holds pages `[10, 11]`
and `[20, 21]`; `kvBuf = 2` holds three initial slots `[1, 2]`, `[3, 4]`, `[5, 6]`. -/
private def kvInitMem : Mem
  | 0, 0 => 2
  | 0, 1 => 0
  | 1, 0 => 10
  | 1, 1 => 11
  | 1, 2 => 20
  | 1, 3 => 21
  | 2, 0 => 1
  | 2, 1 => 2
  | 2, 2 => 3
  | 2, 3 => 4
  | 2, 4 => 5
  | 2, 5 => 6
  | _, _ => 0

/-- End-to-end `KvPageWrite` (`N = 3`, `M = 2`, `P = 2`): two pages written to distinct slots
`[2, 0]`, leaving slot `1` untouched. -/
theorem kvPageWrite_endToEnd (junk : ℕ → ℝ) :
    ∃ m', Examples.KvPageWrite.writePagesRun 3 2 2 0 1 2 junk kvInitMem = some m' ∧
      m'.read 2 0 = 20 ∧ m'.read 2 1 = 21 ∧
      m'.read 2 2 = 3 ∧ m'.read 2 3 = 4 ∧
      m'.read 2 4 = 10 ∧ m'.read 2 5 = 11 := by
  have hreq : (Examples.KvPageWrite.writePagesSpec 3 2 2 0 1 2).requires
      ((Examples.KvPageWrite.writePagesSpec 3 2 2 0 1 2).readSt kvInitMem)
      ((Examples.KvPageWrite.writePagesSpec 3 2 2 0 1 2).readIn kvInitMem) := by
    refine ⟨by decide, ?_, ?_, by decide, by decide, by decide⟩
    · intro i
      fin_cases i
      · exact ⟨2, by decide, (decodeInt32Nat_eq_some_iff _ 2).mpr
          ⟨by norm_num [Examples.KvPageWrite.writePagesSpec, kvInitMem, Mem.read], by decide⟩⟩
      · exact ⟨0, by decide, (decodeInt32Nat_eq_some_iff _ 0).mpr
          ⟨by norm_num [Examples.KvPageWrite.writePagesSpec, kvInitMem, Mem.read], by decide⟩⟩
    · intro i j hij
      fin_cases i <;> fin_cases j <;>
        first
        | rfl
        | simp [Examples.KvPageWrite.writePagesSpec, kvInitMem, Mem.read] at hij
  obtain ⟨-, m', hcall, _, hst, _⟩ :=
    Examples.KvPageWrite.writePages_correct 3 2 2 0 1 2 (by decide) junk kvInitMem hreq
  have hread : ∀ (j : Fin 3) (p : Fin 2),
      m'.read 2 (j.val * 2 + p.val) =
        if h : ∃ i : Fin 2, kvInitMem.read 0 i.val = (j.val : ℝ) then
          kvInitMem.read 1 ((Classical.choose h).val * 2 + p.val)
        else
          kvInitMem.read 2 (j.val * 2 + p.val) := by
    intro j p
    have h := congrArg (fun st => st.kv.get2 j p) hst
    simpa [Examples.KvPageWrite.writePagesSpec, Examples.KvPageWrite.writePagesMut,
      Layout.rowMajor, Layout.read, Tensor.get2, Tensor.of2, Idx.flat_two] using h
  have hex0 : ∃ i : Fin 2, kvInitMem.read 0 i.val = ((0 : ℕ) : ℝ) :=
    ⟨1, by norm_num [kvInitMem, Mem.read]⟩
  have hch0 : Classical.choose hex0 = 1 := by
    have hc := Classical.choose_spec hex0
    generalize Classical.choose hex0 = idx at hc ⊢
    fin_cases idx
    · simp [kvInitMem, Mem.read] at hc
    · rfl
  have hno1 : ¬ ∃ i : Fin 2, kvInitMem.read 0 i.val = ((1 : ℕ) : ℝ) := by
    rintro ⟨i, hi⟩
    fin_cases i <;> simp [kvInitMem, Mem.read] at hi
  have hex2 : ∃ i : Fin 2, kvInitMem.read 0 i.val = ((2 : ℕ) : ℝ) :=
    ⟨0, by norm_num [kvInitMem, Mem.read]⟩
  have hch2 : Classical.choose hex2 = 0 := by
    have hc := Classical.choose_spec hex2
    generalize Classical.choose hex2 = idx at hc ⊢
    fin_cases idx
    · rfl
    · simp [kvInitMem, Mem.read] at hc
  refine ⟨m', hcall, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · have h := hread ⟨0, by decide⟩ ⟨0, by decide⟩
    rw [dite_eq_left hex0, hch0] at h
    norm_num [kvInitMem, Mem.read] at h
    exact h
  · have h := hread ⟨0, by decide⟩ ⟨1, by decide⟩
    rw [dite_eq_left hex0, hch0] at h
    norm_num [kvInitMem, Mem.read] at h
    exact h
  · have h := hread ⟨1, by decide⟩ ⟨0, by decide⟩
    rw [dite_eq_right hno1] at h
    norm_num [kvInitMem, Mem.read] at h
    exact h
  · have h := hread ⟨1, by decide⟩ ⟨1, by decide⟩
    rw [dite_eq_right hno1] at h
    norm_num [kvInitMem, Mem.read] at h
    exact h
  · have h := hread ⟨2, by decide⟩ ⟨0, by decide⟩
    rw [dite_eq_left hex2, hch2] at h
    norm_num [kvInitMem, Mem.read] at h
    exact h
  · have h := hread ⟨2, by decide⟩ ⟨1, by decide⟩
    rw [dite_eq_left hex2, hch2] at h
    norm_num [kvInitMem, Mem.read] at h
    exact h

/-! ### 2. `TrueOnPolicyNorm` -/

/-- Concrete initial memory for `rmsNorm_endToEnd`:
`xBuf = 0` holds two rows `[1, 1]` and `[5, 1]`; `wBuf = 1` holds weight `[4, 8]`. With `eps = 3`,
the row RMS denominators are `√((1 + 1)/2 + 3) = 2` and `√((25 + 1)/2 + 3) = 4`. -/
private def normInitMem : Mem
  | 0, 0 => 1
  | 0, 1 => 1
  | 0, 2 => 5
  | 0, 3 => 1
  | 1, 0 => 4
  | 1, 1 => 8
  | _, _ => 0

/-- End-to-end `TrueOnPolicyNorm` (`M = 2`, `H = 2`, `rows = 1`, `eps = 3`, plain RMSNorm under
`FOps.exact`): outputs are `[[2, 4], [5, 2]]`. -/
theorem rmsNorm_endToEnd (junk : ℕ → ℝ) :
    ∃ m', pallasCall
        ((Examples.TrueOnPolicyNorm.plainKernel Examples.TrueOnPolicyNorm.FOps.exact 3 false 2).gridSpec
          2 1 ![0] 1 ![2])
        (Examples.TrueOnPolicyNorm.plainKernel Examples.TrueOnPolicyNorm.FOps.exact 3 false 2).body
        junk normInitMem = some m' ∧
      m'.read 2 0 = 2 ∧ m'.read 2 1 = 4 ∧ m'.read 2 2 = 5 ∧ m'.read 2 3 = 2 := by
  obtain ⟨m', hcall, hval, -⟩ :=
    Examples.TrueOnPolicyNorm.rmsNorm_correct Examples.TrueOnPolicyNorm.FOps.exact 3 false 2 2 1
      (by decide) (by decide) (by decide) 0 1 2 (by decide) (by decide) junk normInitMem
  refine ⟨m', hcall, ?_, ?_, ?_, ?_⟩
  · have h := hval ⟨0, by decide⟩ ⟨0, by decide⟩
    rw [Examples.TrueOnPolicyNorm.normRow_exact] at h
    simp only [Tensor.rmsNorm, Tensor.get2_of2, Tensor.get1_of1, Fin.sum_univ_two, Fin.val_zero,
      Fin.val_one, normInitMem, Mem.read] at h
    rw [show ((1 : ℝ) * 1 + 1 * 1) / (2 : ℕ) + 3 = 4 by norm_num, sqrt_four] at h
    norm_num at h
    exact h
  · have h := hval ⟨0, by decide⟩ ⟨1, by decide⟩
    rw [Examples.TrueOnPolicyNorm.normRow_exact] at h
    simp only [Tensor.rmsNorm, Tensor.get2_of2, Tensor.get1_of1, Fin.sum_univ_two, Fin.val_zero,
      Fin.val_one, normInitMem, Mem.read] at h
    rw [show ((1 : ℝ) * 1 + 1 * 1) / (2 : ℕ) + 3 = 4 by norm_num, sqrt_four] at h
    norm_num at h
    exact h
  · have h := hval ⟨1, by decide⟩ ⟨0, by decide⟩
    rw [Examples.TrueOnPolicyNorm.normRow_exact] at h
    simp only [Tensor.rmsNorm, Tensor.get2_of2, Tensor.get1_of1, Fin.sum_univ_two, Fin.val_zero,
      Fin.val_one, normInitMem, Mem.read] at h
    rw [show ((5 : ℝ) * 5 + 1 * 1) / (2 : ℕ) + 3 = 16 by norm_num, sqrt_sixteen] at h
    norm_num at h
    exact h
  · have h := hval ⟨1, by decide⟩ ⟨1, by decide⟩
    rw [Examples.TrueOnPolicyNorm.normRow_exact] at h
    simp only [Tensor.rmsNorm, Tensor.get2_of2, Tensor.get1_of1, Fin.sum_univ_two, Fin.val_zero,
      Fin.val_one, normInitMem, Mem.read] at h
    rw [show ((5 : ℝ) * 5 + 1 * 1) / (2 : ℕ) + 3 = 16 by norm_num, sqrt_sixteen] at h
    norm_num at h
    exact h

/-! ### 3. `TrueOnPolicyMatmul` -/

/-- Concrete initial memory for `matmul_endToEnd`:
`x [2, 4] = [[1, 2, 3, 4], [2, 0, 1, 3]]` in buffer `0`,
`w [2, 4] = [[1, 1, 1, 1], [2, 3, 0, 1]]` in buffer `1`. -/
private def matmulInitMem : Mem
  | 0, 0 => 1
  | 0, 1 => 2
  | 0, 2 => 3
  | 0, 3 => 4
  | 0, 4 => 2
  | 0, 5 => 0
  | 0, 6 => 1
  | 0, 7 => 3
  | 1, 0 => 1
  | 1, 1 => 1
  | 1, 2 => 1
  | 1, 3 => 1
  | 1, 4 => 2
  | 1, 5 => 3
  | 1, 6 => 0
  | 1, 7 => 1
  | _, _ => 0

/-- End-to-end `TrueOnPolicyMatmul` (`BK = 2`, `nk = 2`, `M = 2`, `N = 2`, `bm = 1`, `bn = 2`,
under `MOps.exact`): `x @ w^T = [[10, 12], [6, 7]]`. -/
theorem matmul_endToEnd (junk : ℕ → ℝ) :
    ∃ m', pallasCall (Examples.TrueOnPolicyMatmul.gridSpec 2 2 2 2 1 2 0 1 2)
        (Examples.TrueOnPolicyMatmul.body Examples.TrueOnPolicyMatmul.MOps.exact 2 2)
        junk matmulInitMem = some m' ∧
      m'.read 2 0 = 10 ∧ m'.read 2 1 = 12 ∧ m'.read 2 2 = 6 ∧ m'.read 2 3 = 7 := by
  obtain ⟨m', hcall, hval, -⟩ :=
    Examples.TrueOnPolicyMatmul.matmul_correct Examples.TrueOnPolicyMatmul.MOps.exact 2 2 2 2 1 2
      0 1 2 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) junk matmulInitMem
  refine ⟨m', hcall, ?_, ?_, ?_, ?_⟩
  · have h := hval ⟨0, by decide⟩ ⟨0, by decide⟩
    rw [Examples.TrueOnPolicyMatmul.matmulElem_exact, show 2 * 2 = 4 from rfl] at h
    simp only [Finset.sum_range_succ, Finset.sum_range_zero] at h
    norm_num [matmulInitMem, Mem.read] at h
    exact h
  · have h := hval ⟨0, by decide⟩ ⟨1, by decide⟩
    rw [Examples.TrueOnPolicyMatmul.matmulElem_exact, show 2 * 2 = 4 from rfl] at h
    simp only [Finset.sum_range_succ, Finset.sum_range_zero] at h
    norm_num [matmulInitMem, Mem.read] at h
    exact h
  · have h := hval ⟨1, by decide⟩ ⟨0, by decide⟩
    rw [Examples.TrueOnPolicyMatmul.matmulElem_exact, show 2 * 2 = 4 from rfl] at h
    simp only [Finset.sum_range_succ, Finset.sum_range_zero] at h
    norm_num [matmulInitMem, Mem.read] at h
    exact h
  · have h := hval ⟨1, by decide⟩ ⟨1, by decide⟩
    rw [Examples.TrueOnPolicyMatmul.matmulElem_exact, show 2 * 2 = 4 from rfl] at h
    simp only [Finset.sum_range_succ, Finset.sum_range_zero] at h
    norm_num [matmulInitMem, Mem.read] at h
    exact h

/-! ### 4. `Fp8LinearSmallM` -/

/-- Concrete initial memory for `fp8LinearSmallM_endToEnd` at the smallest supported shape
`(tokens = 1, k = 128, n = 128, tile_n = 128)`:
`x` (`0`) has one non-zero entry `x[0, 0] = 3` so the 128-term dot product collapses to a single
term; `w` (`1`) sets `w[0, 0] = 4`, `w[0, 1] = -2`, `w[0, 127] = 7`; `ws` (`2`) sets `ws[0, 0] = 2`,
`ws[0, 1] = 5`, `ws[0, 127] = 3`, and `1` elsewhere. -/
private def fp8InitMem : Mem
  | 0, 0 => 3
  | 1, 0 => 4
  | 1, 1 => -2
  | 1, 127 => 7
  | 2, 0 => 2
  | 2, 1 => 5
  | 2, 127 => 3
  | 2, _ => 1
  | _, _ => 0

private theorem fp8InitMem_x_eq_zero {k : ℕ} (hk : k ≠ 0) : fp8InitMem 0 k = 0 := by
  cases k with
  | zero => exact absurd rfl hk
  | succ _ => rfl

/-- End-to-end `Fp8LinearSmallM` (`tokens = 1`, `k = 128`, `n = 128`, `tile_n = 128`, `fp8E4M3` ->
`bf16` under `FP8LinearOps.exact`): checks four representative columns (`0`, `1`, `2`, `127`). -/
theorem fp8LinearSmallM_endToEnd (junk : ℕ → ℝ) :
    ∃ m', Examples.Fp8LinearSmallM.fp8LinearRun Examples.Fp8LinearSmallM.FP8LinearOps.exact
        .fp8E4M3 .bf16 1 128 128 128 0 1 2 3 junk fp8InitMem = some m' ∧
      m'.read 3 0 = 24 ∧ m'.read 3 1 = -30 ∧ m'.read 3 2 = 0 ∧ m'.read 3 127 = 63 := by
  have hsup : Examples.Fp8LinearSmallM.Supports 1 128 128 128 :=
    ⟨by decide, by decide, by decide, by decide, by decide,
      by simp [Examples.Fp8LinearSmallM.tileNCandidates], by decide, by decide⟩
  obtain ⟨m', hcall, hval, -⟩ :=
    Examples.Fp8LinearSmallM.fp8Linear_correct Examples.Fp8LinearSmallM.FP8LinearOps.exact
      .fp8E4M3 .bf16 rfl 1 128 128 128 hsup 0 1 2 3 (by decide) (by decide) (by decide)
      junk fp8InitMem
  have hcol : ∀ c : Fin 128,
      m'.read 3 c.val = 3 * fp8InitMem.read 1 c.val * fp8InitMem.read 2 c.val := by
    intro c
    have h := hval ⟨0, by decide⟩ c
    rw [Examples.Fp8LinearSmallM.fp8LinearElem_exact rfl] at h
    simp only [zero_mul, zero_add] at h
    have hsum :
        (∑ p : Fin 128, fp8InitMem.read 0 p.val * fp8InitMem.read 1 (p.val * 128 + c.val)) =
          3 * fp8InitMem.read 1 c.val := by
      rw [Finset.sum_eq_single (0 : Fin 128)]
      · simp [fp8InitMem, Mem.read]
      · intro p _ hp
        have hp0 : p.val ≠ 0 := fun h0 => hp (Fin.ext h0)
        simp only [Mem.read, fp8InitMem_x_eq_zero hp0, zero_mul]
      · intro hmem
        exact absurd (Finset.mem_univ _) hmem
    rw [hsum] at h
    exact h
  refine ⟨m', hcall, ?_, ?_, ?_, ?_⟩
  · have h := hcol ⟨0, by decide⟩
    norm_num [fp8InitMem, Mem.read] at h
    exact h
  · have h := hcol ⟨1, by decide⟩
    norm_num [fp8InitMem, Mem.read] at h
    exact h
  · have h := hcol ⟨2, by decide⟩
    norm_num [fp8InitMem, Mem.read] at h
    exact h
  · have h := hcol ⟨127, by decide⟩
    norm_num [fp8InitMem, Mem.read] at h
    exact h

/-! ### 5. `QkNormRope` -/

/-- Concrete initial memory for `qkNormRope_endToEnd` at the smallest shape `requires` admits
(`T = 1`, `numQ = 1`, `numKV = 1`, `h = 1`, `bm = 1`, `eps = 3`):
- `qkv` (`0`): `q = [1, 1]` (RMS denom `2`), `k = [5, 1]` (RMS denom `4`), `v = [99, -42]`;
- `cosSin` (`1`): `cos = 3/5`, `sin = 4/5`;
- `qWeight` (`2`): `[2, 4]`, `kWeight` (`3`): `[4, 8]`. -/
private def qkInitMem : Mem
  | 0, 0 => 1
  | 0, 1 => 1
  | 0, 2 => 5
  | 0, 3 => 1
  | 0, 4 => 99
  | 0, 5 => -42
  | 1, 0 => 3 / 5
  | 1, 1 => 4 / 5
  | 2, 0 => 2
  | 2, 1 => 4
  | 3, 0 => 4
  | 3, 1 => 8
  | _, _ => 0

/-- End-to-end `QkNormRope` (`T = 1`, `numQ = 1`, `numKV = 1`, `h = 1`, `bm = 1`, `eps = 3` under
`FOps.exact`): `qOut = [-1, 2]` and `kOut = [7/5, 26/5]`. -/
theorem qkNormRope_endToEnd (junk : ℕ → ℝ) :
    ∃ m', pallasCall (Examples.QkNormRope.qkNormRopeGridSpec 1 1 1 1 1 0 1 2 3 4 5)
        (Examples.QkNormRope.qkNormRopeBody Examples.QkNormRope.FOps.exact 1 1 1 3)
        junk qkInitMem = some m' ∧
      m'.read 4 0 = -1 ∧ m'.read 4 1 = 2 ∧
      m'.read 5 0 = 7 / 5 ∧ m'.read 5 1 = 26 / 5 := by
  have hdisj : ∀ b ∈ ([4, 5] : List BufId), b ≠ 0 ∧ b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 3 := by
    intro b hb
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hb
    rcases hb with rfl | rfl <;> decide
  obtain ⟨m', hcall, hout, -⟩ :=
    Examples.QkNormRope.qkNormRope_exact_correct 1 1 1 1 1 (by decide) 3 0 1 2 3 4 5 junk qkInitMem
      ⟨by norm_num, by decide, by decide, by decide, by decide, by decide, hdisj⟩
  refine ⟨m', hcall, ?_, ?_, ?_, ?_⟩
  · have h := congrArg (fun o => o.q.get3 ⟨0, by decide⟩ ⟨0, by decide⟩ ⟨0, by decide⟩) hout
    simp only [Examples.QkNormRope.qkNormRopeSpec, Examples.QkNormRope.qkNormRopeExactMath,
      Examples.QkNormRope.extractQ, Examples.QkNormRope.unpackCos, Examples.QkNormRope.unpackSin,
      Tensor.ropeNeox, Tensor.rmsNorm, Tensor.get3, Tensor.get2, Tensor.get1,
      Tensor.of3, Tensor.of2, Layout.rowMajor, Layout.read,
      Idx.flat_three, Idx.flat_two, Idx.flat_one, sum_fin_two_mul_one, Mem.read] at h
    rw [dite_eq_left (by decide)] at h
    norm_num [qkInitMem] at h
    exact h
  · have h := congrArg (fun o => o.q.get3 ⟨0, by decide⟩ ⟨0, by decide⟩ ⟨1, by decide⟩) hout
    simp only [Examples.QkNormRope.qkNormRopeSpec, Examples.QkNormRope.qkNormRopeExactMath,
      Examples.QkNormRope.extractQ, Examples.QkNormRope.unpackCos, Examples.QkNormRope.unpackSin,
      Tensor.ropeNeox, Tensor.rmsNorm, Tensor.get3, Tensor.get2, Tensor.get1,
      Tensor.of3, Tensor.of2, Layout.rowMajor, Layout.read,
      Idx.flat_three, Idx.flat_two, Idx.flat_one, sum_fin_two_mul_one, Mem.read] at h
    rw [dite_eq_right (by decide)] at h
    norm_num [qkInitMem] at h
    exact h
  · have h := congrArg (fun o => o.k.get3 ⟨0, by decide⟩ ⟨0, by decide⟩ ⟨0, by decide⟩) hout
    simp only [Examples.QkNormRope.qkNormRopeSpec, Examples.QkNormRope.qkNormRopeExactMath,
      Examples.QkNormRope.extractK, Examples.QkNormRope.unpackCos, Examples.QkNormRope.unpackSin,
      Tensor.ropeNeox, Tensor.rmsNorm, Tensor.get3, Tensor.get2, Tensor.get1,
      Tensor.of3, Tensor.of2, Layout.rowMajor, Layout.read,
      Idx.flat_three, Idx.flat_two, Idx.flat_one, sum_fin_two_mul_one, Mem.read] at h
    rw [dite_eq_left (by decide)] at h
    norm_num [qkInitMem] at h
    exact h
  · have h := congrArg (fun o => o.k.get3 ⟨0, by decide⟩ ⟨0, by decide⟩ ⟨1, by decide⟩) hout
    simp only [Examples.QkNormRope.qkNormRopeSpec, Examples.QkNormRope.qkNormRopeExactMath,
      Examples.QkNormRope.extractK, Examples.QkNormRope.unpackCos, Examples.QkNormRope.unpackSin,
      Tensor.ropeNeox, Tensor.rmsNorm, Tensor.get3, Tensor.get2, Tensor.get1,
      Tensor.of3, Tensor.of2, Layout.rowMajor, Layout.read,
      Idx.flat_three, Idx.flat_two, Idx.flat_one, sum_fin_two_mul_one, Mem.read] at h
    rw [dite_eq_right (by decide)] at h
    norm_num [qkInitMem] at h
    exact h

/-! ### 6. `BlockScaledMm` (128-block kernel with bias) -/

/-- Concrete initial memory for `blockwise128ScaledMm_endToEnd` (`gm = gn = gk = tileM = stepsK = 1`,
`tileN = 2`, `hasBias = true`):
`a` (`0`) has one non-zero entry `a[0, 0] = 4` so the 128-term block sum collapses to a single term;
`b` (`1`) has `b[0, 0] = 3`, `b[0, 1] = -5`; `scale_a` (`2`) is `[2]`; `scale_b` (`3`) is `[5, 3]`;
`bias` (`4`) is `[7, 10]`. -/
private def block128InitMem : Mem
  | 0, 0 => 4
  | 1, 0 => 3
  | 1, 1 => -5
  | 2, 0 => 2
  | 3, 0 => 5
  | 3, 1 => 3
  | 4, 0 => 7
  | 4, 1 => 10
  | _, _ => 0

private theorem block128InitMem_a_eq_zero {k : ℕ} (hk : k ≠ 0) : block128InitMem 0 k = 0 := by
  cases k with
  | zero => exact absurd rfl hk
  | succ _ => rfl

/-- End-to-end `BlockScaledMm` 128-block kernel with bias under `ScaledMmOps.exact`:
`out[0, 0] = 2 * 5 * (4 * 3) + 7 = 127` and `out[0, 1] = 2 * 3 * (4 * (-5)) + 10 = -110`. -/
theorem blockwise128ScaledMm_endToEnd (junk : ℕ → ℝ) :
    ∃ m', pallasCall (Examples.BlockScaledMm.blockwise128GridSpec 1 1 1 1 2 1 true 0 1 2 3 4 5)
        (Examples.BlockScaledMm.blockwise128Body Examples.BlockScaledMm.ScaledMmOps.exact 1 1 2 1 true)
        junk block128InitMem = some m' ∧
      m'.read 5 0 = 127 ∧ m'.read 5 1 = -110 := by
  obtain ⟨m', hcall, hval, -⟩ :=
    Examples.BlockScaledMm.blockwise128ScaledMm_correct Examples.BlockScaledMm.ScaledMmOps.exact
      1 1 1 1 2 1 true 0 1 2 3 4 5 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (fun _ => by decide)
      junk block128InitMem
  have hcol : ∀ c : Fin 2,
      m'.read 5 c.val =
        block128InitMem.read 2 0 * block128InitMem.read 3 c.val * (4 * block128InitMem.read 1 c.val) +
          block128InitMem.read 4 c.val := by
    intro c
    have h := hval ⟨0, by decide⟩ c
    rw [Examples.BlockScaledMm.blockwise128ScaledMmElem_exact] at h
    simp only [one_mul, Finset.sum_range_one, zero_mul, zero_add, ite_true] at h
    have hsum :
        (∑ k ∈ Finset.range 128,
          block128InitMem.read 0 k * block128InitMem.read 1 (k * 2 + c.val)) =
          4 * block128InitMem.read 1 c.val := by
      rw [Finset.sum_eq_single 0]
      · simp [block128InitMem, Mem.read]
      · intro k _ hk
        simp only [Mem.read, block128InitMem_a_eq_zero hk, zero_mul]
      · simp
    rw [hsum] at h
    exact h
  refine ⟨m', hcall, ?_, ?_⟩
  · have h := hcol ⟨0, by decide⟩
    norm_num [block128InitMem, Mem.read] at h
    exact h
  · have h := hcol ⟨1, by decide⟩
    norm_num [block128InitMem, Mem.read] at h
    exact h

/-! ### 7. `GroupedScaledMm` -/

/-- Dimensions for `groupedScaledMm_endToEnd`, matching `Tests.GroupedScaledMm.validSchedule_shared_tile`:
two real groups (`G = 2`), two M tiles of `tileM = 2` rows (`M = 3` unpadded rows, empty virtual
group `[3, 3)`), one N tile of width `1`, one K tile of width `2`. -/
private def groupedDims : Examples.GroupedScaledMm.Dims where
  G := 2
  gn := 1
  tilesM := 2
  tilesK := 1
  tileM := 2
  tileN := 1
  tileK := 2

/-- Distinct buffer ids for all 12 arguments of `GroupedScaledMm`. -/
private def groupedBufs : Examples.GroupedScaledMm.Bufs where
  goBuf := 0
  gidBuf := 1
  mtBuf := 2
  ngBuf := 3
  mProbeBuf := 4
  kProbeBuf := 5
  nProbeBuf := 6
  aBuf := 7
  bBuf := 8
  saBuf := 9
  sbBuf := 10
  oBuf := 11

/-- Concrete initial memory for `groupedScaledMm_endToEnd`:
scalar-prefetch buffers `0..3` hold the schedule tables of `Tests.GroupedScaledMm` and `G = 2`;
`a` (`7`, `[4, 2]`) has rows `[1, 2]`, `[3, 1]`, `[2, 4]`;
`b` (`8`, `[2, 2, 1]`) has group 0 `[5, 3]` and group 1 `[2, 7]`;
`scale_a` (`9`) has `[2, 3, 5]`; `scale_b` (`10`) has `[4, 2]`. -/
private def groupedInitMem : Mem
  | 0, i => (Tests.GroupedScaledMm.go i : ℝ)
  | 1, v => (Tests.GroupedScaledMm.gid v : ℝ)
  | 2, v => (Tests.GroupedScaledMm.mt v : ℝ)
  | 3, 0 => 2
  | 7, 0 => 1
  | 7, 1 => 2
  | 7, 2 => 3
  | 7, 3 => 1
  | 7, 4 => 2
  | 7, 5 => 4
  | 8, 0 => 5
  | 8, 1 => 3
  | 8, 2 => 2
  | 8, 3 => 7
  | 9, 0 => 2
  | 9, 1 => 3
  | 9, 2 => 5
  | 10, 0 => 4
  | 10, 1 => 2
  | _, _ => 0

/-- End-to-end `GroupedScaledMm` with two real groups sharing M tile `0` (reusing the schedule
tables of `Tests.GroupedScaledMm`, where `go = [0, 1, 3, 3]` has `M = 3` and an empty virtual
group): row `0` (group 0) is `88`, row `1` (group 1, same M tile) is `78`, and row `2` (group 1,
M tile 1) is `320`. -/
theorem groupedScaledMm_endToEnd (junk : ℕ → ℝ) :
    ∃ m', pallasCall (Examples.GroupedScaledMm.gridSpec groupedDims groupedBufs)
        (Examples.GroupedScaledMm.body Examples.GroupedScaledMm.GroupedMmOps.exact groupedDims)
        junk groupedInitMem = some m' ∧
      m'.read 11 0 = 88 ∧ m'.read 11 1 = 78 ∧ m'.read 11 2 = 320 := by
  have hreq : Examples.GroupedScaledMm.Requires groupedDims groupedBufs
      Tests.GroupedScaledMm.go Tests.GroupedScaledMm.gid Tests.GroupedScaledMm.mt 3
      groupedInitMem := by
    refine
      { sched := Tests.GroupedScaledMm.validSchedule_shared_tile
        gn_pos := by decide
        tilesK_pos := by decide
        tileM_pos := by decide
        tileN_pos := by decide
        tileK_pos := by decide
        read_go := fun i hi => by
          dsimp only [groupedDims, groupedBufs] at hi ⊢
          refine (decodeInt32_eq_some_iff _ _).mpr ⟨by simp [groupedInitMem, Mem.read], ?_⟩
          interval_cases i <;> decide
        read_gid := fun v hv => by
          dsimp only [groupedDims, groupedBufs] at hv ⊢
          refine (decodeInt32_eq_some_iff _ _).mpr ⟨by simp [groupedInitMem, Mem.read], ?_⟩
          interval_cases v <;> decide
        read_mt := fun v hv => by
          dsimp only [groupedDims, groupedBufs] at hv ⊢
          refine (decodeInt32_eq_some_iff _ _).mpr ⟨by simp [groupedInitMem, Mem.read], ?_⟩
          interval_cases v <;> decide
        read_ng := (decodeInt32_eq_some_iff _ _).mpr
          ⟨by norm_num [groupedInitMem, Mem.read, groupedDims, groupedBufs], by decide⟩
        out_ne := by decide }
  obtain ⟨m', hcall, hval, -⟩ :=
    Examples.GroupedScaledMm.rowwiseScaledGroupedMm_correct
      Examples.GroupedScaledMm.GroupedMmOps.exact hreq junk
  refine ⟨m', hcall, ?_, ?_, ?_⟩
  · have h := hval 0 (by decide) 0 (by decide) 0 (by decide) (by decide) (by decide)
    simp only [groupedDims, groupedBufs, show (0 : ℕ) < 2 by decide, ite_true,
      Examples.GroupedScaledMm.groupedElem_exact, one_mul, Finset.sum_range_succ,
      Finset.sum_range_zero] at h
    norm_num [groupedInitMem, Mem.read] at h
    exact h
  · have h := hval 1 (by decide) 0 (by decide) 1 (by decide) (by decide) (by decide)
    simp only [groupedDims, groupedBufs, show (1 : ℕ) < 2 by decide, ite_true,
      Examples.GroupedScaledMm.groupedElem_exact, one_mul, Finset.sum_range_succ,
      Finset.sum_range_zero] at h
    norm_num [groupedInitMem, Mem.read] at h
    exact h
  · have h := hval 2 (by decide) 0 (by decide) 1 (by decide) (by decide) (by decide)
    simp only [groupedDims, groupedBufs, show (1 : ℕ) < 2 by decide, ite_true,
      Examples.GroupedScaledMm.groupedElem_exact, one_mul, Finset.sum_range_succ,
      Finset.sum_range_zero] at h
    norm_num [groupedInitMem, Mem.read] at h
    exact h

end

end TpuKernelLean.Tests.EndToEnd
