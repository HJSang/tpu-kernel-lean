import TpuKernelLean.Examples.QkNormRope

/-!
Boundary, non-interference, and negative tests for `Examples/QkNormRope.lean`.
-/

namespace TpuKernelLean.Tests.QkNormRope

open TpuKernelLean.Pallas TpuKernelLean.Examples.QkNormRope

/-- Modifying any `v` column in `qkvBuf` leaves both `q` and `k` outputs identical bit for bit. -/
theorem v_mutation_no_effect (ops : FOps) (T numQ numKV h bm : ℕ) (hbm : 0 < bm)
    (eps : ℝ) (heps : 0 < eps) (hnumQ : 0 < numQ) (hnumKV : 0 < numKV) (hh : 0 < h)
    (junk : ℕ → ℝ) (m0 : Mem) (tv : Fin T) (hv : Fin numKV) (dv : Fin (2 * h)) (val : ℝ) :
    let qkvWidth := (numQ + 2 * numKV) * (2 * h)
    let vOff := tv.val * qkvWidth + ((numQ + numKV) * (2 * h) + hv.val * (2 * h) + dv.val)
    let m0' := m0.write 0 vOff val
    let gs := qkNormRopeGridSpec T numQ numKV h bm 0 1 2 3 4 5
    let body := qkNormRopeBody ops numQ numKV h eps
    ∃ m₁ m₂,
      pallasCall gs body junk m0 = some m₁ ∧
      pallasCall gs body junk m0' = some m₂ ∧
      (∀ (t : Fin T) (hd : Fin numQ) (d : Fin (2 * h)),
        m₁.read 4 (t.val * (numQ * (2 * h)) + (hd.val * (2 * h) + d.val)) =
          m₂.read 4 (t.val * (numQ * (2 * h)) + (hd.val * (2 * h) + d.val))) ∧
      (∀ (t : Fin T) (hd : Fin numKV) (d : Fin (2 * h)),
        m₁.read 5 (t.val * (numKV * (2 * h)) + (hd.val * (2 * h) + d.val)) =
          m₂.read 5 (t.val * (numKV * (2 * h)) + (hd.val * (2 * h) + d.val))) := by
  intro qkvWidth vOff m0' gs body
  have hdisj : ∀ b ∈ ([4, 5] : List BufId), b ≠ 0 ∧ b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 3 := by
    intro b hb
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hb
    rcases hb with rfl | rfl <;> decide
  have hT : 0 < T := Nat.lt_of_le_of_lt (Nat.zero_le _) tv.isLt
  obtain ⟨m₁, h₁, -, -, -⟩ :=
    qkNormRope_pallas_correct ops T numQ numKV h bm hbm eps 0 1 2 3 4 5 junk m0
      ⟨heps, hT, hnumQ, hnumKV, hh, by decide, hdisj⟩
  obtain ⟨m₂, h₂, -, -, -⟩ :=
    qkNormRope_pallas_correct ops T numQ numKV h bm hbm eps 0 1 2 3 4 5 junk m0'
      ⟨heps, hT, hnumQ, hnumKV, hh, by decide, hdisj⟩
  have hbi : ∀ t : Fin T,
      (∀ (hd : Fin numQ) (d : Fin (2 * h)),
        m₁.read 4 (t.val * (numQ * (2 * h)) + (hd.val * (2 * h) + d.val)) =
          m₂.read 4 (t.val * (numQ * (2 * h)) + (hd.val * (2 * h) + d.val))) ∧
      (∀ (hd : Fin numKV) (d : Fin (2 * h)),
        m₁.read 5 (t.val * (numKV * (2 * h)) + (hd.val * (2 * h) + d.val)) =
          m₂.read 5 (t.val * (numKV * (2 * h)) + (hd.val * (2 * h) + d.val))) := fun t =>
    qkNormRope_batch_invariant ops hbm hbm heps hnumQ hnumKV hh
      (by decide) hdisj (by decide) hdisj h₁ h₂ t t
      (fun c hc => by
        symm
        apply Mem.read_write_ne
        intro heq
        have hoff := (Prod.mk.inj heq).2
        have hlt1 : c < qkvWidth := by
          have : (numQ + numKV) * (2 * h) ≤ qkvWidth := by dsimp [qkvWidth]; nlinarith
          omega
        have hlt2 : (numQ + numKV) * (2 * h) + hv.val * (2 * h) + dv.val < qkvWidth := by
          dsimp [qkvWidth]
          nlinarith [hv.isLt, dv.isLt]
        have hmod := congrArg (· % qkvWidth) hoff
        dsimp only [vOff] at hmod
        rw [Nat.mul_comm t.val qkvWidth, Nat.mul_add_mod, Nat.mod_eq_of_lt hlt1,
          Nat.mul_comm tv.val qkvWidth, Nat.mul_add_mod, Nat.mod_eq_of_lt hlt2] at hmod
        omega)
      (fun _ => (Mem.read_write_buf_ne m0 _ _ val (by decide)).symm)
      (fun _ => (Mem.read_write_buf_ne m0 _ _ val (by decide)).symm)
      (fun _ => (Mem.read_write_buf_ne m0 _ _ val (by decide)).symm)
  exact ⟨m₁, m₂, h₁, h₂, fun t => (hbi t).1, fun t => (hbi t).2⟩

/-- Empty batch `T = 0` produces a zero grid dimension and is rejected by `pallasCall` (matching
`tpu_fused_qk_norm_rope` rejecting `m = 0` and Mosaic requiring strictly positive grid dims). -/
theorem empty_batch_rejected (ops : FOps) (eps : ℝ) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall (qkNormRopeGridSpec 0 4 2 64 maxBlockTokens 0 1 2 3 4 5)
      (qkNormRopeBody ops 4 2 64 eps) junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have hpos := h.grid_pos 0 (by simp [qkNormRopeGridSpec, maxBlockTokens])
  omega

/-- Non-divisible batch `T = 300` with `defaultBlockTokens 300 = 256` (two blocks with edge
padding on the second) is well defined and satisfies the spec. -/
theorem non_divisible_default_block (ops : FOps) (eps : ℝ) (heps : 0 < eps)
    (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m',
      pallasCall (qkNormRopeGridSpec 300 4 2 64 (defaultBlockTokens 300) 0 1 2 3 4 5)
        (qkNormRopeBody ops 4 2 64 eps) junk m0 = some m' ∧
      m'.read 0 (299 * (8 * 128) + (6 * 128 + 0)) = m0.read 0 (299 * (8 * 128) + (6 * 128 + 0)) := by
  have hdisj : ∀ b ∈ ([4, 5] : List BufId), b ≠ 0 ∧ b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 3 := by
    intro b hb
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hb
    rcases hb with rfl | rfl <;> decide
  obtain ⟨m', hcall, -, -, -⟩ :=
    qkNormRope_defaultBlock_correct ops 300 4 2 64 (by decide) eps 0 1 2 3 4 5 junk m0
      ⟨heps, by decide, by decide, by decide, by decide, by decide, hdisj⟩
  have hv := qkNormRope_v_unchanged ops 300 4 2 64 (defaultBlockTokens 300)
    (by simp [defaultBlockTokens, maxBlockTokens]) eps heps (by decide) (by decide) (by decide)
    0 1 2 3 4 5 (by decide) hdisj junk m0 m' hcall ⟨299, by decide⟩ ⟨0, by decide⟩ ⟨0, by decide⟩
  exact ⟨m', hcall, by simpa using hv⟩

/-- Negative test: zero head dimension (`h = 0`) on a non-empty batch violates `blocks_ok`, so
`pallasCall` returns `none`. -/
theorem zero_head_dim_rejected (ops : FOps) (eps : ℝ) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall (qkNormRopeGridSpec 1 1 1 0 1 0 1 2 3 4 5)
      (qkNormRopeBody ops 1 1 0 eps) junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have hblk := h.blocks_ok 0 (by simp [qkNormRopeGridSpec, gridPoints_length])
    ({ buf := 2, shape := [0] }, vecBlockSpec 0) (by simp [qkNormRopeGridSpec])
  simp [blockOk, GridSpec.idxAt, qkNormRopeGridSpec, vecBlockSpec] at hblk

/-- Negative test: aliasing `qOutBuf` with `qwBuf` on a 2-block grid violates `no_stale_read`
(step 0 writes block 0 while step 1 re-reads `qwBuf`), so `pallasCall` returns `none`. -/
theorem aliased_weight_rejected (ops : FOps) (eps : ℝ) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall (qkNormRopeGridSpec 2 1 1 1 1 0 1 2 3 2 5)
      (qkNormRopeBody ops 1 1 1 eps) junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  apply h.no_stale_read 0 1 (by decide) (by simp [qkNormRopeGridSpec, gridPoints_length])
    ({ buf := 2, shape := [2] }, vecBlockSpec 2) (by simp [qkNormRopeGridSpec])
    ({ buf := 2, shape := [2, 2] }, rowBlockSpec 1 2) (by simp [qkNormRopeGridSpec])
  refine ⟨rfl, 0, ?_, ?_⟩ <;>
    simp [GridSpec.idxAt, qkNormRopeGridSpec, vecBlockSpec, rowBlockSpec, gridPoints,
      List.range_succ, addrInBlock, coordInBlock, unflatCoord]

end TpuKernelLean.Tests.QkNormRope
