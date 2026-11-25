import TpuKernelLean.Tensor

/-!
Mathematical specification and online-softmax block-equivalence proof for single-head
Flash Attention (`sglang_tpu/kernels/flash_attention.py:232–372, 688–729`).
-/

namespace TpuKernelLean.Examples.FlashAttention

/-- Concatenate a row sequence of length `a` and a non-empty sequence of length `b + 1` into one of length `(a + b) + 1`. -/
def concatRows {a b : ℕ} {α : Type*} (u : Fin a → α) (v : Fin (b + 1) → α) :
    Fin ((a + b) + 1) → α :=
  fun i =>
    if h : i.val < a then
      u ⟨i.val, h⟩
    else
      v ⟨i.val - a, by omega⟩

theorem concatRows_castAdd {a b : ℕ} {α : Type*} (u : Fin a → α) (v : Fin (b + 1) → α)
    (i : Fin a) :
    concatRows u v (Fin.castAdd (b + 1) i) = u i := by
  simp [concatRows, Fin.castAdd, i.isLt]

theorem concatRows_natAdd {a b : ℕ} {α : Type*} (u : Fin a → α) (v : Fin (b + 1) → α)
    (j : Fin (b + 1)) :
    concatRows u v (Fin.natAdd a j) = v j := by
  simp only [concatRows, Fin.natAdd]
  split_ifs with hlt
  · omega
  · congr 1
    exact Fin.ext (by dsimp; omega)

/-- Running online-softmax state `(m, ℓ, acc)` for one query row of head dimension `D`. -/
structure OnlineState (D : ℕ) where
  /-- Running maximum attention logit over all KV blocks consumed so far. -/
  m : ℝ
  /-- Running sum of shifted exponentials `∑ exp(s_j - m)`. -/
  ℓ : ℝ
  /-- Running unnormalized weighted-value accumulator `∑ exp(s_j - m) * V_j`. -/
  acc : Fin D → ℝ

/-- Total KV sequence length minus one for `n + 1` blocks of size `B + 1`. -/
def blockLen (B : ℕ) : ℕ → ℕ
  | 0 => B
  | n + 1 => (blockLen B n + 1) + B

/-- `blockLen B n + 1` equals `(n + 1) * (B + 1)`. -/
theorem blockLen_succ_eq (B : ℕ) : ∀ n : ℕ, blockLen B n + 1 = (n + 1) * (B + 1)
  | 0 => by simp [blockLen]
  | n + 1 => by
    have ih := blockLen_succ_eq B n
    simp only [blockLen]
    rw [ih]
    ring

/-- Flatten `n + 1` KV blocks of size `B + 1` into a single contiguous sequence. -/
def flattenBlocks {B : ℕ} {α : Type*} :
    {n : ℕ} → (Fin (n + 1) → Fin (B + 1) → α) → Fin (blockLen B n + 1) → α
  | 0, blks => blks 0
  | n + 1, blks =>
    concatRows (flattenBlocks fun b => blks b.castSucc) (blks (Fin.last (n + 1)))

noncomputable section

/-- Single-head scaled dot-product attention `softmax(Q Kᵀ * smScale) V` (`flash_attention.py:700–726`). -/
def flashAttentionMath {L S D : ℕ} (smScale : ℝ)
    (Q : Tensor ℝ [L, D]) (K V : Tensor ℝ [S + 1, D]) : Tensor ℝ [L, D] :=
  let scores := Tensor.scale smScale (Tensor.matmul Q (Tensor.transpose K))
  let weights := Tensor.softmax scores
  Tensor.matmul weights V

/-- Single query-row scaled dot-product attention over a non-empty KV sequence of length `S + 1`. -/
def attentionRow {S D : ℕ} (smScale : ℝ) (q : Fin D → ℝ)
    (K V : Fin (S + 1) → Fin D → ℝ) : Fin D → ℝ :=
  let s : Fin (S + 1) → ℝ := fun j => smScale * ∑ d : Fin D, q d * K j d
  let m := Tensor.vecMax s
  let ℓ := ∑ j : Fin (S + 1), Real.exp (s j - m)
  fun d => (∑ j : Fin (S + 1), Real.exp (s j - m) * V j d) / ℓ

/-- Row `i` of `flashAttentionMath` agrees with `attentionRow`. -/
theorem flashAttentionMath_eq_attentionRow {L S D : ℕ} (smScale : ℝ)
    (Q : Tensor ℝ [L, D]) (K V : Tensor ℝ [S + 1, D]) (i : Fin L) (d : Fin D) :
    (flashAttentionMath smScale Q K V).get2 i d =
      attentionRow smScale (fun c => Q.get2 i c) (fun j c => K.get2 j c)
        (fun j c => V.get2 j c) d := by
  simp only [flashAttentionMath, attentionRow, Tensor.get2_matmul, Tensor.softmax,
    Tensor.get2_of2, Tensor.get1_rowMax, Tensor.get2_scale, Tensor.get2_transpose]
  rw [Finset.sum_div]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [div_mul_eq_mul_div]

/-- Initialize the online-softmax state from the first non-empty KV block. -/
def OnlineState.initBlock {B D : ℕ} (smScale : ℝ) (q : Fin D → ℝ)
    (K V : Fin (B + 1) → Fin D → ℝ) : OnlineState D :=
  let s : Fin (B + 1) → ℝ := fun j => smScale * ∑ d : Fin D, q d * K j d
  let m := Tensor.vecMax s
  { m := m
    ℓ := ∑ j : Fin (B + 1), Real.exp (s j - m)
    acc := fun d => ∑ j : Fin (B + 1), Real.exp (s j - m) * V j d }

/-- Consume one non-empty KV block `(K, V)`, rescaling the unnormalized `(ℓ, acc)` accumulator by
`exp(m_prev - m_next)` and normalizing once at the end via `OnlineState.normalize`. Note: this
differs from `sglang_tpu/kernels/flash_attention.py:329–372`, which updates `(m, ℓ)` the same way
(lines 329–343) but renormalizes `acc_scratch_ref` by `1 / l_next` on every step (lines 357–363)
with no final division in `store_output` (lines 365–372). -/
def OnlineState.step {B D : ℕ} (smScale : ℝ) (q : Fin D → ℝ)
    (st : OnlineState D) (K V : Fin (B + 1) → Fin D → ℝ) : OnlineState D :=
  let s : Fin (B + 1) → ℝ := fun j => smScale * ∑ d : Fin D, q d * K j d
  let mCurr := Tensor.vecMax s
  let mNext := max st.m mCurr
  let alpha := Real.exp (st.m - mNext)
  { m := mNext
    ℓ := alpha * st.ℓ + ∑ j : Fin (B + 1), Real.exp (s j - mNext)
    acc := fun d => alpha * st.acc d + ∑ j : Fin (B + 1), Real.exp (s j - mNext) * V j d }

/-- Normalize the accumulated online-softmax output as `acc / ℓ`. -/
def OnlineState.normalize {D : ℕ} (st : OnlineState D) : Fin D → ℝ :=
  fun d => st.acc d / st.ℓ

/-- Run online softmax over `n + 1` consecutive KV blocks of size `B + 1`. -/
def runBlocks {B D : ℕ} (smScale : ℝ) (q : Fin D → ℝ) :
    {n : ℕ} → (Fin (n + 1) → Fin (B + 1) → Fin D → ℝ) →
    (Fin (n + 1) → Fin (B + 1) → Fin D → ℝ) → OnlineState D
  | 0, Ks, Vs => OnlineState.initBlock smScale q (Ks 0) (Vs 0)
  | n + 1, Ks, Vs =>
    let prev := runBlocks smScale q (fun b => Ks b.castSucc) (fun b => Vs b.castSucc)
    OnlineState.step smScale q prev (Ks (Fin.last (n + 1))) (Vs (Fin.last (n + 1)))

/-- The maximum of two concatenated non-empty vectors is the `max` of their maxima. -/
theorem vecMax_concatRows {B₁ B₂ : ℕ} (s₁ : Fin (B₁ + 1) → ℝ) (s₂ : Fin (B₂ + 1) → ℝ) :
    max (Tensor.vecMax s₁) (Tensor.vecMax s₂) = Tensor.vecMax (concatRows s₁ s₂) := by
  apply le_antisymm
  · apply max_le
    · apply Tensor.vecMax_le
      intro i
      have h := Tensor.le_vecMax (concatRows s₁ s₂) (Fin.castAdd (B₂ + 1) i)
      rwa [concatRows_castAdd] at h
    · apply Tensor.vecMax_le
      intro j
      have h := Tensor.le_vecMax (concatRows s₁ s₂) (Fin.natAdd (B₁ + 1) j)
      rwa [concatRows_natAdd] at h
  · apply Tensor.vecMax_le
    intro k
    by_cases hk : k.val < B₁ + 1
    · have h1 : concatRows s₁ s₂ k = s₁ ⟨k.val, hk⟩ := by simp [concatRows, hk]
      rw [h1]
      exact le_trans (Tensor.le_vecMax s₁ ⟨k.val, hk⟩) (le_max_left _ _)
    · have h2 : concatRows s₁ s₂ k = s₂ ⟨k.val - (B₁ + 1), by omega⟩ := by
        simp [concatRows, hk]
      rw [h2]
      exact le_trans (Tensor.le_vecMax s₂ ⟨k.val - (B₁ + 1), by omega⟩) (le_max_right _ _)

/-- Key 2-block invariant: `initBlock` on block 1 followed by `step` on block 2 produces the exact state of `initBlock` on the concatenated sequence. -/
theorem onlineState_two_blocks {B₁ B₂ D : ℕ} (smScale : ℝ) (q : Fin D → ℝ)
    (K₁ V₁ : Fin (B₁ + 1) → Fin D → ℝ) (K₂ V₂ : Fin (B₂ + 1) → Fin D → ℝ) :
    OnlineState.step smScale q (OnlineState.initBlock smScale q K₁ V₁) K₂ V₂ =
      OnlineState.initBlock smScale q (concatRows K₁ K₂) (concatRows V₁ V₂) := by
  let s₁ : Fin (B₁ + 1) → ℝ := fun i => smScale * ∑ d : Fin D, q d * K₁ i d
  let s₂ : Fin (B₂ + 1) → ℝ := fun j => smScale * ∑ d : Fin D, q d * K₂ j d
  let sFull : Fin (((B₁ + 1) + B₂) + 1) → ℝ :=
    fun k => smScale * ∑ d : Fin D, q d * concatRows K₁ K₂ k d
  have hsFull : sFull = concatRows s₁ s₂ := by
    funext k
    dsimp only [sFull, s₁, s₂, concatRows]
    split_ifs <;> rfl
  have hm : max (Tensor.vecMax s₁) (Tensor.vecMax s₂) = Tensor.vecMax sFull := by
    rw [hsFull]
    exact vecMax_concatRows s₁ s₂
  simp only [OnlineState.step, OnlineState.initBlock]
  change OnlineState.mk
      (max (Tensor.vecMax s₁) (Tensor.vecMax s₂))
      (Real.exp (Tensor.vecMax s₁ - max (Tensor.vecMax s₁) (Tensor.vecMax s₂)) *
          (∑ i : Fin (B₁ + 1), Real.exp (s₁ i - Tensor.vecMax s₁)) +
        ∑ j : Fin (B₂ + 1), Real.exp (s₂ j - max (Tensor.vecMax s₁) (Tensor.vecMax s₂)))
      (fun d =>
        Real.exp (Tensor.vecMax s₁ - max (Tensor.vecMax s₁) (Tensor.vecMax s₂)) *
            (∑ i : Fin (B₁ + 1), Real.exp (s₁ i - Tensor.vecMax s₁) * V₁ i d) +
          ∑ j : Fin (B₂ + 1),
            Real.exp (s₂ j - max (Tensor.vecMax s₁) (Tensor.vecMax s₂)) * V₂ j d) =
    OnlineState.mk
      (Tensor.vecMax sFull)
      (∑ k : Fin (((B₁ + 1) + B₂) + 1), Real.exp (sFull k - Tensor.vecMax sFull))
      (fun d => ∑ k : Fin (((B₁ + 1) + B₂) + 1),
        Real.exp (sFull k - Tensor.vecMax sFull) * concatRows V₁ V₂ k d)
  rw [hm]
  have hℓ :
      Real.exp (Tensor.vecMax s₁ - Tensor.vecMax sFull) *
          (∑ i : Fin (B₁ + 1), Real.exp (s₁ i - Tensor.vecMax s₁)) +
        ∑ j : Fin (B₂ + 1), Real.exp (s₂ j - Tensor.vecMax sFull) =
      ∑ k : Fin (((B₁ + 1) + B₂) + 1), Real.exp (sFull k - Tensor.vecMax sFull) := by
    have hsum :
        (∑ k : Fin (((B₁ + 1) + B₂) + 1), Real.exp (sFull k - Tensor.vecMax sFull)) =
        (∑ i : Fin (B₁ + 1),
          Real.exp (sFull (Fin.castAdd (B₂ + 1) i) - Tensor.vecMax sFull)) +
        ∑ j : Fin (B₂ + 1),
          Real.exp (sFull (Fin.natAdd (B₁ + 1) j) - Tensor.vecMax sFull) :=
      Fin.sum_univ_add (f := fun k : Fin ((B₁ + 1) + (B₂ + 1)) =>
        Real.exp (sFull k - Tensor.vecMax sFull))
    rw [hsum, Finset.mul_sum]
    refine congrArg₂ (· + ·) ?_ ?_
    · refine Finset.sum_congr rfl fun i _ => ?_
      rw [← Real.exp_add, hsFull, concatRows_castAdd]
      have hexp : (Tensor.vecMax s₁ - Tensor.vecMax (concatRows s₁ s₂)) +
          (s₁ i - Tensor.vecMax s₁) = s₁ i - Tensor.vecMax (concatRows s₁ s₂) := by ring
      rw [hexp]
    · refine Finset.sum_congr rfl fun j _ => ?_
      rw [hsFull, concatRows_natAdd]
  have hacc :
      (fun d =>
        Real.exp (Tensor.vecMax s₁ - Tensor.vecMax sFull) *
            (∑ i : Fin (B₁ + 1), Real.exp (s₁ i - Tensor.vecMax s₁) * V₁ i d) +
          ∑ j : Fin (B₂ + 1), Real.exp (s₂ j - Tensor.vecMax sFull) * V₂ j d) =
      (fun d => ∑ k : Fin (((B₁ + 1) + B₂) + 1),
        Real.exp (sFull k - Tensor.vecMax sFull) * concatRows V₁ V₂ k d) := by
    funext d
    have hsum :
        (∑ k : Fin (((B₁ + 1) + B₂) + 1),
          Real.exp (sFull k - Tensor.vecMax sFull) * concatRows V₁ V₂ k d) =
        (∑ i : Fin (B₁ + 1),
          Real.exp (sFull (Fin.castAdd (B₂ + 1) i) - Tensor.vecMax sFull) *
            concatRows V₁ V₂ (Fin.castAdd (B₂ + 1) i) d) +
        ∑ j : Fin (B₂ + 1),
          Real.exp (sFull (Fin.natAdd (B₁ + 1) j) - Tensor.vecMax sFull) *
            concatRows V₁ V₂ (Fin.natAdd (B₁ + 1) j) d :=
      Fin.sum_univ_add (f := fun k : Fin ((B₁ + 1) + (B₂ + 1)) =>
        Real.exp (sFull k - Tensor.vecMax sFull) * concatRows V₁ V₂ k d)
    rw [hsum, Finset.mul_sum]
    refine congrArg₂ (· + ·) ?_ ?_
    · refine Finset.sum_congr rfl fun i _ => ?_
      rw [← mul_assoc, ← Real.exp_add, hsFull, concatRows_castAdd, concatRows_castAdd]
      have hexp : (Tensor.vecMax s₁ - Tensor.vecMax (concatRows s₁ s₂)) +
          (s₁ i - Tensor.vecMax s₁) = s₁ i - Tensor.vecMax (concatRows s₁ s₂) := by ring
      rw [hexp]
    · refine Finset.sum_congr rfl fun j _ => ?_
      rw [hsFull, concatRows_natAdd, concatRows_natAdd]
  exact congrArg₂ (OnlineState.mk (Tensor.vecMax sFull)) hℓ hacc

/-- Two-block Flash Attention equivalence: processing two KV blocks with `step` and normalizing equals `attentionRow` over the full sequence. -/
theorem two_blocks_eq_attentionRow {B₁ B₂ D : ℕ} (smScale : ℝ) (q : Fin D → ℝ)
    (K₁ V₁ : Fin (B₁ + 1) → Fin D → ℝ) (K₂ V₂ : Fin (B₂ + 1) → Fin D → ℝ) :
    (OnlineState.step smScale q (OnlineState.initBlock smScale q K₁ V₁) K₂ V₂).normalize =
      attentionRow smScale q (concatRows K₁ K₂) (concatRows V₁ V₂) := by
  rw [onlineState_two_blocks]
  rfl

/-- General `n`-block invariant: processing `n + 1` blocks sequentially equals `initBlock` on the flattened KV sequence. -/
theorem runBlocks_eq_initBlock {B D : ℕ} (smScale : ℝ) (q : Fin D → ℝ) :
    ∀ {n : ℕ} (Ks Vs : Fin (n + 1) → Fin (B + 1) → Fin D → ℝ),
      runBlocks smScale q Ks Vs =
        OnlineState.initBlock smScale q (flattenBlocks Ks) (flattenBlocks Vs)
  | 0, _, _ => rfl
  | n + 1, Ks, Vs => by
    simp only [runBlocks, flattenBlocks]
    rw [runBlocks_eq_initBlock smScale q (fun b => Ks b.castSucc) (fun b => Vs b.castSucc)]
    exact onlineState_two_blocks smScale q _ _ _ _

/-- General `n`-block Flash Attention equivalence: normalizing `runBlocks` equals `attentionRow` over the full flattened sequence. -/
theorem runBlocks_normalize_eq_attentionRow {B D n : ℕ} (smScale : ℝ) (q : Fin D → ℝ)
    (Ks Vs : Fin (n + 1) → Fin (B + 1) → Fin D → ℝ) :
    (runBlocks smScale q Ks Vs).normalize =
      attentionRow smScale q (flattenBlocks Ks) (flattenBlocks Vs) := by
  rw [runBlocks_eq_initBlock]
  rfl

end

end TpuKernelLean.Examples.FlashAttention
