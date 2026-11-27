import TpuKernelLean.Dtype
import TpuKernelLean.Spec
import TpuKernelLean.Pallas.Lemmas

/-!
Specification and end-to-end verification of `sglang_tpu/kernels/router_topk.py`
(`router_topk_jax`): sigmoid MoE routing with a per-expert correction bias. For each token row,
`scores = sigmoid(logits)` rounded to the logits dtype, `choice = scores + bias`, the `k` experts
first in (choice descending, expert id ascending) order are selected, their unbiased scores are
gathered and optionally renormalized.

**Tie rule.** Expert `j` comes before expert `e` (`Before c j e`) when `c e < c j`, or
`c j = c e` and `j < e`: equal biased scores go to the lower expert id. `rank c e` counts the
experts before `e`; `atRank c i` is the expert of rank `i`. `atRank_before_atRank` and
`atRank_before_unselected` state the order the selected ids are returned in and that no
unselected expert beats a selected one.

**Kernel.** `pallas_call` with `grid=(1,)` and whole-array blocks (`router_topk.py:170-183`). The
body (`_topk_kernel`, `router_topk.py:20-89`) has three code paths, all modelled:
- one token, `experts % 128 == 0`, `k <= experts <= 1024`: `_topk_kernel_1tok`
  (`router_topk.py:92-151`), the rank of every expert from tiled compares, then slot `r` takes the
  expert of rank `r`;
- otherwise `k` rounds of argmax-and-mask (`router_topk.py:73-84`): with bf16 logits the argmax
  is one `min` over packed keys `expert * 2^16 + score bits` (`router_topk.py:63-69,76-78`), with
  any other dtype a `min` over tied lanes and a masked sum (`router_topk.py:80-81`).

`routerTopk_correct` proves every path defined and equal to `mathRow`: ids are `atRank` of the
biased scores, weights are the renormalized unbiased scores at those ids, nothing outside the two
output buffers changes. Float ops are opaque (`RouterOps`), so the claim is bit for bit for any
rounding the TPU does. Comparisons, `max`, `min` over integers, `where`, and sums whose terms are
all exact zeros but one are exact on IEEE values and are modelled exactly; `-inf` (the mask) is
`⊥` of `WithBot ℝ`.
-/

namespace TpuKernelLean.Examples.RouterTopk

open TpuKernelLean.Pallas

noncomputable section

open Classical

/-! ### The tie rule and the selection order -/

section Order

variable {E : ℕ}

/-- Expert `j` comes before expert `e`: a higher biased score, or an equal one with a lower id
(`router_topk.py:5-6,97-99`). -/
def Before (c : Fin E → ℝ) (j e : Fin E) : Prop :=
  c e < c j ∨ (c j = c e ∧ j < e)

theorem before_irrefl (c : Fin E → ℝ) (e : Fin E) : ¬ Before c e e := by
  rintro (h | ⟨_, h⟩) <;> exact lt_irrefl _ h

theorem before_trans (c : Fin E → ℝ) {a b d : Fin E} (h₁ : Before c a b) (h₂ : Before c b d) :
    Before c a d := by
  rcases h₁ with h₁ | ⟨e₁, l₁⟩ <;> rcases h₂ with h₂ | ⟨e₂, l₂⟩
  · exact Or.inl (lt_trans h₂ h₁)
  · left
    rw [← e₂]
    exact h₁
  · left
    rw [e₁]
    exact h₂
  · exact Or.inr ⟨e₁.trans e₂, lt_trans l₁ l₂⟩

theorem before_total (c : Fin E → ℝ) {j e : Fin E} (h : j ≠ e) :
    Before c j e ∨ Before c e j := by
  rcases lt_trichotomy (c e) (c j) with hlt | heq | hgt
  · exact Or.inl (Or.inl hlt)
  · rcases lt_or_gt_of_ne h with hl | hl
    · exact Or.inl (Or.inr ⟨heq.symm, hl⟩)
    · exact Or.inr (Or.inr ⟨heq, hl⟩)
  · exact Or.inr (Or.inl hgt)

/-- Position of `e` in (choice descending, expert ascending) order: how many experts come before
it. -/
def rank (c : Fin E → ℝ) (e : Fin E) : ℕ :=
  (Finset.univ.filter fun j => Before c j e).card

theorem rank_lt_rank (c : Fin E → ℝ) {e e' : Fin E} (h : Before c e e') :
    rank c e < rank c e' := by
  apply Finset.card_lt_card
  rw [Finset.ssubset_iff_of_subset]
  · exact ⟨e, by simp [h], by simp [before_irrefl]⟩
  · intro j hj
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj ⊢
    exact before_trans c hj h

theorem rank_lt_rank_iff (c : Fin E → ℝ) {e e' : Fin E} :
    rank c e < rank c e' ↔ Before c e e' := by
  refine ⟨fun h => ?_, rank_lt_rank c⟩
  by_cases he : e = e'
  · subst he
    exact absurd h (lt_irrefl _)
  · rcases before_total c he with h' | h'
    · exact h'
    · exact absurd (rank_lt_rank c h') (not_lt.mpr h.le)

theorem rank_injective (c : Fin E → ℝ) : Function.Injective (rank c) := by
  intro e e' h
  by_contra hne
  rcases before_total c hne with h' | h'
  · exact (rank_lt_rank c h').ne h
  · exact (rank_lt_rank c h').ne h.symm

theorem rank_lt (c : Fin E → ℝ) (e : Fin E) : rank c e < E := by
  have hss : (Finset.univ.filter fun j => Before c j e) ⊂ Finset.univ :=
    (Finset.ssubset_iff_of_subset (Finset.subset_univ _)).mpr
      ⟨e, Finset.mem_univ _, by simp [before_irrefl]⟩
  simpa [rank] using Finset.card_lt_card hss

/-- Every position `r < E` holds exactly one expert. -/
theorem existsUnique_rank (c : Fin E → ℝ) {r : ℕ} (hr : r < E) : ∃! e, rank c e = r := by
  have hinj : Function.Injective fun e : Fin E => (⟨rank c e, rank_lt c e⟩ : Fin E) :=
    fun e e' h => rank_injective c (by simpa using congrArg Fin.val h)
  obtain ⟨e, he⟩ := (Finite.injective_iff_surjective.mp hinj) ⟨r, hr⟩
  have he' : rank c e = r := by simpa using congrArg Fin.val he
  exact ⟨e, he', fun e' h => rank_injective c (h.trans he'.symm)⟩

/-- The expert at position `r < E`. -/
def atRankFin (c : Fin E → ℝ) (r : ℕ) (hr : r < E) : Fin E :=
  Fintype.choose (fun e => rank c e = r) (existsUnique_rank c hr)

theorem rank_atRankFin (c : Fin E → ℝ) (r : ℕ) (hr : r < E) : rank c (atRankFin c r hr) = r :=
  Fintype.choose_spec (fun e => rank c e = r) (existsUnique_rank c hr)

theorem atRankFin_eq (c : Fin E → ℝ) {r : ℕ} (hr : r < E) {e : Fin E} (h : rank c e = r) :
    atRankFin c r hr = e :=
  rank_injective c (by rw [rank_atRankFin, h])

/-- Id of the expert at position `r` (`0` past the end, never used: the spec needs `k ≤ E`). -/
def atRank (c : Fin E → ℝ) (r : ℕ) : ℕ :=
  if hr : r < E then (atRankFin c r hr).val else 0

theorem atRank_of_lt (c : Fin E → ℝ) {r : ℕ} (hr : r < E) :
    atRank c r = (atRankFin c r hr).val := by
  simp [atRank, hr]

theorem atRank_eq_of_rank (c : Fin E → ℝ) {e : Fin E} {r : ℕ} (h : rank c e = r) :
    atRank c r = e.val := by
  have hr : r < E := h ▸ rank_lt c e
  rw [atRank_of_lt c hr, atRankFin_eq c hr h]

/-- `v[i]` for `i < E` (`0` past the end). -/
def gatherAt (v : Fin E → ℝ) (i : ℕ) : ℝ :=
  if hi : i < E then v ⟨i, hi⟩ else 0

theorem gatherAt_val (v : Fin E → ℝ) (e : Fin E) : gatherAt v e.val = v e := by
  simp [gatherAt, e.isLt]

/-- Selected ids come out in selection order: position `i` before position `i'`. -/
theorem atRank_before_atRank (c : Fin E → ℝ) {i i' : ℕ} (h : i < i') (hi' : i' < E) :
    Before c (atRankFin c i (h.trans hi')) (atRankFin c i' hi') := by
  rw [← rank_lt_rank_iff, rank_atRankFin, rank_atRankFin]
  exact h

/-- No unselected expert beats a selected one: an expert `e` outside the first `k` positions
comes after every selected expert, by score or, on a tie, by id. -/
theorem atRank_before_unselected (c : Fin E → ℝ) {i k : ℕ} (hi : i < k) (e : Fin E)
    (he : k ≤ rank c e) :
    Before c (atRankFin c i ((lt_of_lt_of_le hi he).trans (rank_lt c e))) e := by
  rw [← rank_lt_rank_iff, rank_atRankFin]
  exact lt_of_lt_of_le hi he

end Order

/-! ### Opaque operations and the specification -/

/-- Opaque TPU operations of `_topk_kernel` (`router_topk.py:46-89,106-151`). Memory holds each
value as a real. Hardware assumptions for the bitwise claim: every op is a deterministic function
of its arguments, and `rowSum tokens k` (Mosaic's `jnp.sum(weights, axis=-1)` on a `[tokens, k]`
f32 value, `router_topk.py:86,148`) depends only on its row. Comparisons, `max`, integer `min`,
`where` and sums with one non-zero term are exact and not opaque. -/
structure RouterOps where
  /-- `astype(float32)` and `astype(logits.dtype)` (`router_topk.py:49-51,53`). -/
  casts : CastOps
  /-- f32 `jnp.exp` (`router_topk.py:49`). -/
  exp : ℝ → ℝ
  /-- f32 `+` (`router_topk.py:49,53,86`). -/
  add : ℝ → ℝ → ℝ
  /-- f32 `/` (`router_topk.py:49,86`). -/
  div : ℝ → ℝ → ℝ
  /-- f32 `*` (`router_topk.py:87`). -/
  mul : ℝ → ℝ → ℝ
  /-- `jnp.sum(weights, axis=-1)` of one row of a `[tokens, k]` f32 value (`router_topk.py:86`). -/
  rowSum : (tokens k : ℕ) → (Fin k → ℝ) → ℝ
  /-- `jax.lax.bitcast_convert_type(x, jnp.int32)` of an f32 value (`router_topk.py:67`). -/
  f32Bits : ℝ → ℤ
  /-- `jax.lax.bitcast_convert_type(n, jnp.float32)` of the int32 whose two's-complement bits are
  the low 32 bits of `n` (`router_topk.py:78`). -/
  f32OfBits : ℤ → ℝ

/-- The bit-level fact the packed bf16 path relies on (`router_topk.py:64-66,78`): an f32 value
that is a bf16 value promoted to f32 has zero low 16 bits, so the top 16 bits of its pattern
(`bits >> 16 & 0xFFFF`, the floor division and non-negative remainder below), shifted back up and
bitcast, give the value again. True for IEEE f32/bf16; not provable here because memory holds
reals, so it is a stated assumption on the ops. -/
def RouterOps.Bf16BitsExact (ops : RouterOps) : Prop :=
  ∀ x : ℝ,
    ops.f32OfBits (ops.f32Bits (ops.casts.toF32 (ops.casts.toBf16 x)) / 65536 % 65536 * 65536) =
      ops.casts.toF32 (ops.casts.toBf16 x)

/-- Exact real arithmetic. The bitcasts are placeholders: this instance does not satisfy
`Bf16BitsExact`, so it only instantiates the non-bf16 paths. -/
def RouterOps.exact : RouterOps where
  casts := CastOps.exact
  exp := Real.exp
  add := (· + ·)
  div := (· / ·)
  mul := (· * ·)
  rowSum := fun _ _ v => ∑ j, v j
  f32Bits := fun _ => 0
  f32OfBits := fun _ => 0

/-- `sigmoid` in f32, rounded to the logits dtype, promoted to f32
(`router_topk.py:48-52,109-113`). -/
def score (ops : RouterOps) (dtype : Dtype) (x : ℝ) : ℝ :=
  ops.casts.toF32 (ops.casts.cast dtype (ops.div 1 (ops.add 1 (ops.exp (-(ops.casts.toF32 x))))))

/-- Biased choice `scores + bias.astype(float32)` (`router_topk.py:53,114`). -/
def choice (ops : RouterOps) (dtype : Dtype) (x b : ℝ) : ℝ :=
  ops.add (score ops dtype x) (ops.casts.toF32 b)

/-- `weights / (sum(weights) + eps) * scale` when `renormalize` (`router_topk.py:85-87,147-149`). -/
def renormRow (ops : RouterOps) (tokens k : ℕ) (renormalize : Bool) (eps scale : ℝ)
    (w : ℕ → ℝ) : ℕ → ℝ :=
  if renormalize then
    fun i => ops.mul (ops.div (w i) (ops.add (ops.rowSum tokens k fun j => w j.val) eps)) scale
  else w

/-- The specification of one token row: `(weights, ids)` by output column. Ids are the experts at
positions `0..k-1` of the biased order; weights are their unbiased scores, renormalized. -/
def mathRow {E : ℕ} (ops : RouterOps) (dtype : Dtype) (tokens k : ℕ) (renormalize : Bool)
    (eps scale : ℝ) (x b : Fin E → ℝ) : (ℕ → ℝ) × (ℕ → ℝ) :=
  let s : Fin E → ℝ := fun e => score ops dtype (x e)
  let c : Fin E → ℝ := fun e => choice ops dtype (x e) (b e)
  (renormRow ops tokens k renormalize eps scale fun i => gatherAt s (atRank c i),
   fun i => (atRank c i : ℝ))

/-! ### The kernel body -/

section Kernel

variable {E : ℕ}

/-- `jnp.min` over the lanes of a row whose masked lanes hold `init` (`router_topk.py:76,80`). -/
def listMin (init : ℤ) (f : Fin E → ℤ) : ℤ :=
  (List.finRange E).foldl (fun a e => min a (f e)) init

/-- One token row of the round loop: the masked choices, and the ids and weights by column. -/
structure RoundState (E : ℕ) where
  choice : Fin E → WithBot ℝ
  ids : ℕ → ℤ
  weights : ℕ → ℝ

/-- Round `r` of the argmax-and-mask loop on one row (`router_topk.py:73-84`), with the packed
keys of `router_topk.py:63-69` when `packed`. `x >> 16` is `x / 65536` and `x & 0xFFFF` is
`x % 65536` (floor division and non-negative remainder, as on two's-complement int32). -/
def roundStep (ops : RouterOps) (packed : Bool) (s : Fin E → ℝ) (st : RoundState E) (r : ℕ) :
    RoundState E :=
  let m := Finset.univ.sup st.choice
  let noKey : ℤ := (E : ℤ) * 65536
  let key : Fin E → ℤ := fun e => (e.val : ℤ) * 65536 + ops.f32Bits (s e) / 65536 % 65536
  let best : ℤ := listMin noKey fun e => if st.choice e = m then key e else noKey
  let idx : ℤ :=
    if packed then best / 65536
    else listMin (E : ℤ) fun e => if st.choice e = m then (e.val : ℤ) else (E : ℤ)
  let w : ℝ :=
    if packed then ops.f32OfBits (best % 65536 * 65536)
    else ∑ e, if (e.val : ℤ) = idx then s e else 0
  { choice := fun e => if (e.val : ℤ) = idx then ⊥ else st.choice e
    ids := fun i => if i = r then idx else st.ids i
    weights := fun i => if i = r then w else st.weights i }

/-- Initial round state: unmasked choices, zero ids and weights (`router_topk.py:59-60`). -/
def initState (c : Fin E → ℝ) : RoundState E :=
  { choice := fun e => (c e : WithBot ℝ), ids := fun _ => 0, weights := fun _ => 0 }

/-- `k` rounds (`router_topk.py:73`). -/
def roundLoop (ops : RouterOps) (packed : Bool) (k : ℕ) (s c : Fin E → ℝ) : RoundState E :=
  (List.range k).foldl (roundStep ops packed s) (initState c)

/-- `before` of `_topk_kernel_1tok` (`router_topk.py:121-131`): lane `j` counts toward the rank of
lane `e` on `>=` from an earlier 128-lane tile, on `>` from a later one, and in the same tile on
`>` or a tie at a lower position. -/
def tileBefore (c : Fin E → ℝ) (j e : Fin E) : Prop :=
  if j.val / 128 < e.val / 128 then c e ≤ c j
  else if e.val / 128 < j.val / 128 then c e < c j
  else c e < c j ∨ (c j = c e ∧ j.val % 128 < e.val % 128)

/-- `rank` of `_topk_kernel_1tok` (`router_topk.py:121-134`): the count of lanes before `e`, summed
over all tiles (integer sums, exact). -/
def tileRank (c : Fin E → ℝ) (e : Fin E) : ℕ :=
  ∑ j, if tileBefore c j e then 1 else 0

/-- The dispatch to `_topk_kernel_1tok` (`router_topk.py:35`, `_RANK_MAX_EXPERTS = 1024`). -/
def UseRankPath (tokens experts k : ℕ) : Prop :=
  tokens = 1 ∧ experts % 128 = 0 ∧ k ≤ experts ∧ experts ≤ 1024

/-- `_topk_kernel` on one token row `x` with bias `b`: `(weights, ids)` by output column. Every op
of the kernel acts along the expert axis, so the `[tokens, experts]` block is computed row by row.
The rank path: slot `i` sums the ids and the scores of the lanes of rank `i`
(`router_topk.py:135-146`); the round path: `k` rounds (`router_topk.py:73-84`); both renormalize
the same way (`router_topk.py:85-87,147-149`). -/
def kernelRow (ops : RouterOps) (dtype : Dtype) (tokens k : ℕ) (renormalize : Bool)
    (eps scale : ℝ) (x b : Fin E → ℝ) : (ℕ → ℝ) × (ℕ → ℝ) :=
  let s : Fin E → ℝ := fun e => score ops dtype (x e)
  let c : Fin E → ℝ := fun e => ops.add (s e) (ops.casts.toF32 (b e))
  if UseRankPath tokens E k then
    (renormRow ops tokens k renormalize eps scale fun i => ∑ e, if tileRank c e = i then s e else 0,
     fun i => (((∑ e, if tileRank c e = i then (e.val : ℤ) else 0 : ℤ)) : ℝ))
  else
    let st := roundLoop ops (decide (dtype = .bf16)) k s c
    (renormRow ops tokens k renormalize eps scale st.weights, fun i => (st.ids i : ℝ))

end Kernel

/-- `full((tokens, experts))` for `logits` (`router_topk.py:170,182`). -/
def logitsSpec (tokens experts : ℕ) : BlockSpec :=
  { blockShape := [tokens, experts], indexMap := fun _ _ => [0, 0] }

/-- `full((experts,))` for the 1-D f32 `bias` (`router_topk.py:170,182,190`). -/
def biasSpec (experts : ℕ) : BlockSpec :=
  { blockShape := [experts], indexMap := fun _ _ => [0] }

/-- `full((tokens, k))` for both outputs (`router_topk.py:170,183`). -/
def outSpec (tokens k : ℕ) : BlockSpec :=
  { blockShape := [tokens, k], indexMap := fun _ _ => [0, 0] }

/-- `pallas_call` of `router_topk_jax` (`router_topk.py:171-190`): `grid=(1,)`, inputs `logits`
and `bias`, outputs f32 `weights` then int32 `ids`; no aliases, no scratch, default
`dimension_semantics`. -/
def gridSpec (tokens experts k : ℕ) (lBuf bBuf wBuf iBuf : BufId) : GridSpec :=
  { grid := [1]
    inSpecs :=
      [ (⟨lBuf, [tokens, experts]⟩, logitsSpec tokens experts),
        (⟨bBuf, [experts]⟩, biasSpec experts) ]
    outSpecs :=
      [ (⟨wBuf, [tokens, k]⟩, outSpec tokens k),
        (⟨iBuf, [tokens, k]⟩, outSpec tokens k) ] }

/-- Pallas body of `_topk_kernel` (`router_topk.py:20-89`): each output element `[t, i]` is column
`i` of `kernelRow` on logits row `t` and the bias; ids are stored as their integer values. -/
def body (ops : RouterOps) (dtype : Dtype) (tokens experts k : ℕ) (renormalize : Bool)
    (eps scale : ℝ) : Body :=
  fun _ _ ins _ =>
    let row := fun t : ℕ => kernelRow (E := experts) ops dtype tokens k renormalize eps scale
      (fun e => (ins.getD 0 fun _ => 0) [t, e.val]) (fun e => (ins.getD 1 fun _ => 0) [e.val])
    [fun loc => (row (loc.getD 0 0)).1 (loc.getD 1 0),
     fun loc => (row (loc.getD 0 0)).2 (loc.getD 1 0)]

/-- Pallas semantics of `router_topk_jax` (`none` on undefined behaviour). -/
def routerTopkRun (ops : RouterOps) (dtype : Dtype) (tokens experts k : ℕ) (renormalize : Bool)
    (eps scale : ℝ) (lBuf bBuf wBuf iBuf : BufId) : (ℕ → ℝ) → Mem → Option Mem :=
  pallasCall (gridSpec tokens experts k lBuf bBuf wBuf iBuf)
    (body ops dtype tokens experts k renormalize eps scale)

/-- Shapes the theorems cover. `0 < tokens` and `0 < k` give non-empty blocks; `k ≤ experts` makes
the top-k defined (the round loop would otherwise pick a masked lane again). With bf16 logits the
packed keys `expert * 2^16 + bits` and `no_key = experts * 2^16` must fit in int32
(`router_topk.py:68-69`); the model uses unbounded integers, so the bound is a premise. -/
structure Supports (dtype : Dtype) (tokens experts k : ℕ) : Prop where
  tokens_pos : 0 < tokens
  k_pos : 0 < k
  k_le : k ≤ experts
  packed_fits : dtype = .bf16 → experts < 32768

/-! ### The selection proofs -/

section Proofs

variable {E : ℕ}

private theorem foldl_min_eq {α : Type*} (f : α → ℤ) (v : ℤ) :
    ∀ (l : List α) (a : ℤ), (∀ x ∈ l, v ≤ f x) → v ≤ a → (a = v ∨ ∃ x ∈ l, f x = v) →
      l.foldl (fun a x => min a (f x)) a = v
  | [], a, _, _, h => by simpa using h
  | x :: l, a, hl, ha, h => by
    simp only [List.foldl_cons]
    have hx : v ≤ f x := hl x (by simp)
    apply foldl_min_eq f v l
    · intro y hy
      exact hl y (by simp [hy])
    · exact le_min ha hx
    · rcases h with rfl | ⟨y, hy, hfy⟩
      · exact Or.inl (min_eq_left hx)
      · rcases List.mem_cons.mp hy with rfl | hy'
        · left
          rw [hfy]
          exact min_eq_right ha
        · exact Or.inr ⟨y, hy', hfy⟩

/-- `listMin` is attained at `e₀` when `f e₀` is below `init` and every `f e`. -/
theorem listMin_eq (init : ℤ) (f : Fin E → ℤ) (e₀ : Fin E) (hle : ∀ e, f e₀ ≤ f e)
    (hinit : f e₀ ≤ init) : listMin init f = f e₀ :=
  foldl_min_eq f (f e₀) (List.finRange E) init (fun e _ => hle e) hinit
    (Or.inr ⟨e₀, List.mem_finRange e₀, rfl⟩)

/-- The kernel's tiled compare is the tie rule. -/
theorem tileBefore_iff (c : Fin E → ℝ) (j e : Fin E) : tileBefore c j e ↔ Before c j e := by
  unfold tileBefore Before
  split_ifs with h1 h2
  · have hje : j < e := by
      rw [Fin.lt_def]
      omega
    constructor
    · intro h
      rcases h.lt_or_eq with h | h
      · exact Or.inl h
      · exact Or.inr ⟨h.symm, hje⟩
    · rintro (h | ⟨h, _⟩)
      · exact h.le
      · exact h.symm.le
  · have hej : e < j := by
      rw [Fin.lt_def]
      omega
    constructor
    · exact Or.inl
    · rintro (h | ⟨_, h⟩)
      · exact h
      · exact absurd (h.trans hej) (lt_irrefl _)
  · have hiff : j.val % 128 < e.val % 128 ↔ j < e := by
      rw [Fin.lt_def]
      omega
    rw [hiff]

theorem tileRank_eq_rank (c : Fin E → ℝ) (e : Fin E) : tileRank c e = rank c e := by
  unfold tileRank rank
  rw [Finset.card_filter]
  exact Finset.sum_congr rfl fun j _ => if_congr (tileBefore_iff c j e) rfl rfl

/-- Rank path, ids: slot `i` holds the expert of rank `i`. -/
theorem sum_rank_id (c : Fin E → ℝ) {i : ℕ} (hi : i < E) :
    (∑ e, if rank c e = i then (e.val : ℤ) else 0) = ((atRankFin c i hi).val : ℤ) := by
  rw [Finset.sum_eq_single (atRankFin c i hi)]
  · rw [ite_eq_left (rank_atRankFin c i hi)]
  · intro e _ hne
    rw [ite_eq_right]
    intro h
    exact hne (atRankFin_eq c hi h).symm
  · intro h
    exact absurd (Finset.mem_univ _) h

/-- Rank path, weights: slot `i` holds the score of the expert of rank `i`. -/
theorem sum_rank_score (c s : Fin E → ℝ) {i : ℕ} (hi : i < E) :
    (∑ e, if rank c e = i then s e else 0) = s (atRankFin c i hi) := by
  rw [Finset.sum_eq_single (atRankFin c i hi)]
  · rw [ite_eq_left (rank_atRankFin c i hi)]
  · intro e _ hne
    rw [ite_eq_right]
    intro h
    exact hne (atRankFin_eq c hi h).symm
  · intro h
    exact absurd (Finset.mem_univ _) h

/-- Round `r` picks the expert of rank `r`, given that the experts of ranks `0..r-1` are masked:
the max is its biased score and every other lane tied with it has a higher id. -/
theorem roundStep_spec (ops : RouterOps) (packed : Bool) (s c : Fin E → ℝ)
    (hbits : packed = true →
      ∀ e, ops.f32OfBits (ops.f32Bits (s e) / 65536 % 65536 * 65536) = s e)
    (st : RoundState E) (r : ℕ) (hr : r < E)
    (hch : ∀ e, st.choice e = if rank c e < r then ⊥ else (c e : WithBot ℝ)) :
    let st' := roundStep ops packed s st r
    (∀ e, st'.choice e = if rank c e < r + 1 then ⊥ else (c e : WithBot ℝ)) ∧
      (∀ i, st'.ids i = if i = r then ((atRankFin c r hr).val : ℤ) else st.ids i) ∧
      (∀ i, st'.weights i = if i = r then s (atRankFin c r hr) else st.weights i) := by
  set es := atRankFin c r hr with hes
  have hrank : rank c es = r := rank_atRankFin c r hr
  -- every unmasked expert other than `es` comes after it
  have hafter : ∀ e, r ≤ rank c e → e ≠ es → Before c es e := by
    intro e hre hne
    rw [← rank_lt_rank_iff, hrank]
    rcases hre.lt_or_eq with h | h
    · exact h
    · exact absurd (atRankFin_eq c hr h.symm).symm hne
  have hle : ∀ e, r ≤ rank c e → c e ≤ c es := by
    intro e hre
    by_cases hne : e = es
    · rw [hne]
    · rcases hafter e hre hne with h | ⟨h, _⟩
      · exact h.le
      · exact h.symm.le
  have hm : Finset.univ.sup st.choice = (c es : WithBot ℝ) := by
    apply le_antisymm
    · apply Finset.sup_le
      intro e _
      rw [hch e]
      split_ifs with h
      · exact bot_le
      · exact WithBot.coe_le_coe.mpr (hle e (not_lt.mp h))
    · have h := Finset.le_sup (f := st.choice) (Finset.mem_univ es)
      rwa [hch es, ite_eq_right (by omega)] at h
  have htied : ∀ e, st.choice e = (c es : WithBot ℝ) ↔ r ≤ rank c e ∧ c e = c es := by
    intro e
    rw [hch e]
    split_ifs with h
    · simp only [WithBot.bot_ne_coe, false_iff, not_and]
      intro h'
      omega
    · rw [WithBot.coe_inj]
      exact ⟨fun h' => ⟨not_lt.mp h, h'⟩, fun h' => h'.2⟩
  have hmin : ∀ e, r ≤ rank c e → c e = c es → es.val ≤ e.val := by
    intro e hre hce
    by_cases hne : e = es
    · rw [hne]
    · rcases hafter e hre hne with h | ⟨_, h⟩
      · exact absurd hce (ne_of_lt h)
      · exact h.le
  have hes_tied : r ≤ rank c es ∧ c es = c es := ⟨hrank.ge, rfl⟩
  -- the lane the round masks and the weight it gathers
  have hidx_w : (if packed then
        (listMin ((E : ℤ) * 65536) fun e => if st.choice e = Finset.univ.sup st.choice then
          (e.val : ℤ) * 65536 + ops.f32Bits (s e) / 65536 % 65536 else (E : ℤ) * 65536) / 65536
      else listMin (E : ℤ) fun e =>
        if st.choice e = Finset.univ.sup st.choice then (e.val : ℤ) else (E : ℤ)) = es.val ∧
      (if packed then ops.f32OfBits ((listMin ((E : ℤ) * 65536) fun e =>
          if st.choice e = Finset.univ.sup st.choice then
            (e.val : ℤ) * 65536 + ops.f32Bits (s e) / 65536 % 65536 else (E : ℤ) * 65536) %
            65536 * 65536)
      else ∑ e, if (e.val : ℤ) = (es.val : ℤ) then s e else 0) = s es := by
    simp only [hm, htied]
    have hesE : es.val < E := es.isLt
    cases packed with
    | true =>
      set h₀ := ops.f32Bits (s es) / 65536 % 65536 with hh₀
      have hh₀0 : 0 ≤ h₀ := Int.emod_nonneg _ (by norm_num)
      have hh₀1 : h₀ < 65536 := Int.emod_lt_of_pos _ (by norm_num)
      have hbest : (listMin ((E : ℤ) * 65536) fun e =>
          if r ≤ rank c e ∧ c e = c es then
            (e.val : ℤ) * 65536 + ops.f32Bits (s e) / 65536 % 65536 else (E : ℤ) * 65536) =
          (es.val : ℤ) * 65536 + h₀ := by
        rw [listMin_eq _ _ es]
        · rw [ite_eq_left hes_tied]
        · intro e
          rw [ite_eq_left hes_tied]
          split_ifs with h
          · have hl := hmin e h.1 h.2
            have h0 : 0 ≤ ops.f32Bits (s e) / 65536 % 65536 := Int.emod_nonneg _ (by norm_num)
            rcases hl.lt_or_eq with hl | hl
            · have : (es.val : ℤ) + 1 ≤ e.val := by exact_mod_cast hl
              nlinarith
            · rw [show e = es from Fin.ext hl.symm]
          · have : (es.val : ℤ) + 1 ≤ E := by exact_mod_cast hesE
            nlinarith
        · rw [ite_eq_left hes_tied]
          have : (es.val : ℤ) + 1 ≤ E := by exact_mod_cast hesE
          nlinarith
      rw [hbest]
      simp only [↓reduceIte]
      have hq : ((es.val : ℤ) * 65536 + h₀) / 65536 = es.val := by omega
      have hr' : ((es.val : ℤ) * 65536 + h₀) % 65536 = h₀ := by omega
      rw [hq, hr']
      exact ⟨rfl, hbits rfl es⟩
    | false =>
      have hidx : (listMin (E : ℤ) fun e =>
          if r ≤ rank c e ∧ c e = c es then (e.val : ℤ) else (E : ℤ)) = es.val := by
        rw [listMin_eq _ _ es]
        · rw [ite_eq_left hes_tied]
        · intro e
          rw [ite_eq_left hes_tied]
          split_ifs with h
          · exact_mod_cast hmin e h.1 h.2
          · exact_mod_cast hesE.le
        · rw [ite_eq_left hes_tied]
          exact_mod_cast hesE.le
      rw [hidx]
      refine ⟨by simp, ?_⟩
      simp only [Bool.false_eq_true, ↓reduceIte, Nat.cast_inj, Fin.val_inj]
      rw [Finset.sum_ite_eq']
      simp
  obtain ⟨hidx, hw⟩ := hidx_w
  refine ⟨?_, ?_, ?_⟩
  · intro e
    simp only [roundStep]
    rw [hidx]
    by_cases he : e = es
    · subst he
      simp [hrank]
    · have hne : (e.val : ℤ) ≠ es.val := by
        intro h
        exact he (Fin.ext (by exact_mod_cast h))
      rw [ite_eq_right hne, hch e]
      have hrne : rank c e ≠ r := fun h => he (atRankFin_eq c hr h).symm
      by_cases hlt : rank c e < r
      · rw [ite_eq_left hlt, ite_eq_left (by omega)]
      · rw [ite_eq_right hlt, ite_eq_right (by omega)]
  · intro i
    simp only [roundStep]
    rw [hidx]
  · intro i
    simp only [roundStep]
    rw [hidx, hw]

/-- After `r ≤ E` rounds the experts of ranks `0..r-1` are masked and sit in columns `0..r-1`. -/
theorem roundLoop_spec (ops : RouterOps) (packed : Bool) (s c : Fin E → ℝ)
    (hbits : packed = true →
      ∀ e, ops.f32OfBits (ops.f32Bits (s e) / 65536 % 65536 * 65536) = s e) :
    ∀ r, r ≤ E →
      let st := (List.range r).foldl (roundStep ops packed s) (initState c)
      (∀ e, st.choice e = if rank c e < r then ⊥ else (c e : WithBot ℝ)) ∧
        (∀ i, st.ids i = if i < r then (atRank c i : ℤ) else 0) ∧
        (∀ i, st.weights i = if i < r then gatherAt s (atRank c i) else 0)
  | 0, _ => by simp [initState]
  | r + 1, hr => by
    obtain ⟨hch, hids, hws⟩ := roundLoop_spec ops packed s c hbits r (by omega)
    obtain ⟨hch', hids', hws'⟩ :=
      roundStep_spec ops packed s c hbits _ r (by omega) hch
    simp only [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    refine ⟨hch', ?_, ?_⟩
    · intro i
      rw [hids', hids]
      by_cases hi : i = r
      · subst hi
        simp [atRank_of_lt c (show i < E by omega)]
      · rw [ite_eq_right hi]
        by_cases hlt : i < r
        · rw [ite_eq_left hlt, ite_eq_left (by omega)]
        · rw [ite_eq_right hlt, ite_eq_right (by omega)]
    · intro i
      rw [hws', hws]
      by_cases hi : i = r
      · subst hi
        simp [atRank_of_lt c (show i < E by omega), gatherAt_val]
      · rw [ite_eq_right hi]
        by_cases hlt : i < r
        · rw [ite_eq_left hlt, ite_eq_left (by omega)]
        · rw [ite_eq_right hlt, ite_eq_right (by omega)]

theorem renormRow_congr (ops : RouterOps) (tokens k : ℕ) (renormalize : Bool) (eps scale : ℝ)
    {w w' : ℕ → ℝ} (h : ∀ j, j < k → w j = w' j) {i : ℕ} (hi : i < k) :
    renormRow ops tokens k renormalize eps scale w i =
      renormRow ops tokens k renormalize eps scale w' i := by
  have hsum : (fun j : Fin k => w j.val) = fun j => w' j.val := funext fun j => h j.val j.isLt
  cases renormalize <;> simp [renormRow, hsum, h i hi]

/-- Every code path of `_topk_kernel` computes `mathRow` on the first `k` columns. -/
theorem kernelRow_eq_mathRow (ops : RouterOps) (dtype : Dtype)
    (hbits : dtype = .bf16 → ops.Bf16BitsExact) (tokens k : ℕ) (hk : k ≤ E)
    (renormalize : Bool) (eps scale : ℝ) (x b : Fin E → ℝ) {i : ℕ} (hi : i < k) :
    (kernelRow ops dtype tokens k renormalize eps scale x b).1 i =
        (mathRow ops dtype tokens k renormalize eps scale x b).1 i ∧
      (kernelRow ops dtype tokens k renormalize eps scale x b).2 i =
        (mathRow ops dtype tokens k renormalize eps scale x b).2 i := by
  simp only [kernelRow, mathRow, choice]
  split_ifs with hrank
  · simp only [tileRank_eq_rank]
    refine ⟨renormRow_congr ops tokens k renormalize eps scale (fun j hj => ?_) hi, ?_⟩
    · rw [sum_rank_score _ _ (show j < E by omega), atRank_of_lt _ (show j < E by omega),
        gatherAt_val]
    · rw [sum_rank_id _ (show i < E by omega), atRank_of_lt _ (show i < E by omega)]
      simp
  · have hb : decide (dtype = .bf16) = true → ∀ e,
        ops.f32OfBits (ops.f32Bits (score ops dtype (x e)) / 65536 % 65536 * 65536) =
          score ops dtype (x e) := by
      intro hd e
      have hd' : dtype = .bf16 := of_decide_eq_true hd
      subst hd'
      exact hbits rfl _
    obtain ⟨-, hids, hws⟩ := roundLoop_spec ops (decide (dtype = .bf16))
      (fun e => score ops dtype (x e))
      (fun e => ops.add (score ops dtype (x e)) (ops.casts.toF32 (b e))) hb k hk
    refine ⟨renormRow_congr ops tokens k renormalize eps scale (fun j hj => ?_) hi, ?_⟩
    · exact (hws j).trans (ite_eq_left hj)
    · simp [roundLoop, hids, hi]

end Proofs

/-! ### The `pallas_call` -/

theorem wellDefined (ops : RouterOps) (dtype : Dtype) (tokens experts k : ℕ)
    (htok : 0 < tokens) (hE : 0 < experts) (hk : 0 < k) (renormalize : Bool) (eps scale : ℝ)
    (lBuf bBuf wBuf iBuf : BufId) (hwi : wBuf ≠ iBuf) (hwl : wBuf ≠ lBuf) (hwb : wBuf ≠ bBuf)
    (hil : iBuf ≠ lBuf) (hib : iBuf ≠ bBuf) (m0 : Mem) :
    (gridSpec tokens experts k lBuf bBuf wBuf iBuf).WellDefined
      (body ops dtype tokens experts k renormalize eps scale) m0 := by
  have hgp : gridPoints [1] = [[0]] := rfl
  have hpar : (gridSpec tokens experts k lBuf bBuf wBuf iBuf).parGrid = [] := rfl
  constructor
  · intro d hd
    simp only [gridSpec, List.mem_singleton] at hd
    omega
  · intro p hp
    simp only [gridSpec, List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil,
      or_false] at hp
    rcases hp with rfl | rfl | rfl | rfl <;>
      simp [logitsSpec, biasSpec, outSpec, htok, hE, hk]
  · intro j _ p hp
    simp only [gridSpec, List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil,
      or_false] at hp
    rcases hp with rfl | rfl | rfl | rfl <;>
      simp [GridSpec.idxAt, logitsSpec, biasSpec, outSpec, blockOk, htok, hE, hk]
  · intro p hp
    apply noRevisit_of_nodup
    simp [gridSpec, hgp]
  · intro _ _ _ _
    rfl
  · intro s t hst ht
    simp only [gridSpec, hgp, List.length_singleton] at ht
    omega
  · refine GridSpec.AliasesValid.of_empty rfl (by simp [gridSpec, hwi]) ?_ (by simp [gridSpec])
    intro inp hinp outp houtp
    simp only [gridSpec, List.mem_cons, List.not_mem_nil, or_false] at hinp houtp
    rcases hinp with rfl | rfl <;> rcases houtp with rfl | rfl
    · exact hwl.symm
    · exact hil.symm
    · exact hwb.symm
    · exact hib.symm
  · rfl
  · rfl
  · intro pg₁ h₁ pg₂ h₂ hne
    rw [hpar] at h₁ h₂
    simp only [gridPoints, List.mem_singleton] at h₁ h₂
    exact absurd (h₁.trans h₂.symm) hne
  · intro pg₁ h₁ pg₂ h₂ hne
    rw [hpar] at h₁ h₂
    simp only [gridPoints, List.mem_singleton] at h₁ h₂
    exact absurd (h₁.trans h₂.symm) hne
  · intro h
    exact absurd hpar h

theorem storeOnly (ops : RouterOps) (dtype : Dtype) (tokens experts k : ℕ) (renormalize : Bool)
    (eps scale : ℝ) (lBuf bBuf wBuf iBuf : BufId) (hwi : wBuf ≠ iBuf) (hwl : wBuf ≠ lBuf)
    (hwb : wBuf ≠ bBuf) (hil : iBuf ≠ lBuf) (hib : iBuf ≠ bBuf) :
    StoreOnly (gridSpec tokens experts k lBuf bBuf wBuf iBuf)
      (body ops dtype tokens experts k renormalize eps scale) where
  outs_nodup := by simp [gridSpec, hwi]
  ins_disjoint := by
    intro p hp o ho
    simp only [gridSpec, List.mem_cons, List.not_mem_nil, or_false] at hp ho
    rcases hp with rfl | rfl <;> rcases ho with rfl | rfl
    · exact hwl.symm
    · exact hil.symm
    · exact hwb.symm
    · exact hib.symm
  ignores_prev := fun _ _ _ _ => rfl

/-- An element `(t, i)` of a `[tokens, k]` array lies in the whole-array block, at local `[t, i]`. -/
private theorem addrInBlock_full {buf : BufId} {tokens k t i : ℕ} (ht : t < tokens) (hi : i < k) :
    addrInBlock ⟨buf, [tokens, k]⟩ [tokens, k] [0, 0] (t * k + i) = true ∧
      localCoord [tokens, k] [0, 0] (unflatCoord [tokens, k] (t * k + i)) = [t, i] := by
  have hlt : t * k + i < tokens * k := by nlinarith
  rw [unflatCoord_two tokens k t i hi]
  simp [addrInBlock, unflatCoord_two tokens k t i hi, coordInBlock, localCoord, hlt, ht, hi]

/-- End-to-end correctness of `router_topk_jax`: for every shape in `Supports` and every
`RouterOps` (with the bf16 bit fact when the logits are bf16), the run is defined; output element
`(t, i)` of `weights` and of `ids` is column `i` of `mathRow` on logits row `t` and the bias; and
memory outside the two output buffers is unchanged. -/
theorem routerTopk_correct (ops : RouterOps) (dtype : Dtype)
    (hbits : dtype = .bf16 → ops.Bf16BitsExact) (tokens experts k : ℕ)
    (hsup : Supports dtype tokens experts k) (renormalize : Bool) (eps scale : ℝ)
    (lBuf bBuf wBuf iBuf : BufId) (hwi : wBuf ≠ iBuf) (hwl : wBuf ≠ lBuf) (hwb : wBuf ≠ bBuf)
    (hil : iBuf ≠ lBuf) (hib : iBuf ≠ bBuf) (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', routerTopkRun ops dtype tokens experts k renormalize eps scale lBuf bBuf wBuf iBuf
        junk m0 = some m' ∧
      (∀ (t : Fin tokens) (i : Fin k),
        let row := mathRow ops dtype tokens k renormalize eps scale
          (fun e : Fin experts => m0.read lBuf (t.val * experts + e.val))
          (fun e : Fin experts => m0.read bBuf e.val)
        m'.read wBuf (t.val * k + i.val) = row.1 i.val ∧
          m'.read iBuf (t.val * k + i.val) = row.2 i.val) ∧
      Mem.agreeOutside (Mem.bufFootprint wBuf ∪ Mem.bufFootprint iBuf) m0 m' := by
  have hk : 0 < k := hsup.k_pos
  have hE : 0 < experts := lt_of_lt_of_le hk hsup.k_le
  set gs := gridSpec tokens experts k lBuf bBuf wBuf iBuf with hgs
  set bd := body ops dtype tokens experts k renormalize eps scale with hbd
  refine ⟨pallasExec gs bd junk m0,
    pallasCall_of_wellDefined _
      (wellDefined ops dtype tokens experts k hsup.tokens_pos hE hk renormalize eps scale
        lBuf bBuf wBuf iBuf hwi hwl hwb hil hib m0), ?_, ?_⟩
  · intro t i
    have hgp : gridPoints gs.grid = [[0]] := rfl
    have hidx : ∀ sc, (outSpec tokens k).blockIdx [0] sc = [0, 0] := by
      intro sc
      simp [BlockSpec.blockIdx, outSpec]
    obtain ⟨hin, hloc⟩ := addrInBlock_full (buf := wBuf) t.isLt i.isLt
    obtain ⟨hin', hloc'⟩ := addrInBlock_full (buf := iBuf) t.isLt i.isLt
    -- the inputs the body sees are the initial logits row and bias
    have hx : (fun e : Fin experts =>
        (loadIns gs junk m0 [0]).getD 0 (fun _ => 0) [t.val, e.val]) =
        fun e => m0.read lBuf (t.val * experts + e.val) := by
      funext e
      simp only [loadIns, hgs, gridSpec, List.getD_eq_getElem?_getD, List.getElem?_mapIdx,
        List.getElem?_cons_zero, Option.map_some, Option.getD_some]
      rw [loadBlock_eq_of_inBounds]
      · simp [logitsSpec, BlockSpec.blockIdx, globalCoord, flatCoord]
      · simp [logitsSpec, inBounds, t.isLt, e.isLt]
      · simp [logitsSpec, BlockSpec.blockIdx, globalCoord, inBounds, t.isLt, e.isLt]
    have hb : (fun e : Fin experts =>
        (loadIns gs junk m0 [0]).getD 1 (fun _ => 0) [e.val]) =
        fun e => m0.read bBuf e.val := by
      funext e
      simp only [loadIns, hgs, gridSpec, List.getD_eq_getElem?_getD, List.getElem?_mapIdx,
        List.getElem?_cons_succ, List.getElem?_cons_zero, Option.map_some, Option.getD_some]
      rw [loadBlock_eq_of_inBounds]
      · simp [biasSpec, BlockSpec.blockIdx, globalCoord, flatCoord]
      · simp [biasSpec, inBounds, e.isLt]
      · simp [biasSpec, BlockSpec.blockIdx, globalCoord, inBounds, e.isLt]
    obtain ⟨hrw, hri⟩ := kernelRow_eq_mathRow ops dtype hbits tokens k hsup.k_le renormalize eps
      scale (fun e : Fin experts => m0.read lBuf (t.val * experts + e.val))
      (fun e : Fin experts => m0.read bBuf e.val) i.isLt
    refine ⟨?_, ?_⟩
    · have hread := pallasExec_read_out gs bd junk m0
        (storeOnly ops dtype tokens experts k renormalize eps scale lBuf bBuf wBuf iBuf
          hwi hwl hwb hil hib) 0 (⟨wBuf, [tokens, k]⟩, outSpec tokens k)
        (by simp [hgs, gridSpec]) (t.val * k + i.val)
      rw [hread]
      simp only [seqOut, hgp, List.foldl_cons, List.foldl_nil, hidx]
      rw [show (outSpec tokens k).blockShape = [tokens, k] from rfl,
        storeBlock_read_inside _ _ _ _ _ _ hin, hloc]
      simp only [outVal, hbd, body, List.getD_cons_zero, List.getD_cons_succ]
      simp only [hx, hb]
      exact hrw
    · have hread := pallasExec_read_out gs bd junk m0
        (storeOnly ops dtype tokens experts k renormalize eps scale lBuf bBuf wBuf iBuf
          hwi hwl hwb hil hib) 1 (⟨iBuf, [tokens, k]⟩, outSpec tokens k)
        (by simp [hgs, gridSpec]) (t.val * k + i.val)
      rw [hread]
      simp only [seqOut, hgp, List.foldl_cons, List.foldl_nil, hidx]
      rw [show (outSpec tokens k).blockShape = [tokens, k] from rfl,
        storeBlock_read_inside _ _ _ _ _ _ hin', hloc']
      simp only [outVal, hbd, body, List.getD_cons_zero, List.getD_cons_succ]
      simp only [hx, hb]
      exact hri
  · refine Mem.agreeOutside_mono ?_ (pallasExec_agreeOutside_outBufs gs bd junk m0)
    rintro ⟨b, a⟩ ⟨x, hx', hbuf⟩
    simp only [hgs, gridSpec, List.mem_cons, List.not_mem_nil, or_false] at hx'
    rcases hx' with rfl | rfl
    · exact Or.inl (by simpa [Mem.bufFootprint] using hbuf.symm)
    · exact Or.inr (by simpa [Mem.bufFootprint] using hbuf.symm)

/-! ### `KernelSpec` -/

/-- Read-only inputs: BF16 (or other dtype) `logits [tokens, experts]` and the f32 `bias`. -/
structure RouterIn (tokens experts : ℕ) where
  logits : Tensor ℝ [tokens, experts]
  bias : Tensor ℝ [experts]

/-- Outputs: f32 `weights [tokens, k]` and int32 `ids [tokens, k]`. -/
structure RouterOut (tokens k : ℕ) where
  weights : Tensor ℝ [tokens, k]
  ids : Tensor ℝ [tokens, k]

/-- Four-part `KernelSpec` of `router_topk_jax`. -/
def routerTopkSpec (ops : RouterOps) (dtype : Dtype) (tokens experts k : ℕ) (renormalize : Bool)
    (eps scale : ℝ) (lBuf bBuf wBuf iBuf : BufId) :
    KernelSpec (RouterIn tokens experts) Unit (RouterOut tokens k) :=
  { readIn := fun m =>
      { logits := (Layout.rowMajor lBuf 0 [tokens, experts]).read m
        bias := (Layout.rowMajor bBuf 0 [experts]).read m }
    readSt := fun _ => ()
    readOut := fun m =>
      { weights := (Layout.rowMajor wBuf 0 [tokens, k]).read m
        ids := (Layout.rowMajor iBuf 0 [tokens, k]).read m }
    footprint := Mem.bufFootprint wBuf ∪ Mem.bufFootprint iBuf
    math := fun _ inp =>
      let row := fun t => mathRow ops dtype tokens k renormalize eps scale
        (fun e => inp.logits.get2 t e) (fun e => inp.bias.get1 e)
      { weights := Tensor.of2 fun t i => (row t).1 i.val
        ids := Tensor.of2 fun t i => (row t).2 i.val }
    mutate := fun _ _ => ()
    requires := fun _ _ =>
      Supports dtype tokens experts k ∧ wBuf ≠ iBuf ∧ wBuf ≠ lBuf ∧ wBuf ≠ bBuf ∧
        iBuf ≠ lBuf ∧ iBuf ≠ bBuf }

theorem routerTopkSpec_wellFormed (ops : RouterOps) (dtype : Dtype) (tokens experts k : ℕ)
    (renormalize : Bool) (eps scale : ℝ) (lBuf bBuf wBuf iBuf : BufId)
    (hwl : wBuf ≠ lBuf) (hwb : wBuf ≠ bBuf) (hil : iBuf ≠ lBuf) (hib : iBuf ≠ bBuf) :
    (routerTopkSpec ops dtype tokens experts k renormalize eps scale
      lBuf bBuf wBuf iBuf).WellFormed where
  readOut_on_footprint m₁ m₂ hagree := by
    have h : Mem.agreeOn (Mem.bufFootprint wBuf ∪ Mem.bufFootprint iBuf) m₁ m₂ := hagree
    dsimp only [routerTopkSpec]
    rw [Layout.read_eq_of_agreeOn (Layout.rowMajor wBuf 0 [tokens, k])
        ((Layout.footprint_subset_buf _).trans Set.subset_union_left) h,
      Layout.read_eq_of_agreeOn (Layout.rowMajor iBuf 0 [tokens, k])
        ((Layout.footprint_subset_buf _).trans Set.subset_union_right) h]
  readSt_on_footprint _ _ _ := rfl
  readIn_outside_footprint m₁ m₂ hagree := by
    have h : Mem.agreeOutside (Mem.bufFootprint wBuf ∪ Mem.bufFootprint iBuf) m₁ m₂ := hagree
    dsimp only [routerTopkSpec]
    rw [Layout.read_eq_of_agreeOutside (Layout.rowMajor lBuf 0 [tokens, experts])
          (Set.disjoint_union_right.mpr ⟨Layout.disjoint_bufFootprint _ hwl.symm,
            Layout.disjoint_bufFootprint _ hil.symm⟩) h,
      Layout.read_eq_of_agreeOutside (Layout.rowMajor bBuf 0 [experts])
          (Set.disjoint_union_right.mpr ⟨Layout.disjoint_bufFootprint _ hwb.symm,
            Layout.disjoint_bufFootprint _ hib.symm⟩) h]

/-- `router_topk_jax` satisfies its `KernelSpec` for every `RouterOps` (with the bf16 bit fact when
the logits are bf16). -/
theorem routerTopkSpec_correct (ops : RouterOps) (dtype : Dtype)
    (hbits : dtype = .bf16 → ops.Bf16BitsExact) (tokens experts k : ℕ) (renormalize : Bool)
    (eps scale : ℝ) (lBuf bBuf wBuf iBuf : BufId) :
    (routerTopkSpec ops dtype tokens experts k renormalize eps scale lBuf bBuf wBuf iBuf).Correct
      (routerTopkRun ops dtype tokens experts k renormalize eps scale lBuf bBuf wBuf iBuf) := by
  intro junk m0 ⟨hsup, hwi, hwl, hwb, hil, hib⟩
  refine ⟨routerTopkSpec_wellFormed ops dtype tokens experts k renormalize eps scale
    lBuf bBuf wBuf iBuf hwl hwb hil hib, ?_⟩
  obtain ⟨m', hcall, hval, hframe⟩ :=
    routerTopk_correct ops dtype hbits tokens experts k hsup renormalize eps scale
      lBuf bBuf wBuf iBuf hwi hwl hwb hil hib junk m0
  refine ⟨m', hcall, ?_, rfl, hframe⟩
  simp only [routerTopkSpec, RouterOut.mk.injEq]
  constructor <;> apply Tensor.ext2 <;> intro t i
  · simpa [Layout.rowMajor, Layout.read, Tensor.get2, Tensor.of2, Tensor.get1] using
      (hval t i).1
  · simpa [Layout.rowMajor, Layout.read, Tensor.get2, Tensor.of2, Tensor.get1] using
      (hval t i).2

end

end TpuKernelLean.Examples.RouterTopk
