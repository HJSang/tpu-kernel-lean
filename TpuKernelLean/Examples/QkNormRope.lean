import TpuKernelLean.Spec
import TpuKernelLean.Pallas.RowKernel

/-!
Specification and `pallas_call`-level verification of `sglang_tpu/qk_norm_rope.py`:
fused per-head RMSNorm + NeoX RoPE over a raw QKV projection buffer.

* Inputs (`qk_norm_rope.py:79-88`):
  - `qkv : [T, (numQ + 2 * numKV) * (2 * h)]`, loaded through a narrower `[bm, (numQ + numKV) * (2 * h)]`
    block at column 0 (`qk_norm_rope.py:105`), so `v` columns are never fetched into VMEM.
  - `cos_sin : [T, 2 * h]`, packing `cos` in `[:h]` and `sin` in `[h:]` (`qk_norm_rope.py:67-70`).
  - `q_weight : [2 * h]` and `k_weight : [2 * h]` (`qk_norm_rope.py:107-108`).
* Outputs (`qk_norm_rope.py:110-117`):
  - `q : [T, numQ * (2 * h)]` (reshaped `[T, numQ, 2 * h]`).
  - `k : [T, numKV * (2 * h)]` (reshaped `[T, numKV, 2 * h]`).
  - `v` stays untouched in `qkv`, justifying the caller's slice view (`qk_norm_rope.py:276`).

Float operations are kept opaque (`FOps`), giving bitwise correctness and batch invariance under
the stated determinism assumptions, while `FOps.exact` links the result to `Tensor.rmsNorm` and
`Tensor.ropeNeox`. Requiring `0 < eps` guarantees the exact RMSNorm denominator is strictly
positive so division by zero (which Lean's `x / 0 = 0` would mask) cannot occur.
-/

namespace TpuKernelLean.Examples.QkNormRope

/-- Max token rows per Pallas program (`qk_norm_rope.py:21`). -/
def maxBlockTokens : ℕ := 256

/-- Default token block size `min(T, _MAX_BLOCK_TOKENS)` (`qk_norm_rope.py:95`). -/
def defaultBlockTokens (T : ℕ) : ℕ := min T maxBlockTokens

/-- Opaque TPU float operations used by `_qk_norm_rope_kernel` (`qk_norm_rope.py:24-76`).
A theorem proved for every `FOps` holds bit for bit on the TPU provided each op is a deterministic
function of its arguments and `headMean D` (`jnp.mean(..., axis=-1)` over one head of width `D`)
depends only on that head, not on the block's token count or other heads. -/
structure FOps where
  /-- Cast an input element to fp32 (`.astype(jnp.float32)`, `qk_norm_rope.py:60,63,67,72,73`). -/
  toF32 : ℝ → ℝ
  /-- Cast an fp32 output element to `q_ref.dtype` / `k_ref.dtype` (`qk_norm_rope.py:75-76`). -/
  toOut : ℝ → ℝ
  /-- fp32 addition (`var + eps`, `x2 * cos + x1 * sin`). -/
  add : ℝ → ℝ → ℝ
  /-- fp32 subtraction (`x1 * cos - x2 * sin`). -/
  sub : ℝ → ℝ → ℝ
  /-- fp32 multiplication (`x * x`, `x * rsqrt(...) * weight`, rotary products). -/
  mul : ℝ → ℝ → ℝ
  /-- `jax.lax.rsqrt` (`qk_norm_rope.py:29`). -/
  rsqrt : ℝ → ℝ
  /-- `jnp.mean(..., axis=-1)` over one head of width `D` (`qk_norm_rope.py:28`). -/
  headMean : (D : ℕ) → (Fin D → ℝ) → ℝ

/-- Exact real arithmetic, linking `FOps` to `Tensor.rmsNorm` and `Tensor.ropeNeox`. -/
noncomputable def FOps.exact : FOps where
  toF32 := id
  toOut := id
  add := (· + ·)
  sub := (· - ·)
  mul := (· * ·)
  rsqrt := fun v => 1 / Real.sqrt v
  headMean := fun D v => (∑ c, v c) / (D : ℝ)

/-- Read-only inputs for fused QK RMSNorm + NeoX RoPE (`qk_norm_rope.py:79-88`). -/
structure QkNormRopeIn (T numQ numKV h : ℕ) where
  /-- Fused QKV projection `[T, (numQ + 2 * numKV) * (2 * h)]`. -/
  qkv : Tensor ℝ [T, (numQ + 2 * numKV) * (2 * h)]
  /-- Packed rotary table `[T, 2 * h]` (`cos` in `[:h]`, `sin` in `[h:]`). -/
  cosSin : Tensor ℝ [T, 2 * h]
  /-- Per-head Q RMSNorm weight `[2 * h]`. -/
  qWeight : Tensor ℝ [2 * h]
  /-- Per-head K RMSNorm weight `[2 * h]`. -/
  kWeight : Tensor ℝ [2 * h]

/-- Outputs of fused QK RMSNorm + NeoX RoPE (`qk_norm_rope.py:114-117`). -/
structure QkNormRopeOut (T numQ numKV h : ℕ) where
  /-- Normed and rotated queries `[T, numQ, 2 * h]`. -/
  q : Tensor ℝ [T, numQ, 2 * h]
  /-- Normed and rotated keys `[T, numKV, 2 * h]`. -/
  k : Tensor ℝ [T, numKV, 2 * h]

/-- Extract the Q slice `qkv[:, :num_q * hd].reshape(T, num_q, hd)` (`qk_norm_rope.py:60`). -/
def extractQ {T numQ numKV h : ℕ} (inp : QkNormRopeIn T numQ numKV h) :
    Tensor ℝ [T, numQ, 2 * h] :=
  Tensor.of3 fun t hd d =>
    inp.qkv.get2 t ⟨hd.val * (2 * h) + d.val, by nlinarith [hd.isLt, d.isLt]⟩

/-- Extract the K slice `qkv[:, num_q * hd : (num_q + num_kv) * hd].reshape(T, num_kv, hd)`
(`qk_norm_rope.py:61-65`). -/
def extractK {T numQ numKV h : ℕ} (inp : QkNormRopeIn T numQ numKV h) :
    Tensor ℝ [T, numKV, 2 * h] :=
  Tensor.of3 fun t hd d =>
    inp.qkv.get2 t ⟨numQ * (2 * h) + hd.val * (2 * h) + d.val, by nlinarith [hd.isLt, d.isLt]⟩

/-- Unpack `cos = cs[:, :half]` from packed `cos_sin` (`qk_norm_rope.py:69`). -/
def unpackCos {T h : ℕ} (cs : Tensor ℝ [T, 2 * h]) : Tensor ℝ [T, h] :=
  Tensor.of2 fun t j => cs.get2 t ⟨j.val, by omega⟩

/-- Unpack `sin = cs[:, half:]` from packed `cos_sin` (`qk_norm_rope.py:70`). -/
def unpackSin {T h : ℕ} (cs : Tensor ℝ [T, 2 * h]) : Tensor ℝ [T, h] :=
  Tensor.of2 fun t j => cs.get2 t ⟨j.val + h, by omega⟩

/-- Per-head RMSNorm matching `_rms` (`qk_norm_rope.py:24-29`). -/
def rmsHead (ops : FOps) (eps : ℝ) {D : ℕ} (x w : Fin D → ℝ) : Fin D → ℝ :=
  let var := ops.headMean D fun k => ops.mul (x k) (x k)
  let invRms := ops.rsqrt (ops.add var eps)
  fun d => ops.mul (ops.mul (x d) invRms) (w d)

/-- Per-head NeoX RoPE matching `_rope_neox` (`qk_norm_rope.py:32-37`) over packed `cos_sin`
(`qk_norm_rope.py:67-70`). -/
def ropeNeoxHead (ops : FOps) {h : ℕ} (x cs : Fin (2 * h) → ℝ) : Fin (2 * h) → ℝ :=
  fun d =>
    if hd : d.val < h then
      let d2 : Fin (2 * h) := ⟨d.val + h, by omega⟩
      ops.sub (ops.mul (x d) (cs ⟨d.val, by omega⟩)) (ops.mul (x d2) (cs d2))
    else
      let d1 : Fin (2 * h) := ⟨d.val - h, by omega⟩
      ops.add (ops.mul (x d) (cs d1)) (ops.mul (x d1) (cs d))

/-- Full per-head pipeline: cast inputs to fp32, apply `_rms` then `_rope_neox`, and cast to output
dtype (`qk_norm_rope.py:60-76`). -/
def normRopeHead (ops : FOps) (eps : ℝ) {h : ℕ} (x cs w : Fin (2 * h) → ℝ) : Fin (2 * h) → ℝ :=
  let y := rmsHead ops eps (fun d => ops.toF32 (x d)) (fun d => ops.toF32 (w d))
  let r := ropeNeoxHead ops y (fun d => ops.toF32 (cs d))
  fun d => ops.toOut (r d)

/-- Opaque-ops tensor specification of fused QK RMSNorm + NeoX RoPE (`qk_norm_rope.py:40-76`). -/
def qkNormRopeOpsMath (ops : FOps) (eps : ℝ) {T numQ numKV h : ℕ}
    (inp : QkNormRopeIn T numQ numKV h) : QkNormRopeOut T numQ numKV h :=
  { q := Tensor.of3 fun t hd d =>
      normRopeHead ops eps
        (fun d' => (extractQ inp).get3 t hd d')
        (fun d' => inp.cosSin.get2 t d')
        (fun d' => inp.qWeight.get1 d') d
    k := Tensor.of3 fun t hd d =>
      normRopeHead ops eps
        (fun d' => (extractK inp).get3 t hd d')
        (fun d' => inp.cosSin.get2 t d')
        (fun d' => inp.kWeight.get1 d') d }

noncomputable section

/-- Exact real-arithmetic specification in terms of `Tensor.rmsNorm` and `Tensor.ropeNeox`. -/
def qkNormRopeExactMath (eps : ℝ) {T numQ numKV h : ℕ}
    (inp : QkNormRopeIn T numQ numKV h) : QkNormRopeOut T numQ numKV h :=
  let qIn := extractQ inp
  let kIn := extractK inp
  let cos := unpackCos inp.cosSin
  let sin := unpackSin inp.cosSin
  { q := Tensor.of3 fun t hd d =>
      let qNorm := Tensor.rmsNorm eps (Tensor.of2 fun hd' d' => qIn.get3 t hd' d') inp.qWeight
      let cosH := Tensor.of2 fun (_ : Fin numQ) j => cos.get2 t j
      let sinH := Tensor.of2 fun (_ : Fin numQ) j => sin.get2 t j
      (Tensor.ropeNeox qNorm cosH sinH).get2 hd d
    k := Tensor.of3 fun t hd d =>
      let kNorm := Tensor.rmsNorm eps (Tensor.of2 fun hd' d' => kIn.get3 t hd' d') inp.kWeight
      let cosH := Tensor.of2 fun (_ : Fin numKV) j => cos.get2 t j
      let sinH := Tensor.of2 fun (_ : Fin numKV) j => sin.get2 t j
      (Tensor.ropeNeox kNorm cosH sinH).get2 hd d }

/-- When `0 < eps`, the exact RMSNorm denominator `Real.sqrt (meanSq + eps)` is strictly positive
for every head vector, so `Tensor.rmsNorm` and `FOps.exact.rsqrt` never divide by zero. -/
theorem rmsNorm_denom_pos {D : ℕ} {eps : ℝ} (heps : 0 < eps) (x : Fin D → ℝ) :
    0 < Real.sqrt ((∑ k : Fin D, x k * x k) / (D : ℝ) + eps) := by
  apply Real.sqrt_pos.mpr
  have hsum : 0 ≤ ∑ k : Fin D, x k * x k :=
    Finset.sum_nonneg fun k _ => mul_self_nonneg (x k)
  have hmean : 0 ≤ (∑ k : Fin D, x k * x k) / (D : ℝ) :=
    div_nonneg hsum (Nat.cast_nonneg D)
  linarith

/-- With exact arithmetic, `rmsHead` agrees with `Tensor.rmsNorm`. -/
theorem rmsHead_exact {H D : ℕ} (eps : ℝ) (x : Tensor ℝ [H, D]) (w : Tensor ℝ [D])
    (hd : Fin H) (d : Fin D) :
    rmsHead FOps.exact eps (fun k => x.get2 hd k) (fun k => w.get1 k) d =
      (Tensor.rmsNorm eps x w).get2 hd d := by
  simp only [rmsHead, FOps.exact, Tensor.rmsNorm, Tensor.get2_of2]
  ring

/-- With exact arithmetic, `normRopeHead` equals `Tensor.rmsNorm` followed by `Tensor.ropeNeox`. -/
theorem normRopeHead_exact {H h : ℕ} (eps : ℝ) (x : Tensor ℝ [H, 2 * h])
    (cs : Fin (2 * h) → ℝ) (w : Tensor ℝ [2 * h]) (hd : Fin H) (d : Fin (2 * h)) :
    normRopeHead FOps.exact eps (fun d' => x.get2 hd d') cs (fun d' => w.get1 d') d =
      (Tensor.ropeNeox (Tensor.rmsNorm eps x w)
        (Tensor.of2 fun (_ : Fin H) j => cs ⟨j.val, by omega⟩)
        (Tensor.of2 fun (_ : Fin H) j => cs ⟨j.val + h, by omega⟩)).get2 hd d := by
  simp only [normRopeHead, ropeNeoxHead, Tensor.ropeNeox, Tensor.get2_of2]
  change (if hd_lt : d.val < h then
      rmsHead FOps.exact eps (fun d' => x.get2 hd d') (fun d' => w.get1 d') d *
        cs ⟨d.val, by omega⟩ -
      rmsHead FOps.exact eps (fun d' => x.get2 hd d') (fun d' => w.get1 d') ⟨d.val + h, by omega⟩ *
        cs ⟨d.val + h, by omega⟩
    else
      rmsHead FOps.exact eps (fun d' => x.get2 hd d') (fun d' => w.get1 d') d *
        cs ⟨d.val - h, by omega⟩ +
      rmsHead FOps.exact eps (fun d' => x.get2 hd d') (fun d' => w.get1 d') ⟨d.val - h, by omega⟩ *
        cs d) = _
  split_ifs with hd_lt
  · rw [rmsHead_exact eps x w hd d, rmsHead_exact eps x w hd ⟨d.val + h, by omega⟩]
  · rw [rmsHead_exact eps x w hd d, rmsHead_exact eps x w hd ⟨d.val - h, by omega⟩]
    have hd_eq : (⟨d.val - h + h, by omega⟩ : Fin (2 * h)) = d := Fin.ext (by dsimp; omega)
    rw [hd_eq]

/-- Instantiating `FOps.exact` links `qkNormRopeOpsMath` to `Tensor.rmsNorm` + `Tensor.ropeNeox`. -/
theorem qkNormRopeOpsMath_exact (eps : ℝ) {T numQ numKV h : ℕ}
    (inp : QkNormRopeIn T numQ numKV h) :
    qkNormRopeOpsMath FOps.exact eps inp = qkNormRopeExactMath eps inp := by
  dsimp only [qkNormRopeOpsMath, qkNormRopeExactMath]
  refine congrArg₂ QkNormRopeOut.mk ?_ ?_
  · apply Tensor.ext3
    intro t hd d
    simpa [unpackCos, unpackSin] using
      normRopeHead_exact eps (Tensor.of2 fun hd' d' => (extractQ inp).get3 t hd' d')
        (fun d' => inp.cosSin.get2 t d') inp.qWeight hd d
  · apply Tensor.ext3
    intro t hd d
    simpa [unpackCos, unpackSin] using
      normRopeHead_exact eps (Tensor.of2 fun hd' d' => (extractK inp).get3 t hd' d')
        (fun d' => inp.cosSin.get2 t d') inp.kWeight hd d

end

/-- Four-part `KernelSpec` for fused QK RMSNorm + NeoX RoPE over `ops` (`qk_norm_rope.py:79-119`). -/
def qkNormRopeSpec (ops : FOps) (T numQ numKV h : ℕ) (eps : ℝ)
    (qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf : BufId) :
    KernelSpec (QkNormRopeIn T numQ numKV h) Unit (QkNormRopeOut T numQ numKV h) :=
  { readIn := fun m =>
      { qkv := (Layout.rowMajor qkvBuf 0 [T, (numQ + 2 * numKV) * (2 * h)]).read m
        cosSin := (Layout.rowMajor csBuf 0 [T, 2 * h]).read m
        qWeight := (Layout.rowMajor qwBuf 0 [2 * h]).read m
        kWeight := (Layout.rowMajor kwBuf 0 [2 * h]).read m }
    readSt := fun _ => ()
    readOut := fun m =>
      { q := (Layout.rowMajor qOutBuf 0 [T, numQ, 2 * h]).read m
        k := (Layout.rowMajor kOutBuf 0 [T, numKV, 2 * h]).read m }
    footprint := Mem.bufFootprint qOutBuf ∪ Mem.bufFootprint kOutBuf
    math := fun _ inp => qkNormRopeOpsMath ops eps inp
    mutate := fun _ _ => ()
    requires := fun _ _ =>
      0 < eps ∧ 0 < T ∧ 0 < numQ ∧ 0 < numKV ∧ 0 < h ∧
      qOutBuf ≠ kOutBuf ∧
      ∀ b ∈ [qOutBuf, kOutBuf],
        b ≠ qkvBuf ∧ b ≠ csBuf ∧ b ≠ qwBuf ∧ b ≠ kwBuf }

/-- Pallas `GridSpec` of `_qk_norm_rope_jax` (`qk_norm_rope.py:92-119`). The `qkv` input block has
width `numQ * (2 * h) + numKV * (2 * h)` at column 0 of the wider `[T, (numQ + 2 * numKV) * (2 * h)]`
array so `v` columns are never DMAd into VMEM. -/
def qkNormRopeGridSpec (T numQ numKV h bm : ℕ)
    (qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf : BufId) : Pallas.GridSpec :=
  let hd := 2 * h
  let qSize := numQ * hd
  let kvSize := numKV * hd
  { grid := [(T + bm - 1) / bm]
    inSpecs :=
      [ ({ buf := qkvBuf, shape := [T, (numQ + 2 * numKV) * hd] },
         Pallas.rowBlockSpec bm (qSize + kvSize)),
        ({ buf := csBuf, shape := [T, hd] },
         Pallas.rowBlockSpec bm hd),
        ({ buf := qwBuf, shape := [hd] },
         Pallas.vecBlockSpec hd),
        ({ buf := kwBuf, shape := [hd] },
         Pallas.vecBlockSpec hd) ]
    outSpecs :=
      [ ({ buf := qOutBuf, shape := [T, qSize] },
         Pallas.rowBlockSpec bm qSize),
        ({ buf := kOutBuf, shape := [T, kvSize] },
         Pallas.rowBlockSpec bm kvSize) ] }

/-- Pallas kernel body matching `_qk_norm_rope_kernel` (`qk_norm_rope.py:40-76`). -/
def qkNormRopeBody (ops : FOps) (numQ numKV h : ℕ) (eps : ℝ) : Pallas.Body :=
  fun _ _ ins _ =>
    let raw := ins.getD 0 (fun _ => 0)
    let csBlk := ins.getD 1 (fun _ => 0)
    let qwBlk := ins.getD 2 (fun _ => 0)
    let kwBlk := ins.getD 3 (fun _ => 0)
    let hd := 2 * h
    let qSize := numQ * hd
    let qOutBlk : List ℕ → ℝ := fun loc =>
      let row := loc.getD 0 0
      let col := loc.getD 1 0
      if hpos : 0 < hd then
        let headIdx := col / hd
        let dIdx := col % hd
        if headIdx < numQ then
          normRopeHead ops eps
            (fun d => raw [row, headIdx * hd + d.val])
            (fun d => csBlk [row, d.val])
            (fun d => qwBlk [d.val])
            ⟨dIdx, Nat.mod_lt col hpos⟩
        else 0
      else 0
    let kOutBlk : List ℕ → ℝ := fun loc =>
      let row := loc.getD 0 0
      let col := loc.getD 1 0
      if hpos : 0 < hd then
        let headIdx := col / hd
        let dIdx := col % hd
        if headIdx < numKV then
          normRopeHead ops eps
            (fun d => raw [row, qSize + headIdx * hd + d.val])
            (fun d => csBlk [row, d.val])
            (fun d => kwBlk [d.val])
            ⟨dIdx, Nat.mod_lt col hpos⟩
        else 0
      else 0
    [qOutBlk, kOutBlk]

private theorem mul_add_div_eq {a b c : ℕ} (hc : c < b) : (a * b + c) / b = a := by
  have hb : 0 < b := Nat.lt_of_le_of_lt (Nat.zero_le _) hc
  rw [Nat.mul_comm, Nat.mul_add_div hb, Nat.div_eq_of_lt hc, Nat.add_zero]

private theorem mul_add_mod_eq {a b c : ℕ} (hc : c < b) : (a * b + c) % b = c := by
  rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hc]

/-- `_qk_norm_rope_jax` is well defined for every positive `T`, positive block size `bm`, positive
head counts and head dimension, and distinct output buffers disjoint from input buffers. -/
theorem qkNormRope_wellDefined (ops : FOps) (T numQ numKV h bm : ℕ)
    (hT : 0 < T) (hbm : 0 < bm) (hnumQ : 0 < numQ) (hnumKV : 0 < numKV) (hh : 0 < h) (eps : ℝ)
    (qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf : BufId) (hqk : qOutBuf ≠ kOutBuf)
    (hdisj : ∀ b ∈ [qOutBuf, kOutBuf], b ≠ qkvBuf ∧ b ≠ csBuf ∧ b ≠ qwBuf ∧ b ≠ kwBuf)
    (m0 : Mem) :
    (qkNormRopeGridSpec T numQ numKV h bm qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf).WellDefined
      (qkNormRopeBody ops numQ numKV h eps) m0 := by
  have hlen : (Pallas.gridPoints [(T + bm - 1) / bm]).length = (T + bm - 1) / bm := by
    simp [Pallas.gridPoints_length]
  have hhd : 0 < 2 * h := by omega
  have hq : 0 < numQ * (2 * h) := Nat.mul_pos hnumQ hhd
  have hkv : 0 < numKV * (2 * h) := Nat.mul_pos hnumKV hhd
  have hqkW : 0 < numQ * (2 * h) + numKV * (2 * h) := by omega
  have hnumQz : (0 : ℤ) < numQ := by exact_mod_cast hnumQ
  have hnumKVz : (0 : ℤ) < numKV := by exact_mod_cast hnumKV
  have hhz : (0 : ℤ) < h := by exact_mod_cast hh
  have h1 := hdisj qOutBuf (by simp)
  have h2 := hdisj kOutBuf (by simp)
  constructor
  · intro d hd
    simp only [qkNormRopeGridSpec, List.mem_singleton] at hd
    subst hd
    exact Pallas.cdiv_pos hT hbm
  · intro p hp
    simp only [qkNormRopeGridSpec, List.cons_append, List.nil_append, List.mem_cons,
      List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Pallas.rowBlockSpec, Pallas.vecBlockSpec, hbm, hhd, hq, hkv, hqkW]
  · intro j hj p hp
    simp only [qkNormRopeGridSpec, hlen] at hj
    have hjz : (j : ℤ) * bm < T := by exact_mod_cast Pallas.rowBlock_lt_of_cdiv hbm hj
    simp only [qkNormRopeGridSpec, List.cons_append, List.nil_append, List.mem_cons,
      List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp only [Pallas.GridSpec.idxAt, qkNormRopeGridSpec, Pallas.gridPoints_one_getD _ j hj,
        Pallas.rowBlockSpec, Pallas.vecBlockSpec] <;>
      simp [Pallas.blockOk, hbm, hh, hq, hkv, hqkW, hjz] <;>
      positivity
  · intro p hp
    simp only [qkNormRopeGridSpec, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;>
      · simp only [qkNormRopeGridSpec, hlen]
        exact Pallas.noRevisit_rowBlock ((T + bm - 1) / bm) bm _ (fun _ => 0)
  · intro _ _ _ _
    rfl
  · intro s t _ _ i hi o ho hov
    simp only [qkNormRopeGridSpec, List.mem_cons, List.not_mem_nil, or_false] at hi ho
    rcases ho with rfl | rfl <;> rcases hi with rfl | rfl | rfl | rfl
    · exact h1.1 hov.1.symm
    · exact h1.2.1 hov.1.symm
    · exact h1.2.2.1 hov.1.symm
    · exact h1.2.2.2 hov.1.symm
    · exact h2.1 hov.1.symm
    · exact h2.2.1 hov.1.symm
    · exact h2.2.2.1 hov.1.symm
    · exact h2.2.2.2 hov.1.symm
  · exact Pallas.GridSpec.AliasesValid.of_empty rfl (by simp [qkNormRopeGridSpec, hqk]) (by
      intro inp hinp outp houtp
      simp only [qkNormRopeGridSpec, List.mem_cons, List.not_mem_nil, or_false] at hinp houtp
      rcases houtp with rfl | rfl <;> rcases hinp with rfl | rfl | rfl | rfl
      · exact h1.1.symm
      · exact h1.2.1.symm
      · exact h1.2.2.1.symm
      · exact h1.2.2.2.symm
      · exact h2.1.symm
      · exact h2.2.1.symm
      · exact h2.2.2.1.symm
      · exact h2.2.2.2.symm)
      (by simp [qkNormRopeGridSpec])
  · rfl
  · rfl
  · intro pg₁ h₁ pg₂ h₂ hne
    exact absurd ((List.mem_singleton.mp h₁).trans (List.mem_singleton.mp h₂).symm) hne
  · intro pg₁ h₁ pg₂ h₂ hne
    exact absurd ((List.mem_singleton.mp h₁).trans (List.mem_singleton.mp h₂).symm) hne
  · intro hne
    exact absurd rfl hne

/-- `_qk_norm_rope_jax` writes both outputs from scratch into distinct buffers disjoint from all
inputs (`Pallas.StoreOnly`). -/
theorem qkNormRope_storeOnly (ops : FOps) (T numQ numKV h bm : ℕ) (eps : ℝ)
    (qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf : BufId) (hqk : qOutBuf ≠ kOutBuf)
    (hdisj : ∀ b ∈ [qOutBuf, kOutBuf], b ≠ qkvBuf ∧ b ≠ csBuf ∧ b ≠ qwBuf ∧ b ≠ kwBuf) :
    Pallas.StoreOnly
      (qkNormRopeGridSpec T numQ numKV h bm qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf)
      (qkNormRopeBody ops numQ numKV h eps) where
  outs_nodup := by simp [qkNormRopeGridSpec, hqk]
  ins_disjoint := by
    have h1 := hdisj qOutBuf (by simp)
    have h2 := hdisj kOutBuf (by simp)
    intro p hp o ho
    simp only [qkNormRopeGridSpec, List.mem_cons, List.not_mem_nil, or_false] at hp ho
    rcases ho with rfl | rfl <;> rcases hp with rfl | rfl | rfl | rfl
    · exact h1.1.symm
    · exact h1.2.1.symm
    · exact h1.2.2.1.symm
    · exact h1.2.2.2.symm
    · exact h2.1.symm
    · exact h2.2.1.symm
    · exact h2.2.2.1.symm
    · exact h2.2.2.2.symm
  ignores_prev := fun _ _ _ _ => rfl

/-- No grid step's `qkv` input block ever covers a `v` column (`qk_norm_rope.py:55-57,104-105`). -/
theorem qkNormRope_v_block_disjoint (T numQ numKV h bm : ℕ)
    (qkvBuf : BufId) (g : List ℕ) (sc : ℕ → ℤ) (t c : ℕ)
    (hv_lo : numQ * (2 * h) + numKV * (2 * h) ≤ c)
    (hv_hi : c < (numQ + 2 * numKV) * (2 * h)) :
    Pallas.addrInBlock
      { buf := qkvBuf, shape := [T, (numQ + 2 * numKV) * (2 * h)] }
      [bm, numQ * (2 * h) + numKV * (2 * h)]
      ((Pallas.rowBlockSpec bm (numQ * (2 * h) + numKV * (2 * h))).blockIdx g sc)
      (t * ((numQ + 2 * numKV) * (2 * h)) + c) = false := by
  simp only [Pallas.BlockSpec.blockIdx, Pallas.rowBlockSpec, List.map_cons, List.map_nil,
    Int.toNat_natCast, Int.toNat_zero]
  exact Pallas.addrInBlock_rowBlock_col_ge hv_lo hv_hi

/-- Every `loadBlock` of `qkv` is invariant under arbitrary changes to the `v` columns
(columns at or past `numQ * (2 * h) + numKV * (2 * h)`). -/
theorem qkNormRope_loadQkv_ignores_v (T numQ numKV h bm : ℕ)
    (qkvBuf : BufId) (i : ℕ) (junk : ℕ → ℝ) {m₁ m₂ : Mem}
    (hagree : ∀ t < T, ∀ c < numQ * (2 * h) + numKV * (2 * h),
      m₁.read qkvBuf (t * ((numQ + 2 * numKV) * (2 * h)) + c) =
        m₂.read qkvBuf (t * ((numQ + 2 * numKV) * (2 * h)) + c)) :
    Pallas.loadBlock
      { buf := qkvBuf, shape := [T, (numQ + 2 * numKV) * (2 * h)] }
      [bm, numQ * (2 * h) + numKV * (2 * h)] [i, 0] junk m₁ =
    Pallas.loadBlock
      { buf := qkvBuf, shape := [T, (numQ + 2 * numKV) * (2 * h)] }
      [bm, numQ * (2 * h) + numKV * (2 * h)] [i, 0] junk m₂ :=
  Pallas.loadBlock_rowBlock_congr_prefix junk hagree

noncomputable section

/-- `qkNormRopeSpec` has a well-formed layout whenever both output buffers are disjoint from all
four input buffers. -/
theorem qkNormRopeSpec_wellFormed (ops : FOps) (T numQ numKV h : ℕ) (eps : ℝ)
    (qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf : BufId)
    (hdisj : ∀ b ∈ [qOutBuf, kOutBuf], b ≠ qkvBuf ∧ b ≠ csBuf ∧ b ≠ qwBuf ∧ b ≠ kwBuf) :
    (qkNormRopeSpec ops T numQ numKV h eps qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf).WellFormed where
  readOut_on_footprint _ _ hagree :=
    congrArg₂ QkNormRopeOut.mk
      (Layout.read_eq_of_agreeOn _
        ((Layout.footprint_subset_buf _).trans Set.subset_union_left) hagree)
      (Layout.read_eq_of_agreeOn _
        ((Layout.footprint_subset_buf _).trans Set.subset_union_right) hagree)
  readSt_on_footprint _ _ _ := rfl
  readIn_outside_footprint _ _ hagree := by
    have h1 := hdisj qOutBuf (by simp)
    have h2 := hdisj kOutBuf (by simp)
    have hqkv := Layout.read_eq_of_agreeOutside
      (Layout.rowMajor qkvBuf 0 [T, (numQ + 2 * numKV) * (2 * h)])
      (Set.disjoint_union_right.mpr
        ⟨Layout.disjoint_bufFootprint _ h1.1.symm, Layout.disjoint_bufFootprint _ h2.1.symm⟩)
      hagree
    have hcs := Layout.read_eq_of_agreeOutside
      (Layout.rowMajor csBuf 0 [T, 2 * h])
      (Set.disjoint_union_right.mpr
        ⟨Layout.disjoint_bufFootprint _ h1.2.1.symm, Layout.disjoint_bufFootprint _ h2.2.1.symm⟩)
      hagree
    have hqw := Layout.read_eq_of_agreeOutside
      (Layout.rowMajor qwBuf 0 [2 * h])
      (Set.disjoint_union_right.mpr
        ⟨Layout.disjoint_bufFootprint _ h1.2.2.1.symm, Layout.disjoint_bufFootprint _ h2.2.2.1.symm⟩)
      hagree
    have hkw := Layout.read_eq_of_agreeOutside
      (Layout.rowMajor kwBuf 0 [2 * h])
      (Set.disjoint_union_right.mpr
        ⟨Layout.disjoint_bufFootprint _ h1.2.2.2.symm, Layout.disjoint_bufFootprint _ h2.2.2.2.symm⟩)
      hagree
    dsimp only [qkNormRopeSpec]
    rw [hqkv, hcs, hqw, hkw]

/-- Main `pallas_call` correctness theorem for `_qk_norm_rope_jax` (`qk_norm_rope.py:79-119`):
for every `T`, positive block size `bm`, `ops` and junk, whenever the spec preconditions hold
(including `0 < eps`), the run is defined, both outputs match `qkNormRopeOpsMath`, and nothing
outside `{qOutBuf, kOutBuf}` changes. -/
theorem qkNormRope_pallas_correct (ops : FOps) (T numQ numKV h bm : ℕ) (hbm : 0 < bm)
    (eps : ℝ) (qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf : BufId) :
    (qkNormRopeSpec ops T numQ numKV h eps qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf).Correct
      (Pallas.pallasCall (qkNormRopeGridSpec T numQ numKV h bm
        qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf)
        (qkNormRopeBody ops numQ numKV h eps)) := by
  intro junk m0 hreq
  rcases hreq with ⟨_heps, hT, hnumQ, hnumKV, hh, hqk, hdisj⟩
  refine ⟨qkNormRopeSpec_wellFormed ops T numQ numKV h eps
    qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf hdisj, ?_⟩
  set gs := qkNormRopeGridSpec T numQ numKV h bm qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf with hgs
  set body := qkNormRopeBody ops numQ numKV h eps with hbody
  have hwd := qkNormRope_wellDefined ops T numQ numKV h bm hT hbm hnumQ hnumKV hh eps
    qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf hqk hdisj m0
  have hso := qkNormRope_storeOnly ops T numQ numKV h bm eps
    qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf hqk hdisj
  refine ⟨Pallas.pallasExec gs body junk m0,
    Pallas.pallasCall_of_wellDefined junk hwd, ?_, rfl, ?_⟩
  · have hhd : 0 < 2 * h := by omega
    have hbw : numQ * (2 * h) + numKV * (2 * h) ≤ (numQ + 2 * numKV) * (2 * h) := by nlinarith
    have hqkv_in : ∀ (t : Fin T) (c : ℕ) (hc : c < numQ * (2 * h) + numKV * (2 * h)),
        (Pallas.loadIns gs junk m0 [t.val / bm]).getD 0 (fun _ => 0)
          [t.val - t.val / bm * bm, c] =
          m0.read qkvBuf (t.val * ((numQ + 2 * numKV) * (2 * h)) + c) := by
      intro t c hc
      rw [Pallas.loadIns_getD gs junk m0 [t.val / bm] 0
        ({ buf := qkvBuf, shape := [T, (numQ + 2 * numKV) * (2 * h)] },
         Pallas.rowBlockSpec bm (numQ * (2 * h) + numKV * (2 * h)))
        (by simp [hgs, qkNormRopeGridSpec])]
      simp only [Pallas.BlockSpec.blockIdx, Pallas.rowBlockSpec, List.map_cons, List.map_nil,
        List.getD_cons_zero, Int.toNat_natCast, Int.toNat_zero]
      exact Pallas.loadBlock_rowBlock_eq hbm t.isLt hc hbw _ m0
    have hcs_in : ∀ (t : Fin T) (d : Fin (2 * h)),
        (Pallas.loadIns gs junk m0 [t.val / bm]).getD 1 (fun _ => 0)
          [t.val - t.val / bm * bm, d.val] =
          m0.read csBuf (t.val * (2 * h) + d.val) := by
      intro t d
      rw [Pallas.loadIns_getD gs junk m0 [t.val / bm] 1
        ({ buf := csBuf, shape := [T, 2 * h] }, Pallas.rowBlockSpec bm (2 * h))
        (by simp [hgs, qkNormRopeGridSpec])]
      simp only [Pallas.BlockSpec.blockIdx, Pallas.rowBlockSpec, List.map_cons, List.map_nil,
        List.getD_cons_zero, Int.toNat_natCast, Int.toNat_zero]
      exact Pallas.loadBlock_rowBlock_eq hbm t.isLt d.isLt le_rfl _ m0
    have hqw_in : ∀ (t : Fin T) (d : Fin (2 * h)),
        (Pallas.loadIns gs junk m0 [t.val / bm]).getD 2 (fun _ => 0) [d.val] =
          m0.read qwBuf d.val := by
      intro t d
      rw [Pallas.loadIns_getD gs junk m0 [t.val / bm] 2
        ({ buf := qwBuf, shape := [2 * h] }, Pallas.vecBlockSpec (2 * h))
        (by simp [hgs, qkNormRopeGridSpec])]
      simp only [Pallas.BlockSpec.blockIdx, Pallas.vecBlockSpec, List.map_cons, List.map_nil,
        Int.toNat_zero]
      exact Pallas.loadBlock_vecBlock_eq d.isLt _ m0
    have hkw_in : ∀ (t : Fin T) (d : Fin (2 * h)),
        (Pallas.loadIns gs junk m0 [t.val / bm]).getD 3 (fun _ => 0) [d.val] =
          m0.read kwBuf d.val := by
      intro t d
      rw [Pallas.loadIns_getD gs junk m0 [t.val / bm] 3
        ({ buf := kwBuf, shape := [2 * h] }, Pallas.vecBlockSpec (2 * h))
        (by simp [hgs, qkNormRopeGridSpec])]
      simp only [Pallas.BlockSpec.blockIdx, Pallas.vecBlockSpec, List.map_cons, List.map_nil,
        Int.toNat_zero]
      exact Pallas.loadBlock_vecBlock_eq d.isLt _ m0
    dsimp only [qkNormRopeSpec, qkNormRopeOpsMath]
    refine congrArg₂ QkNormRopeOut.mk ?_ ?_
    · apply Tensor.ext3
      intro t hd d
      have hcol : hd.val * (2 * h) + d.val < numQ * (2 * h) := by nlinarith [hd.isLt, d.isLt]
      change (Pallas.pallasExec gs body junk m0).read qOutBuf
          (0 + Idx.flat (s := [T, numQ, 2 * h]) (t, hd, d, ())) = _
      simp only [Idx.flat_three, zero_add]
      rw [Pallas.pallasExec_read_out gs body junk m0 hso 0
        (⟨qOutBuf, [T, numQ * (2 * h)]⟩, Pallas.rowBlockSpec bm (numQ * (2 * h)))
        (by simp [hgs, qkNormRopeGridSpec])]
      rw [show gs.grid = [(T + bm - 1) / bm] by simp [hgs, qkNormRopeGridSpec]]
      rw [Pallas.seqOut_rowBlock_read hbm _ _ _ t ⟨hd.val * (2 * h) + d.val, hcol⟩]
      simp only [Pallas.outVal, hbody, qkNormRopeBody, List.getD_cons_succ, List.getD_cons_zero,
        hhd, dite_true, mul_add_div_eq d.isLt, mul_add_mod_eq d.isLt, hd.isLt, ite_true]
      simp only [Tensor.get3_of3, extractQ, Tensor.get2, Tensor.get1,
        Layout.read, Layout.rowMajor, Idx.flat_two, Idx.flat_one, zero_add]
      have hx : (fun d' : Fin (2 * h) => (Pallas.loadIns gs junk m0 [t.val / bm]).getD 0 (fun _ => 0)
          [t.val - t.val / bm * bm, hd.val * (2 * h) + d'.val]) =
          (fun d' : Fin (2 * h) => m0.read qkvBuf
            (t.val * ((numQ + 2 * numKV) * (2 * h)) + (hd.val * (2 * h) + d'.val))) :=
        funext fun d' => hqkv_in t (hd.val * (2 * h) + d'.val) (by nlinarith [hd.isLt, d'.isLt])
      have hcs : (fun d' : Fin (2 * h) => (Pallas.loadIns gs junk m0 [t.val / bm]).getD 1 (fun _ => 0)
          [t.val - t.val / bm * bm, d'.val]) =
          (fun d' : Fin (2 * h) => m0.read csBuf (t.val * (2 * h) + d'.val)) :=
        funext fun d' => hcs_in t d'
      have hqw : (fun d' : Fin (2 * h) => (Pallas.loadIns gs junk m0 [t.val / bm]).getD 2 (fun _ => 0)
          [d'.val]) = (fun d' : Fin (2 * h) => m0.read qwBuf d'.val) :=
        funext fun d' => hqw_in t d'
      rw [hx, hcs, hqw]
    · apply Tensor.ext3
      intro t hd d
      have hcol : hd.val * (2 * h) + d.val < numKV * (2 * h) := by nlinarith [hd.isLt, d.isLt]
      change (Pallas.pallasExec gs body junk m0).read kOutBuf
          (0 + Idx.flat (s := [T, numKV, 2 * h]) (t, hd, d, ())) = _
      simp only [Idx.flat_three, zero_add]
      rw [Pallas.pallasExec_read_out gs body junk m0 hso 1
        (⟨kOutBuf, [T, numKV * (2 * h)]⟩, Pallas.rowBlockSpec bm (numKV * (2 * h)))
        (by simp [hgs, qkNormRopeGridSpec])]
      rw [show gs.grid = [(T + bm - 1) / bm] by simp [hgs, qkNormRopeGridSpec]]
      rw [Pallas.seqOut_rowBlock_read hbm _ _ _ t ⟨hd.val * (2 * h) + d.val, hcol⟩]
      simp only [Pallas.outVal, hbody, qkNormRopeBody, List.getD_cons_succ, List.getD_cons_zero,
        hhd, dite_true, mul_add_div_eq d.isLt, mul_add_mod_eq d.isLt, hd.isLt, ite_true]
      simp only [Tensor.get3_of3, extractK, Tensor.get2, Tensor.get1,
        Layout.read, Layout.rowMajor, Idx.flat_two, Idx.flat_one, zero_add]
      have hx : (fun d' : Fin (2 * h) => (Pallas.loadIns gs junk m0 [t.val / bm]).getD 0 (fun _ => 0)
          [t.val - t.val / bm * bm, numQ * (2 * h) + hd.val * (2 * h) + d'.val]) =
          (fun d' : Fin (2 * h) => m0.read qkvBuf
            (t.val * ((numQ + 2 * numKV) * (2 * h)) +
              (numQ * (2 * h) + hd.val * (2 * h) + d'.val))) :=
        funext fun d' => hqkv_in t (numQ * (2 * h) + hd.val * (2 * h) + d'.val)
          (by nlinarith [hd.isLt, d'.isLt])
      have hcs : (fun d' : Fin (2 * h) => (Pallas.loadIns gs junk m0 [t.val / bm]).getD 1 (fun _ => 0)
          [t.val - t.val / bm * bm, d'.val]) =
          (fun d' : Fin (2 * h) => m0.read csBuf (t.val * (2 * h) + d'.val)) :=
        funext fun d' => hcs_in t d'
      have hkw : (fun d' : Fin (2 * h) => (Pallas.loadIns gs junk m0 [t.val / bm]).getD 3 (fun _ => 0)
          [d'.val]) = (fun d' : Fin (2 * h) => m0.read kwBuf d'.val) :=
        funext fun d' => hkw_in t d'
      rw [hx, hcs, hkw]
  · refine Mem.agreeOutside_mono ?_ (Pallas.pallasExec_agreeOutside_outBufs gs body junk m0)
    rintro ⟨b, a⟩ ⟨x, hx, hbuf⟩
    simp only [hgs, qkNormRopeGridSpec, List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl
    · exact Or.inl hbuf.symm
    · exact Or.inr hbuf.symm

/-- Correctness with the default `_MAX_BLOCK_TOKENS` block size `min(T, 256)` (`qk_norm_rope.py:21,95`)
for any non-empty batch (`qk_norm_rope.py:224-225`). -/
theorem qkNormRope_defaultBlock_correct (ops : FOps) (T numQ numKV h : ℕ) (hT : 0 < T)
    (eps : ℝ) (qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf : BufId) :
    (qkNormRopeSpec ops T numQ numKV h eps qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf).Correct
      (Pallas.pallasCall (qkNormRopeGridSpec T numQ numKV h (defaultBlockTokens T)
        qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf)
        (qkNormRopeBody ops numQ numKV h eps)) :=
  qkNormRope_pallas_correct ops T numQ numKV h (defaultBlockTokens T)
    (by simp [defaultBlockTokens, maxBlockTokens, hT]) eps
    qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf

/-- Exact real-arithmetic `pallas_call` corollary: with `FOps.exact` and `0 < eps`, both outputs
equal `Tensor.rmsNorm` followed by `Tensor.ropeNeox` and every RMSNorm denominator is strictly
positive. -/
theorem qkNormRope_exact_correct (T numQ numKV h bm : ℕ) (hbm : 0 < bm)
    (eps : ℝ) (qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf : BufId) :
    let spec := qkNormRopeSpec FOps.exact T numQ numKV h eps
      qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf
    ∀ (junk : ℕ → ℝ) (m0 : Mem),
      spec.requires () (spec.readIn m0) →
      ∃ m',
        Pallas.pallasCall (qkNormRopeGridSpec T numQ numKV h bm
          qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf)
          (qkNormRopeBody FOps.exact numQ numKV h eps) junk m0 = some m' ∧
        spec.readOut m' = qkNormRopeExactMath eps (spec.readIn m0) ∧
        Mem.agreeOutside spec.footprint m0 m' := by
  intro spec junk m0 hreq
  obtain ⟨-, m', hcall, hout, -, hframe⟩ :=
    qkNormRope_pallas_correct FOps.exact T numQ numKV h bm hbm eps
      qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf junk m0 hreq
  exact ⟨m', hcall, hout.trans (qkNormRopeOpsMath_exact eps (spec.readIn m0)), hframe⟩

/-- The `v` slice in `qkvBuf` is unchanged after execution, justifying the zero-copy view
`v = qkv[:, q_size + kv_size :]` (`qk_norm_rope.py:274-276`). -/
theorem qkNormRope_v_unchanged (ops : FOps) (T numQ numKV h bm : ℕ) (hbm : 0 < bm)
    (eps : ℝ) (heps : 0 < eps) (hnumQ : 0 < numQ) (hnumKV : 0 < numKV) (hh : 0 < h)
    (qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf : BufId) (hqk : qOutBuf ≠ kOutBuf)
    (hdisj : ∀ b ∈ [qOutBuf, kOutBuf], b ≠ qkvBuf ∧ b ≠ csBuf ∧ b ≠ qwBuf ∧ b ≠ kwBuf)
    (junk : ℕ → ℝ) (m0 m' : Mem)
    (hrun : Pallas.pallasCall
      (qkNormRopeGridSpec T numQ numKV h bm qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf)
      (qkNormRopeBody ops numQ numKV h eps) junk m0 = some m')
    (t : Fin T) (hd : Fin numKV) (d : Fin (2 * h)) :
    let off := t.val * ((numQ + 2 * numKV) * (2 * h)) +
      ((numQ + numKV) * (2 * h) + hd.val * (2 * h) + d.val)
    m'.read qkvBuf off = m0.read qkvBuf off := by
  have hT : 0 < T := Nat.lt_of_le_of_lt (Nat.zero_le _) t.isLt
  obtain ⟨-, n, hcall, -, -, hframe⟩ :=
    qkNormRope_pallas_correct ops T numQ numKV h bm hbm eps
      qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf junk m0
      ⟨heps, hT, hnumQ, hnumKV, hh, hqk, hdisj⟩
  rw [hcall, Option.some.injEq] at hrun
  subst hrun
  have h1 := (hdisj qOutBuf (by simp)).1
  have h2 := (hdisj kOutBuf (by simp)).1
  exact (hframe (qkvBuf, _) (by
    rintro (h | h)
    · exact h1 h.symm
    · exact h2 h.symm)).symm

/-- Batch invariance of `_qk_norm_rope_jax`, bit for bit under the `FOps` assumptions: a token's
`q` and `k` outputs depend only on that token's Q/K slice of `qkv`, its `cos_sin` row, and the
Q/K norm weights — and not on `T`, `bm`, junk, other tokens, or any `v` column. -/
theorem qkNormRope_batch_invariant (ops : FOps) {T T' numQ numKV h bm bm' : ℕ}
    (hbm : 0 < bm) (hbm' : 0 < bm') {eps : ℝ} (heps : 0 < eps)
    (hnumQ : 0 < numQ) (hnumKV : 0 < numKV) (hh : 0 < h)
    {qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf : BufId}
    {qkvBuf' csBuf' qwBuf' kwBuf' qOutBuf' kOutBuf' : BufId}
    (hqk : qOutBuf ≠ kOutBuf)
    (hdisj : ∀ b ∈ [qOutBuf, kOutBuf], b ≠ qkvBuf ∧ b ≠ csBuf ∧ b ≠ qwBuf ∧ b ≠ kwBuf)
    (hqk' : qOutBuf' ≠ kOutBuf')
    (hdisj' : ∀ b ∈ [qOutBuf', kOutBuf'], b ≠ qkvBuf' ∧ b ≠ csBuf' ∧ b ≠ qwBuf' ∧ b ≠ kwBuf')
    {junk junk' : ℕ → ℝ} {m0 m0' m₁ m₂ : Mem}
    (h₁ : Pallas.pallasCall
      (qkNormRopeGridSpec T numQ numKV h bm qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf)
      (qkNormRopeBody ops numQ numKV h eps) junk m0 = some m₁)
    (h₂ : Pallas.pallasCall
      (qkNormRopeGridSpec T' numQ numKV h bm' qkvBuf' csBuf' qwBuf' kwBuf' qOutBuf' kOutBuf')
      (qkNormRopeBody ops numQ numKV h eps) junk' m0' = some m₂)
    (t : Fin T) (t' : Fin T')
    (hqkv : ∀ c < (numQ + numKV) * (2 * h),
      m0.read qkvBuf (t.val * ((numQ + 2 * numKV) * (2 * h)) + c) =
        m0'.read qkvBuf' (t'.val * ((numQ + 2 * numKV) * (2 * h)) + c))
    (hcs : ∀ d : Fin (2 * h),
      m0.read csBuf (t.val * (2 * h) + d.val) = m0'.read csBuf' (t'.val * (2 * h) + d.val))
    (hqw : ∀ d : Fin (2 * h), m0.read qwBuf d.val = m0'.read qwBuf' d.val)
    (hkw : ∀ d : Fin (2 * h), m0.read kwBuf d.val = m0'.read kwBuf' d.val) :
    (∀ (hd : Fin numQ) (d : Fin (2 * h)),
      m₁.read qOutBuf (t.val * (numQ * (2 * h)) + (hd.val * (2 * h) + d.val)) =
        m₂.read qOutBuf' (t'.val * (numQ * (2 * h)) + (hd.val * (2 * h) + d.val))) ∧
    (∀ (hd : Fin numKV) (d : Fin (2 * h)),
      m₁.read kOutBuf (t.val * (numKV * (2 * h)) + (hd.val * (2 * h) + d.val)) =
        m₂.read kOutBuf' (t'.val * (numKV * (2 * h)) + (hd.val * (2 * h) + d.val))) := by
  have hT : 0 < T := Nat.lt_of_le_of_lt (Nat.zero_le _) t.isLt
  have hT' : 0 < T' := Nat.lt_of_le_of_lt (Nat.zero_le _) t'.isLt
  obtain ⟨-, n₁, e₁, v₁, -, -⟩ :=
    qkNormRope_pallas_correct ops T numQ numKV h bm hbm eps
      qkvBuf csBuf qwBuf kwBuf qOutBuf kOutBuf junk m0 ⟨heps, hT, hnumQ, hnumKV, hh, hqk, hdisj⟩
  obtain ⟨-, n₂, e₂, v₂, -, -⟩ :=
    qkNormRope_pallas_correct ops T' numQ numKV h bm' hbm' eps
      qkvBuf' csBuf' qwBuf' kwBuf' qOutBuf' kOutBuf' junk' m0'
      ⟨heps, hT', hnumQ, hnumKV, hh, hqk', hdisj'⟩
  rw [e₁, Option.some.injEq] at h₁
  rw [e₂, Option.some.injEq] at h₂
  subst h₁ h₂
  constructor
  · intro hd d
    have hq₁ := congrArg (fun o => o.q.get3 t hd d) v₁
    have hq₂ := congrArg (fun o => o.q.get3 t' hd d) v₂
    simp only [qkNormRopeSpec, qkNormRopeOpsMath, extractQ, Tensor.get3_of3] at hq₁ hq₂
    simp only [Tensor.get3, Tensor.get2, Tensor.get1, Layout.read, Layout.rowMajor,
      Idx.flat_three, Idx.flat_two, Idx.flat_one, zero_add] at hq₁ hq₂
    rw [hq₁, hq₂]
    have hx : (fun d' : Fin (2 * h) => m0.read qkvBuf
        (t.val * ((numQ + 2 * numKV) * (2 * h)) + (hd.val * (2 * h) + d'.val))) =
        (fun d' : Fin (2 * h) => m0'.read qkvBuf'
        (t'.val * ((numQ + 2 * numKV) * (2 * h)) + (hd.val * (2 * h) + d'.val))) :=
      funext fun d' => hqkv (hd.val * (2 * h) + d'.val) (by nlinarith [hd.isLt, d'.isLt])
    have hcs' : (fun d' : Fin (2 * h) => m0.read csBuf (t.val * (2 * h) + d'.val)) =
        (fun d' : Fin (2 * h) => m0'.read csBuf' (t'.val * (2 * h) + d'.val)) :=
      funext hcs
    have hqw' : (fun d' : Fin (2 * h) => m0.read qwBuf d'.val) =
        (fun d' : Fin (2 * h) => m0'.read qwBuf' d'.val) :=
      funext hqw
    rw [hx, hcs', hqw']
  · intro hd d
    have hk₁ := congrArg (fun o => o.k.get3 t hd d) v₁
    have hk₂ := congrArg (fun o => o.k.get3 t' hd d) v₂
    simp only [qkNormRopeSpec, qkNormRopeOpsMath, extractK, Tensor.get3_of3] at hk₁ hk₂
    simp only [Tensor.get3, Tensor.get2, Tensor.get1, Layout.read, Layout.rowMajor,
      Idx.flat_three, Idx.flat_two, Idx.flat_one, zero_add] at hk₁ hk₂
    rw [hk₁, hk₂]
    have hx : (fun d' : Fin (2 * h) => m0.read qkvBuf
        (t.val * ((numQ + 2 * numKV) * (2 * h)) +
          (numQ * (2 * h) + hd.val * (2 * h) + d'.val))) =
        (fun d' : Fin (2 * h) => m0'.read qkvBuf'
        (t'.val * ((numQ + 2 * numKV) * (2 * h)) +
          (numQ * (2 * h) + hd.val * (2 * h) + d'.val))) :=
      funext fun d' => hqkv (numQ * (2 * h) + hd.val * (2 * h) + d'.val)
        (by nlinarith [hd.isLt, d'.isLt])
    have hcs' : (fun d' : Fin (2 * h) => m0.read csBuf (t.val * (2 * h) + d'.val)) =
        (fun d' : Fin (2 * h) => m0'.read csBuf' (t'.val * (2 * h) + d'.val)) :=
      funext hcs
    have hkw' : (fun d' : Fin (2 * h) => m0.read kwBuf d'.val) =
        (fun d' : Fin (2 * h) => m0'.read kwBuf' d'.val) :=
      funext hkw
    rw [hx, hcs', hkw']

end

end TpuKernelLean.Examples.QkNormRope
