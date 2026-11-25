import TpuKernelLean.Pallas.RowKernel
import TpuKernelLean.Tensor

/-!
Verification of the batch-invariant RMSNorm kernels of sglang-torchtpu PR 2299
(`sglang_tpu/rl/true_on_policy/norm_kernel.py`): `rms_norm` (plain and Gemma), `rms_norm` with a
residual (plain and Gemma), and `gated_rms_norm`.

All three are `RowKernel`s: `_norm_call` tiles `[M, H]` inputs into `[rows, H]` blocks over a
`cdiv(M, rows)` grid with the weight as a `[1, H]` block. The float operations are opaque (`FOps`),
so the theorems hold for every implementation of them. That gives two results at once:

* Bitwise batch invariance (the PR's contract): an output row is one fixed expression in its own
  input row and the weight, for every `M`, row block size, junk and content of the other rows.
  This holds on hardware under the assumptions stated on `FOps`.
* Functional correctness: with exact real arithmetic (`FOps.exact`) the plain kernel is
  `TpuKernelLean.Tensor.rmsNorm`.

`dimension_semantics=("parallel",)` is modelled via `RowKernel.gridSpec`: each grid step writes its
own row block and reads no output buffer, so any multi-core split or reordering agrees with
sequential execution (`Pallas.pallasExecPar_eq_pallasExec`).
-/

namespace TpuKernelLean.Examples.TrueOnPolicyNorm

open TpuKernelLean.Pallas

/-- Opaque TPU float operations, as `_norm_kernel` uses them. Memory holds bf16 or fp32 values as
reals, and `astype(float32)` of a bf16 value is exact, so it is the identity. A theorem proved for
every `FOps` holds bit for bit on the TPU provided each op is a deterministic function of its
arguments and `rowMean H` (Mosaic's reduction of one width-`H` row for `jnp.mean(..., axis=1)`,
including the division by `H`) depends only on that row, not on the block's row count. -/
structure FOps where
  /-- `astype(bfloat16)` of an fp32 value. -/
  toBf16 : ℝ → ℝ
  add : ℝ → ℝ → ℝ
  mul : ℝ → ℝ → ℝ
  rsqrt : ℝ → ℝ
  silu : ℝ → ℝ
  /-- `jnp.mean` of one row of width `H`. -/
  rowMean : (H : ℕ) → (Fin H → ℝ) → ℝ

/-- Exact real arithmetic, for functional correctness. -/
noncomputable def FOps.exact : FOps where
  toBf16 := id
  add := (· + ·)
  mul := (· * ·)
  rsqrt := fun v => 1 / Real.sqrt v
  silu := fun z => z / (1 + Real.exp (-z))
  rowMean := fun H v => (∑ c, v c) / (H : ℝ)

variable (ops : FOps) (eps : ℝ)

/-- `s * rsqrt(mean(s * s) + eps) * w` rounded to bf16, with Gemma's `1 + w` weight
(`norm_kernel.py:32-37`). -/
def normRow (gemma : Bool) {H : ℕ} (s w : Fin H → ℝ) : Fin H → ℝ :=
  let var := ops.rowMean H fun c => ops.mul (s c) (s c)
  fun c => ops.toBf16 (ops.mul (ops.mul (s c) (ops.rsqrt (ops.add var eps)))
    (if gemma then ops.add 1 (w c) else w c))

/-- `rms_norm(x, w)` without a residual (`_norm_kernel` with `has_residual=False`). -/
abbrev plainKernel (gemma : Bool) (H : ℕ) : RowKernel H where
  nIn := 1
  nOut := 1
  f := fun ins w _ => normRow ops eps gemma (ins 0) w

/-- `rms_norm(x, residual, w)`: outputs `(out, new residual)`. The new residual is `bf16(x + r)`;
the plain norm normalizes the unrounded fp32 sum, Gemma the rounded one (`norm_kernel.py:22-28`). -/
abbrev residualKernel (gemma : Bool) (H : ℕ) : RowKernel H where
  nIn := 2
  nOut := 2
  f := fun ins w q =>
    let ro := fun c => ops.toBf16 (ops.add (ins 0 c) (ins 1 c))
    if q = 0 then normRow ops eps gemma (if gemma then ro else fun c => ops.add (ins 0 c) (ins 1 c)) w
    else ro

/-- `gated_rms_norm(x, z, w)`: `bf16(s * rsqrt(mean(s * s) + eps) * w * silu(z))`
(`norm_kernel.py:40-45`). -/
abbrev gatedKernel (H : ℕ) : RowKernel H where
  nIn := 2
  nOut := 1
  f := fun ins w _ =>
    let s := ins 0
    let var := ops.rowMean H fun c => ops.mul (s c) (s c)
    fun c => ops.toBf16 (ops.mul (ops.mul (ops.mul (s c) (ops.rsqrt (ops.add var eps))) (w c))
      (ops.silu (ins 1 c)))

/-- `rms_norm(x, w)` is defined for every positive row count and row block, writes `normRow` of
each input row, and changes nothing outside the output. -/
theorem rmsNorm_correct (gemma : Bool) (M H rows : ℕ) (hM : 0 < M) (hrows : 0 < rows) (hH : 0 < H)
    (xBuf wBuf oBuf : BufId) (hx : oBuf ≠ xBuf) (hw : oBuf ≠ wBuf)
    (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', pallasCall ((plainKernel ops eps gemma H).gridSpec M rows ![xBuf] wBuf ![oBuf])
        (plainKernel ops eps gemma H).body junk m0 = some m' ∧
      (∀ (r : Fin M) (c : Fin H), m'.read oBuf (r.val * H + c.val) =
        normRow ops eps gemma (fun c' => m0.read xBuf (r.val * H + c'.val))
          (fun c' => m0.read wBuf c'.val) c) ∧
      Mem.agreeOutside (Mem.bufFootprint oBuf) m0 m' := by
  obtain ⟨m', e, v, fr⟩ := (plainKernel ops eps gemma H).correct M rows hM hrows hH ![xBuf] wBuf
    ![oBuf] (Function.injective_of_subsingleton _) (fun _ _ => by simpa using hx)
    (fun _ => by simpa using hw) junk m0
  refine ⟨m', e, fun r c => by simpa [plainKernel] using v 0 r c, ?_⟩
  refine Mem.agreeOutside_mono ?_ fr
  rintro a ⟨q, hq⟩
  simpa [Mem.bufFootprint] using hq

/-- `rms_norm(x, residual, w)`: both outputs are defined and correct for every positive row count
and row block, and nothing else changes. -/
theorem rmsNormResidual_correct (gemma : Bool) (M H rows : ℕ) (hM : 0 < M) (hrows : 0 < rows) (hH : 0 < H)
    (xBuf rBuf wBuf oBuf roBuf : BufId) (ho : oBuf ≠ roBuf)
    (hdisj : ∀ b ∈ [oBuf, roBuf], b ≠ xBuf ∧ b ≠ rBuf ∧ b ≠ wBuf)
    (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', pallasCall ((residualKernel ops eps gemma H).gridSpec M rows ![xBuf, rBuf] wBuf
        ![oBuf, roBuf]) (residualKernel ops eps gemma H).body junk m0 = some m' ∧
      ∀ (r : Fin M) (c : Fin H),
        let x := fun c' : Fin H => m0.read xBuf (r.val * H + c'.val)
        let res := fun c' : Fin H => m0.read rBuf (r.val * H + c'.val)
        let ro := fun c' => ops.toBf16 (ops.add (x c') (res c'))
        m'.read roBuf (r.val * H + c.val) = ro c ∧
        m'.read oBuf (r.val * H + c.val) =
          normRow ops eps gemma (if gemma then ro else fun c' => ops.add (x c') (res c'))
            (fun c' => m0.read wBuf c'.val) c := by
  have h1 := hdisj oBuf (by simp)
  have h2 := hdisj roBuf (by simp)
  obtain ⟨m', e, v, -⟩ := (residualKernel ops eps gemma H).correct M rows hM hrows hH ![xBuf, rBuf]
    wBuf ![oBuf, roBuf]
    (by intro a b hab; fin_cases a <;> fin_cases b <;> simp_all [eq_comm])
    (by intro q i; fin_cases q <;> fin_cases i <;> simp [h1, h2])
    (by intro q; fin_cases q <;> simp [h1, h2]) junk m0
  refine ⟨m', e, fun r c => ⟨by simpa [residualKernel] using v 1 r c,
    by simpa [residualKernel] using v 0 r c⟩⟩

/-- `gated_rms_norm(x, z, w)` is defined and correct for every positive row count and row block. -/
theorem gatedRmsNorm_correct (M H rows : ℕ) (hM : 0 < M) (hrows : 0 < rows) (hH : 0 < H)
    (xBuf zBuf wBuf oBuf : BufId) (hx : oBuf ≠ xBuf) (hz : oBuf ≠ zBuf) (hw : oBuf ≠ wBuf)
    (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', pallasCall ((gatedKernel ops eps H).gridSpec M rows ![xBuf, zBuf] wBuf ![oBuf])
        (gatedKernel ops eps H).body junk m0 = some m' ∧
      ∀ (r : Fin M) (c : Fin H),
        let s := fun c' : Fin H => m0.read xBuf (r.val * H + c'.val)
        m'.read oBuf (r.val * H + c.val) =
          ops.toBf16 (ops.mul (ops.mul (ops.mul (s c)
            (ops.rsqrt (ops.add (ops.rowMean H fun c' => ops.mul (s c') (s c')) eps)))
            (m0.read wBuf c.val)) (ops.silu (m0.read zBuf (r.val * H + c.val)))) := by
  obtain ⟨m', e, v, -⟩ := (gatedKernel ops eps H).correct M rows hM hrows hH ![xBuf, zBuf] wBuf
    ![oBuf] (Function.injective_of_subsingleton _)
    (by intro q i; fin_cases i <;> simp [hx, hz]) (fun _ => by simpa using hw) junk m0
  exact ⟨m', e, fun r c => by simpa [gatedKernel] using v 0 r c⟩

/-- The PR's contract for `rms_norm`, bit for bit under the `FOps` assumptions: a row's output is
the same whatever the call's row count `M`, its row block size, its junk and the other rows. -/
theorem rmsNorm_batch_invariant (gemma : Bool) {M M' H rows rows' : ℕ} (hrows : 0 < rows)
    (hrows' : 0 < rows') (hH : 0 < H) {xBuf wBuf oBuf xBuf' wBuf' oBuf' : BufId}
    (hx : oBuf ≠ xBuf) (hw : oBuf ≠ wBuf) (hx' : oBuf' ≠ xBuf') (hw' : oBuf' ≠ wBuf')
    {junk junk' : ℕ → ℝ} {m0 m0' m₁ m₂ : Mem}
    (h₁ : pallasCall ((plainKernel ops eps gemma H).gridSpec M rows ![xBuf] wBuf ![oBuf])
      (plainKernel ops eps gemma H).body junk m0 = some m₁)
    (h₂ : pallasCall ((plainKernel ops eps gemma H).gridSpec M' rows' ![xBuf'] wBuf' ![oBuf'])
      (plainKernel ops eps gemma H).body junk' m0' = some m₂)
    (r : Fin M) (r' : Fin M')
    (hrow : ∀ c : Fin H, m0.read xBuf (r.val * H + c.val) = m0'.read xBuf' (r'.val * H + c.val))
    (hwt : ∀ c : Fin H, m0.read wBuf c.val = m0'.read wBuf' c.val) (c : Fin H) :
    m₁.read oBuf (r.val * H + c.val) = m₂.read oBuf' (r'.val * H + c.val) :=
  (plainKernel ops eps gemma H).batch_invariant hrows hrows' hH
    (Function.injective_of_subsingleton _) (fun _ _ => by simpa using hx)
    (fun _ => by simpa using hw) (Function.injective_of_subsingleton _)
    (fun _ _ => by simpa using hx') (fun _ => by simpa using hw') h₁ h₂ r r'
    (fun i c => by fin_cases i; simpa using hrow c) hwt 0 c

/-- With exact arithmetic, the plain kernel computes `TpuKernelLean.Tensor.rmsNorm`. -/
theorem normRow_exact {H : ℕ} (s w : Fin H → ℝ) (c : Fin H) :
    normRow FOps.exact eps false s w c =
      ((TpuKernelLean.Tensor.rmsNorm eps (TpuKernelLean.Tensor.of2 fun (_ : Fin 1) c' => s c') (TpuKernelLean.Tensor.of1 w)).get2 0 c) := by
  simp [normRow, FOps.exact, TpuKernelLean.Tensor.rmsNorm, div_eq_mul_inv, mul_comm, mul_left_comm]

end TpuKernelLean.Examples.TrueOnPolicyNorm
