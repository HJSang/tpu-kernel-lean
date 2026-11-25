import TpuKernelLean.Dtype
import TpuKernelLean.Examples.Fp8LinearSmallM

/-!
Tests for `TpuKernelLean.Dtype` and `TpuKernelLean.Examples.Fp8LinearSmallM`:
1. Exact `int32` / `int8` decoding rejects non-integer, negative-as-index, and out-of-range values
   without silent flooring or clamping, and `writeInt32`/`readInt32` and `writeInt8`/`readInt8`
   preserve boundaries and frame conditions.
2. `Dtype` FP8 limits and `Fp8LinearSmallM` invariants: `Supports` rejects `n = 0`, exact dynamic
   scaling tightly bounds `|x_p / x_scale| ≤ fp8Max` (and floors zero rows at `1e-12`),
   `FP8LinearOps` does not assume cross-`tokens` batch invariance, and at a fixed compiled shape
   mutating another row of `x` does not affect row 0's output.
-/

namespace TpuKernelLean.Tests.TypedValues

open TpuKernelLean.Examples.Fp8LinearSmallM

/-! ### 1. Exact `int32` / `int8` decoding and memory round-trip tests -/

/-- A non-integer value (`3 / 2`) is rejected by `decodeInt32` rather than silently floored to `1`. -/
theorem test_decodeInt32_rejects_non_int : decodeInt32 (3 / 2 : ℝ) = none := by
  apply decodeInt32_of_not_int
  intro z hz
  have h1 : ((2 * z : ℤ) : ℝ) = ((3 : ℤ) : ℝ) := by push_cast; linarith
  have h2 : 2 * z = 3 := Int.cast_injective h1
  omega

/-- An out-of-range positive value (`2^31 = 2147483648`) is rejected by `decodeInt32` (no clamping). -/
theorem test_decodeInt32_rejects_overflow : decodeInt32 ((2 ^ 31 : ℤ) : ℝ) = none :=
  decodeInt32_of_not_inRange (by decide)

/-- An out-of-range negative value (`-2^31 - 1`) is rejected by `decodeInt32` (no clamping). -/
theorem test_decodeInt32_rejects_underflow : decodeInt32 ((-(2 ^ 31) - 1 : ℤ) : ℝ) = none :=
  decodeInt32_of_not_inRange (by decide)

/-- `-1` decodes to `some (-1)` as an `int32`, and is rejected by `decodeInt32Nat` (not clamped to `0`). -/
theorem test_decodeInt32_neg_one :
    decodeInt32 ((-1 : ℤ) : ℝ) = some (-1) ∧ decodeInt32Nat ((-1 : ℤ) : ℝ) = none :=
  ⟨decodeInt32_intCast (by decide), decodeInt32Nat_of_neg (by norm_num)⟩

/-- Boundary `int32` values `int32Min` and `int32Max` round-trip through `writeInt32` / `readInt32`. -/
theorem test_int32_boundaries (m : Mem) :
    readInt32 (writeInt32 m 0 0 int32Min) 0 0 = some int32Min ∧
    readInt32 (writeInt32 m 0 0 int32Max) 0 0 = some int32Max :=
  ⟨readInt32_writeInt32_same m 0 0 (by decide), readInt32_writeInt32_same m 0 0 (by decide)⟩

/-- Writing an `int32` at `(0, 0)` leaves `(0, 1)` unchanged. -/
theorem test_readInt32_writeInt32_frame (m : Mem) :
    readInt32 (writeInt32 m 0 0 42) 0 1 = readInt32 m 0 1 :=
  readInt32_writeInt32_ne m 42 (by decide)

/-- Out-of-range `int8` values (`128` and `-129`) are rejected by `decodeInt8` (no clamping). -/
theorem test_decodeInt8_rejects_out_of_range :
    decodeInt8 ((128 : ℤ) : ℝ) = none ∧ decodeInt8 ((-129 : ℤ) : ℝ) = none :=
  ⟨decodeInt8_of_not_inRange (by decide), decodeInt8_of_not_inRange (by decide)⟩

/-- A non-integer value (`3 / 2`) is rejected by `decodeInt8` rather than silently floored. -/
theorem test_decodeInt8_rejects_non_int : decodeInt8 (3 / 2 : ℝ) = none := by
  apply decodeInt8_of_not_int
  intro z hz
  have h1 : ((2 * z : ℤ) : ℝ) = ((3 : ℤ) : ℝ) := by push_cast; linarith
  have h2 : 2 * z = 3 := Int.cast_injective h1
  omega

/-- Boundary `int8` values `int8Min` and `int8Max` round-trip through `writeInt8` / `readInt8`. -/
theorem test_int8_boundaries (m : Mem) :
    readInt8 (writeInt8 m 0 0 int8Min) 0 0 = some int8Min ∧
    readInt8 (writeInt8 m 0 0 int8Max) 0 0 = some int8Max :=
  ⟨readInt8_writeInt8_same m 0 0 (by decide), readInt8_writeInt8_same m 0 0 (by decide)⟩

/-- Writing an `int8` at `(0, 0)` leaves `(0, 1)` unchanged. -/
theorem test_readInt8_writeInt8_frame (m : Mem) :
    readInt8 (writeInt8 m 0 0 7) 0 1 = readInt8 m 0 1 :=
  readInt8_writeInt8_ne m 7 (by decide)

/-! ### 2. `Dtype` FP8 limits and `Fp8LinearSmallM` tests -/

/-- `Supports` rejects `n = 0` even though `0 % 128 = 0`. -/
theorem test_fp8_supports_rejects_n_zero :
    ¬ Supports 1 128 0 128 := by
  intro h
  exact Nat.lt_irrefl 0 h.n_pos

/-- `Dtype.isFp8` accepts only FP8 dtypes and `Dtype.fp8Max` matches their hardware finite limits. -/
theorem test_fp8_dtype_limits :
    Dtype.f32.isFp8 = false ∧
    Dtype.bf16.isFp8 = false ∧
    Dtype.fp8E4M3.isFp8 = true ∧
    Dtype.fp8E5M2.isFp8 = true ∧
    Dtype.fp8E4M3.fp8Max = 448 ∧
    Dtype.fp8E5M2.fp8Max = 57344 :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- On a concrete non-zero row `[0, -1000, 500]` with `fp8E4M3`, exact scaling hits `|x_1 / scale| = 448`
tightly and keeps every scaled element within `[-448, 448]`. -/
theorem test_fp8LinearScaled_e4m3_tight :
    let xr : Fin 3 → ℝ := fun i => match i.val with | 0 => 0 | 1 => -1000 | _ => 500
    |fp8LinearScaled FP8LinearOps.exact .fp8E4M3 1 3 128 xr 1| = 448 ∧
    (∀ p : Fin 3, |fp8LinearScaled FP8LinearOps.exact .fp8E4M3 1 3 128 xr p| ≤ 448) := by
  dsimp only
  refine ⟨?_, fun p => fp8LinearScaled_abs_le_exact rfl 1 3 128 _ p⟩
  simp only [fp8LinearScaled, fp8LinearScale, FP8LinearOps.exact, exactRowMax, scaleFloor,
    CastOps.exact, id_eq, Dtype.fp8Max, fp8E4M3Max, Tensor.vecMax, Fin.val_zero, Fin.val_one,
    Fin.val_succ]
  norm_num

/-- On an all-zero row, `fp8LinearScale` hits the `1e-12` `scaleFloor` and every scaled element is `0`. -/
theorem test_fp8LinearScaled_zero_row :
    let xr : Fin 2 → ℝ := fun _ => 0
    fp8LinearScale FP8LinearOps.exact .fp8E4M3 1 2 128 xr = scaleFloor ∧
    (∀ p : Fin 2, fp8LinearScaled FP8LinearOps.exact .fp8E4M3 1 2 128 xr p = 0) := by
  dsimp only
  have hscale :
      fp8LinearScale FP8LinearOps.exact .fp8E4M3 1 2 128 (fun _ => 0) = scaleFloor := by
    simp only [fp8LinearScale, FP8LinearOps.exact, exactRowMax, scaleFloor,
      CastOps.exact, id_eq, Dtype.fp8Max, fp8E4M3Max, Tensor.vecMax]
    norm_num
  refine ⟨hscale, fun p => ?_⟩
  simp [fp8LinearScaled, FP8LinearOps.exact, CastOps.exact]

/-- Witness `FP8LinearOps` whose `dot` depends on `tokens`, modeling Mosaic choosing a different
untiled full-`K` reduction order for different compiled batch sizes. -/
private noncomputable def tokensDependentOps : FP8LinearOps :=
  { FP8LinearOps.exact with
    dot := fun tokens _ _ a b => (tokens : ℝ) + ∑ p, a p * b p }

/-- `FP8LinearOps` does not force cross-`tokens` invariance: the same row, column, and weight
scale can produce different outputs at `tokens = 1` versus `tokens = 2`. -/
theorem test_fp8LinearOps_allows_tokens_dependence :
    fp8LinearElem tokensDependentOps .fp8E4M3 .bf16 1 1 128 (fun _ => 0) (fun _ => 0) 1 ≠
      fp8LinearElem tokensDependentOps .fp8E4M3 .bf16 2 1 128 (fun _ => 0) (fun _ => 0) 1 := by
  simp only [fp8LinearElem, fp8LinearScale, tokensDependentOps,
    FP8LinearOps.exact, exactRowMax, scaleFloor, CastOps.exact, CastOps.cast, id_eq,
    Dtype.fp8Max, fp8E4M3Max, Tensor.vecMax, mul_zero, Finset.sum_const_zero]
  norm_num

/-- End-to-end fixed-shape row independence test: at compiled shape `(2, 128, 128, 128)`, mutating
any entry of row `1` of `x` (and changing `junk`) leaves every column of row `0` bit-for-bit
identical for every `FP8LinearOps`. -/
theorem test_fp8Linear_fixedShape_other_row_mutation
    (ops : FP8LinearOps) (junk junk' : ℕ → ℝ) (m0 : Mem) (p1 : Fin 128) (v : ℝ) :
    let m0' := m0.write 0 (128 + p1.val) v
    ∃ m₁ m₂,
      fp8LinearRun ops .fp8E4M3 .bf16 2 128 128 128 0 1 2 3 junk m0 = some m₁ ∧
      fp8LinearRun ops .fp8E4M3 .bf16 2 128 128 128 0 1 2 3 junk' m0' = some m₂ ∧
      ∀ c : Fin 128, m₁.read 3 c.val = m₂.read 3 c.val := by
  dsimp only
  have hsup : Supports 2 128 128 128 :=
    ⟨by decide, by decide, by decide, by decide, by decide,
      by simp [tileNCandidates], by decide, by decide⟩
  obtain ⟨m₁, hm₁, -⟩ :=
    fp8Linear_correct ops .fp8E4M3 .bf16 rfl 2 128 128 128 hsup
      0 1 2 3 (by decide) (by decide) (by decide) junk m0
  obtain ⟨m₂, hm₂, -⟩ :=
    fp8Linear_correct ops .fp8E4M3 .bf16 rfl 2 128 128 128 hsup
      0 1 2 3 (by decide) (by decide) (by decide) junk' (m0.write 0 (128 + p1.val) v)
  refine ⟨m₁, m₂, hm₁, hm₂, fun c => ?_⟩
  have hrow := fp8Linear_fixedShape_row_independent ops .fp8E4M3 .bf16
    rfl hsup (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    hm₁ hm₂ ⟨0, by decide⟩ ⟨0, by decide⟩
    (fun p => by
      have hne : (0, 0 * 128 + p.val) ≠ (0, 128 + p1.val) := by
        intro h
        have := (Prod.mk.inj h).2
        omega
      exact (Mem.read_write_ne m0 v hne).symm)
    (fun _ _ => (Mem.read_write_buf_ne m0 _ _ v (by decide)).symm)
    (fun _ => (Mem.read_write_buf_ne m0 _ _ v (by decide)).symm)
    c
  simpa using hrow

end TpuKernelLean.Tests.TypedValues
