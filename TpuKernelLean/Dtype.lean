import TpuKernelLean.Memory

/-!
Typed value interpretations and opaque floating-point dtype casts for TPU Pallas kernels.

Memory remains real-valued (`Mem := BufId → ℕ → ℝ`) so existing modules and theorems stay
unchanged. This module provides two layers used by kernels and tests:

1. **Exact signed integers in memory (`int32`, `int8`)**: range predicates (`inInt32Range`,
   `inInt8Range`), value predicates (`IsInt32Val`, `IsInt8Val`), and checked decoders/readers
   (`decodeInt32`, `decodeInt32Nat`, `decodeInt8`, `readInt32`, `writeInt32`, `readInt8`,
   `writeInt8`). Non-integer and out-of-range values return `none` rather than being silently
   floored or clamped.
2. **Floating-point dtypes and opaque casts (`Dtype`, `CastOps`)**: `f32`, `bf16`, `fp8E4M3`,
   `fp8E5M2` with FP8 range limits (`fp8E4M3Max`, `fp8E5M2Max`, `Dtype.fp8Max`) and opaque scalar
   casts (`CastOps`, exact-arithmetic instance `CastOps.exact`), in the style of `FOps`/`MOps`.
-/

namespace TpuKernelLean

/-! ### Floating-point element dtypes and FP8 limits -/

/-- Floating-point TPU Pallas element data types currently used by verified kernels. -/
inductive Dtype where
  /-- IEEE 754 binary32 (`jnp.float32`). -/
  | f32
  /-- Brain floating point 16-bit (`jnp.bfloat16`). -/
  | bf16
  /-- FP8 E4M3FN (`jnp.float8_e4m3fn`, max finite value 448). -/
  | fp8E4M3
  /-- FP8 E5M2 (`jnp.float8_e5m2`, max finite value 57344). -/
  | fp8E5M2
  deriving DecidableEq, Repr

/-- Whether `d` is an 8-bit floating-point dtype (`fp8E4M3` or `fp8E5M2`). -/
def Dtype.isFp8 : Dtype → Bool
  | .fp8E4M3 | .fp8E5M2 => true
  | _ => false

/-- Largest finite value of `jnp.float8_e4m3fn` (`448.0`). -/
def fp8E4M3Max : ℝ := 448

/-- Largest finite value of `jnp.float8_e5m2` (`57344.0`). -/
def fp8E5M2Max : ℝ := 57344

/-- Largest finite value of an FP8 dtype (`0` for non-FP8 dtypes). -/
def Dtype.fp8Max : Dtype → ℝ
  | .fp8E4M3 => fp8E4M3Max
  | .fp8E5M2 => fp8E5M2Max
  | _ => 0

/-- Every FP8 dtype has a strictly positive maximum finite value (`448` or `57344`). -/
theorem Dtype.fp8Max_pos {d : Dtype} (h : d.isFp8 = true) : 0 < d.fp8Max := by
  cases d <;> simp_all [Dtype.isFp8, Dtype.fp8Max, fp8E4M3Max, fp8E5M2Max]

/-! ### Exact signed integers in memory (`int32`, `int8`)

Hardware assumption: an HBM buffer declared with dtype `jnp.int32` (resp. `jnp.int8`) stores
exact two's-complement integers in `[int32Min, int32Max]` (resp. `[int8Min, int8Max]`). Because
memory is modelled as `Mem := BufId → ℕ → ℝ`, valid integer entries are integer-valued reals in
range, and checked readers reject non-integer or out-of-range values with `none` instead of
silently flooring or clamping. -/

/-- Minimum signed 32-bit integer (`-2^31`). -/
def int32Min : ℤ := -(2 ^ 31)

/-- Maximum signed 32-bit integer (`2^31 - 1`). -/
def int32Max : ℤ := 2 ^ 31 - 1

/-- Minimum signed 8-bit integer (`-128`). -/
def int8Min : ℤ := -(2 ^ 7)

/-- Maximum signed 8-bit integer (`127`). -/
def int8Max : ℤ := 2 ^ 7 - 1

/-- Predicate checking that an integer lies in the signed 32-bit range `[-2^31, 2^31 - 1]`. -/
def inInt32Range (z : ℤ) : Prop := int32Min ≤ z ∧ z ≤ int32Max

/-- Predicate checking that an integer lies in the signed 8-bit range `[-128, 127]`. -/
def inInt8Range (z : ℤ) : Prop := int8Min ≤ z ∧ z ≤ int8Max

instance (z : ℤ) : Decidable (inInt32Range z) :=
  inferInstanceAs (Decidable (int32Min ≤ z ∧ z ≤ int32Max))

instance (z : ℤ) : Decidable (inInt8Range z) :=
  inferInstanceAs (Decidable (int8Min ≤ z ∧ z ≤ int8Max))

/-- Predicate holding when a real value stored in `Mem` represents an exact `int32`. -/
def IsInt32Val (x : ℝ) : Prop := ∃ z : ℤ, inInt32Range z ∧ (z : ℝ) = x

/-- Predicate holding when a real value stored in `Mem` represents an exact `int8`. -/
def IsInt8Val (x : ℝ) : Prop := ∃ z : ℤ, inInt8Range z ∧ (z : ℝ) = x

/-- Every in-range integer embeds into `ℝ` as a valid `int32` value. -/
theorem isInt32Val_intCast {z : ℤ} (h : inInt32Range z) : IsInt32Val (z : ℝ) :=
  ⟨z, h, rfl⟩

/-- Every in-range integer embeds into `ℝ` as a valid `int8` value. -/
theorem isInt8Val_intCast {z : ℤ} (h : inInt8Range z) : IsInt8Val (z : ℝ) :=
  ⟨z, h, rfl⟩

noncomputable section

/-- Checked decoder from a real memory value to `int32`: returns `none` for non-integers or
values outside `[-2^31, 2^31 - 1]` (never floors or clamps). -/
def decodeInt32 (x : ℝ) : Option ℤ :=
  open Classical in
  let z := ⌊x⌋
  if (z : ℝ) = x ∧ inInt32Range z then some z else none

/-- Checked decoder from a real memory value to a non-negative `int32` index (`ℕ`): returns `none`
for negative integers (such as `-1`), non-integers, or out-of-range values. -/
def decodeInt32Nat (x : ℝ) : Option ℕ :=
  match decodeInt32 x with
  | some z => if 0 ≤ z then some z.toNat else none
  | none => none

/-- Checked decoder from a real memory value to `int8`: returns `none` for non-integers or values
outside `[-128, 127]` (never floors or clamps). -/
def decodeInt8 (x : ℝ) : Option ℤ :=
  open Classical in
  let z := ⌊x⌋
  if (z : ℝ) = x ∧ inInt8Range z then some z else none

/-- `decodeInt32 x = some z` if and only if `x` is the exact real cast of an in-range `int32` `z`. -/
theorem decodeInt32_eq_some_iff (x : ℝ) (z : ℤ) :
    decodeInt32 x = some z ↔ (z : ℝ) = x ∧ inInt32Range z := by
  unfold decodeInt32
  dsimp only
  split_ifs with h
  · constructor
    · intro heq
      cases heq
      exact h
    · intro ⟨hxz, _⟩
      congr 1
      rw [← hxz, Int.floor_intCast]
  · constructor
    · intro heq
      cases heq
    · intro ⟨hxz, hr⟩
      exfalso
      apply h
      rw [← hxz, Int.floor_intCast]
      exact ⟨rfl, hr⟩

/-- `decodeInt32 x = none` if and only if `x` is not a valid `int32` value. -/
theorem decodeInt32_eq_none_iff (x : ℝ) :
    decodeInt32 x = none ↔ ¬ IsInt32Val x := by
  constructor
  · intro hnone ⟨z, hr, hxz⟩
    have hs := (decodeInt32_eq_some_iff x z).mpr ⟨hxz, hr⟩
    rw [hnone] at hs
    cases hs
  · intro hnot
    cases h : decodeInt32 x with
    | none => rfl
    | some z =>
      have ⟨hxz, hr⟩ := (decodeInt32_eq_some_iff x z).mp h
      exact absurd ⟨z, hr, hxz⟩ hnot

/-- Decoding an in-range integer cast recovers that exact integer. -/
theorem decodeInt32_intCast {z : ℤ} (h : inInt32Range z) :
    decodeInt32 (z : ℝ) = some z :=
  (decodeInt32_eq_some_iff (z : ℝ) z).mpr ⟨rfl, h⟩

/-- Any integer outside `[-2^31, 2^31 - 1]` is rejected by `decodeInt32` (no clamping). -/
theorem decodeInt32_of_not_inRange {z : ℤ} (h : ¬ inInt32Range z) :
    decodeInt32 (z : ℝ) = none := by
  rw [decodeInt32_eq_none_iff]
  rintro ⟨w, hw, hEq⟩
  have hzw : w = z := Int.cast_injective hEq
  subst hzw
  exact h hw

/-- Any non-integer real value is rejected by `decodeInt32` (no flooring). -/
theorem decodeInt32_of_not_int {x : ℝ} (h : ∀ z : ℤ, (z : ℝ) ≠ x) :
    decodeInt32 x = none := by
  rw [decodeInt32_eq_none_iff]
  rintro ⟨z, _, hzx⟩
  exact h z hzx

/-- `decodeInt32Nat x = some n` if and only if `x = n` and `n ≤ int32Max`. -/
theorem decodeInt32Nat_eq_some_iff (x : ℝ) (n : ℕ) :
    decodeInt32Nat x = some n ↔ ((n : ℤ) : ℝ) = x ∧ (n : ℤ) ≤ int32Max := by
  unfold decodeInt32Nat
  constructor
  · intro h
    cases hd : decodeInt32 x with
    | none => rw [hd] at h; cases h
    | some z =>
      rw [hd] at h
      dsimp only at h
      split_ifs at h with hnn
      injection h with htn
      have hzn : z = (n : ℤ) := by omega
      subst hzn
      have ⟨hxz, hr⟩ := (decodeInt32_eq_some_iff x (n : ℤ)).mp hd
      exact ⟨hxz, hr.2⟩
  · rintro ⟨hxn, hle⟩
    have hr : inInt32Range (n : ℤ) := ⟨by dsimp [int32Min]; omega, hle⟩
    rw [(decodeInt32_eq_some_iff x (n : ℤ)).mpr ⟨hxn, hr⟩]
    simp

/-- Every negative real (including `-1`) is rejected by `decodeInt32Nat` (no clamping to `0`). -/
theorem decodeInt32Nat_of_neg {x : ℝ} (hx : x < 0) : decodeInt32Nat x = none := by
  unfold decodeInt32Nat
  cases hd : decodeInt32 x with
  | none => rfl
  | some z =>
    have ⟨hxz, _⟩ := (decodeInt32_eq_some_iff x z).mp hd
    have hzneg : ¬ (0 ≤ z) := by
      intro hnn
      have hzr : (0 : ℝ) ≤ (z : ℝ) := by exact_mod_cast hnn
      linarith
    simp [hzneg]

/-- `decodeInt8 x = some z` if and only if `x` is the exact real cast of an in-range `int8` `z`. -/
theorem decodeInt8_eq_some_iff (x : ℝ) (z : ℤ) :
    decodeInt8 x = some z ↔ (z : ℝ) = x ∧ inInt8Range z := by
  unfold decodeInt8
  dsimp only
  split_ifs with h
  · constructor
    · intro heq
      cases heq
      exact h
    · intro ⟨hxz, _⟩
      congr 1
      rw [← hxz, Int.floor_intCast]
  · constructor
    · intro heq
      cases heq
    · intro ⟨hxz, hr⟩
      exfalso
      apply h
      rw [← hxz, Int.floor_intCast]
      exact ⟨rfl, hr⟩

/-- `decodeInt8 x = none` if and only if `x` is not a valid `int8` value. -/
theorem decodeInt8_eq_none_iff (x : ℝ) :
    decodeInt8 x = none ↔ ¬ IsInt8Val x := by
  constructor
  · intro hnone ⟨z, hr, hxz⟩
    have hs := (decodeInt8_eq_some_iff x z).mpr ⟨hxz, hr⟩
    rw [hnone] at hs
    cases hs
  · intro hnot
    cases h : decodeInt8 x with
    | none => rfl
    | some z =>
      have ⟨hxz, hr⟩ := (decodeInt8_eq_some_iff x z).mp h
      exact absurd ⟨z, hr, hxz⟩ hnot

/-- Decoding an in-range `int8` cast recovers that exact integer. -/
theorem decodeInt8_intCast {z : ℤ} (h : inInt8Range z) :
    decodeInt8 (z : ℝ) = some z :=
  (decodeInt8_eq_some_iff (z : ℝ) z).mpr ⟨rfl, h⟩

/-- Any integer outside `[-128, 127]` is rejected by `decodeInt8` (no clamping). -/
theorem decodeInt8_of_not_inRange {z : ℤ} (h : ¬ inInt8Range z) :
    decodeInt8 (z : ℝ) = none := by
  rw [decodeInt8_eq_none_iff]
  rintro ⟨w, hw, hEq⟩
  have hzw : w = z := Int.cast_injective hEq
  subst hzw
  exact h hw

/-- Any non-integer real value is rejected by `decodeInt8` (no flooring). -/
theorem decodeInt8_of_not_int {x : ℝ} (h : ∀ z : ℤ, (z : ℝ) ≠ x) :
    decodeInt8 x = none := by
  rw [decodeInt8_eq_none_iff]
  rintro ⟨z, _, hzx⟩
  exact h z hzx

/-- Read a checked `int32` value from memory at `(buf, off)`. -/
def readInt32 (m : Mem) (buf : BufId) (off : ℕ) : Option ℤ :=
  decodeInt32 (m.read buf off)

/-- Write an integer value into memory at `(buf, off)`. -/
def writeInt32 (m : Mem) (buf : BufId) (off : ℕ) (z : ℤ) : Mem :=
  m.write buf off (z : ℝ)

/-- Writing an in-range `int32` and reading it back recovers the exact integer. -/
theorem readInt32_writeInt32_same (m : Mem) (b : BufId) (o : ℕ) {z : ℤ} (h : inInt32Range z) :
    readInt32 (writeInt32 m b o z) b o = some z := by
  simp only [readInt32, writeInt32, Mem.read_write_same]
  exact decodeInt32_intCast h

/-- Writing an `int32` leaves all other addresses unchanged. -/
theorem readInt32_writeInt32_ne (m : Mem) {b b' : BufId} {o o' : ℕ} (z : ℤ)
    (hne : (b', o') ≠ (b, o)) :
    readInt32 (writeInt32 m b o z) b' o' = readInt32 m b' o' := by
  simp only [readInt32, writeInt32, Mem.read_write_ne m _ hne]

/-- Read a checked `int8` value from memory at `(buf, off)`. -/
def readInt8 (m : Mem) (buf : BufId) (off : ℕ) : Option ℤ :=
  decodeInt8 (m.read buf off)

/-- Write an `int8` integer value into memory at `(buf, off)`. -/
def writeInt8 (m : Mem) (buf : BufId) (off : ℕ) (z : ℤ) : Mem :=
  m.write buf off (z : ℝ)

/-- Writing an in-range `int8` and reading it back recovers the exact integer. -/
theorem readInt8_writeInt8_same (m : Mem) (b : BufId) (o : ℕ) {z : ℤ} (h : inInt8Range z) :
    readInt8 (writeInt8 m b o z) b o = some z := by
  simp only [readInt8, writeInt8, Mem.read_write_same]
  exact decodeInt8_intCast h

/-- Writing an `int8` leaves all other addresses unchanged. -/
theorem readInt8_writeInt8_ne (m : Mem) {b b' : BufId} {o o' : ℕ} (z : ℤ)
    (hne : (b', o') ≠ (b, o)) :
    readInt8 (writeInt8 m b o z) b' o' = readInt8 m b' o' := by
  simp only [readInt8, writeInt8, Mem.read_write_ne m _ hne]

end

/-! ### Opaque floating-point dtype casts (`CastOps`)

Hardware assumption: each scalar floating-point cast on the TPU VPU is a deterministic function
of its scalar argument, independent of other lanes. Theorems proved for every `CastOps` hold bit
for bit on hardware under this assumption. -/

/-- Opaque scalar floating-point dtype casts used by TPU Pallas kernels. -/
structure CastOps where
  /-- Cast to fp32 (`astype(jnp.float32)`). Exact for bf16 and fp8 inputs on hardware. -/
  toF32 : ℝ → ℝ
  /-- Cast to bfloat16 (`astype(jnp.bfloat16)`). -/
  toBf16 : ℝ → ℝ
  /-- Unsaturated cast to fp8 e4m3fn (`astype(jnp.float8_e4m3fn)`). -/
  toFp8E4M3 : ℝ → ℝ
  /-- Unsaturated cast to fp8 e5m2 (`astype(jnp.float8_e5m2)`). -/
  toFp8E5M2 : ℝ → ℝ

/-- Dispatch a floating-point dtype cast `astype(d)` through `CastOps`. -/
def CastOps.cast (ops : CastOps) : Dtype → ℝ → ℝ
  | .f32 => ops.toF32
  | .bf16 => ops.toBf16
  | .fp8E4M3 => ops.toFp8E4M3
  | .fp8E5M2 => ops.toFp8E5M2

/-- Exact-arithmetic `CastOps` instance: every floating-point cast is the identity (as in
`FOps.exact` and `MOps.exact`). Callers relying on an unsaturated FP8 cast must still establish
that inputs lie in `[-d.fp8Max, d.fp8Max]` to avoid hardware `NaN`/`±inf` overflow. -/
def CastOps.exact : CastOps where
  toF32 := id
  toBf16 := id
  toFp8E4M3 := id
  toFp8E5M2 := id

/-- Under exact arithmetic, every floating-point cast is the identity. -/
theorem CastOps.exact_cast (d : Dtype) (x : ℝ) :
    CastOps.exact.cast d x = x := by
  cases d <;> rfl

end TpuKernelLean
