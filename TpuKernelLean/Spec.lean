import TpuKernelLean.Layout

namespace TpuKernelLean

/-- Pointwise absolute tolerance predicate `∀ i, |a i - b i| ≤ ε` on real tensors. -/
def withinTol {s : List ℕ} (ε : ℝ) (a b : Tensor ℝ s) : Prop :=
  ∀ i : Idx s, |a i - b i| ≤ ε

/-- Zero tolerance is equivalent to exact tensor equality. -/
theorem withinTol_zero_iff {s : List ℕ} {a b : Tensor ℝ s} :
    withinTol 0 a b ↔ a = b := by
  constructor
  · intro h
    funext i
    have hi := h i
    have habs : |a i - b i| = 0 := le_antisymm hi (abs_nonneg _)
    exact sub_eq_zero.mp (abs_eq_zero.mp habs)
  · rintro rfl i
    simp

/-- Exact equality implies `withinTol ε` for any nonnegative tolerance `ε`. -/
theorem withinTol_of_eq {s : List ℕ} {ε : ℝ} {a b : Tensor ℝ s}
    (hab : a = b) (hε : 0 ≤ ε) :
    withinTol ε a b := by
  subst hab
  intro i
  simpa using hε

/-- Four-part kernel specification over input `I`, in-place state `S`, and output `O` records:
`math`, `mutate`, layout (`readIn`, `readSt`, `readOut`, `footprint`), and `requires`. -/
structure KernelSpec (I S O : Type*) where
  /-- Read input tensors from pre-execution memory. -/
  readIn : Mem → I
  /-- Read in-place state tensors from memory. -/
  readSt : Mem → S
  /-- Read output tensors from post-execution memory. -/
  readOut : Mem → O
  /-- Writable footprint (outputs ∪ state); addresses outside this set must remain unchanged. -/
  footprint : Set Addr
  /-- Mathematical specification of outputs from pre-state and inputs. -/
  math : S → I → O
  /-- In-place state mutation specification from pre-state and inputs. -/
  mutate : S → I → S
  /-- Precondition on pre-state and inputs. -/
  requires : S → I → Prop

/-- Layout well-formedness of a `KernelSpec` with respect to its `footprint`:
`readOut` and `readSt` depend only on addresses inside `footprint`, while `readIn` depends only on
addresses outside `footprint`. -/
structure KernelSpec.WellFormed {I S O : Type*} (spec : KernelSpec I S O) : Prop where
  /-- Outputs depend only on memory inside `footprint`. -/
  readOut_on_footprint : ∀ m₁ m₂ : Mem, Mem.agreeOn spec.footprint m₁ m₂ →
    spec.readOut m₁ = spec.readOut m₂
  /-- In-place state depends only on memory inside `footprint`. -/
  readSt_on_footprint : ∀ m₁ m₂ : Mem, Mem.agreeOn spec.footprint m₁ m₂ →
    spec.readSt m₁ = spec.readSt m₂
  /-- Inputs do not depend on memory inside `footprint`. -/
  readIn_outside_footprint : ∀ m₁ m₂ : Mem, Mem.agreeOutside spec.footprint m₁ m₂ →
    spec.readIn m₁ = spec.readIn m₂

/-- A kernel runner satisfies `spec` if on every valid memory and junk oracle `spec` has a
well-formed layout, the run is defined (`run` returns `none` for undefined behaviour), outputs and
post-state equal `math` and `mutate`, and every address outside `footprint` is unchanged. -/
def KernelSpec.Correct {I S O : Type*} (spec : KernelSpec I S O)
    (run : (ℕ → ℝ) → Mem → Option Mem) : Prop :=
  ∀ (junk : ℕ → ℝ) (m : Mem),
    spec.requires (spec.readSt m) (spec.readIn m) →
    spec.WellFormed ∧
    ∃ m', run junk m = some m' ∧
      spec.readOut m' = spec.math (spec.readSt m) (spec.readIn m) ∧
      spec.readSt m' = spec.mutate (spec.readSt m) (spec.readIn m) ∧
      Mem.agreeOutside spec.footprint m m'

end TpuKernelLean
