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

/-- Five-part kernel specification over input `I`, in-place state `S`, and output `O` records. -/
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
  /-- Output closeness relation (defaults to exact equality). -/
  closeOut : O → O → Prop := Eq
  /-- Post-state closeness relation (defaults to exact equality). -/
  closeSt : S → S → Prop := Eq
  /-- Precondition on pre-state and inputs. -/
  requires : S → I → Prop

/-- A kernel runner satisfies `spec` if on every valid memory and junk oracle the run is defined
(`run` returns `none` for undefined behaviour) and meets `math`, `mutate`, and the frame condition. -/
def KernelSpec.Correct {I S O : Type*} (spec : KernelSpec I S O)
    (run : (ℕ → ℝ) → Mem → Option Mem) : Prop :=
  ∀ (junk : ℕ → ℝ) (m : Mem),
    spec.requires (spec.readSt m) (spec.readIn m) →
    ∃ m', run junk m = some m' ∧
      spec.closeOut (spec.readOut m') (spec.math (spec.readSt m) (spec.readIn m)) ∧
      spec.closeSt (spec.readSt m') (spec.mutate (spec.readSt m) (spec.readIn m)) ∧
      Mem.agreeOutside spec.footprint m m'

end TpuKernelLean
