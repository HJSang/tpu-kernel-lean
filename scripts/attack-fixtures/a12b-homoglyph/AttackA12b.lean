import TpuKernelLean.Spec
import TpuKernelLean.Pallas.Core

namespace TpuKernelLean.Examples.AttackA12b

open TpuKernelLean.Pallas

/-- Greek small letter omicron `ο` (U+03BF) instead of Latin `o`. -/
def _root_.TpuKernelLean.KernelSpec.Cοrrect {I S O : Type*}
    (_spec : KernelSpec I S O) (_run : (ℕ → ℝ) → Mem → Option Mem) : Prop :=
  True

noncomputable section

def reluSpec (M : ℕ) (xBuf oBuf : BufId) :
    KernelSpec (Tensor ℝ [M]) Unit (Tensor ℝ [M]) where
  readIn := (Layout.rowMajor xBuf 0 [M]).read
  readSt := fun _ => ()
  readOut := (Layout.rowMajor oBuf 0 [M]).read
  footprint := Mem.bufFootprint oBuf
  math := fun _ x => Tensor.map (fun v => max v 0) x
  mutate := fun _ _ => ()
  requires := fun _ _ => 0 < M ∧ oBuf ≠ xBuf

def reluGridSpec (M : ℕ) (xBuf oBuf : BufId) : GridSpec where
  grid := [1]
  inSpecs := [(⟨xBuf, [M]⟩, { blockShape := [M], indexMap := fun _ _ => [0] })]
  outSpecs := [(⟨oBuf, [M]⟩, { blockShape := [M], indexMap := fun _ _ => [0] })]

def reluBody : Body := fun _ _ _ _ => [fun _ => 0]

theorem homoglyphRelu_correct (M : ℕ) (xBuf oBuf : BufId) :
    (reluSpec M xBuf oBuf).Cοrrect
      (Pallas.pallasCall (reluGridSpec M xBuf oBuf) reluBody) :=
  trivial

end

end TpuKernelLean.Examples.AttackA12b
