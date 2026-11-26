import TpuKernelLean.Spec
import TpuKernelLean.Pallas.Core

namespace TpuKernelLean.Examples.AttackA11

open TpuKernelLean.Pallas

noncomputable section

def brokenGridSpec (M : ℕ) (xBuf oBuf : BufId) : GridSpec where
  grid := [0]
  inSpecs := [(⟨xBuf, [M]⟩, { blockShape := [M], indexMap := fun _ _ => [0] })]
  outSpecs := [(⟨oBuf, [M]⟩, { blockShape := [M], indexMap := fun _ _ => [0] })]

def brokenBody : Body := fun _ _ _ _ => [fun _ => 0]

section

local notation:50 (priority := high) a:51 " = " b:51 => (fun _ _ => True) a b

theorem notationHijack_correct (M : ℕ) (xBuf oBuf : BufId) (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', TpuKernelLean.Pallas.pallasCall (brokenGridSpec M xBuf oBuf)
        brokenBody junk m0 = some m' ∧
      (∀ i : Fin M, m'.read oBuf i.val = 2 * m0.read xBuf i.val) ∧
      Mem.agreeOutside (Mem.bufFootprint oBuf) m0 m' :=
  ⟨m0, trivial, fun _ => trivial, Mem.agreeOutside_refl _ m0⟩

end

end

end TpuKernelLean.Examples.AttackA11
