import TpuKernelLean.Pallas.Core

namespace TpuKernelLean.Examples.AttackA7

open TpuKernelLean.Pallas

/-- Clamped `BlockSpec.indexMap` using truncating `ℕ` subtraction `(i - 1 : ℕ)`. -/
def causalShiftGridSpec (M P : ℕ) (xBuf oBuf : BufId) : GridSpec where
  grid := [M]
  inSpecs := [(⟨xBuf, [M, P]⟩,
    { blockShape := [1, P], indexMap := fun g _ => [((g.getD 0 0 - 1 : ℕ) : ℤ), 0] })]
  outSpecs := [(⟨oBuf, [M, P]⟩,
    { blockShape := [1, P], indexMap := fun g _ => [(g.getD 0 0 : ℤ), 0] })]

end TpuKernelLean.Examples.AttackA7
