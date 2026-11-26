import TpuKernelLean.Pallas.ReductionKernel

namespace TpuKernelLean.Examples.AttackA7Reduction

open TpuKernelLean.Pallas

/-- Clamped `ReductionKernel.outIndexMap` using truncating `ℕ` subtraction. -/
noncomputable def badReductionKernel (M bm nk : ℕ) (xBuf oBuf : BufId) : ReductionKernel 1 where
  outerGrid := [M]
  nk := nk
  inSpecs := [(⟨xBuf, [M]⟩, { blockShape := [bm], indexMap := fun g _ => [(g.getD 0 0 : ℤ)] })]
  outArr := ⟨oBuf, [M]⟩
  outBlockShape := [bm]
  outIndexMap := fun og _ => [((og.getD 0 0 - 1 : ℕ) : ℤ)]
  init := fun _ => 0
  step := fun _ _ _ _ _ acc _ => acc 0
  finalize := fun _ _ _ _ acc => acc 0

end TpuKernelLean.Examples.AttackA7Reduction
