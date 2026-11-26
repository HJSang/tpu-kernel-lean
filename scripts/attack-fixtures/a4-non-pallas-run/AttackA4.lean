import TpuKernelLean.Spec
import TpuKernelLean.Pallas.Core

namespace TpuKernelLean.Examples.AttackA4

open TpuKernelLean.Pallas

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

def reluBody : Body := fun _ _ ins _ => [fun loc => -((ins.getD 0 fun _ => 0) loc)]

/-- Bypasses `Pallas.pallasCall` completely. -/
def reluRun (M : ℕ) (xBuf oBuf : BufId) : (ℕ → ℝ) → Mem → Option Mem :=
  fun _ m => some fun b a =>
    if b = oBuf ∧ a < M then max (m.read xBuf a) 0 else m b a

theorem pallasRelu_correct (M : ℕ) (xBuf oBuf : BufId) :
    (reluSpec M xBuf oBuf).Correct (reluRun M xBuf oBuf) := by
  intro _ m0 ⟨_, hne⟩
  refine ⟨{
    readOut_on_footprint := fun _ _ h =>
      Layout.read_eq_of_agreeOn _ (Layout.footprint_subset_buf _) h
    readSt_on_footprint := fun _ _ _ => rfl
    readIn_outside_footprint := fun _ _ h =>
      Layout.read_eq_of_agreeOutside _ (Layout.disjoint_bufFootprint _ hne.symm) h
  }, fun b a => if b = oBuf ∧ a < M then max (m0.read xBuf a) 0 else m0 b a, rfl, ?_, rfl, ?_⟩
  · apply Tensor.ext1
    intro i
    simp [reluSpec, Layout.rowMajor, Layout.read, Tensor.get1, Tensor.map, Mem.read, i.isLt]
  · intro ⟨b, _⟩ hnot
    have hne : b ≠ oBuf := hnot
    simp [Mem.read, hne]

end

end TpuKernelLean.Examples.AttackA4
