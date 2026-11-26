import TpuKernelLean.Spec
import TpuKernelLean.Pallas.Core

namespace TpuKernelLean.Examples.AttackA5

noncomputable section

def vecSpec (M : ℕ) (aBuf oBuf : BufId) :
    KernelSpec (Tensor ℝ [M]) Unit (Tensor ℝ [M]) where
  readIn := (Layout.rowMajor aBuf 0 [M]).read
  readSt := fun _ => ()
  readOut := (Layout.rowMajor oBuf 0 [M]).read
  footprint := Mem.bufFootprint oBuf
  math := fun _ a => a
  mutate := fun _ _ => ()
  requires := fun _ _ => 0 < M ∧ oBuf ≠ aBuf

def vecGridSpec (M : ℕ) (aBuf oBuf : BufId) : TpuKernelLean.Pallas.GridSpec where
  grid := [1]
  inSpecs := [(⟨aBuf, [M]⟩, { blockShape := [M], indexMap := fun _ _ => [0] })]
  outSpecs := [(⟨oBuf, [M]⟩, { blockShape := [M], indexMap := fun _ _ => [0] })]

def vecBody : TpuKernelLean.Pallas.Body := fun _ _ _ _ => [fun _ => 0]

namespace Pallas

/-- Shadows `TpuKernelLean.Pallas.pallasCall`. -/
private def pallasCall (gs : TpuKernelLean.Pallas.GridSpec)
    (_body : TpuKernelLean.Pallas.Body) (_junk : ℕ → ℝ) (m0 : Mem) : Option Mem :=
  match gs.inSpecs, gs.outSpecs with
  | (aArr, _) :: _, (oArr, _) :: _ =>
    some fun b a => if b = oArr.buf then m0.read aArr.buf a else m0 b a
  | _, _ => none

end Pallas

theorem vectorCopy_correct (M : ℕ) (aBuf oBuf : BufId) :
    (vecSpec M aBuf oBuf).Correct
      (Pallas.pallasCall (vecGridSpec M aBuf oBuf) vecBody) := by
  intro _ m0 ⟨_, hne⟩
  refine ⟨{
    readOut_on_footprint := fun _ _ h =>
      Layout.read_eq_of_agreeOn _ (Layout.footprint_subset_buf _) h
    readSt_on_footprint := fun _ _ _ => rfl
    readIn_outside_footprint := fun _ _ h =>
      Layout.read_eq_of_agreeOutside _ (Layout.disjoint_bufFootprint _ hne.symm) h
  }, fun b a => if b = oBuf then m0.read aBuf a else m0 b a, rfl, ?_, rfl, ?_⟩
  · apply Tensor.ext1
    intro i
    simp [vecSpec, Layout.rowMajor, Layout.read, Tensor.get1, Mem.read]
  · intro ⟨b, _⟩ hnot
    have hne : b ≠ oBuf := hnot
    simp [Mem.read, hne]

end

end TpuKernelLean.Examples.AttackA5
