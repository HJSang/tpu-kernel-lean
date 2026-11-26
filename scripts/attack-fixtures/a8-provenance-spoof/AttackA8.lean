import TpuKernelLean.Spec
import TpuKernelLean.Pallas.Core

namespace TpuKernelLean.Examples.AttackA8

open TpuKernelLean.Pallas

noncomputable section

def spoofSpec (oBuf : BufId) : KernelSpec Unit Unit (Tensor ℝ [1]) where
  readIn := fun _ => ()
  readSt := fun _ => ()
  readOut := (Layout.rowMajor oBuf 0 [1]).read
  footprint := Mem.bufFootprint oBuf
  math := fun _ _ => Tensor.of1 fun _ => 0
  mutate := fun _ _ => ()
  requires := fun _ _ => True

def spoofGridSpec (oBuf : BufId) : GridSpec where
  grid := [1]
  inSpecs := []
  outSpecs := [(⟨oBuf, [1]⟩, { blockShape := [1], indexMap := fun _ _ => [0] })]

def spoofBody : Body := fun _ _ _ _ => [fun _ => 0]

private theorem spoof_wellDefined (oBuf : BufId) (m0 : Mem) :
    (spoofGridSpec oBuf).WellDefined spoofBody m0 where
  grid_pos := by intro d hd; simp [spoofGridSpec] at hd; subst hd; omega
  block_specs_ok := by intro p hp; simp [spoofGridSpec] at hp; subst hp; simp
  blocks_ok := by intro _ _ p hp; simp [spoofGridSpec] at hp; subst hp; simp [GridSpec.idxAt, blockOk]
  outputs_consecutive := by
    intro p hp; simp [spoofGridSpec] at hp; subst hp
    simp [spoofGridSpec, gridPoints_length, noRevisit]
  body_arity := fun _ _ _ _ => rfl
  no_stale_read := by intro _ _ _ _ i hi; simp [spoofGridSpec] at hi
  aliases_ok := GridSpec.AliasesValid.of_empty rfl (by simp [spoofGridSpec])
    (by simp [spoofGridSpec]) (by simp [spoofGridSpec])
  dim_semantics_len := rfl
  dim_semantics_prefix := rfl
  parallel_outputs_disjoint := by
    intro pg₁ h₁ pg₂ h₂ hne
    simp [spoofGridSpec, GridSpec.parGrid, GridSpec.numParallel,
      DimSemantics.isParallel, gridPoints] at h₁ h₂
    exact absurd (h₁.trans h₂.symm) hne
  parallel_read_write_disjoint := by
    intro pg₁ h₁ pg₂ h₂ hne
    simp [spoofGridSpec, GridSpec.parGrid, GridSpec.numParallel,
      DimSemantics.isParallel, gridPoints] at h₁ h₂
    exact absurd (h₁.trans h₂.symm) hne
  parallel_scratch_reset := fun h => absurd rfl h

theorem provenanceSpoof_correct (oBuf : BufId) :
    (spoofSpec oBuf).Correct (pallasCall (spoofGridSpec oBuf) spoofBody) := by
  intro junk m0 _
  refine ⟨{
    readOut_on_footprint := fun _ _ h =>
      Layout.read_eq_of_agreeOn _ (Layout.footprint_subset_buf _) h
    readSt_on_footprint := fun _ _ _ => rfl
    readIn_outside_footprint := fun _ _ _ => rfl
  }, pallasExec (spoofGridSpec oBuf) spoofBody junk m0,
    pallasCall_of_wellDefined junk (spoof_wellDefined oBuf m0), ?_, rfl, ?_⟩
  · apply Tensor.ext1
    intro i
    simp [spoofSpec, Layout.rowMajor, Layout.read, Tensor.get1, Tensor.of1,
      pallasExec, spoofGridSpec, spoofBody, gridPoints, stepGrid, flushAndPrep, makeSlots,
      flushAll, storeBlock, addrInBlock, unflatCoord, coordInBlock, BlockSpec.blockIdx, Mem.read]
  · refine Mem.agreeOutside_mono ?_ (pallasExec_agreeOutside_outBufs _ _ junk m0)
    rintro a ⟨x, hx, hbuf⟩
    simp only [spoofGridSpec, List.mem_singleton] at hx
    subst hx
    exact hbuf.symm

end

end TpuKernelLean.Examples.AttackA8
