import TpuKernelLean.Pallas.Core

/-!
Tests of `scratch_shapes`: a scratch buffer persists across grid steps, and it starts as junk.
-/

namespace TpuKernelLean.Tests.Scratch

open TpuKernelLean.Pallas

/-- Sum `x : [2]` into a one-element scratch accumulator over a 2-step grid, the pattern of
`matmul_kernel.py`: zero the accumulator at step 0, add each block, write the output at the last
step. The output block index is constant, so its VMEM block also persists. -/
def accGridSpec : GridSpec :=
  { grid := [2]
    inSpecs := [(⟨0, [2]⟩, { blockShape := [1], indexMap := fun g _ => g.map (↑) })]
    outSpecs := [(⟨1, [1]⟩, { blockShape := [1], indexMap := fun _ _ => [0] })]
    scratch := [[1]] }

noncomputable def accBody : Body := fun g _ ins prev =>
  let k := g.getD 0 0
  let acc := (if k = 0 then 0 else (prev.getD 1 fun _ => 0) [0]) + (ins.getD 0 fun _ => 0) [0]
  [fun loc => if k = 1 then acc else (prev.getD 0 fun _ => 0) loc, fun _ => acc]

theorem acc_wellDefined (m0 : Mem) : accGridSpec.WellDefined accBody m0 where
  grid_pos := by
    intro d hd
    simp [accGridSpec] at hd
    omega
  block_specs_ok := by
    intro p hp
    simp only [accGridSpec, List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil,
      or_false] at hp
    rcases hp with rfl | rfl <;> simp
  blocks_ok := by
    intro k hk p hp
    simp [accGridSpec, gridPoints_length] at hk
    simp only [accGridSpec, List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil,
      or_false] at hp
    interval_cases k <;> rcases hp with rfl | rfl <;>
      simp [blockOk, GridSpec.idxAt, accGridSpec, gridPoints, List.range_succ]
  outputs_consecutive := by
    intro p hp
    simp only [accGridSpec, List.mem_singleton] at hp
    subst hp
    simp [noRevisit, GridSpec.idxAt, accGridSpec, gridPoints, List.range_succ]
  body_arity := fun _ _ _ _ => rfl
  no_stale_read := by
    intro s t _ _ i hi o ho hov
    simp only [accGridSpec, List.mem_singleton] at hi ho
    subst hi ho
    exact absurd hov.1 (by decide)
  aliases_ok := GridSpec.AliasesValid.of_empty rfl (by simp [accGridSpec]) (by
    intro inp hinp outp houtp
    simp only [accGridSpec, List.mem_singleton] at hinp houtp
    subst hinp houtp
    decide) (by simp [accGridSpec])
  dim_semantics_len := rfl
  dim_semantics_prefix := rfl
  parallel_outputs_disjoint := by
    intro pg₁ h₁ pg₂ h₂ hne
    simp [accGridSpec, GridSpec.parGrid, GridSpec.numParallel, DimSemantics.isParallel,
      gridPoints] at h₁ h₂
    exact absurd (h₁.trans h₂.symm) hne
  parallel_read_write_disjoint := by
    intro pg₁ h₁ pg₂ h₂ hne
    simp [accGridSpec, GridSpec.parGrid, GridSpec.numParallel, DimSemantics.isParallel,
      gridPoints] at h₁ h₂
    exact absurd (h₁.trans h₂.symm) hne
  parallel_scratch_reset := by
    intro hne
    exact absurd rfl hne

/-- The scratch accumulator carries the partial sum from step 0 to step 1. -/
theorem acc_correct (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', pallasCall accGridSpec accBody junk m0 = some m' ∧
      m'.read 1 0 = m0.read 0 0 + m0.read 0 1 := by
  refine ⟨_, pallasCall_of_wellDefined _ (acc_wellDefined m0), ?_⟩
  simp [pallasExec, accGridSpec, accBody, gridPoints, List.range_succ, stepGrid, flushAndPrep,
    flushAll, initOutMem, initOutMemAux, makeSlots, loadBlock, storeBlock,
    addrInBlock, coordInBlock, unflatCoord, globalCoord, inBounds, flatCoord,
    BlockSpec.blockIdx, Mem.read]

/-- Same kernel with the step-0 reset removed: the output then depends on the initial scratch
contents, which are junk, so no theorem can assume scratch starts at zero. -/
noncomputable def noResetBody : Body := fun g _ ins prev =>
  let k := g.getD 0 0
  let acc := (prev.getD 1 fun _ => 0) [0] + (ins.getD 0 fun _ => 0) [0]
  [fun loc => if k = 1 then acc else (prev.getD 0 fun _ => 0) loc, fun _ => acc]

theorem noReset_depends_on_junk (m0 : Mem) :
    pallasExec accGridSpec noResetBody (fun _ => 0) m0 ≠
      pallasExec accGridSpec noResetBody (fun _ => 1) m0 := by
  intro h
  have := congrArg (fun m : Mem => m.read 1 0) h
  simp [pallasExec, accGridSpec, noResetBody, gridPoints, List.range_succ, stepGrid,
    flushAndPrep, flushAll, initOutMem, initOutMemAux, makeSlots, loadBlock,
    storeBlock, addrInBlock, coordInBlock, unflatCoord, globalCoord, inBounds,
    flatCoord, BlockSpec.blockIdx, Mem.read, junkAt] at this

end TpuKernelLean.Tests.Scratch
