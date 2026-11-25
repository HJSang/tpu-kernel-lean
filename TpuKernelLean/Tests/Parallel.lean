import TpuKernelLean.Pallas.Parallel

/-!
Regression tests for `dimension_semantics`: kernels that mark a grid dimension `"parallel"` while
carrying cross-iteration state (scratch accumulator or output-VMEM accumulator), racing on HBM
(read-write overlap across parallel iterations), or violating Mosaic's structural constraints
(length mismatch or `"parallel"` after `"arbitrary"`) are now rejected by `pallasCall`.
-/

namespace TpuKernelLean.Tests.Parallel

open TpuKernelLean.Pallas

/-! ### 1. Scratch accumulator carried across a parallel dimension (disjoint outputs) -/

/-- A 2-step prefix-sum kernel with disjoint outputs (`out[0]`, `out[1]`) and a 1-element scratch
accumulator, erroneously marking its grid dimension `"parallel"`. -/
def scratchCarryGridSpec : GridSpec :=
  { grid := [2]
    inSpecs := [(⟨0, [2]⟩, { blockShape := [1], indexMap := fun g _ => g.map (↑) })]
    outSpecs := [(⟨1, [2]⟩, { blockShape := [1], indexMap := fun g _ => g.map (↑) })]
    scratch := [[1]]
    dimSemantics := [.parallel] }

/-- Kernel body computing `acc := (if k = 0 then 0 else scratch[0]) + x[k]` and writing `acc` to
both output block `k` and scratch. -/
noncomputable def scratchCarryBody : Body := fun g _ ins prev =>
  let k := g.getD 0 0
  let acc := (if k = 0 then 0 else (prev.getD 1 fun _ => 0) [0]) + (ins.getD 0 fun _ => 0) [0]
  [fun _ => acc, fun _ => acc]

/-- Carrying a scratch accumulator across a `"parallel"` dimension is rejected by `pallasCall`,
even when all output blocks are disjoint. -/
theorem scratchCarry_rejected (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall scratchCarryGridSpec scratchCarryBody junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have hres := h.parallel_scratch_reset (by decide) [1]
    (by simp [scratchCarryGridSpec, GridSpec.parGrid, GridSpec.numParallel, List.takeWhile,
      DimSemantics.isParallel, gridPoints, List.range_succ])
    [] (by simp [scratchCarryGridSpec, GridSpec.arbGrid, GridSpec.numParallel, List.takeWhile,
      DimSemantics.isParallel, gridPoints])
    [fun _ => 0] [fun _ => 0] [fun _ => 0] [fun _ => 1] rfl rfl rfl
  have hval := congrArg (fun l => (l.getD 0 fun _ => 0) [0]) hres
  simp [scratchCarryBody] at hval

/-- Witness of the hardware race on 2 cores: running `[0]` on core 0 and `[1]` on core 1 (starting
with junk scratch `fun _ => 1` on core 1) produces a different `out[1]` than sequential `pallasExec`
when `x[0] = 0`. -/
theorem scratchCarry_races_on_megacore (m0 : Mem) (hx0 : m0.read 0 0 = 0) :
    pallasExecPar (C := 2) scratchCarryGridSpec scratchCarryBody (fun _ => 0) m0
      (fun c => if c = 0 then [[0]] else [[1]])
      (fun c => if c = 0 then [none] else [some ([], fun _ => 1)])
      [0, 1] [0, 1] ≠
    pallasExec scratchCarryGridSpec scratchCarryBody (fun _ => 0) m0 := by
  intro h
  have h1 := congrArg (fun m : Mem => m.read 1 1) h
  simp [pallasExecPar, pallasExec, stepCore, scratchCarryGridSpec, scratchCarryBody,
    GridSpec.groupPoints, GridSpec.arbGrid, GridSpec.numParallel, List.takeWhile,
    DimSemantics.isParallel, gridPoints, List.range_succ,
    stepGrid, flushAndPrep, flushAll, initOutMem, initOutMemAux, makeSlots,
    loadBlock, storeBlock, addrInBlock, coordInBlock, unflatCoord, globalCoord,
    inBounds, flatCoord, BlockSpec.blockIdx, Mem.read, Function.update] at h1
  simp only [Mem.read] at hx0
  linarith

/-! ### 2. Output-VMEM accumulator carried across a parallel dimension -/

/-- A 2-step reduction kernel carrying an accumulator in output block `[0]` across a dimension
marked `"parallel"`. -/
def outCarryGridSpec : GridSpec :=
  { grid := [2]
    inSpecs := [(⟨0, [2]⟩, { blockShape := [1], indexMap := fun g _ => g.map (↑) })]
    outSpecs := [(⟨1, [1]⟩, { blockShape := [1], indexMap := fun _ _ => [0] })]
    dimSemantics := [.parallel] }

/-- Kernel body accumulating into the persistent output VMEM block across step 0 and step 1. -/
noncomputable def outCarryBody : Body := fun g _ ins prev =>
  let k := g.getD 0 0
  [fun loc => (if k = 0 then 0 else (prev.getD 0 fun _ => 0) loc) + (ins.getD 0 fun _ => 0) [0]]

/-- Carrying an accumulator in a shared output block across a `"parallel"` dimension is rejected
by `pallasCall` (write-write overlap across parallel coordinates). -/
theorem outCarry_rejected (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall outCarryGridSpec outCarryBody junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  apply h.parallel_outputs_disjoint [0]
    (by simp [outCarryGridSpec, GridSpec.parGrid, GridSpec.numParallel, List.takeWhile,
      DimSemantics.isParallel, gridPoints, List.range_succ])
    [1]
    (by simp [outCarryGridSpec, GridSpec.parGrid, GridSpec.numParallel, List.takeWhile,
      DimSemantics.isParallel, gridPoints, List.range_succ])
    (by decide) []
    (by simp [outCarryGridSpec, GridSpec.arbGrid, GridSpec.numParallel, List.takeWhile,
      DimSemantics.isParallel, gridPoints])
    []
    (by simp [outCarryGridSpec, GridSpec.arbGrid, GridSpec.numParallel, List.takeWhile,
      DimSemantics.isParallel, gridPoints])
    _ (List.head_mem (by simp [outCarryGridSpec]))
    _ (List.head_mem (by simp [outCarryGridSpec]))
  refine ⟨rfl, 0, ?_, ?_⟩ <;>
    simp [outCarryGridSpec, BlockSpec.blockIdx, addrInBlock, coordInBlock, unflatCoord]

/-! ### 3. Read-write (WAR) race across a parallel dimension -/

/-- Step `[0]` reads element `1` and writes element `0`; step `[1]` reads element `1` and writes
element `1`. In sequential row-major order `no_stale_read` and `aliases_ok` pass, but under
`"parallel"` running `[1]` before `[0]` causes a write-after-read race. -/
def warRaceGridSpec : GridSpec :=
  { grid := [2]
    inSpecs := [(⟨0, [2]⟩, { blockShape := [1], indexMap := fun _ _ => [1] })]
    outSpecs := [(⟨0, [2]⟩, { blockShape := [1], indexMap := fun g _ => g.map (↑) })]
    aliases := [(0, 0)]
    dimSemantics := [.parallel] }

/-- A cross-iteration read-write overlap on a `"parallel"` dimension is rejected by `pallasCall`. -/
theorem warRace_rejected (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall warRaceGridSpec body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  apply h.parallel_read_write_disjoint [1]
    (by simp [warRaceGridSpec, GridSpec.parGrid, GridSpec.numParallel, List.takeWhile,
      DimSemantics.isParallel, gridPoints, List.range_succ])
    [0]
    (by simp [warRaceGridSpec, GridSpec.parGrid, GridSpec.numParallel, List.takeWhile,
      DimSemantics.isParallel, gridPoints, List.range_succ])
    (by decide) []
    (by simp [warRaceGridSpec, GridSpec.arbGrid, GridSpec.numParallel, List.takeWhile,
      DimSemantics.isParallel, gridPoints])
    []
    (by simp [warRaceGridSpec, GridSpec.arbGrid, GridSpec.numParallel, List.takeWhile,
      DimSemantics.isParallel, gridPoints])
    _ (List.head_mem (by simp [warRaceGridSpec]))
    _ (List.head_mem (by simp [warRaceGridSpec]))
  refine ⟨rfl, 1, ?_, ?_⟩ <;>
    simp [warRaceGridSpec, BlockSpec.blockIdx, addrInBlock, coordInBlock, unflatCoord]

/-! ### 4. Mosaic structural constraints on `dimSemantics` -/

/-- `dimSemantics` length mismatch with `grid` is rejected by `pallasCall`. -/
theorem badLen_rejected (body : Body) (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall
      { grid := [2, 2], inSpecs := [], outSpecs := [],
        dimSemantics := [.parallel] }
      body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have := h.dim_semantics_len
  simp at this

/-- A `"parallel"` dimension after an `"arbitrary"` dimension is rejected by `pallasCall`. -/
theorem nonLeadingParallel_rejected (body : Body)
    (junk : ℕ → ℝ) (m0 : Mem) :
    pallasCall
      { grid := [2, 2], inSpecs := [], outSpecs := [],
        dimSemantics := [.arbitrary, .parallel] }
      body junk m0 = none := by
  apply pallasCall_of_not_wellDefined
  intro h
  have := h.dim_semantics_prefix
  simp [parallelPrefix, DimSemantics.isParallel] at this

end TpuKernelLean.Tests.Parallel
