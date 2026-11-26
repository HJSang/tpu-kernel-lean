import TpuKernelLean.Pallas.Lemmas
import TpuKernelLean.Spec

/-!
Multi-core / arbitrary-order execution semantics for `dimension_semantics` and its soundness proof.

Under `GridSpec.WellDefined`, parallel dimensions form a leading prefix (`gs.parGrid`) followed by
arbitrary dimensions (`gs.arbGrid`). A parallel execution (`pallasExecPar`) partitions the parallel
iteration groups across `C` cores in arbitrary order, gives each core private VMEM starting from
empty output slots (junk VMEM) and arbitrary initial scratch slots (`initScratch c`), interleaves
grid steps across cores according to an arbitrary schedule `trace`, and flushes the cores' final
VMEM output slots in an arbitrary order `flushOrder`.

`pallasExecPar_eq_pallasExec` proves that every complete parallel execution yields the exact same
final HBM memory as the sequential `pallasExec`.
-/

namespace TpuKernelLean.Pallas

/-! ### Grid and coordinate helpers -/

/-- Row-major grid points decompose as parallel coordinates followed by arbitrary coordinates. -/
theorem gridPoints_append : ∀ (p a : List ℕ),
    gridPoints (p ++ a) = (gridPoints p).flatMap fun pg => (gridPoints a).map (pg ++ ·)
  | [], a => by simp [gridPoints]
  | d :: ds, a => by
    simp only [List.cons_append, gridPoints, gridPoints_append ds a, List.map_flatMap,
      List.flatMap_assoc, List.flatMap_map, List.map_map, Function.comp_def]

/-- `gridPoints grid` has no duplicate grid points. -/
private theorem gridPoints_nodup : ∀ (g : List ℕ), (gridPoints g).Nodup
  | [] => by simp [gridPoints]
  | d :: ds => by
    rw [gridPoints, List.nodup_flatMap]
    refine ⟨fun i _ => (gridPoints_nodup ds).map (fun x y h => by simpa using h), ?_⟩
    apply List.nodup_range.pairwise_of_forall_ne
    intro i _ i' _ hne x hx hx'
    simp only [List.mem_map] at hx hx'
    obtain ⟨_, _, rfl⟩ := hx
    obtain ⟨_, _, h⟩ := hx'
    exact hne (List.cons.inj h).1.symm

/-- Every in-bounds coordinate flattens strictly below `shape.prod`. -/
theorem flatCoord_lt : ∀ (shape c : List ℕ),
    inBounds shape c = true → flatCoord shape c < shape.prod
  | [], [], _ => by simp [flatCoord]
  | [], _ :: _, h => by simp [inBounds] at h
  | _ :: _, [], h => by simp [inBounds] at h
  | d :: ds, x :: xs, h => by
    simp only [inBounds, Bool.and_eq_true, decide_eq_true_eq] at h
    have ih := flatCoord_lt ds xs h.2
    simp only [flatCoord, List.prod_cons]
    have hle : (x + 1) * ds.prod ≤ d * ds.prod := Nat.mul_le_mul_right _ h.1
    rw [Nat.succ_mul] at hle
    omega

/-- Unflattening the row-major offset of an in-bounds coordinate recovers that coordinate. -/
theorem unflatCoord_flatCoord : ∀ (shape c : List ℕ),
    inBounds shape c = true → unflatCoord shape (flatCoord shape c) = c
  | [], [], _ => rfl
  | [], _ :: _, h => by simp [inBounds] at h
  | _ :: _, [], h => by simp [inBounds] at h
  | d :: ds, x :: xs, h => by
    simp only [inBounds, Bool.and_eq_true, decide_eq_true_eq] at h
    have hlt := flatCoord_lt ds xs h.2
    have hpos : 0 < ds.prod := Nat.lt_of_le_of_lt (Nat.zero_le _) hlt
    have hne : ds.prod ≠ 0 := Nat.ne_of_gt hpos
    have hdiv : (x * ds.prod + flatCoord ds xs) / ds.prod = x := by
      rw [Nat.mul_comm, Nat.mul_add_div hpos, Nat.div_eq_of_lt hlt, Nat.add_zero]
    have hmod : (x * ds.prod + flatCoord ds xs) % ds.prod = flatCoord ds xs := by
      rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hlt]
    simp [unflatCoord, flatCoord, hne, hdiv, hmod, unflatCoord_flatCoord ds xs h.2]

private theorem blockOk_lengths : ∀ (shape bs : List ℕ) (idx : List ℤ),
    blockOk shape bs idx = true → shape.length = bs.length ∧ bs.length = idx.length
  | [], [], [], _ => ⟨rfl, rfl⟩
  | _ :: ds, _ :: bs, _ :: is, h => by
    simp only [blockOk, Bool.and_eq_true] at h
    have ih := blockOk_lengths ds bs is h.2
    simp [ih.1, ih.2]
  | [], _ :: _, _, h | [], [], _ :: _, h | _ :: _, [], _, h | _ :: _, _ :: _, [], h => by
    simp [blockOk] at h

private theorem blockOk_zero_inBounds : ∀ (shape bs : List ℕ) (idx : List ℤ),
    blockOk shape bs idx = true →
    inBounds shape (globalCoord bs (idx.map Int.toNat) (bs.map fun _ => 0)) = true ∧
    coordInBlock bs (idx.map Int.toNat) (globalCoord bs (idx.map Int.toNat) (bs.map fun _ => 0)) = true
  | [], [], [], _ => by simp [inBounds, coordInBlock, globalCoord]
  | d :: ds, b :: bs, i :: is, h => by
    simp only [blockOk, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨hb, hi0, hilt⟩, hrest⟩ := h
    have ih := blockOk_zero_inBounds ds bs is hrest
    have hiNat : i.toNat * b < d := by
      have h1 : (i.toNat : ℤ) = i := Int.toNat_of_nonneg hi0
      zify
      rw [h1]
      exact hilt
    simp only [List.map_cons, globalCoord, Nat.add_zero, inBounds, coordInBlock,
      Bool.and_eq_true, decide_eq_true_eq]
    refine ⟨⟨hiNat, ih.1⟩, ⟨⟨le_rfl, ?_⟩, ih.2⟩⟩
    rw [Nat.succ_mul]
    omega
  | [], _ :: _, _, h | [], [], _ :: _, h | _ :: _, [], _, h | _ :: _, _ :: _, [], h => by
    simp [blockOk] at h

/-- Every legal (`blockOk`) block contains at least one in-bounds element of its array. -/
theorem exists_addrInBlock_of_blockOk (arr : ArrayDesc) (bs : List ℕ) (idx : List ℤ)
    (h : blockOk arr.shape bs idx = true) :
    ∃ a, addrInBlock arr bs (idx.map Int.toNat) a = true := by
  have ⟨hin, hcb⟩ := blockOk_zero_inBounds arr.shape bs idx h
  refine ⟨flatCoord arr.shape (globalCoord bs (idx.map Int.toNat) (bs.map fun _ => 0)), ?_⟩
  simp only [addrInBlock, flatCoord_lt _ _ hin, unflatCoord_flatCoord _ _ hin, hcb,
    decide_true, Bool.and_self]

private theorem coordInBlock_globalCoord : ∀ (bs bi loc : List ℕ),
    inBounds bs loc = true → bs.length = bi.length →
    coordInBlock bs bi (globalCoord bs bi loc) = true
  | [], [], [], _, _ => rfl
  | b :: bs, i :: bi, l :: loc, hloc, hlen => by
    simp only [inBounds, Bool.and_eq_true, decide_eq_true_eq] at hloc
    simp only [List.length_cons, Nat.succ.injEq] at hlen
    simp only [globalCoord, coordInBlock, Bool.and_eq_true, decide_eq_true_eq]
    refine ⟨⟨Nat.le_add_right _ _, ?_⟩, coordInBlock_globalCoord bs bi loc hloc.2 hlen⟩
    rw [Nat.succ_mul]
    omega
  | [], _ :: _, _, _, hlen | _ :: _, [], _, _, hlen => by simp at hlen
  | [], [], _ :: _, hloc, _ | _ :: _, _ :: _, [], hloc, _ => by simp [inBounds] at hloc

/-- `loadBlock` depends only on HBM addresses covered by its block. -/
theorem loadBlock_congr_block (arr : ArrayDesc) (bs bi : List ℕ) (hlen : bs.length = bi.length)
    (junk : ℕ → ℝ) {m₁ m₂ : Mem}
    (h : ∀ a, addrInBlock arr bs bi a = true → m₁.read arr.buf a = m₂.read arr.buf a) :
    loadBlock arr bs bi junk m₁ = loadBlock arr bs bi junk m₂ := by
  funext loc
  simp only [loadBlock]
  split_ifs with hcond
  · simp only [Bool.and_eq_true] at hcond
    apply h
    simp only [addrInBlock, Bool.and_eq_true, decide_eq_true_eq]
    refine ⟨flatCoord_lt arr.shape _ hcond.2, ?_⟩
    rw [unflatCoord_flatCoord arr.shape _ hcond.2]
    exact coordInBlock_globalCoord bs bi loc hcond.1 hlen
  · rfl

/-! ### Parallel execution model -/

/-- State of one TensorCore during parallel execution: its remaining grid points and its private
VMEM (`outSpecs` slots followed by `scratch` slots). [PAL-PAR-EXEC] -/
structure CoreState where
  /-- Remaining grid points assigned to this core. -/
  remPoints : List (List ℕ)
  /-- Private VMEM slots (output slots followed by scratch slots). -/
  slots : List OutSlot

noncomputable section

/-- Execute one grid step on core `c`, leaving HBM and all cores unchanged if core `c` has
finished its queue. [PAL-PAR-EXEC] -/
def stepCore {C : ℕ} (gs : GridSpec) (body : Body) (scalars : ℕ → ℤ) (junk : ℕ → ℝ)
    (state : Mem × (Fin C → CoreState)) (c : Fin C) : Mem × (Fin C → CoreState) :=
  let (m, cores) := state
  match (cores c).remPoints with
  | [] => (m, cores)
  | g :: rest =>
    let res := stepGrid gs body scalars junk (m, (cores c).slots) g
    (res.1, Function.update cores c { remPoints := rest, slots := res.2 })

/-- Multi-core / arbitrary-order execution of a `pallas_call` over `C` TensorCores. Each core `c`
receives an ordered list `assign c` of parallel coordinates, starts from empty output VMEM (`none`)
and arbitrary initial scratch `initScratch c`, steps according to an arbitrary interleaving `trace`,
and flushes its final output VMEM in order `flushOrder`. [PAL-PAR-EXEC] -/
def pallasExecPar {C : ℕ} (gs : GridSpec) (body : Body)
    (junk : ℕ → ℝ) (m0 : Mem)
    (assign : Fin C → List (List ℕ))
    (initScratch : Fin C → List OutSlot)
    (trace : List (Fin C))
    (flushOrder : List (Fin C)) : Mem :=
  let scalars := gs.scalarPrefetch m0
  let mInit := initOutMem gs (junkAt junk (Nat.pair 0 0)) m0
  let outNone : List OutSlot := gs.outSpecs.map fun _ => none
  let initCores : Fin C → CoreState := fun c =>
    { remPoints := (assign c).flatMap gs.groupPoints
      slots := outNone ++ initScratch c }
  let finalState := trace.foldl (stepCore gs body scalars junk) (mInit, initCores)
  flushOrder.foldl (fun m c => flushAll gs.outSpecs (finalState.2 c).slots m) finalState.1

/-! ### Flush and isolated-group lemmas -/

private theorem flushAndPrep_outs_length (g : List ℕ) (sc : ℕ → ℤ) (junk : ℕ → ℝ) :
    ∀ (idx : ℕ) (specs : List (ArrayDesc × BlockSpec)) (slots : List OutSlot) (m : Mem),
      (flushAndPrep g sc junk idx specs slots m).2.2.length = specs.length
  | _, [], _, _ => rfl
  | idx, _ :: rest, slots, m => by
    simp only [flushAndPrep, List.length_cons]
    exact congrArg Nat.succ (flushAndPrep_outs_length g sc junk (idx + 1) rest _ _)

private theorem flushAndPrep_snd_indep_mem (g : List ℕ) (sc : ℕ → ℤ) (junk : ℕ → ℝ) :
    ∀ (idx : ℕ) (specs : List (ArrayDesc × BlockSpec)) (slots : List OutSlot) (m₁ m₂ : Mem),
      (flushAndPrep g sc junk idx specs slots m₁).2 =
      (flushAndPrep g sc junk idx specs slots m₂).2
  | _, [], _, _, _ => rfl
  | idx, (outArr, outSpec) :: rest, slots, m₁, m₂ => by
    cases slots with
    | nil =>
      simp only [flushAndPrep]
      rw [flushAndPrep_snd_indep_mem g sc junk (idx + 1) rest [] m₁ m₂]
    | cons s ss =>
      cases s with
      | none =>
        simp only [flushAndPrep]
        rw [flushAndPrep_snd_indep_mem g sc junk (idx + 1) rest ss m₁ m₂]
      | some p =>
        simp only [flushAndPrep]
        rw [flushAndPrep_snd_indep_mem g sc junk (idx + 1) rest ss
          (if p.1 = outSpec.blockIdx g sc then m₁ else storeBlock outArr outSpec.blockShape p.1 p.2 m₁)
          (if p.1 = outSpec.blockIdx g sc then m₂ else storeBlock outArr outSpec.blockShape p.1 p.2 m₂)]

private theorem flushAndPrep_read_congr (g : List ℕ) (sc : ℕ → ℤ) (junk : ℕ → ℝ)
    (b : BufId) (a : ℕ) :
    ∀ (idx : ℕ) (specs : List (ArrayDesc × BlockSpec)) (slots : List OutSlot) (m₁ m₂ : Mem),
      m₁.read b a = m₂.read b a →
      (flushAndPrep g sc junk idx specs slots m₁).1.read b a =
      (flushAndPrep g sc junk idx specs slots m₂).1.read b a
  | _, [], _, _, _, h => h
  | idx, (outArr, outSpec) :: rest, slots, m₁, m₂, h => by
    simp only [flushAndPrep]
    apply flushAndPrep_read_congr g sc junk b a (idx + 1) rest
    cases slots with
    | nil => exact h
    | cons s ss =>
      cases s with
      | none => exact h
      | some p =>
        dsimp only
        split_ifs
        · exact h
        · exact storeBlock_read_congr _ _ _ _ b a h

private theorem flushAll_read_congr (b : BufId) (a : ℕ) :
    ∀ (specs : List (ArrayDesc × BlockSpec)) (slots : List OutSlot) (m₁ m₂ : Mem),
      m₁.read b a = m₂.read b a →
      (flushAll specs slots m₁).read b a = (flushAll specs slots m₂).read b a
  | [], _, _, _, h => h
  | _ :: rest, slots, m₁, m₂, h => by
    simp only [flushAll]
    apply flushAll_read_congr b a rest
    cases slots with
    | nil => exact h
    | cons s ss =>
      cases s with
      | none => exact h
      | some p => exact storeBlock_read_congr _ _ _ _ b a h

private theorem flushAll_outNone :
    ∀ (specs : List (ArrayDesc × BlockSpec)) (rest : List OutSlot) (m : Mem),
      flushAll specs ((specs.map fun _ => none) ++ rest) m = m
  | [], _, _ => rfl
  | _ :: specs, rest, m => flushAll_outNone specs rest m

private theorem flushAndPrep_outNone_fst (g : List ℕ) (sc : ℕ → ℤ) (junk : ℕ → ℝ) :
    ∀ (idx : ℕ) (specs : List (ArrayDesc × BlockSpec)) (rest : List OutSlot) (m : Mem),
      (flushAndPrep g sc junk idx specs ((specs.map fun _ => none) ++ rest) m).1 = m
  | _, [], _, _ => rfl
  | idx, _ :: specs, rest, m => flushAndPrep_outNone_fst g sc junk (idx + 1) specs rest m

private theorem flushAndPrep_outNone_snd (g : List ℕ) (sc : ℕ → ℤ) (junk : ℕ → ℝ) :
    ∀ (idx : ℕ) (specs : List (ArrayDesc × BlockSpec)) (r₁ r₂ : List OutSlot) (m₁ m₂ : Mem),
      (flushAndPrep g sc junk idx specs ((specs.map fun _ => none) ++ r₁) m₁).2 =
      (flushAndPrep g sc junk idx specs ((specs.map fun _ => none) ++ r₂) m₂).2
  | _, [], _, _, _, _ => rfl
  | idx, _ :: specs, r₁, r₂, m₁, m₂ => by
    simp only [flushAndPrep, List.map_cons, List.cons_append]
    have ih := flushAndPrep_outNone_snd g sc junk (idx + 1) specs r₁ r₂ m₁ m₂
    exact congrArg₂ (fun a b => (a, b)) (congrArg _ (congrArg Prod.fst ih))
      (congrArg _ (congrArg Prod.snd ih))

private theorem flushAndPrep_read_outside (g : List ℕ) (sc : ℕ → ℤ) (junk : ℕ → ℝ)
    (b : BufId) (a : ℕ) :
    ∀ (idx : ℕ) (specs : List (ArrayDesc × BlockSpec)) (slots : List OutSlot) (m : Mem),
      (∀ q o, specs[q]? = some o → ∀ oldIdx oldVal, slots.getD q none = some (oldIdx, oldVal) →
        b ≠ o.1.buf ∨ addrInBlock o.1 o.2.blockShape oldIdx a = false) →
      (flushAndPrep g sc junk idx specs slots m).1.read b a = m.read b a
  | _, [], _, _, _ => rfl
  | idx, (outArr, outSpec) :: rest, slots, m, hout => by
    cases slots with
    | nil =>
      simp only [flushAndPrep]
      exact flushAndPrep_read_outside g sc junk b a (idx + 1) rest [] m
        (fun q o hq oldIdx oldVal hs => by simp at hs)
    | cons s ss =>
      have htail : ∀ q o, rest[q]? = some o → ∀ oldIdx oldVal,
          ss.getD q none = some (oldIdx, oldVal) →
          b ≠ o.1.buf ∨ addrInBlock o.1 o.2.blockShape oldIdx a = false :=
        fun q o hq oldIdx oldVal hs => hout (q + 1) o hq oldIdx oldVal hs
      cases s with
      | none =>
        simp only [flushAndPrep]
        exact flushAndPrep_read_outside g sc junk b a (idx + 1) rest ss m htail
      | some p =>
        simp only [flushAndPrep]
        rw [flushAndPrep_read_outside g sc junk b a (idx + 1) rest ss _ htail]
        split_ifs
        · rfl
        · exact storeBlock_read_outside _ _ _ _ _ _ _ (hout 0 _ rfl p.1 p.2 rfl)

private theorem flushAll_read_outside (b : BufId) (a : ℕ) :
    ∀ (specs : List (ArrayDesc × BlockSpec)) (slots : List OutSlot) (m : Mem),
      (∀ q o, specs[q]? = some o → ∀ oldIdx oldVal, slots.getD q none = some (oldIdx, oldVal) →
        b ≠ o.1.buf ∨ addrInBlock o.1 o.2.blockShape oldIdx a = false) →
      (flushAll specs slots m).read b a = m.read b a
  | [], _, _, _ => rfl
  | (outArr, outSpec) :: rest, slots, m, hout => by
    cases slots with
    | nil =>
      simp only [flushAll]
      exact flushAll_read_outside b a rest [] m (fun q o hq oldIdx oldVal hs => by simp at hs)
    | cons s ss =>
      have htail : ∀ q o, rest[q]? = some o → ∀ oldIdx oldVal,
          ss.getD q none = some (oldIdx, oldVal) →
          b ≠ o.1.buf ∨ addrInBlock o.1 o.2.blockShape oldIdx a = false :=
        fun q o hq oldIdx oldVal hs => hout (q + 1) o hq oldIdx oldVal hs
      cases s with
      | none =>
        simp only [flushAll]
        exact flushAll_read_outside b a rest ss m htail
      | some p =>
        simp only [flushAll]
        rw [flushAll_read_outside b a rest ss _ htail]
        exact storeBlock_read_outside _ _ _ _ _ _ _ (hout 0 _ rfl p.1 p.2 rfl)

private theorem flushAndPrep_of_ne (g : List ℕ) (sc : ℕ → ℤ) (junk : ℕ → ℝ) :
    ∀ (idx : ℕ) (specs : List (ArrayDesc × BlockSpec)) (slots rest : List OutSlot) (m m' : Mem),
      (∀ q o, specs[q]? = some o → ∀ oldIdx oldVal, slots.getD q none = some (oldIdx, oldVal) →
        oldIdx ≠ o.2.blockIdx g sc) →
      (flushAndPrep g sc junk idx specs slots m).1 = flushAll specs slots m ∧
      (flushAndPrep g sc junk idx specs slots m).2 =
        (flushAndPrep g sc junk idx specs ((specs.map fun _ => none) ++ rest) m').2
  | _, [], _, _, _, _, _ => ⟨rfl, rfl⟩
  | idx, (outArr, outSpec) :: restSpecs, slots, rest, m, m', hne => by
    have htail : ∀ q o, restSpecs[q]? = some o → ∀ oldIdx oldVal,
        (match slots with | _ :: ss => ss | [] => []).getD q none = some (oldIdx, oldVal) →
        oldIdx ≠ o.2.blockIdx g sc := by
      intro q o hq oldIdx oldVal hs
      cases slots with
      | nil => simp at hs
      | cons s ss => exact hne (q + 1) o hq oldIdx oldVal hs
    cases slots with
    | nil =>
      have ih := flushAndPrep_of_ne g sc junk (idx + 1) restSpecs [] rest m m' htail
      simp only [flushAndPrep, flushAll, List.map_cons, List.cons_append]
      exact ⟨ih.1, by rw [ih.2]⟩
    | cons s ss =>
      cases s with
      | none =>
        have ih := flushAndPrep_of_ne g sc junk (idx + 1) restSpecs ss rest m m' htail
        simp only [flushAndPrep, flushAll, List.map_cons, List.cons_append]
        exact ⟨ih.1, by rw [ih.2]⟩
      | some p =>
        have hp : p.1 ≠ outSpec.blockIdx g sc := hne 0 _ rfl p.1 p.2 rfl
        have ih := flushAndPrep_of_ne g sc junk (idx + 1) restSpecs ss rest
          (storeBlock outArr outSpec.blockShape p.1 p.2 m) m' htail
        simp only [flushAndPrep, flushAll, List.map_cons, List.cons_append, hp, ↓reduceIte]
        exact ⟨ih.1, by rw [ih.2]⟩

/-- An HBM address `(b, a)` belongs to the output region of parallel coordinate `pg` if some output
block of some step in `pg`'s arbitrary group covers `(b, a)`. -/
def inOutRegion (gs : GridSpec) (sc : ℕ → ℤ) (pg : List ℕ) (b : BufId) (a : ℕ) : Prop :=
  ∃ ag ∈ gridPoints gs.arbGrid, ∃ o ∈ gs.outSpecs,
    b = o.1.buf ∧ addrInBlock o.1 o.2.blockShape (o.2.blockIdx (pg ++ ag) sc) a = true

/-- An HBM address `(b, a)` belongs to the input region of parallel coordinate `pg` if some input
block of some step in `pg`'s arbitrary group covers `(b, a)`. -/
def inInRegion (gs : GridSpec) (sc : ℕ → ℤ) (pg : List ℕ) (b : BufId) (a : ℕ) : Prop :=
  ∃ ag ∈ gridPoints gs.arbGrid, ∃ i ∈ gs.inSpecs,
    b = i.1.buf ∧ addrInBlock i.1 i.2.blockShape (i.2.blockIdx (pg ++ ag) sc) a = true

/-- Isolated execution state of running steps `done` starting from `mInit` and empty VMEM/scratch. -/
def isoState (gs : GridSpec) (body : Body) (sc : ℕ → ℤ) (junk : ℕ → ℝ)
    (mInit : Mem) (done : List (List ℕ)) : Mem × List OutSlot :=
  let initSlots := (gs.outSpecs.map fun _ => none) ++ gs.scratch.map fun _ => none
  done.foldl (stepGrid gs body sc junk) (mInit, initSlots)

/-- Isolated final HBM state of running all steps of parallel group `pg` and flushing. -/
def isoMem (gs : GridSpec) (body : Body) (sc : ℕ → ℤ) (junk : ℕ → ℝ)
    (mInit : Mem) (pg : List ℕ) : Mem :=
  let st := isoState gs body sc junk mInit (gs.groupPoints pg)
  flushAll gs.outSpecs st.2 st.1

private theorem outRegion_disjoint {gs : GridSpec} {body : Body} {m0 : Mem}
    (hwd : gs.WellDefined body m0) {pg₁ pg₂ : List ℕ}
    (h₁ : pg₁ ∈ gridPoints gs.parGrid) (h₂ : pg₂ ∈ gridPoints gs.parGrid) (hne : pg₁ ≠ pg₂)
    {b : BufId} {a : ℕ} :
    ¬ (inOutRegion gs (gs.scalarPrefetch m0) pg₁ b a ∧
       inOutRegion gs (gs.scalarPrefetch m0) pg₂ b a) := by
  rintro ⟨⟨ag₁, hag₁, o₁, ho₁, rfl, ha₁⟩, ⟨ag₂, hag₂, o₂, ho₂, hbuf, ha₂⟩⟩
  exact hwd.parallel_outputs_disjoint pg₁ h₁ pg₂ h₂ hne ag₁ hag₁ ag₂ hag₂ o₁ ho₁ o₂ ho₂
    ⟨hbuf, a, ha₁, ha₂⟩

private theorem inOutRegion_disjoint {gs : GridSpec} {body : Body} {m0 : Mem}
    (hwd : gs.WellDefined body m0) {pg₁ pg₂ : List ℕ}
    (h₁ : pg₁ ∈ gridPoints gs.parGrid) (h₂ : pg₂ ∈ gridPoints gs.parGrid) (hne : pg₁ ≠ pg₂)
    {b : BufId} {a : ℕ} :
    ¬ (inInRegion gs (gs.scalarPrefetch m0) pg₂ b a ∧
       inOutRegion gs (gs.scalarPrefetch m0) pg₁ b a) := by
  rintro ⟨⟨ag₂, hag₂, i, hi, rfl, ha₂⟩, ⟨ag₁, hag₁, o, ho, hbuf, ha₁⟩⟩
  exact hwd.parallel_read_write_disjoint pg₁ h₁ pg₂ h₂ hne ag₁ hag₁ ag₂ hag₂ i hi o ho
    ⟨hbuf, a, ha₂, ha₁⟩

private theorem blocks_ok_at_point {gs : GridSpec} {body : Body} {m0 : Mem}
    (hwd : gs.WellDefined body m0) {pg ag : List ℕ}
    (hpg : pg ∈ gridPoints gs.parGrid) (hag : ag ∈ gridPoints gs.arbGrid)
    {p : ArrayDesc × BlockSpec} (hp : p ∈ gs.inSpecs ++ gs.outSpecs) :
    blockOk p.1.shape p.2.blockShape (p.2.indexMap (pg ++ ag) (gs.scalarPrefetch m0)) = true := by
  have hmem : pg ++ ag ∈ gridPoints gs.grid := by
    have hgrid : gs.parGrid ++ gs.arbGrid = gs.grid := List.take_append_drop gs.numParallel gs.grid
    rw [← hgrid, gridPoints_append, List.mem_flatMap]
    exact ⟨pg, hpg, List.mem_map.mpr ⟨ag, hag, rfl⟩⟩
  obtain ⟨k, hk, hget⟩ := List.getElem_of_mem hmem
  have hok := hwd.blocks_ok k hk p hp
  simp only [GridSpec.idxAt, List.getD_eq_getElem _ _ hk, hget] at hok
  exact hok

private theorem blockIdx_ne_of_par_ne {gs : GridSpec} {body : Body} {m0 : Mem}
    (hwd : gs.WellDefined body m0) {pg₁ pg₂ ag₁ ag₂ : List ℕ}
    (h₁ : pg₁ ∈ gridPoints gs.parGrid) (h₂ : pg₂ ∈ gridPoints gs.parGrid) (hne : pg₁ ≠ pg₂)
    (hag₁ : ag₁ ∈ gridPoints gs.arbGrid) (hag₂ : ag₂ ∈ gridPoints gs.arbGrid)
    {o : ArrayDesc × BlockSpec} (ho : o ∈ gs.outSpecs) :
    o.2.blockIdx (pg₁ ++ ag₁) (gs.scalarPrefetch m0) ≠
      o.2.blockIdx (pg₂ ++ ag₂) (gs.scalarPrefetch m0) := by
  intro heq
  have hok := blocks_ok_at_point hwd h₁ hag₁ (List.mem_append_right _ ho)
  obtain ⟨a, ha⟩ := exists_addrInBlock_of_blockOk o.1 o.2.blockShape _ hok
  change addrInBlock o.1 o.2.blockShape (o.2.blockIdx (pg₁ ++ ag₁) (gs.scalarPrefetch m0)) a = true
    at ha
  exact hwd.parallel_outputs_disjoint pg₁ h₁ pg₂ h₂ hne ag₁ hag₁ ag₂ hag₂ o ho o ho
    ⟨rfl, a, ha, heq ▸ ha⟩

private theorem isoState_props {gs : GridSpec} {body : Body} {m0 : Mem}
    (hwd : gs.WellDefined body m0) (junk : ℕ → ℝ) (mInit : Mem) (pg : List ℕ) :
    ∀ (done : List (List ℕ)),
      (∀ g ∈ done, ∃ ag ∈ gridPoints gs.arbGrid, g = pg ++ ag) →
      let st := isoState gs body (gs.scalarPrefetch m0) junk mInit done
      st.2.length = gs.outSpecs.length + gs.scratch.length ∧
      (∀ q o, gs.outSpecs[q]? = some o → ∀ oldIdx oldVal, st.2.getD q none = some (oldIdx, oldVal) →
        ∃ ag ∈ gridPoints gs.arbGrid, oldIdx = o.2.blockIdx (pg ++ ag) (gs.scalarPrefetch m0)) ∧
      (∀ b a, ¬ inOutRegion gs (gs.scalarPrefetch m0) pg b a → st.1.read b a = mInit.read b a) := by
  intro done
  induction done using List.reverseRecOn with
  | nil =>
    intro _
    refine ⟨by simp [isoState], ?_, fun _ _ _ => rfl⟩
    intro q o hq oldIdx oldVal hs
    have hqlt : q < gs.outSpecs.length := by
      by_contra h
      rw [List.getElem?_eq_none (by omega)] at hq
      cases hq
    simp only [isoState, List.foldl_nil] at hs
    rw [List.getD_append _ _ _ _ (by simp [hqlt])] at hs
    simp [List.getD_eq_getElem?_getD, hqlt] at hs
  | append_singleton init g ih =>
    intro hmem
    have hinit : ∀ x ∈ init, ∃ ag ∈ gridPoints gs.arbGrid, x = pg ++ ag :=
      fun x hx => hmem x (by simp [hx])
    obtain ⟨ag, hag, rfl⟩ := hmem g (by simp)
    obtain ⟨hlen, hslots, hout⟩ := ih hinit
    set sc := gs.scalarPrefetch m0
    set st := isoState gs body sc junk mInit init
    have hst' : isoState gs body sc junk mInit (init ++ [pg ++ ag]) =
        stepGrid gs body sc junk st (pg ++ ag) := by
      simp [isoState, st, List.foldl_append]
    rw [hst']
    refine ⟨?_, ?_, ?_⟩
    · simp only [stepGrid, List.length_append, makeSlots_length, flushAndPrep_idxs,
        List.length_map, List.length_drop, hwd.body_arity]
      omega
    · intro q o hq oldIdx oldVal hs
      have hqlt : q < gs.outSpecs.length := by
        by_contra h
        rw [List.getElem?_eq_none (by omega)] at hq
        cases hq
      simp only [stepGrid] at hs
      rw [List.getD_append _ _ _ _ (by rw [makeSlots_length, flushAndPrep_idxs]; simpa using hqlt),
        makeSlots_getD _ _ q (o.2.blockIdx (pg ++ ag) sc) (by rw [flushAndPrep_idxs]; simp [hq])] at hs
      injection hs with hp
      injection hp with hidx _
      exact ⟨ag, hag, hidx.symm⟩
    · intro b a hnot
      simp only [stepGrid]
      rw [flushAndPrep_read_outside _ _ _ b a 0 gs.outSpecs st.2 st.1, hout b a hnot]
      intro q o hq oldIdx oldVal hs
      obtain ⟨ag', hag', rfl⟩ := hslots q o hq oldIdx oldVal hs
      by_cases hbuf : b = o.1.buf
      · right
        cases haddr : addrInBlock o.1 o.2.blockShape (o.2.blockIdx (pg ++ ag') sc) a
        · rfl
        · exfalso
          exact hnot ⟨ag', hag', o, List.mem_of_getElem? hq, hbuf, haddr⟩
      · exact Or.inl hbuf

private theorem stepGrid_congr_in_region {gs : GridSpec} {body : Body} {m0 : Mem}
    (hwd : gs.WellDefined body m0) (junk : ℕ → ℝ) {pg ag : List ℕ}
    (hpg : pg ∈ gridPoints gs.parGrid) (hag : ag ∈ gridPoints gs.arbGrid)
    {m₁ m₂ : Mem} {slots : List OutSlot}
    (hslots : ∀ q o, gs.outSpecs[q]? = some o → ∀ oldIdx oldVal,
      slots.getD q none = some (oldIdx, oldVal) →
      ∃ ag' ∈ gridPoints gs.arbGrid, oldIdx = o.2.blockIdx (pg ++ ag') (gs.scalarPrefetch m0))
    (hin : ∀ b a, inInRegion gs (gs.scalarPrefetch m0) pg b a → m₁.read b a = m₂.read b a) :
    (stepGrid gs body (gs.scalarPrefetch m0) junk (m₁, slots) (pg ++ ag)).2 =
      (stepGrid gs body (gs.scalarPrefetch m0) junk (m₂, slots) (pg ++ ag)).2 ∧
    (∀ b a, m₁.read b a = m₂.read b a →
      (stepGrid gs body (gs.scalarPrefetch m0) junk (m₁, slots) (pg ++ ag)).1.read b a =
      (stepGrid gs body (gs.scalarPrefetch m0) junk (m₂, slots) (pg ++ ag)).1.read b a) ∧
    (∀ b a, ¬ inOutRegion gs (gs.scalarPrefetch m0) pg b a →
      (stepGrid gs body (gs.scalarPrefetch m0) junk (m₁, slots) (pg ++ ag)).1.read b a =
        m₁.read b a) := by
  set sc := gs.scalarPrefetch m0
  set g := pg ++ ag
  have hfp2 : (flushAndPrep g sc (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) 0))
      0 gs.outSpecs slots m₁).2 =
    (flushAndPrep g sc (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) 0))
      0 gs.outSpecs slots m₂).2 :=
    flushAndPrep_snd_indep_mem g sc _ 0 gs.outSpecs slots m₁ m₂
  have hfp1_congr : ∀ b a, m₁.read b a = m₂.read b a →
      (flushAndPrep g sc (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) 0))
        0 gs.outSpecs slots m₁).1.read b a =
      (flushAndPrep g sc (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) 0))
        0 gs.outSpecs slots m₂).1.read b a :=
    fun b a h => flushAndPrep_read_congr g sc _ b a 0 gs.outSpecs slots m₁ m₂ h
  have hfp1_out : ∀ b a, ¬ inOutRegion gs sc pg b a →
      (flushAndPrep g sc (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) 0))
        0 gs.outSpecs slots m₁).1.read b a = m₁.read b a := by
    intro b a hnot
    apply flushAndPrep_read_outside _ _ _ b a 0 gs.outSpecs slots m₁
    intro q o hq oldIdx oldVal hs
    obtain ⟨ag', hag', rfl⟩ := hslots q o hq oldIdx oldVal hs
    by_cases hbuf : b = o.1.buf
    · right
      cases haddr : addrInBlock o.1 o.2.blockShape (o.2.blockIdx (pg ++ ag') sc) a
      · rfl
      · exfalso
        exact hnot ⟨ag', hag', o, List.mem_of_getElem? hq, hbuf, haddr⟩
    · exact Or.inl hbuf
  have hins :
      (gs.inSpecs.mapIdx fun i p =>
        loadBlock p.1 p.2.blockShape (p.2.blockIdx g sc)
          (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) (i + 1)))
          (flushAndPrep g sc (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) 0))
            0 gs.outSpecs slots m₁).1) =
      (gs.inSpecs.mapIdx fun i p =>
        loadBlock p.1 p.2.blockShape (p.2.blockIdx g sc)
          (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) (i + 1)))
          (flushAndPrep g sc (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) 0))
            0 gs.outSpecs slots m₂).1) := by
    apply List.ext_getElem (by simp)
    intro i h1 _
    have hi : i < gs.inSpecs.length := by simpa using h1
    simp only [List.getElem_mapIdx]
    have hmem : gs.inSpecs[i] ∈ gs.inSpecs := List.getElem_mem _
    have hok := blocks_ok_at_point hwd hpg hag (List.mem_append_left _ hmem)
    have hlen : gs.inSpecs[i].2.blockShape.length = (gs.inSpecs[i].2.blockIdx g sc).length := by
      simp only [BlockSpec.blockIdx, List.length_map]
      exact (blockOk_lengths _ _ _ hok).2
    apply loadBlock_congr_block _ _ _ hlen
    intro a ha
    exact hfp1_congr _ a (hin _ a ⟨ag, hag, gs.inSpecs[i], hmem, rfl, ha⟩)
  refine ⟨?_, hfp1_congr, hfp1_out⟩
  simp only [stepGrid, hfp2, hins]

/-! ### Global invariant for concurrent multi-core execution -/

/-- Logical progress of one core: flushed parallel groups, at most one active group `(pg, done, todo)`
with `done ≠ []`, and not-yet-started parallel groups. -/
private structure CoreProg where
  finished : List (List ℕ)
  active : Option (List ℕ × List (List ℕ) × List (List ℕ))
  pending : List (List ℕ)

private def CoreProg.activeList (p : CoreProg) : List (List ℕ) :=
  match p.active with
  | none => []
  | some (pg, _, _) => [pg]

private def CoreProg.allGroups (p : CoreProg) : List (List ℕ) :=
  p.finished ++ p.activeList ++ p.pending

private def CoreProg.remPoints (gs : GridSpec) (p : CoreProg) : List (List ℕ) :=
  (match p.active with | none => [] | some (_, _, todo) => todo) ++
    p.pending.flatMap gs.groupPoints

private def CoreProg.wf (gs : GridSpec) (assigned : List (List ℕ)) (p : CoreProg) : Prop :=
  p.allGroups = assigned ∧
  ∀ pg done todo, p.active = some (pg, done, todo) →
    done ≠ [] ∧ done ++ todo = gs.groupPoints pg

private structure ParInv {C : ℕ} (gs : GridSpec) (body : Body) (sc : ℕ → ℤ) (junk : ℕ → ℝ)
    (mInit : Mem) (assign : Fin C → List (List ℕ)) (initScratch : Fin C → List OutSlot)
    (m : Mem) (cores : Fin C → CoreState) (progs : Fin C → CoreProg) : Prop where
  wf : ∀ c, (progs c).wf gs (assign c)
  rem : ∀ c, (cores c).remPoints = (progs c).remPoints gs
  slots_none : ∀ c, (progs c).active = none →
    (cores c).slots = (gs.outSpecs.map fun _ => none) ++ initScratch c
  slots_some : ∀ c pg done todo, (progs c).active = some (pg, done, todo) →
    (cores c).slots = (isoState gs body sc junk mInit done).2
  mem_finished : ∀ c pg b a, pg ∈ (progs c).finished → inOutRegion gs sc pg b a →
    m.read b a = (isoMem gs body sc junk mInit pg).read b a
  mem_active : ∀ c pg done todo b a, (progs c).active = some (pg, done, todo) →
    inOutRegion gs sc pg b a → m.read b a = (isoState gs body sc junk mInit done).1.read b a
  mem_untouched : ∀ b a,
    (∀ c pg, pg ∈ (progs c).finished → ¬ inOutRegion gs sc pg b a) →
    (∀ c pg done todo, (progs c).active = some (pg, done, todo) → ¬ inOutRegion gs sc pg b a) →
    m.read b a = mInit.read b a

private theorem mem_assign_of_perm {C : ℕ} {gs : GridSpec} {assign : Fin C → List (List ℕ)}
    (hpart : ((List.finRange C).flatMap assign).Perm (gridPoints gs.parGrid))
    {c : Fin C} {pg : List ℕ} (h : pg ∈ assign c) :
    pg ∈ gridPoints gs.parGrid :=
  hpart.subset (List.mem_flatMap.mpr ⟨c, List.mem_finRange c, h⟩)

private theorem assign_nodup {C : ℕ} {gs : GridSpec} {assign : Fin C → List (List ℕ)}
    (hpart : ((List.finRange C).flatMap assign).Perm (gridPoints gs.parGrid)) :
    ((List.finRange C).flatMap assign).Nodup :=
  (hpart.nodup_iff).mpr (gridPoints_nodup gs.parGrid)

private theorem core_assign_nodup {C : ℕ} {gs : GridSpec} {assign : Fin C → List (List ℕ)}
    (hpart : ((List.finRange C).flatMap assign).Perm (gridPoints gs.parGrid)) (c : Fin C) :
    (assign c).Nodup :=
  (List.nodup_flatMap.mp (assign_nodup hpart)).1 c (List.mem_finRange c)

private theorem core_assign_disjoint {C : ℕ} {gs : GridSpec} {assign : Fin C → List (List ℕ)}
    (hpart : ((List.finRange C).flatMap assign).Perm (gridPoints gs.parGrid))
    {c₁ c₂ : Fin C} (hne : c₁ ≠ c₂) {pg : List ℕ} (h₁ : pg ∈ assign c₁) (h₂ : pg ∈ assign c₂) :
    False := by
  have hpw := (List.nodup_flatMap.mp (assign_nodup hpart)).2
  have hdisj : ∀ a ∈ assign c₁, a ∉ assign c₂ := by
    have hmem : ∀ l : List (Fin C), l.Pairwise (fun i j => List.Disjoint (assign i) (assign j)) →
        c₁ ∈ l → c₂ ∈ l → c₁ ≠ c₂ → List.Disjoint (assign c₁) (assign c₂) := by
      intro l hl hc₁ hc₂ hne
      induction l with
      | nil => simp at hc₁
      | cons x xs ih =>
        rw [List.pairwise_cons] at hl
        simp only [List.mem_cons] at hc₁ hc₂
        rcases hc₁ with rfl | hc₁ <;> rcases hc₂ with rfl | hc₂
        · exact absurd rfl hne
        · exact hl.1 c₂ hc₂
        · exact fun y hy₁ hy₂ => hl.1 c₁ hc₁ hy₂ hy₁
        · exact ih hl.2 hc₁ hc₂
    exact hmem (List.finRange C) hpw (List.mem_finRange c₁) (List.mem_finRange c₂) hne
  exact hdisj pg h₁ h₂

private theorem group_in_assign_of_finished {C : ℕ} {gs : GridSpec} {assign : Fin C → List (List ℕ)}
    {progs : Fin C → CoreProg} (hwf : ∀ c, (progs c).wf gs (assign c))
    {c : Fin C} {pg : List ℕ} (h : pg ∈ (progs c).finished) : pg ∈ assign c := by
  rw [← (hwf c).1]; simp [CoreProg.allGroups, h]

private theorem group_in_assign_of_active {C : ℕ} {gs : GridSpec} {assign : Fin C → List (List ℕ)}
    {progs : Fin C → CoreProg} (hwf : ∀ c, (progs c).wf gs (assign c))
    {c : Fin C} {pg : List ℕ} {done todo : List (List ℕ)}
    (h : (progs c).active = some (pg, done, todo)) : pg ∈ assign c := by
  rw [← (hwf c).1]; simp [CoreProg.allGroups, CoreProg.activeList, h]

private theorem group_in_assign_of_pending {C : ℕ} {gs : GridSpec} {assign : Fin C → List (List ℕ)}
    {progs : Fin C → CoreProg} (hwf : ∀ c, (progs c).wf gs (assign c))
    {c : Fin C} {pg : List ℕ} (h : pg ∈ (progs c).pending) : pg ∈ assign c := by
  rw [← (hwf c).1]; simp [CoreProg.allGroups, h]

private theorem par_ne_of_different_buckets {C : ℕ} {gs : GridSpec}
    {assign : Fin C → List (List ℕ)}
    (hpart : ((List.finRange C).flatMap assign).Perm (gridPoints gs.parGrid))
    {progs : Fin C → CoreProg} (hwf : ∀ c, (progs c).wf gs (assign c))
    {c : Fin C} :
    (∀ pg₁ ∈ (progs c).finished, ∀ pg₂ ∈ (progs c).activeList, pg₁ ≠ pg₂) ∧
    (∀ pg₁ ∈ (progs c).finished, ∀ pg₂ ∈ (progs c).pending, pg₁ ≠ pg₂) ∧
    (∀ pg₁ ∈ (progs c).activeList, ∀ pg₂ ∈ (progs c).pending, pg₁ ≠ pg₂) := by
  have hnd : ((progs c).finished ++ (progs c).activeList ++ (progs c).pending).Nodup := by
    have := core_assign_nodup hpart c
    rwa [← (hwf c).1] at this
  rw [List.nodup_append] at hnd
  have hnd12 := List.nodup_append.mp hnd.1
  refine ⟨?_, ?_, ?_⟩
  · intro pg₁ h₁ pg₂ h₂ heq; subst heq; exact hnd12.2.2 pg₁ h₁ pg₁ h₂ rfl
  · intro pg₁ h₁ pg₂ h₂ heq; subst heq; exact hnd.2.2 pg₁ (List.mem_append_left _ h₁) pg₁ h₂ rfl
  · intro pg₁ h₁ pg₂ h₂ heq; subst heq; exact hnd.2.2 pg₁ (List.mem_append_right _ h₁) pg₁ h₂ rfl

private theorem scratchPrev_length {β : Type*} (f : ℕ → OutSlot × List ℕ → β)
    (gs : GridSpec) (slots : List OutSlot)
    (hlen : slots.length = gs.outSpecs.length + gs.scratch.length) :
    (((slots.drop gs.outSpecs.length).zip gs.scratch).mapIdx f).length = gs.scratch.length := by
  simp [hlen]

private theorem stepGrid_first_step_eq {gs : GridSpec} {body : Body} {m0 : Mem}
    (hwd : gs.WellDefined body m0) (junk : ℕ → ℝ) (mInit m : Mem)
    {pg ag₀ : List ℕ} {restAg : List (List ℕ)}
    (hpg : pg ∈ gridPoints gs.parGrid)
    (harb : gridPoints gs.arbGrid = ag₀ :: restAg)
    {slots : List OutSlot}
    (hslen : slots.length = gs.outSpecs.length + gs.scratch.length)
    (hne : ∀ q o, gs.outSpecs[q]? = some o → ∀ oldIdx oldVal,
      slots.getD q none = some (oldIdx, oldVal) →
      oldIdx ≠ o.2.blockIdx (pg ++ ag₀) (gs.scalarPrefetch m0))
    (hscratch : gs.parGrid ≠ [] ∨ slots.drop gs.outSpecs.length = gs.scratch.map fun _ => none)
    (hin : ∀ b a, inInRegion gs (gs.scalarPrefetch m0) pg b a →
      (flushAll gs.outSpecs slots m).read b a = mInit.read b a) :
    (stepGrid gs body (gs.scalarPrefetch m0) junk (m, slots) (pg ++ ag₀)).1 =
      flushAll gs.outSpecs slots m ∧
    (stepGrid gs body (gs.scalarPrefetch m0) junk (m, slots) (pg ++ ag₀)).2 =
      (isoState gs body (gs.scalarPrefetch m0) junk mInit [pg ++ ag₀]).2 := by
  set sc := gs.scalarPrefetch m0
  set g₀ := pg ++ ag₀
  set scratchNone : List OutSlot := gs.scratch.map fun _ => none
  have hag₀ : ag₀ ∈ gridPoints gs.arbGrid := by rw [harb]; simp
  have hfp := flushAndPrep_of_ne g₀ sc (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) 0))
    0 gs.outSpecs slots scratchNone m mInit hne
  have hfpInit :
      (flushAndPrep g₀ sc (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) 0))
        0 gs.outSpecs ((gs.outSpecs.map fun _ => none) ++ scratchNone) mInit).1 = mInit :=
    flushAndPrep_outNone_fst g₀ sc _ 0 gs.outSpecs scratchNone mInit
  have hins :
      (gs.inSpecs.mapIdx fun i p =>
        loadBlock p.1 p.2.blockShape (p.2.blockIdx g₀ sc)
          (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) (i + 1)))
          (flushAndPrep g₀ sc (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) 0))
            0 gs.outSpecs slots m).1) =
      (gs.inSpecs.mapIdx fun i p =>
        loadBlock p.1 p.2.blockShape (p.2.blockIdx g₀ sc)
          (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) (i + 1)))
          (flushAndPrep g₀ sc (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) 0))
            0 gs.outSpecs ((gs.outSpecs.map fun _ => none) ++ scratchNone) mInit).1) := by
    rw [hfp.1, hfpInit]
    apply List.ext_getElem (by simp)
    intro i h1 _
    have hi : i < gs.inSpecs.length := by simpa using h1
    simp only [List.getElem_mapIdx]
    have hmem : gs.inSpecs[i] ∈ gs.inSpecs := List.getElem_mem _
    have hok := blocks_ok_at_point hwd hpg hag₀ (List.mem_append_left _ hmem)
    have hlen : gs.inSpecs[i].2.blockShape.length = (gs.inSpecs[i].2.blockIdx g₀ sc).length := by
      simp only [BlockSpec.blockIdx, List.length_map]
      exact (blockOk_lengths _ _ _ hok).2
    apply loadBlock_congr_block _ _ _ hlen
    intro a ha
    exact hin _ a ⟨ag₀, hag₀, gs.inSpecs[i], hmem, rfl, ha⟩
  refine ⟨hfp.1, ?_⟩
  simp only [isoState, List.foldl_cons, List.foldl_nil, stepGrid]
  have hfp2_1 : (flushAndPrep g₀ sc (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) 0))
      0 gs.outSpecs slots m).2.1 =
    (flushAndPrep g₀ sc (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) 0))
      0 gs.outSpecs ((gs.outSpecs.map fun _ => none) ++ scratchNone) mInit).2.1 :=
    congrArg Prod.fst hfp.2
  have hfp2_2 : (flushAndPrep g₀ sc (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) 0))
      0 gs.outSpecs slots m).2.2 =
    (flushAndPrep g₀ sc (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) 0))
      0 gs.outSpecs ((gs.outSpecs.map fun _ => none) ++ scratchNone) mInit).2.2 :=
    congrArg Prod.snd hfp.2
  rw [hfp2_1, hfp2_2, hins]
  set outNone : List OutSlot := gs.outSpecs.map fun _ => none
  set insInit := gs.inSpecs.mapIdx fun i p =>
    loadBlock p.1 p.2.blockShape (p.2.blockIdx g₀ sc)
      (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) (i + 1)))
      (flushAndPrep g₀ sc (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) 0))
        0 gs.outSpecs (outNone ++ scratchNone) mInit).1
  set outsInit := (flushAndPrep g₀ sc (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) 0))
    0 gs.outSpecs (outNone ++ scratchNone) mInit).2.2
  set scFn : ℕ → OutSlot × List ℕ → List ℕ → ℝ := fun j p loc =>
    match p.1 with
    | some (_, v) =>
      if inBounds p.2 loc then v loc
      else junkAt (junkAt junk (Nat.pair (flatCoord gs.grid g₀ + 1) 0))
        (gs.outSpecs.length + j) (Encodable.encode loc)
    | none => junkAt (junkAt junk (Nat.pair 0 0))
        (gs.outSpecs.length + j) (Encodable.encode loc)
  have hbody :
      body g₀ sc insInit (outsInit ++ ((slots.drop gs.outSpecs.length).zip gs.scratch).mapIdx scFn) =
      body g₀ sc insInit
        (outsInit ++ (((outNone ++ scratchNone).drop gs.outSpecs.length).zip gs.scratch).mapIdx scFn) := by
    rcases hscratch with hpar | hsc
    · apply hwd.parallel_scratch_reset hpar pg hpg ag₀ (by rw [harb]; rfl)
      · exact flushAndPrep_outs_length g₀ sc _ 0 gs.outSpecs _ mInit
      · exact scratchPrev_length scFn gs slots hslen
      · exact scratchPrev_length scFn gs (outNone ++ scratchNone) (by simp [outNone, scratchNone])
    · have hdrop : (outNone ++ scratchNone).drop gs.outSpecs.length = scratchNone := by
        simp [outNone]
      rw [hsc, hdrop]
  exact congrArg (fun v => makeSlots _ v ++ (v.drop gs.outSpecs.length).map fun w => some ([], w))
    hbody

private theorem stepCore_preserves_inv {C : ℕ} {gs : GridSpec} {body : Body} {m0 : Mem}
    (hwd : gs.WellDefined body m0) (junk : ℕ → ℝ) (mInit : Mem)
    {assign : Fin C → List (List ℕ)}
    (hpart : ((List.finRange C).flatMap assign).Perm (gridPoints gs.parGrid))
    {initScratch : Fin C → List OutSlot}
    (hinitLen : ∀ c, (initScratch c).length = gs.scratch.length)
    (hinitScratch : gs.parGrid ≠ [] ∨ ∀ c, initScratch c = gs.scratch.map fun _ => none)
    {ag₀ : List ℕ} {restAg : List (List ℕ)}
    (harb : gridPoints gs.arbGrid = ag₀ :: restAg)
    {m : Mem} {cores : Fin C → CoreState} {progs : Fin C → CoreProg}
    (hinv : ParInv gs body (gs.scalarPrefetch m0) junk mInit assign initScratch m cores progs)
    (c : Fin C) :
    ∃ progs',
      ParInv gs body (gs.scalarPrefetch m0) junk mInit assign initScratch
        (stepCore gs body (gs.scalarPrefetch m0) junk (m, cores) c).1
        (stepCore gs body (gs.scalarPrefetch m0) junk (m, cores) c).2 progs' := by
  set sc := gs.scalarPrefetch m0
  have hgp_eq : ∀ pg, gs.groupPoints pg = (pg ++ ag₀) :: restAg.map (pg ++ ·) := fun pg => by
    simp [GridSpec.groupPoints, harb]
  rcases hact : (progs c).active with _ | ⟨pg, done, todo⟩
  · -- Core `c` has no active group yet.
    rcases hpend : (progs c).pending with _ | ⟨pgNew, pend'⟩
    · -- No pending groups either: core `c` is a no-op.
      have hrem : (cores c).remPoints = [] := by
        rw [hinv.rem c, CoreProg.remPoints, hact, hpend]; rfl
      refine ⟨progs, ?_⟩
      simpa [stepCore, hrem] using hinv
    · -- Case 3: core `c` starts its very first group `pgNew` at `pgNew ++ ag₀`.
      have hrem : (cores c).remPoints =
          (pgNew ++ ag₀) :: (restAg.map (pgNew ++ ·) ++ pend'.flatMap gs.groupPoints) := by
        rw [hinv.rem c, CoreProg.remPoints, hact, hpend]
        simp [hgp_eq]
      have hslots : (cores c).slots = (gs.outSpecs.map fun _ => none) ++ initScratch c :=
        hinv.slots_none c hact
      have hpgNew_assign : pgNew ∈ assign c :=
        group_in_assign_of_pending hinv.wf (by rw [hpend]; simp)
      have hpgNew : pgNew ∈ gridPoints gs.parGrid := mem_assign_of_perm hpart hpgNew_assign
      have hpgNew_not_fin : ∀ c' pg', pg' ∈ (progs c').finished → pg' ≠ pgNew := by
        intro c' pg' hfin heq
        subst heq
        by_cases hc : c' = c
        · subst hc
          exact (par_ne_of_different_buckets hpart hinv.wf).2.1 pg' hfin pg'
            (by rw [hpend]; simp) rfl
        · exact core_assign_disjoint hpart hc (group_in_assign_of_finished hinv.wf hfin)
            hpgNew_assign
      have hpgNew_not_act : ∀ c' pg' d' t', (progs c').active = some (pg', d', t') → pg' ≠ pgNew := by
        intro c' pg' d' t' hact' heq
        subst heq
        by_cases hc : c' = c
        · subst hc; rw [hact] at hact'; cases hact'
        · exact core_assign_disjoint hpart hc (group_in_assign_of_active hinv.wf hact')
            hpgNew_assign
      have hflush_id : flushAll gs.outSpecs (cores c).slots m = m := by
        rw [hslots]; exact flushAll_outNone gs.outSpecs (initScratch c) m
      have hm_in : ∀ b a, inInRegion gs sc pgNew b a → m.read b a = mInit.read b a := by
        intro b a hin
        apply hinv.mem_untouched b a
        · intro c' pg' hfin hout
          have hpg' := mem_assign_of_perm hpart (group_in_assign_of_finished hinv.wf hfin)
          exact inOutRegion_disjoint hwd hpg' hpgNew (hpgNew_not_fin c' pg' hfin) ⟨hin, hout⟩
        · intro c' pg' d' t' hact' hout
          have hpg' := mem_assign_of_perm hpart (group_in_assign_of_active hinv.wf hact')
          exact inOutRegion_disjoint hwd hpg' hpgNew (hpgNew_not_act c' pg' d' t' hact') ⟨hin, hout⟩
      have hstep :
          (stepGrid gs body sc junk (m, (cores c).slots) (pgNew ++ ag₀)).1 =
            flushAll gs.outSpecs (cores c).slots m ∧
          (stepGrid gs body sc junk (m, (cores c).slots) (pgNew ++ ag₀)).2 =
            (isoState gs body sc junk mInit [pgNew ++ ag₀]).2 :=
        stepGrid_first_step_eq hwd junk mInit m hpgNew harb (slots := (cores c).slots)
          (by rw [hslots]; simp [hinitLen c])
          (by
            intro q o hq oldIdx oldVal hs
            have hqlt : q < gs.outSpecs.length := by
              by_contra h; rw [List.getElem?_eq_none (by omega)] at hq; cases hq
            rw [hslots, List.getD_append _ _ _ _ (by simp [hqlt])] at hs
            simp [List.getD_eq_getElem?_getD, hqlt] at hs)
          (by
            rcases hinitScratch with hpar | hsc
            · exact Or.inl hpar
            · right; rw [hslots, List.drop_append_of_le_length (by simp)]; simp [hsc c])
          (by intro b a hin; rw [hflush_id]; exact hm_in b a hin)
      let pNew : CoreProg :=
        { finished := (progs c).finished
          active := some (pgNew, [pgNew ++ ag₀], restAg.map (pgNew ++ ·))
          pending := pend' }
      refine ⟨Function.update progs c pNew, ?_⟩
      simp only [stepCore, hrem, hstep.1, hflush_id, hstep.2]
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · intro c'
        by_cases hc : c' = c
        · subst c'
          simp only [Function.update_self, CoreProg.wf, CoreProg.allGroups, CoreProg.activeList,
            pNew]
          refine ⟨?_, ?_⟩
          · have hwf := (hinv.wf c).1
            simp only [CoreProg.allGroups, CoreProg.activeList, hact, hpend, List.append_nil,
              List.append_assoc, List.singleton_append] at hwf ⊢
            exact hwf
          · intro pg' d' t' h
            obtain ⟨h1, h2, h3⟩ :
                pg' = pgNew ∧ d' = [pgNew ++ ag₀] ∧ t' = restAg.map (pgNew ++ ·) := by
              simpa [Option.some.injEq, Prod.mk.injEq] using h.symm
            subst pg' d' t'
            exact ⟨by simp, by simp [hgp_eq]⟩
        · simpa [Function.update_of_ne hc] using hinv.wf c'
      · intro c'
        by_cases hc : c' = c
        · subst c'; simp [Function.update_self, CoreProg.remPoints, pNew]
        · simpa [Function.update_of_ne hc] using hinv.rem c'
      · intro c' hnone
        by_cases hc : c' = c
        · subst c'; simp [Function.update_self, pNew] at hnone
        · rw [Function.update_of_ne hc] at hnone ⊢
          exact hinv.slots_none c' hnone
      · intro c' pg' d' t' hsome
        by_cases hc : c' = c
        · subst c'
          simp only [Function.update_self, pNew] at hsome ⊢
          obtain ⟨h1, h2, h3⟩ :
              pg' = pgNew ∧ d' = [pgNew ++ ag₀] ∧ t' = restAg.map (pgNew ++ ·) := by
            simpa [Option.some.injEq, Prod.mk.injEq] using hsome.symm
          subst pg' d' t'
          rfl
        · rw [Function.update_of_ne hc] at hsome ⊢
          exact hinv.slots_some c' pg' d' t' hsome
      · intro c' pg' b a hfin hout
        by_cases hc : c' = c
        · subst c'
          simp only [Function.update_self, pNew] at hfin
          exact hinv.mem_finished c pg' b a hfin hout
        · rw [Function.update_of_ne hc] at hfin
          exact hinv.mem_finished c' pg' b a hfin hout
      · intro c' pg' d' t' b a hsome hout
        by_cases hc : c' = c
        · subst c'
          simp only [Function.update_self, pNew] at hsome
          obtain ⟨h1, h2, h3⟩ :
              pg' = pgNew ∧ d' = [pgNew ++ ag₀] ∧ t' = restAg.map (pgNew ++ ·) := by
            simpa [Option.some.injEq, Prod.mk.injEq] using hsome.symm
          subst pg' d' t'
          have hiso1 : (isoState gs body sc junk mInit [pgNew ++ ag₀]).1 = mInit := by
            simp only [isoState, List.foldl_cons, List.foldl_nil, stepGrid]
            exact flushAndPrep_outNone_fst _ _ _ 0 _ _ mInit
          rw [hiso1]
          apply hinv.mem_untouched b a
          · intro c'' pg'' hfin' hout'
            have hpg'' := mem_assign_of_perm hpart (group_in_assign_of_finished hinv.wf hfin')
            exact outRegion_disjoint hwd hpg'' hpgNew (hpgNew_not_fin c'' pg'' hfin') ⟨hout', hout⟩
          · intro c'' pg'' d'' t'' hact'' hout'
            have hpg'' := mem_assign_of_perm hpart (group_in_assign_of_active hinv.wf hact'')
            exact outRegion_disjoint hwd hpg'' hpgNew (hpgNew_not_act c'' pg'' d'' t'' hact'')
              ⟨hout', hout⟩
        · rw [Function.update_of_ne hc] at hsome
          exact hinv.mem_active c' pg' d' t' b a hsome hout
      · intro b a hnofin hnoact
        apply hinv.mem_untouched b a
        · intro c' pg' hfin
          by_cases hc : c' = c
          · subst c'
            exact hnofin c pg' (by simpa [Function.update_self, pNew] using hfin)
          · exact hnofin c' pg' (by simpa [Function.update_of_ne hc] using hfin)
        · intro c' pg' d' t' hact'
          by_cases hc : c' = c
          · subst c'; rw [hact] at hact'; cases hact'
          · exact hnoact c' pg' d' t' (by simpa [Function.update_of_ne hc] using hact')
  · -- Core `c` has an active group `(pg, done, todo)`.
    have ⟨hdone_ne, hdone_todo⟩ := (hinv.wf c).2 pg done todo hact
    have hpg_assign : pg ∈ assign c := group_in_assign_of_active hinv.wf hact
    have hpg : pg ∈ gridPoints gs.parGrid := mem_assign_of_perm hpart hpg_assign
    have hdone_mem : ∀ g ∈ done, ∃ ag ∈ gridPoints gs.arbGrid, g = pg ++ ag := by
      intro g hg
      have hmem : g ∈ gs.groupPoints pg := by rw [← hdone_todo]; exact List.mem_append_left _ hg
      simp only [GridSpec.groupPoints, List.mem_map] at hmem
      obtain ⟨ag, hag, rfl⟩ := hmem
      exact ⟨ag, hag, rfl⟩
    obtain ⟨hisoLen, hisoSlots, hisoOut⟩ := isoState_props hwd junk mInit pg done hdone_mem
    have hslots : (cores c).slots = (isoState gs body sc junk mInit done).2 :=
      hinv.slots_some c pg done todo hact
    have hpg_not_fin : ∀ c' pg', pg' ∈ (progs c').finished → pg' ≠ pg := by
      intro c' pg' hfin heq
      subst heq
      by_cases hc : c' = c
      · subst hc
        exact (par_ne_of_different_buckets hpart hinv.wf).1 pg' hfin pg'
          (by simp [CoreProg.activeList, hact]) rfl
      · exact core_assign_disjoint hpart hc (group_in_assign_of_finished hinv.wf hfin) hpg_assign
    have hpg_not_other_act : ∀ c' pg' d' t', c' ≠ c →
        (progs c').active = some (pg', d', t') → pg' ≠ pg := by
      intro c' pg' d' t' hc hact' heq
      subst heq
      exact core_assign_disjoint hpart hc (group_in_assign_of_active hinv.wf hact') hpg_assign
    cases todo with
    | cons g todo' =>
      -- Case 1: continue active group `pg` with its next step `g`.
      have hg_mem : g ∈ gs.groupPoints pg := by rw [← hdone_todo]; simp
      simp only [GridSpec.groupPoints, List.mem_map] at hg_mem
      obtain ⟨ag, hag, rfl⟩ := hg_mem
      have hrem : (cores c).remPoints = (pg ++ ag) :: (todo' ++ (progs c).pending.flatMap gs.groupPoints) := by
        rw [hinv.rem c, CoreProg.remPoints, hact]; rfl
      have hm_in : ∀ b a, inInRegion gs sc pg b a →
          m.read b a = (isoState gs body sc junk mInit done).1.read b a := by
        intro b a hin
        by_cases hout : inOutRegion gs sc pg b a
        · exact hinv.mem_active c pg done ((pg ++ ag) :: todo') b a hact hout
        · rw [hisoOut b a hout]
          apply hinv.mem_untouched b a
          · intro c' pg' hfin hout'
            have hpg' := mem_assign_of_perm hpart (group_in_assign_of_finished hinv.wf hfin)
            exact inOutRegion_disjoint hwd hpg' hpg (hpg_not_fin c' pg' hfin) ⟨hin, hout'⟩
          · intro c' pg' d' t' hact' hout'
            by_cases hc : c' = c
            · subst c'
              rw [hact] at hact'
              obtain ⟨h1, _, _⟩ : pg' = pg ∧ d' = done ∧ t' = (pg ++ ag) :: todo' := by
                simpa [Option.some.injEq, Prod.mk.injEq] using hact'.symm
              subst pg'
              exact hout hout'
            · have hpg' := mem_assign_of_perm hpart (group_in_assign_of_active hinv.wf hact')
              exact inOutRegion_disjoint hwd hpg' hpg (hpg_not_other_act c' pg' d' t' hc hact')
                ⟨hin, hout'⟩
      have hcongr := stepGrid_congr_in_region hwd junk hpg hag
        (slots := (isoState gs body sc junk mInit done).2) hisoSlots hm_in
      let pNew : CoreProg :=
        { finished := (progs c).finished
          active := some (pg, done ++ [pg ++ ag], todo')
          pending := (progs c).pending }
      refine ⟨Function.update progs c pNew, ?_⟩
      simp only [stepCore, hrem, hslots]
      have hisoNext : isoState gs body sc junk mInit (done ++ [pg ++ ag]) =
          stepGrid gs body sc junk (isoState gs body sc junk mInit done) (pg ++ ag) := by
        simp [isoState, List.foldl_append]
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · intro c'
        by_cases hc : c' = c
        · subst c'
          simp only [Function.update_self, CoreProg.wf, CoreProg.allGroups, CoreProg.activeList,
            pNew]
          refine ⟨?_, ?_⟩
          · have hwf := (hinv.wf c).1
            simp only [CoreProg.allGroups, CoreProg.activeList, hact] at hwf
            exact hwf
          · intro pg' d' t' h
            obtain ⟨h1, h2, h3⟩ : pg' = pg ∧ d' = done ++ [pg ++ ag] ∧ t' = todo' := by
              simpa [Option.some.injEq, Prod.mk.injEq] using h.symm
            subst pg' d' t'
            exact ⟨by simp, by simpa [List.append_assoc] using hdone_todo⟩
        · simpa [Function.update_of_ne hc] using hinv.wf c'
      · intro c'
        by_cases hc : c' = c
        · subst c'; simp [Function.update_self, CoreProg.remPoints, pNew]
        · simpa [Function.update_of_ne hc] using hinv.rem c'
      · intro c' hnone
        by_cases hc : c' = c
        · subst c'; simp [Function.update_self, pNew] at hnone
        · rw [Function.update_of_ne hc] at hnone ⊢
          exact hinv.slots_none c' hnone
      · intro c' pg' d' t' hsome
        by_cases hc : c' = c
        · subst c'
          simp only [Function.update_self, pNew] at hsome ⊢
          obtain ⟨h1, h2, h3⟩ : pg' = pg ∧ d' = done ++ [pg ++ ag] ∧ t' = todo' := by
            simpa [Option.some.injEq, Prod.mk.injEq] using hsome.symm
          subst pg' d' t'
          rw [hisoNext]
          exact hcongr.1
        · rw [Function.update_of_ne hc] at hsome ⊢
          exact hinv.slots_some c' pg' d' t' hsome
      · intro c' pg' b a hfin hout
        have hfin' : pg' ∈ (progs c').finished := by
          by_cases hc : c' = c
          · subst c'; simpa [Function.update_self, pNew] using hfin
          · simpa [Function.update_of_ne hc] using hfin
        have hpg' := mem_assign_of_perm hpart (group_in_assign_of_finished hinv.wf hfin')
        have hnotpg : ¬ inOutRegion gs sc pg b a :=
          fun h => outRegion_disjoint hwd hpg' hpg (hpg_not_fin c' pg' hfin') ⟨hout, h⟩
        rw [hcongr.2.2 b a hnotpg]
        exact hinv.mem_finished c' pg' b a hfin' hout
      · intro c' pg' d' t' b a hsome hout
        by_cases hc : c' = c
        · subst c'
          simp only [Function.update_self, pNew] at hsome
          obtain ⟨h1, h2, h3⟩ : pg' = pg ∧ d' = done ++ [pg ++ ag] ∧ t' = todo' := by
            simpa [Option.some.injEq, Prod.mk.injEq] using hsome.symm
          subst pg' d' t'
          rw [hisoNext]
          exact hcongr.2.1 b a (hinv.mem_active c pg done ((pg ++ ag) :: todo') b a hact hout)
        · rw [Function.update_of_ne hc] at hsome
          have hpg' := mem_assign_of_perm hpart (group_in_assign_of_active hinv.wf hsome)
          have hnotpg : ¬ inOutRegion gs sc pg b a :=
            fun h => outRegion_disjoint hwd hpg' hpg (hpg_not_other_act c' pg' d' t' hc hsome)
              ⟨hout, h⟩
          rw [hcongr.2.2 b a hnotpg]
          exact hinv.mem_active c' pg' d' t' b a hsome hout
      · intro b a hnofin hnoact
        have hnotpg : ¬ inOutRegion gs sc pg b a :=
          hnoact c pg (done ++ [pg ++ ag]) todo' (by simp [Function.update_self, pNew])
        rw [hcongr.2.2 b a hnotpg]
        apply hinv.mem_untouched b a
        · intro c' pg' hfin
          by_cases hc : c' = c
          · subst c'; exact hnofin c pg' (by simpa [Function.update_self, pNew] using hfin)
          · exact hnofin c' pg' (by simpa [Function.update_of_ne hc] using hfin)
        · intro c' pg' d' t' hact'
          by_cases hc : c' = c
          · subst c'
            rw [hact] at hact'
            obtain ⟨h1, _, _⟩ : pg' = pg ∧ d' = done ∧ t' = (pg ++ ag) :: todo' := by
              simpa [Option.some.injEq, Prod.mk.injEq] using hact'.symm
            subst pg'
            exact hnotpg
          · exact hnoact c' pg' d' t' (by simpa [Function.update_of_ne hc] using hact')
    | nil =>
      -- Active group `pg` has finished all its steps (`done = gs.groupPoints pg`).
      have hdone_eq : done = gs.groupPoints pg := by simpa using hdone_todo
      rcases hpend : (progs c).pending with _ | ⟨pgNew, pend'⟩
      · -- No pending groups left: core `c` is a no-op.
        have hrem : (cores c).remPoints = [] := by
          rw [hinv.rem c, CoreProg.remPoints, hact, hpend]; rfl
        refine ⟨progs, ?_⟩
        simpa [stepCore, hrem] using hinv
      · -- Case 2: flush completed `pg` and start `pgNew` at `pgNew ++ ag₀`.
        have hrem : (cores c).remPoints =
            (pgNew ++ ag₀) :: (restAg.map (pgNew ++ ·) ++ pend'.flatMap gs.groupPoints) := by
          rw [hinv.rem c, CoreProg.remPoints, hact, hpend]
          simp [hgp_eq]
        have hpgNew_assign : pgNew ∈ assign c :=
          group_in_assign_of_pending hinv.wf (by rw [hpend]; simp)
        have hpgNew : pgNew ∈ gridPoints gs.parGrid := mem_assign_of_perm hpart hpgNew_assign
        have hpg_ne_new : pg ≠ pgNew :=
          (par_ne_of_different_buckets hpart hinv.wf (c := c)).2.2 pg
            (by simp [CoreProg.activeList, hact]) pgNew (by rw [hpend]; simp)
        have hpar_ne : gs.parGrid ≠ [] := by
          intro hempty
          rw [hempty] at hpg hpgNew
          simp [gridPoints] at hpg hpgNew
          exact hpg_ne_new (hpg.trans hpgNew.symm)
        have hpgNew_not_fin : ∀ c' pg', pg' ∈ (progs c').finished → pg' ≠ pgNew := by
          intro c' pg' hfin heq
          subst heq
          by_cases hc : c' = c
          · subst hc
            exact (par_ne_of_different_buckets hpart hinv.wf).2.1 pg' hfin pg'
              (by rw [hpend]; simp) rfl
          · exact core_assign_disjoint hpart hc (group_in_assign_of_finished hinv.wf hfin)
              hpgNew_assign
        have hpgNew_not_act : ∀ c' pg' d' t', (progs c').active = some (pg', d', t') →
            pg' ≠ pgNew := by
          intro c' pg' d' t' hact' heq
          subst heq
          by_cases hc : c' = c
          · subst hc
            rw [hact] at hact'
            injection hact' with h
            rcases Prod.mk.inj h with ⟨rfl, _⟩
            exact hpg_ne_new rfl
          · exact core_assign_disjoint hpart hc (group_in_assign_of_active hinv.wf hact')
              hpgNew_assign
        have hflush_out : ∀ b a, ¬ inOutRegion gs sc pg b a →
            (flushAll gs.outSpecs (cores c).slots m).read b a = m.read b a := by
          intro b a hnot
          rw [hslots]
          apply flushAll_read_outside b a gs.outSpecs _ m
          intro q o hq oldIdx oldVal hs
          obtain ⟨ag', hag', rfl⟩ := hisoSlots q o hq oldIdx oldVal hs
          by_cases hbuf : b = o.1.buf
          · right
            cases haddr : addrInBlock o.1 o.2.blockShape (o.2.blockIdx (pg ++ ag') sc) a
            · rfl
            · exfalso
              exact hnot ⟨ag', hag', o, List.mem_of_getElem? hq, hbuf, haddr⟩
          · exact Or.inl hbuf
        have hstep :
            (stepGrid gs body sc junk (m, (cores c).slots) (pgNew ++ ag₀)).1 =
              flushAll gs.outSpecs (cores c).slots m ∧
            (stepGrid gs body sc junk (m, (cores c).slots) (pgNew ++ ag₀)).2 =
              (isoState gs body sc junk mInit [pgNew ++ ag₀]).2 :=
          stepGrid_first_step_eq hwd junk mInit m hpgNew harb (slots := (cores c).slots)
            (by rw [hslots]; exact hisoLen)
            (by
              intro q o hq oldIdx oldVal hs
              rw [hslots] at hs
              obtain ⟨ag', hag', rfl⟩ := hisoSlots q o hq oldIdx oldVal hs
              exact blockIdx_ne_of_par_ne hwd hpg hpgNew hpg_ne_new hag'
                (by rw [harb]; simp) (List.mem_of_getElem? hq))
            (Or.inl hpar_ne)
            (by
              intro b a hin
              have hnotpg : ¬ inOutRegion gs sc pg b a :=
                fun hout => inOutRegion_disjoint hwd hpg hpgNew hpg_ne_new ⟨hin, hout⟩
              rw [hflush_out b a hnotpg]
              apply hinv.mem_untouched b a
              · intro c' pg' hfin hout'
                have hpg' := mem_assign_of_perm hpart (group_in_assign_of_finished hinv.wf hfin)
                exact inOutRegion_disjoint hwd hpg' hpgNew (hpgNew_not_fin c' pg' hfin) ⟨hin, hout'⟩
              · intro c' pg' d' t' hact' hout'
                have hpg' := mem_assign_of_perm hpart (group_in_assign_of_active hinv.wf hact')
                exact inOutRegion_disjoint hwd hpg' hpgNew (hpgNew_not_act c' pg' d' t' hact')
                  ⟨hin, hout'⟩)
        let pNew : CoreProg :=
          { finished := (progs c).finished ++ [pg]
            active := some (pgNew, [pgNew ++ ag₀], restAg.map (pgNew ++ ·))
            pending := pend' }
        refine ⟨Function.update progs c pNew, ?_⟩
        simp only [stepCore, hrem, hstep.1, hstep.2]
        refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
        · intro c'
          by_cases hc : c' = c
          · subst c'
            simp only [Function.update_self, CoreProg.wf, CoreProg.allGroups, CoreProg.activeList,
              pNew]
            refine ⟨?_, ?_⟩
            · have hwf := (hinv.wf c).1
              simp only [CoreProg.allGroups, CoreProg.activeList, hact, hpend] at hwf
              simpa [List.append_assoc] using hwf
            · intro pg' d' t' h
              obtain ⟨h1, h2, h3⟩ :
                  pg' = pgNew ∧ d' = [pgNew ++ ag₀] ∧ t' = restAg.map (pgNew ++ ·) := by
                simpa [Option.some.injEq, Prod.mk.injEq] using h.symm
              subst pg' d' t'
              exact ⟨by simp, by simp [hgp_eq]⟩
          · simpa [Function.update_of_ne hc] using hinv.wf c'
        · intro c'
          by_cases hc : c' = c
          · subst c'; simp [Function.update_self, CoreProg.remPoints, pNew]
          · simpa [Function.update_of_ne hc] using hinv.rem c'
        · intro c' hnone
          by_cases hc : c' = c
          · subst c'; simp [Function.update_self, pNew] at hnone
          · rw [Function.update_of_ne hc] at hnone ⊢
            exact hinv.slots_none c' hnone
        · intro c' pg' d' t' hsome
          by_cases hc : c' = c
          · subst c'
            simp only [Function.update_self, pNew] at hsome ⊢
            obtain ⟨h1, h2, h3⟩ :
                pg' = pgNew ∧ d' = [pgNew ++ ag₀] ∧ t' = restAg.map (pgNew ++ ·) := by
              simpa [Option.some.injEq, Prod.mk.injEq] using hsome.symm
            subst pg' d' t'
            rfl
          · rw [Function.update_of_ne hc] at hsome ⊢
            exact hinv.slots_some c' pg' d' t' hsome
        · intro c' pg' b a hfin hout
          by_cases hc : c' = c
          · subst c'
            simp only [Function.update_self, pNew, List.mem_append, List.mem_singleton] at hfin
            rcases hfin with hfinOld | hpgEq
            · have hpg' := mem_assign_of_perm hpart (group_in_assign_of_finished hinv.wf hfinOld)
              have hnotpg : ¬ inOutRegion gs sc pg b a :=
                fun h => outRegion_disjoint hwd hpg' hpg (hpg_not_fin c pg' hfinOld) ⟨hout, h⟩
              rw [hflush_out b a hnotpg]
              exact hinv.mem_finished c pg' b a hfinOld hout
            · subst pg'
              rw [hslots, isoMem]
              rw [← hdone_eq]
              exact flushAll_read_congr b a gs.outSpecs _ m _
                (hinv.mem_active c pg done [] b a hact hout)
          · rw [Function.update_of_ne hc] at hfin
            have hpg' := mem_assign_of_perm hpart (group_in_assign_of_finished hinv.wf hfin)
            have hnotpg : ¬ inOutRegion gs sc pg b a :=
              fun h => outRegion_disjoint hwd hpg' hpg (hpg_not_fin c' pg' hfin) ⟨hout, h⟩
            rw [hflush_out b a hnotpg]
            exact hinv.mem_finished c' pg' b a hfin hout
        · intro c' pg' d' t' b a hsome hout
          by_cases hc : c' = c
          · subst c'
            simp only [Function.update_self, pNew] at hsome
            obtain ⟨h1, h2, h3⟩ :
                pg' = pgNew ∧ d' = [pgNew ++ ag₀] ∧ t' = restAg.map (pgNew ++ ·) := by
              simpa [Option.some.injEq, Prod.mk.injEq] using hsome.symm
            subst pg' d' t'
            have hnotpg : ¬ inOutRegion gs sc pg b a :=
              fun h => outRegion_disjoint hwd hpg hpgNew hpg_ne_new ⟨h, hout⟩
            rw [hflush_out b a hnotpg]
            have hiso1 : (isoState gs body sc junk mInit [pgNew ++ ag₀]).1 = mInit := by
              simp only [isoState, List.foldl_cons, List.foldl_nil, stepGrid]
              exact flushAndPrep_outNone_fst _ _ _ 0 _ _ mInit
            rw [hiso1]
            apply hinv.mem_untouched b a
            · intro c'' pg'' hfin' hout'
              have hpg'' := mem_assign_of_perm hpart (group_in_assign_of_finished hinv.wf hfin')
              exact outRegion_disjoint hwd hpg'' hpgNew (hpgNew_not_fin c'' pg'' hfin')
                ⟨hout', hout⟩
            · intro c'' pg'' d'' t'' hact'' hout'
              have hpg'' := mem_assign_of_perm hpart (group_in_assign_of_active hinv.wf hact'')
              exact outRegion_disjoint hwd hpg'' hpgNew (hpgNew_not_act c'' pg'' d'' t'' hact'')
                ⟨hout', hout⟩
          · rw [Function.update_of_ne hc] at hsome
            have hpg' := mem_assign_of_perm hpart (group_in_assign_of_active hinv.wf hsome)
            have hnotpg : ¬ inOutRegion gs sc pg b a :=
              fun h => outRegion_disjoint hwd hpg' hpg (hpg_not_other_act c' pg' d' t' hc hsome)
                ⟨hout, h⟩
            rw [hflush_out b a hnotpg]
            exact hinv.mem_active c' pg' d' t' b a hsome hout
        · intro b a hnofin hnoact
          have hnotpg : ¬ inOutRegion gs sc pg b a :=
            hnofin c pg (by simp [Function.update_self, pNew])
          rw [hflush_out b a hnotpg]
          apply hinv.mem_untouched b a
          · intro c' pg' hfin
            by_cases hc : c' = c
            · subst c'; exact hnofin c pg' (by simp [Function.update_self, pNew, hfin])
            · exact hnofin c' pg' (by simpa [Function.update_of_ne hc] using hfin)
          · intro c' pg' d' t' hact'
            by_cases hc : c' = c
            · subst c'
              rw [hact] at hact'
              obtain ⟨h1, _, _⟩ : pg' = pg ∧ d' = done ∧ t' = [] := by
                simpa [Option.some.injEq, Prod.mk.injEq] using hact'.symm
              subst pg'
              exact hnotpg
            · exact hnoact c' pg' d' t' (by simpa [Function.update_of_ne hc] using hact')

private theorem foldl_stepCore_remPoints {C : ℕ} (gs : GridSpec) (body : Body) (sc : ℕ → ℤ)
    (junk : ℕ → ℝ) :
    ∀ (trace : List (Fin C)) (m : Mem) (cores : Fin C → CoreState) (c : Fin C),
      ((trace.foldl (stepCore gs body sc junk) (m, cores)).2 c).remPoints =
        (cores c).remPoints.drop (trace.count c)
  | [], _, _, _ => by simp
  | c' :: rest, m, cores, c => by
    simp only [List.foldl_cons]
    rw [foldl_stepCore_remPoints gs body sc junk rest]
    by_cases hc : c = c'
    · subst hc
      simp only [List.count_cons_self]
      cases hrem : (cores c).remPoints with
      | nil => simp [stepCore, hrem]
      | cons g r => simp [stepCore, hrem, Function.update_self]
    · simp only [List.count_cons_of_ne (Ne.symm hc)]
      cases hrem : (cores c').remPoints with
      | nil => simp [stepCore, hrem]
      | cons g r => simp [stepCore, hrem, Function.update_of_ne hc]

private theorem foldl_stepCore_preserves_inv {C : ℕ} {gs : GridSpec} {body : Body} {m0 : Mem}
    (hwd : gs.WellDefined body m0) (junk : ℕ → ℝ) (mInit : Mem)
    {assign : Fin C → List (List ℕ)}
    (hpart : ((List.finRange C).flatMap assign).Perm (gridPoints gs.parGrid))
    {initScratch : Fin C → List OutSlot}
    (hinitLen : ∀ c, (initScratch c).length = gs.scratch.length)
    (hinitScratch : gs.parGrid ≠ [] ∨ ∀ c, initScratch c = gs.scratch.map fun _ => none)
    {ag₀ : List ℕ} {restAg : List (List ℕ)}
    (harb : gridPoints gs.arbGrid = ag₀ :: restAg) :
    ∀ (trace : List (Fin C)) (m : Mem) (cores : Fin C → CoreState) (progs : Fin C → CoreProg),
      ParInv gs body (gs.scalarPrefetch m0) junk mInit assign initScratch m cores progs →
      ∃ progs',
        ParInv gs body (gs.scalarPrefetch m0) junk mInit assign initScratch
          (trace.foldl (stepCore gs body (gs.scalarPrefetch m0) junk) (m, cores)).1
          (trace.foldl (stepCore gs body (gs.scalarPrefetch m0) junk) (m, cores)).2 progs'
  | [], _, _, progs, hinv => ⟨progs, hinv⟩
  | c :: rest, m, cores, progs, hinv => by
    obtain ⟨progs₁, hinv₁⟩ :=
      stepCore_preserves_inv hwd junk mInit hpart hinitLen hinitScratch harb hinv c
    exact foldl_stepCore_preserves_inv hwd junk mInit hpart hinitLen hinitScratch harb rest _ _
      progs₁ hinv₁

private theorem foldl_flushAll_cores {C : ℕ} {gs : GridSpec} {body : Body} {m0 : Mem}
    (hwd : gs.WellDefined body m0) (junk : ℕ → ℝ) (mInit : Mem)
    {assign : Fin C → List (List ℕ)}
    (hpart : ((List.finRange C).flatMap assign).Perm (gridPoints gs.parGrid))
    {initScratch : Fin C → List OutSlot}
    {cores : Fin C → CoreState} {progs : Fin C → CoreProg}
    (hdoneAll : ∀ c pg done todo, (progs c).active = some (pg, done, todo) →
      done = gs.groupPoints pg ∧ todo = []) :
    ∀ (toFlush : List (Fin C)) (flushed : List (Fin C)) (m : Mem),
      (∀ c, c ∈ flushed → ∀ pg done todo, (progs c).active = some (pg, done, todo) →
        ∀ b a, inOutRegion gs (gs.scalarPrefetch m0) pg b a →
          m.read b a = (isoMem gs body (gs.scalarPrefetch m0) junk mInit pg).read b a) →
      (∀ c, c ∉ flushed → ∀ pg done todo, (progs c).active = some (pg, done, todo) →
        ∀ b a, inOutRegion gs (gs.scalarPrefetch m0) pg b a →
          m.read b a = (isoState gs body (gs.scalarPrefetch m0) junk mInit done).1.read b a) →
      (∀ c pg b a, pg ∈ (progs c).finished → inOutRegion gs (gs.scalarPrefetch m0) pg b a →
        m.read b a = (isoMem gs body (gs.scalarPrefetch m0) junk mInit pg).read b a) →
      (∀ b a,
        (∀ c pg, pg ∈ (progs c).finished → ¬ inOutRegion gs (gs.scalarPrefetch m0) pg b a) →
        (∀ c pg done todo, (progs c).active = some (pg, done, todo) →
          ¬ inOutRegion gs (gs.scalarPrefetch m0) pg b a) →
        m.read b a = mInit.read b a) →
      (∀ c, (progs c).wf gs (assign c)) →
      (∀ c, (progs c).active = none →
        (cores c).slots = (gs.outSpecs.map fun _ => none) ++ initScratch c) →
      (∀ c pg done todo, (progs c).active = some (pg, done, todo) →
        (cores c).slots = (isoState gs body (gs.scalarPrefetch m0) junk mInit done).2) →
      (∀ c, c ∈ toFlush → c ∉ flushed) →
      toFlush.Nodup →
      let m' := toFlush.foldl (fun m c => flushAll gs.outSpecs (cores c).slots m) m
      (∀ c, c ∈ flushed ++ toFlush → ∀ pg done todo, (progs c).active = some (pg, done, todo) →
        ∀ b a, inOutRegion gs (gs.scalarPrefetch m0) pg b a →
          m'.read b a = (isoMem gs body (gs.scalarPrefetch m0) junk mInit pg).read b a) ∧
      (∀ c pg b a, pg ∈ (progs c).finished → inOutRegion gs (gs.scalarPrefetch m0) pg b a →
        m'.read b a = (isoMem gs body (gs.scalarPrefetch m0) junk mInit pg).read b a) ∧
      (∀ b a,
        (∀ c pg, pg ∈ (progs c).finished → ¬ inOutRegion gs (gs.scalarPrefetch m0) pg b a) →
        (∀ c pg done todo, (progs c).active = some (pg, done, todo) →
          ¬ inOutRegion gs (gs.scalarPrefetch m0) pg b a) →
        m'.read b a = mInit.read b a)
  | [], flushed, m, hfl, _, hfin, hunt, _, _, _, _, _ => by
    simpa using ⟨hfl, hfin, hunt⟩
  | c :: rest, flushed, m, hfl, hunfl, hfin, hunt, hwf, hnone, hsome, hdisj, hnd => by
    set sc := gs.scalarPrefetch m0
    rw [List.nodup_cons] at hnd
    have hc_not_fl : c ∉ flushed := hdisj c (by simp)
    have hstep :
        (∀ c', c' ∈ c :: flushed → ∀ pg done todo, (progs c').active = some (pg, done, todo) →
          ∀ b a, inOutRegion gs sc pg b a →
            (flushAll gs.outSpecs (cores c).slots m).read b a =
              (isoMem gs body sc junk mInit pg).read b a) ∧
        (∀ c', c' ∉ c :: flushed → ∀ pg done todo, (progs c').active = some (pg, done, todo) →
          ∀ b a, inOutRegion gs sc pg b a →
            (flushAll gs.outSpecs (cores c).slots m).read b a =
              (isoState gs body sc junk mInit done).1.read b a) ∧
        (∀ c' pg b a, pg ∈ (progs c').finished → inOutRegion gs sc pg b a →
          (flushAll gs.outSpecs (cores c).slots m).read b a =
            (isoMem gs body sc junk mInit pg).read b a) ∧
        (∀ b a,
          (∀ c' pg, pg ∈ (progs c').finished → ¬ inOutRegion gs sc pg b a) →
          (∀ c' pg done todo, (progs c').active = some (pg, done, todo) →
            ¬ inOutRegion gs sc pg b a) →
          (flushAll gs.outSpecs (cores c).slots m).read b a = mInit.read b a) := by
      rcases hact : (progs c).active with _ | ⟨pgc, donec, todoc⟩
      · have hid : flushAll gs.outSpecs (cores c).slots m = m := by
          rw [hnone c hact]; exact flushAll_outNone gs.outSpecs (initScratch c) m
        rw [hid]
        refine ⟨?_, ?_, hfin, hunt⟩
        · intro c' hc' pg done todo hact' b a hout
          simp only [List.mem_cons] at hc'
          rcases hc' with rfl | hc'
          · rw [hact] at hact'; cases hact'
          · exact hfl c' hc' pg done todo hact' b a hout
        · intro c' hc' pg done todo hact' b a hout
          exact hunfl c' (fun h => hc' (by simp [h])) pg done todo hact' b a hout
      · obtain ⟨hdone_eq, _⟩ := hdoneAll c pgc donec todoc hact
        have hpgc_assign : pgc ∈ assign c := group_in_assign_of_active hwf hact
        have hpgc : pgc ∈ gridPoints gs.parGrid := mem_assign_of_perm hpart hpgc_assign
        have hdone_mem : ∀ g ∈ donec, ∃ ag ∈ gridPoints gs.arbGrid, g = pgc ++ ag := by
          intro g hg
          rw [hdone_eq] at hg
          simp only [GridSpec.groupPoints, List.mem_map] at hg
          obtain ⟨ag, hag, rfl⟩ := hg
          exact ⟨ag, hag, rfl⟩
        obtain ⟨_, hisoSlots, _⟩ := isoState_props hwd junk mInit pgc donec hdone_mem
        have hflush_out : ∀ b a, ¬ inOutRegion gs sc pgc b a →
            (flushAll gs.outSpecs (cores c).slots m).read b a = m.read b a := by
          intro b a hnot
          rw [hsome c pgc donec todoc hact]
          apply flushAll_read_outside b a gs.outSpecs _ m
          intro q o hq oldIdx oldVal hs
          obtain ⟨ag', hag', rfl⟩ := hisoSlots q o hq oldIdx oldVal hs
          by_cases hbuf : b = o.1.buf
          · right
            cases haddr : addrInBlock o.1 o.2.blockShape (o.2.blockIdx (pgc ++ ag') sc) a
            · rfl
            · exfalso
              exact hnot ⟨ag', hag', o, List.mem_of_getElem? hq, hbuf, haddr⟩
          · exact Or.inl hbuf
        refine ⟨?_, ?_, ?_, ?_⟩
        · intro c' hc' pg done todo hact' b a hout
          simp only [List.mem_cons] at hc'
          rcases hc' with hcc | hc'
          · subst c'
            rw [hact] at hact'
            obtain ⟨h1, h2, h3⟩ : pg = pgc ∧ done = donec ∧ todo = todoc := by
              simpa [Option.some.injEq, Prod.mk.injEq] using hact'.symm
            subst pg done todo
            rw [hsome c pgc donec todoc hact, isoMem, ← hdone_eq]
            exact flushAll_read_congr b a gs.outSpecs _ m _
              (hunfl c hc_not_fl pgc donec todoc hact b a hout)
          · have hcne : c' ≠ c := fun heq => hc_not_fl (heq ▸ hc')
            have hpg' := mem_assign_of_perm hpart (group_in_assign_of_active hwf hact')
            have hpgne : pg ≠ pgc := fun heq =>
              core_assign_disjoint hpart hcne (heq ▸ group_in_assign_of_active hwf hact')
                hpgc_assign
            have hnotpgc : ¬ inOutRegion gs sc pgc b a :=
              fun h => outRegion_disjoint hwd hpg' hpgc hpgne ⟨hout, h⟩
            rw [hflush_out b a hnotpgc]
            exact hfl c' hc' pg done todo hact' b a hout
        · intro c' hc' pg done todo hact' b a hout
          have hcne : c' ≠ c := fun heq => hc' (by simp [heq])
          have hpg' := mem_assign_of_perm hpart (group_in_assign_of_active hwf hact')
          have hpgne : pg ≠ pgc := fun heq =>
            core_assign_disjoint hpart hcne (heq ▸ group_in_assign_of_active hwf hact')
              hpgc_assign
          have hnotpgc : ¬ inOutRegion gs sc pgc b a :=
            fun h => outRegion_disjoint hwd hpg' hpgc hpgne ⟨hout, h⟩
          rw [hflush_out b a hnotpgc]
          exact hunfl c' (fun h => hc' (by simp [h])) pg done todo hact' b a hout
        · intro c' pg b a hfin' hout
          have hpg' := mem_assign_of_perm hpart (group_in_assign_of_finished hwf hfin')
          have hpgne : pg ≠ pgc := by
            intro heq; subst heq
            by_cases hc : c' = c
            · subst hc
              exact (par_ne_of_different_buckets hpart hwf).1 pg hfin' pg
                (by simp [CoreProg.activeList, hact]) rfl
            · exact core_assign_disjoint hpart hc (group_in_assign_of_finished hwf hfin')
                hpgc_assign
          have hnotpgc : ¬ inOutRegion gs sc pgc b a :=
            fun h => outRegion_disjoint hwd hpg' hpgc hpgne ⟨hout, h⟩
          rw [hflush_out b a hnotpgc]
          exact hfin c' pg b a hfin' hout
        · intro b a hnofin hnoact
          rw [hflush_out b a (hnoact c pgc donec todoc hact)]
          exact hunt b a hnofin hnoact
    obtain ⟨h1, h2, h3⟩ := foldl_flushAll_cores hwd junk mInit hpart hdoneAll rest (c :: flushed)
      (flushAll gs.outSpecs (cores c).slots m) hstep.1 hstep.2.1 hstep.2.2.1 hstep.2.2.2
      hwf hnone hsome
      (fun c' hc' => by
        simp only [List.mem_cons, not_or]
        exact ⟨fun heq => hnd.1 (heq ▸ hc'), hdisj c' (by simp [hc'])⟩)
      hnd.2
    refine ⟨?_, h2, h3⟩
    intro c' hc' pg done todo hact' b a hout
    apply h1 c' _ pg done todo hact' b a hout
    simp only [List.mem_append, List.mem_cons] at hc' ⊢
    tauto

/-- Canonical pointwise characterization of any complete parallel execution under `WellDefined`. -/
private theorem pallasExecPar_char {C : ℕ} {gs : GridSpec} {body : Body} {m0 : Mem}
    (hwd : gs.WellDefined body m0) (junk : ℕ → ℝ)
    {assign : Fin C → List (List ℕ)}
    (hpart : ((List.finRange C).flatMap assign).Perm (gridPoints gs.parGrid))
    {initScratch : Fin C → List OutSlot}
    (hinitLen : ∀ c, (initScratch c).length = gs.scratch.length)
    (hinitScratch : gs.parGrid ≠ [] ∨ ∀ c, initScratch c = gs.scratch.map fun _ => none)
    {trace : List (Fin C)}
    (htrace : ∀ c, ((assign c).flatMap gs.groupPoints).length ≤ trace.count c)
    {flushOrder : List (Fin C)}
    (hflush : flushOrder.Perm (List.finRange C)) :
    let sc := gs.scalarPrefetch m0
    let mInit := initOutMem gs (junkAt junk (Nat.pair 0 0)) m0
    let mFinal := pallasExecPar gs body junk m0 assign initScratch trace flushOrder
    (∀ pg ∈ gridPoints gs.parGrid, ∀ b a, inOutRegion gs sc pg b a →
      mFinal.read b a = (isoMem gs body sc junk mInit pg).read b a) ∧
    (∀ b a, (∀ pg ∈ gridPoints gs.parGrid, ¬ inOutRegion gs sc pg b a) →
      mFinal.read b a = mInit.read b a) := by
  set sc := gs.scalarPrefetch m0
  set mInit := initOutMem gs (junkAt junk (Nat.pair 0 0)) m0
  rcases harb : gridPoints gs.arbGrid with _ | ⟨ag₀, restAg⟩
  · -- Empty arbitrary grid: no step runs at all.
    have hgp_nil : ∀ pg, gs.groupPoints pg = [] := fun pg => by
      simp [GridSpec.groupPoints, harb]
    have hrem_nil : ∀ c, ((assign c).flatMap gs.groupPoints) = [] := fun c => by
      simp [hgp_nil]
    have htrace_noop : ∀ (tr : List (Fin C)) (m : Mem) (cores : Fin C → CoreState),
        (∀ c, (cores c).remPoints = []) →
        tr.foldl (stepCore gs body sc junk) (m, cores) = (m, cores) := by
      intro tr
      induction tr with
      | nil => intro _ _ _; rfl
      | cons c rest ih =>
        intro m cores hnil
        simp only [List.foldl_cons, stepCore, hnil c]
        exact ih m cores hnil
    have hflush_noop : ∀ (fl : List (Fin C)) (m : Mem),
        fl.foldl (fun m c => flushAll gs.outSpecs
          ((gs.outSpecs.map fun _ => none) ++ initScratch c) m) m = m := by
      intro fl
      induction fl with
      | nil => intro _; rfl
      | cons c rest ih =>
        intro m
        rw [List.foldl_cons, flushAll_outNone]
        exact ih m
    have hm_eq : pallasExecPar gs body junk m0 assign initScratch trace flushOrder = mInit := by
      simp only [pallasExecPar]
      rw [htrace_noop trace mInit _ (fun c => hrem_nil c)]
      exact hflush_noop flushOrder mInit
    refine ⟨?_, ?_⟩
    · intro pg _ b a ⟨ag, hag, _⟩
      rw [harb] at hag
      simp at hag
    · intro b a _
      rw [hm_eq]
  · -- Non-empty arbitrary grid: run the invariant across `trace` and `flushOrder`.
    let initProgs : Fin C → CoreProg := fun c =>
      { finished := [], active := none, pending := assign c }
    let initCores : Fin C → CoreState := fun c =>
      { remPoints := (assign c).flatMap gs.groupPoints
        slots := (gs.outSpecs.map fun _ => none) ++ initScratch c }
    have hinv0 : ParInv gs body sc junk mInit assign initScratch mInit initCores initProgs :=
      { wf := fun c => ⟨by simp [initProgs, CoreProg.allGroups, CoreProg.activeList],
          fun _ _ _ h => by simp [initProgs] at h⟩
        rem := fun c => by simp [initCores, initProgs, CoreProg.remPoints]
        slots_none := fun _ _ => rfl
        slots_some := fun _ _ _ _ h => by simp [initProgs] at h
        mem_finished := fun _ _ _ _ h => by simp [initProgs] at h
        mem_active := fun _ _ _ _ _ _ h => by simp [initProgs] at h
        mem_untouched := fun _ _ _ _ => rfl }
    obtain ⟨progsF, hinvF⟩ :=
      foldl_stepCore_preserves_inv hwd junk mInit hpart hinitLen hinitScratch harb
        trace mInit initCores initProgs hinv0
    set finalState := trace.foldl (stepCore gs body sc junk) (mInit, initCores)
    have hremF : ∀ c, (finalState.2 c).remPoints = [] := by
      intro c
      have hdrop := foldl_stepCore_remPoints gs body sc junk trace mInit initCores c
      rw [hdrop]
      exact List.drop_eq_nil_of_le (htrace c)
    have hpendF : ∀ c, (progsF c).pending = [] := by
      intro c
      have h1 := hinvF.rem c
      rw [hremF c] at h1
      simp only [CoreProg.remPoints] at h1
      cases hp : (progsF c).pending with
      | nil => rfl
      | cons pg' _ =>
        rw [hp] at h1
        simp [GridSpec.groupPoints, harb] at h1
    have hdoneAll : ∀ c pg done todo, (progsF c).active = some (pg, done, todo) →
        done = gs.groupPoints pg ∧ todo = [] := by
      intro c pg done todo hact
      have h1 := hinvF.rem c
      rw [hremF c] at h1
      simp only [CoreProg.remPoints, hact] at h1
      have htodo : todo = [] := (List.append_eq_nil_iff.mp h1.symm).1
      have hdt := ((hinvF.wf c).2 pg done todo hact).2
      rw [htodo, List.append_nil] at hdt
      exact ⟨hdt, htodo⟩
    have hfl_nd : flushOrder.Nodup := (hflush.nodup_iff).mpr (List.nodup_finRange C)
    obtain ⟨hActF, hFinF, hUntF⟩ :=
      foldl_flushAll_cores hwd junk mInit hpart hdoneAll flushOrder [] finalState.1
        (fun _ h => by simp at h)
        (fun c _ pg d t hact b a hout => hinvF.mem_active c pg d t b a hact hout)
        hinvF.mem_finished hinvF.mem_untouched hinvF.wf hinvF.slots_none hinvF.slots_some
        (fun _ _ => by simp) hfl_nd
    refine ⟨?_, ?_⟩
    · intro pg hpg b a hout
      have hmem : pg ∈ (List.finRange C).flatMap assign := hpart.symm.subset hpg
      rw [List.mem_flatMap] at hmem
      obtain ⟨c, _, hcpg⟩ := hmem
      rw [← (hinvF.wf c).1] at hcpg
      simp only [CoreProg.allGroups, hpendF c, List.append_nil, List.mem_append] at hcpg
      rcases hcpg with hfin | hact
      · exact hFinF c pg b a hfin hout
      · cases hca : (progsF c).active with
        | none => simp [CoreProg.activeList, hca] at hact
        | some p =>
          rcases p with ⟨pg', d', t'⟩
          simp only [CoreProg.activeList, hca, List.mem_singleton] at hact
          subst hact
          exact hActF c (by simpa using hflush.symm.subset (List.mem_finRange c)) pg d' t' hca b a
            hout
    · intro b a hnot
      apply hUntF b a
      · intro c pg hfin
        exact hnot pg (mem_assign_of_perm hpart (group_in_assign_of_finished hinvF.wf hfin))
      · intro c pg d t hact
        exact hnot pg (mem_assign_of_perm hpart (group_in_assign_of_active hinvF.wf hact))

/-- Running a single core on all grid points with `pallasExecPar` is definitionally equivalent to
sequential `pallasExec`. -/
private theorem foldl_stepCore_one (gs : GridSpec) (body : Body) (sc : ℕ → ℤ) (junk : ℕ → ℝ) :
    ∀ (pts : List (List ℕ)) (m : Mem) (slots : List OutSlot),
      let res := (List.replicate pts.length (0 : Fin 1)).foldl
        (stepCore gs body sc junk) (m, fun _ => { remPoints := pts, slots := slots })
      res.1 = (pts.foldl (stepGrid gs body sc junk) (m, slots)).1 ∧
      (res.2 0).slots = (pts.foldl (stepGrid gs body sc junk) (m, slots)).2
  | [], m, slots => ⟨rfl, rfl⟩
  | g :: rest, m, slots => by
    simp only [List.length_cons, List.replicate_succ, List.foldl_cons, stepCore]
    have hupd : Function.update (fun _ : Fin 1 => ({ remPoints := g :: rest, slots := slots } : CoreState)) 0
        ({ remPoints := rest, slots := (stepGrid gs body sc junk (m, slots) g).2 } : CoreState) =
      fun _ : Fin 1 => ({ remPoints := rest, slots := (stepGrid gs body sc junk (m, slots) g).2 } : CoreState) := by
      funext i
      fin_cases i
      rfl
    rw [hupd]
    exact foldl_stepCore_one gs body sc junk rest
      (stepGrid gs body sc junk (m, slots) g).1
      (stepGrid gs body sc junk (m, slots) g).2

private theorem pallasExecPar_seq_eq_pallasExec (gs : GridSpec) (body : Body)
    (junk : ℕ → ℝ) (m0 : Mem) :
    pallasExecPar (C := 1) gs body junk m0
      (fun _ => gridPoints gs.parGrid)
      (fun _ => gs.scratch.map fun _ => none)
      (List.replicate (gridPoints gs.grid).length 0)
      [0] = pallasExec gs body junk m0 := by
  have hgrid : (gridPoints gs.parGrid).flatMap gs.groupPoints = gridPoints gs.grid := by
    have h := List.take_append_drop gs.numParallel gs.grid
    rw [← h, gridPoints_append]
    rfl
  simp only [pallasExecPar, pallasExec, List.foldl_cons, List.foldl_nil, hgrid]
  have h1 := foldl_stepCore_one gs body (gs.scalarPrefetch m0) junk (gridPoints gs.grid)
    (initOutMem gs (junkAt junk (Nat.pair 0 0)) m0)
    ((gs.outSpecs.map fun _ => none) ++ gs.scratch.map fun _ => none)
  rw [h1.1, h1.2]

/-! ### Main soundness theorems -/

/-- Soundness of parallel execution when every core starts with uninitialized (`none`) scratch:
for any grid (including all-`arbitrary` grids), any number of cores `C`, any partition `assign` of
the parallel coordinates across cores, any complete step-level interleaving `trace`, and any final
flush order `flushOrder`, `pallasExecPar` yields the exact same HBM memory as `pallasExec`. -/
theorem pallasExecPar_eq_pallasExec_of_none {C : ℕ} {gs : GridSpec} {body : Body} {m0 : Mem}
    (hwd : gs.WellDefined body m0) (junk : ℕ → ℝ)
    (assign : Fin C → List (List ℕ))
    (hpart : ((List.finRange C).flatMap assign).Perm (gridPoints gs.parGrid))
    (trace : List (Fin C))
    (htrace : ∀ c, ((assign c).flatMap gs.groupPoints).length ≤ trace.count c)
    (flushOrder : List (Fin C))
    (hflush : flushOrder.Perm (List.finRange C)) :
    pallasExecPar gs body junk m0 assign (fun _ => gs.scratch.map fun _ => none)
      trace flushOrder = pallasExec gs body junk m0 := by
  have hpar := pallasExecPar_char hwd junk hpart (fun _ => by simp)
    (Or.inr fun _ => rfl) htrace hflush
  have hgrid : (gridPoints gs.parGrid).flatMap gs.groupPoints = gridPoints gs.grid := by
    have h := List.take_append_drop gs.numParallel gs.grid
    rw [← h, gridPoints_append]
    rfl
  have hseq := pallasExecPar_char (C := 1) hwd junk
    (assign := fun _ => gridPoints gs.parGrid)
    (by simp [List.finRange_succ])
    (initScratch := fun _ => gs.scratch.map fun _ => none)
    (fun _ => by simp) (Or.inr fun _ => rfl)
    (trace := List.replicate (gridPoints gs.grid).length 0)
    (fun c => by fin_cases c; simp [hgrid])
    (flushOrder := [0])
    (by simp [List.finRange_succ])
  rw [pallasExecPar_seq_eq_pallasExec] at hseq
  funext b a
  by_cases hex : ∃ pg ∈ gridPoints gs.parGrid, inOutRegion gs (gs.scalarPrefetch m0) pg b a
  · obtain ⟨pg, hpg, hout⟩ := hex
    exact (hpar.1 pg hpg b a hout).trans (hseq.1 pg hpg b a hout).symm
  · push Not at hex
    exact (hpar.2 b a hex).trans (hseq.2 b a hex).symm

/-- Soundness of parallel execution with arbitrary junk scratch per core: whenever `gs` has at least
one `parallel` dimension (`gs.parGrid ≠ []`) and `gs.WellDefined body m0` holds, every multi-core
execution — with any core count `C`, any partition `assign` of parallel groups, any arbitrary
initial scratch slots `initScratch` per core, any complete step-level interleaving `trace`, and any
final flush order `flushOrder` — yields the exact same HBM memory as sequential `pallasExec`.
[PAL-PAR-EXEC] -/
theorem pallasExecPar_eq_pallasExec {C : ℕ} {gs : GridSpec} {body : Body} {m0 : Mem}
    (hwd : gs.WellDefined body m0) (hparGrid : gs.parGrid ≠ [])
    (junk : ℕ → ℝ)
    (assign : Fin C → List (List ℕ))
    (hpart : ((List.finRange C).flatMap assign).Perm (gridPoints gs.parGrid))
    (initScratch : Fin C → List OutSlot)
    (hinitLen : ∀ c, (initScratch c).length = gs.scratch.length)
    (trace : List (Fin C))
    (htrace : ∀ c, ((assign c).flatMap gs.groupPoints).length ≤ trace.count c)
    (flushOrder : List (Fin C))
    (hflush : flushOrder.Perm (List.finRange C)) :
    pallasExecPar gs body junk m0 assign initScratch trace flushOrder =
      pallasExec gs body junk m0 := by
  have hpar := pallasExecPar_char hwd junk hpart hinitLen (Or.inl hparGrid) htrace hflush
  have hgrid : (gridPoints gs.parGrid).flatMap gs.groupPoints = gridPoints gs.grid := by
    have h := List.take_append_drop gs.numParallel gs.grid
    rw [← h, gridPoints_append]
    rfl
  have hseq := pallasExecPar_char (C := 1) hwd junk
    (assign := fun _ => gridPoints gs.parGrid)
    (by simp [List.finRange_succ])
    (initScratch := fun _ => gs.scratch.map fun _ => none)
    (fun _ => by simp) (Or.inl hparGrid)
    (trace := List.replicate (gridPoints gs.grid).length 0)
    (fun c => by fin_cases c; simp [hgrid])
    (flushOrder := [0])
    (by simp [List.finRange_succ])
  rw [pallasExecPar_seq_eq_pallasExec] at hseq
  funext b a
  by_cases hex : ∃ pg ∈ gridPoints gs.parGrid, inOutRegion gs (gs.scalarPrefetch m0) pg b a
  · obtain ⟨pg, hpg, hout⟩ := hex
    exact (hpar.1 pg hpg b a hout).trans (hseq.1 pg hpg b a hout).symm
  · push Not at hex
    exact (hpar.2 b a hex).trans (hseq.2 b a hex).symm

/-- Any `KernelSpec.Correct` proof for `pallasCall` transfers directly to every valid multi-core
execution `pallasExecPar`. -/
theorem pallasCall_par_correct {I S O : Type*} {C : ℕ}
    {spec : KernelSpec I S O} {gs : GridSpec} {body : Body}
    (hcorr : spec.Correct (pallasCall gs body))
    (hparGrid : gs.parGrid ≠ [])
    (junk : ℕ → ℝ) (m0 : Mem) (hreq : spec.requires (spec.readSt m0) (spec.readIn m0))
    (assign : Fin C → List (List ℕ))
    (hpart : ((List.finRange C).flatMap assign).Perm (gridPoints gs.parGrid))
    (initScratch : Fin C → List OutSlot)
    (hinitLen : ∀ c, (initScratch c).length = gs.scratch.length)
    (trace : List (Fin C))
    (htrace : ∀ c, ((assign c).flatMap gs.groupPoints).length ≤ trace.count c)
    (flushOrder : List (Fin C))
    (hflush : flushOrder.Perm (List.finRange C)) :
    let m' := pallasExecPar gs body junk m0 assign initScratch trace flushOrder
    spec.readOut m' = spec.math (spec.readSt m0) (spec.readIn m0) ∧
    spec.readSt m' = spec.mutate (spec.readSt m0) (spec.readIn m0) ∧
    Mem.agreeOutside spec.footprint m0 m' := by
  obtain ⟨-, mSeq, hcall, hOut, hSt, hFrame⟩ := hcorr junk m0 hreq
  have hwd : gs.WellDefined body m0 := by
    by_contra h
    rw [pallasCall_of_not_wellDefined junk h] at hcall
    cases hcall
  rw [pallasCall_of_wellDefined junk hwd, Option.some.injEq] at hcall
  subst hcall
  dsimp only
  rw [pallasExecPar_eq_pallasExec hwd hparGrid junk assign hpart initScratch hinitLen
    trace htrace flushOrder hflush]
  exact ⟨hOut, hSt, hFrame⟩

end

end TpuKernelLean.Pallas
