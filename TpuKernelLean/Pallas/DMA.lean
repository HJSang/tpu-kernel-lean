namespace TpuKernelLean.Pallas.DMA

/-- TPU memory space participating in explicit async DMA transfers. -/
inductive Space where
  | hbm
  | vmem
  deriving DecidableEq, Repr

/-- Half-open address range `[lo, hi)` within a buffer in a TPU memory space. -/
structure Region where
  /-- Memory space (`hbm` or `vmem`). -/
  space : Space
  /-- Buffer identifier within `space`. -/
  buf : Nat
  /-- Inclusive lower element offset. -/
  lo : Nat
  /-- Exclusive upper element offset. -/
  hi : Nat
  deriving DecidableEq, Repr

/-- Decidable check for whether two half-open memory regions share at least one element. -/
def Region.overlaps (r₁ r₂ : Region) : Bool :=
  decide (r₁.space = r₂.space ∧ r₁.buf = r₂.buf ∧
    r₁.lo < r₁.hi ∧ r₂.lo < r₂.hi ∧
    r₁.lo < r₂.hi ∧ r₂.lo < r₁.hi)

/-- Explicit DMA trace event modelling `make_async_copy(...).start()/.wait()` and local accesses. -/
inductive Ev where
  | start (id : Nat) (src dst : Region)
  | wait (id : Nat)
  | read (r : Region)
  | write (r : Region)
  deriving DecidableEq, Repr

/-- Metadata of a currently in-flight asynchronous DMA copy. -/
structure Copy where
  /-- Semaphore / copy identifier. -/
  id : Nat
  /-- Source region being read asynchronously by the DMA engine. -/
  src : Region
  /-- Destination region being written asynchronously by the DMA engine. -/
  dst : Region
  deriving DecidableEq, Repr

/-- Inductive specification of a hazard-free DMA event suffix starting from `inFlight`.
[PAL-DMA-HAZARD] -/
inductive HazardFreeFrom : List Copy → List Ev → Prop where
  | nil : HazardFreeFrom [] []
  | start {inFlight : List Copy} {id : Nat} {src dst : Region} {evs : List Ev}
      (hSelf : src.overlaps dst = false)
      (hFresh : ∀ c ∈ inFlight, c.id ≠ id)
      (hSrcDst : ∀ c ∈ inFlight, src.overlaps c.dst = false)
      (hDstSrc : ∀ c ∈ inFlight, dst.overlaps c.src = false)
      (hDstDst : ∀ c ∈ inFlight, dst.overlaps c.dst = false)
      (hRest : HazardFreeFrom ({ id := id, src := src, dst := dst } :: inFlight) evs) :
      HazardFreeFrom inFlight (Ev.start id src dst :: evs)
  | wait {inFlight : List Copy} {id : Nat} {evs : List Ev}
      (hMem : ∃ c ∈ inFlight, c.id = id)
      (hRest : HazardFreeFrom (inFlight.filter (fun c => c.id != id)) evs) :
      HazardFreeFrom inFlight (Ev.wait id :: evs)
  | read {inFlight : List Copy} {r : Region} {evs : List Ev}
      (hNoDst : ∀ c ∈ inFlight, r.overlaps c.dst = false)
      (hRest : HazardFreeFrom inFlight evs) :
      HazardFreeFrom inFlight (Ev.read r :: evs)
  | write {inFlight : List Copy} {r : Region} {evs : List Ev}
      (hNoDst : ∀ c ∈ inFlight, r.overlaps c.dst = false)
      (hNoSrc : ∀ c ∈ inFlight, r.overlaps c.src = false)
      (hRest : HazardFreeFrom inFlight evs) :
      HazardFreeFrom inFlight (Ev.write r :: evs)

/-- A complete DMA trace is hazard-free if it begins and ends with no in-flight copies and avoids
all read/write overlaps. [PAL-DMA-HAZARD] -/
def HazardFree (t : List Ev) : Prop :=
  HazardFreeFrom [] t

/-- Computable DMA hazard checker starting from the given in-flight copy list. [PAL-DMA-HAZARD] -/
def checkFrom : List Copy → List Ev → Bool
  | inFlight, [] => inFlight.isEmpty
  | inFlight, Ev.start id src dst :: evs =>
    !src.overlaps dst &&
    inFlight.all (fun c => decide (c.id ≠ id)) &&
    inFlight.all (fun c => !src.overlaps c.dst) &&
    inFlight.all (fun c => !dst.overlaps c.src) &&
    inFlight.all (fun c => !dst.overlaps c.dst) &&
    checkFrom ({ id := id, src := src, dst := dst } :: inFlight) evs
  | inFlight, Ev.wait id :: evs =>
    inFlight.any (fun c => c.id == id) &&
    checkFrom (inFlight.filter (fun c => c.id != id)) evs
  | inFlight, Ev.read r :: evs =>
    inFlight.all (fun c => !r.overlaps c.dst) &&
    checkFrom inFlight evs
  | inFlight, Ev.write r :: evs =>
    inFlight.all (fun c => !r.overlaps c.dst) &&
    inFlight.all (fun c => !r.overlaps c.src) &&
    checkFrom inFlight evs

/-- Computable DMA hazard checker for a complete trace. [PAL-DMA-HAZARD] -/
def check (t : List Ev) : Bool :=
  checkFrom [] t

theorem checkFrom_sound : ∀ (t : List Ev) (inFlight : List Copy),
    checkFrom inFlight t = true → HazardFreeFrom inFlight t
  | [], inFlight, h => by
    cases inFlight with
    | nil => exact HazardFreeFrom.nil
    | cons _ _ => simp [checkFrom] at h
  | Ev.start id src dst :: evs, inFlight, h => by
    simp only [checkFrom, Bool.and_eq_true] at h
    have ⟨⟨⟨⟨⟨hSelf, hFresh⟩, hSrcDst⟩, hDstSrc⟩, hDstDst⟩, hRest⟩ := h
    refine HazardFreeFrom.start (by simpa using hSelf) ?_ ?_ ?_ ?_ (checkFrom_sound evs _ hRest)
    · intro c hc
      have h' := List.all_eq_true.mp hFresh c hc
      simp at h'
      exact h'
    · intro c hc
      have h' := List.all_eq_true.mp hSrcDst c hc
      simp at h'
      exact h'
    · intro c hc
      have h' := List.all_eq_true.mp hDstSrc c hc
      simp at h'
      exact h'
    · intro c hc
      have h' := List.all_eq_true.mp hDstDst c hc
      simp at h'
      exact h'
  | Ev.wait id :: evs, inFlight, h => by
    simp only [checkFrom, Bool.and_eq_true] at h
    have ⟨hAny, hRest⟩ := h
    have ⟨c, hc, hid⟩ := List.any_eq_true.mp hAny
    simp at hid
    exact HazardFreeFrom.wait ⟨c, hc, hid⟩ (checkFrom_sound evs _ hRest)
  | Ev.read r :: evs, inFlight, h => by
    simp only [checkFrom, Bool.and_eq_true] at h
    have ⟨hNoDst, hRest⟩ := h
    refine HazardFreeFrom.read ?_ (checkFrom_sound evs _ hRest)
    intro c hc
    have h' := List.all_eq_true.mp hNoDst c hc
    simp at h'
    exact h'
  | Ev.write r :: evs, inFlight, h => by
    simp only [checkFrom, Bool.and_eq_true] at h
    have ⟨⟨hNoDst, hNoSrc⟩, hRest⟩ := h
    refine HazardFreeFrom.write ?_ ?_ (checkFrom_sound evs _ hRest)
    · intro c hc
      have h' := List.all_eq_true.mp hNoDst c hc
      simp at h'
      exact h'
    · intro c hc
      have h' := List.all_eq_true.mp hNoSrc c hc
      simp at h'
      exact h'

/-- Soundness of the computable DMA hazard checker. -/
theorem check_sound (t : List Ev) (h : check t = true) : HazardFree t :=
  checkFrom_sound t [] h

theorem checkFrom_complete {inFlight : List Copy} {t : List Ev}
    (h : HazardFreeFrom inFlight t) : checkFrom inFlight t = true := by
  induction h with
  | nil => rfl
  | start hSelf hFresh hSrcDst hDstSrc hDstDst _ ih =>
    simp only [checkFrom, Bool.and_eq_true, List.all_eq_true]
    exact ⟨⟨⟨⟨⟨by simp [hSelf], fun c hc => by simpa using hFresh c hc⟩,
      fun c hc => by simp [hSrcDst c hc]⟩, fun c hc => by simp [hDstSrc c hc]⟩,
      fun c hc => by simp [hDstDst c hc]⟩, ih⟩
  | wait hMem _ ih =>
    obtain ⟨c, hc, hid⟩ := hMem
    simp only [checkFrom, Bool.and_eq_true, List.any_eq_true]
    exact ⟨⟨c, hc, by simp [hid]⟩, ih⟩
  | read hNoDst _ ih =>
    simp only [checkFrom, Bool.and_eq_true, List.all_eq_true]
    exact ⟨fun c hc => by simp [hNoDst c hc], ih⟩
  | write hNoDst hNoSrc _ ih =>
    simp only [checkFrom, Bool.and_eq_true, List.all_eq_true]
    exact ⟨⟨fun c hc => by simp [hNoDst c hc], fun c hc => by simp [hNoSrc c hc]⟩, ih⟩

/-- The checker decides `HazardFree`, so `check t = false` proves `¬ HazardFree t`. -/
theorem check_iff (t : List Ev) : check t = true ↔ HazardFree t :=
  ⟨check_sound t, checkFrom_complete⟩

/-- Example double-buffered VMEM pipeline trace: prefetch slot 1 while reading slot 0, then swap. -/
def doubleBufferedTrace : List Ev :=
  let hbmTile (k : Nat) : Region := { space := Space.hbm, buf := 0, lo := k * 128, hi := (k + 1) * 128 }
  let vmemSlot (s : Nat) : Region := { space := Space.vmem, buf := s, lo := 0, hi := 128 }
  [ Ev.start 0 (hbmTile 0) (vmemSlot 0),
    Ev.start 1 (hbmTile 1) (vmemSlot 1),
    Ev.wait 0,
    Ev.read (vmemSlot 0),
    Ev.start 2 (hbmTile 2) (vmemSlot 0),
    Ev.wait 1,
    Ev.read (vmemSlot 1),
    Ev.wait 2,
    Ev.read (vmemSlot 0) ]

theorem doubleBufferedTrace_check : check doubleBufferedTrace = true := rfl

theorem doubleBufferedTrace_hazardFree : HazardFree doubleBufferedTrace :=
  check_sound doubleBufferedTrace rfl

/-- Example buggy trace reading a VMEM destination buffer before waiting on its async DMA copy. -/
def readBeforeWaitTrace : List Ev :=
  let hbmTile0 : Region := { space := Space.hbm, buf := 0, lo := 0, hi := 128 }
  let vmemSlot0 : Region := { space := Space.vmem, buf := 0, lo := 0, hi := 128 }
  [ Ev.start 0 hbmTile0 vmemSlot0,
    Ev.read vmemSlot0,
    Ev.wait 0 ]

theorem readBeforeWaitTrace_check : check readBeforeWaitTrace = false := rfl

theorem readBeforeWaitTrace_not_hazardFree : ¬ HazardFree readBeforeWaitTrace := fun h => by
  have := (check_iff _).mpr h
  simp [readBeforeWaitTrace, check, checkFrom, Region.overlaps] at this

/-- Example buggy trace: one copy whose source and destination overlap in the same VMEM buffer. -/
def selfOverlapTrace : List Ev :=
  [ Ev.start 0 { space := Space.vmem, buf := 0, lo := 0, hi := 64 }
      { space := Space.vmem, buf := 0, lo := 32, hi := 96 },
    Ev.wait 0 ]

theorem selfOverlapTrace_not_hazardFree : ¬ HazardFree selfOverlapTrace := fun h => by
  have := (check_iff _).mpr h
  simp [selfOverlapTrace, check, checkFrom, Region.overlaps] at this

end TpuKernelLean.Pallas.DMA
