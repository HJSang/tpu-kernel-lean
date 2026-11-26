import TpuKernelLean.Spec
import TpuKernelLean.Pallas.Core

/-!
Specification and verification of `sglang_tpu/kv_page_write.py` (`write_pages`),
with trailing page dimensions flattened to a single page width `P`:
- `kv : [N, P]` (in-place state, aliased to output 0 via `input_output_aliases={2: 0}`)
- `pages : [M, P]` (read-only input)
- `block_ids : [M]` (scalar-prefetched page slot indices)
-/

namespace TpuKernelLean.Examples.KvPageWrite

/-- Read-only inputs to `write_pages`: source pages `[M, P]` and target page ids `[M]` (`kv_page_write.py:25`). -/
structure WritePagesIn (M P : ℕ) where
  /-- Source pages to copy into the KV cache. -/
  pages : Tensor ℝ [M, P]
  /-- Target KV cache slot for each source page, as the raw int32 value stored in HBM. -/
  blockIds : Fin M → ℝ

/-- In-place mutable state of `write_pages`: the KV cache buffer `[N, P]` (`kv_page_write.py:25`). -/
structure WritePagesSt (N P : ℕ) where
  /-- KV cache pages flattened to `[N, P]`. -/
  kv : Tensor ℝ [N, P]

noncomputable section

/-- Post-state of `kv`: page `j` receives `pages[i]` if `blockIds[i] = j`, else stays unchanged (`kv_page_write.py:26`). -/
def writePagesMut {N M P : ℕ} (st : WritePagesSt N P) (inp : WritePagesIn M P) :
    WritePagesSt N P :=
  { kv := Tensor.of2 fun j p =>
      if h : ∃ i : Fin M, inp.blockIds i = j.val then
        inp.pages.get2 (Classical.choose h) p
      else
        st.kv.get2 j p }

/-- Four-part `KernelSpec` for `write_pages` with distinct buffers and injective in-range `blockIds`. -/
def writePagesSpec (N M P : ℕ) (idsBuf pagesBuf kvBuf : BufId) :
    KernelSpec (WritePagesIn M P) (WritePagesSt N P) Unit :=
  { readIn := fun m =>
      { pages := (Layout.rowMajor pagesBuf 0 [M, P]).read m
        blockIds := fun i => m.read idsBuf i.val }
    readSt := fun m =>
      { kv := (Layout.rowMajor kvBuf 0 [N, P]).read m }
    readOut := fun _ => ()
    footprint := Mem.bufFootprint kvBuf
    math := fun _ _ => ()
    mutate := writePagesMut
    requires := fun _ inp =>
      0 < M ∧
      (∀ i : Fin M, ∃ n : ℕ, n < N ∧ decodeInt32Nat (inp.blockIds i) = some n) ∧
      Function.Injective inp.blockIds ∧
      kvBuf ≠ pagesBuf ∧ kvBuf ≠ idsBuf ∧ pagesBuf ≠ idsBuf }

/-- Alias table pairing Python arg 2 (`kv`, after scalar-prefetched `block_ids` at arg 0 and
`pages` at arg 1) with output 0 (`kv_page_write.py:46`). -/
def writePagesAliases : List (ℕ × ℕ) := [(2, 0)]

/-- Pallas `GridSpec` for `write_pages` (`kv_page_write.py:31–48`), with `None` block dim
represented as `1`, one scalar-prefetch argument (`block_ids`, decoded with checked `readInt32`),
and `input_output_aliases={2: 0}`. -/
def writePagesGridSpec (N M P : ℕ) (idsBuf pagesBuf kvBuf : BufId) : Pallas.GridSpec :=
  { grid := [M]
    scalarPrefetchBufs := [idsBuf]
    scalarPrefetchFn := fun bufs i => ((bufs.getD 0 fun _ => none) i).getD (-1)
    inSpecs :=
      [ ({ buf := pagesBuf, shape := [M, P] },
         { blockShape := [1, P], indexMap := fun g _ => [(g.getD 0 0 : ℤ), 0] }),
        ({ buf := kvBuf, shape := [N, P] },
         { blockShape := [1, P], indexMap := fun g ids => [ids (g.getD 0 0), 0] }) ]
    outSpecs :=
      [ ({ buf := kvBuf, shape := [N, P] },
         { blockShape := [1, P], indexMap := fun g ids => [ids (g.getD 0 0), 0] }) ]
    aliases := writePagesAliases }

/-- Pallas kernel body for `write_pages`: `out_ref[...] = pages_ref[...]` (`kv_page_write.py:20–22`). -/
def writePagesBody : Pallas.Body :=
  fun _ _ ins _ => [ins.getD 0 (fun _ => 0)]

/-- Pallas semantics of `write_pages` (`none` on undefined behaviour). -/
def writePagesRun (N M P : ℕ) (idsBuf pagesBuf kvBuf : BufId) : (ℕ → ℝ) → Mem → Option Mem :=
  Pallas.pallasCall (writePagesGridSpec N M P idsBuf pagesBuf kvBuf) writePagesBody

/-- Sequential execution of `write_pages`, ignoring undefined behaviour. -/
def writePagesExec (N M P : ℕ) (idsBuf pagesBuf kvBuf : BufId) : (ℕ → ℝ) → Mem → Mem :=
  Pallas.pallasExec (writePagesGridSpec N M P idsBuf pagesBuf kvBuf) writePagesBody

/-- Equivalent junk-free sequential fold of masked `storeBlock` calls over a list of page indices. -/
def writePagesSeq (N P : ℕ) (pagesBuf kvBuf : BufId)
    (ids : ℕ → ℕ) (m0 : Mem) (steps : List ℕ) (m : Mem) : Mem :=
  steps.foldl (fun acc i =>
    Pallas.storeBlock { buf := kvBuf, shape := [N, P] } [1, P] [ids i, 0]
      (fun loc => m0.read pagesBuf (i * P + loc.getD 1 0)) acc) m

private theorem list_flatMap_singleton {α β : Type*} (f : α → β) (l : List α) :
    l.flatMap (fun x => [f x]) = l.map f := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp [ih]

private theorem gridPoints_singleton (M : ℕ) :
    Pallas.gridPoints [M] = (List.range M).map fun i => [i] := by
  simp only [Pallas.gridPoints, List.map_cons, List.map_nil]
  exact list_flatMap_singleton (fun i => [i]) (List.range M)

private theorem writePages_initOutMem (N M P : ℕ) (idsBuf pagesBuf kvBuf : BufId)
    (junk : ℕ → ℝ) (m0 : Mem) :
    Pallas.initOutMem (writePagesGridSpec N M P idsBuf pagesBuf kvBuf) junk m0 = m0 := rfl

private theorem writePagesSeq_cons (N P : ℕ) (pagesBuf kvBuf : BufId)
    (ids : ℕ → ℕ) (m0 m : Mem) (i : ℕ) (rest : List ℕ) :
    writePagesSeq N P pagesBuf kvBuf ids m0 (i :: rest) m =
      writePagesSeq N P pagesBuf kvBuf ids m0 rest
        (Pallas.storeBlock { buf := kvBuf, shape := [N, P] } [1, P] [ids i, 0]
          (fun loc => m0.read pagesBuf (i * P + loc.getD 1 0)) m) := rfl

private theorem writePagesSeq_append (N P : ℕ) (pagesBuf kvBuf : BufId)
    (ids : ℕ → ℕ) (m0 m : Mem) (l₁ l₂ : List ℕ) :
    writePagesSeq N P pagesBuf kvBuf ids m0 (l₁ ++ l₂) m =
      writePagesSeq N P pagesBuf kvBuf ids m0 l₂
        (writePagesSeq N P pagesBuf kvBuf ids m0 l₁ m) := by
  simp [writePagesSeq, List.foldl_append]

private theorem read_rowMajor_get2 {N P : ℕ} (buf : BufId) (m : Mem) (j : Fin N) (p : Fin P) :
    ((Layout.rowMajor buf 0 [N, P]).read m).get2 j p = m.read buf (j.val * P + p.val) := by
  show m.read buf (0 + Idx.flat (s := [N, P]) (j, p, ())) = m.read buf (j.val * P + p.val)
  simp

/-- Storing an in-bounds `loadBlock` of `pages[i]` into `kv` does not depend on `junk`. -/
theorem writePages_storeBlock_loadBlock_eq (N M P : ℕ) (pagesBuf kvBuf : BufId)
    (junk₁ : ℕ → ℝ) (m0 m : Mem) (i target : ℕ) (hi : i < M) :
    Pallas.storeBlock { buf := kvBuf, shape := [N, P] } [1, P] [target, 0]
      (Pallas.loadBlock { buf := pagesBuf, shape := [M, P] } [1, P] [i, 0] junk₁ m0) m =
    Pallas.storeBlock { buf := kvBuf, shape := [N, P] } [1, P] [target, 0]
      (fun loc => m0.read pagesBuf (i * P + loc.getD 1 0)) m := by
  apply Pallas.storeBlock_congr_val
  intro a ha
  simp only [Pallas.addrInBlock, Bool.and_eq_true, decide_eq_true_eq] at ha
  rcases ha with ⟨hlt, hcoord⟩
  have hP : 0 < P := by
    by_contra h0
    have hP0 : P = 0 := by omega
    subst hP0
    simp at hlt
  have hrem : a % P < P := Nat.mod_lt a hP
  have hunflat : Pallas.unflatCoord [N, P] a = [a / P, a % P] := by
    have h1 := Pallas.unflatCoord_two N P (a / P) (a % P) hrem
    have hdiv : (a / P) * P + a % P = a := by
      rw [Nat.mul_comm]
      exact Nat.div_add_mod a P
    rwa [hdiv] at h1
  rw [hunflat] at hcoord ⊢
  simp only [Pallas.coordInBlock, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at hcoord
  have hq : a / P = target := by omega
  have hloc : Pallas.localCoord [1, P] [target, 0] [a / P, a % P] = [0, a % P] := by
    simp [Pallas.localCoord, hq]
  rw [hloc]
  rw [Pallas.loadBlock_eq_of_inBounds]
  · simp [Pallas.globalCoord, Pallas.flatCoord]
  · simp [Pallas.inBounds, hrem]
  · simp [Pallas.globalCoord, Pallas.inBounds, hi, hrem]

/-- Effect of one `write_pages` grid step on element `(j, p)` of `kv`. -/
theorem writePages_storeBlock_read (N P : ℕ) (pagesBuf kvBuf : BufId)
    (m0 m : Mem) (i target : ℕ) (j : Fin N) (p : Fin P) :
    (Pallas.storeBlock { buf := kvBuf, shape := [N, P] } [1, P] [target, 0]
      (fun loc => m0.read pagesBuf (i * P + loc.getD 1 0)) m).read
      kvBuf (j.val * P + p.val) =
    if target = j.val then m0.read pagesBuf (i * P + p.val) else m.read kvBuf (j.val * P + p.val) := by
  have hlt : j.val * P + p.val < ([N, P] : List ℕ).prod := by
    simpa using Idx.flat_lt (s := [N, P]) (j, p, ())
  have hunflat : Pallas.unflatCoord [N, P] (j.val * P + p.val) = [j.val, p.val] :=
    Pallas.unflatCoord_two N P j.val p.val p.isLt
  by_cases ht : target = j.val
  · subst ht
    have hin : Pallas.addrInBlock { buf := kvBuf, shape := [N, P] } [1, P] [j.val, 0]
        (j.val * P + p.val) = true := by
      simp only [Pallas.addrInBlock, hunflat, Pallas.coordInBlock, Bool.and_true,
        Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨hlt, ⟨by omega, by omega⟩, ⟨by omega, by simp⟩⟩
    rw [Pallas.storeBlock_read_inside _ _ _ _ _ _ hin, hunflat]
    simp [Pallas.localCoord]
  · have hout : Pallas.addrInBlock { buf := kvBuf, shape := [N, P] } [1, P] [target, 0]
        (j.val * P + p.val) = false := by
      simp only [Pallas.addrInBlock, hunflat, Pallas.coordInBlock, Bool.and_true]
      have hne : decide (target * 1 ≤ j.val ∧ j.val < (target + 1) * 1) = false := by
        simp only [decide_eq_false_iff_not]
        omega
      rw [hne]
      simp
    rw [Pallas.storeBlock_read_outside _ _ _ _ _ _ _ (Or.inr hout)]
    simp [ht]

/-- Flushed VMEM execution of `write_pages` over in-bounds `steps` equals the junk-free sequential fold. -/
theorem writePages_fold_eq_seq (N M P : ℕ) (idsBuf pagesBuf kvBuf : BufId)
    (hbuf : kvBuf ≠ pagesBuf) (junk : ℕ → ℝ) (m0 : Mem) :
    ∀ (steps : List ℕ) (m : Mem) (slot : Pallas.OutSlot),
      (∀ k ∈ steps, k < M) →
      (∀ a, m.read pagesBuf a = m0.read pagesBuf a) →
      let gs := writePagesGridSpec N M P idsBuf pagesBuf kvBuf
      let ids := gs.scalarPrefetch m0
      let pts := steps.map fun i => [i]
      let finalState := pts.foldl (Pallas.stepGrid gs writePagesBody ids junk) (m, [slot])
      Pallas.flushAll gs.outSpecs finalState.2 finalState.1 =
        writePagesSeq N P pagesBuf kvBuf (fun k => (ids k).toNat) m0 steps
          (Pallas.flushAll gs.outSpecs [slot] m)
  | [], m, slot, _, _ => by
    simp [writePagesSeq]
  | i :: rest, m, slot, hsteps, hpages => by
    dsimp only
    rw [List.map_cons, List.foldl_cons, writePagesSeq_cons]
    have hi : i < M := hsteps i (by simp)
    have hrest : ∀ k ∈ rest, k < M := fun k hk => hsteps k (by simp [hk])
    let gs := writePagesGridSpec N M P idsBuf pagesBuf kvBuf
    let ids := gs.scalarPrefetch m0
    let kvArr : Pallas.ArrayDesc := { buf := kvBuf, shape := [N, P] }
    let pagesArr : Pallas.ArrayDesc := { buf := pagesBuf, shape := [M, P] }
    let step := Pallas.flatCoord gs.grid [i] + 1
    let junkIn0 := Pallas.junkAt junk (Nat.pair step 1)
    let m1 : Mem :=
      match slot with
      | some (oldIdx, oldVal) =>
        if oldIdx = [(ids i).toNat, 0] then m else Pallas.storeBlock kvArr [1, P] oldIdx oldVal m
      | none => m
    have hm1_pages : ∀ a, m1.read pagesBuf a = m0.read pagesBuf a := by
      intro a
      cases slot with
      | none => exact hpages a
      | some p =>
        dsimp only [m1]
        split_ifs
        · exact hpages a
        · rw [Pallas.storeBlock_read_other_buf _ _ _ _ _ _ _ (Ne.symm hbuf)]
          exact hpages a
    have hload :
        Pallas.loadBlock pagesArr [1, P] [i, 0] junkIn0 m1 =
        Pallas.loadBlock pagesArr [1, P] [i, 0] junkIn0 m0 :=
      Pallas.loadBlock_congr_mem pagesArr [1, P] [i, 0] junkIn0 hm1_pages
    have hstep :
        Pallas.stepGrid gs writePagesBody ids junk (m, [slot]) [i] =
        (m1, [some ([(ids i).toNat, 0], Pallas.loadBlock pagesArr [1, P] [i, 0] junkIn0 m0)]) := by
      simp only [Pallas.stepGrid, Pallas.flushAndPrep, gs, writePagesGridSpec,
        writePagesBody, Pallas.makeSlots, List.mapIdx_cons, List.mapIdx_nil,
        Pallas.BlockSpec.blockIdx, List.map_cons, List.map_nil, Int.toNat_zero, Int.toNat_natCast]
      rw [← hload]
      rfl
    have hflush :
        Pallas.flushAll gs.outSpecs
          [some ([(ids i).toNat, 0], Pallas.loadBlock pagesArr [1, P] [i, 0] junkIn0 m0)] m1 =
        Pallas.storeBlock kvArr [1, P] [(ids i).toNat, 0]
          (fun loc => m0.read pagesBuf (i * P + loc.getD 1 0))
          (Pallas.flushAll gs.outSpecs [slot] m) := by
      cases slot with
      | none =>
        exact writePages_storeBlock_loadBlock_eq N M P pagesBuf kvBuf junkIn0 m0 m i (ids i).toNat hi
      | some p =>
        rcases p with ⟨oldIdx, oldVal⟩
        simp only [Pallas.flushAll, gs, writePagesGridSpec, m1, kvArr]
        split_ifs with heq
        · rw [heq, Pallas.storeBlock_storeBlock_same]
          exact writePages_storeBlock_loadBlock_eq N M P pagesBuf kvBuf junkIn0 m0 m i (ids i).toNat hi
        · exact writePages_storeBlock_loadBlock_eq N M P pagesBuf kvBuf junkIn0 m0 _ i (ids i).toNat hi
    rw [hstep]
    have ih := writePages_fold_eq_seq N M P idsBuf pagesBuf kvBuf hbuf junk m0 rest m1
      (some ([(ids i).toNat, 0], Pallas.loadBlock pagesArr [1, P] [i, 0] junkIn0 m0)) hrest hm1_pages
    dsimp only at ih
    rw [ih, hflush]

private theorem readInt32_getD_toNat_eq (m0 : Mem) (idsBuf : BufId) (i : ℕ) :
    ((readInt32 m0 idsBuf i).getD (-1)).toNat = (decodeInt32Nat (m0.read idsBuf i)).getD 0 := by
  simp only [readInt32, decodeInt32Nat]
  cases decodeInt32 (m0.read idsBuf i) with
  | none => rfl
  | some z =>
    dsimp only [Option.getD]
    split_ifs with h
    · rfl
    · exact Int.toNat_of_nonpos (by omega)

/-- End-to-end equivalence between `writePagesExec` and the junk-free sequential fold over `List.range M`. -/
theorem writePagesExec_eq_seq (N M P : ℕ) (idsBuf pagesBuf kvBuf : BufId)
    (hbuf : kvBuf ≠ pagesBuf) (junk : ℕ → ℝ) (m0 : Mem) :
    writePagesExec N M P idsBuf pagesBuf kvBuf junk m0 =
      writePagesSeq N P pagesBuf kvBuf (fun i => (decodeInt32Nat (m0.read idsBuf i)).getD 0) m0
        (List.range M) m0 := by
  simp only [writePagesExec, Pallas.pallasExec]
  have hinit := writePages_initOutMem N M P idsBuf pagesBuf kvBuf
    (Pallas.junkAt junk (Nat.pair 0 0)) m0
  rw [hinit]
  have hgrid : Pallas.gridPoints (writePagesGridSpec N M P idsBuf pagesBuf kvBuf).grid =
      (List.range M).map fun i => [i] := gridPoints_singleton M
  rw [hgrid]
  have hfold := writePages_fold_eq_seq N M P idsBuf pagesBuf kvBuf hbuf junk m0
    (List.range M) m0 none (fun k hk => List.mem_range.mp hk) (fun _ => rfl)
  have hids_eq :
      (fun k => ((writePagesGridSpec N M P idsBuf pagesBuf kvBuf).scalarPrefetch m0 k).toNat) =
        fun i => (decodeInt32Nat (m0.read idsBuf i)).getD 0 := by
    funext i
    simpa [writePagesGridSpec, Pallas.GridSpec.scalarPrefetch] using
      readInt32_getD_toNat_eq m0 idsBuf i
  rw [← hids_eq]
  exact hfold

/-- If no step in `steps` targets page `j`, then element `(j, p)` of `kv` is unchanged. -/
theorem writePagesSeq_read_none (N P : ℕ) (pagesBuf kvBuf : BufId)
    (ids : ℕ → ℕ) (m0 : Mem) (j : Fin N) (p : Fin P) :
    ∀ (steps : List ℕ) (m : Mem),
      (∀ i ∈ steps, ids i ≠ j.val) →
      (writePagesSeq N P pagesBuf kvBuf ids m0 steps m).read kvBuf (j.val * P + p.val) =
        m.read kvBuf (j.val * P + p.val)
  | [], _, _ => rfl
  | i :: rest, m, hne => by
    rw [writePagesSeq_cons]
    have hneq : ids i ≠ j.val := hne i (by simp)
    rw [writePagesSeq_read_none N P pagesBuf kvBuf ids m0 j p rest _
      (fun k hk => hne k (by simp [hk]))]
    rw [writePages_storeBlock_read N P pagesBuf kvBuf m0 m i (ids i) j p]
    simp [hneq]

/-- If `i` is the last step in `steps` targeting page `j`, then `(j, p)` receives `pages[i, p]`. -/
theorem writePagesSeq_read_last (N P : ℕ) (pagesBuf kvBuf : BufId)
    (ids : ℕ → ℕ) (m0 m : Mem)
    (l₁ l₂ : List ℕ) (i : ℕ) (j : Fin N) (p : Fin P)
    (htarget : ids i = j.val)
    (hl₂_ne : ∀ k ∈ l₂, ids k ≠ j.val) :
    (writePagesSeq N P pagesBuf kvBuf ids m0 (l₁ ++ i :: l₂) m).read
      kvBuf (j.val * P + p.val) = m0.read pagesBuf (i * P + p.val) := by
  rw [writePagesSeq_append, writePagesSeq_cons]
  rw [writePagesSeq_read_none N P pagesBuf kvBuf ids m0 j p l₂ _ hl₂_ne]
  rw [writePages_storeBlock_read N P pagesBuf kvBuf m0 _ i (ids i) j p]
  simp [htarget]

private theorem range_split (M i : ℕ) (hi : i < M) :
    ∃ l₁ l₂ : List ℕ, List.range M = l₁ ++ i :: l₂ ∧ (∀ k ∈ l₂, i < k ∧ k < M) := by
  induction M with
  | zero => omega
  | succ M ih =>
    rw [List.range_succ]
    by_cases h : i = M
    · subst h
      exact ⟨List.range i, [], by simp, by simp⟩
    · have hi' : i < M := by omega
      rcases ih hi' with ⟨l₁, l₂, heq, hl₂⟩
      refine ⟨l₁, l₂ ++ [M], ?_, ?_⟩
      · rw [heq]
        simp [List.append_assoc]
      · intro k hk
        simp only [List.mem_append, List.mem_singleton] at hk
        rcases hk with hk | rfl
        · have ⟨h1, h2⟩ := hl₂ k hk
          exact ⟨h1, by omega⟩
        · exact ⟨hi', by omega⟩


/-- An in-block address of row block `[r, 0]` (block shape `[1, P]`) of an `[N, P]` array lies in
row `r`. -/
theorem addrInBlock_row {buf : BufId} {N P r a : ℕ}
    (h : Pallas.addrInBlock { buf := buf, shape := [N, P] } [1, P] [r, 0] a = true) :
    a / P = r := by
  simp only [Pallas.addrInBlock, Bool.and_eq_true, decide_eq_true_eq] at h
  rcases h with ⟨hlt, hcoord⟩
  have hP : 0 < P := by
    by_contra h0
    have hP0 : P = 0 := by omega
    subst hP0
    simp at hlt
  have hunflat : Pallas.unflatCoord [N, P] a = [a / P, a % P] := by
    have h1 := Pallas.unflatCoord_two N P (a / P) (a % P) (Nat.mod_lt a hP)
    have hdiv : (a / P) * P + a % P = a := by
      rw [Nat.mul_comm]
      exact Nat.div_add_mod a P
    rwa [hdiv] at h1
  rw [hunflat] at hcoord
  simp only [Pallas.coordInBlock, Bool.and_true, Bool.and_eq_true, decide_eq_true_eq] at hcoord
  omega

private theorem gridPoints_getD (M k : ℕ) (hk : k < M) :
    (Pallas.gridPoints [M]).getD k [] = [k] := by
  rw [gridPoints_singleton]
  simp [hk]

/-- Under the spec's precondition (ids are distinct non-negative int32s below `N`, batch and page
width positive, `kvBuf` disjoint from `pagesBuf` and `idsBuf`), the `write_pages` run is well
defined: no out-of-range block, no output revisit, no stale read, and valid aliases. -/
theorem writePages_wellDefined (N M P : ℕ) (idsBuf pagesBuf kvBuf : BufId) (hM : 0 < M) (hP : 0 < P)
    (m0 : Mem) (hids : ∀ i : Fin M, ∃ n : ℕ, n < N ∧ decodeInt32Nat (m0.read idsBuf i.val) = some n)
    (hinj : Function.Injective fun i : Fin M => m0.read idsBuf i.val)
    (hbuf : kvBuf ≠ pagesBuf) (hids_buf : kvBuf ≠ idsBuf) :
    (writePagesGridSpec N M P idsBuf pagesBuf kvBuf).WellDefined writePagesBody m0 := by
  have hlen : (Pallas.gridPoints [M]).length = M := by simp [Pallas.gridPoints_length]
  have hsc : ∀ k, k < M → ∃ n : ℕ, n < N ∧ m0.read idsBuf k = n ∧
      (writePagesGridSpec N M P idsBuf pagesBuf kvBuf).scalarPrefetch m0 k = (n : ℤ) := by
    intro k hk
    obtain ⟨n, hn, hdec⟩ := hids ⟨k, hk⟩
    have ⟨hxn, hle⟩ := (decodeInt32Nat_eq_some_iff (m0.read idsBuf k) n).mp hdec
    have hread : readInt32 m0 idsBuf k = some (n : ℤ) := by
      rw [readInt32, decodeInt32_eq_some_iff]
      exact ⟨hxn, by dsimp [inInt32Range, int32Min]; omega⟩
    refine ⟨n, hn, by exact_mod_cast hxn.symm, ?_⟩
    simp [Pallas.GridSpec.scalarPrefetch, writePagesGridSpec, hread]
  constructor
  · intro d hd
    simp only [writePagesGridSpec, List.mem_singleton] at hd
    subst hd
    exact hM
  · intro p hp
    simp only [writePagesGridSpec, List.cons_append, List.nil_append, List.mem_cons,
      List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl <;> simp [hP]
  · intro k hk p hp
    simp only [writePagesGridSpec, hlen] at hk
    simp only [writePagesGridSpec, List.cons_append, List.nil_append, List.mem_cons,
      List.not_mem_nil, or_false] at hp
    obtain ⟨n, hn, -, hfl⟩ := hsc k hk
    rcases hp with rfl | rfl | rfl <;>
      simp only [Pallas.GridSpec.idxAt, writePagesGridSpec, gridPoints_getD M k hk] <;>
      simp [Pallas.blockOk, Pallas.GridSpec.scalarPrefetch, writePagesGridSpec] at hfl ⊢ <;>
      simp [hfl, hk, hn, hP]
  · intro p hp
    simp only [writePagesGridSpec, List.mem_singleton] at hp
    subst hp
    apply Pallas.noRevisit_of_nodup
    simp only [writePagesGridSpec, hlen]
    apply List.Nodup.map_on _ List.nodup_range
    intro a ha b hb hab
    simp only [List.mem_range] at ha hb
    simp only [Pallas.GridSpec.idxAt, gridPoints_getD M a ha,
      gridPoints_getD M b hb, List.getD_cons_zero, List.cons.injEq, and_true] at hab
    change (writePagesGridSpec N M P idsBuf pagesBuf kvBuf).scalarPrefetch m0 a =
      (writePagesGridSpec N M P idsBuf pagesBuf kvBuf).scalarPrefetch m0 b at hab
    obtain ⟨na, -, hxa, hfa⟩ := hsc a ha
    obtain ⟨nb, -, hxb, hfb⟩ := hsc b hb
    rw [hfa, hfb] at hab
    have hn : na = nb := by exact_mod_cast hab
    have hfin := @hinj ⟨a, ha⟩ ⟨b, hb⟩ (by simp only; rw [hxa, hxb, hn])
    simpa using congrArg Fin.val hfin
  · intro _ _ _ _
    rfl
  · intro s t hst ht i hi o ho hov
    simp only [writePagesGridSpec, hlen] at ht
    have hs : s < M := by omega
    simp only [writePagesGridSpec, List.mem_cons, List.not_mem_nil, or_false] at hi ho
    subst ho
    rcases hi with rfl | rfl
    · exact hbuf hov.1.symm
    · obtain ⟨_, a, ha1, ha2⟩ := hov
      simp only [Pallas.GridSpec.idxAt, List.map_cons, List.map_nil, Int.toNat_zero] at ha1 ha2
      have h1 := addrInBlock_row ha1
      have h2 := addrInBlock_row ha2
      have hgrid : (writePagesGridSpec N M P idsBuf pagesBuf kvBuf).grid = [M] := rfl
      rw [hgrid, gridPoints_getD M t ht] at h1
      rw [hgrid, gridPoints_getD M s hs] at h2
      simp only [List.getD_cons_zero] at h1 h2
      obtain ⟨nt, -, hxt, hft⟩ := hsc t ht
      obtain ⟨ns, -, hxs, hfs⟩ := hsc s hs
      rw [hft, Int.toNat_natCast] at h1
      rw [hfs, Int.toNat_natCast] at h2
      have hn : nt = ns := by omega
      have hfin := @hinj ⟨t, ht⟩ ⟨s, hs⟩ (by simp only; rw [hxt, hxs, hn])
      have := congrArg Fin.val hfin
      simp at this
      omega
  · refine
      { entries_ok := ?_
        outs_nodup := by simp [writePagesGridSpec, writePagesAliases]
        out_bufs_nodup := by simp [writePagesGridSpec]
        buf_eq_aliased := ?_
        out_disjoint_scalar := ?_ }
    · intro p hp
      simp only [writePagesGridSpec, writePagesAliases, List.mem_singleton] at hp
      subst hp
      exact ⟨by simp [writePagesGridSpec, Pallas.GridSpec.numScalarPrefetch],
        _, _, rfl, rfl, rfl, rfl, rfl, fun _ _ _ => rfl⟩
    · intro i inp hi q outp ho heq
      rcases q with _ | q
      · simp only [writePagesGridSpec, List.getElem?_cons_zero, Option.some.injEq] at ho
        subst ho
        rcases i with _ | _ | i
        · simp only [writePagesGridSpec, List.getElem?_cons_zero, Option.some.injEq] at hi
          subst hi
          exact absurd heq.symm hbuf
        · simp [writePagesGridSpec, writePagesAliases, Pallas.GridSpec.numScalarPrefetch]
        · simp [writePagesGridSpec] at hi
      · simp [writePagesGridSpec] at ho
    · intro b hb outp houtp
      simp only [writePagesGridSpec, List.mem_singleton] at hb houtp
      subst hb houtp
      exact hids_buf
  · rfl
  · rfl
  · intro pg₁ h₁ pg₂ h₂ hne
    simp [writePagesGridSpec, Pallas.GridSpec.parGrid, Pallas.GridSpec.numParallel,
      Pallas.DimSemantics.isParallel, Pallas.gridPoints] at h₁ h₂
    exact absurd (h₁.trans h₂.symm) hne
  · intro pg₁ h₁ pg₂ h₂ hne
    simp [writePagesGridSpec, Pallas.GridSpec.parGrid, Pallas.GridSpec.numParallel,
      Pallas.DimSemantics.isParallel, Pallas.gridPoints] at h₁ h₂
    exact absurd (h₁.trans h₂.symm) hne
  · intro hne
    exact absurd rfl hne

/-- `writePagesSpec` has a well-formed layout whenever `kvBuf` is distinct from `pagesBuf` and
`idsBuf`. -/
theorem writePagesSpec_wellFormed (N M P : ℕ) (idsBuf pagesBuf kvBuf : BufId)
    (hkv_pages : kvBuf ≠ pagesBuf) (hkv_ids : kvBuf ≠ idsBuf) :
    (writePagesSpec N M P idsBuf pagesBuf kvBuf).WellFormed where
  readOut_on_footprint _ _ _ := rfl
  readSt_on_footprint _ _ hagree :=
    congrArg WritePagesSt.mk
      (Layout.read_eq_of_agreeOn _ (Layout.footprint_subset_buf _) hagree)
  readIn_outside_footprint _ _ hagree :=
    congrArg₂ WritePagesIn.mk
      (Layout.read_eq_of_agreeOutside _ (Layout.disjoint_bufFootprint _ hkv_pages.symm) hagree)
      (funext fun i => hagree (idsBuf, i.val) (fun h => hkv_ids h.symm))

/-- Main correctness theorem: `write_pages` is well defined and satisfies its `KernelSpec`
when ids are distinct non-negative int32s below `N`, buffers are distinct and pages are non-empty. -/
theorem writePages_correct (N M P : ℕ) (idsBuf pagesBuf kvBuf : BufId) (hP : 0 < P) :
    (writePagesSpec N M P idsBuf pagesBuf kvBuf).Correct
      (writePagesRun N M P idsBuf pagesBuf kvBuf) := by
  intro junk m0 hreq
  rcases hreq with ⟨hM, hids, hinj, hkv_pages, hkv_ids, _⟩
  refine ⟨writePagesSpec_wellFormed N M P idsBuf pagesBuf kvBuf hkv_pages hkv_ids, ?_⟩
  have hids' : ∀ i : Fin M, ∃ n : ℕ, n < N ∧ decodeInt32Nat (m0.read idsBuf i.val) = some n := hids
  have hinj' : Function.Injective fun i : Fin M => m0.read idsBuf i.val := hinj
  have hwf := writePages_wellDefined N M P idsBuf pagesBuf kvBuf hM hP m0 hids' hinj'
    hkv_pages hkv_ids
  have hdec : ∀ k (j : ℕ), k < M →
      ((decodeInt32Nat (m0.read idsBuf k)).getD 0 = j ↔ m0.read idsBuf k = j) := by
    intro k j hk
    obtain ⟨n, -, hdn⟩ := hids' ⟨k, hk⟩
    have ⟨hxn, _⟩ := (decodeInt32Nat_eq_some_iff (m0.read idsBuf k) n).mp hdn
    rw [hdn]
    dsimp only [Option.getD]
    rw [← hxn]
    constructor <;> intro h <;> exact_mod_cast h
  refine ⟨writePagesExec N M P idsBuf pagesBuf kvBuf junk m0,
    Pallas.pallasCall_of_wellDefined _ hwf, rfl, ?_, ?_⟩
  · dsimp only [writePagesSpec, writePagesMut]
    congr 1
    apply Tensor.ext2
    intro j p
    simp only [Tensor.get2_of2]
    rw [read_rowMajor_get2, writePagesExec_eq_seq N M P idsBuf pagesBuf kvBuf hkv_pages junk m0]
    split_ifs with hex
    · rw [read_rowMajor_get2]
      let i := Classical.choose hex
      have hi_eq : m0.read idsBuf i.val = j.val := Classical.choose_spec hex
      rcases range_split M i.val i.isLt with ⟨l₁, l₂, hrange, hl₂⟩
      rw [hrange]
      exact writePagesSeq_read_last N P pagesBuf kvBuf
        (fun k => (decodeInt32Nat (m0.read idsBuf k)).getD 0) m0 m0 l₁ l₂ i.val j p
        ((hdec i.val j.val i.isLt).mpr hi_eq)
        (fun k hk heq => by
          have ⟨hgt, hlt⟩ := hl₂ k hk
          have hx : m0.read idsBuf k = j.val := (hdec k j.val hlt).mp heq
          have hfin : (⟨k, hlt⟩ : Fin M) = i := hinj' (hx.trans hi_eq.symm)
          have hval : k = i.val := congrArg Fin.val hfin
          omega)
    · rw [read_rowMajor_get2]
      exact writePagesSeq_read_none N P pagesBuf kvBuf
        (fun k => (decodeInt32Nat (m0.read idsBuf k)).getD 0) m0 j p (List.range M) m0
        (fun k hk heq => hex ⟨⟨k, List.mem_range.mp hk⟩,
          (hdec k j.val (List.mem_range.mp hk)).mp heq⟩)
  · have hframe := Pallas.pallasExec_agreeOutside_outBufs
      (writePagesGridSpec N M P idsBuf pagesBuf kvBuf) writePagesBody junk m0
    refine Mem.agreeOutside_mono ?_ hframe
    rintro a ⟨x, hx, hbuf⟩
    simp only [writePagesGridSpec, List.mem_singleton] at hx
    subst hx
    exact hbuf.symm

end

end TpuKernelLean.Examples.KvPageWrite
