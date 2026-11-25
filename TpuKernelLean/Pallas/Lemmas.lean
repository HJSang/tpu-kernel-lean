import TpuKernelLean.Pallas.Core

/-!
Reusable lifting lemmas for `pallasExec`.

For a kernel whose body ignores its previous output blocks and whose outputs live in distinct
buffers that no input reads, `pallasExec_read_out` says each output buffer ends up as if every
grid step stored its block immediately, in grid order (`seqOut`), with inputs read from the
initial memory (`outVal`). The delayed write-back of the pipeline is invisible. The
`foldl_storeBlock_read_*` lemmas then read one address out of that sequence of stores.
-/

namespace TpuKernelLean.Pallas

noncomputable section

/-- Write back one output's previous VMEM block before the grid point with block index `newIdx`. -/
def flushOne (o : ArrayDesc × BlockSpec) (newIdx : List ℕ) : OutSlot → Mem → Mem
  | some (oldIdx, oldVal), m =>
    if oldIdx = newIdx then m else storeBlock o.1 o.2.blockShape oldIdx oldVal m
  | none, m => m

/-- Write back one output's VMEM block unconditionally, as at the end of the grid. -/
def flushFinal (o : ArrayDesc × BlockSpec) : OutSlot → Mem → Mem
  | some (oldIdx, oldVal), m => storeBlock o.1 o.2.blockShape oldIdx oldVal m
  | none, m => m

theorem storeBlock_read_congr (arr : ArrayDesc) (bs idx : List ℕ) (v : List ℕ → ℝ)
    {m m' : Mem} (b : BufId) (a : ℕ) (h : m.read b a = m'.read b a) :
    (storeBlock arr bs idx v m).read b a = (storeBlock arr bs idx v m').read b a := by
  simp only [storeBlock, Mem.read] at *
  split_ifs <;> simp_all

theorem flushOne_read_congr (o : ArrayDesc × BlockSpec) (newIdx : List ℕ) (s : OutSlot)
    {m m' : Mem} (b : BufId) (a : ℕ) (h : m.read b a = m'.read b a) :
    (flushOne o newIdx s m).read b a = (flushOne o newIdx s m').read b a := by
  rcases s with _ | ⟨oldIdx, oldVal⟩
  · exact h
  · simp only [flushOne]
    split_ifs
    · exact h
    · exact storeBlock_read_congr _ _ _ _ b a h

theorem flushFinal_read_congr (o : ArrayDesc × BlockSpec) (s : OutSlot)
    {m m' : Mem} (b : BufId) (a : ℕ) (h : m.read b a = m'.read b a) :
    (flushFinal o s m).read b a = (flushFinal o s m').read b a := by
  rcases s with _ | ⟨oldIdx, oldVal⟩
  · exact h
  · exact storeBlock_read_congr _ _ _ _ b a h

theorem flushOne_read_other_buf (o : ArrayDesc × BlockSpec) (newIdx : List ℕ) (s : OutSlot)
    (m : Mem) (b : BufId) (a : ℕ) (h : b ≠ o.1.buf) :
    (flushOne o newIdx s m).read b a = m.read b a := by
  rcases s with _ | ⟨oldIdx, oldVal⟩
  · rfl
  · simp only [flushOne]
    split_ifs
    · rfl
    · exact storeBlock_read_other_buf _ _ _ _ _ b a h

theorem flushFinal_read_other_buf (o : ArrayDesc × BlockSpec) (s : OutSlot)
    (m : Mem) (b : BufId) (a : ℕ) (h : b ≠ o.1.buf) :
    (flushFinal o s m).read b a = m.read b a := by
  rcases s with _ | ⟨oldIdx, oldVal⟩
  · rfl
  · exact storeBlock_read_other_buf _ _ _ _ _ b a h

/-- Outside the new block, writing back conditionally and unconditionally agree. -/
theorem flushOne_read_eq_flushFinal (o : ArrayDesc × BlockSpec) (newIdx : List ℕ) (s : OutSlot)
    (m : Mem) (a : ℕ) (ha : addrInBlock o.1 o.2.blockShape newIdx a = false) :
    (flushOne o newIdx s m).read o.1.buf a = (flushFinal o s m).read o.1.buf a := by
  rcases s with _ | ⟨oldIdx, oldVal⟩
  · rfl
  · simp only [flushOne, flushFinal]
    split_ifs with h
    · subst h
      exact (storeBlock_read_outside _ _ _ _ _ _ _ (Or.inr ha)).symm
    · rfl

private theorem buf_ne_of_nodup {specs : List (ArrayDesc × BlockSpec)} {o' : ArrayDesc × BlockSpec}
    (hnd : ((o' :: specs).map fun p => p.1.buf).Nodup) : ∀ x ∈ specs, x.1.buf ≠ o'.1.buf := by
  intro x hx heq
  simp only [List.map_cons, List.nodup_cons, List.mem_map] at hnd
  exact hnd.1 ⟨x, hx, heq⟩

/-- `flushAndPrep` returns the block index of every output at `g`. -/
theorem flushAndPrep_idxs (g : List ℕ) (sc : ℕ → ℤ) (junk : ℕ → ℝ) :
    ∀ (idx : ℕ) (specs : List (ArrayDesc × BlockSpec)) (slots : List OutSlot) (m : Mem),
      (flushAndPrep g sc junk idx specs slots m).2.1 = specs.map fun o => o.2.blockIdx g sc
  | _, [], _, _ => rfl
  | idx, o :: rest, slots, m => by
    simp only [flushAndPrep, List.map_cons]
    rw [flushAndPrep_idxs g sc junk (idx + 1) rest]

/-- Reading output `o`'s buffer after `flushAndPrep` sees only `o`'s own write-back. -/
theorem flushAndPrep_read_at (g : List ℕ) (sc : ℕ → ℤ) (junk : ℕ → ℝ) :
    ∀ (specs : List (ArrayDesc × BlockSpec)) (idx : ℕ) (slots : List OutSlot) (m : Mem)
      (q : ℕ) (o : ArrayDesc × BlockSpec), specs[q]? = some o →
      (specs.map fun p => p.1.buf).Nodup → ∀ a,
      (flushAndPrep g sc junk idx specs slots m).1.read o.1.buf a =
        (flushOne o (o.2.blockIdx g sc) (slots.getD q none) m).read o.1.buf a
  | [], _, _, _, q, o, hq, _, _ => by simp at hq
  | o' :: rest, idx, slots, m, q, o, hq, hnd, a => by
    have hrest : ∀ x ∈ rest, x.1.buf ≠ o'.1.buf := buf_ne_of_nodup hnd
    simp only [flushAndPrep]
    cases q with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hq
      subst hq
      rw [flushAndPrep_read_other_buf g sc junk _ a (idx + 1) rest _ _ hrest]
      rcases slots with _ | ⟨s, ss⟩
      · rfl
      · rcases s with _ | ⟨oldIdx, oldVal⟩ <;> rfl
    | succ q' =>
      simp only [List.getElem?_cons_succ] at hq
      have hnd' : (rest.map fun p => p.1.buf).Nodup := (List.nodup_cons.mp hnd).2
      have hne : o.1.buf ≠ o'.1.buf := hrest o (List.mem_of_getElem? hq)
      rw [flushAndPrep_read_at g sc junk rest (idx + 1) _ _ q' o hq hnd' a]
      rcases slots with _ | ⟨s, ss⟩
      · rfl
      · apply flushOne_read_congr
        rcases s with _ | ⟨oldIdx, oldVal⟩
        · rfl
        · dsimp only
          split_ifs
          · rfl
          · exact storeBlock_read_other_buf _ _ _ _ _ _ a hne

/-- Reading output `o`'s buffer after `flushAll` sees only `o`'s own write-back. -/
theorem flushAll_read_at :
    ∀ (specs : List (ArrayDesc × BlockSpec)) (slots : List OutSlot) (m : Mem)
      (q : ℕ) (o : ArrayDesc × BlockSpec), specs[q]? = some o →
      (specs.map fun p => p.1.buf).Nodup → ∀ a,
      (flushAll specs slots m).read o.1.buf a = (flushFinal o (slots.getD q none) m).read o.1.buf a
  | [], _, _, q, o, hq, _, _ => by simp at hq
  | o' :: rest, slots, m, q, o, hq, hnd, a => by
    have hrest : ∀ x ∈ rest, x.1.buf ≠ o'.1.buf := buf_ne_of_nodup hnd
    simp only [flushAll]
    cases q with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hq
      subst hq
      rw [flushAll_read_other_buf _ a rest _ _ hrest]
      rcases slots with _ | ⟨s, ss⟩
      · rfl
      · rcases s with _ | ⟨oldIdx, oldVal⟩ <;> rfl
    | succ q' =>
      simp only [List.getElem?_cons_succ] at hq
      have hnd' : (rest.map fun p => p.1.buf).Nodup := (List.nodup_cons.mp hnd).2
      have hne : o.1.buf ≠ o'.1.buf := hrest o (List.mem_of_getElem? hq)
      rw [flushAll_read_at rest _ _ q' o hq hnd' a]
      rcases slots with _ | ⟨s, ss⟩
      · rfl
      · apply flushFinal_read_congr
        rcases s with _ | ⟨oldIdx, oldVal⟩
        · rfl
        · exact storeBlock_read_other_buf _ _ _ _ _ _ a hne

theorem makeSlots_getD :
    ∀ (idxs : List (List ℕ)) (vals : List (List ℕ → ℝ)) (q : ℕ) (i : List ℕ),
      idxs[q]? = some i → (makeSlots idxs vals).getD q none = some (i, vals.getD q fun _ => 0)
  | [], _, q, i, hq => by simp at hq
  | idx :: idxs, [], 0, i, hq => by simp_all [makeSlots]
  | idx :: idxs, [], q + 1, i, hq => by
    simp only [List.getElem?_cons_succ] at hq
    simp only [makeSlots, List.getD_cons_succ]
    rw [makeSlots_getD idxs [] q i hq]
    simp
  | idx :: idxs, v :: vals, 0, i, hq => by simp_all [makeSlots]
  | idx :: idxs, v :: vals, q + 1, i, hq => by
    simp only [List.getElem?_cons_succ] at hq
    simp only [makeSlots, List.getD_cons_succ]
    exact makeSlots_getD idxs vals q i hq

theorem makeSlots_length :
    ∀ (idxs : List (List ℕ)) (vals : List (List ℕ → ℝ)), (makeSlots idxs vals).length = idxs.length
  | [], _ => rfl
  | _ :: idxs, [] => by simp [makeSlots, makeSlots_length idxs []]
  | _ :: idxs, _ :: vals => by simp [makeSlots, makeSlots_length idxs vals]

/-- Inputs of grid point `g` read from memory `m`, with the junk tags `stepGrid` uses. -/
def loadIns (gs : GridSpec) (junk : ℕ → ℝ) (m : Mem) (g : List ℕ) : List (List ℕ → ℝ) :=
  gs.inSpecs.mapIdx fun i p =>
    loadBlock p.1 p.2.blockShape (p.2.blockIdx g (gs.scalarPrefetch m))
      (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) (i + 1))) m

/-- Block written to output `q` at grid point `g` by a body that ignores its previous output
blocks, with inputs read from the initial memory `m0`. -/
def outVal (gs : GridSpec) (body : Body) (junk : ℕ → ℝ) (m0 : Mem) (q : ℕ) (g : List ℕ) :
    List ℕ → ℝ :=
  (body g (gs.scalarPrefetch m0) (loadIns gs junk m0 g) []).getD q fun _ => 0

/-- Store every block of output `o` immediately, in grid order. -/
def seqOut (o : ArrayDesc × BlockSpec) (val : List ℕ → List ℕ → ℝ) (sc : ℕ → ℤ)
    (pts : List (List ℕ)) (m : Mem) : Mem :=
  pts.foldl (fun m g => storeBlock o.1 o.2.blockShape (o.2.blockIdx g sc) (val g) m) m

/-- Assumptions of `pallasExec_read_out`: outputs in distinct buffers that no input reads, and a
body that ignores its previous output blocks (it writes every output block from scratch). -/
structure StoreOnly (gs : GridSpec) (body : Body) : Prop where
  outs_nodup : (gs.outSpecs.map fun p => p.1.buf).Nodup
  ins_disjoint : ∀ p ∈ gs.inSpecs, ∀ o ∈ gs.outSpecs, p.1.buf ≠ o.1.buf
  ignores_prev : ∀ g s ins outs, body g s ins outs = body g s ins []

/-- `loadIns` depends only on the contents of the input buffers. -/
theorem loadIns_congr (gs : GridSpec) (junk : ℕ → ℝ) (sc : ℕ → ℤ) (m m' : Mem) (g : List ℕ)
    (hm : ∀ p ∈ gs.inSpecs, ∀ a, m.read p.1.buf a = m'.read p.1.buf a) :
    (gs.inSpecs.mapIdx fun i p =>
      loadBlock p.1 p.2.blockShape (p.2.blockIdx g sc)
        (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) (i + 1))) m) =
    (gs.inSpecs.mapIdx fun i p =>
      loadBlock p.1 p.2.blockShape (p.2.blockIdx g sc)
        (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) (i + 1))) m') := by
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp only [List.getElem_mapIdx]
  exact loadBlock_congr_mem _ _ _ _ (hm _ (List.getElem_mem _))

private theorem fold_read_out (gs : GridSpec) (body : Body) (junk : ℕ → ℝ) (m0 : Mem)
    (hs : StoreOnly gs body) (q : ℕ) (o : ArrayDesc × BlockSpec) (hq : gs.outSpecs[q]? = some o) :
    ∀ (pts : List (List ℕ)) (m : Mem) (slots : List OutSlot) (mseq : Mem),
      (∀ b, (∀ x ∈ gs.outSpecs, x.1.buf ≠ b) → ∀ a, m.read b a = m0.read b a) →
      (∀ a, (flushFinal o (slots.getD q none) m).read o.1.buf a = mseq.read o.1.buf a) →
      ∀ a,
        (flushFinal o ((pts.foldl (stepGrid gs body (gs.scalarPrefetch m0) junk) (m, slots)).2.getD
            q none) (pts.foldl (stepGrid gs body (gs.scalarPrefetch m0) junk) (m, slots)).1).read
          o.1.buf a =
        (seqOut o (outVal gs body junk m0 q) (gs.scalarPrefetch m0) pts mseq).read o.1.buf a
  | [], _, _, _, _, hinv, a => hinv a
  | g :: rest, m, slots, mseq, hframe, hinv, a => by
    have ho : o ∈ gs.outSpecs := List.mem_of_getElem? hq
    set sc := gs.scalarPrefetch m0
    set step := stepGrid gs body sc junk (m, slots) g with hstep
    have hframe' : ∀ b, (∀ x ∈ gs.outSpecs, x.1.buf ≠ b) → ∀ a, step.1.read b a = m0.read b a :=
      fun b hb a => (stepGrid_read_other_buf gs body sc junk b a hb (m, slots) g).trans
        (hframe b hb a)
    -- The value stored for `o` at `g` equals `outVal`, since inputs still hold `m0`.
    have hval : ∀ outs, (body g sc
        (gs.inSpecs.mapIdx fun i p => loadBlock p.1 p.2.blockShape (p.2.blockIdx g sc)
          (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) (i + 1)))
          (flushAndPrep g sc (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) 0)) 0
            gs.outSpecs slots m).1) outs).getD q (fun _ => 0) = outVal gs body junk m0 q g := by
      intro outs
      rw [hs.ignores_prev, outVal, loadIns]
      congr 2
      apply loadIns_congr
      intro p hp a
      rw [flushAndPrep_read_other_buf _ _ _ _ a 0 _ _ _
        (fun x hx heq => hs.ins_disjoint p hp x hx heq.symm)]
      exact hframe _ (fun x hx heq => hs.ins_disjoint p hp x hx heq.symm) a
    have hslot : step.2.getD q none =
        some (o.2.blockIdx g sc, outVal gs body junk m0 q g) := by
      have hqlt : q < gs.outSpecs.length := by
        by_contra h
        rw [List.getElem?_eq_none (by omega)] at hq
        cases hq
      rw [hstep]
      simp only [stepGrid]
      rw [List.getD_append _ _ _ _ (by rw [makeSlots_length, flushAndPrep_idxs]; simpa using hqlt),
        makeSlots_getD _ _ q (o.2.blockIdx g sc) (by rw [flushAndPrep_idxs]; simp [hq]), hval]
    have hinv' : ∀ a, (flushFinal o (step.2.getD q none) step.1).read o.1.buf a =
        (storeBlock o.1 o.2.blockShape (o.2.blockIdx g sc) (outVal gs body junk m0 q g)
          mseq).read o.1.buf a := by
      intro a
      rw [hslot]
      simp only [flushFinal]
      cases hin : addrInBlock o.1 o.2.blockShape (o.2.blockIdx g sc) a
      · rw [storeBlock_read_outside _ _ _ _ _ _ _ (Or.inr hin),
          storeBlock_read_outside _ _ _ _ _ _ _ (Or.inr hin)]
        have h1 : step.1.read o.1.buf a =
            (flushOne o (o.2.blockIdx g sc) (slots.getD q none) m).read o.1.buf a := by
          rw [hstep]
          exact flushAndPrep_read_at g sc _ gs.outSpecs 0 slots m q o hq hs.outs_nodup a
        rw [h1, flushOne_read_eq_flushFinal o _ _ m a hin]
        exact hinv a
      · rw [storeBlock_read_inside _ _ _ _ _ _ hin, storeBlock_read_inside _ _ _ _ _ _ hin]
    have ih := fold_read_out gs body junk m0 hs q o hq rest step.1 step.2
      (storeBlock o.1 o.2.blockShape (o.2.blockIdx g sc) (outVal gs body junk m0 q g) mseq)
      hframe' hinv' a
    simpa only [List.foldl_cons, seqOut] using ih

/-- Each output buffer of a `StoreOnly` kernel ends up as the in-order sequence of its blocks
(`seqOut`) applied to the initialized memory. -/
theorem pallasExec_read_out (gs : GridSpec) (body : Body)
    (junk : ℕ → ℝ) (m0 : Mem) (hs : StoreOnly gs body) (q : ℕ) (o : ArrayDesc × BlockSpec)
    (hq : gs.outSpecs[q]? = some o) (a : ℕ) :
    (pallasExec gs body junk m0).read o.1.buf a =
      (seqOut o (outVal gs body junk m0 q) (gs.scalarPrefetch m0) (gridPoints gs.grid)
        (initOutMem gs (junkAt junk (Nat.pair 0 0)) m0)).read
        o.1.buf a := by
  simp only [pallasExec]
  rw [flushAll_read_at _ _ _ q o hq hs.outs_nodup a]
  apply fold_read_out gs body junk m0 hs q o hq
  · intro b hb a
    exact initOutMemAux_read_other_buf _ _ b a 0 _ m0 hb
  · intro a
    simp [flushFinal]

/-- A sequence of block stores leaves an address no block covers unchanged. -/
theorem foldl_storeBlock_read_none {α : Type*} (arr : ArrayDesc) (bs : List ℕ)
    (idx : α → List ℕ) (val : α → List ℕ → ℝ) (a : ℕ) :
    ∀ (l : List α) (m : Mem), (∀ x ∈ l, addrInBlock arr bs (idx x) a = false) →
      (l.foldl (fun m x => storeBlock arr bs (idx x) (val x) m) m).read arr.buf a =
        m.read arr.buf a
  | [], _, _ => rfl
  | x :: rest, m, h => by
    simp only [List.foldl_cons]
    rw [foldl_storeBlock_read_none arr bs idx val a rest _ (fun y hy => h y (by simp [hy])),
      storeBlock_read_outside _ _ _ _ _ _ _ (Or.inr (h x (by simp)))]

/-- An address covered by block `x` and by no later block reads `x`'s value. -/
theorem foldl_storeBlock_read_last {α : Type*} (arr : ArrayDesc) (bs : List ℕ)
    (idx : α → List ℕ) (val : α → List ℕ → ℝ) (a : ℕ) (l₁ l₂ : List α) (x : α) (m : Mem)
    (hx : addrInBlock arr bs (idx x) a = true)
    (h₂ : ∀ y ∈ l₂, addrInBlock arr bs (idx y) a = false) :
    ((l₁ ++ x :: l₂).foldl (fun m x => storeBlock arr bs (idx x) (val x) m) m).read arr.buf a =
      val x (localCoord bs (idx x) (unflatCoord arr.shape a)) := by
  rw [List.foldl_append, List.foldl_cons, foldl_storeBlock_read_none arr bs idx val a l₂ _ h₂,
    storeBlock_read_inside _ _ _ _ _ _ hx]

/-- Split `range n` at `i < n`, with every later element above `i`. -/
theorem range_split (n i : ℕ) (hi : i < n) :
    ∃ l₁ l₂ : List ℕ, List.range n = l₁ ++ i :: l₂ ∧ ∀ k ∈ l₂, i < k ∧ k < n := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [List.range_succ]
    by_cases h : i = n
    · subst h
      exact ⟨List.range i, [], by simp, by simp⟩
    · obtain ⟨l₁, l₂, heq, hl₂⟩ := ih (by omega)
      refine ⟨l₁, l₂ ++ [n], by rw [heq]; simp, ?_⟩
      intro k hk
      simp only [List.mem_append, List.mem_singleton] at hk
      rcases hk with hk | rfl
      · exact ⟨(hl₂ k hk).1, by have := (hl₂ k hk).2; omega⟩
      · exact ⟨by omega, by omega⟩

end

end TpuKernelLean.Pallas
