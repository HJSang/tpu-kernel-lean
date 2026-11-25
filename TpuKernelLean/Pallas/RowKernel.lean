import TpuKernelLean.Pallas.Lemmas

/-!
Row-wise kernels: `pallas_call`s that tile `[M, H]` arrays into `[rows, H]` row blocks over a
`cdiv(M, rows)` grid, broadcast one `[1, H]` weight block, and compute each output row from the
same row of every input. `RowKernel.correct` proves, for every `M`, `rows`, junk and memory, that
the run is defined, each output row is `f` of the matching input rows, and nothing else changes.
-/

namespace TpuKernelLean.Pallas

/-- A row-wise kernel of row width `H`: `nIn` row inputs, one weight row, `nOut` row outputs, and
the function computing every output row from the input rows and the weight row. -/
structure RowKernel (H : ℕ) where
  nIn : ℕ
  nOut : ℕ
  f : (Fin nIn → Fin H → ℝ) → (Fin H → ℝ) → Fin nOut → Fin H → ℝ

/-- `[rows, H]` row block `i` of an `[M, H]` array (`pl.BlockSpec((rows, h), lambda i: (i, 0))`). -/
def rowBlockSpec (rows H : ℕ) : BlockSpec :=
  { blockShape := [rows, H], indexMap := fun g _ => [((g.getD 0 0 : ℕ) : ℤ), 0] }

/-- The weight row (`pl.BlockSpec((1, h), lambda i: (0, 0))`). -/
def weightBlockSpec (H : ℕ) : BlockSpec :=
  { blockShape := [1, H], indexMap := fun _ _ => [0, 0] }

/-- 1-D broadcast block (`pl.BlockSpec((W,), lambda i: (0,))`). -/
def vecBlockSpec (W : ℕ) : BlockSpec :=
  { blockShape := [W], indexMap := fun _ _ => [0] }

namespace RowKernel

variable {H : ℕ} (k : RowKernel H)

/-- The `pallas_call` grid of the kernel on `M` rows with row blocks of `rows` rows. -/
def gridSpec (M rows : ℕ) (inBufs : Fin k.nIn → BufId) (wBuf : BufId)
    (outBufs : Fin k.nOut → BufId) : GridSpec :=
  { grid := [(M + rows - 1) / rows]
    inSpecs := List.ofFn (fun i => (⟨inBufs i, [M, H]⟩, rowBlockSpec rows H)) ++
      [(⟨wBuf, [1, H]⟩, weightBlockSpec H)]
    outSpecs := List.ofFn fun q => (⟨outBufs q, [M, H]⟩, rowBlockSpec rows H)
    dimSemantics := [.parallel] }

/-- The kernel body: output `q` at local row `r`, column `c` is `f` of local row `r` of every input
block and of the weight block. -/
def body : Body := fun _ _ ins _ =>
  List.ofFn fun q : Fin k.nOut => fun loc =>
    if hc : loc.getD 1 0 < H then
      k.f (fun i c' => (ins.getD i.val fun _ => 0) [loc.getD 0 0, c'.val])
        (fun c' => (ins.getD k.nIn fun _ => 0) [0, c'.val]) q ⟨loc.getD 1 0, hc⟩
    else 0

end RowKernel

theorem gridPoints_one (G : ℕ) : gridPoints [G] = (List.range G).map fun i => [i] := by
  simp only [gridPoints, List.map_cons, List.map_nil]
  induction (List.range G) with
  | nil => rfl
  | cons x xs ih => simp [ih]

theorem gridPoints_one_getD (G k : ℕ) (hk : k < G) : (gridPoints [G]).getD k [] = [k] := by
  rw [gridPoints_one]
  simp [hk]

/-- An in-bounds element `(r, c)` of an `[M, H]` array lies in row block `r / rows`. -/
theorem addrInBlock_rowBlock {buf : BufId} {M H rows r c i : ℕ} (hrows : 0 < rows) (hr : r < M)
    (hc : c < H) :
    addrInBlock ⟨buf, [M, H]⟩ [rows, H] [i, 0] (r * H + c) = decide (i = r / rows) := by
  have hlt : r * H + c < M * H := by nlinarith
  have key : (i * rows ≤ r ∧ r < (i + 1) * rows) ↔ i = r / rows := by
    constructor
    · rintro ⟨h1, h2⟩
      exact (Nat.div_eq_of_lt_le h1 h2).symm
    · rintro rfl
      refine ⟨Nat.div_mul_le_self r rows, ?_⟩
      rw [Nat.succ_mul]
      exact Nat.lt_div_mul_add hrows
  simp [addrInBlock, unflatCoord_two M H r c hc, coordInBlock, hlt, hc, key]

/-- `cdiv(M, rows)` is strictly positive whenever `M` and `rows` are strictly positive. -/
theorem cdiv_pos {M rows : ℕ} (hM : 0 < M) (hrows : 0 < rows) : 0 < (M + rows - 1) / rows :=
  Nat.div_pos (by omega) hrows

/-- Every grid index below `cdiv(M, rows)` starts strictly below `M`. -/
theorem rowBlock_lt_of_cdiv {M rows j : ℕ} (hrows : 0 < rows)
    (hj : j < (M + rows - 1) / rows) : j * rows < M := by
  have h := (Nat.le_div_iff_mul_le hrows).mp hj
  rw [Nat.succ_mul] at h
  omega

/-- Any row `r < M` falls into a grid block `r / rows < cdiv(M, rows)`. -/
theorem div_lt_cdiv {M rows r : ℕ} (hrows : 0 < rows) (hr : r < M) :
    r / rows < (M + rows - 1) / rows := by
  apply (Nat.le_div_iff_mul_le hrows).mpr
  have := Nat.div_mul_le_self r rows
  rw [Nat.succ_mul]
  have h2 := Nat.lt_div_mul_add (a := r) hrows
  omega

/-- Reading input `i` from `loadIns` yields the `loadBlock` of `gs.inSpecs[i]?`. -/
theorem loadIns_getD (gs : GridSpec) (junk : ℕ → ℝ) (m0 : Mem) (g : List ℕ) (i : ℕ)
    (p : ArrayDesc × BlockSpec) (hp : gs.inSpecs[i]? = some p) :
    (loadIns gs junk m0 g).getD i (fun _ => 0) =
      loadBlock p.1 p.2.blockShape (p.2.blockIdx g (gs.scalarPrefetch m0))
        (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) (i + 1))) m0 := by
  simp [loadIns, List.getD_eq_getElem?_getD, List.getElem?_mapIdx, hp]

/-- Reading local coordinate `(r % rows, c)` of row block `r / rows` of a `[M, W]` array with
block width `bw ≤ W` fetches HBM element `(r, c)` independently of `junk`. -/
theorem loadBlock_rowBlock_eq {buf : BufId} {M W rows bw r c : ℕ} (hrows : 0 < rows)
    (hr : r < M) (hc : c < bw) (hbw : bw ≤ W) (junk : ℕ → ℝ) (m : Mem) :
    loadBlock ⟨buf, [M, W]⟩ [rows, bw] [r / rows, 0] junk m [r - r / rows * rows, c] =
      m.read buf (r * W + c) := by
  have hrl : r - r / rows * rows < rows := by
    have := Nat.lt_div_mul_add (a := r) hrows
    omega
  have hglob : r / rows * rows + (r - r / rows * rows) = r := by
    have := Nat.div_mul_le_self r rows
    omega
  have hcW : c < W := Nat.lt_of_lt_of_le hc hbw
  rw [loadBlock_eq_of_inBounds]
  · simp [globalCoord, flatCoord, hglob]
  · simp [inBounds, hrl, hc]
  · simp [globalCoord, inBounds, hglob, hr, hcW]

/-- Columns at or past `bw` of a `[M, W]` array are never covered by a `[rows, bw]` row block
anchored at column 0. -/
theorem addrInBlock_rowBlock_col_ge {buf : BufId} {M W rows bw i r c : ℕ}
    (hbw : bw ≤ c) (hc : c < W) :
    addrInBlock ⟨buf, [M, W]⟩ [rows, bw] [i, 0] (r * W + c) = false := by
  simp [addrInBlock, unflatCoord_two M W r c hc, coordInBlock, hbw]

/-- A `[rows, bw]` row block anchored at column 0 of a `[M, W]` array depends only on columns
below `bw`. -/
theorem loadBlock_rowBlock_congr_prefix {buf : BufId} {M W rows bw i : ℕ}
    (junk : ℕ → ℝ) {m₁ m₂ : Mem}
    (h : ∀ r < M, ∀ c < bw, m₁.read buf (r * W + c) = m₂.read buf (r * W + c)) :
    loadBlock ⟨buf, [M, W]⟩ [rows, bw] [i, 0] junk m₁ =
      loadBlock ⟨buf, [M, W]⟩ [rows, bw] [i, 0] junk m₂ := by
  funext loc
  simp only [loadBlock]
  split_ifs with hcond
  · rcases loc with _ | ⟨rl, _ | ⟨cl, _ | ⟨_, _⟩⟩⟩
    · simp [inBounds] at hcond
    · simp [inBounds] at hcond
    · simp only [Bool.and_eq_true, inBounds, globalCoord, Bool.and_true,
        decide_eq_true_eq] at hcond
      rcases hcond with ⟨⟨_, hcl⟩, ⟨hr, _⟩⟩
      simp only [globalCoord, flatCoord, zero_mul, zero_add, List.prod_cons, List.prod_nil,
        mul_one, add_zero]
      exact h (i * rows + rl) hr cl hcl
    · simp [inBounds] at hcond
  · rfl

/-- Reading an in-bounds coordinate `c < W` of a 1-D `[W]` broadcast block fetches HBM element
`c` independently of `junk`. -/
theorem loadBlock_vecBlock_eq {buf : BufId} {W c : ℕ} (hc : c < W)
    (junk : ℕ → ℝ) (m : Mem) :
    loadBlock ⟨buf, [W]⟩ [W] [0] junk m [c] = m.read buf c := by
  rw [loadBlock_eq_of_inBounds]
  · simp [globalCoord, flatCoord]
  · simp [inBounds, hc]
  · simp [globalCoord, inBounds, hc]

/-- A `rowBlockSpec` output block is never revisited across grid steps `0 .. G - 1`. -/
theorem noRevisit_rowBlock (G rows W : ℕ) (sc : ℕ → ℤ) :
    noRevisit ((List.range G).map fun k =>
      (rowBlockSpec rows W).indexMap ((gridPoints [G]).getD k []) sc) = true := by
  apply noRevisit_of_nodup
  apply List.Nodup.map_on _ List.nodup_range
  intro a ha b hb hab
  simp only [List.mem_range] at ha hb
  simp only [rowBlockSpec, gridPoints_one_getD _ a ha, gridPoints_one_getD _ b hb] at hab
  simpa using hab

/-- Reading `(r, c)` after storing every `rowBlockSpec rows W` block of an `[M, W]` output in
grid order returns the value written by step `r / rows`. -/
theorem seqOut_rowBlock_read {buf : BufId} {M W rows : ℕ} (hrows : 0 < rows)
    (val : List ℕ → List ℕ → ℝ) (sc : ℕ → ℤ) (m : Mem) (r : Fin M) (c : Fin W) :
    (seqOut (⟨buf, [M, W]⟩, rowBlockSpec rows W) val sc
      (gridPoints [(M + rows - 1) / rows]) m).read buf (r.val * W + c.val) =
      val [r.val / rows] [r.val - r.val / rows * rows, c.val] := by
  obtain ⟨l₁, l₂, hsplit, hl₂⟩ := range_split _ (r.val / rows) (div_lt_cdiv hrows r.isLt)
  simp only [seqOut, gridPoints_one, List.foldl_map, hsplit]
  have hidx : ∀ i : ℕ, (rowBlockSpec rows W).blockIdx [i] sc = [i, 0] := fun i => by
    simp [BlockSpec.blockIdx, rowBlockSpec]
  simp only [hidx]
  refine (foldl_storeBlock_read_last ⟨buf, [M, W]⟩ [rows, W] (fun i => [i, 0])
    (fun i => val [i]) (r.val * W + c.val) l₁ l₂ (r.val / rows) m
    (by rw [addrInBlock_rowBlock hrows r.isLt c.isLt]; simp)
    (by
      intro y hy
      rw [addrInBlock_rowBlock hrows r.isLt c.isLt]
      have := (hl₂ y hy).1
      simp only [decide_eq_false_iff_not]
      omega)).trans ?_
  rw [unflatCoord_two M W r.val c.val c.isLt]
  simp [localCoord]

namespace RowKernel

variable {H : ℕ} (k : RowKernel H)

theorem wellDefined (M rows : ℕ) (hM : 0 < M) (hrows : 0 < rows) (hH : 0 < H) (inBufs : Fin k.nIn → BufId)
    (wBuf : BufId) (outBufs : Fin k.nOut → BufId) (hout : Function.Injective outBufs)
    (hdisj : ∀ q i, outBufs q ≠ inBufs i) (hw : ∀ q, outBufs q ≠ wBuf) (m0 : Mem) :
    (k.gridSpec M rows inBufs wBuf outBufs).WellDefined k.body m0 := by
  have hlen : (gridPoints [(M + rows - 1) / rows]).length = (M + rows - 1) / rows := by
    simp [gridPoints_length]
  have hrow : ∀ j, j < (M + rows - 1) / rows → j * rows < M := by
    intro j hj
    have h := (Nat.le_div_iff_mul_le hrows).mp hj
    rw [Nat.succ_mul] at h
    omega
  constructor
  · intro d hd
    simp only [gridSpec, List.mem_singleton] at hd
    subst hd
    exact cdiv_pos hM hrows
  · intro p hp
    simp only [gridSpec, List.append_assoc, List.mem_append, List.mem_ofFn, List.mem_singleton]
      at hp
    rcases hp with ⟨i, rfl⟩ | rfl | ⟨q, rfl⟩ <;>
      simp [rowBlockSpec, weightBlockSpec, hrows, hH]
  · intro j hj p hp
    simp only [gridSpec, hlen] at hj
    simp only [gridSpec, List.append_assoc, List.mem_append, List.mem_ofFn, List.mem_singleton]
      at hp
    have hjz : (j : ℤ) * rows < M := by exact_mod_cast hrow j hj
    rcases hp with ⟨i, rfl⟩ | rfl | ⟨q, rfl⟩ <;>
      simp only [GridSpec.idxAt, gridSpec, gridPoints_one_getD _ j hj, rowBlockSpec,
        weightBlockSpec] <;>
      simp [blockOk, hrows, hH, hjz]
  · intro p hp
    simp only [gridSpec, List.mem_ofFn] at hp
    obtain ⟨q, rfl⟩ := hp
    apply noRevisit_of_nodup
    simp only [gridSpec, hlen]
    apply List.Nodup.map_on _ List.nodup_range
    intro a ha b hb hab
    simp only [List.mem_range] at ha hb
    simp only [GridSpec.idxAt, rowBlockSpec, gridPoints_one_getD _ a ha,
      gridPoints_one_getD _ b hb] at hab
    simpa using hab
  · intro _ _ _ _
    simp [body, gridSpec]
  · intro s t _ _ i hi o ho hov
    simp only [gridSpec, List.mem_append, List.mem_ofFn, List.mem_singleton] at hi ho
    obtain ⟨q, rfl⟩ := ho
    rcases hi with ⟨j, rfl⟩ | rfl
    · exact hdisj q j hov.1.symm
    · exact hw q hov.1.symm
  · exact GridSpec.AliasesValid.of_empty rfl
      (by simp only [gridSpec, List.map_ofFn]; exact List.nodup_ofFn.mpr hout)
      (by
        intro inp hinp outp houtp
        simp only [gridSpec, List.mem_append, List.mem_ofFn, List.mem_singleton] at hinp houtp
        obtain ⟨q, rfl⟩ := houtp
        rcases hinp with ⟨j, rfl⟩ | rfl
        · exact (hdisj q j).symm
        · exact (hw q).symm)
      (by simp [gridSpec])
  · rfl
  · rfl
  · intro pg₁ h₁ pg₂ h₂ hne ag₁ _ ag₂ _ o₁ ho₁ o₂ ho₂ hov
    simp only [gridSpec, GridSpec.parGrid, GridSpec.numParallel, List.takeWhile,
      DimSemantics.isParallel, List.length_cons, List.length_nil, List.take_succ_cons,
      List.take_zero, gridPoints_one, List.mem_map, List.mem_range] at h₁ h₂
    obtain ⟨i₁, _, rfl⟩ := h₁
    obtain ⟨i₂, _, rfl⟩ := h₂
    simp only [gridSpec, List.mem_ofFn] at ho₁ ho₂
    obtain ⟨q₁, rfl⟩ := ho₁
    obtain ⟨q₂, rfl⟩ := ho₂
    obtain ⟨_, a, ha₁, ha₂⟩ := hov
    simp only [BlockSpec.blockIdx, rowBlockSpec, List.cons_append, List.getD_cons_zero,
      List.map_cons, List.map_nil, Int.toNat_natCast, Int.toNat_zero] at ha₁ ha₂
    have hlt : a < M * H := by
      simp only [addrInBlock, Bool.and_eq_true, decide_eq_true_eq, List.prod_cons,
        List.prod_nil, mul_one] at ha₁
      exact ha₁.1
    have hr : a / H < M := (Nat.div_lt_iff_lt_mul hH).mpr hlt
    have hc : a % H < H := Nat.mod_lt a hH
    have haeq : (a / H) * H + a % H = a := by
      rw [Nat.mul_comm]; exact Nat.div_add_mod a H
    rw [← haeq, addrInBlock_rowBlock hrows hr hc, decide_eq_true_eq] at ha₁ ha₂
    exact hne (by rw [ha₁, ha₂])
  · intro pg₁ _ pg₂ _ _ ag₁ _ ag₂ _ i hi o ho hov
    simp only [gridSpec, List.mem_append, List.mem_ofFn, List.mem_singleton] at hi ho
    obtain ⟨q, rfl⟩ := ho
    rcases hi with ⟨j, rfl⟩ | rfl
    · exact hdisj q j hov.1.symm
    · exact hw q hov.1.symm
  · intro _ _ _ _ _ _ _ _ _ _ _ _
    rfl

theorem storeOnly (M rows : ℕ) (inBufs : Fin k.nIn → BufId) (wBuf : BufId)
    (outBufs : Fin k.nOut → BufId) (hout : Function.Injective outBufs)
    (hdisj : ∀ q i, outBufs q ≠ inBufs i) (hw : ∀ q, outBufs q ≠ wBuf) :
    StoreOnly (k.gridSpec M rows inBufs wBuf outBufs) k.body where
  outs_nodup := by
    simp only [gridSpec, List.map_ofFn]
    exact List.nodup_ofFn.mpr hout
  ins_disjoint := by
    intro p hp o ho
    simp only [gridSpec, List.mem_append, List.mem_ofFn, List.mem_singleton] at hp ho
    obtain ⟨q, rfl⟩ := ho
    rcases hp with ⟨j, rfl⟩ | rfl
    · exact (hdisj q j).symm
    · exact (hw q).symm
  ignores_prev := fun _ _ _ _ => rfl

private theorem inSpecs_getElem?_row (M rows : ℕ) (inBufs : Fin k.nIn → BufId) (wBuf : BufId)
    (outBufs : Fin k.nOut → BufId) (i : Fin k.nIn) :
    (k.gridSpec M rows inBufs wBuf outBufs).inSpecs[i.val]? =
      some (⟨inBufs i, [M, H]⟩, rowBlockSpec rows H) := by
  simp [gridSpec, i.isLt]

private theorem inSpecs_getElem?_weight (M rows : ℕ) (inBufs : Fin k.nIn → BufId) (wBuf : BufId)
    (outBufs : Fin k.nOut → BufId) :
    (k.gridSpec M rows inBufs wBuf outBufs).inSpecs[k.nIn]? =
      some (⟨wBuf, [1, H]⟩, weightBlockSpec H) := by
  simp [gridSpec]

/-- Correctness of every row-wise kernel: for any positive `M`, any positive row block size and any
junk, the run is defined, output row `r` is `f` of input row `r`, and memory outside the outputs is
unchanged. -/
theorem correct (M rows : ℕ) (hM : 0 < M) (hrows : 0 < rows) (hH : 0 < H) (inBufs : Fin k.nIn → BufId)
    (wBuf : BufId) (outBufs : Fin k.nOut → BufId) (hout : Function.Injective outBufs)
    (hdisj : ∀ q i, outBufs q ≠ inBufs i) (hw : ∀ q, outBufs q ≠ wBuf)
    (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', pallasCall (k.gridSpec M rows inBufs wBuf outBufs) k.body junk m0 = some m' ∧
      (∀ (q : Fin k.nOut) (r : Fin M) (c : Fin H),
        m'.read (outBufs q) (r.val * H + c.val) =
          k.f (fun i c' => m0.read (inBufs i) (r.val * H + c'.val)) (fun c' => m0.read wBuf c'.val)
            q c) ∧
      Mem.agreeOutside {a | ∃ q, a.1 = outBufs q} m0 m' := by
  set gs := k.gridSpec M rows inBufs wBuf outBufs with hgs
  refine ⟨pallasExec gs k.body junk m0,
    pallasCall_of_wellDefined _ (k.wellDefined M rows hM hrows hH inBufs wBuf outBufs hout hdisj hw m0),
    ?_, ?_⟩
  · intro q r c
    set o : ArrayDesc × BlockSpec := (⟨outBufs q, [M, H]⟩, rowBlockSpec rows H)
    have hq : gs.outSpecs[q.val]? = some o := by simp [hgs, gridSpec, o]
    have hread := pallasExec_read_out gs k.body junk m0
      (k.storeOnly M rows inBufs wBuf outBufs hout hdisj hw) q.val o hq (r.val * H + c.val)
    simp only [o] at hread
    rw [hread]
    have hG : r.val / rows < (M + rows - 1) / rows := by
      apply (Nat.le_div_iff_mul_le hrows).mpr
      have := Nat.div_mul_le_self r.val rows
      rw [Nat.succ_mul]
      have h2 := Nat.lt_div_mul_add (a := r.val) hrows
      omega
    obtain ⟨l₁, l₂, hsplit, hl₂⟩ := range_split _ (r.val / rows) hG
    simp only [seqOut, hgs, gridSpec, gridPoints_one, List.foldl_map, hsplit]
    have hidx : ∀ (i : ℕ) (sc : ℕ → ℤ), (rowBlockSpec rows H).blockIdx [i] sc = [i, 0] := by
      intro i sc
      simp [BlockSpec.blockIdx, rowBlockSpec]
    simp only [hidx]
    refine (foldl_storeBlock_read_last ⟨outBufs q, [M, H]⟩ [rows, H] (fun i => [i, 0])
      (fun i => outVal gs k.body junk m0 q.val [i]) (r.val * H + c.val) l₁ l₂ (r.val / rows) _
      (by rw [addrInBlock_rowBlock hrows r.isLt c.isLt]; simp)
      (by
        intro y hy
        rw [addrInBlock_rowBlock hrows r.isLt c.isLt]
        have := (hl₂ y hy).1
        simp only [decide_eq_false_iff_not]
        omega)).trans ?_
    dsimp only
    rw [unflatCoord_two M H r.val c.val c.isLt]
    simp only [localCoord, zero_mul, Nat.sub_zero]
    have hrl : r.val - r.val / rows * rows < rows := by
      have := Nat.lt_div_mul_add (a := r.val) hrows
      omega
    have hglob : r.val / rows * rows + (r.val - r.val / rows * rows) = r.val := by
      have := Nat.div_mul_le_self r.val rows
      omega
    have hrowin : ∀ (i : Fin k.nIn) (c' : Fin H),
        (loadIns gs junk m0 [r.val / rows]).getD i.val (fun _ => 0)
          [r.val - r.val / rows * rows, c'.val] = m0.read (inBufs i) (r.val * H + c'.val) := by
      intro i c'
      rw [loadIns_getD gs junk m0 _ i.val _ (k.inSpecs_getElem?_row M rows inBufs wBuf outBufs i),
        hidx, loadBlock_eq_of_inBounds]
      · simp [rowBlockSpec, globalCoord, flatCoord, hglob]
      · simp [rowBlockSpec, inBounds, hrl, c'.isLt]
      · simp [rowBlockSpec, globalCoord, inBounds, hglob, r.isLt, c'.isLt]
    have hwin : ∀ c' : Fin H, (loadIns gs junk m0 [r.val / rows]).getD k.nIn (fun _ => 0)
        [0, c'.val] = m0.read wBuf c'.val := by
      intro c'
      rw [loadIns_getD gs junk m0 _ k.nIn _ (k.inSpecs_getElem?_weight M rows inBufs wBuf outBufs),
        loadBlock_eq_of_inBounds]
      · simp [weightBlockSpec, BlockSpec.blockIdx, globalCoord, flatCoord]
      · simp [weightBlockSpec, inBounds, c'.isLt]
      · simp [weightBlockSpec, BlockSpec.blockIdx, globalCoord, inBounds, c'.isLt]
    simp only [List.getD_eq_getElem?_getD] at hrowin hwin
    simp [outVal, RowKernel.body, List.getD_eq_getElem?_getD, c.isLt,
      hrowin, hwin]
  · refine Mem.agreeOutside_mono ?_ (pallasExec_agreeOutside_outBufs gs k.body junk m0)
    rintro ⟨b, a⟩ ⟨x, hx, hbuf⟩
    simp only [hgs, gridSpec, List.mem_ofFn] at hx
    obtain ⟨q, rfl⟩ := hx
    exact ⟨q, hbuf.symm⟩

/-- Batch invariance of every row-wise kernel: two calls, with any row counts, row block sizes,
junk and other rows, give the same output row whenever the input rows and the weight row agree. -/
theorem batch_invariant {M M' rows rows' : ℕ} (hrows : 0 < rows) (hrows' : 0 < rows') (hH : 0 < H)
    {inBufs inBufs' : Fin k.nIn → BufId} {wBuf wBuf' : BufId} {outBufs outBufs' : Fin k.nOut → BufId}
    (hout : Function.Injective outBufs) (hdisj : ∀ q i, outBufs q ≠ inBufs i)
    (hw : ∀ q, outBufs q ≠ wBuf) (hout' : Function.Injective outBufs')
    (hdisj' : ∀ q i, outBufs' q ≠ inBufs' i) (hw' : ∀ q, outBufs' q ≠ wBuf')
    {junk junk' : ℕ → ℝ} {m0 m0' m₁ m₂ : Mem}
    (h₁ : pallasCall (k.gridSpec M rows inBufs wBuf outBufs) k.body junk m0 = some m₁)
    (h₂ : pallasCall (k.gridSpec M' rows' inBufs' wBuf' outBufs') k.body junk' m0' =
      some m₂)
    (r : Fin M) (r' : Fin M')
    (hin : ∀ i (c : Fin H), m0.read (inBufs i) (r.val * H + c.val) =
      m0'.read (inBufs' i) (r'.val * H + c.val))
    (hwt : ∀ c : Fin H, m0.read wBuf c.val = m0'.read wBuf' c.val) (q : Fin k.nOut) (c : Fin H) :
    m₁.read (outBufs q) (r.val * H + c.val) = m₂.read (outBufs' q) (r'.val * H + c.val) := by
  have hM : 0 < M := Nat.lt_of_le_of_lt (Nat.zero_le _) r.isLt
  have hM' : 0 < M' := Nat.lt_of_le_of_lt (Nat.zero_le _) r'.isLt
  obtain ⟨n₁, e₁, v₁, -⟩ := k.correct M rows hM hrows hH inBufs wBuf outBufs hout hdisj hw junk m0
  obtain ⟨n₂, e₂, v₂, -⟩ :=
    k.correct M' rows' hM' hrows' hH inBufs' wBuf' outBufs' hout' hdisj' hw' junk' m0'
  rw [e₁, Option.some.injEq] at h₁
  rw [e₂, Option.some.injEq] at h₂
  subst h₁ h₂
  rw [v₁, v₂]
  congr 1
  · funext i c'
    exact hin i c'
  · funext c'
    exact hwt c'

end RowKernel

end TpuKernelLean.Pallas
