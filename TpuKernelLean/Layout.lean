import TpuKernelLean.Tensor
import TpuKernelLean.Memory

namespace TpuKernelLean

/-- Memory layout mapping each index of a tensor of shape `s` to an element offset in one buffer. -/
structure Layout (s : List ℕ) where
  /-- Backing HBM buffer identifier. -/
  buf : BufId
  /-- Element offset within `buf` for each tensor index. -/
  off : Idx s → ℕ

namespace Layout

/-- Contiguous row-major layout in buffer `buf` starting at element offset `base`. -/
def rowMajor (buf : BufId) (base : ℕ) (s : List ℕ) : Layout s :=
  { buf := buf, off := fun idx => base + Idx.flat idx }

/-- Set of all logical addresses occupied by a layout. -/
def footprint {s : List ℕ} (l : Layout s) : Set Addr :=
  { a | ∃ idx : Idx s, a = (l.buf, l.off idx) }

/-- Read a tensor of shape `s` from memory `m` according to layout `l`. -/
def read {s : List ℕ} (l : Layout s) (m : Mem) : Tensor ℝ s :=
  fun idx => m.read l.buf (l.off idx)

/-- A layout is injective if distinct tensor indices map to distinct offsets. -/
def Injective {s : List ℕ} (l : Layout s) : Prop :=
  Function.Injective l.off

/-- Two layouts are disjoint if their address footprints do not intersect. -/
def Disjoint {s₁ s₂ : List ℕ} (l₁ : Layout s₁) (l₂ : Layout s₂) : Prop :=
  _root_.Disjoint l₁.footprint l₂.footprint

@[simp] theorem read_apply {s : List ℕ} (l : Layout s) (m : Mem) (idx : Idx s) :
    l.read m idx = m.read l.buf (l.off idx) := rfl

/-- Reading through a layout is unchanged by writes outside a region disjoint from its footprint. -/
theorem read_eq_of_agreeOutside {s : List ℕ} (l : Layout s) {S : Set Addr} {m m' : Mem}
    (hdisj : _root_.Disjoint l.footprint S) (hagree : Mem.agreeOutside S m m') :
    l.read m = l.read m' := by
  funext idx
  apply hagree (l.buf, l.off idx)
  intro hmem
  exact Set.disjoint_left.mp hdisj ⟨idx, rfl⟩ hmem

/-- Reading through a layout depends only on memory inside any region containing its footprint. -/
theorem read_eq_of_agreeOn {s : List ℕ} (l : Layout s) {S : Set Addr} {m m' : Mem}
    (hsub : l.footprint ⊆ S) (hagree : Mem.agreeOn S m m') :
    l.read m = l.read m' := by
  funext idx
  exact hagree (l.buf, l.off idx) (hsub ⟨idx, rfl⟩)

/-- Contiguous row-major layouts are injective. -/
theorem rowMajor_injective (buf : BufId) (base : ℕ) (s : List ℕ) :
    (rowMajor buf base s).Injective := by
  intro i₁ i₂ h
  exact Idx.flat_injective (Nat.add_left_cancel h)

/-- Layouts backed by distinct buffers are disjoint. -/
theorem disjoint_of_buf_ne {s₁ s₂ : List ℕ} {l₁ : Layout s₁} {l₂ : Layout s₂}
    (hne : l₁.buf ≠ l₂.buf) : l₁.Disjoint l₂ := by
  rw [Layout.Disjoint, Set.disjoint_left]
  rintro a ⟨i₁, rfl⟩ ⟨i₂, h⟩
  exact hne (Prod.mk.inj h).1

/-- Every address in a layout's footprint belongs to its backing buffer. -/
theorem footprint_subset_buf {s : List ℕ} (l : Layout s) :
    l.footprint ⊆ Mem.bufFootprint l.buf := by
  rintro a ⟨idx, rfl⟩
  rfl

/-- A layout's footprint is disjoint from any other buffer's footprint. -/
theorem disjoint_bufFootprint {s : List ℕ} (l : Layout s) {b : BufId} (hne : l.buf ≠ b) :
    _root_.Disjoint l.footprint (Mem.bufFootprint b) := by
  rw [Set.disjoint_left]
  rintro a ⟨idx, rfl⟩ hmem
  exact hne hmem

end Layout

end TpuKernelLean
