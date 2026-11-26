import Mathlib

namespace TpuKernelLean

/-- Identifier of a logical HBM buffer. -/
abbrev BufId := ℕ

/-- Logical element-addressed HBM memory mapping `(buffer, offset)` to a real value. -/
abbrev Mem := BufId → ℕ → ℝ

/-- A single logical memory address `(buffer, offset)`. -/
abbrev Addr := BufId × ℕ

namespace Mem

/-- Read the real value stored at `(buf, off)` in memory `m`. -/
def read (m : Mem) (buf : BufId) (off : ℕ) : ℝ := m buf off

/-- Functional update of memory `m` at address `(buf, off)` with value `v`. -/
def write (m : Mem) (buf : BufId) (off : ℕ) (v : ℝ) : Mem :=
  fun b o => if b = buf ∧ o = off then v else m b o

/-- Frame condition: memories `m` and `m'` agree on every address outside `S`. -/
def agreeOutside (S : Set Addr) (m m' : Mem) : Prop :=
  ∀ a : Addr, a ∉ S → m.read a.1 a.2 = m'.read a.1 a.2

/-- Memories `m` and `m'` agree on every address inside `S`. -/
def agreeOn (S : Set Addr) (m m' : Mem) : Prop :=
  ∀ a : Addr, a ∈ S → m.read a.1 a.2 = m'.read a.1 a.2

/-- The set of all addresses belonging to buffer `b`. -/
def bufFootprint (b : BufId) : Set Addr :=
  { a | a.1 = b }

@[simp] theorem read_write_same (m : Mem) (b : BufId) (o : ℕ) (v : ℝ) :
    (m.write b o v).read b o = v := by
  simp [read, write]

theorem read_write_ne (m : Mem) {b b' : BufId} {o o' : ℕ} (v : ℝ)
    (hne : (b', o') ≠ (b, o)) :
    (m.write b o v).read b' o' = m.read b' o' := by
  simp only [read, write]
  split_ifs with h
  · exact absurd (Prod.ext h.1 h.2) hne
  · rfl

theorem read_write_buf_ne (m : Mem) {b b' : BufId} (o o' : ℕ) (v : ℝ) (hne : b' ≠ b) :
    (m.write b o v).read b' o' = m.read b' o' :=
  read_write_ne m v fun h => hne (Prod.mk.inj h).1

theorem agreeOutside_refl (S : Set Addr) (m : Mem) :
    agreeOutside S m m :=
  fun _ _ => rfl

theorem agreeOutside_symm {S : Set Addr} {m m' : Mem} (h : agreeOutside S m m') :
    agreeOutside S m' m :=
  fun a ha => (h a ha).symm

theorem agreeOutside_trans {S : Set Addr} {m₁ m₂ m₃ : Mem}
    (h₁₂ : agreeOutside S m₁ m₂) (h₂₃ : agreeOutside S m₂ m₃) :
    agreeOutside S m₁ m₃ :=
  fun a ha => (h₁₂ a ha).trans (h₂₃ a ha)

theorem agreeOutside_mono {S T : Set Addr} {m m' : Mem}
    (hsub : S ⊆ T) (h : agreeOutside S m m') :
    agreeOutside T m m' :=
  fun a ha => h a fun hs => ha (hsub hs)

theorem agreeOutside_write {S : Set Addr} (m : Mem) {b : BufId} {o : ℕ} (v : ℝ)
    (hmem : (b, o) ∈ S) :
    agreeOutside S m (m.write b o v) := by
  intro ⟨b', o'⟩ ha
  have hne : (b', o') ≠ (b, o) := fun heq => ha (by rw [heq]; exact hmem)
  exact (read_write_ne m v hne).symm

end Mem

end TpuKernelLean
