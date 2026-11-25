import Mathlib

namespace TpuKernelLean

/-- Multi-dimensional index space for a tensor of static shape `s`. -/
def Idx : List ℕ → Type
  | [] => Unit
  | n :: s => Fin n × Idx s

/-- Decidable equality on tensor indices of shape `s`. -/
def Idx.decEq : (s : List ℕ) → DecidableEq (Idx s)
  | [] => inferInstanceAs (DecidableEq Unit)
  | n :: s => by
    letI : DecidableEq (Idx s) := Idx.decEq s
    exact inferInstanceAs (DecidableEq (Fin n × Idx s))

instance {s : List ℕ} : DecidableEq (Idx s) := Idx.decEq s

/-- Finiteness of tensor index spaces of shape `s`. -/
@[instance_reducible] def Idx.fintype : (s : List ℕ) → Fintype (Idx s)
  | [] => inferInstanceAs (Fintype Unit)
  | n :: s => by
    letI : Fintype (Idx s) := Idx.fintype s
    exact inferInstanceAs (Fintype (Fin n × Idx s))

instance {s : List ℕ} : Fintype (Idx s) := Idx.fintype s

/-- Convert a typed tensor index into a list of natural coordinates. -/
def Idx.toList : {s : List ℕ} → Idx s → List ℕ
  | [], () => []
  | _ :: _, (i, rest) => i.val :: Idx.toList rest

/-- Row-major linear flattening of a tensor index within shape `s`. -/
def Idx.flat : {s : List ℕ} → Idx s → ℕ
  | [], () => 0
  | _ :: s, (i, rest) => i.val * s.prod + Idx.flat rest

@[simp] theorem Idx.flat_nil : Idx.flat (s := []) () = 0 := rfl

@[simp] theorem Idx.flat_one {m : ℕ} (i : Fin m) :
    Idx.flat (s := [m]) (i, ()) = i.val := by
  simp [Idx.flat]

@[simp] theorem Idx.flat_two {m n : ℕ} (i : Fin m) (j : Fin n) :
    Idx.flat (s := [m, n]) (i, j, ()) = i.val * n + j.val := by
  simp [Idx.flat]

@[simp] theorem Idx.flat_three {l m n : ℕ} (i : Fin l) (j : Fin m) (k : Fin n) :
    Idx.flat (s := [l, m, n]) (i, j, k, ()) = i.val * (m * n) + (j.val * n + k.val) := by
  simp [Idx.flat]

/-- Row-major flattening always lands strictly below `s.prod`. -/
theorem Idx.flat_lt : {s : List ℕ} → (idx : Idx s) → Idx.flat idx < s.prod
  | [], () => by simp [Idx.flat]
  | n :: s, (i, rest) => by
    simp only [Idx.flat, List.prod_cons]
    have hrest : Idx.flat rest < s.prod := Idx.flat_lt rest
    have h1 : i.val * s.prod + Idx.flat rest < (i.val + 1) * s.prod := by
      rw [Nat.add_mul, Nat.one_mul]
      exact Nat.add_lt_add_left hrest (i.val * s.prod)
    exact Nat.lt_of_lt_of_le h1 (Nat.mul_le_mul_right s.prod i.isLt)

/-- Row-major flattening is injective on `Idx s`. -/
theorem Idx.flat_injective : {s : List ℕ} → Function.Injective (Idx.flat (s := s))
  | [], (), (), _ => rfl
  | _ :: s, (i₁, r₁), (i₂, r₂), h => by
    simp only [Idx.flat] at h
    have hr₁ : Idx.flat r₁ < s.prod := Idx.flat_lt r₁
    have hr₂ : Idx.flat r₂ < s.prod := Idx.flat_lt r₂
    have hpos : 0 < s.prod := Nat.lt_of_le_of_lt (Nat.zero_le _) hr₁
    have hdiv : i₁.val = i₂.val := by
      have h1 : (i₁.val * s.prod + Idx.flat r₁) / s.prod = i₁.val := by
        rw [Nat.mul_comm, Nat.mul_add_div hpos, Nat.div_eq_of_lt hr₁, Nat.add_zero]
      have h2 : (i₂.val * s.prod + Idx.flat r₂) / s.prod = i₂.val := by
        rw [Nat.mul_comm, Nat.mul_add_div hpos, Nat.div_eq_of_lt hr₂, Nat.add_zero]
      rw [← h1, h, h2]
    have hi : i₁ = i₂ := Fin.ext hdiv
    subst hi
    have hrest : Idx.flat r₁ = Idx.flat r₂ := Nat.add_left_cancel h
    exact Prod.ext rfl (Idx.flat_injective hrest)

/-- A tensor of elements of type `α` indexed by static shape `s`. -/
abbrev Tensor (α : Type*) (s : List ℕ) : Type _ := Idx s → α

namespace Tensor

/-- Construct a rank-1 tensor from a function on `Fin m`. -/
def of1 {α : Type*} {m : ℕ} (f : Fin m → α) : Tensor α [m] :=
  fun (i, ()) => f i

/-- Read an element of a rank-1 tensor. -/
def get1 {α : Type*} {m : ℕ} (t : Tensor α [m]) (i : Fin m) : α :=
  t (i, ())

/-- Construct a rank-2 tensor from a function on `Fin m → Fin n`. -/
def of2 {α : Type*} {m n : ℕ} (f : Fin m → Fin n → α) : Tensor α [m, n] :=
  fun (i, j, ()) => f i j

/-- Read an element of a rank-2 tensor. -/
def get2 {α : Type*} {m n : ℕ} (t : Tensor α [m, n]) (i : Fin m) (j : Fin n) : α :=
  t (i, j, ())

/-- Construct a rank-3 tensor from a function on `Fin l → Fin m → Fin n`. -/
def of3 {α : Type*} {l m n : ℕ} (f : Fin l → Fin m → Fin n → α) : Tensor α [l, m, n] :=
  fun (i, j, k, ()) => f i j k

/-- Read an element of a rank-3 tensor. -/
def get3 {α : Type*} {l m n : ℕ} (t : Tensor α [l, m, n]) (i : Fin l) (j : Fin m) (k : Fin n) :
    α :=
  t (i, j, k, ())

@[simp] theorem get1_of1 {α : Type*} {m : ℕ} (f : Fin m → α) (i : Fin m) :
    (of1 f).get1 i = f i := rfl

@[simp] theorem of1_get1 {α : Type*} {m : ℕ} (t : Tensor α [m]) :
    of1 t.get1 = t := by
  funext ⟨i, ⟨⟩⟩
  rfl

@[simp] theorem get2_of2 {α : Type*} {m n : ℕ} (f : Fin m → Fin n → α) (i : Fin m) (j : Fin n) :
    (of2 f).get2 i j = f i j := rfl

@[simp] theorem of2_get2 {α : Type*} {m n : ℕ} (t : Tensor α [m, n]) :
    of2 t.get2 = t := by
  funext ⟨i, j, ⟨⟩⟩
  rfl

@[simp] theorem get3_of3 {α : Type*} {l m n : ℕ} (f : Fin l → Fin m → Fin n → α)
    (i : Fin l) (j : Fin m) (k : Fin n) :
    (of3 f).get3 i j k = f i j k := rfl

@[simp] theorem of3_get3 {α : Type*} {l m n : ℕ} (t : Tensor α [l, m, n]) :
    of3 t.get3 = t := by
  funext ⟨i, j, k, ⟨⟩⟩
  rfl

theorem ext1 {α : Type*} {m : ℕ} {a b : Tensor α [m]} (h : ∀ i, a.get1 i = b.get1 i) :
    a = b := by
  funext ⟨i, ⟨⟩⟩
  exact h i

theorem ext2 {α : Type*} {m n : ℕ} {a b : Tensor α [m, n]}
    (h : ∀ i j, a.get2 i j = b.get2 i j) : a = b := by
  funext ⟨i, j, ⟨⟩⟩
  exact h i j

theorem ext3 {α : Type*} {l m n : ℕ} {a b : Tensor α [l, m, n]}
    (h : ∀ i j k, a.get3 i j k = b.get3 i j k) : a = b := by
  funext ⟨i, j, k, ⟨⟩⟩
  exact h i j k

/-- Elementwise unary map over a tensor. -/
def map {α β : Type*} {s : List ℕ} (f : α → β) (t : Tensor α s) : Tensor β s :=
  fun i => f (t i)

/-- Elementwise binary map over two tensors of identical shape. -/
def zipWith {α β γ : Type*} {s : List ℕ} (f : α → β → γ)
    (a : Tensor α s) (b : Tensor β s) : Tensor γ s :=
  fun i => f (a i) (b i)

/-- Transpose a rank-2 tensor. -/
def transpose {α : Type*} {m n : ℕ} (t : Tensor α [m, n]) : Tensor α [n, m] :=
  of2 fun i j => t.get2 j i

@[simp] theorem map_apply {α β : Type*} {s : List ℕ} (f : α → β) (t : Tensor α s) (i : Idx s) :
    map f t i = f (t i) := rfl

@[simp] theorem zipWith_apply {α β γ : Type*} {s : List ℕ} (f : α → β → γ)
    (a : Tensor α s) (b : Tensor β s) (i : Idx s) :
    zipWith f a b i = f (a i) (b i) := rfl

@[simp] theorem get2_transpose {α : Type*} {m n : ℕ} (t : Tensor α [m, n])
    (i : Fin n) (j : Fin m) :
    t.transpose.get2 i j = t.get2 j i := rfl

/-- Maximum entry of a non-empty real vector of length `n + 1`. -/
noncomputable def vecMax : {n : ℕ} → (Fin (n + 1) → ℝ) → ℝ
  | 0, v => v 0
  | _ + 1, v => max (v 0) (vecMax fun i => v i.succ)

/-- Every entry of `v` is bounded above by `vecMax v`. -/
theorem le_vecMax : {n : ℕ} → (v : Fin (n + 1) → ℝ) → (i : Fin (n + 1)) → v i ≤ vecMax v
  | 0, v, i => by
    have hi : i = 0 := Fin.ext (by omega)
    subst hi
    exact le_rfl
  | n + 1, v, i => by
    refine Fin.cases ?_ ?_ i
    · exact le_max_left _ _
    · intro j
      exact le_trans (le_vecMax (fun k => v k.succ) j) (le_max_right _ _)

/-- `vecMax v` is bounded above by any upper bound on all entries of `v`. -/
theorem vecMax_le : {n : ℕ} → {v : Fin (n + 1) → ℝ} → {b : ℝ} →
    (∀ i, v i ≤ b) → vecMax v ≤ b
  | 0, _, _, h => h 0
  | _ + 1, _, _, h => max_le (h 0) (vecMax_le fun j => h j.succ)

/-- `vecMax` depends only on the values of the vector. -/
theorem vecMax_congr {n : ℕ} {u v : Fin (n + 1) → ℝ} (h : ∀ i, u i = v i) :
    vecMax u = vecMax v := by
  have huv : u = v := funext h
  rw [huv]

noncomputable section

/-- Multiply every entry of a real tensor by a scalar `c`. -/
def scale {s : List ℕ} (c : ℝ) (t : Tensor ℝ s) : Tensor ℝ s :=
  map (fun x => c * x) t

/-- Matrix multiplication of `[m, k]` and `[k, n]` real tensors. -/
def matmul {m k n : ℕ} (a : Tensor ℝ [m, k]) (b : Tensor ℝ [k, n]) : Tensor ℝ [m, n] :=
  of2 fun i j => ∑ p : Fin k, a.get2 i p * b.get2 p j

/-- Sum of each row of a rank-2 real tensor. -/
def rowSum {m n : ℕ} (t : Tensor ℝ [m, n]) : Tensor ℝ [m] :=
  of1 fun i => ∑ j : Fin n, t.get2 i j

/-- Maximum entry of each non-empty row of a rank-2 real tensor. -/
def rowMax {m n : ℕ} (t : Tensor ℝ [m, n + 1]) : Tensor ℝ [m] :=
  of1 fun i => vecMax fun j => t.get2 i j

/-- Elementwise real exponential of a tensor. -/
def exp {s : List ℕ} (t : Tensor ℝ s) : Tensor ℝ s :=
  map Real.exp t

/-- Row-wise numerically-stable softmax `exp(x - max) / sum(exp(x - max))`. -/
def softmax {m n : ℕ} (t : Tensor ℝ [m, n + 1]) : Tensor ℝ [m, n + 1] :=
  of2 fun i j =>
    let rMax := (rowMax t).get1 i
    let denom := ∑ k : Fin (n + 1), Real.exp (t.get2 i k - rMax)
    Real.exp (t.get2 i j - rMax) / denom

/-- Row-wise RMSNorm `x / sqrt(mean(x^2) + eps) * w` over the last dimension. -/
def rmsNorm {m n : ℕ} (eps : ℝ) (x : Tensor ℝ [m, n]) (w : Tensor ℝ [n]) : Tensor ℝ [m, n] :=
  of2 fun i j =>
    let meanSq := (∑ k : Fin n, x.get2 i k * x.get2 i k) / (n : ℝ)
    x.get2 i j / Real.sqrt (meanSq + eps) * w.get1 j

/-- Row-wise NeoX RoPE splitting the last dimension into halves `[x1*cos - x2*sin, x2*cos + x1*sin]`. -/
def ropeNeox {m h : ℕ} (x : Tensor ℝ [m, 2 * h])
    (cos sin : Tensor ℝ [m, h]) : Tensor ℝ [m, 2 * h] :=
  of2 fun i j =>
    if hj : j.val < h then
      let k : Fin h := ⟨j.val, hj⟩
      let k2 : Fin (2 * h) := ⟨j.val + h, by omega⟩
      x.get2 i j * cos.get2 i k - x.get2 i k2 * sin.get2 i k
    else
      let k : Fin h := ⟨j.val - h, by omega⟩
      let k1 : Fin (2 * h) := ⟨j.val - h, by omega⟩
      x.get2 i j * cos.get2 i k + x.get2 i k1 * sin.get2 i k

@[simp] theorem get2_scale {m n : ℕ} (c : ℝ) (t : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    (scale c t).get2 i j = c * t.get2 i j := rfl

@[simp] theorem get2_matmul {m k n : ℕ} (a : Tensor ℝ [m, k]) (b : Tensor ℝ [k, n])
    (i : Fin m) (j : Fin n) :
    (matmul a b).get2 i j = ∑ p : Fin k, a.get2 i p * b.get2 p j := rfl

@[simp] theorem get1_rowSum {m n : ℕ} (t : Tensor ℝ [m, n]) (i : Fin m) :
    (rowSum t).get1 i = ∑ j : Fin n, t.get2 i j := rfl

@[simp] theorem get1_rowMax {m n : ℕ} (t : Tensor ℝ [m, n + 1]) (i : Fin m) :
    (rowMax t).get1 i = vecMax fun j => t.get2 i j := rfl

end

end Tensor

end TpuKernelLean
