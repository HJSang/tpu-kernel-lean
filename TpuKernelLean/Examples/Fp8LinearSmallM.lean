import TpuKernelLean.Dtype
import TpuKernelLean.Spec
import TpuKernelLean.Pallas.RowKernel

/-!
Specification and end-to-end verification of `sglang_tpu/kernels/fp8_linear_small_m.py`:
per-row dynamic FP8 quantization of `x [tokens, k]`, FP8-in / fp32-accumulated dot product with
column tiles of `w [k, n]`, rescaling by `x_scale * ws`, and casting to the output dtype.

The kernel is tiled across columns of `w`, `ws`, and `out`:
- grid: `(n / tile_n,)` (`fp8_linear_small_m.py:64`)
- `x`: block `(tokens, k)` at `(0, 0)` (`fp8_linear_small_m.py:66`)
- `w`: block `(k, tile_n)` at `(0, j)` (`fp8_linear_small_m.py:67`)
- `ws`: block `(1, tile_n)` at `(0, j)` (`fp8_linear_small_m.py:68`, already broadcast to `[1, n]`
  before `pallas_call` at lines 55–57)
- `out`: block `(tokens, tile_n)` at `(0, j)` (`fp8_linear_small_m.py:70`)

`fp8_max` is tied to `fp8_dtype` as `fp8Dtype.fp8Max` (`fp8_linear_small_m.py:61-62`), with
`fp8Dtype.isFp8 = true` so `0 < fp8Dtype.fp8Max`. Floating-point, quantization, and dot operations
are kept opaque (`FP8LinearOps`, reusing `Dtype.CastOps`), with `rowMax` and `dot` parameterised
by the compiled block shape `(tokens, k, tile_n)` because `_kernel` issues a single untiled
`dot_general` on `(tokens, k) @ (k, tile_n)` (`fp8_linear_small_m.py:38-43`) whose Mosaic
K-blocking and fp32 accumulation order may vary across different `tokens` or `tile_n`:
- `fp8Linear_correct` and `fp8LinearSpec_correct` prove the kernel well defined and correct for
  every `(tokens, k, n, tile_n)` meeting `Supports` and every FP8 `fp8Dtype`, with frame condition
  `Mem.bufFootprint oBuf`;
- `fp8Linear_fixedShape_elem_independent` and `fp8Linear_fixedShape_row_independent` prove what
  the kernel structure enforces at a fixed compiled shape `(tokens, k, n, tile_n)`: an output row
  depends only on its own `x` row, `w`, and `ws` (not on other rows' values or junk);
- `fp8LinearScaled_abs_le_exact` proves in exact arithmetic that every scaled input satisfies
  `|x_p / x_scale| ≤ fp8Dtype.fp8Max`, justifying why `fp8_linear_small_m.py:36-37` omits
  `jnp.clip` before the unsaturated `.astype(fp8_dtype)` cast;
- `fp8LinearElem_exact` and `fp8LinearMath_exact` show that under exact arithmetic
  (`FP8LinearOps.exact`) with an FP8 dtype (`fp8Dtype.isFp8 = true`) the positive `1e-12` scale
  floor guarantees exact cancellation of `x_scale`, recovering `(x @ w) * ws`.

`dimension_semantics=("parallel",)` (`fp8_linear_small_m.py:72`) is modelled on `gridSpec` and
proved race-free in `wellDefined`, so every multi-core split or reordering agrees with sequential
execution (`Pallas.pallasExecPar_eq_pallasExec`).
-/

namespace TpuKernelLean.Examples.Fp8LinearSmallM

open TpuKernelLean.Pallas

/-- Maximum batch size handled by `fp8_linear_small_m` (`fp8_linear_small_m.py:18`). -/
def maxTokens : ℕ := 16

/-- Candidate column tile sizes tried by `_tile_n` (`fp8_linear_small_m.py:19`). -/
def tileNCandidates : List ℕ := [1024, 512, 256, 128]

/-- Double-buffered FP8 weight tile budget in bytes (`fp8_linear_small_m.py:21`). -/
def weightTileBudget : ℕ := 24 * 1024 * 1024

/-- Minimum per-row quantization scale `1e-12` (`fp8_linear_small_m.py:36`). -/
noncomputable def scaleFloor : ℝ := 1 / 10 ^ 12

/-- Preconditions of `supports(tokens, k, n)` and `_tile_n(k, n)` (`fp8_linear_small_m.py:18-31`),
together with `0 < k` and `0 < n` (required for positive Pallas `BlockSpec` dims, non-empty
`jnp.max`, and a non-empty grid `n / tile_n`). -/
structure Supports (tokens k n tile_n : ℕ) : Prop where
  /-- At least one token (`0 < tokens`, `fp8_linear_small_m.py:31`). -/
  tokens_pos : 0 < tokens
  /-- At most `MAX_TOKENS = 16` tokens (`fp8_linear_small_m.py:18,31`). -/
  tokens_le : tokens ≤ maxTokens
  /-- Reduction axis is non-empty (required by `BlockSpec` and `jnp.max`). -/
  k_pos : 0 < k
  /-- Reduction axis is a multiple of 128 (`fp8_linear_small_m.py:31`). -/
  k_mod : k % 128 = 0
  /-- Output width is non-empty (so the grid `n / tile_n` is non-empty). -/
  n_pos : 0 < n
  /-- `tile_n` is one of `(1024, 512, 256, 128)` (`fp8_linear_small_m.py:19,26`). -/
  tile_mem : tile_n ∈ tileNCandidates
  /-- Output width is divisible by `tile_n` (`fp8_linear_small_m.py:26`). -/
  n_mod : n % tile_n = 0
  /-- Double-buffered FP8 weight tile fits in VMEM budget (`fp8_linear_small_m.py:21,26`). -/
  budget : 2 * k * tile_n ≤ weightTileBudget

/-- Every candidate tile size in `tileNCandidates` is positive. -/
theorem Supports.tile_pos {tokens k n tile_n : ℕ} (h : Supports tokens k n tile_n) :
    0 < tile_n := by
  have hm := h.tile_mem
  simp only [tileNCandidates, List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm with rfl | rfl | rfl | rfl <;> decide

/-- Under `Supports`, the column-tile grid `n / tile_n` has at least one step. -/
theorem Supports.grid_pos {tokens k n tile_n : ℕ} (h : Supports tokens k n tile_n) :
    0 < n / tile_n := by
  have hdm := Nat.div_add_mod n tile_n
  rw [h.n_mod, Nat.add_zero] at hdm
  cases hd : n / tile_n with
  | zero =>
    rw [hd, Nat.mul_zero] at hdm
    have := h.n_pos
    omega
  | succ _ => exact Nat.succ_pos _

/-- Opaque TPU operations used by `_kernel` (`fp8_linear_small_m.py:34-44`).

Hardware assumptions for fixed-shape bit-for-bit theorems:
- Each scalar op (`abs`, `div`, `maximum`, `mul`, and the `CastOps` casts) is a deterministic
  function of its arguments.
- For a fixed compiled block shape `(tokens, k, tile_n)`, `rowMax tokens k tile_n` (Mosaic's
  `jnp.max(..., axis=-1)` over the `(tokens, k)` `x` block) depends only on its length-`k` row,
  not on the values in other rows of the block.
- For a fixed compiled block shape `(tokens, k, tile_n)`, `dot tokens k tile_n` (Mosaic's untiled
  full-`K` `dot_general` on `(tokens, k) @ (k, tile_n)`) depends only on its length-`k` row of
  `x_q` and length-`k` column of `w`, not on the values in other rows or columns.

Neither `rowMax` nor `dot` is assumed invariant across different `tokens` or `tile_n`, because
changing the compiled block shape can change Mosaic's K-blocking and fp32 accumulation order. -/
structure FP8LinearOps where
  /-- Dtype casts (`toF32`, `toFp8E4M3`, `toFp8E5M2`, `toBf16`). -/
  casts : CastOps
  /-- fp32 elementwise absolute value (`jnp.abs`, `fp8_linear_small_m.py:36`). -/
  abs : ℝ → ℝ
  /-- Row maximum over one length-`k` fp32 row within a compiled `(tokens, k, tile_n)` kernel
  (`jnp.max(..., axis=-1)`, `fp8_linear_small_m.py:36`). -/
  rowMax : (tokens k tile_n : ℕ) → (Fin k → ℝ) → ℝ
  /-- fp32 division (`/`, `fp8_linear_small_m.py:36-37`). -/
  div : ℝ → ℝ → ℝ
  /-- fp32 elementwise maximum (`jnp.maximum`, `fp8_linear_small_m.py:36`). -/
  maximum : ℝ → ℝ → ℝ
  /-- fp32 multiplication (`*`, `fp8_linear_small_m.py:44`). -/
  mul : ℝ → ℝ → ℝ
  /-- One output element of `jax.lax.dot_general(x_q, w, ..., preferred_element_type=float32)` on
  a `(tokens, k) @ (k, tile_n)` block (`fp8_linear_small_m.py:38-43`). -/
  dot : (tokens k tile_n : ℕ) → (Fin k → ℝ) → (Fin k → ℝ) → ℝ

/-- Exact row maximum of a length-`k` real vector (`0` when `k = 0`). -/
noncomputable def exactRowMax : (k : ℕ) → (Fin k → ℝ) → ℝ
  | 0, _ => 0
  | _ + 1, v => Tensor.vecMax v

/-- Exact real-arithmetic instance of `FP8LinearOps`, linking the kernel to `(x @ w) * ws`. -/
noncomputable def FP8LinearOps.exact : FP8LinearOps where
  casts := CastOps.exact
  abs := fun x => |x|
  rowMax := fun _ k _ => exactRowMax k
  div := (· / ·)
  maximum := max
  mul := (· * ·)
  dot := fun _ _ _ a b => ∑ p, a p * b p

/-- Per-row dynamic FP8 quantization scale `maximum(max(|x|) / fp8Dtype.fp8Max, 1e-12)`
(`fp8_linear_small_m.py:36,61-62`). -/
noncomputable def fp8LinearScale (ops : FP8LinearOps) (fp8Dtype : Dtype)
    (tokens k tile_n : ℕ) (xr : Fin k → ℝ) : ℝ :=
  let xF32 : Fin k → ℝ := fun p => ops.casts.toF32 (xr p)
  ops.maximum
    (ops.div (ops.rowMax tokens k tile_n fun p => ops.abs (xF32 p)) fp8Dtype.fp8Max)
    scaleFloor

/-- Pre-cast scaled activation `x_p / x_scale` passed to the unsaturated `.astype(fp8_dtype)` cast
(`fp8_linear_small_m.py:37`). -/
noncomputable def fp8LinearScaled (ops : FP8LinearOps) (fp8Dtype : Dtype)
    (tokens k tile_n : ℕ) (xr : Fin k → ℝ) (p : Fin k) : ℝ :=
  ops.div (ops.casts.toF32 (xr p)) (fp8LinearScale ops fp8Dtype tokens k tile_n xr)

/-- Per-element specification of `_kernel` (`fp8_linear_small_m.py:34-44,61-62`): cast row `xr` to
fp32, compute `x_scale = maximum(max(|x|) / fp8Dtype.fp8Max, 1e-12)`, cast `x / x_scale` to
`fp8Dtype`, dot with column `wc` in fp32 for compiled block shape `(tokens, k, tile_n)`, rescale by
`x_scale * ws`, and cast to `outDtype`. -/
noncomputable def fp8LinearElem (ops : FP8LinearOps) (fp8Dtype outDtype : Dtype)
    (tokens k tile_n : ℕ) (xr wc : Fin k → ℝ) (ws : ℝ) : ℝ :=
  let xScale : ℝ := fp8LinearScale ops fp8Dtype tokens k tile_n xr
  let xQ : Fin k → ℝ := fun p =>
    ops.casts.cast fp8Dtype (fp8LinearScaled ops fp8Dtype tokens k tile_n xr p)
  let acc : ℝ := ops.dot tokens k tile_n xQ wc
  ops.casts.cast outDtype (ops.mul (ops.mul acc xScale) ws)

/-- Tensor-level mathematical specification of `fp8_linear_small_m` (`fp8_linear_small_m.py:34-44`). -/
noncomputable def fp8LinearMath (ops : FP8LinearOps) (fp8Dtype outDtype : Dtype) (tile_n : ℕ)
    {tokens k n : ℕ} (x : Tensor ℝ [tokens, k]) (w : Tensor ℝ [k, n]) (ws : Tensor ℝ [1, n]) :
    Tensor ℝ [tokens, n] :=
  Tensor.of2 fun r c =>
    fp8LinearElem ops fp8Dtype outDtype tokens k tile_n
      (fun p => x.get2 r p) (fun p => w.get2 p c) (ws.get2 0 c)

/-- Read-only inputs to `fp8_linear_small_m`: activations `x [tokens, k]`, FP8 weights `w [k, n]`,
and broadcast weight scale `ws [1, n]` (`fp8_linear_small_m.py:47-57`). -/
structure Fp8LinearIn (tokens k n : ℕ) where
  /-- Input activations `[tokens, k]`. -/
  x : Tensor ℝ [tokens, k]
  /-- FP8 weight matrix `[k, n]`. -/
  w : Tensor ℝ [k, n]
  /-- Broadcast fp32 weight scale `[1, n]`. -/
  ws : Tensor ℝ [1, n]

/-- `pl.BlockSpec((tokens, k), lambda j: (0, 0))` (`fp8_linear_small_m.py:66`). -/
def xSpec (tokens k : ℕ) : BlockSpec :=
  { blockShape := [tokens, k], indexMap := fun _ _ => [0, 0] }

/-- `pl.BlockSpec((k, tile_n), lambda j: (0, j))` (`fp8_linear_small_m.py:67`). -/
def wSpec (k tile_n : ℕ) : BlockSpec :=
  { blockShape := [k, tile_n], indexMap := fun g _ => [0, ((g.getD 0 0 : ℕ) : ℤ)] }

/-- `pl.BlockSpec((1, tile_n), lambda j: (0, j))` (`fp8_linear_small_m.py:68`). -/
def wsSpec (tile_n : ℕ) : BlockSpec :=
  { blockShape := [1, tile_n], indexMap := fun g _ => [0, ((g.getD 0 0 : ℕ) : ℤ)] }

/-- `pl.BlockSpec((tokens, tile_n), lambda j: (0, j))` (`fp8_linear_small_m.py:70`). -/
def oSpec (tokens tile_n : ℕ) : BlockSpec :=
  { blockShape := [tokens, tile_n], indexMap := fun g _ => [0, ((g.getD 0 0 : ℕ) : ℤ)] }

/-- `pallas_call` grid and block specs of `fp8_linear_small_m` (`fp8_linear_small_m.py:58-74`). -/
def gridSpec (tokens k n tile_n : ℕ) (xBuf wBuf wsBuf oBuf : BufId) : GridSpec :=
  { grid := [n / tile_n]
    inSpecs :=
      [ (⟨xBuf, [tokens, k]⟩, xSpec tokens k),
        (⟨wBuf, [k, n]⟩, wSpec k tile_n),
        (⟨wsBuf, [1, n]⟩, wsSpec tile_n) ]
    outSpecs :=
      [ (⟨oBuf, [tokens, n]⟩, oSpec tokens tile_n) ]
    dimSemantics := [.parallel] }

/-- Pallas body of `_kernel` (`fp8_linear_small_m.py:34-44`): dynamically quantize each row of `x`
to `fp8Dtype` using `fp8Dtype.fp8Max`, compute the fp32-accumulated dot product with the current
`w` column tile for compiled shape `(tokens, k, tile_n)`, rescale by `x_scale * ws`, and cast to
`outDtype`. -/
noncomputable def body (ops : FP8LinearOps) (fp8Dtype outDtype : Dtype)
    (tokens k tile_n : ℕ) : Body :=
  fun _ _ ins _ =>
    [fun loc =>
      let r := loc.getD 0 0
      let c := loc.getD 1 0
      fp8LinearElem ops fp8Dtype outDtype tokens k tile_n
        (fun p => (ins.getD 0 fun _ => 0) [r, p.val])
        (fun p => (ins.getD 1 fun _ => 0) [p.val, c])
        ((ins.getD 2 fun _ => 0) [0, c])]

/-- Pallas semantics of `fp8_linear_small_m` (`none` on undefined behaviour). -/
noncomputable def fp8LinearRun (ops : FP8LinearOps) (fp8Dtype outDtype : Dtype)
    (tokens k n tile_n : ℕ) (xBuf wBuf wsBuf oBuf : BufId) :
    (ℕ → ℝ) → Mem → Option Mem :=
  pallasCall (gridSpec tokens k n tile_n xBuf wBuf wsBuf oBuf)
    (body ops fp8Dtype outDtype tokens k tile_n)

/-- Four-part `KernelSpec` for `fp8_linear_small_m`. -/
noncomputable def fp8LinearSpec (ops : FP8LinearOps) (fp8Dtype outDtype : Dtype)
    (tokens k n tile_n : ℕ) (xBuf wBuf wsBuf oBuf : BufId) :
    KernelSpec (Fp8LinearIn tokens k n) Unit (Tensor ℝ [tokens, n]) :=
  { readIn := fun m =>
      { x := (Layout.rowMajor xBuf 0 [tokens, k]).read m
        w := (Layout.rowMajor wBuf 0 [k, n]).read m
        ws := (Layout.rowMajor wsBuf 0 [1, n]).read m }
    readSt := fun _ => ()
    readOut := fun m => (Layout.rowMajor oBuf 0 [tokens, n]).read m
    footprint := Mem.bufFootprint oBuf
    math := fun _ inp => fp8LinearMath ops fp8Dtype outDtype tile_n inp.x inp.w inp.ws
    mutate := fun _ _ => ()
    requires := fun _ _ =>
      Supports tokens k n tile_n ∧ fp8Dtype.isFp8 = true ∧
        oBuf ≠ xBuf ∧ oBuf ≠ wBuf ∧ oBuf ≠ wsBuf }

/-- An in-bounds element `(r, c)` of a `[tokens, n]` array lies in column tile `[0, c / tile_n]`. -/
theorem addrInBlock_colTile {buf : BufId} {tokens n tile_n r c j : ℕ}
    (htile : 0 < tile_n) (hr : r < tokens) (hc : c < n) :
    addrInBlock ⟨buf, [tokens, n]⟩ [tokens, tile_n] [0, j] (r * n + c) =
      decide (j = c / tile_n) := by
  have hlt : r * n + c < tokens * n := by nlinarith
  have key : (j * tile_n ≤ c ∧ c < (j + 1) * tile_n) ↔ j = c / tile_n := by
    constructor
    · rintro ⟨h1, h2⟩
      exact (Nat.div_eq_of_lt_le h1 h2).symm
    · rintro rfl
      refine ⟨Nat.div_mul_le_self c tile_n, ?_⟩
      rw [Nat.succ_mul]
      exact Nat.lt_div_mul_add htile
  simp [addrInBlock, unflatCoord_two tokens n r c hc, coordInBlock, hlt, hr, key]

/-- Under positive dimensions and an output buffer disjoint from inputs, `fp8_linear_small_m` is
well defined (`GridSpec.WellDefined`). -/
theorem wellDefined (ops : FP8LinearOps) (fp8Dtype outDtype : Dtype)
    (tokens k n tile_n : ℕ) (htok : 0 < tokens) (hk : 0 < k) (htile : 0 < tile_n)
    (hgrid : 0 < n / tile_n)
    (xBuf wBuf wsBuf oBuf : BufId) (hx : oBuf ≠ xBuf) (hw : oBuf ≠ wBuf) (hws : oBuf ≠ wsBuf)
    (m0 : Mem) :
    (gridSpec tokens k n tile_n xBuf wBuf wsBuf oBuf).WellDefined
      (body ops fp8Dtype outDtype tokens k tile_n) m0 := by
  have hlen : (gridPoints [n / tile_n]).length = n / tile_n := by
    simp [gridPoints_length]
  constructor
  · intro d hd
    simp only [gridSpec, List.mem_singleton] at hd
    subst hd
    exact hgrid
  · intro p hp
    simp only [gridSpec, List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil,
      or_false] at hp
    rcases hp with rfl | rfl | rfl | rfl <;>
      simp [xSpec, wSpec, wsSpec, oSpec, htok, hk, htile]
  · intro j hj p hp
    simp only [gridSpec, hlen] at hj
    have hj_lt : j * tile_n < n := by
      have h1 : (j + 1) * tile_n ≤ (n / tile_n) * tile_n := Nat.mul_le_mul_right tile_n hj
      have h2 : (n / tile_n) * tile_n ≤ n := Nat.div_mul_le_self n tile_n
      rw [Nat.succ_mul] at h1
      omega
    have hjz : (j : ℤ) * tile_n < n := by exact_mod_cast hj_lt
    simp only [gridSpec, List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil,
      or_false] at hp
    rcases hp with rfl | rfl | rfl | rfl <;>
      simp only [GridSpec.idxAt, gridSpec, gridPoints_one_getD _ j hj,
        xSpec, wSpec, wsSpec, oSpec] <;>
      simp [blockOk, htok, hk, htile, hjz]
  · intro p hp
    simp only [gridSpec, List.mem_singleton] at hp
    subst hp
    apply noRevisit_of_nodup
    simp only [gridSpec, hlen]
    apply List.Nodup.map_on _ List.nodup_range
    intro a ha b hb hab
    simp only [List.mem_range] at ha hb
    simp only [GridSpec.idxAt, oSpec, gridPoints_one_getD _ a ha,
      gridPoints_one_getD _ b hb] at hab
    simpa using hab
  · intro _ _ _ _
    rfl
  · intro s t _ _ i hi o ho hov
    simp only [gridSpec, List.mem_cons, List.not_mem_nil, or_false] at hi ho
    subst ho
    rcases hi with rfl | rfl | rfl
    · exact hx hov.1.symm
    · exact hw hov.1.symm
    · exact hws hov.1.symm
  · exact GridSpec.AliasesValid.of_empty rfl (by simp [gridSpec]) (by
      intro inp hinp outp houtp
      simp only [gridSpec, List.mem_cons, List.not_mem_nil, or_false] at hinp houtp
      subst houtp
      rcases hinp with rfl | rfl | rfl
      · exact hx.symm
      · exact hw.symm
      · exact hws.symm)
      (by simp [gridSpec])
  · rfl
  · rfl
  · intro pg₁ h₁ pg₂ h₂ hne ag₁ _ ag₂ _ o₁ ho₁ o₂ ho₂ hov
    simp only [gridSpec, GridSpec.parGrid, GridSpec.numParallel, List.takeWhile,
      DimSemantics.isParallel, List.length_cons, List.length_nil, List.take_succ_cons,
      List.take_zero] at h₁ h₂
    rw [mem_gridPoints] at h₁ h₂
    rcases pg₁ with _ | ⟨j₁, _ | ⟨_, _⟩⟩ <;> simp [inBounds] at h₁
    rcases pg₂ with _ | ⟨j₂, _ | ⟨_, _⟩⟩ <;> simp [inBounds] at h₂
    simp only [gridSpec, List.mem_singleton] at ho₁ ho₂
    subst ho₁ ho₂
    obtain ⟨_, a, ha₁, ha₂⟩ := hov
    simp only [BlockSpec.blockIdx, oSpec, List.cons_append, List.getD_cons_zero, List.map_cons,
      List.map_nil, Int.toNat_zero, Int.toNat_natCast] at ha₁ ha₂
    have hlt : a < tokens * n := by
      simp only [addrInBlock, Bool.and_eq_true, decide_eq_true_eq, List.prod_cons,
        List.prod_nil, mul_one] at ha₁
      exact ha₁.1
    have hn : 0 < n := by
      by_contra h0
      have : n = 0 := by omega
      subst this
      simp at hlt
    have hr : a / n < tokens := (Nat.div_lt_iff_lt_mul hn).mpr hlt
    have hc : a % n < n := Nat.mod_lt a hn
    have haeq : (a / n) * n + a % n = a := by
      rw [Nat.mul_comm]; exact Nat.div_add_mod a n
    rw [← haeq, addrInBlock_colTile htile hr hc, decide_eq_true_eq] at ha₁ ha₂
    exact hne (by rw [ha₁, ha₂])
  · intro _ _ _ _ _ _ _ _ _ i hi o ho hov
    simp only [gridSpec, List.mem_cons, List.not_mem_nil, or_false] at hi ho
    subst ho
    rcases hi with rfl | rfl | rfl
    · exact hx hov.1.symm
    · exact hw hov.1.symm
    · exact hws hov.1.symm
  · intro _ _ _ _ _ _ _ _ _ _ _ _
    rfl

/-- `fp8_linear_small_m` is `StoreOnly`: one output buffer disjoint from inputs, body ignores
previous output blocks. -/
theorem storeOnly (ops : FP8LinearOps) (fp8Dtype outDtype : Dtype)
    (tokens k n tile_n : ℕ) (xBuf wBuf wsBuf oBuf : BufId)
    (hx : oBuf ≠ xBuf) (hw : oBuf ≠ wBuf) (hws : oBuf ≠ wsBuf) :
    StoreOnly (gridSpec tokens k n tile_n xBuf wBuf wsBuf oBuf)
      (body ops fp8Dtype outDtype tokens k tile_n) where
  outs_nodup := by simp [gridSpec]
  ins_disjoint := by
    intro p hp o ho
    simp only [gridSpec, List.mem_cons, List.not_mem_nil, or_false] at hp ho
    subst ho
    rcases hp with rfl | rfl | rfl
    · exact hx.symm
    · exact hw.symm
    · exact hws.symm
  ignores_prev := fun _ _ _ _ => rfl

private theorem loadIns_getD_eq (gs : GridSpec) (junk : ℕ → ℝ) (m0 : Mem) (g : List ℕ) (i : ℕ)
    (p : ArrayDesc × BlockSpec) (hp : gs.inSpecs[i]? = some p) :
    (loadIns gs junk m0 g).getD i (fun _ => 0) =
      loadBlock p.1 p.2.blockShape (p.2.blockIdx g (gs.scalarPrefetch m0))
        (junkAt junk (Nat.pair (flatCoord gs.grid g + 1) (i + 1))) m0 := by
  simp [loadIns, List.getD_eq_getElem?_getD, List.getElem?_mapIdx, hp]

/-- End-to-end correctness of `fp8_linear_small_m` via `pallasExec_read_out` and
`foldl_storeBlock_read_last`: for every `(tokens, k, n, tile_n)` meeting `Supports` and every FP8
`fp8Dtype`, the run is defined, every output element `(r, c)` equals `fp8LinearElem` of row `r` of
`x`, column `c` of `w`, and `ws[c]` at compiled shape `(tokens, k, tile_n)`, and memory outside
`oBuf` is unchanged. -/
theorem fp8Linear_correct (ops : FP8LinearOps) (fp8Dtype outDtype : Dtype)
    (hfp8 : fp8Dtype.isFp8 = true)
    (tokens k n tile_n : ℕ) (hsup : Supports tokens k n tile_n)
    (xBuf wBuf wsBuf oBuf : BufId) (hx : oBuf ≠ xBuf) (hw : oBuf ≠ wBuf) (hws : oBuf ≠ wsBuf)
    (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', fp8LinearRun ops fp8Dtype outDtype tokens k n tile_n xBuf wBuf wsBuf oBuf
        junk m0 = some m' ∧
      (∀ (r : Fin tokens) (c : Fin n),
        m'.read oBuf (r.val * n + c.val) =
          fp8LinearElem ops fp8Dtype outDtype tokens k tile_n
            (fun p => m0.read xBuf (r.val * k + p.val))
            (fun p => m0.read wBuf (p.val * n + c.val))
            (m0.read wsBuf c.val)) ∧
      Mem.agreeOutside (Mem.bufFootprint oBuf) m0 m' := by
  have _ := Dtype.fp8Max_pos hfp8
  have htok : 0 < tokens := hsup.tokens_pos
  have hk : 0 < k := hsup.k_pos
  have htile : 0 < tile_n := hsup.tile_pos
  set gs := gridSpec tokens k n tile_n xBuf wBuf wsBuf oBuf with hgs
  set bd := body ops fp8Dtype outDtype tokens k tile_n with hbd
  refine ⟨pallasExec gs bd junk m0,
    pallasCall_of_wellDefined _
      (wellDefined ops fp8Dtype outDtype tokens k n tile_n htok hk htile hsup.grid_pos
        xBuf wBuf wsBuf oBuf hx hw hws m0),
    ?_, ?_⟩
  · intro r c
    set o : ArrayDesc × BlockSpec := (⟨oBuf, [tokens, n]⟩, oSpec tokens tile_n)
    have hq : gs.outSpecs[0]? = some o := by simp [hgs, gridSpec, o]
    have hread := pallasExec_read_out gs bd junk m0
      (storeOnly ops fp8Dtype outDtype tokens k n tile_n xBuf wBuf wsBuf oBuf hx hw hws)
      0 o hq (r.val * n + c.val)
    simp only [o] at hread
    rw [hread]
    have hG : c.val / tile_n < n / tile_n := by
      have hmul : tile_n * (n / tile_n) = n := by
        have hdm := Nat.div_add_mod n tile_n
        rw [hsup.n_mod, Nat.add_zero] at hdm
        exact hdm
      exact Nat.div_lt_of_lt_mul (hmul.symm ▸ c.isLt)
    obtain ⟨l₁, l₂, hsplit, hl₂⟩ := range_split (n / tile_n) (c.val / tile_n) hG
    simp only [seqOut, hgs, gridSpec, gridPoints_one, List.foldl_map, hsplit]
    have hidx : ∀ (j : ℕ) (sc : ℕ → ℤ), (oSpec tokens tile_n).blockIdx [j] sc = [0, j] := by
      intro j sc
      simp [BlockSpec.blockIdx, oSpec]
    simp only [hidx]
    refine (foldl_storeBlock_read_last ⟨oBuf, [tokens, n]⟩ [tokens, tile_n] (fun j => [0, j])
      (fun j => outVal gs bd junk m0 0 [j]) (r.val * n + c.val) l₁ l₂ (c.val / tile_n) _
      (by rw [addrInBlock_colTile htile r.isLt c.isLt]; simp)
      (by
        intro y hy
        rw [addrInBlock_colTile htile r.isLt c.isLt]
        have := (hl₂ y hy).1
        simp only [decide_eq_false_iff_not]
        omega)).trans ?_
    dsimp only
    rw [unflatCoord_two tokens n r.val c.val c.isLt]
    simp only [localCoord, zero_mul, Nat.sub_zero]
    have hcl : c.val - c.val / tile_n * tile_n < tile_n := by
      have := Nat.lt_div_mul_add (a := c.val) htile
      omega
    have hcglob : c.val / tile_n * tile_n + (c.val - c.val / tile_n * tile_n) = c.val := by
      have := Nat.div_mul_le_self c.val tile_n
      omega
    have hxin : ∀ p : Fin k,
        (loadIns gs junk m0 [c.val / tile_n]).getD 0 (fun _ => 0) [r.val, p.val] =
          m0.read xBuf (r.val * k + p.val) := by
      intro p
      rw [loadIns_getD_eq gs junk m0 _ 0 (⟨xBuf, [tokens, k]⟩, xSpec tokens k)
        (by simp [hgs, gridSpec])]
      rw [loadBlock_eq_of_inBounds]
      · simp [xSpec, BlockSpec.blockIdx, globalCoord, flatCoord]
      · simp [xSpec, inBounds, r.isLt, p.isLt]
      · simp [xSpec, BlockSpec.blockIdx, globalCoord, inBounds, r.isLt, p.isLt]
    have hwidx : ∀ (j : ℕ) (sc : ℕ → ℤ), (wSpec k tile_n).blockIdx [j] sc = [0, j] := by
      intro j sc
      simp [BlockSpec.blockIdx, wSpec]
    have hwsidx : ∀ (j : ℕ) (sc : ℕ → ℤ), (wsSpec tile_n).blockIdx [j] sc = [0, j] := by
      intro j sc
      simp [BlockSpec.blockIdx, wsSpec]
    have hwin : ∀ p : Fin k,
        (loadIns gs junk m0 [c.val / tile_n]).getD 1 (fun _ => 0)
          [p.val, c.val - c.val / tile_n * tile_n] =
          m0.read wBuf (p.val * n + c.val) := by
      intro p
      rw [loadIns_getD_eq gs junk m0 _ 1 (⟨wBuf, [k, n]⟩, wSpec k tile_n)
        (by simp [hgs, gridSpec]), hwidx, loadBlock_eq_of_inBounds]
      · simp [wSpec, globalCoord, flatCoord, hcglob]
      · simp [wSpec, inBounds, p.isLt, hcl]
      · simp [wSpec, globalCoord, inBounds, p.isLt, hcglob, c.isLt]
    have hwsin :
        (loadIns gs junk m0 [c.val / tile_n]).getD 2 (fun _ => 0)
          [0, c.val - c.val / tile_n * tile_n] =
          m0.read wsBuf c.val := by
      rw [loadIns_getD_eq gs junk m0 _ 2 (⟨wsBuf, [1, n]⟩, wsSpec tile_n)
        (by simp [hgs, gridSpec]), hwsidx, loadBlock_eq_of_inBounds]
      · simp [wsSpec, globalCoord, flatCoord, hcglob]
      · simp [wsSpec, inBounds, hcl]
      · simp [wsSpec, globalCoord, inBounds, hcglob, c.isLt]
    simp only [outVal, hbd, body, List.getD_cons_zero, List.getD_cons_succ, hxin, hwin, hwsin]
  · refine Mem.agreeOutside_mono ?_ (pallasExec_agreeOutside_outBufs gs bd junk m0)
    rintro ⟨b, a⟩ ⟨x, hx', hbuf⟩
    simp only [hgs, gridSpec, List.mem_singleton] at hx'
    subst hx'
    simpa [Mem.bufFootprint] using hbuf.symm

/-- `fp8LinearSpec` has a well-formed layout whenever `oBuf` is disjoint from all three input
buffers. -/
theorem fp8LinearSpec_wellFormed (ops : FP8LinearOps) (fp8Dtype outDtype : Dtype)
    (tokens k n tile_n : ℕ) (xBuf wBuf wsBuf oBuf : BufId)
    (hx : oBuf ≠ xBuf) (hw : oBuf ≠ wBuf) (hws : oBuf ≠ wsBuf) :
    (fp8LinearSpec ops fp8Dtype outDtype tokens k n tile_n xBuf wBuf wsBuf oBuf).WellFormed where
  readOut_on_footprint _ _ hagree :=
    Layout.read_eq_of_agreeOn _ (Layout.footprint_subset_buf _) hagree
  readSt_on_footprint _ _ _ := rfl
  readIn_outside_footprint _ _ hagree := by
    dsimp only [fp8LinearSpec]
    rw [Layout.read_eq_of_agreeOutside _ (Layout.disjoint_bufFootprint _ hx.symm) hagree,
        Layout.read_eq_of_agreeOutside _ (Layout.disjoint_bufFootprint _ hw.symm) hagree,
        Layout.read_eq_of_agreeOutside _ (Layout.disjoint_bufFootprint _ hws.symm) hagree]

/-- `fp8_linear_small_m` satisfies its `KernelSpec`. -/
theorem fp8LinearSpec_correct (ops : FP8LinearOps) (fp8Dtype outDtype : Dtype)
    (tokens k n tile_n : ℕ) (xBuf wBuf wsBuf oBuf : BufId) :
    (fp8LinearSpec ops fp8Dtype outDtype tokens k n tile_n xBuf wBuf wsBuf oBuf).Correct
      (fp8LinearRun ops fp8Dtype outDtype tokens k n tile_n xBuf wBuf wsBuf oBuf) := by
  intro junk m0 ⟨hsup, hfp8, hx, hw, hws⟩
  refine ⟨fp8LinearSpec_wellFormed ops fp8Dtype outDtype tokens k n tile_n
    xBuf wBuf wsBuf oBuf hx hw hws, ?_⟩
  obtain ⟨m', hcall, hval, hframe⟩ :=
    fp8Linear_correct ops fp8Dtype outDtype hfp8 tokens k n tile_n hsup
      xBuf wBuf wsBuf oBuf hx hw hws junk m0
  refine ⟨m', hcall, ?_, rfl, hframe⟩
  apply Tensor.ext2
  intro r c
  simpa [fp8LinearSpec, fp8LinearMath, Layout.rowMajor, Layout.read, Tensor.get2, Tensor.of2]
    using hval r c

/-- Fixed-shape element independence under the `FP8LinearOps` assumptions: at a fixed compiled shape
`(tokens, k, n, tile_n)`, output element `(r, c)` is bit-for-bit identical across any two runs
(any buffers, junk, and values in other rows/columns) whenever row `r` of `x`, column `c` of `w`,
and `ws[c]` match. Does not assert invariance across different `tokens` or `tile_n`. -/
theorem fp8Linear_fixedShape_elem_independent (ops : FP8LinearOps) (fp8Dtype outDtype : Dtype)
    (hfp8 : fp8Dtype.isFp8 = true)
    {tokens k n tile_n : ℕ} (hsup : Supports tokens k n tile_n)
    {xBuf wBuf wsBuf oBuf xBuf' wBuf' wsBuf' oBuf' : BufId}
    (hx : oBuf ≠ xBuf) (hw : oBuf ≠ wBuf) (hws : oBuf ≠ wsBuf)
    (hx' : oBuf' ≠ xBuf') (hw' : oBuf' ≠ wBuf') (hws' : oBuf' ≠ wsBuf')
    {junk junk' : ℕ → ℝ} {m0 m0' m₁ m₂ : Mem}
    (h₁ : fp8LinearRun ops fp8Dtype outDtype tokens k n tile_n xBuf wBuf wsBuf oBuf
      junk m0 = some m₁)
    (h₂ : fp8LinearRun ops fp8Dtype outDtype tokens k n tile_n xBuf' wBuf' wsBuf' oBuf'
      junk' m0' = some m₂)
    (r r' : Fin tokens) (c c' : Fin n)
    (hxr : ∀ p : Fin k, m0.read xBuf (r.val * k + p.val) = m0'.read xBuf' (r'.val * k + p.val))
    (hwc : ∀ p : Fin k, m0.read wBuf (p.val * n + c.val) = m0'.read wBuf' (p.val * n + c'.val))
    (hwsc : m0.read wsBuf c.val = m0'.read wsBuf' c'.val) :
    m₁.read oBuf (r.val * n + c.val) = m₂.read oBuf' (r'.val * n + c'.val) := by
  obtain ⟨n₁, e₁, v₁, -⟩ :=
    fp8Linear_correct ops fp8Dtype outDtype hfp8 tokens k n tile_n hsup
      xBuf wBuf wsBuf oBuf hx hw hws junk m0
  obtain ⟨n₂, e₂, v₂, -⟩ :=
    fp8Linear_correct ops fp8Dtype outDtype hfp8 tokens k n tile_n hsup
      xBuf' wBuf' wsBuf' oBuf' hx' hw' hws' junk' m0'
  rw [e₁, Option.some.injEq] at h₁
  rw [e₂, Option.some.injEq] at h₂
  subst h₁ h₂
  rw [v₁ r c, v₂ r' c']
  congr 1
  · funext p
    exact hxr p
  · funext p
    exact hwc p

/-- Fixed-shape row independence: at a fixed compiled shape `(tokens, k, n, tile_n)` with matching
weights `w` and scales `ws`, output row `r` depends only on input row `r` of `x`, independent of
junk and the values in all other rows of `x`. Does not assert invariance across different `tokens`
or `tile_n`. -/
theorem fp8Linear_fixedShape_row_independent (ops : FP8LinearOps) (fp8Dtype outDtype : Dtype)
    (hfp8 : fp8Dtype.isFp8 = true)
    {tokens k n tile_n : ℕ} (hsup : Supports tokens k n tile_n)
    {xBuf wBuf wsBuf oBuf xBuf' wBuf' wsBuf' oBuf' : BufId}
    (hx : oBuf ≠ xBuf) (hw : oBuf ≠ wBuf) (hws : oBuf ≠ wsBuf)
    (hx' : oBuf' ≠ xBuf') (hw' : oBuf' ≠ wBuf') (hws' : oBuf' ≠ wsBuf')
    {junk junk' : ℕ → ℝ} {m0 m0' m₁ m₂ : Mem}
    (h₁ : fp8LinearRun ops fp8Dtype outDtype tokens k n tile_n xBuf wBuf wsBuf oBuf
      junk m0 = some m₁)
    (h₂ : fp8LinearRun ops fp8Dtype outDtype tokens k n tile_n xBuf' wBuf' wsBuf' oBuf'
      junk' m0' = some m₂)
    (r r' : Fin tokens)
    (hxr : ∀ p : Fin k, m0.read xBuf (r.val * k + p.val) = m0'.read xBuf' (r'.val * k + p.val))
    (hwAll : ∀ (p : Fin k) (c : Fin n),
      m0.read wBuf (p.val * n + c.val) = m0'.read wBuf' (p.val * n + c.val))
    (hwsAll : ∀ c : Fin n, m0.read wsBuf c.val = m0'.read wsBuf' c.val)
    (c : Fin n) :
    m₁.read oBuf (r.val * n + c.val) = m₂.read oBuf' (r'.val * n + c.val) :=
  fp8Linear_fixedShape_elem_independent ops fp8Dtype outDtype hfp8 hsup
    hx hw hws hx' hw' hws' h₁ h₂ r r' c c hxr (fun p => hwAll p c) (hwsAll c)

/-- Under exact real arithmetic with an FP8 dtype (`fp8Dtype.isFp8 = true`, so
`0 < fp8Dtype.fp8Max`), every scaled element `x_p / x_scale` lies in `[-fp8Max, fp8Max]`,
justifying why `fp8_linear_small_m.py:36-37` omits `jnp.clip` before the unsaturated
`.astype(fp8_dtype)` cast. -/
theorem fp8LinearScaled_abs_le_exact {fp8Dtype : Dtype} (hfp8 : fp8Dtype.isFp8 = true)
    (tokens k tile_n : ℕ) (xr : Fin k → ℝ) (p : Fin k) :
    |fp8LinearScaled FP8LinearOps.exact fp8Dtype tokens k tile_n xr p| ≤ fp8Dtype.fp8Max := by
  cases k with
  | zero => exact p.elim0
  | succ m =>
    have hmax_pos : 0 < fp8Dtype.fp8Max := Dtype.fp8Max_pos hfp8
    have hfloor_pos : (0 : ℝ) < scaleFloor := by norm_num [scaleFloor]
    change |xr p / max (Tensor.vecMax (fun q : Fin (m + 1) => |xr q|) / fp8Dtype.fp8Max)
      scaleFloor| ≤ fp8Dtype.fp8Max
    set amax : ℝ := Tensor.vecMax fun q : Fin (m + 1) => |xr q|
    set xScale : ℝ := max (amax / fp8Dtype.fp8Max) scaleFloor
    have hxScale_pos : 0 < xScale := lt_of_lt_of_le hfloor_pos (le_max_right _ _)
    have hle_amax : |xr p| ≤ amax := Tensor.le_vecMax (fun q : Fin (m + 1) => |xr q|) p
    have hamax_div : amax / fp8Dtype.fp8Max ≤ xScale := le_max_left _ _
    have hamax_le : amax ≤ xScale * fp8Dtype.fp8Max := (div_le_iff₀ hmax_pos).mp hamax_div
    have hnum : |xr p| ≤ fp8Dtype.fp8Max * xScale := by
      rw [mul_comm]
      exact le_trans hle_amax hamax_le
    rw [abs_div, abs_of_pos hxScale_pos]
    exact (div_le_iff₀ hxScale_pos).mpr hnum

/-- Under exact real arithmetic with an FP8 dtype (`fp8Dtype.isFp8 = true`, tying `fp8Max` to
`fp8Dtype.fp8Max > 0`), the positive `1e-12` scale floor ensures `x_scale > 0` even on all-zero
rows, so `x_scale` cancels out exactly and `fp8LinearElem` equals `(∑ p, xr p * wc p) * ws`. -/
theorem fp8LinearElem_exact {fp8Dtype : Dtype} (hfp8 : fp8Dtype.isFp8 = true)
    (outDtype : Dtype) (tokens k tile_n : ℕ) (xr wc : Fin k → ℝ) (ws : ℝ) :
    fp8LinearElem FP8LinearOps.exact fp8Dtype outDtype tokens k tile_n xr wc ws =
      (∑ p : Fin k, xr p * wc p) * ws := by
  have _ := Dtype.fp8Max_pos hfp8
  simp only [fp8LinearElem, fp8LinearScale, fp8LinearScaled, FP8LinearOps.exact,
    CastOps.exact_cast]
  change (∑ p : Fin k,
      xr p / max (exactRowMax k (fun p => |xr p|) / fp8Dtype.fp8Max) scaleFloor * wc p) *
      max (exactRowMax k (fun p => |xr p|) / fp8Dtype.fp8Max) scaleFloor * ws =
    (∑ p : Fin k, xr p * wc p) * ws
  set xScale : ℝ := max (exactRowMax k (fun p => |xr p|) / fp8Dtype.fp8Max) scaleFloor
  have hpos : 0 < xScale :=
    lt_of_lt_of_le (by norm_num [scaleFloor]) (le_max_right _ _)
  have hne : xScale ≠ 0 := ne_of_gt hpos
  congr 1
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun p _ => ?_
  rw [mul_right_comm, div_mul_cancel₀ _ hne]

/-- Under exact real arithmetic with an FP8 dtype (`fp8Dtype.isFp8 = true`), `fp8LinearMath` equals
`Tensor.matmul x w` scaled column-wise by `ws`. -/
theorem fp8LinearMath_exact {fp8Dtype : Dtype} (hfp8 : fp8Dtype.isFp8 = true)
    (outDtype : Dtype) (tile_n : ℕ) {tokens k n : ℕ}
    (x : Tensor ℝ [tokens, k]) (w : Tensor ℝ [k, n]) (ws : Tensor ℝ [1, n]) :
    fp8LinearMath FP8LinearOps.exact fp8Dtype outDtype tile_n x w ws =
      Tensor.of2 fun r c => (Tensor.matmul x w).get2 r c * ws.get2 0 c := by
  apply Tensor.ext2
  intro r c
  simp only [fp8LinearMath, Tensor.get2_of2, Tensor.get2_matmul]
  exact fp8LinearElem_exact hfp8 outDtype tokens k tile_n _ _ _

end TpuKernelLean.Examples.Fp8LinearSmallM
