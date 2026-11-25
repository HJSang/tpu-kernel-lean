import TpuKernelLean.Examples.BlockScaledMm

/-!
Tests of the block-scaled matmul kernels:
1. the hypotheses of `blockwise128ScaledMm_correct` (with the bias input) are satisfiable: a
   concrete shape with six distinct buffers;
2. the smallest K1 shape under exact arithmetic computes `a * b * sa * sb`.
-/

namespace TpuKernelLean.Tests.BlockScaledMm

open TpuKernelLean.Pallas
open TpuKernelLean.Examples.BlockScaledMm

/-- K2 with bias at `gm = 2`, `gn = 1`, `gk = 2`, `tile_m = 1`, `tile_n = 2`, `steps_k = 1`
(`a [2, 256]`, `b [256, 2]`, `scale_a [2, 2]`, `scale_b [2, 2]`, `bias [2]`) on buffers `0..5`. -/
theorem blockwise128ScaledMm_correct_concrete (ops : ScaledMmOps) (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', pallasCall (blockwise128GridSpec 2 1 2 1 2 1 true 0 1 2 3 4 5)
        (blockwise128Body ops 2 1 2 1 true) junk m0 = some m' ∧
      (∀ (r : Fin (2 * 1)) (c : Fin (1 * 2)),
        m'.read 5 (r.val * (1 * 2) + c.val) =
          blockwise128Elem ops 2 1 2 1 true
            (fun k => m0.read 0 (r.val * (2 * (1 * 128)) + k))
            (fun k => m0.read 1 (k * (1 * 2) + c.val))
            (fun j => m0.read 2 (j * (2 * 1) + r.val))
            (fun j => m0.read 3 (j * (1 * 2) + c.val))
            (m0.read 4 c.val)) ∧
      Mem.agreeOutside (Mem.bufFootprint 5) m0 m' :=
  blockwise128ScaledMm_correct ops 2 1 2 1 2 1 true 0 1 2 3 4 5 (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (fun _ => by decide) junk m0

/-- K1 at `gm = gn = gk = tile_m = tile_n = steps_k = bs = 1` with exact arithmetic: the single
output element is `a * b * sa * sb`. -/
theorem blockwiseScaledMm_smallest_exact (aBuf bBuf saBuf sbBuf oBuf : BufId)
    (ha : oBuf ≠ aBuf) (hb : oBuf ≠ bBuf) (hsa : oBuf ≠ saBuf) (hsb : oBuf ≠ sbBuf)
    (junk : ℕ → ℝ) (m0 : Mem) :
    ∃ m', pallasCall (blockwiseGridSpec 1 1 1 1 1 1 1 aBuf bBuf saBuf sbBuf oBuf)
        (blockwiseBody ScaledMmOps.exact 1 1 1 1 1) junk m0 = some m' ∧
      m'.read oBuf 0 =
        m0.read aBuf 0 * m0.read bBuf 0 * m0.read saBuf 0 * m0.read sbBuf 0 := by
  obtain ⟨m', hcall, hval, -⟩ := blockwiseScaledMm_correct ScaledMmOps.exact 1 1 1 1 1 1 1
    aBuf bBuf saBuf sbBuf oBuf (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) ha hb hsa hsb junk m0
  refine ⟨m', hcall, ?_⟩
  have h0 := hval ⟨0, by decide⟩ ⟨0, by decide⟩
  rw [blockwiseScaledMmElem_exact] at h0
  have h1 : m'.read oBuf 0 =
      m0.read saBuf 0 * m0.read sbBuf 0 * (m0.read aBuf 0 * m0.read bBuf 0) := by
    simpa using h0
  rw [h1]
  ring

end TpuKernelLean.Tests.BlockScaledMm
