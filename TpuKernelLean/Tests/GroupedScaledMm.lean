import TpuKernelLean.Examples.GroupedScaledMm

/-!
Test of the grouped matmul schedule predicate: `ValidSchedule` is satisfiable on the case the
kernel's masked store exists for, two real groups sharing an M tile.
-/

namespace TpuKernelLean.Tests.GroupedScaledMm

open TpuKernelLean.Examples.GroupedScaledMm

/-- Group offsets for group ends `[1, 3]` and `M = 3` rows: group 0 owns row 0, group 1 owns rows
1 and 2, and the virtual group owns no row. -/
def go : ℕ → ℕ := fun i => [0, 1, 3, 3].getD i 3

/-- The group of each of the four visits: group 0, group 1 twice, the virtual group. -/
def gid : ℕ → ℕ := fun v => [0, 1, 1, 2].getD v 2

/-- The M tile of each of the four visits. -/
def mt : ℕ → ℕ := fun v => [0, 0, 1, 1].getD v 1

/-- `M = 3` rows, two M tiles of two rows, `G = 2` groups with ends `[1, 3]`, `V = 4` visits:
groups 0 and 1 both visit M tile 0 (visits 0 and 1), and group 1 also visits M tile 1. -/
theorem validSchedule_shared_tile : ValidSchedule go gid mt 2 2 2 3 4 := by
  constructor <;> decide

end TpuKernelLean.Tests.GroupedScaledMm
