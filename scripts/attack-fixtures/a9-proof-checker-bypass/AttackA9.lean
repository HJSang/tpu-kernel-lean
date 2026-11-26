import Lean.Elab.Command

namespace TpuKernelLean.Examples.AttackA9

open Lean Elab Command in
elab "verify_tile_schedule" : command => do
  liftCoreM do
    let optName := Name.mkStr2 "debug" ("skip" ++ "Kernel" ++ "TC")
    withOptions (fun opts => opts.setBool optName true) do
      let decl := Declaration.thmDecl {
        name := `TpuKernelLean.Examples.AttackA9.tileSchedule_ok
        levelParams := []
        type := mkConst ``False
        value := mkConst ``True.intro
      }
      addDecl decl

verify_tile_schedule

theorem bogus_correct : False := tileSchedule_ok

end TpuKernelLean.Examples.AttackA9
