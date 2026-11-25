import Lean

/-!
Standalone axiom, safety, trusted-import, and syntax gate for `TpuKernelLean`.

Run from the repository root with:
  lake env lean --run scripts/CheckAxioms.lean
  lake env lean --run scripts/CheckAxioms.lean --base DIR --root NAME

Fails (exit code 1) if:
1. Any declaration under the target root (public, private, or defined in a target module)
   is `unsafe`/`partial`, `opaque`, `@[extern]`, `@[implemented_by]`, or depends on
   any axiom outside `{propext, Classical.choice, Quot.sound}`.
2. For `TpuKernelLean`, any trusted model module (`Tensor`, `Memory`, `Layout`, `Spec`,
   `Dtype`, `Pallas.Core`, `Pallas.DMA`) imports a non-trusted local module.
3. Any `.lean` file under the target root is not imported by `<root>.lean`.
4. Any `.lean` file under the target root fails to parse, contains an `example` declaration,
   or contains a forbidden identifier/token outside comments and string/char literals.
-/

namespace CheckAxioms

open Lean

/-- The three standard Lean/Mathlib axioms permitted for every declaration. -/
private def standardAxioms : List Name :=
  [``propext, ``Classical.choice, ``Quot.sound]

/-- Identifiers and tokens forbidden anywhere in parsed non-comment/non-string syntax. -/
private def forbiddenIdents : List String :=
  [ "sorry", "admit", "native_decide", "sorryAx",
    "ofReduceBool", "trustCompiler", "skipKernelTC",
    "addDeclWithoutChecking", "mkSorry", "admitGoal",
    "implemented_by", "extern" ]

/-- Trusted model modules whose direct imports must stay within the trusted layer and standard/Mathlib libraries. -/
private def trustedModules : List Name :=
  [ `TpuKernelLean.Tensor,
    `TpuKernelLean.Memory,
    `TpuKernelLean.Layout,
    `TpuKernelLean.Spec,
    `TpuKernelLean.Dtype,
    `TpuKernelLean.Pallas.Core,
    `TpuKernelLean.Pallas.DMA ]

/-- Check whether `root` is the outermost component of `name`. -/
private def hasRoot (root : Name) : Name → Bool
  | .anonymous => false
  | .str .anonymous s => root == .str .anonymous s
  | .str p _ => hasRoot root p
  | .num p _ => hasRoot root p

/-- Check whether a constant in `env` belongs to `targetRoot` (public, private, or module-owned). -/
private def isTargetConstant (env : Environment) (targetRoot : Name) (name : Name) : Bool :=
  hasRoot targetRoot name ||
  (isPrivateName name && (privateToUserName? name).any (hasRoot targetRoot)) ||
  match env.getModuleIdxFor? name with
  | some idx => (env.header.moduleNames[idx]?).any (hasRoot targetRoot)
  | none => false

/-- Format a constant name for diagnostics, noting private declarations. -/
private def displayName (name : Name) : String :=
  if isPrivateName name then
    match privateToUserName? name with
    | some userName => s!"{userName} (private)"
    | none => name.toString
  else
    name.toString

/-- Check whether `m` is an allowed direct import of a trusted model module. -/
private def isAllowedTrustedImport (m : Name) : Bool :=
  trustedModules.contains m ||
  let s := m.toString
  s == "Mathlib" || s.startsWith "Mathlib." ||
  s == "Init" || s.startsWith "Init." ||
  s == "Std" || s.startsWith "Std." ||
  s == "Lean" || s.startsWith "Lean." ||
  s == "Batteries" || s.startsWith "Batteries."

/-- Verify that every trusted model module imports only trusted modules and standard/Mathlib libraries. -/
private def checkTrustedImports (env : Environment) : Array String := Id.run do
  let mut errs : Array String := #[]
  for m in trustedModules do
    let some idx := env.getModuleIdx? m
      | errs := errs.push s!"trusted module `{m}` not found in environment"; continue
    for imp in env.header.moduleData[idx]!.imports do
      unless isAllowedTrustedImport imp.module do
        errs := errs.push s!"trusted module `{m}` has forbidden import `{imp.module}`"
  return errs

/-- Relative path components of `p` with respect to `base`. -/
private def relComponents (base p : System.FilePath) : List String :=
  let b := base.components.dropWhile (· == ".")
  let c := p.components.dropWhile (· == ".")
  if b.isPrefixOf c then c.drop b.length else c

/-- Convert relative `.lean` path components such as `["TpuKernelLean", "Spec.lean"]` to a module `Name`. -/
private def pathToModuleName (comps : List String) : Option Name :=
  match comps.reverse with
  | [] => none
  | last :: revInit =>
    if last.endsWith ".lean" then
      let stem := (last.dropEnd 5).toString
      let parts := revInit.reverse ++ [stem]
      some (parts.foldl (fun acc s => Name.str acc s) Name.anonymous)
    else
      none

/-- Walk parsed Lean `Syntax` (immune to comments, raw strings, and guillemets). -/
private partial def scanSyntax (file : System.FilePath) (fm : FileMap) (stx : Syntax)
    (acc : Array String) : Array String := Id.run do
  let mut out := acc
  let lineOf (s : Syntax) : Nat :=
    match s.getPos? with
    | some pos => (fm.toPosition pos).line
    | none => 0
  match stx with
  | .node _ kind args =>
    -- Skip string/char literals, interpolated-string literal chunks, and doc comments.
    -- Expressions inside `s!"{...}"` are still scanned as child nodes of `interpolatedStrKind`.
    if kind == strLitKind || kind == charLitKind || kind == interpolatedStrLitKind ||
       kind == ``Lean.Parser.Command.docComment || kind == ``Lean.Parser.Command.moduleDoc then
      return out
    if kind == ``Lean.Parser.Command.example then
      out := out.push s!"{file}:{lineOf stx}: forbidden `example` declaration (use a named `theorem`)"
    for arg in args do
      out := scanSyntax file fm arg out
  | .ident _ _ val _ =>
    let tail := match val with | .str _ s => s | _ => val.toString
    if forbiddenIdents.contains tail then
      out := out.push s!"{file}:{lineOf stx}: forbidden identifier `{val}`"
  | .atom _ val =>
    if forbiddenIdents.contains val then
      out := out.push s!"{file}:{lineOf stx}: forbidden token `{val}`"
  | .missing => pure ()
  return out

/-- Parse a `.lean` file with Lean's own parser and scan every command's `Syntax`. -/
private def scanFileSyntax (env : Environment) (p : System.FilePath) : IO (Array String) := do
  let src ← IO.FS.readFile p
  let inputCtx := Parser.mkInputContext src p.toString
  let (_header, parserState, messages) ← Parser.parseHeader inputCtx
  let pmctx : Parser.ParserModuleContext := { env := env, options := {} }
  let mut st := parserState
  let mut msgs := messages
  let mut failures : Array String := #[]
  if msgs.hasErrors then
    failures := failures.push s!"{p}: failed to parse module header"
    return failures
  repeat do
    let (cmd, st', msgs') := Parser.parseCommand inputCtx pmctx st msgs
    st := st'
    msgs := msgs'
    if Parser.isTerminalCommand cmd then break
    failures := scanSyntax p inputCtx.fileMap cmd failures
  if msgs.hasErrors then
    let mut errTexts : Array String := #[]
    for msg in msgs.toList do
      if msg.severity == .error then
        errTexts := errTexts.push s!"L{msg.pos.line}: {← msg.data.toString}"
        if errTexts.size >= 1 then break
    let detail := String.intercalate "; " (Array.toList errTexts)
    failures := failures.push s!"{p}: failed to parse source file cleanly ({detail})"
  return failures

/-- Scan `<root>.lean` and `<root>/` source files for unimported modules and forbidden syntax. -/
private def scanSources (env parseEnv : Environment) (base : System.FilePath) (targetRoot : Name) :
    IO (Nat × Array String) := do
  let rootStr := targetRoot.toString
  let rootDir : System.FilePath := base / rootStr
  let rootFile : System.FilePath := base / (rootStr ++ ".lean")
  let mut leanFiles : Array System.FilePath := #[]
  if ← rootFile.pathExists then
    leanFiles := leanFiles.push rootFile
  if ← rootDir.pathExists then
    for p in ← rootDir.walkDir do
      if !(← p.isDir) && p.extension == some "lean" then
        leanFiles := leanFiles.push p
  leanFiles := leanFiles.qsort fun a b => a.toString < b.toString
  if leanFiles.isEmpty then
    return (0, #[s!"no `.lean` files found for `{targetRoot}` under `{base}`"])
  let importedMods := env.header.moduleNames
  let mut failures : Array String := #[]
  for p in leanFiles do
    let comps := relComponents base p
    match pathToModuleName comps with
    | some modName =>
      if !importedMods.contains modName then
        failures := failures.push s!"{p}: module `{modName}` is not imported by `{targetRoot}.lean`"
    | none => pure ()
    failures := failures ++ (← scanFileSyntax parseEnv p)
  return (leanFiles.size, failures)

/-- True if `e` is a (possibly dependent) function type whose return type is `Nat`. -/
private def returnsNat : Expr → Bool
  | .forallE _ _ b _ => returnsNat b
  | .const ``Nat [] => true
  | _ => false

/-- True for compiler-generated `I.ctorIdx` and `I.ctorIdx._impl` on multi-constructor inductives. -/
private def isCompilerCtorIdx (env : Environment) (name : Name) (cinfo : ConstantInfo) : Bool :=
  returnsNat cinfo.type &&
  match name with
  | .str parent "ctorIdx" =>
    match env.find? parent with
    | some (.inductInfo info) =>
      !info.isUnsafe && info.ctors.length > 1 &&
      Lean.Compiler.implementedByAttr.getParam? env name == some (name ++ `_impl)
    | _ => false
  | .str (.str parent "ctorIdx") "_impl" =>
    match env.find? parent, env.find? (parent ++ `ctorIdx) with
    | some (.inductInfo info), some ctorIdxInfo =>
      !info.isUnsafe && info.ctors.length > 1 && !ctorIdxInfo.isUnsafe &&
      ctorIdxInfo.type == cinfo.type &&
      Lean.Compiler.implementedByAttr.getParam? env (parent ++ `ctorIdx) == some name
    | _, _ => false
  | _ => false

/-- Audit every target declaration's safety/axioms, trusted imports, and source files. -/
private def checkEnv (parseEnv : Environment) (base : System.FilePath) (targetRoot : Name) :
    CoreM (UInt32 × String) := do
  let env ← getEnv
  let mut failures : Array String := #[]
  let rawTargets := env.constants.fold (init := (#[] : Array (Name × ConstantInfo))) fun acc name cinfo =>
    if isTargetConstant env targetRoot name then acc.push (name, cinfo) else acc
  let targets := rawTargets.qsort fun a b => a.1.toString < b.1.toString
  let mut thmCount : Nat := 0
  for (name, cinfo) in targets do
    if cinfo matches .thmInfo _ then
      thmCount := thmCount + 1
    let isCtorIdx := isCompilerCtorIdx env name cinfo
    if cinfo.isUnsafe && !isCtorIdx then
      failures := failures.push s!"{displayName name}: `unsafe` (or `partial`) declaration is forbidden"
    if cinfo matches .opaqueInfo _ then
      failures := failures.push s!"{displayName name}: `opaque` declaration is forbidden"
    if Lean.isExtern env name then
      failures := failures.push s!"{displayName name}: `@[extern]` declaration is forbidden"
    if (Lean.Compiler.implementedByAttr.getParam? env name).isSome && !isCtorIdx then
      failures := failures.push s!"{displayName name}: `@[implemented_by]` declaration is forbidden"
    let axioms ← Lean.collectAxioms name
    let bad := axioms.filter fun ax => !standardAxioms.contains ax
    if !bad.isEmpty then
      let axList := String.intercalate ", " ((Array.toList bad).map Name.toString)
      failures := failures.push s!"{displayName name}: disallowed axiom(s): {axList}"
  if targetRoot == `TpuKernelLean then
    failures := failures ++ checkTrustedImports env
  let (fileCount, srcFailures) ← scanSources env parseEnv base targetRoot
  failures := failures ++ srcFailures
  if !failures.isEmpty then
    let report := String.intercalate "\n  " (Array.toList failures)
    return (1, s!"Axiom gate FAILED ({failures.size} issue(s)):\n  {report}")
  return (0, s!"Axiom gate passed: checked {targets.size} declarations ({thmCount} theorems) in {targetRoot} and scanned {fileCount} source files.")

private def parseArgs : List String → System.FilePath × Name
  | "--base" :: b :: rest =>
    let (_, r) := parseArgs rest; (b, r)
  | "--root" :: r :: rest =>
    let (b, _) := parseArgs rest; (b, r.toName)
  | _ :: rest => parseArgs rest
  | [] => (".", `TpuKernelLean)

end CheckAxioms

unsafe def main (args : List String) : IO UInt32 := do
  let (base, targetRoot) := CheckAxioms.parseArgs args
  Lean.initSearchPath (← Lean.findSysroot)
  -- One environment serves both the declaration checks and the parser (which needs the
  -- extensions loaded). Loading Mathlib a second time doubled memory and got the gate killed
  -- on a GitHub runner.
  Lean.enableInitializersExecution
  let env ← Lean.importModules #[{ module := targetRoot }] {} (trustLevel := 0) (loadExts := true)
  let ctx : Lean.Core.Context := { fileName := "<CheckAxioms>", fileMap := default }
  let ((code, msg), _) ← (CheckAxioms.checkEnv env base targetRoot).toIO ctx { env := env }
  if code == 0 then IO.println msg else IO.eprintln msg
  return code
