import Lean

/-!
Standalone axiom, safety, trusted-import, syntax, protected-name, index-map, and headline-theorem
gate for `TpuKernelLean`.

Run from the repository root with:
  lake env lean --run scripts/CheckAxioms.lean --scan-only
  lake env lean --run scripts/CheckAxioms.lean --check-env
  lake env lean --run scripts/CheckAxioms.lean [--base DIR] [--root NAME] [--audit-root NAME]
                                               [--parse-root NAME] [--manifest PATH]

Execution architecture (two phases, one Mathlib environment per process):
- Phase 1 (`--scan-only`): loads ONLY a trusted parser environment (`--parse-root`, default
  `Mathlib`, never the target library) so no untrusted `initialize` block can execute before the
  source scan passes. Scans `<base>/<root>.lean` and `<base>/<root>/**/*.lean`.
- Phase 2 (`--check-env`): loads `<root>` and audits declarations, imports, protected names,
  index maps, and the `manifest.json` headline-theorem / witness registry.
- When invoked with neither flag, `main` spawns `--scan-only` as a child process FIRST and exits
  immediately on failure before loading `<root>`.

Fails (exit code 1) if:
1. Source scan (`--scan-only`), matched on parsed `Syntax` node kinds (comments/strings ignored):
   - Any file fails to parse cleanly.
   - Any file imports a module whose root is `Lean` or `Lake`.
   - Any file contains `example`, `elab`, `elab_rules`, `macro`, `macro_rules`, `syntax`,
     `declare_syntax_cat`, `notation`, `infix`/`infixl`/`infixr`/`prefix`/`postfix` (including
     `local`/`scoped` forms), `run_cmd`, `run_elab`, `run_meta`, `initialize`,
     `builtin_initialize`, or `#eval`.
   - Any file contains `set_option` for an option outside `allowedOptions` (currently empty,
     matching what `TpuKernelLean` uses today).
   - Any declaration (including structure fields, constructors, and `namespace` headers) has a
     non-ASCII character in its name (binder and local variable names such as `ε` remain allowed).
   - Any non-comment/non-string identifier or token is in `forbiddenIdents`.
2. Declaration safety & axioms (`--check-env`):
   - Any declaration under `auditRoot` (public, private, or module-owned) is `unsafe`/`partial`
     (excluding compiler-generated `ctorIdx` / `ctorIdx._impl`), `opaque`, `@[extern]`,
     `@[implemented_by]`, has a non-ASCII user name, or depends on any axiom outside
     `{propext, Classical.choice, Quot.sound}`.
   - Any module under `auditRoot` imports `Lean.*` or `Lake.*`, or any `.lean` file under
     `<base>/<auditRoot>/` is not imported by `<root>`.
   - For `TpuKernelLean`, any trusted model module (`Tensor`, `Memory`, `Layout`, `Spec`, `Dtype`,
     `Pallas.Core`, `Pallas.DMA`) imports a non-trusted local module.
3. Protected names & namespaces (for `TpuKernelLean`, case-sensitive on user names including
   private declarations):
   - No declaration outside the trusted model modules (`Tensor`, `Memory`, `Layout`, `Spec`,
     `Dtype`, `Pallas.Core`, `Pallas.DMA`) may have:
     (a) a final name component in `protectedFinalComponents` (`pallasCall`, `pallasExec`,
         `Correct`, `WellDefined`, `agreeOutside`, `bufFootprint`, `KernelSpec`, `GridSpec`,
         `BlockSpec`, `Body`, `Mem`, `ArrayDesc`, `AliasesValid`, `withinTol`); or
     (b) final two name components in `protectedFinalTwo` (`Pallas.*` model types/runners,
         `KernelSpec.Correct`, `GridSpec.WellDefined`, `GridSpec.AliasesValid`,
         `Mem.{agreeOutside,bufFootprint,read,write}`,
         `Layout.{read,footprint,rowMajor,Injective,Disjoint}`); or
     (c) a prefix inside any protected model namespace (`TpuKernelLean.KernelSpec`,
         `TpuKernelLean.Mem`, `TpuKernelLean.Layout`, `TpuKernelLean.Tensor`,
         `TpuKernelLean.Dtype`, `TpuKernelLean.Pallas.{GridSpec,BlockSpec,ArrayDesc,Body,
         DimSemantics,DMA}`);
   - Furthermore, no module outside `TpuKernelLean.Pallas.*` may declare any constant inside the
     `TpuKernelLean.Pallas` namespace.
   Compiler-generated equational lemmas and matcher auxiliaries (`.eq_*`, `.eq_def`, `.eq_unfold`,
   `.splitter*`, `.congr_eq_*`, `._sparseCasesOn_*`) lazily synthesized by Lean in downstream
   modules for an existing upstream constant declared in a trusted or `TpuKernelLean.Pallas.*`
   module are exempted from the namespace-prefix rules (the reserved final-component checks still
   apply).
   Legitimate lowercase names such as `RowKernel.correct`, `ReductionKernel.correct`, and
   `ReductionKernel.wellDefined` remain allowed because comparison is case-sensitive.
4. Index-map natural-subtraction ban (for `TpuKernelLean`):
   - For every non-theorem declaration, every `indexMap` argument passed to `Pallas.BlockSpec.mk`
     and every `outIndexMap` argument passed to `Pallas.ReductionKernel.mk` (following non-theorem
     helper definitions in `TpuKernelLean`) must not contain truncating natural-number subtraction
     or predecessor (`Nat.sub`, `Nat.pred`, or `HSub.hSub`/`Sub.sub` at type `ℕ`). Signed `ℤ`
     subtraction is allowed.
5. Headline-theorem and witness audit against `manifest.json` (for `TpuKernelLean` or `--manifest`):
   - Every `correct` and `other` theorem exists, is a non-private `theorem`, is declared in the
     entry's `lean_module`, and is not trivially `True`.
   - Every `pallas_call` entry has non-empty `correct` and a non-null `witness`.
   - Every `correct` theorem reaches the genuine `TpuKernelLean.Pallas.pallasCall` when following
     non-theorem definitions in its own module, `Pallas.RowKernel`, `Pallas.ReductionKernel`, and
     `Spec`.
   - Every `witness` theorem exists, is a non-private `theorem` in a `TpuKernelLean.Tests.*`
     module, is not trivially `True`, reaches `TpuKernelLean.Pallas.pallasCall`, and shares at
     least one kernel-defining non-theorem definition (`GridSpec`, `Body`, `RowKernel`, or
     `ReductionKernel`) declared in the entry's `lean_module` with a `correct` theorem of the entry.
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

/-- Allowlist of options permitted in `set_option` commands.
Derived from what `TpuKernelLean/` uses today (none). -/
private def allowedOptions : List Name := []

/-- Trusted model modules whose direct imports must stay within the trusted layer and standard/Mathlib libraries. -/
private def trustedModules : List Name :=
  [ `TpuKernelLean.Tensor,
    `TpuKernelLean.Memory,
    `TpuKernelLean.Layout,
    `TpuKernelLean.Spec,
    `TpuKernelLean.Dtype,
    `TpuKernelLean.Pallas.Core,
    `TpuKernelLean.Pallas.DMA ]

/-- Protected final name components (case-sensitive) forbidden outside `trustedModules`. -/
private def protectedFinalComponents : List String :=
  [ "pallasCall", "pallasExec", "Correct", "WellDefined",
    "agreeOutside", "bufFootprint", "KernelSpec", "GridSpec",
    "BlockSpec", "Body", "Mem", "ArrayDesc", "AliasesValid",
    "withinTol" ]

/-- Protected final two name components (case-sensitive) forbidden outside `trustedModules`. -/
private def protectedFinalTwo : List (String × String) :=
  [ ("Pallas", "pallasCall"), ("Pallas", "pallasExec"),
    ("Pallas", "GridSpec"), ("Pallas", "BlockSpec"),
    ("Pallas", "Body"), ("Pallas", "ArrayDesc"),
    ("Pallas", "DimSemantics"),
    ("KernelSpec", "Correct"),
    ("GridSpec", "WellDefined"), ("GridSpec", "AliasesValid"),
    ("Mem", "agreeOutside"), ("Mem", "bufFootprint"),
    ("Mem", "read"), ("Mem", "write"),
    ("Layout", "read"), ("Layout", "footprint"),
    ("Layout", "rowMajor"), ("Layout", "Injective"),
    ("Layout", "Disjoint") ]

/-- Protected namespace prefixes forbidden for declarations outside `trustedModules`. -/
private def protectedNamespaces : List Name :=
  [ `TpuKernelLean.KernelSpec,
    `TpuKernelLean.Mem,
    `TpuKernelLean.Layout,
    `TpuKernelLean.Tensor,
    `TpuKernelLean.Dtype,
    `TpuKernelLean.Pallas.GridSpec,
    `TpuKernelLean.Pallas.BlockSpec,
    `TpuKernelLean.Pallas.ArrayDesc,
    `TpuKernelLean.Pallas.Body,
    `TpuKernelLean.Pallas.DimSemantics,
    `TpuKernelLean.Pallas.DMA ]

/-- Kernel-defining return-type heads checked when matching a `witness` to a `correct` theorem. -/
private def kernelDefTypeHeads : List Name :=
  [ `TpuKernelLean.Pallas.GridSpec,
    `TpuKernelLean.Pallas.Body,
    `TpuKernelLean.Pallas.RowKernel,
    `TpuKernelLean.Pallas.ReductionKernel ]

/-- Check whether `root` is the outermost component of `name`. -/
private def hasRoot (root : Name) : Name → Bool
  | .anonymous => false
  | .str .anonymous s => root == .str .anonymous s
  | .str p _ => hasRoot root p
  | .num p _ => hasRoot root p

/-- Check whether `prefixName` is a proper or equal namespace prefix of `name`. -/
private def hasNamePrefix (prefixName : Name) (name : Name) : Bool :=
  prefixName.isPrefixOf name

/-- True iff every string component of `name` consists solely of ASCII characters (`< 128`). -/
private def isAsciiName : Name → Bool
  | Name.anonymous => true
  | Name.str p s => s.all (fun (c : Char) => c.toNat < 128) && isAsciiName p
  | Name.num p _ => isAsciiName p

/-- User-visible name of a constant (stripping `_private.<mod>.<idx>.` when private). -/
private def userNameOf (name : Name) : Name :=
  if isPrivateName name then
    (privateToUserName? name).getD name
  else
    name

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

/-- Simple substring test without extra dependencies. -/
private def containsSubstr (haystack needle : String) : Bool :=
  (haystack.splitOn needle).length > 1

/-- Check whether `m` is a forbidden direct import (`Lean.*` or `Lake.*`). -/
private def isForbiddenImport (m : Name) : Bool :=
  hasRoot `Lean m || hasRoot `Lake m

/-- Check whether `m` is an allowed direct import of a trusted model module. -/
private def isAllowedTrustedImport (m : Name) : Bool :=
  trustedModules.contains m ||
  let s := m.toString
  s == "Mathlib" || s.startsWith "Mathlib." ||
  s == "Init" || s.startsWith "Init." ||
  s == "Std" || s.startsWith "Std." ||
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

/-- Belt-and-suspenders check on loaded environment: no module under `auditRoot` may directly import `Lean.*` or `Lake.*`. -/
private def checkEnvForbiddenImports (env : Environment) (auditRoot : Name) : Array String := Id.run do
  let mut errs : Array String := #[]
  for i in [:env.header.moduleNames.size] do
    let modName := env.header.moduleNames[i]!
    if hasRoot auditRoot modName then
      for imp in env.header.moduleData[i]!.imports do
        if isForbiddenImport imp.module then
          errs := errs.push s!"module `{modName}` has forbidden import `{imp.module}`"
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

/-- Collect all `.lean` files for `targetRoot` under `base` (sorted deterministically). -/
private def collectLeanFiles (base : System.FilePath) (targetRoot : Name) : IO (Array System.FilePath) := do
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
  return leanFiles.qsort fun a b => a.toString < b.toString

/-- Extract the first `.ident` name inside a syntax subtree. -/
private partial def firstIdent? : Syntax → Option Name
  | .ident _ _ val _ => some val
  | .node _ _ args => args.findSome? firstIdent?
  | _ => none

/-- Extract all `.ident` names directly inside a syntax node (or one level of null-node list). -/
private def collectDirectIdents (stx : Syntax) : Array Name := Id.run do
  let mut out : Array Name := #[]
  match stx with
  | .ident _ _ val _ => out := out.push val
  | .node _ _ args =>
    for a in args do
      match a with
      | .ident _ _ val _ => out := out.push val
      | _ => pure ()
  | _ => pure ()
  return out

/-- Return an error label if `kind` is a forbidden metaprogramming or notation syntax node kind. -/
private def forbiddenCommandKind? (kind : SyntaxNodeKind) : Option String :=
  if kind == ``Lean.Parser.Command.example then
    some "forbidden `example` declaration (use a named `theorem`)"
  else if kind == ``Lean.Parser.Command.elab then
    some "forbidden `elab` command"
  else if kind == ``Lean.Parser.Command.elab_rules then
    some "forbidden `elab_rules` command"
  else if kind == ``Lean.Parser.Command.macro then
    some "forbidden `macro` command"
  else if kind == ``Lean.Parser.Command.macro_rules then
    some "forbidden `macro_rules` command"
  else if kind == ``Lean.Parser.Command.syntax then
    some "forbidden `syntax` command"
  else if kind == ``Lean.Parser.Command.syntaxCat then
    some "forbidden `declare_syntax_cat` command"
  else if kind == ``Lean.Parser.Command.notation then
    some "forbidden `notation` command"
  else if kind == ``Lean.Parser.Command.mixfix then
    some "forbidden `infix`/`prefix`/`postfix` notation command"
  else if kind == ``Lean.Parser.Command.initialize then
    some "forbidden `initialize`/`builtin_initialize` command"
  else if kind == ``Lean.Parser.Command.eval then
    some "forbidden `#eval` command"
  else
    let ks := kind.toString.toLower
    if containsSubstr ks "run_cmd" || containsSubstr ks "runcmd" then
      some s!"forbidden `run_cmd` command ({kind})"
    else if containsSubstr ks "run_elab" || containsSubstr ks "runelab" then
      some s!"forbidden `run_elab` command ({kind})"
    else if containsSubstr ks "run_meta" || containsSubstr ks "runmeta" then
      some s!"forbidden `run_meta` command ({kind})"
    else if containsSubstr ks "builtin_initialize" then
      some s!"forbidden `builtin_initialize` command ({kind})"
    else if containsSubstr ks "declare_syntax_cat" then
      some s!"forbidden `declare_syntax_cat` command ({kind})"
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
    if let some reason := forbiddenCommandKind? kind then
      out := out.push s!"{file}:{lineOf stx}: {reason}"
    let isSetOptionKind :=
      kind == ``Lean.Parser.Command.set_option ||
      match kind with
      | Name.str _ s => s == "set_option" || s == "setOption"
      | _ => false
    if isSetOptionKind then
      match (args[1]?.bind firstIdent?).orElse (fun () => firstIdent? stx) with
      | some optName =>
        if !allowedOptions.contains optName then
          out := out.push s!"{file}:{lineOf stx}: forbidden `set_option` `{optName}` (not in allowlist)"
      | none =>
        out := out.push s!"{file}:{lineOf stx}: forbidden `set_option` (not in allowlist)"
    -- Check declaration names (not binders/local variables) for non-ASCII characters.
    if kind == ``Lean.Parser.Command.declId then
      if let some (Syntax.ident _ _ val _) := args[0]? then
        if !isAsciiName val then
          out := out.push s!"{file}:{lineOf stx}: non-ASCII character in declaration name `{val}`"
    else if kind == ``Lean.Parser.Command.ctor then
      for a in args do
        match a with
        | Syntax.ident _ _ val _ =>
          if !isAsciiName val then
            out := out.push s!"{file}:{lineOf stx}: non-ASCII character in constructor name `{val}`"
          break
        | _ => pure ()
    else if kind == ``Lean.Parser.Command.structSimpleBinder then
      if let some (Syntax.ident _ _ val _) := args[0]? then
        if !isAsciiName val then
          out := out.push s!"{file}:{lineOf stx}: non-ASCII character in field name `{val}`"
    else if kind == ``Lean.Parser.Command.structExplicitBinder then
      if let some namesStx := args[1]? then
        for val in collectDirectIdents namesStx do
          if !isAsciiName val then
            out := out.push s!"{file}:{lineOf stx}: non-ASCII character in field name `{val}`"
    else if kind == ``Lean.Parser.Command.namespace then
      if let some (Syntax.ident _ _ val _) := args[1]? then
        if !isAsciiName val then
          out := out.push s!"{file}:{lineOf stx}: non-ASCII character in namespace name `{val}`"
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

/-- Parse a `.lean` file with Lean's own parser (using a trusted parser env) and scan its header and commands. -/
private def scanFileSyntax (parseEnv : Environment) (p : System.FilePath) : IO (Array String) := do
  let src ← IO.FS.readFile p
  let inputCtx := Parser.mkInputContext src p.toString
  let (header, parserState, messages) ← Parser.parseHeader inputCtx
  let pmctx : Parser.ParserModuleContext := { env := parseEnv, options := {} }
  let mut st := parserState
  let mut msgs := messages
  let mut failures : Array String := #[]
  if msgs.hasErrors then
    failures := failures.push s!"{p}: failed to parse module header"
    return failures
  -- Check module imports before scanning commands.
  for imp in Elab.headerToImports header do
    if isForbiddenImport imp.module then
      failures := failures.push
        s!"{p}: forbidden import `{imp.module}` (importing Lean/Lake internals is forbidden)"
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

/-- Phase 1: scan all `.lean` source files under `<base>/<targetRoot>` using ONLY `parseEnv`. -/
private def runScanOnly (parseEnv : Environment) (base : System.FilePath) (targetRoot : Name) :
    IO (UInt32 × String) := do
  let leanFiles ← collectLeanFiles base targetRoot
  if leanFiles.isEmpty then
    return (1, s!"Source scan FAILED: no `.lean` files found for `{targetRoot}` under `{base}`")
  let mut failures : Array String := #[]
  for p in leanFiles do
    failures := failures ++ (← scanFileSyntax parseEnv p)
  if !failures.isEmpty then
    let report := String.intercalate "\n  " (Array.toList failures)
    return (1, s!"Source scan FAILED ({failures.size} issue(s)):\n  {report}")
  return (0, s!"Source scan passed: scanned {leanFiles.size} source files for {targetRoot}.")

/-- Verify that every `.lean` file under `<base>/<auditRoot>` is imported by `env`. -/
private def checkAllFilesImported (env : Environment) (base : System.FilePath) (auditRoot targetRoot : Name) :
    IO (Nat × Array String) := do
  let leanFiles ← collectLeanFiles base auditRoot
  if leanFiles.isEmpty then
    return (0, #[s!"no `.lean` files found for `{auditRoot}` under `{base}`"])
  let importedMods := env.header.moduleNames
  let mut failures : Array String := #[]
  for p in leanFiles do
    let comps := relComponents base p
    match pathToModuleName comps with
    | some modName =>
      if !importedMods.contains modName then
        failures := failures.push s!"{p}: module `{modName}` is not imported by `{targetRoot}.lean`"
    | none => pure ()
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

/-- Extract the last string component and last two string components of `name`. -/
private def lastTwoStrComponents (name : Name) : Option String × Option (String × String) :=
  match name with
  | .str (.str _ p) s => (some s, some (p, s))
  | .str _ s => (some s, none)
  | _ => (none, none)

/-- True iff `m` is `TpuKernelLean.Pallas` or a submodule `TpuKernelLean.Pallas.*`. -/
private def isPallasModule (m : Name) : Bool :=
  m == `TpuKernelLean.Pallas || m.toString.startsWith "TpuKernelLean.Pallas."

/-- True if `s` is a Lean compiler-synthesized equational or matcher-auxiliary name suffix. -/
private def isEqnOrSplitterComp (s : String) : Bool :=
  s == "eq_def" || s == "eq_unfold" || s == "splitter" ||
  s.startsWith "eq_" || s.startsWith "congr_eq_" || s.startsWith "_sparseCasesOn_"

/-- True for Lean's lazily-generated equational lemmas and matcher auxiliaries (`.eq_*`, `.eq_def`,
`.eq_unfold`, `.splitter*`, `.congr_eq_*`, `._sparseCasesOn_*`) synthesized in a downstream module
for an existing constant already declared in a trusted or `TpuKernelLean.Pallas.*` module. -/
private def isUpstreamEqnOrSplitterAux (env : Environment) (name : Name) : Bool :=
  let rec go (cur : Name) : Bool :=
    match cur with
    | .str parent s =>
      if isEqnOrSplitterComp s then
        let parentIsUpstream :=
          match (env.getModuleIdxFor? parent).bind fun idx => env.header.moduleNames[idx]? with
          | some m => trustedModules.contains m || isPallasModule m
          | none => false
        parentIsUpstream || go parent
      else
        false
    | _ => false
  go (userNameOf name)

/-- Check protected names and namespaces for a single declaration outside `trustedModules`. -/
private def checkProtectedName (env : Environment) (name : Name) : Array String := Id.run do
  let declMod? := (env.getModuleIdxFor? name).bind fun idx => env.header.moduleNames[idx]?
  match declMod? with
  | some m =>
    if trustedModules.contains m then return #[]
  | none => pure ()
  let u := userNameOf name
  let mut errs : Array String := #[]
  let (last?, lastTwo?) := lastTwoStrComponents u
  if let some s := last? then
    if protectedFinalComponents.contains s then
      errs := errs.push s!"{displayName name}: protected name component `{s}` is reserved for trusted model modules"
  if let some (p, s) := lastTwo? then
    if protectedFinalTwo.contains (p, s) then
      errs := errs.push s!"{displayName name}: protected name `{p}.{s}` is reserved for trusted model modules"
  if !isUpstreamEqnOrSplitterAux env name then
    for ns in protectedNamespaces do
      if hasNamePrefix ns u then
        errs := errs.push s!"{displayName name}: declaring inside protected namespace `{ns}` is forbidden outside trusted model modules"
    -- Also forbid any module outside `TpuKernelLean.Pallas.*` from declaring inside `TpuKernelLean.Pallas`.
    let inPallasMod := declMod?.any isPallasModule
    if !inPallasMod && hasNamePrefix `TpuKernelLean.Pallas u then
      errs := errs.push s!"{displayName name}: declaring inside namespace `TpuKernelLean.Pallas` is forbidden outside `TpuKernelLean.Pallas.*`"
  return errs

/-- Collect all `indexMap` expressions passed to `BlockSpec.mk` or `ReductionKernel.mk` within `e`. -/
private partial def collectIndexMapExprs (e : Expr) (acc : Array Expr := #[]) : Array Expr := Id.run do
  let mut out := acc
  match e with
  | .app .. =>
    out := e.withApp fun fn args => Id.run do
      let mut res := out
      if fn.isConstOf `TpuKernelLean.Pallas.BlockSpec.mk then
        if h : 1 < args.size then
          res := res.push args[1]
      else if fn.isConstOf `TpuKernelLean.Pallas.ReductionKernel.mk then
        if h : 8 < args.size then
          res := res.push args[8]
      res := collectIndexMapExprs fn res
      for a in args do
        res := collectIndexMapExprs a res
      return res
  | .lam _ t b _ =>
    out := collectIndexMapExprs t out
    out := collectIndexMapExprs b out
  | .forallE _ t b _ =>
    out := collectIndexMapExprs t out
    out := collectIndexMapExprs b out
  | .letE _ t v b _ =>
    out := collectIndexMapExprs t out
    out := collectIndexMapExprs v out
    out := collectIndexMapExprs b out
  | .mdata _ b =>
    out := collectIndexMapExprs b out
  | .proj _ _ b =>
    out := collectIndexMapExprs b out
  | _ => pure ()
  return out

/-- True iff `e` denotes the `ℕ` type. -/
private def isNatType (e : Expr) : Bool :=
  e.isConstOf ``Nat

/-- Search an `indexMap` expression (following non-theorem `TpuKernelLean` helper defs) for
truncating natural-number subtraction or predecessor (`Nat.sub`, `Nat.pred`, `HSub.hSub`/`Sub.sub` at `ℕ`). -/
private partial def findNatSubInExpr (env : Environment) (e : Expr)
    (visited : NameSet := {}) : Option String × NameSet := Id.run do
  let mut vis := visited
  match e with
  | .const c _ =>
    if c == ``Nat.sub then return (some "Nat.sub", vis)
    if c == ``Nat.pred then return (some "Nat.pred", vis)
    if !vis.contains c then
      let inTkl := match env.getModuleIdxFor? c with
        | some idx => (env.header.moduleNames[idx]?).any (hasRoot `TpuKernelLean)
        | none => false
      if inTkl then
        vis := vis.insert c
        if let some (.defnInfo val) := env.find? c then
          let (hit?, vis') := findNatSubInExpr env val.value vis
          vis := vis'
          if hit?.isSome then return (hit?, vis)
    return (none, vis)
  | .app .. =>
    let fn := e.getAppFn
    let args := e.getAppArgs
    if fn.isConstOf ``Nat.sub then
      return (some "Nat.sub", vis)
    if fn.isConstOf ``Nat.pred then
      return (some "Nat.pred", vis)
    if fn.isConstOf ``HSub.hSub then
      if (args[0]?.any isNatType) || (args[1]?.any isNatType) || (args[2]?.any isNatType) then
        return (some "HSub.hSub at ℕ", vis)
    if fn.isConstOf ``Sub.sub then
      if args[0]?.any isNatType then
        return (some "Sub.sub at ℕ", vis)
    let (hitFn?, vis') := findNatSubInExpr env fn vis
    vis := vis'
    if hitFn?.isSome then return (hitFn?, vis)
    for a in args do
      let (hitA?, vis'') := findNatSubInExpr env a vis
      vis := vis''
      if hitA?.isSome then return (hitA?, vis)
    return (none, vis)
  | .lam _ _ b _ =>
    findNatSubInExpr env b vis
  | .forallE _ _ b _ =>
    findNatSubInExpr env b vis
  | .letE _ _ v b _ =>
    let (hitV?, vis') := findNatSubInExpr env v vis
    if hitV?.isSome then return (hitV?, vis')
    findNatSubInExpr env b vis'
  | .mdata _ b =>
    findNatSubInExpr env b vis
  | .proj _ _ b =>
    findNatSubInExpr env b vis
  | _ => return (none, vis)

/-- Audit all non-theorem declarations for `Nat` subtraction inside `BlockSpec.indexMap` or `ReductionKernel.outIndexMap`. -/
private def checkIndexMaps (env : Environment) (targets : Array (Name × ConstantInfo)) : Array String := Id.run do
  let mut errs : Array String := #[]
  for (name, cinfo) in targets do
    match cinfo with
    | .defnInfo val =>
      let idxMaps := collectIndexMapExprs val.value
      for im in idxMaps do
        let (hit?, _) := findNatSubInExpr env im
        if let some detail := hit? then
          errs := errs.push
            s!"{displayName name}: indexMap uses truncating natural-number subtraction ({detail}); use signed ℤ arithmetic"
          break
    | _ => pure ()
  return errs

/-- Collect all constant names occurring directly in `e`. -/
private def exprConsts (e : Expr) : NameSet :=
  e.foldConsts {} fun c s => s.insert c

/-- Compute all constants reachable from `e`, unfolding `.defnInfo` bodies declared in `allowedMods`. -/
private def reachableConsts (env : Environment) (allowedMods : List Name) (e : Expr) : NameSet := Id.run do
  let mut visited : NameSet := {}
  let mut worklist : List Name := (exprConsts e).toList
  while !worklist.isEmpty do
    match worklist with
    | [] => break
    | c :: rest =>
      worklist := rest
      if visited.contains c then continue
      visited := visited.insert c
      let inAllowed := match env.getModuleIdxFor? c with
        | some idx => (env.header.moduleNames[idx]?).any (allowedMods.contains ·)
        | none => false
      if inAllowed then
        match env.find? c with
        | some (.defnInfo val) =>
          for c' in (exprConsts val.value).toList do
            if !visited.contains c' then
              worklist := c' :: worklist
        | _ => pure ()
  return visited

/-- Strip leading `forallE` and `letE` binders from a proposition type. -/
private partial def stripForallsAndLets : Expr → Expr
  | .forallE _ _ b _ => stripForallsAndLets b
  | .letE _ _ _ b _ => stripForallsAndLets b
  | .mdata _ b => stripForallsAndLets b
  | e => e

/-- Strip leading `lam` binders from a definition body. -/
private partial def stripLams : Expr → Expr
  | .lam _ _ b _ => stripLams b
  | .mdata _ b => stripLams b
  | e => e

/-- True if `e` is trivially `True` (directly, under universal quantifiers, or via unfolding
a predicate defined in `allowedMods` whose body is `True`). -/
private partial def isTriviallyTrue (env : Environment) (allowedMods : List Name) (e : Expr)
    (fuel : Nat := 16) : Bool :=
  if fuel == 0 then false
  else
    let body := stripForallsAndLets e
    if body.isConstOf ``True then true
    else
      match body.getAppFn with
      | .const c _ =>
        let inAllowed := match env.getModuleIdxFor? c with
          | some idx => (env.header.moduleNames[idx]?).any (allowedMods.contains ·)
          | none => false
        if inAllowed then
          match env.find? c with
          | some (.defnInfo val) => isTriviallyTrue env allowedMods (stripLams val.value) (fuel - 1)
          | _ => false
        else false
      | _ => false

/-- Return type head constant of a (possibly function-typed) definition. -/
private partial def returnTypeHead? : Expr → Option Name
  | .forallE _ _ b _ => returnTypeHead? b
  | .letE _ _ _ b _ => returnTypeHead? b
  | .mdata _ b => returnTypeHead? b
  | e =>
    match e.getAppFn with
    | .const c _ => some c
    | _ => none

/-- True iff `c` is a non-theorem kernel-defining definition declared in `kernelMod`
(return type head is `GridSpec`, `Body`, `RowKernel`, or `ReductionKernel`). -/
private def isKernelDefInModule (env : Environment) (kernelMod : Name) (c : Name) : Bool :=
  let inMod := match env.getModuleIdxFor? c with
    | some idx => env.header.moduleNames[idx]? == some kernelMod
    | none => false
  if !inMod then false
  else
    match env.find? c with
    | some (.defnInfo val) =>
      match returnTypeHead? val.type with
      | some head => kernelDefTypeHeads.contains head
      | none => false
    | _ => false

/-- Parse a JSON array of strings from field `fieldName` of `obj`. -/
private def getStringArrayField (ctx : String) (obj : Json) (fieldName : String) :
    Except String (Array String) := do
  let v ← obj.getObjVal? fieldName |>.mapError fun _ => s!"{ctx}: missing field `{fieldName}`"
  let arr ← v.getArr? |>.mapError fun _ => s!"{ctx}: field `{fieldName}` must be an array"
  arr.mapM fun item =>
    item.getStr? |>.mapError fun _ => s!"{ctx}: elements of `{fieldName}` must be strings"

/-- Audit every entry in `manifest.json` against the compiled Lean environment. -/
private def checkManifestRegistry (env : Environment) (manifestPath : System.FilePath) :
    IO (Array String) := do
  if !(← manifestPath.pathExists) then
    return #[s!"manifest file not found: {manifestPath}"]
  let raw ← IO.FS.readFile manifestPath
  let json ← match Json.parse raw with
    | .ok j => pure j
    | .error err => return #[s!"failed to parse {manifestPath}: {err}"]
  let kernelsObj ← match json.getObjVal? "kernels" with
    | .ok (.obj kvs) => pure kvs.toList
    | _ => return #[s!"{manifestPath}: `kernels` must be a JSON object"]
  let mut errs : Array String := #[]
  let pallasCallName : Name := `TpuKernelLean.Pallas.pallasCall
  let sharedFollowMods : List Name :=
    [ `TpuKernelLean.Pallas.RowKernel,
      `TpuKernelLean.Pallas.ReductionKernel,
      `TpuKernelLean.Spec ]

  for (key, entry) in kernelsObj do
    let ctx := s!"manifest[{key}]"
    let leanModStr ← match (entry.getObjVal? "lean_module").bind Json.getStr? with
      | .ok s => pure s
      | .error _ => errs := errs.push s!"{ctx}: missing string field `lean_module`"; continue
    let level ← match (entry.getObjVal? "level").bind Json.getStr? with
      | .ok s => pure s
      | .error _ => errs := errs.push s!"{ctx}: missing string field `level`"; continue
    let correctArr ← match getStringArrayField ctx entry "correct" with
      | .ok a => pure a
      | .error e => errs := errs.push e; continue
    let otherArr ← match getStringArrayField ctx entry "other" with
      | .ok a => pure a
      | .error e => errs := errs.push e; continue
    let witnessOpt : Option String ← match entry.getObjVal? "witness" with
      | .ok .null => pure none
      | .ok (.str s) => pure (some s)
      | .ok _ => errs := errs.push s!"{ctx}: `witness` must be a string or null"; continue
      | .error _ => errs := errs.push s!"{ctx}: missing field `witness`"; continue

    let leanMod := leanModStr.toName
    let some expectedModIdx := env.getModuleIdx? leanMod
      | errs := errs.push s!"{ctx}: module `{leanMod}` is not loaded in environment"; continue

    if level == "pallas_call" then
      if correctArr.isEmpty then
        errs := errs.push s!"{ctx}: `pallas_call` entry must list at least one theorem in `correct`"
      if witnessOpt.isNone then
        errs := errs.push s!"{ctx}: `pallas_call` entry must specify a non-null `witness` theorem"
    else if level == "algorithm_only" then
      if correctArr.isEmpty && otherArr.isEmpty then
        errs := errs.push s!"{ctx}: `algorithm_only` entry must list at least one theorem"

    let resolveInMod (thm : String) : Name :=
      if thm.startsWith (leanModStr ++ ".") then thm.toName else leanMod ++ thm.toName

    let followMods := leanMod :: sharedFollowMods
    let mut correctKernelDefs : NameSet := {}

    -- Check `correct` theorems.
    for thm in correctArr do
      let fullName := resolveInMod thm
      if isPrivateName fullName then
        errs := errs.push s!"{ctx}: `correct` theorem `{thm}` must not be private"
        continue
      match env.find? fullName with
      | none =>
        errs := errs.push s!"{ctx}: `correct` theorem `{fullName}` not found in compiled environment"
      | some cinfo =>
        if !(cinfo matches .thmInfo _) then
          errs := errs.push s!"{ctx}: `{fullName}` must be a `theorem`"
          continue
        if env.getModuleIdxFor? fullName != some expectedModIdx then
          errs := errs.push s!"{ctx}: `correct` theorem `{fullName}` is not declared in `{leanMod}`"
          continue
        if isTriviallyTrue env followMods cinfo.type then
          errs := errs.push s!"{ctx}: `correct` theorem `{fullName}` has a trivially `True` statement"
          continue
        let reach := reachableConsts env followMods cinfo.type
        if !reach.contains pallasCallName then
          errs := errs.push s!"{ctx}: `correct` theorem `{fullName}` does not reach `{pallasCallName}`"
        for c in reach.toList do
          if isKernelDefInModule env leanMod c then
            correctKernelDefs := correctKernelDefs.insert c

    -- Check `other` theorems.
    for thm in otherArr do
      let fullName := resolveInMod thm
      if isPrivateName fullName then
        errs := errs.push s!"{ctx}: `other` theorem `{thm}` must not be private"
        continue
      match env.find? fullName with
      | none =>
        errs := errs.push s!"{ctx}: `other` theorem `{fullName}` not found in compiled environment"
      | some cinfo =>
        if !(cinfo matches .thmInfo _) then
          errs := errs.push s!"{ctx}: `{fullName}` must be a `theorem`"
          continue
        if env.getModuleIdxFor? fullName != some expectedModIdx then
          errs := errs.push s!"{ctx}: `other` theorem `{fullName}` is not declared in `{leanMod}`"
          continue
        if isTriviallyTrue env followMods cinfo.type then
          errs := errs.push s!"{ctx}: `other` theorem `{fullName}` has a trivially `True` statement"

    -- Check `witness` theorem.
    if let some wStr := witnessOpt then
      let wName :=
        if (env.find? wStr.toName).isSome then wStr.toName
        else `TpuKernelLean.Tests.EndToEnd ++ wStr.toName
      if isPrivateName wName then
        errs := errs.push s!"{ctx}: witness theorem `{wStr}` must not be private"
      else
        match env.find? wName with
        | none =>
          errs := errs.push s!"{ctx}: witness theorem `{wStr}` not found in compiled environment"
        | some cinfo =>
          if !(cinfo matches .thmInfo _) then
            errs := errs.push s!"{ctx}: witness `{wName}` must be a `theorem`"
          else
            let wMod? := (env.getModuleIdxFor? wName).bind fun idx => env.header.moduleNames[idx]?
            match wMod? with
            | none =>
              errs := errs.push s!"{ctx}: witness `{wName}` has no declaring module"
            | some wMod =>
              if !wMod.toString.startsWith "TpuKernelLean.Tests." then
                errs := errs.push s!"{ctx}: witness `{wName}` must be declared in a `TpuKernelLean.Tests.*` module (found `{wMod}`)"
              let wFollowMods := wMod :: followMods
              if isTriviallyTrue env wFollowMods cinfo.type then
                errs := errs.push s!"{ctx}: witness `{wName}` has a trivially `True` statement"
              else
                let wReach := reachableConsts env wFollowMods cinfo.type
                if !wReach.contains pallasCallName then
                  errs := errs.push s!"{ctx}: witness `{wName}` does not reach `{pallasCallName}`"
                let shared := wReach.toList.any fun c =>
                  isKernelDefInModule env leanMod c && correctKernelDefs.contains c
                if !correctArr.isEmpty && !shared then
                  errs := errs.push
                    s!"{ctx}: witness `{wName}` does not share any `GridSpec`/`Body`/`RowKernel`/`ReductionKernel` definition from `{leanMod}` with a `correct` theorem"
  return errs

/-- Phase 2: audit every target declaration's safety/axioms, imports, protected names, index maps, and manifest registry. -/
private def checkEnv (base : System.FilePath) (targetRoot auditRoot : Name)
    (manifestOpt : Option System.FilePath) : CoreM (UInt32 × String) := do
  let env ← getEnv
  let mut failures : Array String := #[]
  let rawTargets := env.constants.fold (init := (#[] : Array (Name × ConstantInfo))) fun acc name cinfo =>
    if isTargetConstant env auditRoot name then acc.push (name, cinfo) else acc
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
    let u := userNameOf name
    if !isAsciiName u then
      failures := failures.push s!"{displayName name}: non-ASCII character in declaration name"
    let axioms ← Lean.collectAxioms name
    let bad := axioms.filter fun ax => !standardAxioms.contains ax
    if !bad.isEmpty then
      let axList := String.intercalate ", " ((Array.toList bad).map Name.toString)
      failures := failures.push s!"{displayName name}: disallowed axiom(s): {axList}"
    if auditRoot == `TpuKernelLean then
      failures := failures ++ checkProtectedName env name
  failures := failures ++ checkEnvForbiddenImports env auditRoot
  if auditRoot == `TpuKernelLean then
    failures := failures ++ checkTrustedImports env
    failures := failures ++ checkIndexMaps env targets
  let effectiveManifest : Option System.FilePath :=
    match manifestOpt with
    | some p => some p
    | none => if auditRoot == `TpuKernelLean then some (base / "manifest.json") else none
  if let some mp := effectiveManifest then
    failures := failures ++ (← checkManifestRegistry env mp)
  let (fileCount, importFailures) ←
    if auditRoot == targetRoot then
      checkAllFilesImported env base auditRoot targetRoot
    else
      pure (0, #[])
  failures := failures ++ importFailures
  if !failures.isEmpty then
    let report := String.intercalate "\n  " (Array.toList failures)
    return (1, s!"Axiom gate FAILED ({failures.size} issue(s)):\n  {report}")
  return (0, s!"Axiom gate passed: checked {targets.size} declarations ({thmCount} theorems) in {auditRoot} (checked {fileCount} source imports).")

private structure Config where
  base : System.FilePath := "."
  root : Name := `TpuKernelLean
  auditRoot? : Option Name := none
  parseRoot : Name := `Mathlib
  manifest? : Option System.FilePath := none
  scanOnly : Bool := false
  checkEnvOnly : Bool := false

private def parseArgs : List String → Config → Config
  | "--base" :: b :: rest, cfg => parseArgs rest { cfg with base := b }
  | "--root" :: r :: rest, cfg => parseArgs rest { cfg with root := r.toName }
  | "--audit-root" :: r :: rest, cfg => parseArgs rest { cfg with auditRoot? := some r.toName }
  | "--parse-root" :: r :: rest, cfg => parseArgs rest { cfg with parseRoot := r.toName }
  | "--manifest" :: m :: rest, cfg => parseArgs rest { cfg with manifest? := some m }
  | "--scan-only" :: rest, cfg => parseArgs rest { cfg with scanOnly := true }
  | "--check-env" :: rest, cfg => parseArgs rest { cfg with checkEnvOnly := true }
  | _ :: rest, cfg => parseArgs rest cfg
  | [], cfg => cfg

end CheckAxioms

unsafe def main (args : List String) : IO UInt32 := do
  let cfg := CheckAxioms.parseArgs args {}
  let auditRoot := cfg.auditRoot?.getD cfg.root
  Lean.initSearchPath (← Lean.findSysroot)

  if cfg.scanOnly then
    -- Phase 1: load ONLY the trusted parser environment (never the untrusted target library)
    -- so library initializers cannot execute before source scanning succeeds.
    Lean.enableInitializersExecution
    let parseEnv ← Lean.importModules #[{ module := cfg.parseRoot }] {} (trustLevel := 0) (loadExts := true)
    let (code, msg) ← CheckAxioms.runScanOnly parseEnv cfg.base cfg.root
    if code == 0 then IO.println msg else IO.eprintln msg
    return code

  if !cfg.checkEnvOnly then
    -- Default mode: run Phase 1 (`--scan-only`) in a separate child process BEFORE loading the
    -- target library with initializers enabled. Peak memory stays at one environment per process.
    let childArgs : Array String :=
      #[ "--run", "scripts/CheckAxioms.lean", "--scan-only",
         "--base", cfg.base.toString,
         "--root", cfg.root.toString,
         "--parse-root", cfg.parseRoot.toString ]
    let res ← IO.Process.output { cmd := "lean", args := childArgs }
    if !res.stdout.isEmpty then IO.print res.stdout
    if !res.stderr.isEmpty then IO.eprint res.stderr
    if res.exitCode != 0 then
      return res.exitCode

  -- Phase 2: source scan has already passed; now load the target library and audit declarations.
  Lean.enableInitializersExecution
  let env ← Lean.importModules #[{ module := cfg.root }] {} (trustLevel := 0) (loadExts := true)
  let ctx : Lean.Core.Context := { fileName := "<CheckAxioms>", fileMap := default }
  let ((code, msg), _) ←
    (CheckAxioms.checkEnv cfg.base cfg.root auditRoot cfg.manifest?).toIO ctx { env := env }
  if code == 0 then IO.println msg else IO.eprintln msg
  return code
