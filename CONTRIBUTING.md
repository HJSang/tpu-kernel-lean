# Contributing

## Build and checks

Build on a Linux machine (not a laptop; see `README.md` for toolchain setup):

```bash
lake exe cache get
lake build
bash scripts/ci.sh
```

If an end-to-end differential or hardware script `scripts/e2e.sh` is present, run:

```bash
bash scripts/e2e.sh
```

## Rules for changes

- **Zero warnings and clean Lean:** `lakefile.toml` sets `warningAsError = true`, so `lake build` fails on any Lean compiler or linter warning. `autoImplicit` is off. Do not use `sorry`, `admit`, `native_decide`, new `axiom`s, `unsafe`/`partial`, `opaque`, `@[extern]`, or `@[implemented_by]`.
- **No metaprogramming, notation, or non-ASCII declaration names:** `scripts/CheckAxioms.lean` rejects `import Lean*`/`Lake*`, `elab`/`elab_rules`, `macro`/`macro_rules`, `syntax`/`declare_syntax_cat`, `notation`/`infix*`/`prefix`/`postfix` (including `local`/`scoped`), `run_cmd`/`run_elab`/`run_meta`, `initialize`/`builtin_initialize`, `#eval`, `set_option` outside the allowlist, and non-ASCII characters in declaration names (binders and local variables such as `ε` remain fine).
- **Named theorems only:** write tests as named `theorem`s; `scripts/CheckAxioms.lean` rejects `example` declarations. Every `.lean` file under `TpuKernelLean/` must be imported by `TpuKernelLean.lean`.
- **Signed integer arithmetic in index maps:** `BlockSpec.indexMap` and `ReductionKernel.outIndexMap` must not use truncating `ℕ` subtraction or `Nat.pred`; cast to `ℤ` before subtracting so negative block indices are caught by `GridSpec.WellDefined`.
- **Pallas model ledger:** every semantic rule in `TpuKernelLean/Pallas/Core.lean` (and `Parallel.lean`, `DMA.lean`) must carry a `PAL-` id with a matching heading in `docs/pallas-model-ledger.md` (`scripts/check-ledger.sh`).
- **Provenance, theorem registry, and decisions:** every kernel under `TpuKernelLean/Examples/` (recursively) must have an entry in `manifest.json` listing its `correct`, `other`, and `witness` theorems (`scripts/check-manifest.sh --offline`), a matching `e2e/test_<key>.py` test if it is a public `pallas_call` kernel, its modelling choices recorded in `DECISIONS.md`, and a row in the *What is verified* table in `README.md` that passes `scripts/check-readme.sh`.
- **Non-public sources:** never add a repository name, file path, URL, commit, or internal function name for a kernel whose source is not public. `"source": "unreferenced"` is gated by an explicit allowlist in `scripts/check-manifest.sh` and must omit `repository`, `path`, and `commit`. Only public sources (`sgl-project/sglang-torchtpu`, `jax-ml/jax`) may be named.

## What the checks cannot catch

Automated checks enforce kernel soundness, syntax/metaprogramming bans, protected names, `pallasCall` reachability, index-map arithmetic, and manifest/README consistency, but they cannot prove that a specification means what the author claims. Reviewers must read every new kernel's spec and Python source directly for the following residual attack classes:

- **Unsatisfiable or degenerate preconditions (`requires`):** A theorem can hold vacuously if `requires` contains a disguised contradiction (for example two helper predicates on tile alignment and padding that cannot both be true), or it can pass a non-vacuity witness while collapsing the input domain to a trivial set such as the all-zero tensor where a broken body happens to match. Look closely at every predicate in `requires`, check that helper predicates do not restrict data values beyond genuine domain requirements (such as `0 < eps` or distinct integer page ids), and verify that the end-to-end witness theorem in `TpuKernelLean/Tests/` uses non-zero, asymmetric inputs and asserts the actual expected output values.
- **A specification that mirrors the kernel's bug:** If both the Lean `Body` and the spec's `math` function contain the same mistake (for example transposing `B` in a matmul so both compute `A @ B^T` while the README advertises `A @ B`), the Lean proof is completely sound and passes every gate. Compare `math` against canonical primitives in `TpuKernelLean/Tensor.lean` (`Tensor.matmul`, `Tensor.rmsNorm`, `Tensor.softmax`, `Tensor.ropeNeox`) wherever possible, check concrete asymmetric examples in `Tests/EndToEnd.lean`, and inspect the independent NumPy reference in `e2e/test_<key>.py`.
- **A Lean file that does not model the pinned Python source:** `check-manifest.sh` verifies git blob hashes, unique `(repository, path)` pins, and that `e2e/test_<key>.py` exists, and `CheckAxioms.lean` ties the witness to the Lean kernel's `GridSpec`/`Body`, but no tool checks that the Lean `GridSpec`, `Body`, scalar prefetch, aliases, and `dimension_semantics` actually transcribe the pinned Python `pallas_call`. Read the Python file side by side with the Lean `GridSpec` and `Body`, checking every `BlockSpec` shape and `index_map`, `grid`, `input_output_aliases`, `dimension_semantics`, and `e2e/test_<key>.py` to confirm the e2e test invokes the pinned functions against the proved formula.
- **A statement that mentions the model without saying the right thing about it:** the statement audit only checks that a `correct` theorem reaches the real `pallasCall` and is not literally `True`. A run that calls `pallasCall` and then discards its result, or a conclusion that holds for any memory, still passes. Read each headline statement and check that it has the shape "`pallasCall gridSpec body` returns `some` memory, and that memory has these values at every output element and is unchanged elsewhere".

## Adding a kernel

Follow the step-by-step workflow in the **Adding a kernel** section of [README.md](README.md#adding-a-kernel).
