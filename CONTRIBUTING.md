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
- **Named theorems only:** write tests as named `theorem`s; `scripts/CheckAxioms.lean` rejects `example` declarations. Every `.lean` file under `TpuKernelLean/` must be imported by `TpuKernelLean.lean`.
- **Pallas model ledger:** every semantic rule in `TpuKernelLean/Pallas/Core.lean` (and `Parallel.lean`, `DMA.lean`) must carry a `PAL-` id with a matching heading in `docs/pallas-model-ledger.md` (`scripts/check-ledger.sh`).
- **Provenance and decisions:** every kernel under `TpuKernelLean/Examples/` must have an entry in `manifest.json` (`scripts/check-manifest.sh --offline`), its modelling choices recorded in `DECISIONS.md`, and a row in the *What is verified* table in `README.md` whose theorem names pass `scripts/check-readme.sh`.
- **Non-public sources:** never add a repository name, file path, URL, commit, or internal function name for a kernel whose source is not public. Use `"source": "unreferenced"` in `manifest.json` with no `repository`, `path`, or `commit` field. Only public sources (`sgl-project/sglang-torchtpu`, `jax-ml/jax`) may be named.

## Adding a kernel

Follow the step-by-step workflow in the **Adding a kernel** section of [README.md](README.md#adding-a-kernel).
