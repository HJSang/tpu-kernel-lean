"""Shared fixtures and helpers for CPU Pallas TPU-interpret-mode e2e tests."""

from __future__ import annotations

import contextlib
import importlib.util
import os
from pathlib import Path
import sys
import types
from typing import Iterator, Sequence

import jax
from jax.experimental.pallas import tpu as pltpu
import pytest


def _get_interpret_helpers():
    force = getattr(pltpu, "force_tpu_interpret_mode", None)
    reset = getattr(pltpu, "reset_tpu_interpret_mode_state", None)
    if force is None or reset is None:
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call as _interp

        force = force or _interp.force_tpu_interpret_mode
        reset = reset or _interp.reset_tpu_interpret_mode_state
    return force, reset


class _InertOpHandle:
    """Inert placeholder returned by stubbed torch/sglang op registration helpers."""

    def __init__(self, name: str):
        self._name = name

    def register_fake(self, fn):
        return fn

    def register_autograd(self, *args, **kwargs):
        return None

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"Inert stub op {self._name!r} was called at runtime; "
            "e2e tests must call the JAX-level kernel directly."
        )


def _make_inert_torch_stubs() -> dict[str, types.ModuleType]:
    """Create minimal inert `torch` stub modules for importing JAX-level kernels."""
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.dtype = object
    torch_mod.bfloat16 = object()
    torch_mod.float32 = object()
    torch_mod.float16 = object()
    torch_mod.int32 = object()

    autograd_mod = types.ModuleType("torch.autograd")

    class _Function:
        pass

    autograd_mod.Function = _Function
    torch_mod.autograd = autograd_mod

    library_mod = types.ModuleType("torch.library")

    def _inert_custom_op(name, *args, **kwargs):
        def _decorator(fn):
            return _InertOpHandle(name)

        return _decorator

    library_mod.custom_op = _inert_custom_op
    torch_mod.library = library_mod

    nn_mod = types.ModuleType("torch.nn")
    nn_functional_mod = types.ModuleType("torch.nn.functional")
    nn_mod.functional = nn_functional_mod
    torch_mod.nn = nn_mod

    torch_tpu_mod = types.ModuleType("torch_tpu")
    torch_tpu_mod.__path__ = []
    torch_tpu_internal = types.ModuleType("torch_tpu._internal")
    torch_tpu_internal.__path__ = []
    torch_tpu_pallas = types.ModuleType("torch_tpu._internal.pallas")

    def _inert_custom_jax_kernel(fn, *, name="<unknown>", **kwargs):
        return _InertOpHandle(str(name))

    def _inert_jax_op(name, fn, **kwargs):
        return _InertOpHandle(str(name))

    torch_tpu_pallas.custom_jax_kernel = _inert_custom_jax_kernel
    torch_tpu_pallas.jax_op = _inert_jax_op
    torch_tpu_internal.pallas = torch_tpu_pallas
    torch_tpu_mod._internal = torch_tpu_internal

    return {
        "torch": torch_mod,
        "torch.autograd": autograd_mod,
        "torch.library": library_mod,
        "torch.nn": nn_mod,
        "torch.nn.functional": nn_functional_mod,
        "torch_tpu": torch_tpu_mod,
        "torch_tpu._internal": torch_tpu_internal,
        "torch_tpu._internal.pallas": torch_tpu_pallas,
    }


def _make_inert_sglang_stubs() -> dict[str, types.ModuleType]:
    """Create minimal inert stubs for `sglang_tpu` registration helpers."""

    def _inert_cpu_export_jax_op(name, *args, **kwargs):
        return _InertOpHandle(str(name))

    def _forbid(*args, **kwargs):
        raise RuntimeError(
            "Inert sglang_tpu helper stub was called at runtime; "
            "e2e tests must invoke the JAX-level kernel directly."
        )

    pkg = types.ModuleType("sglang_tpu")
    pkg.__path__ = []

    pallas_jax_op = types.ModuleType("sglang_tpu.pallas_jax_op")
    pallas_jax_op.cpu_export_jax_op = _inert_cpu_export_jax_op

    pallas_export = types.ModuleType("sglang_tpu.pallas_export")
    pallas_export.tpu_info_for_cpu_export = _forbid

    return {
        "sglang_tpu": pkg,
        "sglang_tpu.pallas_jax_op": pallas_jax_op,
        "sglang_tpu.pallas_export": pallas_export,
    }


def _ensure_parent_packages(
    dotted_name: str, stubs: dict[str, types.ModuleType]
) -> None:
    parts = dotted_name.split(".")[:-1]
    for i in range(1, len(parts) + 1):
        pkg_name = ".".join(parts[:i])
        if pkg_name not in stubs:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = []
            stubs[pkg_name] = pkg


def kernels_dir() -> Path:
    raw = os.environ.get("TKL_E2E_KERNELS_DIR")
    if not raw:
        raise RuntimeError(
            "TKL_E2E_KERNELS_DIR is not set; run via `bash scripts/e2e.sh`."
        )
    return Path(raw)


def _exec_module_from_path(module_name: str, file_path: Path) -> types.ModuleType:
    if not file_path.is_file():
        raise FileNotFoundError(f"Fetched file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_kernel_module(
    rel_path: str,
    module_name: str,
    *,
    stub_torch: bool = False,
    stub_sglang: bool = False,
    helper_paths: Sequence[str] = (),
) -> types.ModuleType:
    """Load a fetched Python kernel file from `TKL_E2E_KERNELS_DIR`.

    Any `helper_paths` (repository-relative `.py` paths fetched at the same
    manifest commit) are loaded as real modules under their package names
    before executing the kernel module. Unavailable host/TPU registration
    imports (`torch`, `sglang_tpu.pallas_jax_op`) are replaced with inert stubs
    only for the duration of import.
    """
    base = kernels_dir()
    stubs: dict[str, types.ModuleType] = {}
    if stub_torch:
        stubs.update(_make_inert_torch_stubs())
    if stub_sglang:
        stubs.update(_make_inert_sglang_stubs())

    helper_specs: list[tuple[str, Path]] = []
    for hp in helper_paths:
        mod_name = hp.removesuffix(".py").replace("/", ".")
        _ensure_parent_packages(mod_name, stubs)
        helper_specs.append((mod_name, base / hp))

    saved: dict[str, types.ModuleType | None] = {}

    def _inject(name: str, mod: types.ModuleType) -> None:
        if name not in saved:
            saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    try:
        for name, mod in stubs.items():
            if name not in sys.modules:
                _inject(name, mod)
        for h_mod_name, h_path in helper_specs:
            h_mod = _exec_module_from_path(h_mod_name, h_path)
            _inject(h_mod_name, h_mod)
        return _exec_module_from_path(module_name, base / rel_path)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


@contextlib.contextmanager
def tpu_interpret_mode(
    *,
    detect_races: bool = True,
    uninitialized_memory: str = "nan",
) -> Iterator[pltpu.InterpretParams]:
    """Run Pallas TPU kernels on CPU using JAX's TPU interpret mode."""
    force_mode, reset_state = _get_interpret_helpers()
    params = pltpu.InterpretParams(
        detect_races=detect_races,
        on_race="raise",
        out_of_bounds_reads="raise",
        uninitialized_memory=uninitialized_memory,
    )
    try:
        with force_mode(params):
            yield params
    finally:
        reset_state()


@pytest.fixture(scope="session", autouse=True)
def _verify_cpu_backend():
    assert jax.default_backend() == "cpu", (
        f"Expected JAX CPU backend, got {jax.default_backend()}"
    )
