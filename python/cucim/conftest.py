#
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Session-wide pytest configuration for the cuCIM Python test suite.

On ROCm/HIP, CuPy compiles kernels at runtime through hiprtc/comgr. Unlike the
C++ build, hiprtc has no knowledge of GCC installations that live outside the
default compiler search path (such as ``gcc-toolset`` on RHEL-family systems).
It therefore falls back to the system GCC headers, which can be too old for the
bundled ROCm clang and make every kernel compilation fail (e.g. an error in
``cuda_wrappers/bits/c++config.h``).

The C++ build already works around this by passing ``--gcc-toolchain`` to hipcc
(see ``run_amd`` and ``GCC_TOOLCHAIN_ROOT``). This module applies the equivalent
flag to CuPy's runtime compiler so the Python test suite can build kernels on
HIP. It is a no-op on CUDA/NVIDIA and when no suitable GCC toolchain is found.
"""

import os
import shutil

import pytest


def _gcc_toolchain_root():
    """Return the GCC toolchain root, mirroring the logic in ``run_amd``.

    Prefers the ``GCC_TOOLCHAIN_ROOT`` environment variable (exported by
    ``run_amd``) and otherwise derives it from ``gcc`` on ``PATH`` as
    ``dirname(dirname(realpath(gcc)))`` (e.g. ``/opt/rh/gcc-toolset-14/root/usr``).
    """
    root = os.environ.get("GCC_TOOLCHAIN_ROOT")
    if root:
        return root
    gcc = shutil.which("gcc")
    if not gcc:
        return None
    return os.path.dirname(os.path.dirname(os.path.realpath(gcc)))


def _patch_cupy_gcc_toolchain():
    """Inject ``--gcc-toolchain`` into CuPy's runtime kernel compiler on HIP."""
    try:
        import cupy
        from cupy.cuda import compiler
    except Exception:
        # CuPy is optional for parts of the suite; nothing to do without it.
        return

    if not getattr(cupy.cuda.runtime, "is_hip", False):
        # Only HIP needs the toolchain hint; CUDA/NVIDIA is unaffected.
        return

    if getattr(compiler, "_cucim_gcc_toolchain_patched", False):
        return

    root = _gcc_toolchain_root()
    if not root or not os.path.isdir(root):
        return

    flag = f"--gcc-toolchain={root}"
    _orig_compile = compiler._compile_module_with_cache

    def _compile_with_gcc_toolchain(source, options=(), **kwargs):
        if flag not in options:
            options = tuple(options) + (flag,)
        return _orig_compile(source, options, **kwargs)

    compiler._compile_module_with_cache = _compile_with_gcc_toolchain
    compiler._cucim_gcc_toolchain_patched = True


_patch_cupy_gcc_toolchain()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Reset reused test-class instances so ``pytest-rerunfailures`` can retry them.

    ``cupy.testing.parameterize`` installs an autouse fixture that asserts the
    test instance's ``__dict__`` is empty and then fills it with the parameter
    set. pytest normally hands each test attempt a fresh instance, but
    ``pytest-rerunfailures`` reuses the same instance across reruns, so on a retry
    the ``__dict__`` still holds the previous attempt's parameters and the
    assertion raises. That surfaces as a *setup ERROR* which both masks the
    original failure and prevents ``--reruns`` from ever succeeding for such
    tests.

    Clearing the instance's ``__dict__`` after each attempt restores the
    fresh-instance invariant. pytest forbids ``__init__`` on test classes, so
    there is no constructor state to preserve, and for non-rerun tests the
    instance is discarded right after teardown, making this a no-op in the common
    case. The post-yield placement runs after all fixture finalizers, so it never
    removes state a teardown still needs.
    """
    yield
    instance = getattr(item, "instance", None)
    if instance is not None:
        try:
            instance.__dict__.clear()
        except Exception:
            # Test-instance cleanup must never break teardown.
            pass

