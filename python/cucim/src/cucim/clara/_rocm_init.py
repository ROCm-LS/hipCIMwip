# SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preload ROCm shared libraries via the ROCm (TheRock) Python packages.

When hipCIM is installed as a wheel alongside the ROCm Python packages
(``pip install "rocm[libraries]"``), the ROCm runtime libraries live inside the
``_rocm_sdk_*`` trees in ``site-packages`` rather than on the default dynamic
loader search path. Asking ``rocm_sdk`` to load them into the global symbol
namespace lets hipCIM's own native extensions resolve their ROCm dependencies
(``libamdhip64``, ``librocjpeg``, ...) regardless of the wheel's install
location or the active virtual environment.

This is a best-effort no-op when the ``rocm_sdk`` package is not installed --
for example when ROCm is provided by a classic system ``/opt/rocm`` install that
is instead picked up through the libraries' RPATH.
"""

# Short (SONAME-less) names of the ROCm libraries that hipCIM's native
# extensions link against directly. Their own transitive dependencies
# (hsa-runtime64, amd_comgr, hiprtc, ...) are resolved by the ROCm packages'
# RPATHs once these are loaded.
_PRELOAD_SHORTNAMES = [
    "amdhip64",
    "rocjpeg",
]


def initialize() -> None:
    """Preload ROCm libraries if the ROCm Python packages are available."""
    try:
        import rocm_sdk
    except ModuleNotFoundError:
        return

    try:
        rocm_sdk.initialize_process(preload_shortnames=_PRELOAD_SHORTNAMES)
    except Exception:
        # Best effort: fall back to RPATH / LD_LIBRARY_PATH based resolution.
        pass
