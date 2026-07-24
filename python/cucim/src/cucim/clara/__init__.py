#
# SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import os

# Preload ROCm shared libraries before importing any native extension so that
# hipCIM resolves its ROCm dependencies regardless of where ROCm is installed
# (pip/venv wheels vs. a system /opt/rocm). No-op if the rocm-sdk package is
# not present.
try:
    from . import _rocm_init
except ModuleNotFoundError:
    pass
else:
    _rocm_init.initialize()
    del _rocm_init

from . import cli, converter

# import hidden methods
from ._cucim import CuImage, DLDataType, DLDataTypeCode, cache, filesystem, io

__all__ = [
    "cli",
    "CuImage",
    "DLDataType",
    "DLDataTypeCode",
    "filesystem",
    "io",
    "cache",
    "converter",
]


from ._cucim import _get_plugin_root  # isort:skip
from ._cucim import _set_plugin_root  # isort:skip

# Set plugin root path
_set_plugin_root(os.path.dirname(os.path.realpath(__file__)))
