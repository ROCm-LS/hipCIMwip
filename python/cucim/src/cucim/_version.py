# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Modifications Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
import importlib.resources

__version__ = (
    importlib.resources.files(__package__)
    .joinpath("HIPCIM_VERSION")
    .read_text()
    .strip()
)
try:
    __git_commit__ = (
        importlib.resources.files(__package__)
        .joinpath("GIT_COMMIT")
        .read_text()
        .strip()
    )
except FileNotFoundError:
    __git_commit__ = ""

__all__ = ["__git_commit__", "__version__"]
