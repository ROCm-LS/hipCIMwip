#
# SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for ``cucim.clara.io.Device`` name parsing and validation."""

import pytest

from cucim.clara import io


@pytest.mark.parametrize(
    "name, expected",
    [
        ("cpu", "cpu"),
        ("cuda", "cuda"),
        ("cuda:0", "cuda:0"),
        ("cpu:1", "cpu:1"),
        ("cpu:2[s]", "cpu:2[s]"),
        ("cpu[shm0]", "cpu:-1[shm0]"),
        ("cuda:0[cuda_shm0]", "cuda:0[cuda_shm0]"),
        ("cpu[my-shm_1.0]", "cpu:-1[my-shm_1.0]"),
    ],
)
def test_device_valid_names(name, expected):
    # A device string is "type[:index][[shm_name]]"; round-tripping through
    # str() exercises the type / index / shm-name parsing and validation.
    assert str(io.Device(name)) == expected


@pytest.mark.parametrize(
    "name",
    [
        "cuda:00",  # leading zero not allowed in an index
        "cpu:01",  # leading zero not allowed in an index
        "cpu[bad!shm]",  # '!' is not a valid shared-memory-name character
    ],
)
def test_device_invalid_names(name):
    with pytest.raises(RuntimeError):
        io.Device(name)
