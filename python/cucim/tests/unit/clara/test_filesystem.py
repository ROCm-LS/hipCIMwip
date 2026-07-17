#
# SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the ``cucim.clara.filesystem`` (cuFile) driver API.

GPUDirect Storage is unavailable in most environments (``is_gds_available()``
returns ``False``), so these tests exercise the POSIX fallback paths of the
cuFile driver, which are otherwise untested from Python.
"""

import numpy as np
import pytest

from cucim.clara import filesystem as fs


def test_is_gds_available():
    # Should return a bool regardless of whether GDS is present.
    assert isinstance(fs.is_gds_available(), bool)


def test_pread_host_buffer(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(bytes(range(256)) * 4)

    fd = fs.open(str(path), "r")
    try:
        buf = np.zeros(256, dtype=np.uint8)
        assert fs.pread(fd, buf, 256, 0) == 256
        assert np.array_equal(buf, np.arange(256, dtype=np.uint8))

        # Read from a non-zero file offset into a non-zero buffer offset.
        buf2 = np.full(128, 255, dtype=np.uint8)
        assert fs.pread(fd, buf2, 64, 256, 64) == 64
        assert np.array_equal(buf2[64:128], np.arange(64, dtype=np.uint8))
    finally:
        fs.close(fd)


def test_pwrite_then_read_back(tmp_path):
    path = tmp_path / "out.bin"
    src = np.arange(200, dtype=np.uint8)

    with fs.open(str(path), "w") as fd:
        assert fs.pwrite(fd, src, 200, 0) == 200

    with fs.open(str(path), "r") as fd:
        buf = np.zeros(200, dtype=np.uint8)
        assert fs.pread(fd, buf, 200, 0) == 200
    assert np.array_equal(buf, src)


def test_discard_page_cache(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"\x00" * 4096)
    # Safe to call regardless of GDS availability.
    fs.discard_page_cache(str(path))


def test_pread_cuda_buffer(tmp_path):
    cp = pytest.importorskip("cupy")
    path = tmp_path / "data.bin"
    path.write_bytes(bytes(range(256)) * 4)

    with fs.open(str(path), "r") as fd:
        buf = cp.zeros(256, dtype=cp.uint8)
        assert fs.pread(fd, buf, 256, 0) == 256
    assert bool((buf.get() == np.arange(256, dtype=np.uint8)).all())
