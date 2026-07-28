#
# SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the cumed (medical image) plugin: NIfTI-1 and DICOM readers.

The inputs are tiny, hand-crafted files so the suite does not depend on external
sample data or optional packages (nibabel / pydicom).
"""

import gzip
import struct

import numpy as np
import pytest

from cucim import CuImage

# NIfTI datatype code -> (numpy dtype, bits per voxel)
_NIFTI_DTYPES = {
    2: (np.uint8, 8),
    4: (np.int16, 16),
    16: (np.float32, 32),
    64: (np.float64, 64),
    512: (np.uint16, 16),
}


def _write_nifti(
    path, nx, ny, nz, datatype=2, big_endian=False, gzip_it=False
):
    """Write a minimal, valid single-file NIfTI-1 volume of arange() voxels."""
    np_dtype, bitpix = _NIFTI_DTYPES[datatype]
    endian = ">" if big_endian else "<"

    header = bytearray(348)
    struct.pack_into(endian + "i", header, 0, 348)  # sizeof_hdr
    struct.pack_into(endian + "h", header, 40, 3)  # dim[0] = ndims
    struct.pack_into(endian + "h", header, 42, nx)  # dim[1]
    struct.pack_into(endian + "h", header, 44, ny)  # dim[2]
    struct.pack_into(endian + "h", header, 46, nz)  # dim[3]
    struct.pack_into(endian + "h", header, 70, datatype)
    struct.pack_into(endian + "h", header, 72, bitpix)
    for offset in (76, 80, 84, 88):  # pixdim[0..3]
        struct.pack_into(endian + "f", header, offset, 1.0)
    struct.pack_into(endian + "f", header, 108, 352.0)  # vox_offset
    header[123] = 2  # xyzt_units = millimeter
    header[344:348] = b"n+1\x00"  # magic

    voxels = np.arange(nx * ny * nz, dtype=np_dtype)
    if big_endian:
        voxels = voxels.byteswap()
    payload = bytes(header) + b"\x00" * 4 + voxels.tobytes()
    if gzip_it:
        payload = gzip.compress(payload)
    path.write_bytes(payload)


def _dicom_element(group, element, vr, value, implicit=False):
    """Encode one DICOM data element (Explicit or Implicit VR Little Endian)."""
    tag = struct.pack("<HH", group, element)
    if implicit:
        return tag + struct.pack("<I", len(value)) + value
    if vr in (b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"):
        return tag + vr + b"\x00\x00" + struct.pack("<I", len(value)) + value
    return tag + vr + struct.pack("<H", len(value)) + value


def _write_dicom(path, rows, cols, samples=1, implicit_vr=False, geometry=False):
    """Write a minimal uncompressed single-frame DICOM.

    Supports Explicit or Implicit VR Little Endian, grayscale or RGB, and
    optional geometry tags (pixel spacing / slice thickness / position /
    orientation / rescale, all DS multi-value).
    """
    transfer_syntax = (
        b"1.2.840.10008.1.2\x00" if implicit_vr  # Implicit VR LE
        else b"1.2.840.10008.1.2.1\x00"  # Explicit VR LE
    )
    # The file-meta group is always Explicit VR LE.
    meta = _dicom_element(0x0002, 0x0010, b"UI", transfer_syntax)
    # (0002,0000) FileMetaInformationGroupLength is the byte count of all file-
    # meta elements that follow it. TransferSyntaxUID is the only one here, so
    # len(meta) is the full group length; extend this sum if more are added.
    meta_group_length = _dicom_element(
        0x0002, 0x0000, b"UL", struct.pack("<I", len(meta))
    )

    imp = implicit_vr
    photometric = b"RGB " if samples == 3 else b"MONOCHROME2 "
    elements = [
        _dicom_element(0x0028, 0x0002, b"US", struct.pack("<H", samples), imp),
        _dicom_element(0x0028, 0x0004, b"CS", photometric, imp),
        _dicom_element(0x0028, 0x0010, b"US", struct.pack("<H", rows), imp),
        _dicom_element(0x0028, 0x0011, b"US", struct.pack("<H", cols), imp),
        _dicom_element(0x0028, 0x0100, b"US", struct.pack("<H", 8), imp),
        _dicom_element(0x0028, 0x0101, b"US", struct.pack("<H", 8), imp),
        _dicom_element(0x0028, 0x0103, b"US", struct.pack("<H", 0), imp),
    ]
    if samples == 3:
        elements.append(
            _dicom_element(0x0028, 0x0006, b"US", struct.pack("<H", 0), imp)
        )
    if geometry:
        elements += [
            _dicom_element(0x0018, 0x0050, b"DS", b"1.0 ", imp),
            _dicom_element(0x0020, 0x0032, b"DS", b"0\\0\\0 ", imp),
            _dicom_element(0x0020, 0x0037, b"DS", b"1\\0\\0\\0\\1\\0 ", imp),
            _dicom_element(0x0028, 0x0030, b"DS", b"0.5\\0.5 ", imp),
            _dicom_element(0x0028, 0x1052, b"DS", b"0 ", imp),
            _dicom_element(0x0028, 0x1053, b"DS", b"1 ", imp),
        ]
    elements.append(
        _dicom_element(
            # 8-bit uncompressed Pixel Data uses VR=OB under Explicit VR LE (OW
            # is for >8-bit); matches the encapsulated OB helper below.
            0x7FE0, 0x0010, b"OB", bytes(range(rows * cols * samples)), imp
        )
    )
    path.write_bytes(
        b"\x00" * 128 + b"DICM" + meta_group_length + meta + b"".join(elements)
    )


def _encapsulated_pixel_data(frame):
    """Wrap one compressed frame as an encapsulated (undefined-length) PixelData."""
    if len(frame) % 2:
        frame += b"\x00"  # DICOM values (and items) must be even length
    return b"".join(
        [
            struct.pack("<HH", 0x7FE0, 0x0010) + b"OB\x00\x00"
            + struct.pack("<I", 0xFFFFFFFF),  # OB, undefined length
            struct.pack("<HH", 0xFFFE, 0xE000) + struct.pack("<I", 0),  # empty BOT
            struct.pack("<HH", 0xFFFE, 0xE000)
            + struct.pack("<I", len(frame)) + frame,  # frame item
            struct.pack("<HH", 0xFFFE, 0xE0DD) + struct.pack("<I", 0),  # seq delim
        ]
    )


def _write_dicom_compressed(path, rows, cols, samples, transfer_syntax, frame):
    """Write a DICOM whose single frame is compressed (encapsulated)."""
    meta = _dicom_element(0x0002, 0x0010, b"UI", transfer_syntax)
    meta_group_length = _dicom_element(
        0x0002, 0x0000, b"UL", struct.pack("<I", len(meta))
    )
    photometric = b"RGB " if samples == 3 else b"MONOCHROME2 "
    elements = [
        _dicom_element(0x0028, 0x0002, b"US", struct.pack("<H", samples)),
        _dicom_element(0x0028, 0x0004, b"CS", photometric),
        _dicom_element(0x0028, 0x0010, b"US", struct.pack("<H", rows)),
        _dicom_element(0x0028, 0x0011, b"US", struct.pack("<H", cols)),
        _dicom_element(0x0028, 0x0100, b"US", struct.pack("<H", 8)),
        _dicom_element(0x0028, 0x0101, b"US", struct.pack("<H", 8)),
        _dicom_element(0x0028, 0x0103, b"US", struct.pack("<H", 0)),
    ]
    if samples == 3:
        elements.append(
            _dicom_element(0x0028, 0x0006, b"US", struct.pack("<H", 0))
        )
    elements.append(_encapsulated_pixel_data(frame))
    path.write_bytes(
        b"\x00" * 128 + b"DICM" + meta_group_length + meta + b"".join(elements)
    )


def test_read_nifti(tmp_path):
    path = tmp_path / "volume.nii"
    _write_nifti(path, nx=4, ny=3, nz=2)

    img = CuImage(str(path))
    assert img.dims == "ZYXC"
    assert img.ndim == 4
    assert img.shape == [2, 3, 4, 1]

    arr = np.asarray(img.read_region())
    assert arr.dtype == np.uint8
    assert np.array_equal(arr.ravel(), np.arange(24))


def test_read_nifti_gzip(tmp_path):
    path = tmp_path / "volume.nii.gz"
    _write_nifti(path, nx=4, ny=3, nz=2, gzip_it=True)

    arr = np.asarray(CuImage(str(path)).read_region())
    assert np.array_equal(arr.ravel(), np.arange(24))


def test_read_nifti_big_endian(tmp_path):
    path = tmp_path / "be.nii"
    # Use a multi-byte datatype (512 = uint16) so the read actually exercises
    # the big-endian byte-swap; uint8 would be byte-order-agnostic.
    _write_nifti(path, nx=4, ny=3, nz=2, datatype=512, big_endian=True)

    arr = np.asarray(CuImage(str(path)).read_region())
    assert arr.dtype == np.uint16
    assert np.array_equal(arr.ravel(), np.arange(24, dtype=np.uint16))


@pytest.mark.parametrize("datatype", [4, 16, 64, 512])
def test_read_nifti_datatypes(tmp_path, datatype):
    np_dtype, _ = _NIFTI_DTYPES[datatype]
    path = tmp_path / f"dt{datatype}.nii"
    _write_nifti(path, nx=4, ny=3, nz=2, datatype=datatype)

    arr = np.asarray(CuImage(str(path)).read_region())
    assert arr.dtype == np_dtype
    assert np.array_equal(arr.ravel(), np.arange(24, dtype=np_dtype))


def test_read_dicom(tmp_path):
    path = tmp_path / "image.dcm"
    _write_dicom(path, rows=4, cols=6)

    img = CuImage(str(path))
    assert img.dims == "YXC"
    assert img.ndim == 3
    assert img.shape == [4, 6, 1]

    arr = np.asarray(img.read_region())
    assert arr.dtype == np.uint8
    assert np.array_equal(arr.ravel(), np.arange(24))


def test_read_dicom_implicit_vr_with_geometry(tmp_path):
    path = tmp_path / "implicit.dcm"
    _write_dicom(path, rows=4, cols=6, implicit_vr=True, geometry=True)

    img = CuImage(str(path))
    assert img.dims == "YXC"
    # DS multi-value geometry tags are parsed into the physical spacing
    # (PixelSpacing 0.5x0.5 mm, SliceThickness 1.0 mm).
    assert img.spacing() == [0.5, 0.5, 1.0]

    arr = np.asarray(img.read_region())
    assert np.array_equal(arr.ravel(), np.arange(24))


def test_read_dicom_rgb(tmp_path):
    path = tmp_path / "rgb.dcm"
    _write_dicom(path, rows=2, cols=2, samples=3)

    img = CuImage(str(path))
    assert img.shape == [2, 2, 3]

    arr = np.asarray(img.read_region())
    assert arr.dtype == np.uint8
    assert np.array_equal(arr.ravel(), np.arange(12))


def test_read_dicom_jpeg_baseline(tmp_path):
    imagecodecs = pytest.importorskip("imagecodecs")
    rows = cols = 32
    source = np.zeros((rows, cols, 3), np.uint8)
    source[..., 0] = np.arange(cols, dtype=np.uint8)  # R gradient
    source[..., 1] = 64  # constant G
    frame = imagecodecs.jpeg_encode(source)

    path = tmp_path / "jpeg.dcm"
    _write_dicom_compressed(
        path, rows, cols, 3, b"1.2.840.10008.1.2.4.50\x00", frame
    )  # JPEG Baseline transfer syntax

    arr = np.asarray(CuImage(str(path)).read_region())
    assert arr.shape == (rows, cols, 3)
    assert arr.dtype == np.uint8
    # JPEG is lossy, but the constant green channel is roughly preserved.
    assert abs(int(arr[..., 1].mean()) - 64) < 20


def test_read_dicom_jpeg2000(tmp_path):
    imagecodecs = pytest.importorskip("imagecodecs")
    rows = cols = 32
    source = np.zeros((rows, cols, 3), np.uint8)
    source[..., 0] = np.arange(cols, dtype=np.uint8)
    source[..., 2] = 128  # constant B
    # DICOM JPEG 2000 uses a raw J2K codestream, not the JP2 box format.
    frame = imagecodecs.jpeg2k_encode(source, codecformat="J2K")

    path = tmp_path / "jpeg2000.dcm"
    _write_dicom_compressed(
        path, rows, cols, 3, b"1.2.840.10008.1.2.4.90\x00", frame
    )  # JPEG 2000 transfer syntax

    arr = np.asarray(CuImage(str(path)).read_region())
    assert arr.shape == (rows, cols, 3)
    assert arr.dtype == np.uint8
    assert abs(int(arr[..., 2].mean()) - 128) < 20
