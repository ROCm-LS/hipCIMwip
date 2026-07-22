#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate synthetic NIfTI/DICOM fixtures for the cumed (MedicalImage) C++ tests.

Writes deterministic .nii/.nii.gz/.dcm files under test_data/generated/medical/
(only if absent), invoked from gen_images.sh. Requires nibabel and pydicom; if
either is missing it warns and exits 0 so other test flows are unaffected.
"""

import argparse
import os
import struct
import sys


def _need(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def gen_nifti(dest):
    import nibabel as nib
    import numpy as np

    def save(name, arr, dtype, affine=None, units="mm", scl=None):
        path = os.path.join(dest, name)
        if os.path.exists(path):
            return
        if affine is None:
            affine = np.diag([1.0, 1.0, 1.0, 1.0])
        img = nib.Nifti1Image(arr.astype(dtype), affine)
        img.header.set_xyzt_units(units)
        if scl is not None:
            img.header["scl_slope"], img.header["scl_inter"] = scl
        nib.save(img, path)

    base = np.arange(8 * 8 * 2).reshape(8, 8, 2)

    # Core datatype coverage for cumed::nifti::nifti_datatype_to_dl().
    save("vol_uint8.nii", (np.arange(64 * 64 * 8) % 256).reshape(64, 64, 8),
         np.uint8, affine=np.diag([0.5, 0.5, 2.0, 1.0]))
    save("vol_int16.nii", (np.arange(32 * 32 * 4) % 1000 - 100).reshape(32, 32, 4),
         np.int16, scl=(2.0, 5.0))
    save("vol_i32.nii", base, np.int32)
    save("vol_f32.nii", base / 7.0, np.float32)
    save("vol_f64.nii", base / 3.0, np.float64)
    save("vol_u16.nii", base % 500, np.uint16)

    # 4D time-series -> ndim==5 (TZYXC) branch.
    save("vol_4d.nii", np.arange(8 * 8 * 2 * 3).reshape(8, 8, 2, 3), np.uint8)

    # gzip-compressed single-file NIfTI -> libdeflate gunzip path.
    save("vol_gz.nii.gz", (np.arange(16 * 16 * 3) % 256).reshape(16, 16, 3), np.uint8)

    # qform-only (sform disabled) -> quaternion direction reconstruction.
    qpath = os.path.join(dest, "vol_qform.nii")
    if not os.path.exists(qpath):
        aff = np.array([[0.0, -1.0, 0.0, 10.0],
                        [1.0, 0.0, 0.0, 20.0],
                        [0.0, 0.0, 2.0, 30.0],
                        [0.0, 0.0, 0.0, 1.0]])
        vol = (np.arange(8 * 8 * 4) % 256).astype(np.uint8).reshape(8, 8, 4)
        img = nib.Nifti1Image(vol, aff)
        img.set_sform(None, code=0)
        img.set_qform(aff, code=1)
        img.header.set_xyzt_units("mm")
        nib.save(img, qpath)

    # Hand-crafted big-endian header -> cumed byteswap_header() path. nibabel
    # always writes little-endian, so the 348-byte header is packed manually.
    bepath = os.path.join(dest, "vol_be.nii")
    if not os.path.exists(bepath):
        h = bytearray(348)
        struct.pack_into(">i", h, 0, 348)            # sizeof_hdr
        struct.pack_into(">8h", h, 40, 3, 16, 16, 3, 1, 1, 1, 1)  # dim
        struct.pack_into(">h", h, 70, 2)             # datatype = uint8
        struct.pack_into(">h", h, 72, 8)             # bitpix
        struct.pack_into(">8f", h, 76, 1, 1, 1, 1, 0, 0, 0, 0)    # pixdim
        struct.pack_into(">f", h, 108, 352.0)        # vox_offset
        struct.pack_into(">b", h, 123, 2)            # xyzt_units = mm
        h[344:348] = b"n+1\x00"                       # magic
        data = (np.arange(16 * 16 * 3) % 256).astype(np.uint8).tobytes()
        with open(bepath, "wb") as f:
            f.write(bytes(h) + b"\x00" * 4 + data)


def gen_dicom(dest):
    import numpy as np
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian

    def make(name, ts, rows, cols, samples, photometric, bits, pixrepr, data):
        path = os.path.join(dest, name)
        if os.path.exists(path):
            return
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ts
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5"
        ds.preamble = b"\x00" * 128
        ds.Rows, ds.Columns, ds.SamplesPerPixel = rows, cols, samples
        ds.PhotometricInterpretation = photometric
        ds.BitsAllocated = ds.BitsStored = bits
        ds.HighBit = bits - 1
        ds.PixelRepresentation = pixrepr
        if samples > 1:
            ds.PlanarConfiguration = 0
        ds.PixelSpacing = [0.5, 0.5]
        ds.SliceThickness = 1.0
        ds.NumberOfFrames = 1
        ds.ImagePositionPatient = [1.0, 2.0, 3.0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.RescaleSlope = 1.5
        ds.RescaleIntercept = -1024
        ds.PixelData = data.tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = (ts == ImplicitVRLittleEndian)
        ds.save_as(path, enforce_file_format=True)

    make("ct_mono16.dcm", ExplicitVRLittleEndian, 64, 64, 1, "MONOCHROME2", 16, 0,
         (np.arange(64 * 64) % 4000).astype(np.uint16).reshape(64, 64))
    make("rgb_u8.dcm", ExplicitVRLittleEndian, 48, 48, 3, "RGB", 8, 0,
         (np.arange(48 * 48 * 3) % 256).astype(np.uint8).reshape(48, 48, 3))
    make("ct_implicit.dcm", ImplicitVRLittleEndian, 40, 40, 1, "MONOCHROME2", 16, 0,
         (np.arange(40 * 40) % 3000).astype(np.uint16).reshape(40, 40))
    make("ct_signed.dcm", ExplicitVRLittleEndian, 40, 40, 1, "MONOCHROME1", 16, 1,
         (np.arange(40 * 40) % 2000 - 1000).astype(np.int16).reshape(40, 40))


def main():
    parser = argparse.ArgumentParser(description="Generate NIfTI/DICOM test fixtures")
    parser.add_argument("--dest", "-d", default="test_data/generated/medical",
                        help="destination folder")
    args = parser.parse_args()
    os.makedirs(args.dest, exist_ok=True)

    missing = [m for m in ("nibabel", "pydicom", "numpy") if not _need(m)]
    if missing:
        print(f"[gen_medical] Skipping: missing packages: {', '.join(missing)} "
              f"(install with: pip install {' '.join(missing)})", file=sys.stderr)
        return 0

    gen_nifti(args.dest)
    gen_dicom(args.dest)
    print(f"[gen_medical] NIfTI/DICOM fixtures ready in {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
