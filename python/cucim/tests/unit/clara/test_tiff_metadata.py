#
# SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for cuslide vendor-specific TIFF metadata and associated images.

The inputs are tiny synthetic TIFFs whose tags mimic the Aperio SVS and
Philips TIFF whole-slide-image formats, so the vendor metadata parsers and the
associated-image (label / macro / thumbnail) reader are exercised without
needing real (large, proprietary) slide files.
"""

import numpy as np
import pytest
import tifffile

# JPEG tile compression in the generated TIFFs requires imagecodecs.
pytest.importorskip("imagecodecs")

from cucim import CuImage  # noqa: E402

_APERIO_DESCRIPTION = (
    "Aperio Image Library v1.0\r\n"
    "256x256 [0,0 256x256] "
    "|AppMag = 20|MPP = 0.4990|ScanScope ID = TEST"
)

_PHILIPS_DESCRIPTION = (
    '<DataObject ObjectType="DPUfsImport">'
    '<Attribute Name="PIM_DP_SCANNED_IMAGES"><Array>'
    '<DataObject ObjectType="DPScannedImage">'
    '<Attribute Name="PIM_DP_IMAGE_TYPE" PMSVR="IString">WSI</Attribute>'
    '<Attribute Name="PIIM_PIXEL_DATA_REPRESENTATION_SEQUENCE"><Array>'
    '<DataObject ObjectType="PixelDataRepresentation">'
    '<Attribute Name="DICOM_PIXEL_SPACING" PMSVR="IStringArray">'
    '"0.00025" "0.00025"</Attribute>'
    "</DataObject></Array></Attribute>"
    "</DataObject></Array></Attribute></DataObject>"
)


def _write_aperio_svs(path, with_associated=True):
    level0 = np.zeros((256, 256, 3), np.uint8)
    level0[::2] = (200, 0, 0)
    level0[1::2] = (0, 0, 200)
    small = np.full((64, 64, 3), (0, 200, 0), np.uint8)

    with tifffile.TiffWriter(str(path)) as tif:
        tif.write(
            level0,
            tile=(128, 128),
            photometric="RGB",
            compression="jpeg",
            description=_APERIO_DESCRIPTION,
            subfiletype=0,
        )
        if with_associated:
            # Non-tiled extra pages become associated images, identified by
            # NewSubfileType: page 1 with 0 -> thumbnail, 1 -> label, 9 -> macro.
            tif.write(small, photometric="RGB", compression="jpeg", subfiletype=0)
            tif.write(small, photometric="RGB", compression="jpeg", subfiletype=1)
            tif.write(
                small,
                photometric="RGB",
                compression="jpeg",
                extratags=[(254, 4, 1, 9)],
            )


def _write_philips_tiff(path):
    level0 = np.zeros((256, 256, 3), np.uint8)
    level0[::2] = (200, 0, 0)
    with tifffile.TiffWriter(str(path)) as tif:
        tif.write(
            level0,
            tile=(128, 128),
            photometric="RGB",
            compression="jpeg",
            description=_PHILIPS_DESCRIPTION,
            software="Philips DP v1.0",
        )


def test_aperio_svs_metadata(tmp_path):
    path = tmp_path / "aperio.svs"
    _write_aperio_svs(path, with_associated=False)

    metadata = CuImage(str(path)).metadata
    assert "aperio" in metadata
    aperio = metadata["aperio"]
    assert aperio["AppMag"] == "20"
    assert aperio["MPP"] == "0.4990"
    assert aperio["ScanScope ID"] == "TEST"


def test_aperio_svs_associated_images(tmp_path):
    path = tmp_path / "aperio.svs"
    _write_aperio_svs(path, with_associated=True)

    img = CuImage(str(path))
    assert img.associated_images == {"thumbnail", "label", "macro"}
    for name in img.associated_images:
        arr = np.asarray(img.associated_image(name))
        assert arr.shape == (64, 64, 3)


def test_philips_tiff_metadata(tmp_path):
    path = tmp_path / "philips.tif"
    _write_philips_tiff(path)

    # Opening a Philips-formatted TIFF exercises the Philips XML metadata
    # parser; the level-0 image is still a normal readable WSI.
    img = CuImage(str(path))
    assert img.shape == [256, 256, 3]
    arr = np.asarray(img.read_region((0, 0), (64, 64)))
    assert arr.shape == (64, 64, 3)
