/*
 * SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sys/stat.h>

#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "config.h"
#include "cucim/cuimage.h"
#include "cucim/memory/dlpack.h"

// Reads generated NIfTI/DICOM fixtures through cucim::CuImage (which loads the
// cumed plugin) to cover cumed.cpp/nifti.h/dicom.h -- header parsing, datatype
// mapping, metadata construction and the raw voxel read path -- none of which
// the Python suite reaches. Fixtures come from gen_medical.py (via gen_images.sh).

namespace
{
// A correct decode yields a raster that is neither empty nor uniform (the
// fixtures carry varied, non-zero voxels), catching a blank/zeroed buffer that
// the shape/dtype checks alone would accept.
bool region_has_content(cucim::CuImage& region)
{
    const cucim::memory::DLTContainer container = region.container();
    DLTensor* handle = static_cast<DLTensor*>(container);
    if (handle == nullptr || handle->data == nullptr)
    {
        return false;
    }
    const size_t n = container.size();
    const auto* p = static_cast<const uint8_t*>(handle->data);
    const uint8_t first = (n > 0) ? p[0] : 0;
    bool nonzero = false;
    bool varied = false;
    for (size_t i = 0; i < n && !(nonzero && varied); ++i)
    {
        nonzero = nonzero || (p[i] != 0);
        varied = varied || (p[i] != first);
    }
    return nonzero && varied;
}

// True if `path` names an existing filesystem entry.
bool file_exists(const std::string& path)
{
    struct stat buffer;
    return stat(path.c_str(), &buffer) == 0;
}

// Resolve a generated medical fixture, or SKIP the test when it is absent.
// gen_medical.py no-ops (and gen_images.sh still exits 0) when nibabel/pydicom
// are unavailable, so these fixtures may legitimately not exist. Skipping turns
// that missing optional dependency into a clear, self-explanatory skip instead
// of a confusing CuImage decode failure later in the case.
std::string medical_fixture_or_skip(const std::string& rel_path)
{
    const std::string path = g_config.get_input_path(rel_path);
    if (!file_exists(path))
    {
        SKIP("Medical fixture '" + rel_path + "' not found; run test_data/gen_medical.py "
             "(needs nibabel/pydicom). Skipping.");
    }
    return path;
}
} // namespace

SCENARIO("cumed parses NIfTI volumes", "[test_medical.cpp]")
{
    GIVEN("A 3D uint8 NIfTI volume")
    {
        const std::string path = medical_fixture_or_skip("generated/medical/vol_uint8.nii");
        cucim::CuImage image(path);

        THEN("Header geometry and dtype are parsed into metadata")
        {
            REQUIRE(image.is_loaded());
            // NIfTI is exposed as ZYXC (depth, height, width, channel).
            REQUIRE(image.dims() == "ZYXC");
            REQUIRE(image.ndim() >= 3);

            const cucim::Shape shape = image.shape();
            REQUIRE(shape.size() >= 3);
            REQUIRE(shape[0] == 8); // Z
            REQUIRE(shape[1] == 64); // Y
            REQUIRE(shape[2] == 64); // X

            const DLDataType dtype = image.dtype();
            REQUIRE(dtype.code == kDLUInt);
            REQUIRE(dtype.bits == 8);
        }

        AND_THEN("Spacing, units and the metadata JSON are populated")
        {
            const std::vector<float> spacing = image.spacing();
            REQUIRE(spacing.size() >= 3);

            const std::vector<std::string> units = image.spacing_units();
            REQUIRE_FALSE(units.empty());

            // cumed serializes a "nifti1" object into the metadata JSON.
            const cucim::Metadata meta = image.metadata();
            REQUIRE_FALSE(meta.empty());
        }

        AND_WHEN("A whole-volume region is read on the host")
        {
            // cumed readers ignore the request window; location/size are nominal.
            auto region = image.read_region({ 0, 0 }, { 1, 1 });
            THEN("The decoded raster has the expected shape and dtype")
            {
                REQUIRE(region.is_loaded());
                const cucim::Shape rshape = region.shape();
                REQUIRE(rshape[0] == 8);
                REQUIRE(rshape[1] == 64);
                REQUIRE(rshape[2] == 64);
                REQUIRE(region.dtype().bits == 8);
                REQUIRE(region_has_content(region));
            }
        }
    }

    GIVEN("A 3D int16 NIfTI volume with intensity scaling")
    {
        const std::string path = medical_fixture_or_skip("generated/medical/vol_int16.nii");
        cucim::CuImage image(path);

        THEN("The signed 16-bit datatype is mapped and the volume is readable")
        {
            REQUIRE(image.is_loaded());
            const DLDataType dtype = image.dtype();
            REQUIRE(dtype.code == kDLInt);
            REQUIRE(dtype.bits == 16);

            auto region = image.read_region({ 0, 0 }, { 1, 1 });
            REQUIRE(region.is_loaded());
            REQUIRE(region.dtype().bits == 16);
        }
    }
}

SCENARIO("cumed parses uncompressed DICOM images", "[test_medical.cpp]")
{
    GIVEN("A single-frame MONOCHROME2 uint16 DICOM")
    {
        const std::string path = medical_fixture_or_skip("generated/medical/ct_mono16.dcm");
        cucim::CuImage image(path);

        THEN("Rows/Columns/BitsAllocated map onto the image metadata")
        {
            REQUIRE(image.is_loaded());
            REQUIRE(image.dims() == "YXC");

            const cucim::Shape shape = image.shape();
            REQUIRE(shape[0] == 64); // rows (Y)
            REQUIRE(shape[1] == 64); // columns (X)
            REQUIRE(shape[2] == 1); // samples per pixel

            const DLDataType dtype = image.dtype();
            REQUIRE(dtype.code == kDLUInt);
            REQUIRE(dtype.bits == 16);
        }

        AND_WHEN("The pixel data is read (raw transfer syntax path)")
        {
            auto region = image.read_region({ 0, 0 }, { 1, 1 });
            THEN("A 16-bit raster of the full frame is returned")
            {
                REQUIRE(region.is_loaded());
                const cucim::Shape rshape = region.shape();
                REQUIRE(rshape[0] == 64);
                REQUIRE(rshape[1] == 64);
                REQUIRE(region.dtype().bits == 16);
                REQUIRE(region_has_content(region));
            }
        }
    }

    GIVEN("A single-frame RGB uint8 DICOM")
    {
        const std::string path = medical_fixture_or_skip("generated/medical/rgb_u8.dcm");
        cucim::CuImage image(path);

        THEN("Three interleaved samples per pixel are reported and read")
        {
            REQUIRE(image.is_loaded());
            const cucim::Shape shape = image.shape();
            REQUIRE(shape[0] == 48);
            REQUIRE(shape[1] == 48);
            REQUIRE(shape[2] == 3);

            const DLDataType dtype = image.dtype();
            REQUIRE(dtype.code == kDLUInt);
            REQUIRE(dtype.bits == 8);

            auto region = image.read_region({ 0, 0 }, { 1, 1 });
            REQUIRE(region.is_loaded());
            const cucim::Shape rshape = region.shape();
            REQUIRE(rshape[2] == 3);
            REQUIRE(region_has_content(region));
        }
    }
}

SCENARIO("cumed handles NIfTI and DICOM encoding variants", "[test_medical.cpp]")
{
    GIVEN("A big-endian NIfTI file")
    {
        // Forces the header byte-swap path in nifti.h (sizeof_hdr != 348 on read).
        const std::string path = medical_fixture_or_skip("generated/medical/vol_be.nii");
        cucim::CuImage image(path);
        THEN("The byte-swapped header parses and the volume reads")
        {
            REQUIRE(image.is_loaded());
            REQUIRE(image.dims() == "ZYXC");
            auto region = image.read_region({ 0, 0 }, { 1, 1 });
            REQUIRE(region.is_loaded());
        }
    }

    GIVEN("A gzip-compressed NIfTI file (.nii.gz)")
    {
        // Forces the libdeflate gunzip path in nifti.h.
        const std::string path = medical_fixture_or_skip("generated/medical/vol_gz.nii.gz");
        cucim::CuImage image(path);
        THEN("The decompressed volume parses and reads")
        {
            REQUIRE(image.is_loaded());
            auto region = image.read_region({ 0, 0 }, { 1, 1 });
            REQUIRE(region.is_loaded());
        }
    }

    GIVEN("An Implicit VR Little Endian DICOM")
    {
        // Forces the implicit-VR mini-dictionary path in dicom.h implicit_vr().
        const std::string path = medical_fixture_or_skip("generated/medical/ct_implicit.dcm");
        cucim::CuImage image(path);
        THEN("Tags are resolved via the implicit-VR dictionary and pixels read")
        {
            REQUIRE(image.is_loaded());
            const cucim::Shape shape = image.shape();
            REQUIRE(shape[0] == 40);
            REQUIRE(shape[1] == 40);
            auto region = image.read_region({ 0, 0 }, { 1, 1 });
            REQUIRE(region.is_loaded());
            REQUIRE(region.dtype().bits == 16);
        }
    }

    GIVEN("A signed 16-bit MONOCHROME1 DICOM")
    {
        // PixelRepresentation=1 exercises the signed-integer dtype mapping.
        const std::string path = medical_fixture_or_skip("generated/medical/ct_signed.dcm");
        cucim::CuImage image(path);
        THEN("The signed datatype is reported and pixels read")
        {
            REQUIRE(image.is_loaded());
            const DLDataType dtype = image.dtype();
            REQUIRE(dtype.code == kDLInt);
            REQUIRE(dtype.bits == 16);
            auto region = image.read_region({ 0, 0 }, { 1, 1 });
            REQUIRE(region.is_loaded());
        }
    }
}

SCENARIO("cumed maps NIfTI datatypes and time-series dimensions", "[test_medical.cpp]")
{
    struct DtypeCase
    {
        const char* file;
        uint8_t code;
        uint8_t bits;
    };
    const std::vector<DtypeCase> cases = {
        { "vol_i32.nii", kDLInt, 32 },
        { "vol_f32.nii", kDLFloat, 32 },
        { "vol_f64.nii", kDLFloat, 64 },
        { "vol_u16.nii", kDLUInt, 16 },
    };

    for (const auto& c : cases)
    {
        GIVEN(std::string("A NIfTI volume: ") + c.file)
        {
            const std::string path =
                medical_fixture_or_skip(std::string("generated/medical/") + c.file);
            cucim::CuImage image(path);
            THEN("nifti_datatype_to_dl maps it and the volume reads")
            {
                REQUIRE(image.is_loaded());
                REQUIRE(image.dtype().code == c.code);
                REQUIRE(image.dtype().bits == c.bits);
                auto region = image.read_region({ 0, 0 }, { 1, 1 });
                REQUIRE(region.is_loaded());
            }
        }
    }

    GIVEN("A 4D NIfTI time-series")
    {
        // A non-unit time dimension takes the ndim==5 (TZYXC) branch in cumed.
        const std::string path = medical_fixture_or_skip("generated/medical/vol_4d.nii");
        cucim::CuImage image(path);
        THEN("The temporal dimension is exposed")
        {
            REQUIRE(image.is_loaded());
            REQUIRE(image.dims() == "TZYXC");
            REQUIRE(image.ndim() >= 4);
            auto region = image.read_region({ 0, 0 }, { 1, 1 });
            REQUIRE(region.is_loaded());
        }
    }

    GIVEN("A qform-only NIfTI (sform disabled)")
    {
        // sform_code==0, qform_code>0 -> direction rebuilt from the quaternion.
        const std::string path = medical_fixture_or_skip("generated/medical/vol_qform.nii");
        cucim::CuImage image(path);
        THEN("The quaternion-derived orientation is exposed")
        {
            REQUIRE(image.is_loaded());
            const auto direction = image.direction();
            REQUIRE(direction.size() >= 3);
            auto region = image.read_region({ 0, 0 }, { 1, 1 });
            REQUIRE(region.is_loaded());
        }
    }
}


