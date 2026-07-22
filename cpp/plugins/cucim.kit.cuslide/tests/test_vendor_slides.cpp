/*
 * SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>
#include <sys/stat.h>

#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include <cucim/memory/memory_manager.h>

#include "config.h"
#include "cuslide/tiff/tiff.h"

// Reads real OpenSlide vendor slides (fetched by download_test_data.sh) to cover
// tiff.cpp/ifd.cpp paths the generic fixtures cannot reach: Aperio SVS metadata
// + JPEG/JPEG2000 (33003/33005) decode, and the Philips XML parser. Slides are
// large and not committed, so each case SKIPs when its file is absent.

namespace
{

bool file_exists(const std::string& path)
{
    struct stat buffer;
    return stat(path.c_str(), &buffer) == 0;
}

// Construct IFDs (parse vendor metadata) and decode a level-0 region.
void read_slide_region(const std::string& path, int64_t width, int64_t height)
{
    auto tif = std::make_shared<cuslide::tiff::TIFF>(path.c_str(), O_RDONLY);
    tif->construct_ifds();

    cucim::io::format::ImageMetadata metadata{};
    cucim::io::format::ImageReaderRegionRequestDesc request{};
    cucim::io::format::ImageDataDesc image_data{};

    metadata.level_count(1).level_downsamples({ 1.0 }).level_ndim(3);

    int64_t request_location[2] = { 0, 0 };
    request.location = request_location;
    request.level = 0;
    int64_t request_size[2] = { width, height };
    request.size = request_size;
    request.device = const_cast<char*>("cpu");

    tif->read(&metadata.desc(), &request, &image_data);
    REQUIRE(image_data.container.data != nullptr);

    // Associated images (label/macro/thumbnail): best-effort into a separate
    // buffer; ignore slides that lack a given image.
    for (const char* name : { "label", "macro", "thumbnail" })
    {
        try
        {
            cucim::io::format::ImageDataDesc assoc_data{};
            request.associated_image_name = const_cast<char*>(name);
            tif->read(&metadata.desc(), &request, &assoc_data, nullptr /*out_metadata*/);
        }
        catch (...)
        {
        }
    }
    request.associated_image_name = nullptr;

    tif->close();

    REQUIRE(image_data.container.data != nullptr);
}

} // namespace

TEST_CASE("Read an Aperio JPEG SVS slide", "[test_vendor_slides.cpp]")
{
    const std::string path = g_config.get_input_path("private/CMU-1-Small-Region.svs");
    if (!file_exists(path))
    {
        SKIP("Vendor test file private/CMU-1-Small-Region.svs not available - skipping");
    }
    read_slide_region(path, 256, 256);
}

TEST_CASE("Read an Aperio JPEG 2000 SVS slide", "[test_vendor_slides.cpp]")
{
    const std::string path = g_config.get_input_path("private/JP2K-33003-1.svs");
    if (!file_exists(path))
    {
        SKIP("Vendor test file private/JP2K-33003-1.svs not available - skipping");
    }
    // 512x512 spans several tiles -> multiple 33003 (YCbCr) decodes.
    read_slide_region(path, 512, 512);
}

TEST_CASE("Read a Philips TIFF slide", "[test_vendor_slides.cpp]")
{
    const std::string path = g_config.get_input_path("private/Philips-1.tiff");
    if (!file_exists(path))
    {
        SKIP("Vendor test file private/Philips-1.tiff not available - skipping");
    }
    // construct_ifds() parses the Philips XML -> parse_philips_tiff_metadata().
    read_slide_region(path, 512, 512);
}
