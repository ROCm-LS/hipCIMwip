/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucim/memory/memory_manager.h>
#include <openslide/openslide.h>
#include "cuslide/tiff/tiff.h"
#include "config.h"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <sys/stat.h>

TEST_CASE("Verify philips tiff file", "[generic_tiff_000.tif]")
{
    fmt::print("Read Philips TIFF File\n");

    std::string file_path = g_config.get_input_path("private/generic_tiff_000.tif");

    // Skip test if vendor test file is not available
    struct stat buffer;
    if (stat(file_path.c_str(), &buffer) != 0) {
        SKIP("Vendor test file generic_tiff_000.tif not available - skipping test");
    }

    auto tif = std::make_shared<cuslide::tiff::TIFF>(file_path.c_str(),
                                                     O_RDONLY); // , cuslide::tiff::TIFF::kUseLibTiff
    tif->construct_ifds();

    int64_t test_sx = 1;
    int64_t test_sy = 1;

    int64_t test_width = 500;
    int64_t test_height = 500;

    cucim::io::format::ImageMetadata metadata{};
    cucim::io::format::ImageReaderRegionRequestDesc request{};
    cucim::io::format::ImageDataDesc image_data{};

    metadata.level_count(1).level_downsamples({ 1.0 }).level_ndim(3);

    int64_t request_location[2] = { test_sx, test_sy };
    request.location = request_location;
    request.level = 0;
    int64_t request_size[2] = { test_width, test_height };
    request.size = request_size;
    request.device = const_cast<char*>("cuda");

    tif->read(&metadata.desc(), &request, &image_data);

    request.associated_image_name = const_cast<char*>("label");
    tif->read(&metadata.desc(), &request, &image_data, nullptr /*out_metadata*/);

    tif->close();

    REQUIRE(1 == 1);
}
