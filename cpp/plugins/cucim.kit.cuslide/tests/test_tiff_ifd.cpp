/*
 * SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>

#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "config.h"
#include "cuslide/tiff/ifd.h"
#include "cuslide/tiff/tiff.h"

// Opens the generated tiled TIFF directly to cover the TIFF construction/config
// accessors and IFD metadata getters that the Python read path does not reach.

TEST_CASE("TIFF construction and configuration accessors", "[test_tiff_ifd.cpp]")
{
    const std::string input_path = g_config.get_input_path();

    SECTION("static open() overloads")
    {
        auto tif = cuslide::tiff::TIFF::open(input_path.c_str(), O_RDONLY);
        REQUIRE(tif != nullptr);
        tif->close();

        auto tif_cfg =
            cuslide::tiff::TIFF::open(input_path.c_str(), O_RDONLY, cuslide::tiff::TIFF::kUseLibTiff);
        REQUIRE(tif_cfg != nullptr);
        REQUIRE(tif_cfg->is_in_read_config(cuslide::tiff::TIFF::kUseLibTiff));
        tif_cfg->close();
    }

    SECTION("three-argument constructor and config mutation")
    {
        auto tif = std::make_shared<cuslide::tiff::TIFF>(
            input_path.c_str(), O_RDONLY, cuslide::tiff::TIFF::kUseDirectJpegTurbo);
        REQUIRE(tif != nullptr);

        // read_config() reflects the config passed at construction.
        REQUIRE((tif->read_config() & cuslide::tiff::TIFF::kUseDirectJpegTurbo) != 0);

        // add_read_config() ORs in additional flags.
        tif->add_read_config(cuslide::tiff::TIFF::kUseLibTiff);
        REQUIRE(tif->is_in_read_config(cuslide::tiff::TIFF::kUseLibTiff));
        REQUIRE((tif->read_config() & cuslide::tiff::TIFF::kUseLibTiff) != 0);

        // Generated stripe TIFF is a little-endian generic TIFF.
        REQUIRE_FALSE(tif->is_big_endian());
        REQUIRE(tif->tiff_type() == cuslide::tiff::TiffType::Generic);
    }
}

TEST_CASE("IFD metadata accessors", "[test_tiff_ifd.cpp]")
{
    const std::string input_path = g_config.get_input_path();

    auto tif = std::make_shared<cuslide::tiff::TIFF>(input_path.c_str(), O_RDONLY);
    tif->construct_ifds();

    REQUIRE(tif->ifd_count() >= 1);

    // ifd_offsets() is populated by construct_ifds().
    const std::vector<cuslide::tiff::ifd_offset_t>& offsets = tif->ifd_offsets();
    REQUIRE(offsets.size() == tif->ifd_count());

    std::shared_ptr<cuslide::tiff::IFD> ifd = tif->ifd(0);
    REQUIRE(ifd != nullptr);

    SECTION("scalar getters")
    {
        REQUIRE(ifd->index() == 0);
        REQUIRE(ifd->offset() == offsets[0]);
        REQUIRE(ifd->hash_value() != 0);
        REQUIRE(ifd->bits_per_sample() > 0);
        REQUIRE(ifd->planar_config() >= 1);
        REQUIRE(ifd->predictor() >= 1);

        // rows_per_strip is 0 for tiled TIFFs; exercise the accessor.
        (void)ifd->rows_per_strip();
    }

    SECTION("sub-IFD accessors")
    {
        const uint16_t subifd_count = ifd->subifd_count();
        std::vector<uint64_t>& subifd_offsets = ifd->subifd_offsets();
        REQUIRE(subifd_offsets.size() == subifd_count);
    }

    SECTION("image-piece (tile) accessors")
    {
        const uint32_t piece_count = ifd->image_piece_count();
        REQUIRE(piece_count > 0);

        const std::vector<uint64_t>& piece_offsets = ifd->image_piece_offsets();
        const std::vector<uint64_t>& piece_bytecounts = ifd->image_piece_bytecounts();
        REQUIRE(piece_offsets.size() == piece_count);
        REQUIRE(piece_bytecounts.size() == piece_count);
    }
}
