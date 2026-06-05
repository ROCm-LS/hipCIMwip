/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests for the JPEGTables-merge helpers. These exercise the
 * host-side byte-splicing logic that RocJpegProcessor uses to turn an
 * abbreviated Aperio SVS tile into a self-contained JPEG before handing
 * it to rocJpegStreamParse. Pure-CPU tests — no GPU / no rocJPEG runtime
 * required.
 */
#include "cuslide/loader/rocjpeg_jpegtables.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

using cuslide::loader::detail::jpegtable_prefix_length;
using cuslide::loader::detail::plan_merged_jpeg_size;
using cuslide::loader::detail::splice_jpegtable_prefix;

namespace
{

// Minimal well-formed JPEGTables payload: SOI + 2 placeholder DQT/DHT bytes + EOI.
// Per JPEG B.5: [FF D8] [tables...] [FF D9]. Exact table contents do not
// matter to the splicing logic — only the framing markers do.
std::vector<std::uint8_t> make_jpegtable(std::size_t body_bytes)
{
    std::vector<std::uint8_t> t;
    t.reserve(4 + body_bytes);
    t.push_back(0xFF); t.push_back(0xD8);              // SOI
    for (std::size_t i = 0; i < body_bytes; ++i)
    {
        t.push_back(static_cast<std::uint8_t>(0xA0 + (i & 0x0F)));
    }
    t.push_back(0xFF); t.push_back(0xD9);              // EOI
    return t;
}

// Minimal abbreviated tile: SOI + payload + EOI. The payload stands in for
// [SOF][SOS][entropy-coded scan]; only the framing matters for the splice.
std::vector<std::uint8_t> make_abbreviated_tile(std::size_t payload_bytes)
{
    std::vector<std::uint8_t> tile;
    tile.reserve(4 + payload_bytes);
    tile.push_back(0xFF); tile.push_back(0xD8);        // SOI
    for (std::size_t i = 0; i < payload_bytes; ++i)
    {
        tile.push_back(static_cast<std::uint8_t>(i & 0xFF));
    }
    tile.push_back(0xFF); tile.push_back(0xD9);        // EOI
    return tile;
}

} // namespace

TEST_CASE("jpegtable_prefix_length: typical SVS table is shortened by 2 bytes",
          "[rocjpeg][jpegtables]")
{
    const auto t = make_jpegtable(8);
    REQUIRE(jpegtable_prefix_length(t.data(), t.size()) == t.size() - 2);
}

TEST_CASE("jpegtable_prefix_length: degenerate inputs return 0",
          "[rocjpeg][jpegtables]")
{
    const std::uint8_t dummy[4] = {0xFF, 0xD8, 0xFF, 0xD9};

    // No IFD JPEGTables tag → no prefix, fall through to pass-through.
    REQUIRE(jpegtable_prefix_length(nullptr, 0) == 0);
    REQUIRE(jpegtable_prefix_length(nullptr, 16) == 0);

    // Below the minimum [SOI][EOI] envelope of 4 bytes.
    REQUIRE(jpegtable_prefix_length(dummy, 0) == 0);
    REQUIRE(jpegtable_prefix_length(dummy, 1) == 0);
    REQUIRE(jpegtable_prefix_length(dummy, 2) == 0);
    REQUIRE(jpegtable_prefix_length(dummy, 3) == 0);

    // Boundary: exactly 4 bytes is the smallest legal table.
    REQUIRE(jpegtable_prefix_length(dummy, 4) == 2);
}

TEST_CASE("plan_merged_jpeg_size: happy path returns prefix_len + tile_size - 2",
          "[rocjpeg][jpegtables]")
{
    const auto table = make_jpegtable(16);
    const auto tile  = make_abbreviated_tile(64);

    const std::size_t prefix_len = jpegtable_prefix_length(table.data(), table.size());
    REQUIRE(prefix_len == table.size() - 2);

    const std::size_t slot_cap   = prefix_len + tile.size();  // ample
    const std::size_t merged_len = plan_merged_jpeg_size(prefix_len, tile.data(),
                                                         tile.size(), slot_cap);
    REQUIRE(merged_len == prefix_len + tile.size() - 2);
}

TEST_CASE("plan_merged_jpeg_size: returns 0 when no prefix is configured",
          "[rocjpeg][jpegtables]")
{
    // prefix_len == 0 means the IFD had no JPEGTables tag; tiles must be
    // passed through unchanged regardless of their framing.
    const auto tile = make_abbreviated_tile(64);
    REQUIRE(plan_merged_jpeg_size(0, tile.data(), tile.size(), /*slot_cap=*/4096) == 0);
}

TEST_CASE("plan_merged_jpeg_size: tile not starting with SOI is passed through",
          "[rocjpeg][jpegtables]")
{
    // Constructor docstring requires the splice path only triggers on tiles
    // beginning with FF D8. Anything else has to be handed to
    // rocJpegStreamParse unmodified so the error surfaces there.
    std::uint8_t bad_tile[8] = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03};
    REQUIRE(plan_merged_jpeg_size(/*prefix_len=*/8, bad_tile, sizeof(bad_tile),
                                  /*slot_cap=*/4096) == 0);

    // Tiles shorter than 2 bytes never have a complete SOI marker.
    std::uint8_t tiny[1] = {0xFF};
    REQUIRE(plan_merged_jpeg_size(8, tiny, 1, 4096) == 0);
    REQUIRE(plan_merged_jpeg_size(8, tiny, 0, 4096) == 0);
    REQUIRE(plan_merged_jpeg_size(8, nullptr, 0, 4096) == 0);
}

TEST_CASE("plan_merged_jpeg_size: refuses to overflow the slot capacity",
          "[rocjpeg][jpegtables]")
{
    const auto table = make_jpegtable(32);  // prefix_len = 34
    const auto tile  = make_abbreviated_tile(128);
    const std::size_t prefix_len = jpegtable_prefix_length(table.data(), table.size());
    const std::size_t merged_required = prefix_len + tile.size() - 2;

    // One byte too small → must refuse.
    REQUIRE(plan_merged_jpeg_size(prefix_len, tile.data(), tile.size(),
                                  merged_required - 1) == 0);

    // Exactly the required size → must accept (boundary, not overflow).
    REQUIRE(plan_merged_jpeg_size(prefix_len, tile.data(), tile.size(),
                                  merged_required) == merged_required);
}

TEST_CASE("splice_jpegtable_prefix: produces a well-framed merged JPEG",
          "[rocjpeg][jpegtables]")
{
    const auto table = make_jpegtable(12);
    const auto tile  = make_abbreviated_tile(48);
    const std::size_t prefix_len = jpegtable_prefix_length(table.data(), table.size());
    const std::size_t slot_cap   = prefix_len + tile.size();
    const std::size_t merged_len = plan_merged_jpeg_size(prefix_len, tile.data(),
                                                         tile.size(), slot_cap);
    REQUIRE(merged_len > 0);

    std::vector<std::uint8_t> out(slot_cap, 0xCC);
    splice_jpegtable_prefix(table.data(), prefix_len,
                            tile.data(), tile.size(), out.data());

    // Merged stream starts with the JPEGTables prefix (SOI + DQT/DHT, no EOI).
    REQUIRE(out[0] == 0xFF);
    REQUIRE(out[1] == 0xD8);
    REQUIRE(std::memcmp(out.data(), table.data(), prefix_len) == 0);

    // The EOI that originally terminated the JPEGTables payload must not
    // appear at the splice boundary — that is the whole point of dropping
    // the last 2 bytes in jpegtable_prefix_length(). Parenthesize the &&
    // so Catch2's expression decomposer doesn't reject it as chained.
    REQUIRE_FALSE((out[prefix_len - 2] == 0xFF && out[prefix_len - 1] == 0xD9));

    // Tile body is appended starting at tile[2] (skipping its own SOI).
    REQUIRE(std::memcmp(out.data() + prefix_len, tile.data() + 2, tile.size() - 2) == 0);

    // Tile's terminating EOI must survive intact at the end of the merged stream.
    REQUIRE(out[merged_len - 2] == 0xFF);
    REQUIRE(out[merged_len - 1] == 0xD9);

    // Beyond the merged region the caller's buffer is untouched.
    for (std::size_t i = merged_len; i < out.size(); ++i)
    {
        REQUIRE(out[i] == 0xCC);
    }
}

TEST_CASE("splice_jpegtable_prefix: minimum-size table + minimum-size tile",
          "[rocjpeg][jpegtables]")
{
    // Smallest legal SVS abbreviated stream we can splice: 4-byte table
    // (SOI+EOI only — prefix is just the SOI) and a 4-byte tile (SOI+EOI).
    // The resulting merged stream is just SOI+EOI: 2 + (4 - 2) = 4 bytes.
    const std::uint8_t table[4] = {0xFF, 0xD8, 0xFF, 0xD9};
    const std::uint8_t tile[4]  = {0xFF, 0xD8, 0xFF, 0xD9};

    const std::size_t prefix_len = jpegtable_prefix_length(table, sizeof(table));
    REQUIRE(prefix_len == 2);

    const std::size_t merged_len = plan_merged_jpeg_size(prefix_len, tile,
                                                         sizeof(tile),
                                                         /*slot_cap=*/8);
    REQUIRE(merged_len == 4);

    std::uint8_t out[8] = {};
    splice_jpegtable_prefix(table, prefix_len, tile, sizeof(tile), out);
    const std::uint8_t expected[4] = {0xFF, 0xD8, 0xFF, 0xD9};
    REQUIRE(std::memcmp(out, expected, 4) == 0);
}
