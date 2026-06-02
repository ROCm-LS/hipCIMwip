// SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Unit tests for the IFD-index ordering. Locks down the
 * "width desc, height desc, file-index asc" tiebreak so it can't drift
 * back to std::sort's unspecified ordering across equal keys.
 */
#include "cuslide/tiff/ifd_sort.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

using cuslide::tiff::detail::ifd_index_less;

namespace
{

struct Fixture
{
    std::vector<std::uint32_t> widths;
    std::vector<std::uint32_t> heights;
    auto width()  const { return [this](std::size_t i) { return widths[i]; }; }
    auto height() const { return [this](std::size_t i) { return heights[i]; }; }
};

// Sort indices [0..N) by the comparator under test and return the result.
std::vector<std::size_t> sorted_indices(const Fixture& f)
{
    std::vector<std::size_t> idx(f.widths.size());
    for (std::size_t i = 0; i < idx.size(); ++i)
    {
        idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(), [&f](std::size_t a, std::size_t b) {
        return ifd_index_less(a, b, f.width(), f.height());
    });
    return idx;
}

} // namespace

TEST_CASE("ifd_sort: Aperio-style pyramid (descending widths) is unchanged",
          "[cuslide][ifd_sort]")
{
    // Typical SVS: full res + 4 thumbnails at /2, /4, /16, /32. Already in
    // descending order by file index, must stay that way.
    Fixture f{{32000, 16000, 8000, 2000, 1000},
              {24000, 12000, 6000, 1500,  750}};
    std::vector<std::size_t> expected{0, 1, 2, 3, 4};
    REQUIRE(sorted_indices(f) == expected);
}

TEST_CASE("ifd_sort: shuffled pyramid lands largest-first",
          "[cuslide][ifd_sort]")
{
    // Same set as above but in reverse file order — must sort to put the
    // 32000×24000 IFD first.
    Fixture f{{ 1000,  2000,  8000, 16000, 32000},
              {  750,  1500,  6000, 12000, 24000}};
    std::vector<std::size_t> expected{4, 3, 2, 1, 0};
    REQUIRE(sorted_indices(f) == expected);
}

TEST_CASE("ifd_sort: same H×W → file-order ascending (tiebreak)",
          "[cuslide][ifd_sort]")
{
    // OME-TIFF / BBBC pattern: every IFD same 696×520. Without the
    // file-index tiebreak the order is std::sort-implementation-defined.
    // With it, indices come out in their original file order.
    Fixture f{{696, 696, 696, 696, 696},
              {520, 520, 520, 520, 520}};
    std::vector<std::size_t> expected{0, 1, 2, 3, 4};
    REQUIRE(sorted_indices(f) == expected);
}

TEST_CASE("ifd_sort: mixed — pyramid base ties at full res, then descends",
          "[cuslide][ifd_sort]")
{
    // 3 full-res same-size IFDs (e.g. 3-channel OME-TIFF) plus 2 downsamples.
    // Expect file-order across the full-res tie, then descending pyramid.
    Fixture f{{8000, 8000, 8000, 4000, 2000},
              {6000, 6000, 6000, 3000, 1500}};
    std::vector<std::size_t> expected{0, 1, 2, 3, 4};
    REQUIRE(sorted_indices(f) == expected);
}

TEST_CASE("ifd_sort: same width, different heights — height descending",
          "[cuslide][ifd_sort]")
{
    // Width tie alone — height should be the secondary key.
    Fixture f{{8000, 8000, 8000},
              {2000, 8000, 4000}};
    std::vector<std::size_t> expected{1, 2, 0};   // h: 8000, 4000, 2000
    REQUIRE(sorted_indices(f) == expected);
}

TEST_CASE("ifd_sort: comparator is strict weak ordering",
          "[cuslide][ifd_sort]")
{
    // Irreflexive: !comp(x, x). Asymmetric: comp(a,b) implies !comp(b,a).
    Fixture f{{100, 100, 200},
              {100, 200, 100}};
    auto w = f.width();
    auto h = f.height();
    for (std::size_t i = 0; i < 3; ++i)
    {
        REQUIRE_FALSE(ifd_index_less(i, i, w, h));
        for (std::size_t j = 0; j < 3; ++j)
        {
            if (i == j) continue;
            if (ifd_index_less(i, j, w, h))
            {
                REQUIRE_FALSE(ifd_index_less(j, i, w, h));
            }
        }
    }
}

TEST_CASE("ifd_sort: single IFD is trivially sorted",
          "[cuslide][ifd_sort]")
{
    Fixture f{{1024}, {768}};
    std::vector<std::size_t> expected{0};
    REQUIRE(sorted_indices(f) == expected);
}
