// SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Stable comparator for ordering IFD indices by descending resolution.
 * Pulled out of TIFF::construct_ifds() so the tiebreak behavior
 * (file-order on equal H×W) can be exercised in unit tests without
 * standing up a TIFF fixture.
 *
 * Width descending, then height descending, then IFD-index ascending.
 * The last key matters because many real-world multi-IFD
 * TIFFs (OME-TIFF channel splits / time-series, Bio-Formats BBBC plate
 * datasets) have every IFD at the same resolution. std::sort across
 * equal keys is unspecified ordering; using the original file index as
 * a tertiary key makes the level→IFD mapping deterministic across runs
 * and stdlib implementations.
 */
#ifndef CUSLIDE_TIFF_IFD_SORT_H
#define CUSLIDE_TIFF_IFD_SORT_H

#include <cstddef>
#include <cstdint>

namespace cuslide::tiff::detail
{

/**
 * Strict weak ordering on IFD indices a and b given accessors for
 * width(idx) and height(idx). Returns true iff a should come before b.
 *
 * Templated on the accessor types so unit tests can pass plain arrays
 * without dragging in the IFD class.
 */
template <typename WidthFn, typename HeightFn>
inline bool ifd_index_less(std::size_t a, std::size_t b, WidthFn width, HeightFn height)
{
    const std::uint32_t wa = width(a);
    const std::uint32_t wb = width(b);
    if (wa != wb)
    {
        return wa > wb;
    }
    const std::uint32_t ha = height(a);
    const std::uint32_t hb = height(b);
    if (ha != hb)
    {
        return ha > hb;
    }
    return a < b;
}

} // namespace cuslide::tiff::detail

#endif // CUSLIDE_TIFF_IFD_SORT_H
