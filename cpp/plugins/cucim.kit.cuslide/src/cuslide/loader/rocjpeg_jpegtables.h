/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Pure helpers for splicing an IFD's JPEGTables prefix into an
 * abbreviated per-tile JPEG. Kept dependency-free (no ROCm / no GPU /
 * no rocJPEG headers) so it can be exercised in unit tests on any host.
 *
 * Background. Aperio SVS and many other tiled-JPEG TIFFs use abbreviated
 * JPEG streams: the quantization (DQT) and Huffman (DHT) tables live in
 * the IFD's JPEGTables tag (0x015B) once for the IFD, and the per-tile
 * bitstream contains only [SOI][SOF][SOS][scan][EOI]. rocJPEG has no
 * equivalent of nvJPEG's nvjpegDecodeBatchedParseJpegTables(), so we
 * pre-merge per tile before calling rocJpegStreamParse:
 *
 *   merged = [SOI][DQT/DHT...]  ++  [tile bytes after the SOI marker]
 *
 * The JPEGTables payload itself follows the standard JPEG abbreviated
 * format from ITU-T T.81 B.5: [SOI 0xFF 0xD8] [DQT/DHT ...] [EOI 0xFF 0xD9].
 * We keep bytes [0, size-2) — SOI through the end of DHT — and drop the
 * trailing EOI so it does not appear in the middle of the merged stream.
 */
#ifndef CUSLIDE_LOADER_ROCJPEG_JPEGTABLES_H
#define CUSLIDE_LOADER_ROCJPEG_JPEGTABLES_H

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cuslide::loader::detail
{

/**
 * Number of bytes from `jpegtable_data` to use as the merged-stream prefix.
 *
 * Returns 0 (no merge needed) when:
 *   - `jpegtable_data` is null, OR
 *   - `jpegtable_size` < 4 (smaller than the minimum SOI+EOI envelope).
 *
 * Otherwise returns `jpegtable_size - 2` — the input with its trailing EOI
 * marker dropped.
 */
inline std::size_t jpegtable_prefix_length(const std::uint8_t* jpegtable_data,
                                           std::size_t jpegtable_size)
{
    if (jpegtable_data == nullptr || jpegtable_size < 4)
    {
        return 0;
    }
    return jpegtable_size - 2;
}

/**
 * Plan a merge of `prefix_len` bytes of an IFD JPEGTables prefix with a
 * single abbreviated tile.
 *
 * Returns the size of the merged stream that would be written into a slot
 * of `slot_cap` bytes, or 0 to signal "do not merge, pass the tile through
 * unchanged" when any of these holds:
 *   - prefix_len == 0          (no IFD JPEGTables, tile is already complete)
 *   - tile_size < 2            (not enough bytes for the leading SOI marker)
 *   - tile does not start with 0xFF 0xD8  (not an SOI; tile bitstream is
 *     malformed or non-abbreviated, defer to rocJpegStreamParse to diagnose)
 *   - prefix_len + tile_size - 2 > slot_cap  (would overflow the per-slot
 *     arena allocated at construction)
 */
inline std::size_t plan_merged_jpeg_size(std::size_t prefix_len,
                                         const std::uint8_t* tile_data,
                                         std::size_t tile_size,
                                         std::size_t slot_cap)
{
    if (prefix_len == 0)
    {
        return 0;
    }
    if (tile_data == nullptr || tile_size < 2)
    {
        return 0;
    }
    if (tile_data[0] != 0xFF || tile_data[1] != 0xD8)
    {
        return 0;
    }
    const std::size_t merged_len = prefix_len + tile_size - 2;
    if (merged_len > slot_cap)
    {
        return 0;
    }
    return merged_len;
}

/**
 * Splice [prefix] ++ [tile bytes after the SOI marker] into `out`.
 *
 * Caller must ensure that `plan_merged_jpeg_size(prefix_len, tile_data,
 * tile_size, slot_cap)` returned a non-zero value before calling — this
 * function performs the copy unconditionally.
 */
inline void splice_jpegtable_prefix(const std::uint8_t* prefix_data,
                                    std::size_t prefix_len,
                                    const std::uint8_t* tile_data,
                                    std::size_t tile_size,
                                    std::uint8_t* out)
{
    std::memcpy(out, prefix_data, prefix_len);
    std::memcpy(out + prefix_len, tile_data + 2, tile_size - 2);
}

} // namespace cuslide::loader::detail

#endif // CUSLIDE_LOADER_ROCJPEG_JPEGTABLES_H
