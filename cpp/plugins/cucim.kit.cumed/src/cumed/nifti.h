// SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// NIfTI-1 header parsing and voxel loading helpers for the hipCIM cumed plugin.
// Supports .nii (plain) and .nii.gz (gzip-compressed single-file NIfTI).
// Decompression uses libdeflate (already vendored in the tree via cuslide).

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <fmt/format.h>
#include <dlpack/dlpack.h>
#include <libdeflate.h>

namespace cumed::nifti
{

// ── NIfTI-1 header (348 bytes, fixed layout per the NIfTI-1 standard) ────────
#pragma pack(push, 1)
struct Nifti1Header
{
    int32_t  sizeof_hdr;    // 0   Must be 348
    char     data_type[10]; // 4   unused in NIfTI-1
    char     db_name[18];   // 14  unused in NIfTI-1
    int32_t  extents;       // 32  unused in NIfTI-1
    int16_t  session_error; // 36  unused in NIfTI-1
    char     regular;       // 38  unused in NIfTI-1
    char     dim_info;      // 39  MRI phase/frequency/slice encoding
    int16_t  dim[8];        // 40  dim[0]=ndims; dim[1..7]=sizes
    float    intent_p1;     // 56
    float    intent_p2;     // 60
    float    intent_p3;     // 64
    int16_t  intent_code;   // 68
    int16_t  datatype;      // 70  data type code
    int16_t  bitpix;        // 72  bits per voxel
    int16_t  slice_start;   // 74
    float    pixdim[8];     // 76  pixdim[0]=qfac; pixdim[1..7]=voxel sizes
    float    vox_offset;    // 108 byte offset to voxel data
    float    scl_slope;     // 112 intensity scale factor
    float    scl_inter;     // 116 intensity intercept
    int16_t  slice_end;     // 120
    char     slice_code;    // 122
    char     xyzt_units;    // 123 spatial + temporal units
    float    cal_max;       // 124
    float    cal_min;       // 128
    float    slice_duration;// 132
    float    toffset;       // 136
    int32_t  glmax;         // 140 unused in NIfTI-1
    int32_t  glmin;         // 144 unused in NIfTI-1
    char     descrip[80];   // 148 any text
    char     aux_file[24];  // 228
    int16_t  qform_code;    // 252 orientation: 0=unknown, >0=method 2
    int16_t  sform_code;    // 254 orientation: 0=unknown, >0=method 3
    float    quatern_b;     // 256
    float    quatern_c;     // 260
    float    quatern_d;     // 264
    float    qoffset_x;     // 268
    float    qoffset_y;     // 272
    float    qoffset_z;     // 276
    float    srow_x[4];     // 280
    float    srow_y[4];     // 296
    float    srow_z[4];     // 312
    char     intent_name[16];// 328
    char     magic[4];      // 344  "n+1\0" or "ni1\0"
};
#pragma pack(pop)
static_assert(sizeof(Nifti1Header) == 348, "Nifti1Header must be exactly 348 bytes");

// ── NIfTI datatype codes → DLDataType ────────────────────────────────────────
static inline DLDataType nifti_datatype_to_dl(int16_t datatype)
{
    switch (datatype)
    {
    case 2:   return {kDLUInt,  8,  1}; // UINT8
    case 4:   return {kDLInt,   16, 1}; // INT16
    case 8:   return {kDLInt,   32, 1}; // INT32
    case 16:  return {kDLFloat, 32, 1}; // FLOAT32
    case 64:  return {kDLFloat, 64, 1}; // FLOAT64
    case 256: return {kDLInt,   8,  1}; // INT8
    case 512: return {kDLUInt,  16, 1}; // UINT16
    case 768: return {kDLUInt,  32, 1}; // UINT32
    case 1024:return {kDLInt,   64, 1}; // INT64
    case 1280:return {kDLUInt,  64, 1}; // UINT64
    default:
        throw std::runtime_error(fmt::format("Unsupported NIfTI datatype: {}", datatype));
    }
}

// ── Byte-swap helpers (for big-endian NIfTI files) ───────────────────────────
static inline int16_t bswap16(int16_t v) {
    return static_cast<int16_t>(__builtin_bswap16(static_cast<uint16_t>(v)));
}
static inline int32_t bswap32(int32_t v) {
    return static_cast<int32_t>(__builtin_bswap32(static_cast<uint32_t>(v)));
}
static inline float bswap_float(float v) {
    uint32_t u; memcpy(&u, &v, 4); u = __builtin_bswap32(u); float r; memcpy(&r, &u, 4); return r;
}

static inline void byteswap_header(Nifti1Header& h)
{
    h.sizeof_hdr = bswap32(h.sizeof_hdr);
    for (int i=0;i<8;i++) h.dim[i] = bswap16(h.dim[i]);
    h.datatype   = bswap16(h.datatype);
    h.bitpix     = bswap16(h.bitpix);
    for (int i=0;i<8;i++) h.pixdim[i] = bswap_float(h.pixdim[i]);
    h.vox_offset = bswap_float(h.vox_offset);
    h.scl_slope  = bswap_float(h.scl_slope);
    h.scl_inter  = bswap_float(h.scl_inter);
    h.qform_code = bswap16(h.qform_code);
    h.sform_code = bswap16(h.sform_code);
    h.quatern_b  = bswap_float(h.quatern_b);
    h.quatern_c  = bswap_float(h.quatern_c);
    h.quatern_d  = bswap_float(h.quatern_d);
    h.qoffset_x  = bswap_float(h.qoffset_x);
    h.qoffset_y  = bswap_float(h.qoffset_y);
    h.qoffset_z  = bswap_float(h.qoffset_z);
    for (int i=0;i<4;i++) { h.srow_x[i]=bswap_float(h.srow_x[i]);
                             h.srow_y[i]=bswap_float(h.srow_y[i]);
                             h.srow_z[i]=bswap_float(h.srow_z[i]); }
}

// ── File content loader (returns full file bytes) ─────────────────────────────
static inline std::vector<uint8_t> read_file_bytes(int fd)
{
    struct stat sb;
    fstat(fd, &sb);
    size_t fsz = static_cast<size_t>(sb.st_size);
    std::vector<uint8_t> buf(fsz);
    size_t got = 0;
    while (got < fsz) {
        ssize_t r = ::read(fd, buf.data() + got, fsz - got);
        if (r <= 0) throw std::runtime_error("NIfTI: read error");
        got += r;
    }
    return buf;
}

// ── gzip decompressor using libdeflate ────────────────────────────────────────
static inline std::vector<uint8_t> gunzip(const std::vector<uint8_t>& compressed)
{
    struct libdeflate_decompressor* d = libdeflate_alloc_decompressor();
    if (!d) throw std::runtime_error("NIfTI: libdeflate_alloc_decompressor failed");

    // Grow output buffer until it fits — NIfTI volumes are a few hundred MB max
    std::vector<uint8_t> out(compressed.size() * 4);
    size_t actual = 0;
    libdeflate_result res = LIBDEFLATE_INSUFFICIENT_SPACE;
    while (res == LIBDEFLATE_INSUFFICIENT_SPACE)
    {
        out.resize(out.size() * 2);
        res = libdeflate_gzip_decompress(d, compressed.data(), compressed.size(),
                                          out.data(), out.size(), &actual);
    }
    libdeflate_free_decompressor(d);
    if (res != LIBDEFLATE_SUCCESS)
        throw std::runtime_error(fmt::format("NIfTI: gzip decompression failed ({})", static_cast<int>(res)));
    out.resize(actual);
    return out;
}

// ── Main parse result ─────────────────────────────────────────────────────────
struct NiftiInfo
{
    Nifti1Header hdr;
    bool         big_endian = false;
    bool         is_gz      = false;
    uint16_t     ndim       = 0;    // spatial dims (1-3, or 4 if time series)
    int64_t      nx=1, ny=1, nz=1, nt=1;
    size_t       vox_bytes  = 0;    // bytes per voxel
    size_t       data_offset= 0;    // byte offset to voxel data in decompressed stream
    std::vector<uint8_t> raw_bytes; // full decompressed file content
};

static inline NiftiInfo parse(int fd, const std::string& path)
{
    NiftiInfo info;
    bool gz = path.size() >= 7 &&
              path.compare(path.size()-7, 7, ".nii.gz") == 0;
    info.is_gz = gz;

    // Read raw file bytes
    lseek(fd, 0, SEEK_SET);
    std::vector<uint8_t> raw = read_file_bytes(fd);

    // Decompress if needed
    if (gz)
        info.raw_bytes = gunzip(raw);
    else
        info.raw_bytes = std::move(raw);

    if (info.raw_bytes.size() < 348)
        throw std::runtime_error("NIfTI: file too small for header");

    // Read header
    memcpy(&info.hdr, info.raw_bytes.data(), sizeof(Nifti1Header));

    // Detect endianness and byte-swap if needed
    if (info.hdr.sizeof_hdr != 348)
    {
        byteswap_header(info.hdr);
        info.big_endian = true;
        if (info.hdr.sizeof_hdr != 348)
            throw std::runtime_error("NIfTI: invalid sizeof_hdr (not 348)");
    }

    // Validate magic
    if (strncmp(info.hdr.magic, "n+1", 3) != 0 && strncmp(info.hdr.magic, "ni1", 3) != 0)
        throw std::runtime_error("NIfTI: invalid magic bytes");

    // Dimensionality
    int16_t hdr_ndim = info.hdr.dim[0];
    if (hdr_ndim < 1 || hdr_ndim > 7)
        throw std::runtime_error(fmt::format("NIfTI: invalid dim[0]={}", hdr_ndim));
    info.ndim = static_cast<uint16_t>(std::min(hdr_ndim, static_cast<int16_t>(4)));
    info.nx = (hdr_ndim >= 1) ? info.hdr.dim[1] : 1;
    info.ny = (hdr_ndim >= 2) ? info.hdr.dim[2] : 1;
    info.nz = (hdr_ndim >= 3) ? info.hdr.dim[3] : 1;
    info.nt = (hdr_ndim >= 4) ? info.hdr.dim[4] : 1;
    if (info.nx <= 0 || info.ny <= 0 || info.nz <= 0 || info.nt <= 0)
        throw std::runtime_error("NIfTI: zero or negative dimension");

    // Voxel size
    info.vox_bytes = static_cast<size_t>(info.hdr.bitpix) / 8;
    if (info.vox_bytes == 0)
        throw std::runtime_error(fmt::format("NIfTI: bitpix={} is invalid", info.hdr.bitpix));

    // Data offset
    info.data_offset = static_cast<size_t>(info.hdr.vox_offset);
    if (info.data_offset < 348) info.data_offset = 352; // NIfTI-1 spec minimum
    if (info.data_offset + info.nx * info.ny * info.nz * info.nt * info.vox_bytes > info.raw_bytes.size())
        throw std::runtime_error("NIfTI: voxel data extends beyond file");

    return info;
}

// ── Build spacing_units string from xyzt_units nibble ────────────────────────
static inline const char* spatial_unit_string(char xyzt_units)
{
    switch (xyzt_units & 0x07) // low 3 bits = spatial
    {
    case 1: return "meter";
    case 2: return "millimeter";
    case 3: return "micrometer";
    default: return "unknown";
    }
}

} // namespace cumed::nifti
