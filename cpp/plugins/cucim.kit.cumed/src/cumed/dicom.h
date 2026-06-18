// SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Minimal DICOM Phase 1 reader for the hipCIM cumed plugin.
// Supports: classic single-frame DICOM, uncompressed (Explicit/Implicit VR LE)
// and JPEG2000 / JPEG Baseline transfer syntaxes.
// Does NOT require DCMTK or GDCM — hand-rolled tag parser for the ~10 tags needed.

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <fmt/format.h>
#include <dlpack/dlpack.h>

namespace cumed::dicom
{

// ── Tag encoding helpers ──────────────────────────────────────────────────────
static inline uint32_t make_tag(uint16_t group, uint16_t element)
{
    return (static_cast<uint32_t>(group) << 16) | element;
}
// Known tags
static constexpr uint32_t TAG_META_LENGTH         = 0x00020000;
static constexpr uint32_t TAG_TRANSFER_SYNTAX_UID = 0x00020010;
static constexpr uint32_t TAG_ROWS                = 0x00280010;
static constexpr uint32_t TAG_COLUMNS             = 0x00280011;
static constexpr uint32_t TAG_SAMPLES_PER_PIXEL   = 0x00280002;
static constexpr uint32_t TAG_BITS_ALLOCATED      = 0x00280100;
static constexpr uint32_t TAG_BITS_STORED         = 0x00280101;
static constexpr uint32_t TAG_PIXEL_REPRESENTATION= 0x00280103;
static constexpr uint32_t TAG_PHOTOMETRIC_INTERP  = 0x00280004;
static constexpr uint32_t TAG_NUMBER_OF_FRAMES    = 0x00280008;
static constexpr uint32_t TAG_PLANAR_CONFIGURATION= 0x00280006;
static constexpr uint32_t TAG_PIXEL_SPACING       = 0x00280030;
static constexpr uint32_t TAG_SLICE_THICKNESS     = 0x00180050;
static constexpr uint32_t TAG_IMAGE_POS_PATIENT   = 0x00200032;
static constexpr uint32_t TAG_IMAGE_ORIENT_PATIENT= 0x00200037;
static constexpr uint32_t TAG_RESCALE_INTERCEPT   = 0x00281052;
static constexpr uint32_t TAG_RESCALE_SLOPE       = 0x00281053;
static constexpr uint32_t TAG_PIXEL_DATA          = 0x7FE00010;
// Sequence delimiters
static constexpr uint32_t TAG_ITEM               = 0xFFFEE000;
static constexpr uint32_t TAG_ITEM_DELIM         = 0xFFFEE00D;
static constexpr uint32_t TAG_SEQ_DELIM          = 0xFFFEE0DD;
static constexpr uint32_t UNDEF_LENGTH           = 0xFFFFFFFF;

// ── VRs that use 4-byte length (with 2 reserved bytes) ───────────────────────
static inline bool vr_uses_4byte_length(const char vr[2])
{
    // OB, OD, OF, OL, OW, SQ, UC, UN, UR, UT
    static const char* list[] = {"OB","OD","OF","OL","OW","SQ","UC","UN","UR","UT",nullptr};
    for (int i=0; list[i]; ++i)
        if (vr[0]==list[i][0] && vr[1]==list[i][1]) return true;
    return false;
}

// ── Implicit VR mini-dictionary (group 0028 + 0020 tags we care about) ────────
static inline const char* implicit_vr(uint32_t tag)
{
    switch(tag) {
    case TAG_ROWS:                 return "US";
    case TAG_COLUMNS:              return "US";
    case TAG_SAMPLES_PER_PIXEL:   return "US";
    case TAG_BITS_ALLOCATED:       return "US";
    case TAG_BITS_STORED:          return "US";
    case TAG_PIXEL_REPRESENTATION: return "US";
    case TAG_PHOTOMETRIC_INTERP:   return "CS";
    case TAG_NUMBER_OF_FRAMES:     return "IS";
    case TAG_PLANAR_CONFIGURATION: return "US";
    case TAG_PIXEL_SPACING:        return "DS";
    case TAG_SLICE_THICKNESS:      return "DS";
    case TAG_IMAGE_POS_PATIENT:    return "DS";
    case TAG_IMAGE_ORIENT_PATIENT: return "DS";
    case TAG_RESCALE_INTERCEPT:    return "DS";
    case TAG_RESCALE_SLOPE:        return "DS";
    case TAG_PIXEL_DATA:           return "OW";
    default:                       return "UN";
    }
}

// ── Read helpers (little-endian) ──────────────────────────────────────────────
static inline uint16_t read_u16(const uint8_t* p) {
    uint16_t v; memcpy(&v, p, 2); return v;
}
static inline uint32_t read_u32(const uint8_t* p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}
static inline float parse_float_ds(const std::string& s) {
    try { return std::stof(s); } catch(...) { return 0.f; }
}
static std::vector<float> parse_ds_multi(const uint8_t* data, size_t len) {
    std::string s(reinterpret_cast<const char*>(data), len);
    std::vector<float> out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t end = s.find('\\', pos);
        if (end == std::string::npos) end = s.size();
        out.push_back(parse_float_ds(s.substr(pos, end-pos)));
        pos = end + 1;
    }
    return out;
}
static std::string parse_string(const uint8_t* data, size_t len) {
    std::string s(reinterpret_cast<const char*>(data), len);
    while (!s.empty() && (s.back()==' '||s.back()=='\0')) s.pop_back();
    return s;
}

// ── Transfer syntax categories ────────────────────────────────────────────────
enum class Compression { Raw, JpegBaseline, Jpeg2000, Unsupported };

static inline Compression classify_ts(const std::string& ts)
{
    if (ts == "1.2.840.10008.1.2" ||   // Implicit VR LE
        ts == "1.2.840.10008.1.2.1")    // Explicit VR LE
        return Compression::Raw;
    if (ts == "1.2.840.10008.1.2.4.50") // JPEG Baseline
        return Compression::JpegBaseline;
    if (ts == "1.2.840.10008.1.2.4.90" || // JP2K lossless
        ts == "1.2.840.10008.1.2.4.91")   // JP2K lossy
        return Compression::Jpeg2000;
    return Compression::Unsupported;
}

// ── Main parse result ─────────────────────────────────────────────────────────
struct DicomInfo
{
    // Pixel geometry
    uint16_t rows    = 0;
    uint16_t columns = 0;
    uint16_t samples = 1;
    uint16_t bits_alloc = 8;
    uint16_t bits_stored = 8;
    uint16_t pixel_repr = 0;  // 0=unsigned, 1=signed
    uint16_t planar_config = 0;
    int      n_frames = 1;
    std::string photometric;  // "MONOCHROME1","MONOCHROME2","RGB","YBR_FULL","YBR_FULL_422"

    // Transfer syntax / compression
    std::string transfer_syntax_uid;
    Compression compression = Compression::Raw;
    bool        implicit_vr = false;

    // Pixel data location (in raw_bytes)
    bool        pixel_data_encapsulated = false;
    size_t      pixel_data_offset = 0; // offset into raw_bytes for start of pixel value (raw case)
    size_t      pixel_data_length = 0; // raw case only

    // Encapsulated frames (compressed case)
    std::vector<std::pair<size_t,size_t>> frames; // (offset, length) into raw_bytes for each frame

    // Geometry (optional)
    float pixel_spacing[2] = {1.f, 1.f};  // row, col (mm)
    float slice_thickness  = 1.f;
    float image_pos[3]     = {0.f,0.f,0.f};
    float image_orient[6]  = {1,0,0,0,1,0};  // row cosines (3) + col cosines (3)
    float rescale_slope     = 1.f;
    float rescale_intercept = 0.f;

    std::vector<uint8_t> raw_bytes; // full file content
};

// ── File reader ───────────────────────────────────────────────────────────────
static inline std::vector<uint8_t> read_file_bytes(int fd)
{
    struct stat sb; fstat(fd, &sb);
    size_t fsz = static_cast<size_t>(sb.st_size);
    std::vector<uint8_t> buf(fsz);
    size_t got = 0;
    while (got < fsz) {
        ssize_t r = ::read(fd, buf.data()+got, fsz-got);
        if (r <= 0) throw std::runtime_error("DICOM: read error");
        got += r;
    }
    return buf;
}

// ── Main parser ───────────────────────────────────────────────────────────────
static inline DicomInfo parse(int fd)
{
    DicomInfo info;
    lseek(fd, 0, SEEK_SET);
    info.raw_bytes = read_file_bytes(fd);
    const uint8_t* data = info.raw_bytes.data();
    size_t size = info.raw_bytes.size();

    // Validate DICM magic
    if (size < 132 || memcmp(data+128, "DICM", 4) != 0)
        throw std::runtime_error("DICOM: missing DICM magic at byte 128");

    size_t pos = 132; // start of File Meta group

    // ── Parse File Meta group (always Explicit VR LE) ─────────────────────────
    // Read (0002,0000) MetaInformationGroupLength first
    if (pos + 12 > size) throw std::runtime_error("DICOM: truncated meta header");
    uint32_t meta_tag = make_tag(read_u16(data+pos), read_u16(data+pos+2));
    if (meta_tag != TAG_META_LENGTH) throw std::runtime_error("DICOM: first tag must be (0002,0000)");
    // skip VR "UL" + 2-byte len
    uint32_t meta_value_len = read_u32(data+pos+8);
    uint32_t meta_group_end = static_cast<uint32_t>(pos + 12 + 4 + meta_value_len); // overshoot ok
    // Read meta group length from value
    uint32_t meta_len = read_u32(data+pos+12);
    size_t meta_end = pos + 12 + 4 + meta_len; // actual end of meta group

    // Scan meta group for TransferSyntaxUID
    size_t p = pos;
    while (p < meta_end && p + 8 <= size) {
        uint32_t tag = make_tag(read_u16(data+p), read_u16(data+p+2));
        char vr[2] = {(char)data[p+4], (char)data[p+5]};
        size_t len, next;
        if (vr_uses_4byte_length(vr)) {
            len = read_u32(data+p+8); next = p+12;
        } else {
            len = read_u16(data+p+6); next = p+8;
        }
        if (tag == TAG_TRANSFER_SYNTAX_UID)
            info.transfer_syntax_uid = parse_string(data+next, len);
        p = next + len;
    }

    if (info.transfer_syntax_uid.empty())
        info.transfer_syntax_uid = "1.2.840.10008.1.2.1"; // default explicit VR LE
    info.compression = classify_ts(info.transfer_syntax_uid);
    info.implicit_vr = (info.transfer_syntax_uid == "1.2.840.10008.1.2");

    // ── Parse main dataset ────────────────────────────────────────────────────
    p = meta_end;
    bool found_pixel_data = false;

    while (p + 4 <= size && !found_pixel_data) {
        uint32_t tag = make_tag(read_u16(data+p), read_u16(data+p+2));

        // Sequence/item delimiters — skip
        if (tag == TAG_ITEM || tag == TAG_ITEM_DELIM || tag == TAG_SEQ_DELIM) {
            p += 8; continue;
        }

        size_t value_start;
        size_t len;

        if (info.implicit_vr) {
            // Implicit VR: group/element/len32/value
            if (p+8 > size) break;
            len = read_u32(data+p+4);
            value_start = p+8;
        } else {
            // Explicit VR
            if (p+8 > size) break;
            char vr[2] = {(char)data[p+4], (char)data[p+5]};
            if (vr_uses_4byte_length(vr)) {
                if (p+12 > size) break;
                len = read_u32(data+p+8);
                value_start = p+12;
            } else {
                len = read_u16(data+p+6);
                value_start = p+8;
            }
        }

        if (tag == TAG_PIXEL_DATA) {
            found_pixel_data = true;
            if (len == UNDEF_LENGTH) {
                // Compressed encapsulation
                info.pixel_data_encapsulated = true;
                size_t q = value_start;
                // Skip Basic Offset Table (first item)
                if (q+8 <= size && make_tag(read_u16(data+q), read_u16(data+q+2)) == TAG_ITEM) {
                    uint32_t bot_len = read_u32(data+q+4);
                    q += 8 + bot_len;
                }
                // Read frame items
                while (q+8 <= size) {
                    uint32_t item_tag = make_tag(read_u16(data+q), read_u16(data+q+2));
                    uint32_t item_len = read_u32(data+q+4);
                    if (item_tag == TAG_SEQ_DELIM || item_tag == TAG_ITEM_DELIM) break;
                    if (item_tag == TAG_ITEM)
                        info.frames.emplace_back(q+8, item_len);
                    q += 8 + item_len;
                }
            } else {
                info.pixel_data_offset = value_start;
                info.pixel_data_length = len;
            }
            break; // pixel data is always last meaningful tag
        }

        // Parse known tags
        if (value_start + len <= size) {
            switch(tag) {
            case TAG_ROWS:                  info.rows = read_u16(data+value_start); break;
            case TAG_COLUMNS:               info.columns = read_u16(data+value_start); break;
            case TAG_SAMPLES_PER_PIXEL:     info.samples = read_u16(data+value_start); break;
            case TAG_BITS_ALLOCATED:        info.bits_alloc = read_u16(data+value_start); break;
            case TAG_BITS_STORED:           info.bits_stored = read_u16(data+value_start); break;
            case TAG_PIXEL_REPRESENTATION:  info.pixel_repr = read_u16(data+value_start); break;
            case TAG_PLANAR_CONFIGURATION:  info.planar_config = read_u16(data+value_start); break;
            case TAG_PHOTOMETRIC_INTERP:    info.photometric = parse_string(data+value_start, len); break;
            case TAG_NUMBER_OF_FRAMES:      try { info.n_frames = std::stoi(parse_string(data+value_start,len)); } catch(...) {} break;
            case TAG_RESCALE_INTERCEPT:     try { info.rescale_intercept = std::stof(parse_string(data+value_start,len)); } catch(...) {} break;
            case TAG_RESCALE_SLOPE:         try { info.rescale_slope = std::stof(parse_string(data+value_start,len)); } catch(...) {} break;
            case TAG_PIXEL_SPACING: {
                auto v = parse_ds_multi(data+value_start, len);
                if (v.size()>=2){ info.pixel_spacing[0]=v[0]; info.pixel_spacing[1]=v[1]; } break; }
            case TAG_SLICE_THICKNESS: {
                auto v = parse_ds_multi(data+value_start, len);
                if (!v.empty()) info.slice_thickness=v[0]; break; }
            case TAG_IMAGE_POS_PATIENT: {
                auto v = parse_ds_multi(data+value_start, len);
                for (int i=0;i<3&&i<(int)v.size();i++) info.image_pos[i]=v[i]; break; }
            case TAG_IMAGE_ORIENT_PATIENT: {
                auto v = parse_ds_multi(data+value_start, len);
                for (int i=0;i<6&&i<(int)v.size();i++) info.image_orient[i]=v[i]; break; }
            default: break;
            }
        }

        if (len == UNDEF_LENGTH) break; // undefined length non-pixel tag — stop
        p = value_start + ((len+1)&~1); // advance (values are even-padded)
    }

    if (!found_pixel_data)
        throw std::runtime_error("DICOM: PixelData (7FE0,0010) not found");
    if (info.rows == 0 || info.columns == 0)
        throw std::runtime_error("DICOM: Rows/Columns not found");

    return info;
}

// ── DLDataType from DICOM pixel attributes ────────────────────────────────────
static inline DLDataType dicom_dtype(const DicomInfo& d)
{
    uint8_t bits = d.bits_alloc;
    if (d.pixel_repr == 1) // signed
        return {kDLInt, bits, 1};
    else
        return {kDLUInt, bits, 1};
}

} // namespace cumed::dicom
