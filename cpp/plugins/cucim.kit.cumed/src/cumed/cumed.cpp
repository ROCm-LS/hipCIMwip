// SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: Modifications Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define CUCIM_EXPORTS

#include "cumed.h"
#include "nifti.h"

#include <cmath>
#include <fcntl.h>
#include <filesystem>

#include <fmt/format.h>

#include <cucim/core/framework.h>
#include <cucim/core/plugin_util.h>
#include <cucim/filesystem/file_path.h>
#include <cucim/io/format/image_format.h>
#include <cucim/memory/memory_manager.h>


const struct cucim::PluginImplDesc kPluginImpl = {
    "cucim.kit.cumed", // name
    { 0, 1, 0 }, // version
    "dev", // build
    "clara team", // author
    "cumed", // description
    "cumed plugin", // long_description
    "Apache-2.0", // license
    "https://github.com/rapidsai/cucim", // url
    "linux", // platforms,
    cucim::PluginHotReload::kDisabled, // hot_reload
};

CUCIM_PLUGIN_IMPL_MINIMAL(kPluginImpl, cucim::io::format::IImageFormat)
CUCIM_PLUGIN_IMPL_NO_DEPS()


static void set_enabled(bool val) { (void)val; }
static bool is_enabled() { return true; }
static const char* get_format_name() { return "MedicalImage"; }

// ── Detect whether a path is a NIfTI file ─────────────────────────────────────
static bool is_nifti(const std::string& path)
{
    if (path.size() >= 4 && path.compare(path.size()-4, 4, ".nii") == 0)
        return true;
    if (path.size() >= 7 && path.compare(path.size()-7, 7, ".nii.gz") == 0)
        return true;
    return false;
}

static bool CUCIM_ABI checker_is_valid(const char* file_name, const char* buf, size_t size)
{
    (void)buf; (void)size;
    std::string path(file_name);
    if (is_nifti(path)) return true;
    auto ext = std::filesystem::path(file_name).extension().string();
    if (ext == ".mhd") return true;
    return false;
}

// ── Per-file state stored in handle->client_data ──────────────────────────────
struct CumedFileState
{
    std::string path;
    bool        is_nifti = false;
};

static CuCIMFileHandle_share CUCIM_ABI parser_open(const char* file_path_)
{
    const cucim::filesystem::Path& file_path = file_path_;
    int mode = O_RDONLY;

    char* file_path_cstr = static_cast<char*>(malloc(file_path.size() + 1));
    memcpy(file_path_cstr, file_path.c_str(), file_path.size());
    file_path_cstr[file_path.size()] = '\0';

    int fd = ::open(file_path_cstr, mode, 0666);
    if (fd == -1)
    {
        free(file_path_cstr);
        throw std::invalid_argument(fmt::format("Cannot open {}!", file_path));
    }

    auto* state = new CumedFileState{std::string(file_path_cstr), is_nifti(file_path_cstr)};
    auto file_handle = std::make_shared<CuCIMFileHandle>(fd, nullptr, FileHandleType::kPosix, file_path_cstr, state);
    return new std::shared_ptr<CuCIMFileHandle>(std::move(file_handle));
}

// ── NIfTI parser_parse: fill ImageMetadataDesc from the 348-byte header ───────
static bool parser_parse_nifti(CuCIMFileHandle* handle,
                                cucim::io::format::ImageMetadata& out_metadata,
                                const std::string& path)
{
    auto info = cumed::nifti::parse(handle->fd, path);
    auto& hdr = info.hdr;
    auto& resource = out_metadata.get_resource();

    // Determine dims string and shape.
    // Present as "ZYXC" with C=1 for a 3D scalar volume, or "TZYXC" for 4D.
    // This keeps C always last (required by the existing cuimage read path).
    // The dims string drives how downstream code interprets the shape array.
    bool has_time = (info.nt > 1);
    uint16_t ndim = has_time ? 5 : 4; // T Z Y X C  or  Z Y X C
    std::string_view dims = has_time ? std::string_view{"TZYXC"} : std::string_view{"ZYXC"};

    std::pmr::vector<int64_t> shape(&resource);
    if (has_time) shape.emplace_back(info.nt);
    shape.emplace_back(info.nz);
    shape.emplace_back(info.ny);
    shape.emplace_back(info.nx);
    shape.emplace_back(1); // C=1 scalar volume

    DLDataType dtype = cumed::nifti::nifti_datatype_to_dl(hdr.datatype);

    std::pmr::vector<std::string_view> channel_names(&resource);
    channel_names.emplace_back("intensity");

    // Spacing from pixdim[1..3] (spatial) and optionally pixdim[4] (temporal)
    const char* sunit = cumed::nifti::spatial_unit_string(hdr.xyzt_units);
    std::pmr::vector<float> spacing(&resource);
    std::pmr::vector<std::string_view> spacing_units(&resource);
    if (has_time) {
        spacing.emplace_back(std::abs(hdr.pixdim[4])); // TR
        spacing_units.emplace_back("second");
    }
    spacing.emplace_back(std::abs(hdr.pixdim[3])); // Z
    spacing.emplace_back(std::abs(hdr.pixdim[2])); // Y
    spacing.emplace_back(std::abs(hdr.pixdim[1])); // X
    spacing.emplace_back(1.0f);                    // C (no physical spacing)
    spacing_units.emplace_back(sunit);
    spacing_units.emplace_back(sunit);
    spacing_units.emplace_back(sunit);
    spacing_units.emplace_back("color");

    // Affine / direction / origin
    // Prefer sform (method 3) over qform (method 2) per NIfTI spec
    std::pmr::vector<float> origin(&resource);
    std::pmr::vector<float> direction(&resource);

    if (hdr.sform_code > 0)
    {
        // srow_x/y/z encode a full affine; last column = translation (origin)
        // Extract 3x3 direction (normalise out pixdim scaling) + translation
        origin.emplace_back(hdr.srow_x[3]);
        origin.emplace_back(hdr.srow_y[3]);
        origin.emplace_back(hdr.srow_z[3]);

        float sx = (hdr.pixdim[1] > 0.f) ? hdr.pixdim[1] : 1.f;
        float sy = (hdr.pixdim[2] > 0.f) ? hdr.pixdim[2] : 1.f;
        float sz = (hdr.pixdim[3] > 0.f) ? hdr.pixdim[3] : 1.f;
        // rows are X, Y, Z direction cosines (divided by voxel size)
        direction.emplace_back(hdr.srow_x[0] / sx);
        direction.emplace_back(hdr.srow_x[1] / sy);
        direction.emplace_back(hdr.srow_x[2] / sz);
        direction.emplace_back(hdr.srow_y[0] / sx);
        direction.emplace_back(hdr.srow_y[1] / sy);
        direction.emplace_back(hdr.srow_y[2] / sz);
        direction.emplace_back(hdr.srow_z[0] / sx);
        direction.emplace_back(hdr.srow_z[1] / sy);
        direction.emplace_back(hdr.srow_z[2] / sz);
    }
    else if (hdr.qform_code > 0)
    {
        // Reconstruct rotation matrix from quaternion (quatern_b/c/d, a = sqrt(1-b²-c²-d²))
        float b = hdr.quatern_b, c = hdr.quatern_c, d = hdr.quatern_d;
        float a = std::sqrt(std::max(0.f, 1.f - b*b - c*c - d*d));
        float qfac = (hdr.pixdim[0] >= 0) ? 1.f : -1.f;

        origin.emplace_back(hdr.qoffset_x);
        origin.emplace_back(hdr.qoffset_y);
        origin.emplace_back(hdr.qoffset_z);

        // Quaternion → rotation matrix; Z column scaled by qfac
        direction.emplace_back(a*a + b*b - c*c - d*d);
        direction.emplace_back(2*(b*c - a*d));
        direction.emplace_back(qfac * 2*(b*d + a*c));
        direction.emplace_back(2*(b*c + a*d));
        direction.emplace_back(a*a + c*c - b*b - d*d);
        direction.emplace_back(qfac * 2*(c*d - a*b));
        direction.emplace_back(2*(b*d - a*c));
        direction.emplace_back(2*(c*d + a*b));
        direction.emplace_back(qfac * (a*a + d*d - b*b - c*c));
    }
    else
    {
        // Method 1: identity
        origin.insert(origin.end(), 3, 0.f);
        direction = {1,0,0, 0,1,0, 0,0,1};
        direction = std::pmr::vector<float>({1,0,0,0,1,0,0,0,1}, &resource);
    }

    // coord_sys is always RAS for NIfTI
    std::string_view coord_sys{"RAS"};

    // Level info (no pyramid)
    size_t level_count = 1;
    const uint16_t level_ndim = 2;
    std::pmr::vector<int64_t> level_dimensions(&resource);
    level_dimensions.emplace_back(info.nx);
    level_dimensions.emplace_back(info.ny);
    std::pmr::vector<float> level_downsamples(&resource);
    level_downsamples.emplace_back(1.f);
    std::pmr::vector<uint32_t> level_tile_sizes(&resource);
    level_tile_sizes.emplace_back(static_cast<uint32_t>(info.nx));
    level_tile_sizes.emplace_back(static_cast<uint32_t>(info.ny));

    // json_data: compact header summary (slope/intercept, codes)
    std::string json = fmt::format(
        "{{\"nifti1\":{{\"datatype\":{},\"scl_slope\":{},\"scl_inter\":{}"
        ",\"qform_code\":{},\"sform_code\":{}}}}}",
        hdr.datatype, hdr.scl_slope, hdr.scl_inter, hdr.qform_code, hdr.sform_code);
    char* json_ptr = static_cast<char*>(cucim_malloc(json.size() + 1));
    memcpy(json_ptr, json.c_str(), json.size() + 1);

    out_metadata.ndim(ndim);
    out_metadata.dims(std::move(dims));
    out_metadata.shape(std::move(shape));
    out_metadata.dtype(dtype);
    out_metadata.channel_names(std::move(channel_names));
    out_metadata.spacing(std::move(spacing));
    out_metadata.spacing_units(std::move(spacing_units));
    out_metadata.origin(std::move(origin));
    out_metadata.direction(std::move(direction));
    out_metadata.coord_sys(std::move(coord_sys));
    out_metadata.level_count(level_count);
    out_metadata.level_ndim(level_ndim);
    out_metadata.level_dimensions(std::move(level_dimensions));
    out_metadata.level_downsamples(std::move(level_downsamples));
    out_metadata.level_tile_sizes(std::move(level_tile_sizes));
    out_metadata.image_count(0);
    out_metadata.image_names(std::pmr::vector<std::string_view>(&resource));
    out_metadata.raw_data(std::string_view{""});
    out_metadata.json_data(std::string_view{json_ptr, json.size()});

    return true;
}

static bool CUCIM_ABI parser_parse(CuCIMFileHandle_ptr handle_ptr,
                                   cucim::io::format::ImageMetadataDesc* out_metadata_desc)
{
    CuCIMFileHandle* handle = reinterpret_cast<CuCIMFileHandle*>(handle_ptr);
    if (!out_metadata_desc || !out_metadata_desc->handle)
        throw std::runtime_error("out_metadata_desc shouldn't be nullptr!");

    cucim::io::format::ImageMetadata& out_metadata =
        *reinterpret_cast<cucim::io::format::ImageMetadata*>(out_metadata_desc->handle);

    auto* state = static_cast<CumedFileState*>(handle->client_data);
    if (state && state->is_nifti)
        return parser_parse_nifti(handle, out_metadata, state->path);

    // ── Fallback: MetaIO skeleton (unchanged) ─────────────────────────────────
    auto& resource = out_metadata.get_resource();
    const uint16_t ndim = 3;
    std::pmr::vector<int64_t> shape({ 256, 256, 3 }, &resource);
    DLDataType dtype{ kDLUInt, 8, 1 };
    std::pmr::vector<std::string_view> channel_names(
        { std::string_view{"R"}, std::string_view{"G"}, std::string_view{"B"} }, &resource);
    std::pmr::vector<float> spacing(&resource); spacing.insert(spacing.end(), ndim, 1.f);
    std::pmr::vector<std::string_view> spacing_units(&resource);
    spacing_units.emplace_back("pixel"); spacing_units.emplace_back("pixel"); spacing_units.emplace_back("color");
    std::pmr::vector<float> origin({ 0.f, 0.f, 0.f }, &resource);
    std::pmr::vector<float> direction({ 1.f,0.f,0.f, 0.f,1.f,0.f, 0.f,0.f,1.f }, &resource);
    std::pmr::vector<int64_t> level_dims(&resource); level_dims.emplace_back(256); level_dims.emplace_back(256);
    std::pmr::vector<float> level_ds(&resource); level_ds.emplace_back(1.f);
    std::pmr::vector<uint32_t> level_ts(&resource); level_ts.emplace_back(256); level_ts.emplace_back(256);
    const std::string& json_str = std::string{};
    char* json_ptr = static_cast<char*>(cucim_malloc(json_str.size() + 1));
    memcpy(json_ptr, json_str.data(), json_str.size() + 1);

    out_metadata.ndim(ndim);
    out_metadata.dims(std::string_view{"YXC"});
    out_metadata.shape(std::move(shape));
    out_metadata.dtype(dtype);
    out_metadata.channel_names(std::move(channel_names));
    out_metadata.spacing(std::move(spacing));
    out_metadata.spacing_units(std::move(spacing_units));
    out_metadata.origin(std::move(origin));
    out_metadata.direction(std::move(direction));
    out_metadata.coord_sys(std::string_view{"LPS"});
    out_metadata.level_count(1);
    out_metadata.level_ndim(2);
    out_metadata.level_dimensions(std::move(level_dims));
    out_metadata.level_downsamples(std::move(level_ds));
    out_metadata.level_tile_sizes(std::move(level_ts));
    out_metadata.image_count(0);
    out_metadata.image_names(std::pmr::vector<std::string_view>(&resource));
    out_metadata.raw_data(std::string_view{""});
    out_metadata.json_data(std::string_view{json_ptr, json_str.size()});
    return true;
}

static bool CUCIM_ABI parser_close(CuCIMFileHandle_ptr handle_ptr)
{
    CuCIMFileHandle* handle = reinterpret_cast<CuCIMFileHandle*>(handle_ptr);
    if (handle->client_data)
    {
        delete static_cast<CumedFileState*>(handle->client_data);
        handle->client_data = nullptr;
    }
    return true;
}

// ── NIfTI reader_read ─────────────────────────────────────────────────────────
static bool reader_read_nifti(CuCIMFileHandle* handle,
                               const cucim::io::format::ImageMetadataDesc* metadata,
                               const cucim::io::format::ImageReaderRegionRequestDesc* request,
                               cucim::io::format::ImageDataDesc* out_image_data,
                               const std::string& path)
{
    (void)metadata;

    std::string device_name(request->device ? request->device : "cpu");
    if (request->shm_name)
        device_name += fmt::format("[{}]", request->shm_name);
    cucim::io::Device out_device(device_name);

    // Parse (decompress if .nii.gz and re-read header)
    auto info = cumed::nifti::parse(handle->fd, path);

    size_t nvox = static_cast<size_t>(info.nx) * info.ny * info.nz * info.nt;
    size_t raster_size = nvox * info.vox_bytes;

    // Allocate host buffer and copy voxel data from the decompressed stream
    uint8_t* raster = static_cast<uint8_t*>(cucim_malloc(raster_size));
    memcpy(raster, info.raw_bytes.data() + info.data_offset, raster_size);

    // Stage to GPU if requested
    cucim::memory::move_raster_from_host((void**)&raster, raster_size, out_device);

    // Build output shape (matching dims from parser_parse: ZYX[T]C layout)
    bool has_time = (info.nt > 1);
    const uint16_t ndim = has_time ? 5 : 4;
    int64_t* container_shape = static_cast<int64_t*>(cucim_malloc(sizeof(int64_t) * ndim));
    size_t idx = 0;
    if (has_time) container_shape[idx++] = info.nt;
    container_shape[idx++] = info.nz;
    container_shape[idx++] = info.ny;
    container_shape[idx++] = info.nx;
    container_shape[idx]   = 1; // C

    auto& out_image_container = out_image_data->container;
    out_image_container.data       = raster;
    out_image_container.device     = DLDevice{static_cast<DLDeviceType>(out_device.type()), out_device.index()};
    out_image_container.ndim       = ndim;
    out_image_container.dtype      = cumed::nifti::nifti_datatype_to_dl(info.hdr.datatype);
    out_image_container.shape      = container_shape;
    out_image_container.strides    = nullptr; // compact row-major
    out_image_container.byte_offset = 0;

    auto& shm_name = out_device.shm_name();
    if (!shm_name.empty())
    {
        out_image_data->shm_name = static_cast<char*>(cucim_malloc(shm_name.size() + 1));
        memcpy(out_image_data->shm_name, shm_name.c_str(), shm_name.size() + 1);
    }
    else
    {
        out_image_data->shm_name = nullptr;
    }

    return true;
}

static bool CUCIM_ABI reader_read(const CuCIMFileHandle_ptr handle_ptr,
                                  const cucim::io::format::ImageMetadataDesc* metadata,
                                  const cucim::io::format::ImageReaderRegionRequestDesc* request,
                                  cucim::io::format::ImageDataDesc* out_image_data,
                                  cucim::io::format::ImageMetadataDesc* out_metadata_desc)
{
    (void)out_metadata_desc;
    CuCIMFileHandle* handle = reinterpret_cast<CuCIMFileHandle*>(handle_ptr);
    auto* state = static_cast<CumedFileState*>(handle->client_data);

    if (state && state->is_nifti)
        return reader_read_nifti(handle, metadata, request, out_image_data, state->path);

    // ── Fallback: MetaIO skeleton (returns zeroed 256x256x3) ─────────────────
    std::string device_name(request->device ? request->device : "cpu");
    cucim::io::Device out_device(device_name);

    constexpr size_t raster_size = 256 * 256 * 3;
    uint8_t* raster = static_cast<uint8_t*>(cucim_malloc(raster_size));
    memset(raster, 0, raster_size);
    cucim::memory::move_raster_from_host((void**)&raster, raster_size, out_device);

    int64_t* container_shape = static_cast<int64_t*>(cucim_malloc(sizeof(int64_t) * 3));
    container_shape[0] = 256; container_shape[1] = 256; container_shape[2] = 3;

    auto& out_image_container = out_image_data->container;
    out_image_container.data        = raster;
    out_image_container.device      = DLDevice{static_cast<DLDeviceType>(out_device.type()), out_device.index()};
    out_image_container.ndim        = 3;
    out_image_container.dtype       = {kDLUInt, 8, 1};
    out_image_container.shape       = container_shape;
    out_image_container.strides     = nullptr;
    out_image_container.byte_offset = 0;
    out_image_data->shm_name        = nullptr;
    return true;
}

static bool CUCIM_ABI writer_write(const CuCIMFileHandle_ptr handle_ptr,
                                   const cucim::io::format::ImageMetadataDesc* metadata,
                                   const cucim::io::format::ImageDataDesc* image_data)
{
    (void)handle_ptr; (void)metadata; (void)image_data;
    return true;
}

void fill_interface(cucim::io::format::IImageFormat& iface)
{
    static cucim::io::format::ImageCheckerDesc image_checker = { 0, 0, checker_is_valid };
    static cucim::io::format::ImageParserDesc image_parser = { parser_open, parser_parse, parser_close };
    static cucim::io::format::ImageReaderDesc image_reader = { reader_read };
    static cucim::io::format::ImageWriterDesc image_writer = { writer_write };

    // clang-format off
    static cucim::io::format::ImageFormatDesc image_format_desc = {
        set_enabled, is_enabled, get_format_name,
        image_checker, image_parser, image_reader, image_writer
    };
    iface = { &image_format_desc, 1 };
    // clang-format on
}
