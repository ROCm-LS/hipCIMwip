// SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUSLIDE_NVJPEG_PROCESSOR_H
#define CUSLIDE_NVJPEG_PROCESSOR_H
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>
#include <rocjpeg/rocjpeg.h>
#include <cucim/filesystem/cufile_driver.h>
#include <cucim/filesystem/file_handle.h>
#include <cucim/io/device.h>
#include <cucim/loader/batch_data_processor.h>
#include <cucim/loader/tile_info.h>
#include "cuslide/tiff/ifd.h"

#include <stdexcept>
#include <string>
// Raise a recoverable C++ exception on a rocJPEG/HIP API failure instead of
// terminating the host process. cuslide callers already treat a failed batch
// decode as a soft signal to fall back to the CPU path, so a thrown error lets
// a single bad tile degrade gracefully rather than killing the whole process.
static inline void PrintError(const char* file, const int line,
                              const char* context, const char* error) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             " error: " + context + " returned '" + error + "'");
}

#define CHECK_ROCJPEG(call) {                                 \
    RocJpegStatus rocjpeg_status = (call);                    \
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {           \
        PrintError(__FILE__, __LINE__, #call,                 \
                   rocJpegGetErrorName(rocjpeg_status));      \
    }                                                         \
}

#define CHECK_HIP(call) {                                     \
    hipError_t hip_status = (call);                           \
    if (hip_status != hipSuccess) {                           \
        PrintError(__FILE__, __LINE__, #call,                 \
                   hipGetErrorName(hip_status));              \
    }                                                         \
}

namespace cuslide::loader
{
class RocJpegProcessor : public cucim::loader::BatchDataProcessor
{
public:
    RocJpegProcessor(CuCIMFileHandle* file_handle,
                    const cuslide::tiff::IFD* ifd,
                    const int64_t* request_location,
                    const int64_t* request_size,
                    uint64_t location_len,
                    uint32_t batch_size,
                    uint32_t maximum_tile_count,
                    const uint8_t* jpegtable_data,
                    uint32_t jpegtable_size);
    ~RocJpegProcessor();
    uint32_t request(std::deque<uint32_t>& batch_item_counts, uint32_t num_remaining_patches) override;
    uint32_t wait_batch(uint32_t index_in_task,
                        std::deque<uint32_t>& batch_item_counts,
                        uint32_t num_remaining_patches) override;
    std::shared_ptr<cucim::cache::ImageCacheValue> wait_for_processing(uint32_t index) override;
    void shutdown() override;
    uint32_t preferred_loader_prefetch_factor();
private:
    bool stopped_ = false;
    uint32_t preferred_loader_prefetch_factor_ = 2;
    CuCIMFileHandle* file_handle_ = nullptr;
    const cuslide::tiff::IFD* ifd_ = nullptr;
    std::shared_ptr<cucim::filesystem::CuFileDriver> cufile_;
    size_t tile_width_ = 0;
    size_t tile_width_bytes_ = 0;
    size_t tile_height_ = 0;
    size_t tile_raster_nbytes_ = 0;
    uint32_t cuda_batch_size_ = 1;

    RocJpegStreamHandle handle_ = nullptr;
    RocJpegOutputFormat output_format_ = ROCJPEG_OUTPUT_RGB;
    // Aperio/Generic RGB-photometric tiles store JPEG components that are already
    // R,G,B (not YCbCr). ROCJPEG_OUTPUT_RGB unconditionally applies a YCbCr->RGB
    // matrix, which corrupts genuine-RGB data. For those IFDs we decode NATIVE
    // (no colour transform) and interleave the three planes ourselves. YCbCr
    // tiles keep the standard ROCJPEG_OUTPUT_RGB path.
    bool decode_native_rgb_ = false;
    RocJpegStatus state_;
    RocJpegBackend backend_ = ROCJPEG_BACKEND_HARDWARE;
    hipStream_t stream_ = nullptr;

    std::vector<RocJpegStreamHandle> stream_handles_;
    std::vector<RocJpegDecodeParams> decode_params_;

    std::condition_variable cuda_batch_cond_;
    std::shared_ptr<cucim::cache::ImageCache> cuda_image_cache_; // process-level GPU tile cache
    uint64_t processed_cuda_batch_count_ = 0;
    cucim::loader::TileInfo fetch_after_{ -1, -1, 0, 0 };
    std::deque<uint32_t> cache_tile_queue_;
    std::unordered_map<uint32_t, cucim::loader::TileInfo> cache_tile_map_;

    // Per-batch host arena holding the raw compressed bytes of each tile in the
    // current batch. Only the tiles actually decoded by a request() are read
    // (gathered via per-tile pread) into their cuda_batch_size_ slots of
    // tile_slot_bytes_ each -- so a spatially-scattered batch never reads more
    // than the tiles it needs, instead of mirroring the whole [min..max] file
    // span. Host-resident because rocJpegStreamParse reads the JPEG header on
    // the host CPU even for ROCJPEG_BACKEND_HARDWARE; device memory is used only
    // for the decoded output (raw_cuda_outputs_).
    uint8_t* tile_arena_host_ = nullptr;
    size_t tile_slot_bytes_ = 0;

    std::vector<const unsigned char*> raw_cuda_inputs_;
    std::vector<size_t> raw_cuda_inputs_len_;
    std::vector<RocJpegImage> raw_cuda_outputs_;

    // SVS / TIFF abbreviated-JPEG support.
    //
    // Aperio SVS stores JPEG quantization + Huffman tables once in the TIFF
    // IFD's JPEGTables tag (0x015B); individual tile bitstreams contain only
    // [SOI][SOS][scan][EOI] without the tables. rocJPEG has no
    // tables-injection API equivalent to nvJPEG's
    // nvjpegDecodeBatchedParseJpegTables, so we pre-merge per tile before
    // calling rocJpegStreamParse:
    //
    //   merged = jpegtable[0 .. size-2)  ++  tile[2 .. tile_size)
    //          = [SOI][DQT/DHT...]       ++  [...SOS][scan][EOI]
    //
    // jpegtable_prefix_host_ : host copy of jpegtable[0..size-2]
    // merged_arena_host_     : single contiguous HOST arena sliced into
    //                          cuda_batch_size_ slots of merged_slot_bytes_
    //                          each, so per-batch allocation cost is paid once
    //                          at construction. Host-resident because the
    //                          merged stream is what rocJpegStreamParse reads.
    std::vector<uint8_t> jpegtable_prefix_host_;
    uint8_t* merged_arena_host_ = nullptr;
    size_t merged_slot_bytes_ = 0;
};
} // namespace cuslide::loader
#endif // CUSLIDE_ROCJPEG_PROCESSOR_H
