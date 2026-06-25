// SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rocjpeg_processor.h"
#include "rocjpeg_jpegtables.h"

#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <unistd.h>
#include <cucim/cuda_runtime.h>
#include <cucim/cache/image_cache_manager.h>
#include <cucim/codec/hash_function.h>
#include <cucim/io/device.h>
#include <cucim/util/cuda.h>
#include <fmt/format.h>
#include <tiffio.h> // PHOTOMETRIC_RGB

#define ALIGN_UP(x, align_to) (((uint64_t)(x) + ((uint64_t)(align_to)-1)) & ~((uint64_t)(align_to)-1))
#define ALIGN_DOWN(x, align_to) ((uint64_t)(x) & ~((uint64_t)(align_to)-1))
namespace cuslide::loader
{

namespace
{
// Interleave three planar 8-bit channels (R, G, B) into a single
// width*3-pitched interleaved RGB raster. Used for the NATIVE-decode path on
// RGB-photometric tiles, where rocJPEG returns the components as separate
// planes and must NOT colour-transform them.
__global__ void interleave_rgb_planes_kernel(const uint8_t* __restrict__ r,
                                             const uint8_t* __restrict__ g,
                                             const uint8_t* __restrict__ b,
                                             uint32_t src_pitch_r,
                                             uint32_t src_pitch_g,
                                             uint32_t src_pitch_b,
                                             uint8_t* __restrict__ dst,
                                             size_t dst_pitch,
                                             uint32_t width,
                                             uint32_t height)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    uint8_t* out = dst + static_cast<size_t>(y) * dst_pitch + static_cast<size_t>(x) * 3;
    out[0] = r[static_cast<size_t>(y) * src_pitch_r + x];
    out[1] = g[static_cast<size_t>(y) * src_pitch_g + x];
    out[2] = b[static_cast<size_t>(y) * src_pitch_b + x];
}
} // namespace

constexpr uint32_t MAX_CUDA_BATCH_SIZE = 1024;

// Process-level pool of rocJPEG decode handles.
//
// A new RocJpegProcessor is constructed for every read_region() call, and
// rocJpegCreate()/rocJpegDestroy() are among the most expensive rocJPEG calls
// (they bring up the VCN decode context). For small reads (e.g. a single 256px
// patch = ~4 tiles) this per-call handle setup dominates wall time, making the
// GPU path slower than CPU. The handle is independent of tile geometry and
// batch size -- it only depends on the backend -- so a single pooled handle can
// be reused across all read_region() calls.
//
// Pooled handles are intentionally never destroyed at process exit (calling
// rocJpegDestroy in a static destructor after the runtime is torn down would
// crash); the OS reclaims everything at exit.
class RocJpegHandlePool
{
public:
    static RocJpegHandlePool& instance()
    {
        static RocJpegHandlePool* pool = new RocJpegHandlePool();
        return *pool;
    }

    // Acquire an idle handle for the given backend, or create one on miss.
    RocJpegStreamHandle acquire(RocJpegBackend backend)
    {
        {
            std::lock_guard<std::mutex> g(mutex_);
            auto it = idle_.find(backend);
            if (it != idle_.end() && !it->second.empty())
            {
                RocJpegStreamHandle h = it->second.back();
                it->second.pop_back();
                return h;
            }
        }
        // Miss: create outside the lock.
        RocJpegStreamHandle h = nullptr;
        CHECK_ROCJPEG(rocJpegCreate(backend, 0, &h));
        return h;
    }

    // Return a handle to the pool for reuse (does NOT destroy it).
    void release(RocJpegBackend backend, RocJpegStreamHandle h)
    {
        if (h == nullptr)
            return;
        std::lock_guard<std::mutex> g(mutex_);
        idle_[backend].push_back(h);
    }

private:
    std::mutex mutex_;
    std::unordered_map<int, std::vector<RocJpegStreamHandle>> idle_;
};

// Process-level GPU tile cache shared across all RocJpegProcessor instances.
// Tiles decoded by one read_region() call are available on cache hits in subsequent calls,
// matching the CPU path's use of the global ImageCache.
//
// The cache is intentionally leaked (never destroyed) via a raw pointer to avoid freeing
// device memory in a static destructor after the HIP runtime has been torn down, which would
// crash or hang. At process exit the OS reclaims all device memory anyway.
static std::shared_ptr<cucim::cache::ImageCache>& s_gpu_tile_cache()
{
    static std::shared_ptr<cucim::cache::ImageCache>* cache = []() {
        cucim::cache::ImageCacheConfig cfg{};
        cfg.type = cucim::cache::CacheType::kPerProcess;
        // memory_capacity is in MiB (capacity_nbytes_ = kOneMiB * memory_capacity); this is a
        // ~1 TiB byte budget used as a non-reserved sentinel so the byte ceiling never drives
        // eviction. Device memory is hipMalloc'd lazily per tile, not pre-reserved.
        cfg.memory_capacity = 1024 * 1024;
        // The real bound is the tile COUNT: 4096 tiles ≈ 4× MAX_CUDA_BATCH_SIZE (1024), leaving
        // headroom for a full batch plus cross-call reuse without self-eviction within a batch.
        // At 256×256×3 ≈ 192 KiB/tile a full cache is ~768 MiB of VRAM — negligible on
        // MI300X/MI355X (192–288 GB), and only reached if 4096 distinct tiles are touched.
        cfg.capacity = 4096;
        cfg.record_stat = true;   // track hit/miss to validate reuse in benchmarks
        return new std::shared_ptr<cucim::cache::ImageCache>(
            cucim::cache::ImageCacheManager::create_cache(cfg, cucim::io::DeviceType::kCUDA));
    }();
    return *cache;
}

RocJpegProcessor::RocJpegProcessor(CuCIMFileHandle* file_handle,
                                 const cuslide::tiff::IFD* ifd,
                                 const int64_t* request_location,
                                 const int64_t* request_size,
                                 const uint64_t location_len,
                                 const uint32_t batch_size,
                                 uint32_t maximum_tile_count,
                                 const uint8_t* jpegtable_data,
                                 const uint32_t jpegtable_size)
    : cucim::loader::BatchDataProcessor(batch_size), file_handle_(file_handle), ifd_(ifd)
{
    // Capture the IFD's JPEGTables prefix once at construction.
    // See rocjpeg_jpegtables.h for the merged-stream layout rationale.
    {
        const size_t prefix_len =
            detail::jpegtable_prefix_length(jpegtable_data, jpegtable_size);
        if (prefix_len > 0)
        {
            jpegtable_prefix_host_.assign(jpegtable_data,
                                          jpegtable_data + prefix_len);
        }
    }
    // If the IFD has no JPEGTables tag, jpegtable_prefix_host_ stays empty
    // and the merge step is a no-op (tile is passed through unchanged), which
    // is correct for tiles that embed their own tables.

    if (maximum_tile_count > 1)
    {
        // Calculate nearlest power of 2 that is equal or larger than the given number.
        // (Test with https://godbolt.org/z/n7qhPYzfP)
        int next_candidate = maximum_tile_count & (maximum_tile_count - 1);
        if (next_candidate > 0)
        {
            maximum_tile_count <<= 1;
            while (true)
            {
                next_candidate = maximum_tile_count & (maximum_tile_count - 1);
                if (next_candidate == 0)
                {
                    break;
                }
                maximum_tile_count = next_candidate;
            }
        }

        // Batch size selection
        // Do not exceed MAX_CUDA_BATCH_SIZE for decoding JPEG with nvJPEG
        uint32_t cuda_batch_size = std::min(maximum_tile_count, MAX_CUDA_BATCH_SIZE);

        // Update prefetch_factor
        // (We can decode/cache tiles at least two times of the number of tiles for batch decoding)
        // E.g., (128 - 1) / 32 + 1 ~= 4 => 8 (for 256 tiles) for cuda_batch_size(=128) and batch_size(=32)
        preferred_loader_prefetch_factor_ = ((cuda_batch_size - 1) / batch_size_ + 1) * 2;

        // Use the process-level GPU tile cache so decoded tiles survive across read_region()
        // calls. The CPU path already shares a global ImageCache (CuImage::cache_manager().cache());
        // the GPU path previously created a per-call cache that was destroyed with the
        // RocJpegProcessor, discarding all cached tiles at the end of every read_region().
        // A process-level cache allows the second and subsequent read_region() calls that
        // re-sample the same tiles (e.g. multi-epoch training) to hit the cache and skip
        // rocJPEG decode entirely.
        //
        // The cache is keyed by (ifd_hash, tile_index) and stores device pointers
        // (DeviceType::kCUDA). It is separate from the CPU global cache to avoid
        // device-type mismatches: a tile cached as kCPU must not be returned to a GPU
        // consumer without a host→device copy.
        cuda_image_cache_ = s_gpu_tile_cache();

        cuda_batch_size_ = cuda_batch_size;

        // NOTE: the rocJPEG handle and per-slot stream handles are created at the
        // end of this constructor, once cuda_batch_size_ is known. The compressed
        // input is host-resident (see the gather/parse notes below), so there is
        // no device file-block backend decision and the default backend_ is used.

        // Inputs to rocJPEG for decoding
        raw_cuda_inputs_.resize(cuda_batch_size_);
        raw_cuda_inputs_len_.resize(cuda_batch_size_);
        decode_params_.resize(cuda_batch_size_);
        for (uint32_t i = 0; i < cuda_batch_size_; ++i)
        {
            // Add default-constructed RocJpegImage object
            raw_cuda_outputs_.emplace_back();
        }

        // Tile size setup
        tile_width_ = ifd->tile_width();
        tile_width_bytes_ = tile_width_ * ifd->pixel_size_nbytes();
        tile_height_ = ifd->tile_height();
        tile_raster_nbytes_ = tile_width_bytes_ * tile_height_;

        // RGB-photometric tiles (e.g. Aperio SVS, Generic TIFF) store JPEG
        // components that are already R,G,B. ROCJPEG_OUTPUT_RGB would apply a
        // spurious YCbCr->RGB conversion (corrupting the colours), so for these
        // we decode NATIVE (no colour transform) and interleave the planes
        // ourselves. Genuine YCbCr tiles keep the ROCJPEG_OUTPUT_RGB path.
        decode_native_rgb_ = (ifd->photometric() == PHOTOMETRIC_RGB);
        output_format_ = decode_native_rgb_ ? ROCJPEG_OUTPUT_NATIVE : ROCJPEG_OUTPUT_RGB;

        // rocJpegStreamParse() parses the JPEG header on the HOST CPU -- it
        // dereferences the data pointer directly (e.g. the SOI check
        // `*stream_ != 0xFF` and std::memcpy of the DQT/DHT tables in
        // rocJpegStreamParser::ParseJpegStream). It therefore requires a HOST
        // pointer for the compressed input, even for ROCJPEG_BACKEND_HARDWARE.
        // (rocJPEG's own batched samples read the compressed JPEG into a host
        // std::vector and pass host .data() to rocJpegStreamParse regardless of
        // backend; only the decoded OUTPUT is device memory.) So the compressed
        // tile bytes and the merged-JPEG arena are always host-resident; device
        // memory is reserved for the decode output only.
        //
        // Tiles are gathered per batch in request() rather than mirroring the
        // whole [min..max] file span here. A spatially-scattered batch (e.g.
        // random patch sampling) can span nearly the entire slide, so a single
        // pread of that span would read hundreds of MB to use a few small
        // tiles. Instead each request() reads only the tiles it will decode,
        // in parallel, into a per-batch host tile arena, using the slide's own
        // buffered file descriptor (file_handle_->fd) -- the same descriptor the
        // CPU libjpeg path uses, so reads are page-cache-eligible (no O_DIRECT
        // self-penalty) and positional pread() is thread-safe across tiles.

        // Per-batch host arena holding the raw compressed bytes of each tile in
        // the current batch. tile_raster_nbytes_ is a safe upper bound on a
        // compressed JPEG tile (it never exceeds its uncompressed raster size
        // in practice); +64 for alignment/safety slack.
        tile_slot_bytes_ = tile_raster_nbytes_ + 64;
        tile_arena_host_ = static_cast<uint8_t*>(
            cucim_malloc(static_cast<size_t>(cuda_batch_size_) * tile_slot_bytes_));

        // Allocate the merged-JPEG arena (host). Each batch slot gets
        // merged_slot_bytes_ contiguous bytes; tile_raster_nbytes_ is a safe
        // upper bound on the compressed JPEG size plus the prefix length, plus
        // a small safety margin.
        if (!jpegtable_prefix_host_.empty())
        {
            const size_t prefix_len = jpegtable_prefix_host_.size();
            merged_slot_bytes_ = prefix_len + tile_raster_nbytes_ + 64;

            const size_t arena_bytes = static_cast<size_t>(cuda_batch_size_) * merged_slot_bytes_;

            merged_arena_host_ = static_cast<uint8_t*>(cucim_malloc(arena_bytes));
        }

        // Create the rocJPEG decode handle and one stream handle per batch slot.
        // Acquire the rocJPEG decode handle from the process-level pool (creates
        // one on a cold miss, reuses an idle one otherwise). rocJpegCreate is
        // expensive and geometry-independent, so reusing it across read_region()
        // calls removes the per-call setup that dominates small reads. The handle
        // is returned to the pool (not destroyed) in the destructor.
        handle_ = RocJpegHandlePool::instance().acquire(backend_);

        // request() dereferences stream_handles_[i] for each tile and passes
        // handle_ + stream_handles_.data() to rocJpegDecodeBatched; both must be
        // initialized. (Stream handles are still per-instance for now; they are
        // cheap relative to rocJpegCreate.)
        stream_handles_.resize(cuda_batch_size_);
        for (uint32_t i = 0; i < cuda_batch_size_; ++i)
        {
            CHECK_ROCJPEG(rocJpegStreamCreate(&stream_handles_[i]));
        }
    }
}

RocJpegProcessor::~RocJpegProcessor()
{
    // Free the per-batch host tile arena (gathered compressed tile bytes).
    if (tile_arena_host_)
    {
        cucim_free(tile_arena_host_);
        tile_arena_host_ = nullptr;
    }

    // Free the host-resident merged-JPEG arena.
    if (merged_arena_host_)
    {
        cucim_free(merged_arena_host_);
        merged_arena_host_ = nullptr;
    }

    // Release output buffers for each batch and channel.
    // NOTE: best-effort, non-throwing cleanup only. PrintError now throws, and
    // CHECK_HIP/CHECK_ROCJPEG route through it; throwing from a destructor would
    // call std::terminate(). Ignore teardown errors instead.
    for (uint32_t i = 0; i < cuda_batch_size_; ++i)
    {
        for (int ch = 0; ch < ROCJPEG_MAX_COMPONENT; ++ch)
        {
            if (raw_cuda_outputs_[i].channel[ch])
            {
                // Free CUDA memory for each channel buffer
                (void)hipFree(raw_cuda_outputs_[i].channel[ch]);
                raw_cuda_outputs_[i].channel[ch] = nullptr;
            }
        }
    }

    // Destroy rocJPEG stream handles
    for (auto& stream : stream_handles_)
    {
        (void)rocJpegStreamDestroy(stream);
        stream = nullptr;
    }
    stream_handles_.clear();

    // Return the rocJPEG handle to the process-level pool for reuse instead of
    // destroying it (rocJpegDestroy is expensive and the handle is reusable
    // across read_region() calls). The pool never destroys handles; the OS
    // reclaims them at process exit.
    if (handle_)
    {
        RocJpegHandlePool::instance().release(backend_, handle_);
        handle_ = nullptr;
    }

    // Close any opened CuFile
    if (cufile_)
    {
        cufile_->close();
        cufile_.reset(); // Release the shared_ptr ownership, deleting if refcount==0
    }

    // cuda_image_cache_ is a shared_ptr to the process-level GPU tile cache; releasing
    // our reference here does not destroy the cache (the static in s_gpu_tile_cache()
    // holds the last reference for the process lifetime).
    cuda_image_cache_.reset();
}

// Requests image tile(s) to be decoded and processed using the rocJpeg backend.
uint32_t RocJpegProcessor::request(std::deque<uint32_t>& batch_item_counts, const uint32_t num_remaining_patches)
{
    // Unused parameter -- suppress diagnostics
    (void)batch_item_counts;

    std::vector<cucim::loader::TileInfo> tile_to_request;
    if (tiles_.empty())
    {
        return 0;
    }

    // Return if we need to wait until previous cuda batch is consumed.
    auto& first_tile = tiles_.front();
    if (first_tile.location_index <= fetch_after_.location_index)
    {
        if (first_tile.location_index < fetch_after_.location_index || first_tile.index < fetch_after_.index)
        {
            return 0;
        }
    }

    // Set fetch_after_ to the last tile info of previously processed cuda batch
    if (!cache_tile_queue_.empty())
    {
        fetch_after_ = cache_tile_map_[cache_tile_queue_.back()];
    }

    // Remove previous batch (keep last 'cuda_batch_size_' items) before adding/processing new cuda batch
    std::vector<cucim::loader::TileInfo> removed_tiles;
    while (cache_tile_queue_.size() > cuda_batch_size_)
    {
        uint32_t removed_tile_index = cache_tile_queue_.front();
        auto removed_tile = cache_tile_map_.find(removed_tile_index);
        removed_tiles.push_back(removed_tile->second);
        cache_tile_queue_.pop_front();
        cache_tile_map_.erase(removed_tile_index);
    }

    // Collect candidates
    for (auto tile : tiles_)
    {
        auto index = tile.index;
        if (tile_to_request.size() >= cuda_batch_size_)
            break;

        if (cache_tile_map_.find(index) == cache_tile_map_.end() && tile.size > 0)
        {
            // Cross-call reuse: if this tile was already decoded by a previous
            // read_region() and still lives in the process-level GPU tile cache,
            // skip decoding it. The consumer's wait_for_processing() looks the
            // tile up in the same cache by (ifd_hash, index) and finds it
            // directly, so no decode/insert is needed here. This is what turns
            // the process-level cache into an actual decode-skip on repeated /
            // overlapping reads (e.g. multi-epoch training).
            auto cached_key = cuda_image_cache_->create_key(ifd_->hash_value(), index);
            if (cuda_image_cache_->find(cached_key))
            {
                continue;
            }

            cache_tile_queue_.emplace_back(index);
            cache_tile_map_.emplace(index, tile);
            tile_to_request.emplace_back(tile);
        }
    }

    // Return if not enough for a full batch and more are expected, else move on
    size_t request_count = tile_to_request.size();
    if (request_count < cuda_batch_size_)
    {
        if (num_remaining_patches > 0)
        {
            // Restore cache_tile_queue_ and cache_tile_map_
            for (auto& added_tile : tile_to_request)
            {
                uint32_t added_index = added_tile.index;
                cache_tile_queue_.pop_back();
                cache_tile_map_.erase(added_index);
            }
            for (auto rit = removed_tiles.rbegin(); rit != removed_tiles.rend(); ++rit)
            {
                uint32_t removed_index = rit->index;
                cache_tile_queue_.emplace_front(removed_index);
                cache_tile_map_.emplace(removed_index, *rit);
            }
            return 0;
        }
        else
        {
            // Completed, set fetch_after_ to the last tile info.
            fetch_after_ = tiles_.back();
        }
    }

    // DO NOT REUSE existing input vectors, always reset
    raw_cuda_inputs_.clear();
    raw_cuda_inputs_len_.clear();

    // Make sure outputs, decode_params vectors are also set for correct size
    if (raw_cuda_outputs_.size() < request_count)
        raw_cuda_outputs_.resize(request_count);

    if (decode_params_.size() < request_count)
        decode_params_.resize(request_count);

    // Setup inputs, outputs, and decode_params for just as many as request_count
    const bool need_merge = !jpegtable_prefix_host_.empty();
    const size_t prefix_len = jpegtable_prefix_host_.size();
    bool any_parse_failed = false;

    // Phase 1: gather the compressed bytes of every valid tile in this batch in
    // PARALLEL. Scattered tiles sit at random file offsets, so the per-tile read
    // is latency-bound; issuing them concurrently overlaps that latency instead
    // of paying it serially. Positional pread() on the same fd is thread-safe
    // (no shared seek pointer), and each tile writes a disjoint arena slot, so
    // no synchronization is needed beyond joining. A tile that is zero-length or
    // larger than its slot is left unread (marked invalid) so it falls back to
    // CPU decode instead of overrunning the arena.
    const int fd = file_handle_->fd;
    std::vector<uint8_t> tile_valid(request_count, 0);
    {
        unsigned int hw = std::thread::hardware_concurrency();
        size_t n_threads = std::min<size_t>(request_count, hw ? hw : 8);
        if (n_threads < 1)
            n_threads = 1;
        auto read_range = [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                const size_t tsize = tile_to_request[i].size;
                if (tsize == 0 || tsize > tile_slot_bytes_)
                    continue; // leave invalid -> CPU fallback
                uint8_t* dst = tile_arena_host_ + (i * tile_slot_bytes_);
                size_t got = 0;
                const off_t base = static_cast<off_t>(tile_to_request[i].offset);
                while (got < tsize)
                {
                    ssize_t r = ::pread(fd, dst + got, tsize - got, base + static_cast<off_t>(got));
                    if (r <= 0)
                        break;
                    got += static_cast<size_t>(r);
                }
                if (got == tsize)
                    tile_valid[i] = 1;
            }
        };
        if (n_threads <= 1)
        {
            read_range(0, request_count);
        }
        else
        {
            std::vector<std::thread> pool;
            pool.reserve(n_threads);
            const size_t chunk = (request_count + n_threads - 1) / n_threads;
            for (size_t t = 0; t < n_threads; ++t)
            {
                const size_t b = t * chunk;
                if (b >= request_count)
                    break;
                const size_t e = std::min(request_count, b + chunk);
                pool.emplace_back(read_range, b, e);
            }
            for (auto& th : pool)
                th.join();
        }
    }

    // Phase 2: merge (splice JPEGTables prefix) + parse, reading from the bytes
    // already gathered in phase 1.
    for (size_t i = 0; i < request_count; ++i)
    {
        const size_t tile_size = tile_to_request[i].size;

        const unsigned char* input_ptr = nullptr;
        size_t input_len = 0;

        if (!tile_valid[i])
        {
            std::cerr << "**warning**: tile " << i << " (size " << tile_size
                      << ") is zero/oversized or its read was short"
                      << "; skipping GPU decode for this tile" << std::endl;
            raw_cuda_inputs_.push_back(nullptr);
            raw_cuda_inputs_len_.push_back(0);
            any_parse_failed = true;
        }
        else
        {
            uint8_t* tile_mem_offset = tile_arena_host_ + (i * tile_slot_bytes_);

            // For SVS / TIFF abbreviated-JPEG tiles, splice the
            // IFD's JPEGTables prefix into a per-slot merged buffer so the
            // resulting stream is a complete decodable JPEG.
            //
            //   merged = [SOI][DQT/DHT...]  ++  [tile after SOI marker]
            //
            // Tile is expected to start with FF D8 (SOI); if not, fall through
            // to the pass-through path and let rocJpegStreamParse report the
            // problem rather than silently corrupting the bitstream.
            input_ptr = static_cast<const unsigned char*>(tile_mem_offset);
            input_len = tile_size;

            const size_t merged_len = need_merge
                ? detail::plan_merged_jpeg_size(prefix_len, tile_mem_offset,
                                                tile_size, merged_slot_bytes_)
                : 0;
            if (merged_len > 0)
            {
                // Splice [JPEGTables prefix] ++ [tile after SOI] into a host
                // slot. The merged buffer is what rocJpegStreamParse reads, so
                // it must be host memory for every backend.
                uint8_t* slot = merged_arena_host_ + (i * merged_slot_bytes_);
                detail::splice_jpegtable_prefix(jpegtable_prefix_host_.data(),
                                                prefix_len,
                                                tile_mem_offset, tile_size,
                                                slot);
                input_ptr = slot;
                input_len = merged_len;
            }

            raw_cuda_inputs_.push_back(input_ptr);
            raw_cuda_inputs_len_.push_back(input_len);

#ifndef NDEBUG
            // Quick SOI marker check on the *original* tile (the merged buffer
            // is guaranteed to start with SOI by construction).
            if (tile_size < 2 || tile_mem_offset[0] != 0xFF || tile_mem_offset[1] != 0xD8) {
                std::cerr << "**warning**: tile " << i
                          << " does not start with SOI (FF D8); using raw bytes" << std::endl;
            }
#endif // !NDEBUG

            // Each tile must have a valid RocJpegStreamHandle.
            // Do NOT use CHECK_ROCJPEG (which throws) — we want to fall
            // through to a CPU fallback path on BAD_JPEG so a single bad tile
            // can't abort the whole batch.
            RocJpegStatus s = rocJpegStreamParse(raw_cuda_inputs_[i],
                                                 raw_cuda_inputs_len_[i],
                                                 stream_handles_[i]);
            if (s != ROCJPEG_STATUS_SUCCESS)
            {
                std::cerr << "**warning**: rocJpegStreamParse failed for tile " << i
                          << " (" << rocJpegGetErrorName(s) << ")"
                          << std::endl;
                raw_cuda_inputs_len_.back() = 0;
                any_parse_failed = true;
            }
        }

        decode_params_[i].output_format = output_format_;

        // Set crop_rectangle and target_dimension if needed
        decode_params_[i].crop_rectangle = {0, 0, 0, 0}; // No cropping by default
        decode_params_[i].target_dimension = {0, 0};     // No resizing by default

        // Allocate output buffer(s).
        if (decode_native_rgb_)
        {
            // NATIVE decode returns three separate width-wide planes (one per
            // component). Allocate channel[0..2], each tile_width_ bytes wide.
            for (int ch = 0; ch < 3; ++ch)
            {
                if (!raw_cuda_outputs_[i].channel[ch] ||
                    raw_cuda_outputs_[i].pitch[ch] != tile_width_)
                {
                    if (raw_cuda_outputs_[i].channel[ch])
                        hipFree(raw_cuda_outputs_[i].channel[ch]);

                    CHECK_HIP(hipMallocPitch(
                        reinterpret_cast<void**>(&raw_cuda_outputs_[i].channel[ch]),
                        reinterpret_cast<size_t*>(&raw_cuda_outputs_[i].pitch[ch]),
                        tile_width_, tile_height_));
                }
            }
        }
        else
        {
            // ROCJPEG_OUTPUT_RGB returns a single interleaved width*3 raster.
            size_t expected_pitch = tile_width_bytes_;
            if (!raw_cuda_outputs_[i].channel[0] ||
                raw_cuda_outputs_[i].pitch[0] != expected_pitch)
            {
                if (raw_cuda_outputs_[i].channel[0])
                    hipFree(raw_cuda_outputs_[i].channel[0]);

                CHECK_HIP(hipMallocPitch(
                    reinterpret_cast<void**>(&raw_cuda_outputs_[i].channel[0]),
                    reinterpret_cast<size_t*>(&raw_cuda_outputs_[i].pitch[0]),
                    tile_width_bytes_, tile_height_));
            }
        }
    }

    // Batch decode: if any per-tile parse failed above, skip
    // rocJpegDecodeBatched (it would either fail outright or produce
    // garbage for the bad slot); the caller treats missing cache entries
    // as misses and the IFD::read_region CPU fallback path takes over.
    // Use soft status check (not CHECK_ROCJPEG, which throws) so a
    // single corrupt tile cannot abort the whole batch.
    if (!any_parse_failed)
    {
        RocJpegStatus s = rocJpegDecodeBatched(handle_, stream_handles_.data(),
                                               request_count,
                                               decode_params_.data(),
                                               raw_cuda_outputs_.data());
        if (s != ROCJPEG_STATUS_SUCCESS)
        {
            std::cerr << "**warning**: rocJpegDecodeBatched failed ("
                      << rocJpegGetErrorName(s)
                      << "); CPU fallback path will be used for this batch"
                      << std::endl;
        }
    }
    else
    {
        std::cerr << "**warning**: batch contained " << request_count
                  << " tile(s) with parse errors; skipping GPU decode"
                  << std::endl;
    }

    // The process-level cache manages its own capacity (LRU eviction up to capacity=1024 tiles).
    // We no longer manually evict tiles here — doing so would discard cached tiles that could
    // be reused by the next read_region() call (cross-call reuse is the point of the shared cache).

    // Add to image cache
    for (uint32_t i = 0; i < request_count; ++i)
    {
        auto& added_tile = tile_to_request[i];

        uint32_t index = added_tile.index;
        uint64_t index_hash = cucim::codec::splitmix64(index);

        auto key = cuda_image_cache_->create_key(ifd_->hash_value(), index);

        cuda_image_cache_->lock(index_hash);

        uint8_t* tile_data = static_cast<uint8_t*>(cuda_image_cache_->allocate(tile_raster_nbytes_));

        cudaError_t cuda_status;
        if (decode_native_rgb_)
        {
            // NATIVE decode gave three R,G,B planes (no colour transform).
            // Interleave them into the cache's width*3 raster.
            dim3 block(16, 16);
            dim3 grid((static_cast<uint32_t>(tile_width_) + block.x - 1) / block.x,
                      (static_cast<uint32_t>(tile_height_) + block.y - 1) / block.y);
            hipLaunchKernelGGL(interleave_rgb_planes_kernel, grid, block, 0, 0,
                               raw_cuda_outputs_[i].channel[0],
                               raw_cuda_outputs_[i].channel[1],
                               raw_cuda_outputs_[i].channel[2],
                               raw_cuda_outputs_[i].pitch[0],
                               raw_cuda_outputs_[i].pitch[1],
                               raw_cuda_outputs_[i].pitch[2],
                               tile_data, tile_width_bytes_,
                               static_cast<uint32_t>(tile_width_),
                               static_cast<uint32_t>(tile_height_));
            CHECK_HIP(hipGetLastError());
            CHECK_HIP(hipDeviceSynchronize());
        }
        else
        {
            CUDA_TRY(
                cudaMemcpy2D(tile_data, tile_width_bytes_,
                             raw_cuda_outputs_[i].channel[0],
                             raw_cuda_outputs_[i].pitch[0],
                             tile_width_bytes_, tile_height_,
                             cudaMemcpyDeviceToDevice)
            );
        }

        const size_t tile_raster_nbytes = raw_cuda_inputs_len_[i];
        auto value = cuda_image_cache_->create_value(tile_data, tile_raster_nbytes, cucim::io::DeviceType::kCUDA);
        cuda_image_cache_->insert(key, value);
        cuda_image_cache_->unlock(index_hash);
    }

    ++processed_cuda_batch_count_;

    cuda_batch_cond_.notify_all();
    return request_count;
}

uint32_t RocJpegProcessor::wait_batch(const uint32_t index_in_task,
                                     std::deque<uint32_t>& batch_item_counts,
                                     const uint32_t num_remaining_patches)
{
    // Check if the next (cuda) batch needs to be requested whenever an index in a task is divided by cuda batch size.
    // (each task which is for a patch consists of multiple tile processing)
    if (index_in_task % cuda_batch_size_ == 0)
    {
        // Call request and propagate its result
        return request(batch_item_counts, num_remaining_patches);
    }
    return 0;
}

std::shared_ptr<cucim::cache::ImageCacheValue> RocJpegProcessor::wait_for_processing(const uint32_t index)
{
    uint64_t index_hash = cucim::codec::splitmix64(index);
    std::mutex* m = reinterpret_cast<std::mutex*>(cuda_image_cache_->mutex(index_hash));
    std::shared_ptr<cucim::cache::ImageCacheValue> value;

    std::unique_lock<std::mutex> lock(*m);
    cuda_batch_cond_.wait(lock, [this, index, &value] {
        // Exit waiting if the thread needs to be stopped or cache value is available.
        if (stopped_)
        {
            value = std::shared_ptr<cucim::cache::ImageCacheValue>();
            return true;
        }
        auto key = cuda_image_cache_->create_key(ifd_->hash_value(), index);
        value = cuda_image_cache_->find(key);
        return static_cast<bool>(value);
    });
    return value;
}

void RocJpegProcessor::shutdown()
{
    stopped_ = true;
    cuda_batch_cond_.notify_all();
}

uint32_t RocJpegProcessor::preferred_loader_prefetch_factor()
{
    return preferred_loader_prefetch_factor_;
}

} // namespace cuslide::loader
