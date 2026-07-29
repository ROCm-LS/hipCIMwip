/*
 * SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <string>
#include <vector>

#include <functional>

#include <catch2/catch_test_macros.hpp>

#include "config.h"
#include "cucim/cuimage.h"
#include "cucim/io/device.h"
#include "cucim/memory/dlpack.h"

// Exercises cucim::CuImage C++-only entry points not reached through the Python
// pybind layer: detect_format(), the two-arg constructor, plural ResolutionInfo
// accessors, DLDataType comparisons, iterators, and the in-memory crop path.

SCENARIO("CuImage format detection and metadata accessors", "[test_cuimage.cpp]")
{
    const std::string input_path = g_config.get_input_path();

    GIVEN("A generic TIFF input path")
    {
        WHEN("detect_format() is called")
        {
            const cucim::DetectedFormat format = cucim::detect_format(input_path);
            THEN("It reports a non-empty format and at least one plugin")
            {
                REQUIRE_FALSE(format.first.empty());
                REQUIRE_FALSE(format.second.empty());
            }
        }

        WHEN("The image is opened with CuImage(path)")
        {
            cucim::CuImage image(input_path);

            THEN("Its metadata is parsed and available")
            {
                // Opening parses immediately (is_loaded()==true). Per its
                // documented contract (see CuImage::operator bool() in
                // cuimage.h), operator bool() reports only the detected-but-not-
                // yet-parsed state, so it is intentionally false once loaded;
                // this pins that behaviour and is not a validity check.
                REQUIRE(image.is_loaded());
                REQUIRE_FALSE(static_cast<bool>(image));
                REQUIRE(image.dims() == "YXC");
                REQUIRE(image.ndim() >= 2);
            }

            AND_THEN("The plural ResolutionInfo accessors are self-consistent")
            {
                const cucim::ResolutionInfo res = image.resolutions();
                const uint16_t level_count = res.level_count();
                REQUIRE(level_count >= 1);

                const std::vector<int64_t>& level_dims = res.level_dimensions();
                const std::vector<uint32_t>& tile_sizes = res.level_tile_sizes();

                // Both plural vectors hold level_count * level_ndim entries.
                REQUIRE_FALSE(level_dims.empty());
                REQUIRE_FALSE(tile_sizes.empty());
                REQUIRE(level_dims.size() % level_count == 0);
                REQUIRE(tile_sizes.size() % level_count == 0);

                // The plural vector must agree with the singular accessor.
                const std::vector<int64_t> dim0 = res.level_dimension(0);
                REQUIRE(dim0.size() == level_dims.size() / level_count);
                for (size_t i = 0; i < dim0.size(); ++i)
                {
                    REQUIRE(level_dims[i] == dim0[i]);
                }
            }

            AND_THEN("DLDataType equality and inequality behave consistently")
            {
                const DLDataType a = image.dtype();
                const DLDataType b = image.dtype();
                REQUIRE(a == b);
                REQUIRE_FALSE(a != b);

                DLDataType c = a;
                c.bits = static_cast<uint8_t>(a.bits + 1);
                REQUIRE(a != c);
                REQUIRE_FALSE(a == c);
            }
        }

        WHEN("The two-argument CuImage(path, plugin_name) constructor is used")
        {
            // Stub overload: must construct/destroy safely and stay unloaded.
            cucim::CuImage image(input_path, std::string("hipcim.kit.hipslide"));
            THEN("The resulting image is not loaded")
            {
                REQUIRE_FALSE(image.is_loaded());
                REQUIRE_FALSE(static_cast<bool>(image));
            }
        }
    }
}

SCENARIO("CuImage iteration and in-memory crop", "[test_cuimage.cpp]")
{
    const std::string input_path = g_config.get_input_path();
    cucim::CuImage image(input_path);

    GIVEN("A region read from the file into host memory")
    {
        // 256x256 region at level 0; default workers -> non-batched path.
        auto region = std::make_shared<cucim::CuImage>(
            image.read_region({ 0, 0 }, { 256, 256 }, /*level=*/0));

        REQUIRE(region->is_loaded());
        REQUIRE(region->dims() == "YXC");

        WHEN("Iterating with the non-const iterator")
        {
            uint64_t visited = 0;
            for (auto it = region->begin(); it != region->end(); ++it)
            {
                const std::shared_ptr<cucim::CuImage> current = *it; // operator*
                REQUIRE(current != nullptr);
                REQUIRE(it->is_loaded()); // operator->
                REQUIRE(it.size() >= 1); // iterator batch count
                ++visited;
            }

            THEN("Exactly one batch is visited")
            {
                REQUIRE(visited == 1);
            }

            AND_THEN("Postfix increment advances to end and reports the batch index")
            {
                auto it = region->begin();
                REQUIRE(it.index() == 0);
                auto prev = it++; // operator++(int)
                REQUIRE(prev != it);
                REQUIRE(it == region->end());
            }
        }

        WHEN("Iterating with the const iterator")
        {
            const cucim::CuImage& const_region = *region;
            uint64_t visited = 0;
            for (auto it = const_region.begin(); it != const_region.end(); ++it)
            {
                REQUIRE((*it) != nullptr);
                REQUIRE(it->is_loaded()); // operator->
                REQUIRE(it.size() >= 1); // size()
                ++visited;
            }
            THEN("Exactly one batch is visited")
            {
                REQUIRE(visited == 1);
            }
            AND_THEN("const iterator index, postfix increment and equality work")
            {
                auto it = const_region.begin();
                REQUIRE(it.index() == 0); // index()
                auto prev = it++; // operator++(int)
                REQUIRE(prev != it);
                REQUIRE(it == const_region.end()); // operator==
            }
        }

        WHEN("read_region() is called again on the in-memory region")
        {
            // image_data_ is already populated, so this takes the crop_image()
            // path rather than reading from the file handle.
            cucim::CuImage cropped = region->read_region({ 0, 0 }, { 64, 64 }, /*level=*/0);

            THEN("The cropped region has the requested shape")
            {
                REQUIRE(cropped.is_loaded());
                const cucim::Shape shape = cropped.shape();
                REQUIRE(shape.size() >= 2);
                // dims are "YXC": shape[0]=Y(height), shape[1]=X(width).
                REQUIRE(shape[0] == 64);
                REQUIRE(shape[1] == 64);
            }
        }
    }
}

SCENARIO("CuImage batched multi-worker read exercises the thread loader", "[test_cuimage.cpp]")
{
    const std::string input_path = g_config.get_input_path();
    cucim::CuImage image(input_path);

    GIVEN("A multi-location region request with a worker thread")
    {
        // num_workers=1 routes through ThreadBatchDataLoader (num_workers>0).
        auto region = std::make_shared<cucim::CuImage>(
            image.read_region({ 0, 0, 0, 0 }, { 256, 256 }, /*level=*/0, /*num_workers=*/1));

        THEN("A thread batch data loader is created")
        {
            REQUIRE(region->loader() != nullptr);
        }

        AND_THEN("Iterating advances the loader through every batch")
        {
            auto it = region->begin();
            const uint64_t total = it.size();
            REQUIRE(total >= 1);

            uint64_t seen = 0;
            for (; it != region->end(); ++it)
            {
                REQUIRE((*it) != nullptr);
                ++seen;
                if (seen > total + 2) // safety valve against a non-terminating loop
                {
                    break;
                }
            }
            REQUIRE(seen == total);
        }
    }
}

SCENARIO("CuImage DLPack container, Device helpers and DLDataType hash", "[test_cuimage.cpp]")
{
    const std::string input_path = g_config.get_input_path();
    cucim::CuImage image(input_path);

    GIVEN("A loaded host region")
    {
        auto region = std::make_shared<cucim::CuImage>(
            image.read_region({ 0, 0 }, { 128, 128 }, /*level=*/0));
        REQUIRE(region->is_loaded());

        WHEN("The DLTContainer is obtained via container()")
        {
            const cucim::memory::DLTContainer container = region->container();

            THEN("Its accessors report a consistent, non-empty tensor")
            {
                // operator bool() -> a loaded region has backing data.
                REQUIRE(static_cast<bool>(container));

                // size() multiplies shape extents by the element byte width.
                const size_t byte_size = container.size();
                REQUIRE(byte_size > 0);

                // For an 8-bit RGB region this is width * height * channels.
                REQUIRE(byte_size == 128u * 128u * 3u);

                // numpy_dtype()/dtype() describe the same element type.
                REQUIRE(std::string(container.numpy_dtype()) == "|u1");
                const DLDataType ctype = container.dtype();
                REQUIRE(ctype.bits == region->dtype().bits);

                // No shared-memory name for a plain host allocation.
                REQUIRE(container.shm_name() == nullptr);
            }

            AND_THEN("The DLTensor conversion operators expose the raw handle")
            {
                // operator DLTensor*() returns the underlying handle...
                DLTensor* handle = static_cast<DLTensor*>(container);
                REQUIRE(handle != nullptr);
                REQUIRE(handle->data != nullptr);

                // ...and operator DLTensor() returns a value copy of it.
                const DLTensor value = static_cast<DLTensor>(container);
                REQUIRE(value.ndim == handle->ndim);
                REQUIRE(value.data == handle->data);
            }
        }
    }

    GIVEN("Directly constructed io::Device values")
    {
        WHEN("Using the (type, index) constructor and set_values()")
        {
            cucim::io::Device dev(cucim::io::DeviceType::kCUDA, 0);
            THEN("The type and index round-trip")
            {
                REQUIRE(dev.type() == cucim::io::DeviceType::kCUDA);
                REQUIRE(dev.index() == 0);
            }

            AND_THEN("set_values() mutates the device in place")
            {
                dev.set_values(cucim::io::DeviceType::kCPU, -1);
                REQUIRE(dev.type() == cucim::io::DeviceType::kCPU);
            }
        }
    }
}

SCENARIO("CuImage CUDA device read, save and re-crop", "[test_cuimage.cpp][gpu]")
{
    // Device-resident save() and the crop_image() CUDA branches, never taken by
    // the CPU-only Python suite. GPU-guarded: skipped when no accelerator.
    const std::string input_path = g_config.get_input_path();
    cucim::CuImage image(input_path);

    std::shared_ptr<cucim::CuImage> dev_tile;
    bool gpu_ok = true;
    try
    {
        // device="cuda" -> batch loader path; the container must be a shared_ptr
        // for the iterator's shared_from_this. Advancing materializes on the GPU.
        auto dev = std::make_shared<cucim::CuImage>(
            image.read_region({ 0, 0 }, { 256, 256 }, /*level=*/0, /*num_workers=*/0,
                              /*batch_size=*/1, /*drop_last=*/false, /*prefetch_factor=*/2,
                              /*shuffle=*/false, /*seed=*/0, {}, cucim::io::Device("cuda")));
        auto it = dev->begin();
        dev_tile = *it;
    }
    catch (const std::exception& e)
    {
        gpu_ok = false;
        WARN("GPU device read unavailable, skipping: " << e.what());
    }

    if (gpu_ok && dev_tile && dev_tile->is_loaded() &&
        dev_tile->device().type() == cucim::io::DeviceType::kCUDA)
    {
        GIVEN("A device-resident region")
        {
            WHEN("It is saved to a PPM file")
            {
                const std::string tmp = g_config.temp_folder + "/cuimage_dev.ppm";
                THEN("save() copies the raster device->host without error")
                {
                    REQUIRE_NOTHROW(dev_tile->save(tmp));
                }
            }

            WHEN("It is re-cropped to a host output")
            {
                // crop_image(): in_device=CUDA, out_device=CPU (bulk D2H + per-row).
                auto host_crop = dev_tile->read_region({ 0, 0 }, { 64, 64 }, /*level=*/0);
                THEN("A host raster of the requested size is returned")
                {
                    REQUIRE(host_crop.is_loaded());
                    const cucim::Shape shape = host_crop.shape();
                    REQUIRE(shape[0] == 64);
                    REQUIRE(shape[1] == 64);
                }
            }

            WHEN("It is re-cropped to a device output")
            {
                // crop_image(): CUDA in, CUDA out (D2D + move_raster_from_device).
                auto dev_crop = dev_tile->read_region({ 0, 0 }, { 64, 64 }, /*level=*/0,
                                                      /*num_workers=*/0, /*batch_size=*/1,
                                                      /*drop_last=*/false, /*prefetch_factor=*/2,
                                                      /*shuffle=*/false, /*seed=*/0, {},
                                                      cucim::io::Device("cuda"));
                THEN("A device-resident crop is produced")
                {
                    REQUIRE(dev_crop.device().type() == cucim::io::DeviceType::kCUDA);
                }
            }
        }
    }
}

SCENARIO("CuImage rejects invalid levels and reports associated images", "[test_cuimage.cpp]")
{
    cucim::CuImage image(g_config.get_input_path());
    const cucim::ResolutionInfo res = image.resolutions();
    const uint16_t lc = res.level_count();

    THEN("Resolution accessors throw for out-of-range levels")
    {
        REQUIRE_THROWS_AS(res.level_dimension(lc + 3), std::invalid_argument);
        REQUIRE_THROWS_AS(res.level_downsample(lc + 3), std::invalid_argument);
        REQUIRE_THROWS_AS(res.level_tile_size(lc + 3), std::invalid_argument);
    }

    THEN("read_region throws for an out-of-range level")
    {
        REQUIRE_THROWS(image.read_region({ 0, 0 }, { 16, 16 }, static_cast<uint16_t>(lc + 3)));
    }

    THEN("associated_images() is queryable")
    {
        // Generated stripe pyramids carry no associated (label/macro) images.
        const std::set<std::string> names = image.associated_images();
        REQUIRE(names.empty());
    }

    THEN("spacing()/spacing_units() accept an explicit dim order with unknown axes")
    {
        // 'Q' is not a real axis, so those entries fall back to 1.0 / \"\".
        const std::vector<float> sp = image.spacing("XYQ");
        REQUIRE(sp.size() == 3);
        REQUIRE(sp[2] == 1.0f);
        const std::vector<std::string> su = image.spacing_units("XYQ");
        REQUIRE(su.size() == 3);
    }
}

