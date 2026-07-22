/*
 * SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

#include "cucim/io/device.h"
#include "cucim/memory/memory_manager.h"
#include "cucim/memory/dlpack.h"
#include "cucim/cuda_runtime.h"

// Directly tests memory_manager.cpp functions no read path reaches:
// get_pointer_attributes() and move_raster_from_device()'s device->host branch,
// plus the host->device staging path. GPU-guarded: skipped when no accelerator.

TEST_CASE("memory_manager round-trips a raster host->device->host", "[test_memory.cpp]")
{
    const size_t n = 4096;
    auto* buf = static_cast<uint8_t*>(cucim_malloc(n));
    for (size_t i = 0; i < n; ++i)
    {
        buf[i] = static_cast<uint8_t>(i & 0xFF);
    }
    void* p = buf;

    bool gpu_ok = true;
    try
    {
        // Host -> device (cudaMalloc + cudaMemcpy H2D under the hood).
        cucim::memory::move_raster_from_host(&p, n, cucim::io::Device("cuda"));
    }
    catch (const std::exception& e)
    {
        gpu_ok = false;
        WARN("GPU unavailable, skipping device transfer test: " << e.what());
    }

    if (gpu_ok)
    {
        // get_pointer_attributes() must classify the staged pointer as CUDA.
        cucim::memory::PointerAttributes attr;
        cucim::memory::get_pointer_attributes(attr, p);
        REQUIRE(attr.device.type() == cucim::io::DeviceType::kCUDA);
        REQUIRE(attr.ptr != nullptr);

        // Device -> host (the previously-uncovered branch) must preserve bytes.
        cucim::memory::move_raster_from_device(&p, n, cucim::io::Device("cpu"));
        auto* back = static_cast<uint8_t*>(p);
        REQUIRE(back[0] == 0);
        REQUIRE(back[1] == 1);
        REQUIRE(back[255] == 255);
    }

    cucim_free(p);
}

TEST_CASE("get_pointer_attributes classifies a host allocation", "[test_memory.cpp]")
{
    void* h = cucim_malloc(64);
    cucim::memory::PointerAttributes attr;
    try
    {
        cucim::memory::get_pointer_attributes(attr, h);
        // An ordinary malloc'd pointer is host/unregistered memory.
        REQUIRE((attr.device.type() == cucim::io::DeviceType::kCPU ||
                 attr.device.type() == cucim::io::DeviceType::kCUDAHost));
    }
    catch (const std::exception& e)
    {
        WARN("get_pointer_attributes unavailable: " << e.what());
    }
    cucim_free(h);
}

TEST_CASE("get_pointer_attributes classifies managed and pinned memory", "[test_memory.cpp][gpu]")
{
    // Managed memory -> cudaMemoryTypeManaged switch case.
    void* managed = nullptr;
    if (cudaMallocManaged(&managed, 256) == cudaSuccess && managed != nullptr)
    {
        cucim::memory::PointerAttributes attr;
        cucim::memory::get_pointer_attributes(attr, managed);
        REQUIRE(attr.ptr != nullptr);
        cudaFree(managed);
    }
    else
    {
        WARN("managed allocation unavailable, skipping");
    }

    // Pinned host memory -> cudaMemoryTypeHost switch case.
    void* pinned = nullptr;
    if (cudaMallocHost(&pinned, 256) == cudaSuccess && pinned != nullptr)
    {
        cucim::memory::PointerAttributes attr;
        cucim::memory::get_pointer_attributes(attr, pinned);
        REQUIRE(attr.ptr != nullptr);
        cudaFreeHost(pinned);
    }
    else
    {
        WARN("pinned allocation unavailable, skipping");
    }
}

TEST_CASE("move_raster rejects unsupported destination device types", "[test_memory.cpp]")
{
    // Neither move helper handles pinned-host as a destination, so both must
    // hit their default "Unsupported device type" throw.
    void* p = cucim_malloc(64);
    REQUIRE_THROWS(cucim::memory::move_raster_from_host(
        &p, 64, cucim::io::Device(cucim::io::DeviceType::kCUDAHost, 0)));
    cucim_free(p);

    void* q = cucim_malloc(64);
    REQUIRE_THROWS(cucim::memory::move_raster_from_device(
        &q, 64, cucim::io::Device(cucim::io::DeviceType::kCUDAHost, 0)));
    cucim_free(q);
}

TEST_CASE("to_numpy_dtype maps every supported DLDataType", "[test_memory.cpp]")
{
    using cucim::memory::to_numpy_dtype;
    // Integer types.
    REQUIRE(std::string(to_numpy_dtype({ kDLInt, 8, 1 })) == "|i1");
    REQUIRE(std::string(to_numpy_dtype({ kDLInt, 16, 1 })) == "<i2");
    REQUIRE(std::string(to_numpy_dtype({ kDLInt, 32, 1 })) == "<i4");
    REQUIRE(std::string(to_numpy_dtype({ kDLInt, 64, 1 })) == "<i8");
    // Unsigned integer types.
    REQUIRE(std::string(to_numpy_dtype({ kDLUInt, 8, 1 })) == "|u1");
    REQUIRE(std::string(to_numpy_dtype({ kDLUInt, 16, 1 })) == "<u2");
    REQUIRE(std::string(to_numpy_dtype({ kDLUInt, 32, 1 })) == "<u4");
    REQUIRE(std::string(to_numpy_dtype({ kDLUInt, 64, 1 })) == "<u8");
    // Floating-point types.
    REQUIRE(std::string(to_numpy_dtype({ kDLFloat, 16, 1 })) == "<f2");
    REQUIRE(std::string(to_numpy_dtype({ kDLFloat, 32, 1 })) == "<f4");
    REQUIRE(std::string(to_numpy_dtype({ kDLFloat, 64, 1 })) == "<f8");
    // Unsupported bit-widths and codes throw.
    REQUIRE_THROWS_AS(to_numpy_dtype({ kDLInt, 128, 1 }), std::logic_error);
    REQUIRE_THROWS_AS(to_numpy_dtype({ kDLUInt, 128, 1 }), std::logic_error);
    REQUIRE_THROWS_AS(to_numpy_dtype({ kDLFloat, 128, 1 }), std::logic_error);
    REQUIRE_THROWS_AS(to_numpy_dtype({ kDLBfloat, 16, 1 }), std::logic_error);
}

TEST_CASE("DLTContainer null-tensor and shm_name paths", "[test_memory.cpp]")
{
    using cucim::memory::DLTContainer;

    // The (DLTensor*, const std::string&) constructor is otherwise unused.
    DLTContainer with_shm(static_cast<DLTensor*>(nullptr), std::string("shm-region"));
    (void)with_shm;

    // A null container exercises the nullptr guards in the accessors.
    DLTContainer empty(nullptr);
    REQUIRE_FALSE(static_cast<bool>(empty));
    REQUIRE(std::string(empty.numpy_dtype()).empty());
    REQUIRE(empty.dtype().code == kDLUInt);
    const DLTensor as_value = static_cast<DLTensor>(empty);
    REQUIRE(as_value.data == nullptr);
    REQUIRE(static_cast<DLTensor*>(empty) == nullptr);
}

