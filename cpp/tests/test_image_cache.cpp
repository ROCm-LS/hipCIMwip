/*
 * SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "cucim/cache/cache_type.h"
#include "cucim/cache/image_cache.h"
#include "cucim/cache/image_cache_config.h"
#include "cucim/cache/image_cache_manager.h"
#include "cucim/io/device_type.h"

// Drives the per-process FIFO image cache directly: type_str()/mutex()/find()
// and FIFO eviction, which the Python suite only reaches indirectly.

SCENARIO("PerProcess image cache operations", "[test_image_cache.cpp]")
{
    using namespace cucim::cache;

    ImageCacheConfig config;
    config.type = CacheType::kPerProcess;
    config.memory_capacity = 1; // 1 MiB
    config.capacity = 8;
    config.record_stat = true;

    auto cache = ImageCacheManager::create_cache(config);
    REQUIRE(cache != nullptr);

    GIVEN("A freshly created per-process cache")
    {
        THEN("It reports per-process type and configured capacities")
        {
            REQUIRE(cache->type() == CacheType::kPerProcess);
            REQUIRE(std::string(cache->type_str()) == "per_process");
            REQUIRE(cache->device_type() == cucim::io::DeviceType::kCPU);
            REQUIRE(cache->capacity() == 8);
            REQUIRE(cache->memory_capacity() == static_cast<uint64_t>(1) * 1024 * 1024);
            REQUIRE(cache->size() == 0);
            REQUIRE(cache->free_memory() == cache->memory_capacity());
        }

        WHEN("A value is inserted, looked up and mutated via the stat/mutex API")
        {
            cache->record(true);
            REQUIRE(cache->record() == true);

            const uint64_t value_size = 64 * 1024; // 64 KiB
            auto key = cache->create_key(/*file_hash=*/1, /*index=*/1);
            void* buf = cache->allocate(value_size);
            REQUIRE(buf != nullptr);
            auto value = cache->create_value(buf, value_size);
            REQUIRE(static_cast<bool>(*value));

            REQUIRE(cache->insert(key, value));
            REQUIRE(cache->size() == 1);
            REQUIRE(cache->memory_size() == value_size);

            THEN("find() records a hit for a present key and a miss for an absent one")
            {
                auto hit = cache->find(key);
                REQUIRE(hit != nullptr);

                auto absent_key = cache->create_key(999, 999);
                auto miss = cache->find(absent_key);
                REQUIRE(miss == nullptr);

                REQUIRE(cache->hit_count() >= 1);
                REQUIRE(cache->miss_count() >= 1);
            }

            AND_THEN("The per-index mutex pool and lock helpers are usable")
            {
                REQUIRE(cache->mutex(1) != nullptr);
                cache->lock(2);
                cache->unlock(2);
            }
        }

        WHEN("More data than the memory budget is inserted")
        {
            // ~256 KiB values against a 1 MiB budget force FIFO eviction.
            const uint64_t value_size = 256 * 1024;
            for (uint64_t i = 0; i < 10; ++i)
            {
                auto key = cache->create_key(1, i + 1);
                // allocate() can return nullptr under memory pressure; fail the
                // test clearly here rather than wrapping/inserting a null value
                // that could be dereferenced later.
                void* buf = cache->allocate(value_size);
                REQUIRE(buf != nullptr);
                auto value = cache->create_value(buf, value_size);
                cache->insert(key, value);
            }

            THEN("The cache stays within its memory capacity")
            {
                REQUIRE(cache->memory_size() <= cache->memory_capacity());
                REQUIRE(cache->size() >= 1);
                REQUIRE(cache->free_memory() <= cache->memory_capacity());
            }
        }
    }
}

SCENARIO("PerProcess image cache on a CUDA device", "[test_image_cache.cpp][gpu]")
{
    using namespace cucim::cache;

    ImageCacheConfig config;
    config.type = CacheType::kPerProcess;
    config.memory_capacity = 1; // 1 MiB
    config.capacity = 8;

    std::unique_ptr<ImageCache> cache;
    bool gpu_ok = true;
    try
    {
        // A CUDA-backed cache routes allocate()/free through cudaMalloc/cudaFree.
        cache = ImageCacheManager::create_cache(config, cucim::io::DeviceType::kCUDA);
    }
    catch (const std::exception& e)
    {
        gpu_ok = false;
        WARN("GPU unavailable, skipping CUDA cache test: " << e.what());
    }

    if (gpu_ok && cache)
    {
        GIVEN("A CUDA per-process cache")
        {
            REQUIRE(cache->device_type() == cucim::io::DeviceType::kCUDA);

            WHEN("Device values are allocated, inserted and evicted")
            {
                const uint64_t vsize = 256 * 1024; // 256 KiB against a 1 MiB budget
                void* buf = cache->allocate(vsize); // cudaMalloc path
                REQUIRE(buf != nullptr);
                auto key = cache->create_key(1, 1);
                auto value = cache->create_value(buf, vsize, cucim::io::DeviceType::kCUDA);
                REQUIRE(cache->insert(key, value));
                REQUIRE(cache->find(key) != nullptr);

                // Overflow the budget so evicted device values are cudaFree'd.
                for (uint64_t i = 0; i < 12; ++i)
                {
                    auto k = cache->create_key(1, i + 2);
                    auto v = cache->create_value(
                        cache->allocate(vsize), vsize, cucim::io::DeviceType::kCUDA);
                    cache->insert(k, v);
                }

                THEN("The cache honours its memory budget")
                {
                    REQUIRE(cache->memory_size() <= cache->memory_capacity());
                }
            }
        }
    }
}

SCENARIO("ImageCacheManager reserve and preferred capacity", "[test_image_cache.cpp]")
{
    using namespace cucim::cache;

    GIVEN("A manager holding a per-process cache")
    {
        ImageCacheManager mgr;
        ImageCacheConfig cfg;
        cfg.type = CacheType::kPerProcess;
        cfg.memory_capacity = 1;
        cfg.capacity = 8;
        auto cache = mgr.cache(cfg);
        REQUIRE(cache != nullptr);

        WHEN("Capacity is reserved via the manager")
        {
            mgr.reserve(4); // grow memory capacity (MiB)
            mgr.reserve(8, 32); // grow memory + entry capacity
            THEN("The cache reflects a non-shrinking memory capacity")
            {
                REQUIRE(mgr.cache().memory_capacity() >= static_cast<uint64_t>(1) * 1024 * 1024);
            }
        }

        AND_THEN("preferred_memory_capacity() computes a positive MiB estimate")
        {
            const uint32_t mc = preferred_memory_capacity(
                { 1024, 1024 }, { 256, 256 }, { 256, 256 }, /*bytes_per_pixel=*/3);
            REQUIRE(mc >= 1);
        }
    }
}
