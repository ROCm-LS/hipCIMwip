#
# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Modifications Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
#

import pytest

# skip if imagecodecs package not available (needed by ImageGenerator utility)
pytest.importorskip("imagecodecs")


def test_get_nocache():
    from cucim import CuImage

    cache = CuImage.cache()

    assert int(cache.type) == 0
    assert cache.memory_size == 0
    assert cache.memory_capacity == 0
    assert cache.free_memory == 0
    assert cache.size == 0
    assert cache.capacity == 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    config = cache.config
    # Check essential properties
    #  {'type': 'nocache', 'memory_capacity': 1024, 'capacity': 5461,
    #   'mutex_pool_capacity': 11117, 'list_padding': 10000,
    #   'extra_shared_memory_size': 100, 'record_stat': False}
    assert config["type"] == "nocache"
    assert not config["record_stat"]


def test_get_per_process_cache():
    from cucim import CuImage

    cache = CuImage.cache("per_process", memory_capacity=2048)
    assert int(cache.type) == 1
    assert cache.memory_size == 0
    assert cache.memory_capacity == 2**20 * 2048
    assert cache.free_memory == 2**20 * 2048
    assert cache.size == 0
    assert cache.capacity > 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    config = cache.config
    # Check essential properties
    #  {'type': 'per_process', 'memory_capacity': 2048, 'capacity': 10922,
    #   'mutex_pool_capacity': 11117, 'list_padding': 10000,
    #   'extra_shared_memory_size': 100, 'record_stat': False}
    assert config["type"] == "per_process"
    assert config["memory_capacity"] == 2048
    assert not config["record_stat"]

def test_get_shared_memory_cache():
    from cucim import CuImage

    cache = CuImage.cache("shared_memory", memory_capacity=2048)
    assert int(cache.type) == 2
    assert cache.memory_size == 0
    # It allocates additional memory
    assert cache.memory_capacity > 2**20 * 2048
    assert cache.free_memory > 2**20 * 2048
    assert cache.size == 0
    assert cache.capacity > 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    config = cache.config
    # Check essential properties
    #  {'type': 'shared_memory', 'memory_capacity': 2048, 'capacity': 10922,
    #   'mutex_pool_capacity': 11117, 'list_padding': 10000,
    #   'extra_shared_memory_size': 100, 'record_stat': False}
    assert config["type"] == "shared_memory"
    assert config["memory_capacity"] == 2048
    assert not config["record_stat"]


def test_preferred_memory_capacity(testimg_tiff_stripe_32x24_16_jpeg):
    from cucim import CuImage
    from cucim.clara.cache import preferred_memory_capacity

    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)

    # same with `img.resolutions["level_dimensions"][0]`
    image_size = img.size("XY")  # 32x24
    tile_size = img.resolutions["level_tile_sizes"][0]  # 16x16
    patch_size = (tile_size[0] * 2, tile_size[0] * 2)
    bytes_per_pixel = 3  # default: 3

    # Below three statements are the same.
    memory_capacity = preferred_memory_capacity(img, patch_size=patch_size)
    memory_capacity2 = preferred_memory_capacity(
        None, image_size, tile_size, patch_size, bytes_per_pixel
    )
    memory_capacity3 = preferred_memory_capacity(
        None, image_size, patch_size=patch_size
    )

    assert memory_capacity == memory_capacity2  # 1 == 1
    assert memory_capacity2 == memory_capacity3  # 1 == 1

    # You can also manually set capacity` (e.g., `capacity=500`)
    cache = CuImage.cache("per_process", memory_capacity=memory_capacity)
    assert int(cache.type) == 1
    assert cache.memory_size == 0
    assert cache.memory_capacity == 2**20 * 1
    assert cache.free_memory == 2**20 * 1
    assert cache.size == 0
    assert cache.capacity > 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    basic_memory_capacity = preferred_memory_capacity(
        None,
        image_size=(1024 * 1024, 1024 * 1024),
        tile_size=(256, 256),
        patch_size=(256, 256),
        bytes_per_pixel=3,
    )
    assert basic_memory_capacity == 1536  # https://godbolt.org/z/jY7G84xzT


def test_reserve_more_cache_memory():
    from cucim import CuImage
    from cucim.clara.cache import preferred_memory_capacity

    memory_capacity = preferred_memory_capacity(
        None,
        image_size=(1024 * 1024, 1024 * 1024),
        tile_size=(256, 256),
        patch_size=(256, 256),
        bytes_per_pixel=3,
    )
    new_memory_capacity = preferred_memory_capacity(
        None,
        image_size=(1024 * 1024, 1024 * 1024),
        tile_size=(256, 256),
        patch_size=(512, 512),
        bytes_per_pixel=3,
    )

    cache = CuImage.cache("per_process", memory_capacity=memory_capacity)
    assert int(cache.type) == 1
    assert cache.memory_size == 0
    assert cache.memory_capacity == 2**20 * 1536
    assert cache.free_memory == 2**20 * 1536
    assert cache.size == 0
    assert cache.capacity > 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    cache.reserve(new_memory_capacity)
    assert int(cache.type) == 1
    assert cache.memory_size == 0
    assert cache.memory_capacity == 2**20 * 2304
    assert cache.free_memory == 2**20 * 2304
    assert cache.size == 0
    assert cache.capacity > 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    cache.reserve(memory_capacity, capacity=500)
    # Smaller `memory_capacity` value does not change this')
    assert int(cache.type) == 1
    assert cache.memory_size == 0
    assert cache.memory_capacity == 2**20 * 2304
    assert cache.free_memory == 2**20 * 2304
    assert cache.size == 0
    assert cache.capacity > 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0

    cache = CuImage.cache("no_cache")
    # Set new cache will reset memory size
    assert int(cache.type) == 0
    assert cache.memory_size == 0
    assert cache.memory_capacity == 0
    assert cache.free_memory == 0
    assert cache.size == 0
    assert cache.capacity == 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0


def test_cache_hit_miss(testimg_tiff_stripe_32x24_16_jpeg):
    from cucim import CuImage
    from cucim.clara.cache import preferred_memory_capacity

    # Reset to a known (empty) cache first so the hit/miss counts below do not
    # depend on cache state left over from earlier tests. gh-626 tracked an
    # isolation-order failure of this test; starting from "no_cache" makes it
    # deterministic regardless of execution order.
    CuImage.cache("no_cache")

    img = CuImage(testimg_tiff_stripe_32x24_16_jpeg)
    memory_capacity = preferred_memory_capacity(img, patch_size=(16, 16))
    cache = CuImage.cache(
        "per_process", memory_capacity=memory_capacity, record_stat=True
    )

    img.read_region((0, 0), (8, 8))
    assert (cache.hit_count, cache.miss_count) == (0, 1)

    _ = img.read_region((0, 0), (8, 8))
    assert (cache.hit_count, cache.miss_count) == (1, 1)

    _ = img.read_region((0, 0), (8, 8))
    assert (cache.hit_count, cache.miss_count) == (2, 1)
    assert cache.record()

    cache.record(False)
    assert not cache.record()

    _ = img.read_region((0, 0), (8, 8))
    assert (cache.hit_count, cache.miss_count) == (0, 0)

    assert int(cache.type) == 1
    assert cache.memory_size == 768
    assert cache.memory_capacity == 2**20 * 1
    assert cache.free_memory == 2**20 * 1 - 768
    assert cache.size == 1
    assert cache.capacity == 5

    cache = CuImage.cache("no_cache")

    assert int(cache.type) == 0
    assert cache.memory_size == 0
    assert cache.memory_capacity == 0
    assert cache.free_memory == 0
    assert cache.size == 0
    assert cache.capacity == 0


def test_cache_eviction(testimg_tiff_stripe_4096x4096_256_jpeg):
    import numpy as np

    from cucim import CuImage

    # Start from a clean cache so counts are independent of execution order.
    CuImage.cache("no_cache")

    img = CuImage(testimg_tiff_stripe_4096x4096_256_jpeg)
    tile_width, tile_height = img.resolutions["level_tile_sizes"][0]
    width, _ = img.size("XY")

    # A 1 MiB per-process cache holds only a handful of tiles, so reading more
    # distinct tiles than it can hold forces least-recently-used eviction
    # (exercises remove_front()/erase() in the per-process cache).
    cache = CuImage.cache("per_process", memory_capacity=1, record_stat=True)
    capacity = cache.capacity
    assert capacity > 0

    num_reads = capacity + 6
    for i in range(num_reads):
        x = (i * tile_width) % width
        y = ((i * tile_width) // width) * tile_height
        np.asarray(img.read_region((x, y), (tile_width, tile_height)))

    # Every distinct tile missed, and the cache never grew past its capacity,
    # so older entries must have been evicted.
    assert cache.hit_count == 0
    assert cache.miss_count >= num_reads
    assert 0 < cache.size <= capacity
    assert cache.memory_size <= cache.memory_capacity

    # Reset the global cache so later tests start from a known state.
    CuImage.cache("no_cache")
