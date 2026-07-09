#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Modifications Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
# cmake-format: on
#

if (NOT TARGET deps::googlebenchmark)
    FetchContent_Declare(
            deps-googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.5.5
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
    )

    message(STATUS "Fetching googlebenchmark sources")

    # Create static library
    cucim_set_build_shared_libs(OFF)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
    FetchContent_MakeAvailable(deps-googlebenchmark)
    message(STATUS "Fetching googlebenchmark sources - done")

    # Suppress -Wc2y-extensions error: Clang 23+ treats __COUNTER__ (used in
    # benchmark.h) as a C2y extension, which -pedantic-errors promotes to a
    # hard error in benchmark v1.5.5.
    target_compile_options(benchmark PRIVATE -Wno-c2y-extensions)
    target_compile_options(benchmark_main PRIVATE -Wno-c2y-extensions)

    cucim_restore_build_shared_libs()

    add_library(deps::googlebenchmark INTERFACE IMPORTED GLOBAL)
    target_link_libraries(deps::googlebenchmark INTERFACE benchmark::benchmark)
    # Propagate -Wno-c2y-extensions to consumers that include benchmark.h
    target_compile_options(deps::googlebenchmark INTERFACE -Wno-c2y-extensions)
    set(deps-googlebenchmark_SOURCE_DIR ${deps-googlebenchmark_SOURCE_DIR} CACHE INTERNAL "" FORCE)
    mark_as_advanced(deps-googlebenchmark_SOURCE_DIR)
endif ()
