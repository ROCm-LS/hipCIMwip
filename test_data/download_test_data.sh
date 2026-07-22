#!/bin/bash
# SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Downloads OpenSlide vendor slides for the cuslide vendor-path tests
# (tests/test_vendor_slides.cpp) to exercise tiff.cpp/ifd.cpp paths the generic
# fixtures cannot reach. Slides are large and not committed; tests SKIP when a
# slide is absent, so this is best-effort (a failure never breaks the build).
#
# Cached by expected size (present-and-correct reused, truncated re-fetched), so
# it is safe to run at docker build time or against a persistent cache. Set
# CUCIM_TESTDATA_DIR to fetch into a stable location (default test_data/private).
#
# CUCIM_TESTDATA_TIER selects what to fetch:
#   none  : nothing
#   small : Aperio JPEG              (default, ~2 MB)
#   medium: + Aperio JPEG 2000       (~66 MB)
#   large : + Philips TIFF           (~390 MB)
#
# Source: https://openslide.org/  (test data under CC0 / see LICENSE-3rdparty).

set -u

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
TOP="$(git rev-parse --show-toplevel 2>/dev/null || echo "${SCRIPT_DIR}/..")"
# Destination (default: test_data/private, where the tests look). Override with
# CUCIM_TESTDATA_DIR for a persistent cache; run_amd bridges it back into place.
DEST="${CUCIM_TESTDATA_DIR:-${TOP}/test_data/private}"
BASE_URL="https://openslide.cs.cmu.edu/download/openslide-testdata"
TIER="${CUCIM_TESTDATA_TIER:-small}"

mkdir -p "${DEST}"

# Manifest: "tier|relative_url|filename|expected_size". expected_size (bytes) is
# the cache key: a present file of a different size is re-fetched, not trusted.
SLIDES=(
    "small|Aperio/CMU-1-Small-Region.svs|CMU-1-Small-Region.svs|1938955"
    "medium|Aperio/JP2K-33003-1.svs|JP2K-33003-1.svs|63847265"
    "large|Philips-TIFF/Philips-1.tiff|Philips-1.tiff|326607275"
)

# Numeric rank for a tier name so a slide is fetched when its tier is at or below
# the requested tier. Unknown names fall back to "small".
tier_level() {
    case "$1" in
        none | off | NONE | OFF) echo 0 ;;
        small | SMALL)           echo 1 ;;
        medium | MEDIUM)         echo 2 ;;
        large | LARGE)           echo 3 ;;
        *)                       echo 1 ;;
    esac
}

file_size() {
    stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0
}

download_one() {
    local rel_url="$1"
    local name="$2"
    local expected_size="$3"
    local out="${DEST}/${name}"

    # Cache hit: already present and (when a size is known) the expected size.
    if [ -f "${out}" ]; then
        local have
        have="$(file_size "${out}")"
        if [ -z "${expected_size}" ] || [ "${expected_size}" = "0" ] || [ "${have}" = "${expected_size}" ]; then
            echo "[download_test_data] cached: ${name}"
            return 0
        fi
        echo "[download_test_data] stale ${name} (size ${have} != ${expected_size}); re-fetching" >&2
        rm -f "${out}"
    fi

    echo "[download_test_data] fetching ${name} ..."
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 --connect-timeout 20 -o "${out}.part" "${BASE_URL}/${rel_url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "${out}.part" "${BASE_URL}/${rel_url}"
    else
        echo "[download_test_data] WARN: neither curl nor wget available; skipping ${name}" >&2
        return 1
    fi
    local rc=$?

    if [ ${rc} -ne 0 ] || [ ! -s "${out}.part" ]; then
        rm -f "${out}.part"
        echo "[download_test_data] WARN: failed to download ${name} (tests will skip)" >&2
        return 1
    fi

    # Validate the download size before publishing it into the cache.
    if [ -n "${expected_size}" ] && [ "${expected_size}" != "0" ]; then
        local got
        got="$(file_size "${out}.part")"
        if [ "${got}" != "${expected_size}" ]; then
            echo "[download_test_data] WARN: ${name} size ${got} != expected ${expected_size}; discarding" >&2
            rm -f "${out}.part"
            return 1
        fi
    fi

    mv -f "${out}.part" "${out}"
    echo "[download_test_data] done: ${name}"
}

WANT="$(tier_level "${TIER}")"
if [ "${WANT}" -eq 0 ]; then
    echo "[download_test_data] CUCIM_TESTDATA_TIER=${TIER}; not downloading vendor slides."
    exit 0
fi

echo "[download_test_data] tier=${TIER}, destination=${DEST}"

for entry in "${SLIDES[@]}"; do
    IFS='|' read -r s_tier s_url s_name s_size <<< "${entry}"
    if [ "$(tier_level "${s_tier}")" -le "${WANT}" ]; then
        download_one "${s_url}" "${s_name}" "${s_size}" || true
    fi
done

exit 0
