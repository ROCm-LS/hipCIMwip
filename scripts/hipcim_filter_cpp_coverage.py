# SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filter an llvm-cov JSON export to only the files listed in a diff list.

llvm-cov export records absolute build-time paths (e.g.
/src/hipcim/cpp/src/cucim/loader/cuimage_loader.cpp) while the diff list
produced by `git diff --name-only` contains repository-relative paths (e.g.
cpp/src/cucim/loader/cuimage_loader.cpp).  This script matches entries using
str.endswith() so that the path prefix introduced by the build environment is
irrelevant.

Usage:
    python3 hipcim_filter_cpp_coverage.py <difflist> <input_json> <output_json>

Arguments:
    difflist     Path to difflist.txt (one relative source path per line).
    input_json   Path to the full llvm-cov JSON export.
    output_json  Path to write the filtered JSON (may be the same as input_json).
"""

import json
import sys


def _recompute_totals(orig_totals, files):
    """Re-aggregate an llvm-cov 'totals' block over the retained files so a
    consumer reading 'totals' sees diff-scoped coverage, not whole-repo. Uses
    the original totals as the metric/sub-key template; 'percent' is re-derived
    as 100*covered/count (0 when count is 0, matching llvm-cov).
    """
    new_totals = {}
    for metric, orig in orig_totals.items():
        if not isinstance(orig, dict):
            new_totals[metric] = orig
            continue
        agg = {key: 0 for key in orig if key != "percent"}
        for file_entry in files:
            summary = file_entry.get("summary", {}).get(metric, {})
            for key in agg:
                value = summary.get(key, 0)
                if isinstance(value, (int, float)):
                    agg[key] += value
        if "percent" in orig:
            count = agg.get("count", 0)
            agg["percent"] = (100.0 * agg.get("covered", 0) / count) if count else 0.0
        new_totals[metric] = agg
    return new_totals


def main():
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <difflist> <input_json> <output_json>",
            file=sys.stderr,
        )
        sys.exit(1)

    difflist_path, input_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(difflist_path) as f:
        diff_paths = [line.strip() for line in f if line.strip()]

    if not diff_paths:
        print("Diff list is empty; writing unfiltered coverage JSON.")
        if input_path != output_path:
            import shutil
            shutil.copy2(input_path, output_path)
        return

    with open(input_path) as f:
        data = json.load(f)

    total_before = sum(len(entry["files"]) for entry in data["data"])

    for entry in data["data"]:
        entry["files"] = [
            file_entry for file_entry in entry["files"]
            if any(file_entry["filename"].endswith(p) for p in diff_paths)
        ]
        # Filter functions to match retained files
        retained = {file_entry["filename"] for file_entry in entry["files"]}
        entry["functions"] = [
            fn for fn in entry.get("functions", [])
            if fn.get("filenames") and any(f in retained for f in fn["filenames"])
        ]
        # Re-aggregate totals over the retained files so 'totals' reflects the
        # diff-scoped set rather than the whole codebase.
        if "totals" in entry:
            entry["totals"] = _recompute_totals(entry["totals"], entry["files"])

    total_after = sum(len(entry["files"]) for entry in data["data"])
    print(
        f"Filtered coverage: {total_before} -> {total_after} files "
        f"({len(diff_paths)} paths in diff list)"
    )

    with open(output_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
