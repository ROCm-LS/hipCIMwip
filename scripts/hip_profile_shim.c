/*
 * SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * hip_profile_shim.c -- LD_PRELOAD interposer that makes host-only LLVM
 * source-based coverage work on HIP code.
 *
 * Problem
 * -------
 * When AMD clang compiles a HIP translation unit with -fprofile-instr-generate
 * (even scoped to the host via -Xarch_host), it emits an offload "profile
 * shadow" variable named __llvm_profile_sections_<cuid> and registers it with
 * the HIP runtime via __hipRegisterVar. This is part of AMD's optional GPU
 * coverage support: the shadow variable is meant to have a matching device-side
 * symbol so device counters can be located.
 *
 * With host-only instrumentation there is no device-side __llvm_profile_sections
 * symbol, so when the HIP runtime tears down the fat binary it fails to bind the
 * registered variable and aborts the whole process:
 *
 *     :.../hipamd/src/hip_global.cpp:209 : Cannot create GlobalVar Obj for
 *         symbol: __llvm_profile_sections_<cuid>
 *     Aborted (core dumped)
 *
 * There is no compiler flag to suppress this registration, so this shim drops it
 * at runtime instead.
 *
 * Behavior
 * --------
 * __hipRegisterVar is interposed. Registrations for __llvm_profile_sections*
 * symbols are dropped; every other registration (genuine __device__ /
 * __constant__ / __managed__ variables) is forwarded unchanged to the real HIP
 * runtime via dlsym(RTLD_NEXT). Dropping the shadow registration is safe: host
 * coverage counters are written by the LLVM profile runtime's atexit handler,
 * which is independent of HIP variable registration.
 *
 * Build:  gcc -O2 -fPIC -shared -o hip_profile_shim.so hip_profile_shim.c -ldl
 * Use:    LD_PRELOAD=hip_profile_shim.so <instrumented program>
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Matches the HIP runtime ABI for __hipRegisterVar (all args are pointers/ints,
 * so the exact const-qualification of the name arguments is irrelevant). */
typedef void (*hip_register_var_fn)(void **modules, char *var,
                                    const char *host_name,
                                    const char *device_name, int ext,
                                    unsigned long size, int constant,
                                    int global);

static const char kProfileShadowPrefix[] = "__llvm_profile_sections";

void __hipRegisterVar(void **modules, char *var, const char *host_name,
                      const char *device_name, int ext, unsigned long size,
                      int constant, int global)
{
    const unsigned long prefix_len = sizeof(kProfileShadowPrefix) - 1;

    /* Drop host-only LLVM coverage shadow variables: they have no device-side
     * counterpart and would abort the HIP runtime at fat-binary teardown. */
    if ((host_name && strncmp(host_name, kProfileShadowPrefix, prefix_len) == 0) ||
        (device_name && strncmp(device_name, kProfileShadowPrefix, prefix_len) == 0)) {
        return;
    }

    static hip_register_var_fn real_register_var = NULL;
    if (!real_register_var) {
        real_register_var =
            (hip_register_var_fn)dlsym(RTLD_NEXT, "__hipRegisterVar");
        if (!real_register_var) {
            fprintf(stderr,
                    "hip_profile_shim: dlsym(RTLD_NEXT, \"__hipRegisterVar\") "
                    "failed: %s\n", dlerror());
            abort();
        }
    }
    real_register_var(modules, var, host_name, device_name, ext, size,
                      constant, global);
}
