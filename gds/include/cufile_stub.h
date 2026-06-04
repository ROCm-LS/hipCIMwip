/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modifications Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef CUCIM_CUFILE_STUB_H
#define CUCIM_CUFILE_STUB_H

#include "cufile_amd.h"

#include "cucim/dynlib/helper.h"

class CuFileStub
{
public:
    void load();
    void unload();
    ~CuFileStub();

private:
    cucim::dynlib::LibraryHandle handle_ = nullptr;
};

#endif // CUCIM_CUFILE_STUB_H
