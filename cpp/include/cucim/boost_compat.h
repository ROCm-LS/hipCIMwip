// SPDX-FileCopyrightText: Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef BOOST_COMPAT_H
#define BOOST_COMPAT_H

#include <boost/config.hpp>

#ifdef BOOST_HAS_LONG_LONG
    namespace boost {
        typedef long long long_long_type;
        typedef unsigned long long ulong_long_type;
    }
#endif

#endif // BOOST_COMPAT_H
