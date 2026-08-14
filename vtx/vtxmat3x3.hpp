// MIT License
//
// Copyright (c) 2026 SpectreFestival
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * @file      vtxmat3x3.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     3x3 matrix type aliases.
 *
 * @defgroup  vtx_matrix Matrices
 * @ingroup   vtx
 * @brief     3x3 matrix aliases for all arithmetic types.
 */

#ifndef VERTEX_MATRIX3X3_HPP
#define VERTEX_MATRIX3X3_HPP
#pragma once

#include "vtxbasic.hpp"    ///< arithmetic_t, vi32, vf32, etc.
#include "vtxmatcxr.hpp"   ///< matrix_template<T, R, C>

namespace vtx {

    /// @brief 3x3 matrix template alias.
    template <arithmetic_t Ty>
    using mat3x3 = matrix_template<Ty, 3, 3>;

    //==========================================================================
    // Type-specific 3x3 matrix aliases
    //==========================================================================

    using mat3x3i08 = mat3x3<vi08>;  ///< 3x3 matrix of signed 8-bit integers
    using mat3x3i16 = mat3x3<vi16>;  ///< 3x3 matrix of signed 16-bit integers
    using mat3x3i32 = mat3x3<vi32>;  ///< 3x3 matrix of signed 32-bit integers
    using mat3x3i64 = mat3x3<vi64>;  ///< 3x3 matrix of signed 64-bit integers

    using mat3x3u08 = mat3x3<vu08>;  ///< 3x3 matrix of unsigned 8-bit integers
    using mat3x3u16 = mat3x3<vu16>;  ///< 3x3 matrix of unsigned 16-bit integers
    using mat3x3u32 = mat3x3<vu32>;  ///< 3x3 matrix of unsigned 32-bit integers
    using mat3x3u64 = mat3x3<vu64>;  ///< 3x3 matrix of unsigned 64-bit integers

    using mat3x3f32 = mat3x3<vf32>;  ///< 3x3 matrix of 32-bit floats
    using mat3x3f64 = mat3x3<vf64>;  ///< 3x3 matrix of 64-bit floats

    //==========================================================================
    // Default convenience aliases
    //==========================================================================

    using mat3x3i = mat3x3i32;       ///< Default signed 3x3 matrix (32-bit)
    using mat3x3u = mat3x3u32;       ///< Default unsigned 3x3 matrix (32-bit)
    using mat3x3f = mat3x3f32;       ///< Default float 3x3 matrix (32-bit)
    using mat3x3d = mat3x3f64;       ///< Default double 3x3 matrix (64-bit)

} // namespace vtx

#endif // VERTEX_MATRIX3X3_HPP