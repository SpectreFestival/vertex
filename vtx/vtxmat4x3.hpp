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
 * @file      vtxmat4x3.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     4x3 matrix type aliases.
 *
 * @defgroup  vtx_matrix Matrices
 * @ingroup   vtx
 * @brief     4x3 matrix aliases for all arithmetic types.
 *
 * @note      Matrix dimensions are (R rows, C columns) = (3 rows, 4 columns).
 *            The alias name "4x3" indicates the **logical** dimension (4 rows, 3 columns),
 *            but the template parameter order is matrix_template<Ty, R, C>.
 *            Therefore mat4x3 = matrix_template<Ty, 3, 4>.
 */

#ifndef VERTEX_MATRIX4X3_HPP
#define VERTEX_MATRIX4X3_HPP
#pragma once

#include "vtxbasic.hpp"    ///< arithmetic_t, vi32, vf32, etc.
#include "vtxmatcxr.hpp"   ///< matrix_template<T, R, C>

namespace vtx {

    /// @brief 4x3 matrix template alias (3 rows, 4 columns in storage).
    template <arithmetic_t Ty>
    using mat4x3 = matrix_template<Ty, 3, 4>;

    //==========================================================================
    // Type-specific 4x3 matrix aliases
    //==========================================================================

    using mat4x3i08 = mat4x3<vi08>;  ///< 4x3 matrix of signed 8-bit integers
    using mat4x3i16 = mat4x3<vi16>;  ///< 4x3 matrix of signed 16-bit integers
    using mat4x3i32 = mat4x3<vi32>;  ///< 4x3 matrix of signed 32-bit integers
    using mat4x3i64 = mat4x3<vi64>;  ///< 4x3 matrix of signed 64-bit integers

    using mat4x3u08 = mat4x3<vu08>;  ///< 4x3 matrix of unsigned 8-bit integers
    using mat4x3u16 = mat4x3<vu16>;  ///< 4x3 matrix of unsigned 16-bit integers
    using mat4x3u32 = mat4x3<vu32>;  ///< 4x3 matrix of unsigned 32-bit integers
    using mat4x3u64 = mat4x3<vu64>;  ///< 4x3 matrix of unsigned 64-bit integers

    using mat4x3f32 = mat4x3<vf32>;  ///< 4x3 matrix of 32-bit floats
    using mat4x3f64 = mat4x3<vf64>;  ///< 4x3 matrix of 64-bit floats

    //==========================================================================
    // Default convenience aliases
    //==========================================================================

    using mat4x3i = mat4x3i32;       ///< Default signed 4x3 matrix (32-bit)
    using mat4x3u = mat4x3u32;       ///< Default unsigned 4x3 matrix (32-bit)
    using mat4x3f = mat4x3f32;       ///< Default float 4x3 matrix (32-bit)
    using mat4x3d = mat4x3f64;       ///< Default double 4x3 matrix (64-bit)

} // namespace vtx

#endif // VERTEX_MATRIX4X3_HPP