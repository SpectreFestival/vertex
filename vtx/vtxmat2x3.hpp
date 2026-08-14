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
 * @file      vtxmat2x3.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     2x3 matrix type aliases.
 *
 * @defgroup  vtx_matrix Matrices
 * @ingroup   vtx
 * @brief     2x3 matrix aliases for all arithmetic types.
 *
 * @note      Matrix dimensions are (R rows, C columns) = (3 rows, 2 columns).
 *            The alias name "2x3" indicates the **logical** dimension (2 rows, 3 columns),
 *            but the template parameter order is matrix_template<Ty, R, C>.
 *            Therefore mat2x3 = matrix_template<Ty, 3, 2>.
 */

#ifndef VERTEX_MATRIX2X3_HPP
#define VERTEX_MATRIX2X3_HPP
#pragma once

#include "vtxbasic.hpp"    ///< arithmetic_t, vi32, vf32, etc.
#include "vtxmatcxr.hpp"   ///< matrix_template<T, R, C>

namespace vtx {

    /// @brief 2x3 matrix template alias (3 rows, 2 columns in storage).
    template <arithmetic_t Ty>
    using mat2x3 = matrix_template<Ty, 3, 2>;

    //==========================================================================
    // Type-specific 2x3 matrix aliases
    //==========================================================================

    using mat2x3i08 = mat2x3<vi08>;  ///< 2x3 matrix of signed 8-bit integers
    using mat2x3i16 = mat2x3<vi16>;  ///< 2x3 matrix of signed 16-bit integers
    using mat2x3i32 = mat2x3<vi32>;  ///< 2x3 matrix of signed 32-bit integers
    using mat2x3i64 = mat2x3<vi64>;  ///< 2x3 matrix of signed 64-bit integers

    using mat2x3u08 = mat2x3<vu08>;  ///< 2x3 matrix of unsigned 8-bit integers
    using mat2x3u16 = mat2x3<vu16>;  ///< 2x3 matrix of unsigned 16-bit integers
    using mat2x3u32 = mat2x3<vu32>;  ///< 2x3 matrix of unsigned 32-bit integers
    using mat2x3u64 = mat2x3<vu64>;  ///< 2x3 matrix of unsigned 64-bit integers

    using mat2x3f32 = mat2x3<vf32>;  ///< 2x3 matrix of 32-bit floats
    using mat2x3f64 = mat2x3<vf64>;  ///< 2x3 matrix of 64-bit floats

    //==========================================================================
    // Default convenience aliases
    //==========================================================================

    using mat2x3i = mat2x3i32;       ///< Default signed 2x3 matrix (32-bit)
    using mat2x3u = mat2x3u32;       ///< Default unsigned 2x3 matrix (32-bit)
    using mat2x3f = mat2x3f32;       ///< Default float 2x3 matrix (32-bit)
    using mat2x3d = mat2x3f64;       ///< Default double 2x3 matrix (64-bit)

} // namespace vtx

#endif // VERTEX_MATRIX2X3_HPP