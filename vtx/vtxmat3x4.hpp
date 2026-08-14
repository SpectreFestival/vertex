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
 * @file      vtxmat3x4.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     3x4 matrix type aliases.
 *
 * @defgroup  vtx_matrix Matrices
 * @ingroup   vtx
 * @brief     3x4 matrix aliases for all arithmetic types.
 *
 * @note      Matrix dimensions are (R rows, C columns) = (4 rows, 3 columns).
 *            The alias name "3x4" indicates the **logical** dimension (3 rows, 4 columns),
 *            but the template parameter order is matrix_template<Ty, R, C>.
 *            Therefore mat3x4 = matrix_template<Ty, 4, 3>.
 */

#ifndef VERTEX_MATRIX3X4_HPP
#define VERTEX_MATRIX3X4_HPP
#pragma once

#include "vtxbasic.hpp"    ///< arithmetic_t, vi32, vf32, etc.
#include "vtxmatcxr.hpp"   ///< matrix_template<T, R, C>

namespace vtx {

    /// @brief 3x4 matrix template alias (4 rows, 3 columns in storage).
    template <arithmetic_t Ty>
    using mat3x4 = matrix_template<Ty, 4, 3>;

    //==========================================================================
    // Type-specific 3x4 matrix aliases
    //==========================================================================

    using mat3x4i08 = mat3x4<vi08>;  ///< 3x4 matrix of signed 8-bit integers
    using mat3x4i16 = mat3x4<vi16>;  ///< 3x4 matrix of signed 16-bit integers
    using mat3x4i32 = mat3x4<vi32>;  ///< 3x4 matrix of signed 32-bit integers
    using mat3x4i64 = mat3x4<vi64>;  ///< 3x4 matrix of signed 64-bit integers

    using mat3x4u08 = mat3x4<vu08>;  ///< 3x4 matrix of unsigned 8-bit integers
    using mat3x4u16 = mat3x4<vu16>;  ///< 3x4 matrix of unsigned 16-bit integers
    using mat3x4u32 = mat3x4<vu32>;  ///< 3x4 matrix of unsigned 32-bit integers
    using mat3x4u64 = mat3x4<vu64>;  ///< 3x4 matrix of unsigned 64-bit integers

    using mat3x4f32 = mat3x4<vf32>;  ///< 3x4 matrix of 32-bit floats
    using mat3x4f64 = mat3x4<vf64>;  ///< 3x4 matrix of 64-bit floats

    //==========================================================================
    // Default convenience aliases
    //==========================================================================

    using mat3x4i = mat3x4i32;       ///< Default signed 3x4 matrix (32-bit)
    using mat3x4u = mat3x4u32;       ///< Default unsigned 3x4 matrix (32-bit)
    using mat3x4f = mat3x4f32;       ///< Default float 3x4 matrix (32-bit)
    using mat3x4d = mat3x4f64;       ///< Default double 3x4 matrix (64-bit)

} // namespace vtx

#endif // VERTEX_MATRIX3X4_HPP