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
 * @file      vtxmat3x2.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     3x2 matrix type aliases.
 *
 * @defgroup  vtx_matrix Matrices
 * @ingroup   vtx
 * @brief     3x2 matrix aliases for all arithmetic types.
 *
 * @note      Matrix dimensions are (R rows, C columns) = (2 rows, 3 columns).
 *            The alias name "3x2" indicates the **logical** dimension (3 rows, 2 columns),
 *            but the template parameter order is matrix_template<Ty, R, C>.
 *            Therefore mat3x2 = matrix_template<Ty, 2, 3>.
 */

#ifndef VERTEX_MATRIX3X2_HPP
#define VERTEX_MATRIX3X2_HPP
#pragma once

#include "vtxbasic.hpp"    ///< arithmetic_t, vi32, vf32, etc.
#include "vtxmatcxr.hpp"   ///< matrix_template<T, R, C>

namespace vtx {

    /// @brief 3x2 matrix template alias (2 rows, 3 columns in storage).
    template <arithmetic_t Ty>
    using mat3x2 = matrix_template<Ty, 2, 3>;

    //==========================================================================
    // Type-specific 3x2 matrix aliases
    //==========================================================================

    using mat3x2i08 = mat3x2<vi08>;  ///< 3x2 matrix of signed 8-bit integers
    using mat3x2i16 = mat3x2<vi16>;  ///< 3x2 matrix of signed 16-bit integers
    using mat3x2i32 = mat3x2<vi32>;  ///< 3x2 matrix of signed 32-bit integers
    using mat3x2i64 = mat3x2<vi64>;  ///< 3x2 matrix of signed 64-bit integers

    using mat3x2u08 = mat3x2<vu08>;  ///< 3x2 matrix of unsigned 8-bit integers
    using mat3x2u16 = mat3x2<vu16>;  ///< 3x2 matrix of unsigned 16-bit integers
    using mat3x2u32 = mat3x2<vu32>;  ///< 3x2 matrix of unsigned 32-bit integers
    using mat3x2u64 = mat3x2<vu64>;  ///< 3x2 matrix of unsigned 64-bit integers

    using mat3x2f32 = mat3x2<vf32>;  ///< 3x2 matrix of 32-bit floats
    using mat3x2f64 = mat3x2<vf64>;  ///< 3x2 matrix of 64-bit floats

    //==========================================================================
    // Default convenience aliases
    //==========================================================================

    using mat3x2i = mat3x2i32;       ///< Default signed 3x2 matrix (32-bit)
    using mat3x2u = mat3x2u32;       ///< Default unsigned 3x2 matrix (32-bit)
    using mat3x2f = mat3x2f32;       ///< Default float 3x2 matrix (32-bit)
    using mat3x2d = mat3x2f64;       ///< Default double 3x2 matrix (64-bit)

} // namespace vtx

#endif // VERTEX_MATRIX3X2_HPP