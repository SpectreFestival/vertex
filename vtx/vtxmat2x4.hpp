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
 * @file      vtxmat2x4.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     2x4 matrix type aliases.
 *
 * @defgroup  vtx_matrix Matrices
 * @ingroup   vtx
 * @brief     2x4 matrix aliases for all arithmetic types.
 *
 * @note      Matrix dimensions are (R rows, C columns) = (4 rows, 2 columns).
 *            The alias name "2x4" indicates the **logical** dimension (2 rows, 4 columns),
 *            but the template parameter order is matrix_template<Ty, R, C>.
 *            Therefore mat2x4 = matrix_template<Ty, 4, 2>.
 */

#ifndef VERTEX_MATRIX2X4_HPP
#define VERTEX_MATRIX2X4_HPP
#pragma once

#include "vtxbasic.hpp"    ///< arithmetic_t, vi32, vf32, etc.
#include "vtxmatcxr.hpp"   ///< matrix_template<T, R, C>

namespace vtx {

    /// @brief 2x4 matrix template alias (4 rows, 2 columns in storage).
    template <arithmetic_t Ty>
    using mat2x4 = matrix_template<Ty, 4, 2>;

    //==========================================================================
    // Type-specific 2x4 matrix aliases
    //==========================================================================

    using mat2x4i08 = mat2x4<vi08>;  ///< 2x4 matrix of signed 8-bit integers
    using mat2x4i16 = mat2x4<vi16>;  ///< 2x4 matrix of signed 16-bit integers
    using mat2x4i32 = mat2x4<vi32>;  ///< 2x4 matrix of signed 32-bit integers
    using mat2x4i64 = mat2x4<vi64>;  ///< 2x4 matrix of signed 64-bit integers

    using mat2x4u08 = mat2x4<vu08>;  ///< 2x4 matrix of unsigned 8-bit integers
    using mat2x4u16 = mat2x4<vu16>;  ///< 2x4 matrix of unsigned 16-bit integers
    using mat2x4u32 = mat2x4<vu32>;  ///< 2x4 matrix of unsigned 32-bit integers
    using mat2x4u64 = mat2x4<vu64>;  ///< 2x4 matrix of unsigned 64-bit integers

    using mat2x4f32 = mat2x4<vf32>;  ///< 2x4 matrix of 32-bit floats
    using mat2x4f64 = mat2x4<vf64>;  ///< 2x4 matrix of 64-bit floats

    //==========================================================================
    // Default convenience aliases
    //==========================================================================

    using mat2x4i = mat2x4i32;       ///< Default signed 2x4 matrix (32-bit)
    using mat2x4u = mat2x4u32;       ///< Default unsigned 2x4 matrix (32-bit)
    using mat2x4f = mat2x4f32;       ///< Default float 2x4 matrix (32-bit)
    using mat2x4d = mat2x4f64;       ///< Default double 2x4 matrix (64-bit)

} // namespace vtx

#endif // VERTEX_MATRIX2X4_HPP