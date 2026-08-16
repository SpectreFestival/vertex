/**
 * +-------------------------------------------------------------------------------+
 * | MIT License                                                                   |
 * +-------------------------------------------------------------------------------+
 * |                                                                               |
 * | Copyright (c) 2026 SpectreFestival                                            |
 * |                                                                               |
 * | Permission is hereby granted, free of charge, to any person obtaining a copy  |
 * | of this software and associated documentation files (the "Software"), to deal |
 * | in the Software without restriction, including without limitation the rights  |
 * | to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     |
 * | copies of the Software, and to permit persons to whom the Software is         |
 * | furnished to do so, subject to the following conditions:                      |
 * |                                                                               |
 * | The above copyright notice and this permission notice shall be included in    |
 * | all copies or substantial portions of the Software.                           |
 * +-------------------------------------------------------------------------------+
 * |                                                                               |
 * | THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    |
 * | IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      |
 * | FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   |
 * | AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        |
 * | LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, |
 * | OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE |
 * | SOFTWARE.                                                                     |
 * +-------------------------------------------------------------------------------+
 *
 * @file      vtxmat4x2.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     4x2 matrix type aliases.
 *
 * @defgroup  vtx_matrix Matrices
 * @ingroup   vtx
 * @brief     4x2 matrix aliases for all arithmetic types.
 *
 * @note      Matrix dimensions are (R rows, C columns) = (2 rows, 4 columns).
 *            The alias name "4x2" indicates the **logical** dimension (4 rows, 2 columns),
 *            but the template parameter order is matrix_template<Ty, R, C>.
 *            Therefore mat4x2 = matrix_template<Ty, 2, 4>.
 */

#ifndef VERTEX_MATRIX4X2_HPP
#define VERTEX_MATRIX4X2_HPP
#pragma once

#include "vtxbasic.hpp"    ///< arithmetic_t, vi32, vf32, etc.
#include "vtxmatcxr.hpp"   ///< matrix_template<T, R, C>

namespace vtx {

    template <arithmetic_t Ty>
    using mat4x2 = matrix_template<Ty, 2, 4>;

    //==========================================================================
    // Type-specific 4x2 matrix aliases
    //==========================================================================

    using mat4x2i08 = mat4x2<vi08>;  ///< 4x2 matrix of signed  8-bit integers
    using mat4x2i16 = mat4x2<vi16>;  ///< 4x2 matrix of signed 16-bit integers
    using mat4x2i32 = mat4x2<vi32>;  ///< 4x2 matrix of signed 32-bit integers
    using mat4x2i64 = mat4x2<vi64>;  ///< 4x2 matrix of signed 64-bit integers

    using mat4x2u08 = mat4x2<vu08>;  ///< 4x2 matrix of unsigned  8-bit integers
    using mat4x2u16 = mat4x2<vu16>;  ///< 4x2 matrix of unsigned 16-bit integers
    using mat4x2u32 = mat4x2<vu32>;  ///< 4x2 matrix of unsigned 32-bit integers
    using mat4x2u64 = mat4x2<vu64>;  ///< 4x2 matrix of unsigned 64-bit integers

    using mat4x2f32 = mat4x2<vf32>;  ///< 4x2 matrix of 32-bit floats
    using mat4x2f64 = mat4x2<vf64>;  ///< 4x2 matrix of 64-bit floats

    //==========================================================================
    // Default convenience aliases
    //==========================================================================

    using mat4x2i = mat4x2i32;       ///< Default signed   4x2 matrix (32-bit)
    using mat4x2u = mat4x2u32;       ///< Default unsigned 4x2 matrix (32-bit)
    using mat4x2f = mat4x2f32;       ///< Default float    4x2 matrix (32-bit)
    using mat4x2d = mat4x2f64;       ///< Default double   4x2 matrix (64-bit)
} // namespace vtx

#endif // VERTEX_MATRIX4X2_HPP