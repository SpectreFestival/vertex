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
 * @file      vtxmatrix4x4.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     4x4 matrix type aliases.
 *
 * @defgroup  vtx_matrix Matrices
 * @ingroup   vtx
 * @brief     4x4 matrix aliases for all arithmetic types.
 */
#ifndef VERTEX_MATRIX4X4_HPP
#define VERTEX_MATRIX4X4_HPP
#pragma once
#include "vtxmatrix.hpp"

namespace vtx {
    using mat4x4i08 = matrix_template<vi08 , 4 , 4>;
    using mat4x4i16 = matrix_template<vi16 , 4 , 4>;
    using mat4x4i32 = matrix_template<vi32 , 4 , 4>;
    using mat4x4i64 = matrix_template<vi64 , 4 , 4>;
    using mat4x4u08 = matrix_template<vu08 , 4 , 4>;
    using mat4x4u16 = matrix_template<vu16 , 4 , 4>;
    using mat4x4u32 = matrix_template<vu32 , 4 , 4>;
    using mat4x4u64 = matrix_template<vu64 , 4 , 4>;
    using mat4x4f32 = matrix_template<vf32 , 4 , 4>;
    using mat4x4f64 = matrix_template<vf64 , 4 , 4>;
}

#endif