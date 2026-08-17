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
 * @file      vtxmatrix4x3.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     4x3 matrix type aliases.
 *
 * @defgroup  vtx_matrix Matrices
 * @ingroup   vtx
 * @brief     4x3 matrix aliases for all arithmetic types.
 */
#ifndef VERTEX_MATRIX4X3_HPP
#define VERTEX_MATRIX4X3_HPP
#pragma once
#include "vtxmatrix.hpp"

namespace vtx {
    using mat4x3i08 = matrix_template<vi08 , 3 , 4>;
    using mat4x3i16 = matrix_template<vi16 , 3 , 4>;
    using mat4x3i32 = matrix_template<vi32 , 3 , 4>;
    using mat4x3i64 = matrix_template<vi64 , 3 , 4>;
    using mat4x3u08 = matrix_template<vu08 , 3 , 4>;
    using mat4x3u16 = matrix_template<vu16 , 3 , 4>;
    using mat4x3u32 = matrix_template<vu32 , 3 , 4>;
    using mat4x3u64 = matrix_template<vu64 , 3 , 4>;
    using mat4x3f32 = matrix_template<vf32 , 3 , 4>;
    using mat4x3f64 = matrix_template<vf64 , 3 , 4>;
}

#endif