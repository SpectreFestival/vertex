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
 * @file      vtxvector4.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     4-dimensional vector type aliases.
 *
 * @defgroup  vtx_vector Vectors
 * @ingroup   vtx
 * @brief     4D vector aliases for all arithmetic types.
 */
#ifndef VERTEX_VECTOR4_HPP
#define VERTEX_VECTOR4_HPP
#pragma once
#include "vtxvector.hpp"

namespace vtx {
    template <arithmetic_t Ty>
    using vec4 = vector_template<Ty , 4>;

    using vec4i08 = vec4<vi08>;
    using vec4i16 = vec4<vi16>;
    using vec4i32 = vec4<vi32>;
    using vec4i64 = vec4<vi64>;
    using vec4u08 = vec4<vu08>;
    using vec4u16 = vec4<vu16>;
    using vec4u32 = vec4<vu32>;
    using vec4u64 = vec4<vu64>;
    using vec4f32 = vec4<vf32>;
    using vec4f64 = vec4<vf64>;
}

#endif