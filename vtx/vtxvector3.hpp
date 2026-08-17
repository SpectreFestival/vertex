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
 * @file      vtxvector3.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     3-dimensional vector type aliases.
 *
 * @defgroup  vtx_vector Vectors
 * @ingroup   vtx
 * @brief     3D vector aliases for all arithmetic types.
 */
#ifndef VERTEX_VECTOR3_HPP
#define VERTEX_VECTOR3_HPP
#pragma once
#include "vtxvector.hpp"

namespace vtx {
    template <arithmetic_t Ty>
    using vec3 = vector_template<Ty , 3>;

    using vec3i08 = vec3<vi08>;
    using vec3i16 = vec3<vi16>;
    using vec3i32 = vec3<vi32>;
    using vec3i64 = vec3<vi64>;
    using vec3u08 = vec3<vu08>;
    using vec3u16 = vec3<vu16>;
    using vec3u32 = vec3<vu32>;
    using vec3u64 = vec3<vu64>;
    using vec3f32 = vec3<vf32>;
    using vec3f64 = vec3<vf64>;
}

#endif