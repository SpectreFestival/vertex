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
 * @file      vtxvec2.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     2-dimensional vector type aliases.
 *
 * @defgroup  vtx_vector Vectors
 * @ingroup   vtx
 * @brief     2D vector aliases for all arithmetic types.
 */

#ifndef VERTEX_VECTOR2_HPP
#define VERTEX_VECTOR2_HPP
#pragma once

#include "vtxbasic.hpp"   ///< arithmetic_t, vi32, vf32, etc.
#include "vtxvecx.hpp"    ///< vector_template<T, N>

namespace vtx {
    
    template <arithmetic_t Ty>
    using vec2 = vector_template<Ty, 2>;

    //==========================================================================
    // Type-specific 2D vector aliases
    //==========================================================================

    using vec2i08 = vec2<vi08>;  ///< 2D vector of signed  8-bit integers
    using vec2i16 = vec2<vi16>;  ///< 2D vector of signed 16-bit integers
    using vec2i32 = vec2<vi32>;  ///< 2D vector of signed 32-bit integers
    using vec2i64 = vec2<vi64>;  ///< 2D vector of signed 64-bit integers

    using vec2u08 = vec2<vu08>;  ///< 2D vector of unsigned  8-bit integers
    using vec2u16 = vec2<vu16>;  ///< 2D vector of unsigned 16-bit integers
    using vec2u32 = vec2<vu32>;  ///< 2D vector of unsigned 32-bit integers
    using vec2u64 = vec2<vu64>;  ///< 2D vector of unsigned 64-bit integers

    using vec2f32 = vec2<vf32>;  ///< 2D vector of 32-bit floats
    using vec2f64 = vec2<vf64>;  ///< 2D vector of 64-bit floats

    //==========================================================================
    // Default convenience aliases
    //==========================================================================

    using vec2i  = vec2i32;      ///< Default signed   2D vector (32-bit)
    using vec2u  = vec2u32;      ///< Default unsigned 2D vector (32-bit)
    using vec2f  = vec2f32;      ///< Default float    2D vector (32-bit)
    using vec2d  = vec2f64;      ///< Default double   2D vector (64-bit)
} // namespace vtx

#endif // VERTEX_VECTOR2_HPP
