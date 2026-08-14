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
* @file      vtxvec4.hpp
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

#include "vtxbasic.hpp"   ///< arithmetic_t, vi32, vf32, etc.
#include "vtxvecx.hpp"    ///< vector_template<T, N>

namespace vtx {

    /// @brief 4D vector template alias.
    template <arithmetic_t Ty>
    using vec4 = vector_template<Ty, 4>;

    //==========================================================================
    // Type-specific 4D vector aliases
    //==========================================================================

    using vec4i08 = vec4<vi08>;  ///< 4D vector of signed 8-bit integers
    using vec4i16 = vec4<vi16>;  ///< 4D vector of signed 16-bit integers
    using vec4i32 = vec4<vi32>;  ///< 4D vector of signed 32-bit integers
    using vec4i64 = vec4<vi64>;  ///< 4D vector of signed 64-bit integers

    using vec4u08 = vec4<vu08>;  ///< 4D vector of unsigned 8-bit integers
    using vec4u16 = vec4<vu16>;  ///< 4D vector of unsigned 16-bit integers
    using vec4u32 = vec4<vu32>;  ///< 4D vector of unsigned 32-bit integers
    using vec4u64 = vec4<vu64>;  ///< 4D vector of unsigned 64-bit integers

    using vec4f32 = vec4<vf32>;  ///< 4D vector of 32-bit floats
    using vec4f64 = vec4<vf64>;  ///< 4D vector of 64-bit floats

    //==========================================================================
    // Default convenience aliases
    //==========================================================================

    using vec4i = vec4i32;       ///< Default signed 4D vector (32-bit)
    using vec4u = vec4u32;       ///< Default unsigned 4D vector (32-bit)
    using vec4f = vec4f32;       ///< Default float 4D vector (32-bit)
    using vec4d = vec4f64;       ///< Default double 4D vector (64-bit)

} // namespace vtx

#endif // VERTEX_VECTOR4_HPP