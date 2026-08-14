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
 * @file      vtxbasic.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     Core type aliases and utility macros for Vertex library.
 *
 * @defgroup  vtx_basic Basic Types
 * @ingroup   vtx
 * @brief     Fundamental type definitions and compiler utilities.
 *
 * @details
 * This header provides:
 * - Integer and floating-point type aliases with explicit bit-width
 * - C++20 arithmetic concept constraint
 * - Cross-platform force-inline macro
 * - Conditional constexpr macro (enabled for C++20 and above)
 * - Debug mode detection
 */

#ifndef VERTEX_BASIC_HPP
#define VERTEX_BASIC_HPP
#pragma once

#include <cstdint>      ///< std::int8_t, std::uint32_t, etc.
#include <type_traits>  ///< std::is_arithmetic_v

/**
 * @def VERTEX_DEBUG
 * @brief Defined when NDEBUG is not set (Debug build).
 */
#ifndef NDEBUG
#   define VERTEX_DEBUG
#endif

/**
 * @def VERTEX_FORCEINLINE
 * @brief Cross-platform forced inline hint.
 * @details
 * Expands to:
 * - MSVC:   __forceinline
 * - GCC:    __inline__ __attribute__((always_inline))
 * - Clang:  __inline__ __attribute__((always_inline))
 * - Intel:  __forceinline or __inline__ __attribute__((always_inline))
 * - Others: inline
 */
#ifdef _MSC_VER
#   define VERTEX_FORCEINLINE __forceinline
#elif defined( __GNUC__ )
#   if (__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1)
#       define VERTEX_FORCEINLINE __inline__ __attribute__((always_inline))
#   else
#       define VERTEX_FORCEINLINE __inline__
#   endif
#elif defined( __clang__ )
#       define VERTEX_FORCEINLINE __inline__ __attribute__((always_inline))
#elif defined( __INTEL_COMPILER )
#   ifdef __GNUC__
#       define VERTEX_FORCEINLINE __inline__ __attribute__((always_inline))
#   else
#       define VERTEX_FORCEINLINE __forceinline
#   endif
#else
#       define VERTEX_FORCEINLINE inline
#endif

/**
 * @def VERTEX_CONSTEXPR
 * @brief Conditional constexpr.
 * @details
 * Expands to `constexpr` when C++20 or higher is detected
 * (__cplusplus >= 202302L), otherwise expands to nothing.
 */
#if __cplusplus >= 202302L
#   define VERTEX_CONSTEXPR constexpr
#else
#   define VERTEX_CONSTEXPR
#endif

/**
 * @namespace vtx
 * @brief Root namespace for Vertex library.
 */
namespace vtx {

    //==========================================================================
    // Integer Types
    //==========================================================================

    using vi08 = std::int8_t;   ///< signed 8-bit integer
    using vi16 = std::int16_t;  ///< signed 16-bit integer
    using vi32 = std::int32_t;  ///< signed 32-bit integer
    using vi64 = std::int64_t;  ///< signed 64-bit integer

    using vu08 = std::uint8_t;  ///< unsigned 8-bit integer
    using vu16 = std::uint16_t; ///< unsigned 16-bit integer
    using vu32 = std::uint32_t; ///< unsigned 32-bit integer
    using vu64 = std::uint64_t; ///< unsigned 64-bit integer

    //==========================================================================
    // Floating-Point Types
    //==========================================================================

    using vf32 = float;   ///< single-precision 32-bit floating-point
    using vf64 = double;  ///< double-precision 64-bit floating-point

    //==========================================================================
    // Concepts
    //==========================================================================

    /**
     * @concept arithmetic_t
     * @brief      Constraint for arithmetic (integral or floating-point) types.
     * @tparam T   Type to be checked.
     *
     * @details
     * Usage:
     * @code
     * template <arithmetic_t T>
     * T add(T a, T b) { return a + b; }
     * @endcode
     */
    template <typename T>
    concept arithmetic_t = std::is_arithmetic_v<T>;

} // namespace vtx

#endif // VERTEX_BASIC_HPP