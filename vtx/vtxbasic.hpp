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
 * @def     VERTEX_DEBUG
 * @brief   Defined when NDEBUG is not set (Debug build).
 */
#ifndef NDEBUG
#define VERTEX_DEBUG
#endif

/**
 * @def     VERTEX_FORCEINLINE
 * @brief   Cross-platform forced inline hint.
 * @details
 * Expands to:
 * - MSVC:   __forceinline
 * - Intel:  __forceinline
 * - GCC:    __inline__ __attribute__((always_inline))
 * - Clang:  __inline__ __attribute__((always_inline))
 * - Intel:  __inline__ __attribute__((always_inline))
 * - Others: inline
 */
#ifdef _MSC_VER
#   define VERTEX_FORCEINLINE __forceinline
#elif defined(__GNUC__)
#   if (__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1)
#       define VERTEX_FORCEINLINE __inline__ __attribute__((always_inline))
#   else
#       define VERTEX_FORCEINLINE __inline__
#   endif
#elif defined(__clang__)
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
 * @def     VERTEX_CONSTEXPR
 * @brief   Conditional constexpr.
 * @details
 * Expands to `constexpr` when C++20 or higher is detected
 * (__cplusplus >= 202302L), otherwise expands to nothing.
 */
#if __cplusplus >= 202302L
#   define VERTEX_CONSTEXPR constexpr
#else
#   define VERTEX_CONSTEXPR
#endif

#if defined(__clang__) || defined(__GNUC__) || defined(__INTEL_COMPILER)
#	define VERTEX_RESTRICT __restrict__
#elif defined(_MSC_VER)
#	define VERTEX_RESTRICT __restrict
#else
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#	define VERTEX_RESTRICT restrict
#else
#	define VERTEX_RESTRICT 
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define VERTEX_SUPPORTS_TARGET_ATTRIBUTE 1

    #define VERTEX_FUNCTION_OPTIMIZATION(...)  __attribute__((target(__VA_ARGS__)))
    #define VERTEX_OPT_SSE4_NAME    "sse4.2"
    #define VERTEX_OPT_AVX1_NAME    "avx"
    #define VERTEX_OPT_AVX2_NAME    "avx2"
    #define VERTEX_OPT_AVX5_NAME    "avx512f"
    #define VERTEX_OPT_NEON_NAME    "neon"
    #define VERTEX_OPT_MFMA_NAME    "avx2,fma"

    #define VERTEX_OPT_SSE4 VERTEX_FUNCTION_OPTIMIZATION("sse4.2")
    #define VERTEX_OPT_AVX1 VERTEX_FUNCTION_OPTIMIZATION("avx")
    #define VERTEX_OPT_AVX2 VERTEX_FUNCTION_OPTIMIZATION("avx2")
    #define VERTEX_OPT_AVX5 VERTEX_FUNCTION_OPTIMIZATION("avx512f")
    #define VERTEX_OPT_NEON VERTEX_FUNCTION_OPTIMIZATION("neon")

#elif defined(_MSC_VER)
    #define VERTEX_SUPPORTS_TARGET_ATTRIBUTE 0
    #define VERTEX_FUNCTION_OPTIMIZATION(...)
    #define VERTEX_OPT_SSE4_NAME
    #define VERTEX_OPT_AVX1_NAME
    #define VERTEX_OPT_AVX2_NAME
    #define VERTEX_OPT_AVX5_NAME
    #define VERTEX_OPT_NEON_NAME
    #define VERTEX_OPT_MFMA_NAME
    #define VERTEX_OPT_SSE4
    #define VERTEX_OPT_AVX1
    #define VERTEX_OPT_AVX2
    #define VERTEX_OPT_AVX5
    #define VERTEX_OPT_NEON
#else
    #define VERTEX_SUPPORTS_TARGET_ATTRIBUTE 0
    #define VERTEX_FUNCTION_OPTIMIZATION(...)
    #define VERTEX_OPT_SSE4_NAME
    #define VERTEX_OPT_AVX1_NAME
    #define VERTEX_OPT_AVX2_NAME
    #define VERTEX_OPT_AVX5_NAME
    #define VERTEX_OPT_NEON_NAME
    #define VERTEX_OPT_MFMA_NAME
    #define VERTEX_OPT_SSE4
    #define VERTEX_OPT_AVX1
    #define VERTEX_OPT_AVX2
    #define VERTEX_OPT_AVX5
    #define VERTEX_OPT_NEON
#endif

/**
 * @namespace   vtx
 * @brief       Root namespace for Vertex library.
 */
namespace vtx {

    //+------------------------------------------------------------------------+
    //| Integer Types                                                          |
    //+------------------------------------------------------------------------+

    using vi08 = std::int8_t;   ///< signed  8-bit integer
    using vi16 = std::int16_t;  ///< signed 16-bit integer
    using vi32 = std::int32_t;  ///< signed 32-bit integer
    using vi64 = std::int64_t;  ///< signed 64-bit integer

    using vu08 = std::uint8_t;  ///< unsigned  8-bit integer
    using vu16 = std::uint16_t; ///< unsigned 16-bit integer
    using vu32 = std::uint32_t; ///< unsigned 32-bit integer
    using vu64 = std::uint64_t; ///< unsigned 64-bit integer

    //+------------------------------------------------------------------------+
    //| Floating-Point Types                                                   |
    //+------------------------------------------------------------------------+

    using vf32 = float;   ///< single-precision 32-bit floating-point
    using vf64 = double;  ///< double-precision 64-bit floating-point

    //+------------------------------------------------------------------------+
    //| Concepts                                                               |
    //+------------------------------------------------------------------------+

    /**
     * @concept     arithmetic_t
     * @brief       Constraint for arithmetic (integral or floating-point) types.
     * @tparam      T Type to be checked.
     */
    template <typename T>
    concept arithmetic_t = std::is_arithmetic_v<T>;

    template <typename T>
    inline constexpr auto vertex_alignment = alignof(T);

    template <> inline constexpr std::size_t vertex_alignment<vi32> = 32;
    template <> inline constexpr std::size_t vertex_alignment<vu32> = 32;
    template <> inline constexpr std::size_t vertex_alignment<vi64> = 64;
    template <> inline constexpr std::size_t vertex_alignment<vu64> = 64;
    template <> inline constexpr std::size_t vertex_alignment<vf32> = 64;
    template <> inline constexpr std::size_t vertex_alignment<vf64> = 64;
} // namespace vtx

#endif // VERTEX_BASIC_HPP