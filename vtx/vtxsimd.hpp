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
* @file      vtxsimd.hpp
* @author    SpectreFestival
* @license   MIT
* @brief     SIMD AVX2 optimizations for Vertex library operations.
*
* @defgroup  vtx_ext Extensions
* @ingroup   vtx
* @brief     SIMD-accelerated linear algebra operations.
*
* @details
* This header provides AVX2-accelerated implementations for:
* - Matrix addition (vf32/vf64, aligned to SIMD lanes)
* - Matrix subtraction (vf32/vf64, aligned to SIMD lanes)
* - 4x4 matrix inversion (vf32/vf64, SIMD-optimized)
*
* All SIMD operations are conditionally compiled when VERTEX_ENABLE_EXTENSIONS
* is defined. The implementation uses AVX2 intrinsics (<immintrin.h>) and
* requires appropriate compiler flags (-mavx2, /arch:AVX2).
*
* @note  This module is optional and must be explicitly enabled.
* @note  Matrix dimensions must be multiples of SIMD lane width:
*        - vf32: total elements must be multiple of 8
*        - vf64: total elements must be multiple of 4
* @note  Fallback scalar implementations are used when SIMD is not enabled.
*
* @warning SIMD-accelerated inverse functions are experimental
*          (VERTEX_ENABLE_TESTEXT) and may not handle all edge cases.
*/

#ifndef VERTEX_SIMD_HPP
#define VERTEX_SIMD_HPP
#pragma once

#include "vtxvecx.hpp"       ///< vector_template<T, N>
#include "vtxvec2.hpp"       ///< vec2<T>
#include "vtxvec3.hpp"       ///< vec3<T>
#include "vtxvec4.hpp"       ///< vec4<T>

#include "vtxmatcxr.hpp"     ///< matrix_template<T, R, C>
#include "vtxmat2x2.hpp"     ///< mat2x2<T>
#include "vtxmat2x3.hpp"     ///< mat2x3<T>
#include "vtxmat2x4.hpp"     ///< mat2x4<T>
#include "vtxmat3x2.hpp"     ///< mat3x2<T>
#include "vtxmat3x3.hpp"     ///< mat3x3<T>
#include "vtxmat3x4.hpp"     ///< mat3x4<T>
#include "vtxmat4x2.hpp"     ///< mat4x2<T>
#include "vtxmat4x3.hpp"     ///< mat4x3<T>
#include "vtxmat4x4.hpp"     ///< mat4x4<T>

#include <immintrin.h>       ///< AVX2 intrinsics

namespace vtx {
    /**
    * @brief    Internal SIMD implementation details.
    * @namespace impl
    */
    namespace impl {
        /**
         * @brief AVX2-accelerated vector addition for double-precision.
         * @tparam I... Compile-time indices for loop unrolling.
         * @param veca Source vector A (vf64*).
         * @param vecb Source vector B (vf64*).
         * @param vecc Destination vector C = A + B (vf64*).
         * @param seq Index sequence for compile-time unrolling.
         *
         * @details
         * Processes 4 double-precision elements (256 bits) per iteration.
         * Uses AVX2 SIMD intrinsics: _mm256_loadu_pd / _mm256_add_pd / _mm256_storeu_pd.
         *
         * @note Total elements must be multiple of 4.
         * @note Uses unaligned loads/stores for flexibility.
         */
        template <size_t... I>
        void compute_vecadd( const vf64* veca , const vf64* vecb , vf64* vecc , std::index_sequence<I...> ) noexcept {
            ( ( _mm256_storeu_pd( vecc + I * 4 , _mm256_add_pd( _mm256_loadu_pd( veca + I * 4 ) , _mm256_loadu_pd( vecb + I * 4 ) ) ) ) , ... );
        }

        /**
         * @brief AVX2-accelerated vector subtraction for double-precision.
         * @tparam I... Compile-time indices for loop unrolling.
         * @param veca Source vector A (vf64*).
         * @param vecb Source vector B (vf64*).
         * @param vecc Destination vector C = A - B (vf64*).
         * @param seq Index sequence for compile-time unrolling.
         *
         * @details
         * Processes 4 double-precision elements (256 bits) per iteration.
         * Uses AVX2 SIMD intrinsics: _mm256_loadu_pd / _mm256_sub_pd / _mm256_storeu_pd.
         *
         * @note Total elements must be multiple of 4.
         * @note Uses unaligned loads/stores for flexibility.
         */
        template <size_t... I>
        void compute_vecsub( const vf64* veca , const vf64* vecb , vf64* vecc , std::index_sequence<I...> ) noexcept {
            ( ( _mm256_storeu_pd( vecc + I * 4 , _mm256_sub_pd( _mm256_loadu_pd( veca + I * 4 ) , _mm256_loadu_pd( vecb + I * 4 ) ) ) ) , ... );
        }

        /**
         * @brief AVX2-accelerated vector addition for single-precision.
         * @tparam I... Compile-time indices for loop unrolling.
         * @param veca Source vector A (vf32*).
         * @param vecb Source vector B (vf32*).
         * @param vecc Destination vector C = A + B (vf32*).
         * @param seq Index sequence for compile-time unrolling.
         *
         * @details
         * Processes 8 single-precision elements (256 bits) per iteration.
         * Uses AVX2 SIMD intrinsics: _mm256_loadu_ps / _mm256_add_ps / _mm256_storeu_ps.
         *
         * @note Total elements must be multiple of 8.
         * @note Uses unaligned loads/stores for flexibility.
         */
        template <size_t... I>
        void compute_vecadd( const vf32* veca , const vf32* vecb , vf32* vecc , std::index_sequence<I...> ) noexcept {
            ( ( _mm256_storeu_ps( vecc + I * 8 , _mm256_add_ps( _mm256_loadu_ps( veca + I * 8 ) , _mm256_loadu_ps( vecb + I * 8 ) ) ) ) , ... );
        }

        /**
         * @brief AVX2-accelerated vector subtraction for single-precision.
         * @tparam I... Compile-time indices for loop unrolling.
         * @param veca Source vector A (vf32*).
         * @param vecb Source vector B (vf32*).
         * @param vecc Destination vector C = A - B (vf32*).
         * @param seq Index sequence for compile-time unrolling.
         *
         * @details
         * Processes 8 single-precision elements (256 bits) per iteration.
         * Uses AVX2 SIMD intrinsics: _mm256_loadu_ps / _mm256_sub_ps / _mm256_storeu_ps.
         *
         * @note Total elements must be multiple of 8.
         * @note Uses unaligned loads/stores for flexibility.
         */
        template <size_t... I>
        void compute_vecsub( const vf32* veca , const vf32* vecb , vf32* vecc , std::index_sequence<I...> ) noexcept {
            ( ( _mm256_storeu_ps( vecc + I * 8 , _mm256_sub_ps( _mm256_loadu_ps( veca + I * 8 ) , _mm256_loadu_ps( vecb + I * 8 ) ) ) ) , ... );
        }
    }

    /**
     * @brief AVX2-accelerated compound addition for double-precision matrices.
     * @tparam R Number of rows.
     * @tparam C Number of columns.
     * @param mata Left-hand side matrix (modified in-place).
     * @param matb Right-hand side matrix.
     * @return Reference to mata.
     *
     * @details
     * Requires total elements (R * C) to be:
     * - Multiple of 4 (AVX2 processes 4 doubles at a time)
     * - Greater than 4 (fallback for small matrices)
     *
     * Uses AVX2 intrinsics to perform element-wise addition.
     *
     * @note Enabled only when VERTEX_ENABLE_EXTENSIONS is defined.
     */
    template <size_t R , size_t C> requires( ( R* C ) % 4 == 0 and ( R * C ) > 4 )
        matrix_template<vf64 , R , C>& operator+=( matrix_template<vf64 , R , C>& mata , const matrix_template<vf64 , R , C>& matb ) noexcept {
        impl::compute_vecadd( &mata.data [ 0 ][ 0 ] , &matb.data [ 0 ][ 0 ] , &mata.data [ 0 ][ 0 ] , std::make_index_sequence< ( R * C ) / 4 >( ) );
        return mata;
    }

    /**
     * @brief AVX2-accelerated compound subtraction for double-precision matrices.
     * @tparam R Number of rows.
     * @tparam C Number of columns.
     * @param mata Left-hand side matrix (modified in-place).
     * @param matb Right-hand side matrix.
     * @return Reference to mata.
     *
     * @details
     * Requires total elements (R * C) to be:
     * - Multiple of 4 (AVX2 processes 4 doubles at a time)
     * - Greater than 4 (fallback for small matrices)
     *
     * Uses AVX2 intrinsics to perform element-wise subtraction.
     *
     * @note Enabled only when VERTEX_ENABLE_EXTENSIONS is defined.
     */
    template <size_t R , size_t C> requires( ( R* C ) % 4 == 0 and ( R * C ) > 4 )
        matrix_template<vf64 , R , C>& operator-=( matrix_template<vf64 , R , C>& mata , const matrix_template<vf64 , R , C>& matb ) noexcept {
        impl::compute_vecsub( &mata.data [ 0 ][ 0 ] , &matb.data [ 0 ][ 0 ] , &mata.data [ 0 ][ 0 ] , std::make_index_sequence< ( R * C ) / 4 >( ) );
        return mata;
    }

    /**
     * @brief AVX2-accelerated compound addition for single-precision matrices.
     * @tparam R Number of rows.
     * @tparam C Number of columns.
     * @param mata Left-hand side matrix (modified in-place).
     * @param matb Right-hand side matrix.
     * @return Reference to mata.
     *
     * @details
     * Requires total elements (R * C) to be:
     * - Multiple of 8 (AVX2 processes 8 floats at a time)
     * - Greater than 8 (fallback for small matrices)
     *
     * Uses AVX2 intrinsics to perform element-wise addition.
     *
     * @note Enabled only when VERTEX_ENABLE_EXTENSIONS is defined.
     */
    template <size_t R , size_t C> requires( ( R* C ) % 8 == 0 and ( R * C ) > 8 )
        matrix_template<vf32 , R , C>& operator+=( matrix_template<vf32 , R , C>& mata , const matrix_template<vf32 , R , C>& matb ) noexcept {
        impl::compute_vecadd( &mata.data [ 0 ][ 0 ] , &matb.data [ 0 ][ 0 ] , &mata.data [ 0 ][ 0 ] , std::make_index_sequence< ( R * C ) / 8 >( ) );
        return mata;
    }

    /**
     * @brief AVX2-accelerated compound subtraction for single-precision matrices.
     * @tparam R Number of rows.
     * @tparam C Number of columns.
     * @param mata Left-hand side matrix (modified in-place).
     * @param matb Right-hand side matrix.
     * @return Reference to mata.
     *
     * @details
     * Requires total elements (R * C) to be:
     * - Multiple of 8 (AVX2 processes 8 floats at a time)
     * - Greater than 8 (fallback for small matrices)
     * 
     * Uses AVX2 intrinsics to perform element-wise subtraction.
     *
     * @note Enabled only when VERTEX_ENABLE_EXTENSIONS is defined.
     */
    template <size_t R , size_t C> requires( ( R* C ) % 8 == 0 and ( R * C ) > 8 )
        matrix_template<vf32 , R , C>& operator-=( matrix_template<vf32 , R , C>& mata , const matrix_template<vf32 , R , C>& matb ) noexcept {
        impl::compute_vecsub( &mata.data [ 0 ][ 0 ] , &matb.data [ 0 ][ 0 ] , &mata.data [ 0 ][ 0 ] , std::make_index_sequence< ( R * C ) / 8 >( ) );
        return mata;
    }

#ifdef VERTEX_ENABLE_TESTEXT
    namespace test_impl {
        inline vf32 impl_sum1m256( const __m256& v0 ) noexcept {
            auto vl = _mm256_castps256_ps128( v0 );
            auto vh = _mm256_extractf128_ps( v0 , 1 );
            auto vs = _mm_add_ps( vl , vh );
            vs = _mm_hadd_ps( vs , vs );
            vs = _mm_hadd_ps( vs , vs );
            return _mm_cvtss_f32( vs );
        }

        inline  __m256 impl_mul3m256( const __m256& v1 , const __m256& v2 , const __m256& v3 ) noexcept {
            return _mm256_mul_ps( _mm256_mul_ps( v1 , v2 ) , v3 );
        }

        inline  __m256 compute_temp(
            const __m256& v1 , const __m256& v2 , const __m256& v3 ,
            const __m256& v4 , const __m256& v5 , const __m256& v6 ,
            const __m256& v7 , const __m256& v8 , const __m256& v9
        ) noexcept {
            return
                _mm256_add_ps(
                _mm256_add_ps(
                _mm256_mul_ps( _mm256_mul_ps( v1 , v2 ) , v3 ) ,
                _mm256_mul_ps( _mm256_mul_ps( v4 , v5 ) , v6 ) ) ,
                _mm256_mul_ps( _mm256_mul_ps( v7 , v8 ) , v9 ) );

            //return
            //    _mm256_fmadd_ps( _mm256_mul_ps( v7 , v8 ) , v9 ,
            //    _mm256_fmadd_ps( _mm256_mul_ps( v4 , v5 ) , v6 ,
            //      _mm256_mul_ps( _mm256_mul_ps( v1 , v2 ) , v3 ) ) );
        }

        /**
         * @brief AVX2-optimized 4x4 matrix inversion for single-precision.
         * @param imsrc Input 4x4 matrix in column-major order (vf32*).
         * @param omdst Output inverse matrix in column-major order (vf32*).
         *
         * @details
         * Uses AVX2 SIMD intrinsics to compute the inverse of a 4x4 matrix.
         * The algorithm computes all 16 cofactors using 8-lane AVX2 operations,
         * processes them in parallel, then computes the determinant and divides.
         *
         * Performance characteristics:
         * - 8-wide SIMD lanes (AVX2)
         * - All cofactors computed simultaneously
         * - Single division for the entire matrix (shared determinant)
         *
         * @note This implementation is optimized for column-major storage layout.
         * @note The input matrix is assumed to be invertible (det != 0).
         * @note This is an experimental implementation enabled by VERTEX_ENABLE_TESTEXT.
         *
         * @warning The algorithm performs no determinant check.
         *          Division by zero may occur for singular matrices.
         */
        inline void inverse_mat4f32( const vf32* imsrc , vf32* omdst ) noexcept {
            auto& x00 = imsrc [ 0x0 ]; auto& x01 = imsrc [ 0x4 ]; auto& x02 = imsrc [ 0x8 ]; auto& x03 = imsrc [ 0xC ];
            auto& x10 = imsrc [ 0x1 ]; auto& x11 = imsrc [ 0x5 ]; auto& x12 = imsrc [ 0x9 ]; auto& x13 = imsrc [ 0xD ];
            auto& x20 = imsrc [ 0x2 ]; auto& x21 = imsrc [ 0x6 ]; auto& x22 = imsrc [ 0xA ]; auto& x23 = imsrc [ 0xE ];
            auto& x30 = imsrc [ 0x3 ]; auto& x31 = imsrc [ 0x7 ]; auto& x32 = imsrc [ 0xB ]; auto& x33 = imsrc [ 0xF ];

            //      [   x00,    x01,    x02,    x03     ]           [   a00,    a10,    a20,    a30     ]
            //      [   x10,    x11,    x12,    x13     ]   __\     [   a01,    a11,    a21,    a31     ]
            //      [   x20,    x21,    x22,    x23     ]     /     [   a02,    a12,    a22,    a32     ]
            //      [   x30,    x31,    x32,    x33     ]           [   a03,    a13,    a23,    a33     ]
            //
            //  A00 = x11 * x22 * x33 + x12 * x23 * x31 + x13 * x21 * x32 - x13 * x22 * x31 - x12 * x21 * x33 - x11 * x23 * x32
            //  A02 = x10 * x21 * x33 + x11 * x23 * x30 + x13 * x20 * x31 - x13 * x21 * x30 - x11 * x20 * x33 - x10 * x23 * x31
            //  A01 = x13 * x22 * x30 + x12 * x20 * x33 + x10 * x23 * x32 - x10 * x22 * x33 - x12 * x23 * x30 - x13 * x20 * x32
            //  A03 = x12 * x21 * x30 + x11 * x20 * x32 + x10 * x22 * x31 - x10 * x21 * x32 - x11 * x22 * x30 - x12 * x20 * x31
            //  A11 = x00 * x22 * x33 + x02 * x23 * x30 + x03 * x20 * x32 - x03 * x22 * x30 - x02 * x20 * x33 - x00 * x23 * x32
            //  A13 = x00 * x21 * x32 + x01 * x22 * x30 + x02 * x20 * x31 - x02 * x21 * x30 - x01 * x20 * x32 - x00 * x22 * x31
            //  A10 = x03 * x22 * x31 + x02 * x21 * x33 + x01 * x23 * x32 - x01 * x22 * x33 - x02 * x23 * x31 - x03 * x21 * x32
            //  A12 = x03 * x21 * x30 + x01 * x20 * x33 + x00 * x23 * x31 - x00 * x21 * x33 - x01 * x23 * x30 - x03 * x20 * x31
            //         ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^ 
            //  A20 = x01 * x12 * x33 + x02 * x13 * x31 + x03 * x11 * x32 - x03 * x12 * x31 - x02 * x11 * x33 - x01 * x13 * x32
            //  A22 = x00 * x11 * x33 + x01 * x13 * x30 + x03 * x10 * x31 - x03 * x11 * x30 - x01 * x10 * x33 - x00 * x13 * x31
            //  A21 = x03 * x12 * x30 + x02 * x10 * x33 + x00 * x13 * x32 - x00 * x12 * x33 - x02 * x13 * x30 - x03 * x10 * x32
            //  A23 = x02 * x11 * x30 + x01 * x10 * x32 + x00 * x12 * x31 - x00 * x11 * x32 - x01 * x12 * x30 - x02 * x10 * x31
            //  A31 = x00 * x12 * x23 + x02 * x13 * x20 + x03 * x10 * x22 - x03 * x12 * x20 - x02 * x10 * x23 - x00 * x13 * x22
            //  A33 = x00 * x11 * x22 + x01 * x12 * x20 + x02 * x10 * x21 - x02 * x11 * x20 - x01 * x10 * x22 - x00 * x12 * x21
            //  A30 = x03 * x12 * x21 + x02 * x11 * x23 + x01 * x13 * x22 - x01 * x12 * x23 - x02 * x13 * x21 - x03 * x11 * x22
            //  A32 = x03 * x11 * x20 + x01 * x10 * x23 + x00 * x13 * x21 - x00 * x11 * x23 - x01 * x13 * x20 - x03 * x10 * x21
            //         ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^ 

            auto x11_10_13_12_00_00_03_03 = _mm256_set_ps( x03 , x03 , x00 , x00 , x12 , x13 , x10 , x11 );
            auto x22_21_22_21_22_21_22_21 = _mm256_set_ps( x21 , x22 , x21 , x22 , x21 , x22 , x21 , x22 );
            auto x33_33_30_30_33_32_31_30 = _mm256_set_ps( x30 , x31 , x32 , x33 , x30 , x30 , x33 , x33 );
            auto x12_11_12_11_02_01_02_01 = _mm256_set_ps( x01 , x02 , x01 , x02 , x11 , x12 , x11 , x12 );
            auto x23_23_20_20_23_22_21_20 = _mm256_set_ps( x20 , x21 , x22 , x23 , x20 , x20 , x23 , x23 );
            auto x31_30_33_32_30_30_33_33 = _mm256_set_ps( x33 , x33 , x30 , x30 , x32 , x33 , x30 , x31 );
            auto x13_13_10_10_03_02_01_00 = _mm256_set_ps( x00 , x01 , x02 , x03 , x10 , x10 , x13 , x13 );
            auto x21_20_23_22_20_20_23_23 = _mm256_set_ps( x23 , x23 , x20 , x20 , x22 , x23 , x20 , x21 );
            auto x32_31_32_31_32_31_32_31 = _mm256_set_ps( x31 , x32 , x31 , x32 , x31 , x32 , x31 , x32 );
            auto x01_00_03_02_00_00_03_03 = _mm256_set_ps( x03 , x03 , x00 , x00 , x02 , x03 , x00 , x01 );
            auto x12_11_12_11_12_11_12_11 = _mm256_set_ps( x11 , x12 , x11 , x12 , x11 , x12 , x11 , x12 );
            auto x33_33_30_30_23_22_21_20 = _mm256_set_ps( x20 , x21 , x22 , x23 , x30 , x30 , x33 , x33 );
            auto x02_01_02_01_02_01_02_01 = _mm256_set_ps( x01 , x02 , x01 , x02 , x01 , x02 , x01 , x02 );
            auto x13_13_10_10_13_12_11_10 = _mm256_set_ps( x10 , x11 , x12 , x13 , x10 , x10 , x13 , x13 );
            auto x31_30_33_32_20_20_23_23 = _mm256_set_ps( x23 , x23 , x20 , x20 , x32 , x33 , x30 , x31 );
            auto x03_03_00_00_03_02_01_00 = _mm256_set_ps( x00 , x01 , x02 , x03 , x00 , x00 , x03 , x03 );
            auto x11_10_13_12_10_10_13_13 = _mm256_set_ps( x13 , x13 , x10 , x10 , x12 , x13 , x10 , x11 );
            auto x32_31_32_31_22_21_22_21 = _mm256_set_ps( x21 , x22 , x21 , x22 , x31 , x32 , x31 , x32 );

            auto a00_02_01_03_11_13_10_12 = _mm256_sub_ps(
                compute_temp( x11_10_13_12_00_00_03_03 , x22_21_22_21_22_21_22_21 , x33_33_30_30_33_32_31_30 ,
                              x12_11_12_11_02_01_02_01 , x23_23_20_20_23_22_21_20 , x31_30_33_32_30_30_33_33 ,
                              x13_13_10_10_03_02_01_00 , x21_20_23_22_20_20_23_23 , x32_31_32_31_32_31_32_31 ) ,
                compute_temp( x13_13_10_10_03_02_01_00 , x22_21_22_21_22_21_22_21 , x31_30_33_32_20_20_23_23 ,
                              x12_11_12_11_02_01_02_01 , x21_20_23_22_20_20_23_23 , x33_33_30_30_23_22_21_20 ,
                              x11_10_13_12_00_00_03_03 , x23_23_20_20_23_22_21_20 , x22_21_22_21_22_21_22_21 ) );
            auto a20_22_21_23_31_33_30_32 = _mm256_sub_ps(
                compute_temp( x01_00_03_02_00_00_03_03 , x12_11_12_11_02_01_02_01 , x33_33_30_30_23_22_21_20 ,
                              x02_01_02_01_02_01_02_01 , x13_13_10_10_03_02_01_00 , x31_30_33_32_20_20_23_23 ,
                              x03_03_00_00_03_02_01_00 , x11_10_13_12_00_00_03_03 , x32_31_32_31_22_21_22_21 ) ,
                compute_temp( x03_03_00_00_03_02_01_00 , x12_11_12_11_02_01_02_01 , x31_30_33_32_20_20_23_23 ,
                              x02_01_02_01_02_01_02_01 , x11_10_13_12_00_00_03_03 , x33_33_30_30_23_22_21_20 ,
                              x01_00_03_02_00_00_03_03 , x13_13_10_10_03_02_01_00 , x32_31_32_31_22_21_22_21 ) );

            auto idx = _mm256_setr_epi32( 0 , 2 , 1 , 3 , 6 , 4 , 7 , 5 );

            auto a00_01_02_03_10_11_12_13 = _mm256_permutevar8x32_ps( a00_02_01_03_11_13_10_12 , idx );
            auto a20_21_22_23_30_31_32_33 = _mm256_permutevar8x32_ps( a20_22_21_23_31_33_30_32 , idx );

            auto x00_01_02_03_xx_xx_xx_xx = _mm256_set_ps( 0.f , 0.f , 0.f , 0.f , x03 , x02 , x01 , x00 );

            auto det = 1.f / impl_sum1m256(
                            _mm256_mul_ps( a00_01_02_03_10_11_12_13 , x00_01_02_03_xx_xx_xx_xx ) );
            auto scl = _mm256_set_ps( det , det , det , det , det , det , det , det );

            a00_01_02_03_10_11_12_13 = _mm256_mul_ps( a00_01_02_03_10_11_12_13 , scl );
            a20_21_22_23_30_31_32_33 = _mm256_mul_ps( a20_21_22_23_30_31_32_33 , scl );

            _mm256_store_ps( omdst + 0x00 , a00_01_02_03_10_11_12_13 );
            _mm256_store_ps( omdst + 0x08 , a20_21_22_23_30_31_32_33 );
        }

        /**
         * @brief AVX2-optimized 4x4 matrix inversion for double-precision.
         * @param imsrc Input 4x4 matrix in column-major order (vf64*).
         * @param omdst Output inverse matrix in column-major order (vf64*).
         *
         * @details
         * Uses AVX2 SIMD intrinsics to compute the inverse of a 4x4 matrix.
         * The algorithm computes all 16 cofactors using 4-lane AVX2 operations,
         * processes them in parallel, then computes the determinant and divides.
         *
         * Performance characteristics:
         * - 4-wide SIMD lanes (AVX2 for double-precision)
         * - All cofactors computed simultaneously
         * - Single division for the entire matrix (shared determinant)
         *
         * @note This implementation is optimized for column-major storage layout.
         * @note The input matrix is assumed to be invertible (det != 0).
         * @note This is an experimental implementation enabled by VERTEX_ENABLE_TESTEXT.
         *
         * @warning The algorithm performs no determinant check.
         *          Division by zero may occur for singular matrices.
         */
        inline void inverse_mat4f64( const vf64* imsrc , vf64* omdst ) noexcept {
            auto& x00 = imsrc [ 0x0 ]; auto& x01 = imsrc [ 0x4 ]; auto& x02 = imsrc [ 0x8 ]; auto& x03 = imsrc [ 0xC ];
            auto& x10 = imsrc [ 0x1 ]; auto& x11 = imsrc [ 0x5 ]; auto& x12 = imsrc [ 0x9 ]; auto& x13 = imsrc [ 0xD ];
            auto& x20 = imsrc [ 0x2 ]; auto& x21 = imsrc [ 0x6 ]; auto& x22 = imsrc [ 0xA ]; auto& x23 = imsrc [ 0xE ];
            auto& x30 = imsrc [ 0x3 ]; auto& x31 = imsrc [ 0x7 ]; auto& x32 = imsrc [ 0xB ]; auto& x33 = imsrc [ 0xF ];

            //     [   x00,    x01,    x02,    x03     ]
            //     [   x10,    x11,    x12,    x13     ]
            //     [   x20,    x21,    x22,    x23     ]
            //     [   x30,    x31,    x32,    x33     ]

            auto x11_10_00_00 = _mm256_set_pd( x00 , x00 , x10 , x11 );
            auto x22_21_22_21 = _mm256_set_pd( x21 , x22 , x21 , x22 );
            auto x33_33_33_32 = _mm256_set_pd( x32 , x33 , x33 , x33 );
            auto x12_11_02_01 = _mm256_set_pd( x01 , x02 , x11 , x12 );
            auto x23_23_23_22 = _mm256_set_pd( x22 , x23 , x23 , x23 );
            auto x31_30_30_30 = _mm256_set_pd( x30 , x30 , x30 , x31 );
            auto x13_13_03_02 = _mm256_set_pd( x02 , x03 , x13 , x13 );
            auto x21_20_20_20 = _mm256_set_pd( x20 , x20 , x20 , x21 );
            auto x32_31_32_31 = _mm256_set_pd( x31 , x32 , x31 , x32 );
            auto x01_00_00_00 = _mm256_set_pd( x00 , x00 , x00 , x01 );
            auto x12_11_12_11 = _mm256_set_pd( x11 , x12 , x11 , x12 );
            auto x33_33_23_22 = _mm256_set_pd( x22 , x23 , x33 , x33 );
            auto x02_01_02_01 = _mm256_set_pd( x01 , x02 , x01 , x02 );
            auto x13_13_13_12 = _mm256_set_pd( x12 , x13 , x13 , x13 );
            auto x31_30_20_20 = _mm256_set_pd( x20 , x20 , x30 , x31 );
            auto x03_03_03_02 = _mm256_set_pd( x02 , x03 , x03 , x03 );
            auto x11_10_10_10 = _mm256_set_pd( x10 , x10 , x10 , x11 );
            auto x32_31_22_21 = _mm256_set_pd( x21 , x22 , x31 , x32 );
            auto x10_10_01_00 = _mm256_set_pd( x00 , x01 , x10 , x10 );
            auto x23_22_23_23 = _mm256_set_pd( x23 , x23 , x22 , x23 );
            auto x20_20_21_20 = _mm256_set_pd( x20 , x21 , x20 , x20 );
            auto x33_32_33_33 = _mm256_set_pd( x33 , x33 , x32 , x33 );
            auto x13_12_03_03 = _mm256_set_pd( x03 , x03 , x12 , x13 );
            auto x30_30_31_30 = _mm256_set_pd( x30 , x31 , x30 , x30 );
            auto x00_00_01_00 = _mm256_set_pd( x00 , x01 , x00 , x00 );
            auto x13_12_13_13 = _mm256_set_pd( x13 , x13 , x12 , x13 );
            auto x10_10_11_10 = _mm256_set_pd( x10 , x11 , x10 , x10 );
            auto x33_32_23_23 = _mm256_set_pd( x23 , x23 , x32 , x33 );
            auto x03_02_03_03 = _mm256_set_pd( x03 , x03 , x02 , x03 );
            auto x30_30_21_20 = _mm256_set_pd( x20 , x21 , x30 , x30 );

            auto a00_02_11_13 =
                _mm256_sub_pd(
                _mm256_add_pd( _mm256_mul_pd( x11_10_00_00 , _mm256_mul_pd( x22_21_22_21 , x33_33_33_32 ) ) ,
                _mm256_add_pd( _mm256_mul_pd( x12_11_02_01 , _mm256_mul_pd( x23_23_23_22 , x31_30_30_30 ) ) ,
                               _mm256_mul_pd( x13_13_03_02 , _mm256_mul_pd( x21_20_20_20 , x32_31_32_31 ) ) ) ) ,
                _mm256_add_pd( _mm256_mul_pd( x13_13_03_02 , _mm256_mul_pd( x22_21_22_21 , x31_30_30_30 ) ) ,
                _mm256_add_pd( _mm256_mul_pd( x12_11_02_01 , _mm256_mul_pd( x21_20_20_20 , x33_33_33_32 ) ) ,
                               _mm256_mul_pd( x11_10_00_00 , _mm256_mul_pd( x23_23_23_22 , x32_31_32_31 ) ) ) ) );

            auto a20_22_31_33 =
                _mm256_sub_pd(
                _mm256_add_pd( _mm256_mul_pd( x01_00_00_00 , _mm256_mul_pd( x12_11_12_11 , x33_33_23_22 ) ) ,
                _mm256_add_pd( _mm256_mul_pd( x02_01_02_01 , _mm256_mul_pd( x13_13_13_12 , x31_30_20_20 ) ) ,
                               _mm256_mul_pd( x03_03_03_02 , _mm256_mul_pd( x11_10_10_10 , x32_31_22_21 ) ) ) ) ,
                _mm256_add_pd( _mm256_mul_pd( x03_03_03_02 , _mm256_mul_pd( x12_11_12_11 , x31_30_20_20 ) ) ,
                _mm256_add_pd( _mm256_mul_pd( x02_01_02_01 , _mm256_mul_pd( x11_10_10_10 , x33_33_23_22 ) ) ,
                               _mm256_mul_pd( x01_00_00_00 , _mm256_mul_pd( x13_13_13_12 , x32_31_22_21 ) ) ) ) );

            auto a01_03_10_12 =
                _mm256_sub_pd(
                _mm256_add_pd( _mm256_mul_pd( x10_10_01_00 , _mm256_mul_pd( x23_22_23_23 , x32_31_32_31 ) ) ,
                _mm256_add_pd( _mm256_mul_pd( x12_11_02_01 , _mm256_mul_pd( x20_20_21_20 , x33_32_33_33 ) ) ,
                               _mm256_mul_pd( x13_12_03_03 , _mm256_mul_pd( x22_21_22_21 , x30_30_31_30 ) ) ) ) ,
                _mm256_add_pd( _mm256_mul_pd( x10_10_01_00 , _mm256_mul_pd( x22_21_22_21 , x33_32_33_33 ) ) ,
                _mm256_add_pd( _mm256_mul_pd( x12_11_02_01 , _mm256_mul_pd( x23_22_23_23 , x30_30_31_30 ) ) ,
                               _mm256_mul_pd( x13_12_03_03 , _mm256_mul_pd( x20_20_21_20 , x32_31_32_31 ) ) ) ) );

            auto a21_23_30_32 =
                _mm256_sub_pd(
                _mm256_add_pd( _mm256_mul_pd( x00_00_01_00 , _mm256_mul_pd( x13_12_13_13 , x32_31_22_21 ) ) ,
                _mm256_add_pd( _mm256_mul_pd( x02_01_02_01 , _mm256_mul_pd( x10_10_11_10 , x33_32_23_23 ) ) ,
                               _mm256_mul_pd( x03_02_03_03 , _mm256_mul_pd( x12_11_12_11 , x30_30_21_20 ) ) ) ) ,
                _mm256_add_pd( _mm256_mul_pd( x00_00_01_00 , _mm256_mul_pd( x12_11_12_11 , x33_32_23_23 ) ) ,
                _mm256_add_pd( _mm256_mul_pd( x02_01_02_01 , _mm256_mul_pd( x13_12_13_13 , x30_30_21_20 ) ) ,
                               _mm256_mul_pd( x03_02_03_03 , _mm256_mul_pd( x10_10_11_10 , x32_31_22_21 ) ) ) ) );

// a00_02_11_13  
// a20_22_31_33 
// a01_03_10_12
// a21_23_30_32

            auto l00 = _mm256_castpd256_pd128( a00_02_11_13 );
            auto h00 = _mm256_extractf128_pd( a00_02_11_13 , 1 );
            auto A00 = _mm_cvtsd_f64( l00 );
            auto A02 = _mm_cvtsd_f64( _mm_shuffle_pd( l00 , l00 , 1 ) );
            auto A11 = _mm_cvtsd_f64( h00 );
            auto A13 = _mm_cvtsd_f64( _mm_shuffle_pd( h00 , h00 , 1 ) );

            auto l01 = _mm256_castpd256_pd128( a01_03_10_12 );
            auto h01 = _mm256_extractf128_pd( a01_03_10_12 , 1 );
            auto A01 = _mm_cvtsd_f64( l01 );
            auto A03 = _mm_cvtsd_f64( _mm_shuffle_pd( l01 , l01 , 1 ) );
            auto A10 = _mm_cvtsd_f64( h01 );
            auto A12 = _mm_cvtsd_f64( _mm_shuffle_pd( h01 , h01 , 1 ) );

            auto l20 = _mm256_castpd256_pd128( a20_22_31_33 );
            auto h20 = _mm256_extractf128_pd( a20_22_31_33 , 1 );
            auto A20 = _mm_cvtsd_f64( l20 );
            auto A22 = _mm_cvtsd_f64( _mm_shuffle_pd( l20 , l20 , 1 ) );
            auto A31 = _mm_cvtsd_f64( h20 );
            auto A33 = _mm_cvtsd_f64( _mm_shuffle_pd( h20 , h20 , 1 ) );

            auto l21 = _mm256_castpd256_pd128( a21_23_30_32 );
            auto h21 = _mm256_extractf128_pd( a21_23_30_32 , 1 );
            auto A21 = _mm_cvtsd_f64( l21 );
            auto A23 = _mm_cvtsd_f64( _mm_shuffle_pd( l21 , l21 , 1 ) );
            auto A30 = _mm_cvtsd_f64( h21 );
            auto A32 = _mm_cvtsd_f64( _mm_shuffle_pd( h21 , h21 , 1 ) );

            auto a00_01_02_03 = _mm256_set_pd( A03 , A02 , A01 , A00 ); // [A00, A01, A02, A03]
            auto a10_11_12_13 = _mm256_set_pd( A13 , A12 , A11 , A10 ); // [A10, A11, A12, A13]
            auto a20_21_22_23 = _mm256_set_pd( A23 , A22 , A21 , A20 ); // [A20, A21, A22, A23]
            auto a30_31_32_33 = _mm256_set_pd( A33 , A32 , A31 , A30 ); // [A30, A31, A32, A33]

            auto x00_01_02_03 = _mm256_set_pd( x03 , x02 , x01 , x00 );
            auto ax00_01_02_03 = _mm256_mul_pd( x00_01_02_03 , a00_01_02_03 );

            auto l = _mm256_castpd256_pd128( ax00_01_02_03 );
            auto h = _mm256_extractf128_pd( ax00_01_02_03 , 1 );
            auto sum128 = _mm_add_pd( l , h );
            sum128 = _mm_hadd_pd( sum128 , sum128 );
            double dt = _mm_cvtsd_f64( sum128 );

            auto v3 = _mm256_set_pd( dt , dt , dt , dt );

            auto c0 = _mm256_div_pd( a00_01_02_03 , v3 );
            auto c1 = _mm256_div_pd( a10_11_12_13 , v3 );
            auto c2 = _mm256_div_pd( a20_21_22_23 , v3 );
            auto c3 = _mm256_div_pd( a30_31_32_33 , v3 );

            _mm256_store_pd( omdst + 0x00 , c0 );
            _mm256_store_pd( omdst + 0x04 , c1 );
            _mm256_store_pd( omdst + 0x08 , c2 );
            _mm256_store_pd( omdst + 0x0C , c3 );
        }
    }

    /**
     * @brief SIMD-accelerated 4x4 matrix inversion (single-precision).
     * @param mat Input 4x4 matrix.
     * @return Inverse 4x4 matrix.
     *
     * @details
     * Uses AVX2-optimized SIMD implementation for maximum performance.
     * Computes the inverse using the cofactor method with 8-wide SIMD lanes.
     *
     * Performance characteristics:
     * - All 16 cofactors computed in parallel using SIMD
     * - Single determinant computation
     * - Single division for the entire matrix
     *
     * @note Enabled only when VERTEX_ENABLE_TESTEXT is defined.
     * @note The input matrix is assumed to be invertible (det != 0).
     *
     * @warning No determinant check is performed. Division by zero may occur
     *          for singular matrices.
     *
     * @see inverse_mat4f32
     */
    matrix_template<vf32 , 4 , 4> inverse( const matrix_template<vf32 , 4 , 4>& mat ) noexcept {
        matrix_template<vf32 , 4 , 4> ans {};
        test_impl::inverse_mat4f32( &mat.data [ 0 ][ 0 ] , &ans.data [ 0 ][ 0 ] );
        return ans;
    }

    /**
     * @brief SIMD-accelerated 4x4 matrix inversion (double-precision).
     * @param mat Input 4x4 matrix.
     * @return Inverse 4x4 matrix.
     *
     * @details
     * Uses AVX2-optimized SIMD implementation for double-precision.
     * Computes the inverse using the cofactor method with 4-wide SIMD lanes.
     *
     * Performance characteristics:
     * - All 16 cofactors computed in parallel using SIMD
     * - Single determinant computation
     * - Single division for the entire matrix
     *
     * @note Enabled only when VERTEX_ENABLE_TESTEXT is defined.
     * @note The input matrix is assumed to be invertible (det != 0).
     *
     * @warning No determinant check is performed. Division by zero may occur
     *          for singular matrices.
     *
     * @see inverse_mat4f64
     */
    matrix_template<vf64 , 4 , 4> inverse( const matrix_template<vf64 , 4 , 4>& mat ) noexcept {
        matrix_template<vf64 , 4 , 4> ans {};
        test_impl::inverse_mat4f64( &mat.data [ 0 ][ 0 ] , &ans.data [ 0 ][ 0 ] );
        return ans;
    }
#endif
}

#endif