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
 */

/**
 * @file      vtxsimd1.hpp
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

#ifndef VERTEX_SIMD1_HPP
#define VERTEX_SIMD1_HPP
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
     * @brief     Internal SIMD implementation details.
     * @namespace impl
     */
    namespace impl {

        template <size_t... I> VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
        void compute_vecadd( const vf64* VERTEX_RESTRICT veca , const vf64* VERTEX_RESTRICT vecb , vf64* VERTEX_RESTRICT vecc , std::index_sequence<I...> ) noexcept {
            ( ( _mm256_store_pd( vecc + I * 4 , _mm256_add_pd( _mm256_load_pd( veca + I * 4 ) , _mm256_load_pd( vecb + I * 4 ) ) ) ) , ... );
        }

        template <size_t... I> VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
        void compute_vecsub( const vf64* VERTEX_RESTRICT veca , const vf64* VERTEX_RESTRICT vecb , vf64* VERTEX_RESTRICT vecc , std::index_sequence<I...> ) noexcept {
            ( ( _mm256_store_pd( vecc + I * 4 , _mm256_sub_pd( _mm256_load_pd( veca + I * 4 ) , _mm256_load_pd( vecb + I * 4 ) ) ) ) , ... );
        }

        template <size_t... I> VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
        void compute_vecadd( const vf32* VERTEX_RESTRICT veca , const vf32* VERTEX_RESTRICT vecb , vf32* VERTEX_RESTRICT vecc , std::index_sequence<I...> ) noexcept {
            ( ( _mm256_store_ps( vecc + I * 8 , _mm256_add_ps( _mm256_load_ps( veca + I * 8 ) , _mm256_load_ps( vecb + I * 8 ) ) ) ) , ... );
        }

        template <size_t... I> VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
        void compute_vecsub( const vf32* VERTEX_RESTRICT veca , const vf32* VERTEX_RESTRICT vecb , vf32* VERTEX_RESTRICT vecc , std::index_sequence<I...> ) noexcept {
            ( ( _mm256_store_ps( vecc + I * 8 , _mm256_sub_ps( _mm256_load_ps( veca + I * 8 ) , _mm256_load_ps( vecb + I * 8 ) ) ) ) , ... );
        }

        template <size_t... I> VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
        void compute_vecadd( const vi32* VERTEX_RESTRICT veca , const vi32* VERTEX_RESTRICT vecb , vi32* VERTEX_RESTRICT vecc , std::index_sequence<I...> ) noexcept {
            ( ( _mm256_store_epi32( vecc + I * 8 , _mm256_add_epi32( _mm256_load_epi32( veca + I * 8 ) , _mm256_load_epi32( vecb + I * 8 ) ) ) ) , ... );
        }

        template <size_t... I> VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
        void compute_vecsub( const vi32* VERTEX_RESTRICT veca , const vi32* VERTEX_RESTRICT vecb , vi32* VERTEX_RESTRICT vecc , std::index_sequence<I...> ) noexcept {
            ( ( _mm256_store_epi32( vecc + I * 8 , _mm256_sub_epi32( _mm256_load_epi32( veca + I * 8 ) , _mm256_load_epi32( vecb + I * 8 ) ) ) ) , ... );
        }

        template <size_t... I> VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
       void compute_vecadd( const vu32* VERTEX_RESTRICT veca , const vu32* VERTEX_RESTRICT vecb , vu32* VERTEX_RESTRICT vecc , std::index_sequence<I...> ) noexcept {
            ( ( _mm256_store_epi32( vecc + I * 8 , _mm256_add_epi32( _mm256_load_epi32( veca + I * 8 ) , _mm256_load_epi32( vecb + I * 8 ) ) ) ) , ... );
        }

        template <size_t... I> VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
        void compute_vecsub( const vu32* VERTEX_RESTRICT veca , const vu32* VERTEX_RESTRICT vecb , vu32* VERTEX_RESTRICT vecc , std::index_sequence<I...> ) noexcept {
            ( ( _mm256_store_epi32( vecc + I * 8 , _mm256_sub_epi32( _mm256_load_epi32( veca + I * 8 ) , _mm256_load_epi32( vecb + I * 8 ) ) ) ) , ... );
        }
    }

    template <size_t R , size_t C> requires( ( R * C ) % 4 == 0 and ( R * C ) > 4 )
        matrix_template<vf64 , R , C>& operator+=( matrix_template<vf64 , R , C>& mata , const matrix_template<vf64 , R , C>& matb ) noexcept {
        impl::compute_vecadd( &mata.data [ 0 ][ 0 ] , &matb.data [ 0 ][ 0 ] , &mata.data [ 0 ][ 0 ] , std::make_index_sequence< ( R * C ) / 4 >( ) );
        return mata;
    }

    template <size_t R , size_t C> requires( ( R * C ) % 4 == 0 and ( R * C ) > 4 )
        matrix_template<vf64 , R , C>& operator-=( matrix_template<vf64 , R , C>& mata , const matrix_template<vf64 , R , C>& matb ) noexcept {
        impl::compute_vecsub( &mata.data [ 0 ][ 0 ] , &matb.data [ 0 ][ 0 ] , &mata.data [ 0 ][ 0 ] , std::make_index_sequence< ( R * C ) / 4 >( ) );
        return mata;
    }

    template <size_t R , size_t C> requires( ( R * C ) % 8 == 0 and ( R * C ) > 8 )
        matrix_template<vf32 , R , C>& operator+=( matrix_template<vf32 , R , C>& mata , const matrix_template<vf32 , R , C>& matb ) noexcept {
        impl::compute_vecadd( &mata.data [ 0 ][ 0 ] , &matb.data [ 0 ][ 0 ] , &mata.data [ 0 ][ 0 ] , std::make_index_sequence< ( R * C ) / 8 >( ) );
        return mata;
    }

    template <size_t R , size_t C> requires( ( R * C ) % 8 == 0 and ( R * C ) > 8 )
        matrix_template<vf32 , R , C>& operator-=( matrix_template<vf32 , R , C>& mata , const matrix_template<vf32 , R , C>& matb ) noexcept {
        impl::compute_vecsub( &mata.data [ 0 ][ 0 ] , &matb.data [ 0 ][ 0 ] , &mata.data [ 0 ][ 0 ] , std::make_index_sequence< ( R * C ) / 8 >( ) );
        return mata;
    }

    template <size_t R , size_t C> requires( ( R * C ) % 8 == 0 and ( R * C ) > 8 )
        matrix_template<vi32 , R , C>& operator+=( matrix_template<vi32 , R , C>& mata , const matrix_template<vi32 , R , C>& matb ) noexcept {
        impl::compute_vecadd( &mata.data [ 0 ][ 0 ] , &matb.data [ 0 ][ 0 ] , &mata.data [ 0 ][ 0 ] , std::make_index_sequence< ( R * C ) / 8 >( ) );
        return mata;
    }

    template <size_t R , size_t C> requires( ( R * C ) % 8 == 0 and ( R * C ) > 8 )
        matrix_template<vi32 , R , C>& operator-=( matrix_template<vi32 , R , C>& mata , const matrix_template<vi32 , R , C>& matb ) noexcept {
        impl::compute_vecsub( &mata.data [ 0 ][ 0 ] , &matb.data [ 0 ][ 0 ] , &mata.data [ 0 ][ 0 ] , std::make_index_sequence< ( R * C ) / 8 >( ) );
        return mata;
    }

    template <size_t R , size_t C> requires( ( R * C ) % 8 == 0 and ( R * C ) > 8 )
        matrix_template<vu32 , R , C>& operator+=( matrix_template<vu32 , R , C>& mata , const matrix_template<vu32 , R , C>& matb ) noexcept {
        impl::compute_vecadd( &mata.data [ 0 ][ 0 ] , &matb.data [ 0 ][ 0 ] , &mata.data [ 0 ][ 0 ] , std::make_index_sequence< ( R * C ) / 8 >( ) );
        return mata;
    }

    template <size_t R , size_t C> requires( ( R * C ) % 8 == 0 and ( R * C ) > 8 )
        matrix_template<vu32 , R , C>& operator-=( matrix_template<vu32 , R , C>& mata , const matrix_template<vu32 , R , C>& matb ) noexcept {
        impl::compute_vecsub( &mata.data [ 0 ][ 0 ] , &matb.data [ 0 ][ 0 ] , &mata.data [ 0 ][ 0 ] , std::make_index_sequence< ( R * C ) / 8 >( ) );
        return mata;
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
    inline auto& operator+=( vector_template<vf64, 4>& vca, const vector_template<vf64, 4>& vcb ) noexcept {
        _mm256_store_pd( vca.data, _mm256_add_pd( _mm256_load_pd( vca.data ), _mm256_load_pd( vcb.data ) ) );
        return vca;
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
    inline auto& operator-=( vector_template<vf64, 4>& vca, const vector_template<vf64, 4>& vcb ) noexcept {
        _mm256_store_pd( vca.data, _mm256_sub_pd( _mm256_load_pd( vca.data ), _mm256_load_pd( vcb.data ) ) );
        return vca;
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
    inline auto& operator*=( vector_template<vf64, 4>& vca, const vector_template<vf64, 4>& vcb ) noexcept {
        _mm256_store_pd( vca.data, _mm256_mul_pd( _mm256_load_pd( vca.data ), _mm256_load_pd( vcb.data ) ) );
        return vca;
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
    inline auto& operator/=( vector_template<vf64, 4>& vca, const vector_template<vf64, 4>& vcb ) noexcept {
        _mm256_store_pd( vca.data, _mm256_div_pd( _mm256_load_pd( vca.data ), _mm256_load_pd( vcb.data ) ) );
        return vca;
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_SSE4_NAME )
    inline auto& operator+=( vector_template<vf32, 4>& vca, const vector_template<vf32, 4>& vcb ) noexcept {
        _mm_store_ps( vca.data, _mm_add_ps( _mm_load_ps( vca.data ), _mm_load_ps( vcb.data ) ) );
        return vca;
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_SSE4_NAME )
    inline auto& operator-=( vector_template<vf32, 4>& vca, const vector_template<vf32, 4>& vcb ) noexcept {
        _mm_store_ps( vca.data, _mm_sub_ps( _mm_load_ps( vca.data ), _mm_load_ps( vcb.data ) ) );
        return vca;
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_SSE4_NAME )
    inline auto& operator*=( vector_template<vf32, 4>& vca, const vector_template<vf32, 4>& vcb ) noexcept {
        _mm_store_ps( vca.data, _mm_mul_ps( _mm_load_ps( vca.data ), _mm_load_ps( vcb.data ) ) );
        return vca;
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_SSE4_NAME )
    inline auto& operator/=( vector_template<vf32, 4>& vca, const vector_template<vf32, 4>& vcb ) noexcept {
        _mm_store_ps( vca.data, _mm_div_ps( _mm_load_ps( vca.data ), _mm_load_ps( vcb.data ) ) );
        return vca;
    }
}

#endif