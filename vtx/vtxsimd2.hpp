
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

#ifndef VERTEX_SIMD2_HPP
#define VERTEX_SIMD2_HPP
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
#include <thread>

#ifdef VERTEX_ENABLE_MTHREAD
#include <thread>
#endif

#include "vtxtuple.hpp"

namespace vtx {
    /**
     * @brief       Internal SIMD implementation tmpZails.
     * @namespace   impl
     */
    namespace impl {
        VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
        inline auto expand_mm128( const __m128& v ) noexcept {
            auto tmpV = _mm256_castps128_ps256( v );
            return _mm256_permute2f128_ps( tmpV, tmpV, 0x00 );
        }

        VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_MFMA_NAME )
        inline auto compute_temp(
            const __m128& v1 , const __m128& v2 , const __m128& v3 ,
            const __m128& v4 , const __m128& v5 , const __m128& v6 ,
            const __m128& v7 , const __m128& v8 , const __m128& v9
        ) noexcept {
            return
                _mm_fmadd_ps( _mm_mul_ps( v7 , v8 ) , v9 ,
                _mm_fmadd_ps( _mm_mul_ps( v4 , v5 ) , v6 ,
                  _mm_mul_ps( _mm_mul_ps( v1 , v2 ) , v3 ) ) );
        }

        VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_MFMA_NAME )
        inline auto compute_temp(
            const __m256& v1 , const __m256& v2 , const __m256& v3 ,
            const __m256& v4 , const __m256& v5 , const __m256& v6 ,
            const __m256& v7 , const __m256& v8 , const __m256& v9
        ) noexcept {
            return
                _mm256_fmadd_ps( _mm256_mul_ps( v7 , v8 ) , v9 ,
                _mm256_fmadd_ps( _mm256_mul_ps( v4 , v5 ) , v6 ,
                  _mm256_mul_ps( _mm256_mul_ps( v1 , v2 ) , v3 ) ) );
        }

        VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_MFMA_NAME )
        inline auto compute_temp(
            const __m256d& v1 , const __m256d& v2 , const __m256d& v3 ,
            const __m256d& v4 , const __m256d& v5 , const __m256d& v6 ,
            const __m256d& v7 , const __m256d& v8 , const __m256d& v9
        ) noexcept {
            return
                _mm256_fmadd_pd( _mm256_mul_pd( v7 , v8 ) , v9 ,
                _mm256_fmadd_pd( _mm256_mul_pd( v4 , v5 ) , v6 ,
                  _mm256_mul_pd( _mm256_mul_pd( v1 , v2 ) , v3 ) ) );
        }

        VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
        inline vf32 compute_sum( __m128 v ) noexcept {
            v = _mm_hadd_ps( v , v );
            v = _mm_hadd_ps( v , v );
            return _mm_cvtss_f32( v );
        }

        VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
        inline vf32 compute_sum( const __m256& v ) noexcept {
            auto vl = _mm256_castps256_ps128( v );
            auto vh = _mm256_extractf128_ps( v , 1 );
            auto vs = _mm_add_ps ( vl , vh );
            vs = _mm_hadd_ps( vs , vs );
            vs = _mm_hadd_ps( vs , vs );
            return _mm_cvtss_f32( vs );
        }

        VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
        inline vf64 compute_sum( const __m256d& v ) noexcept {
            auto vl = _mm256_castpd256_pd128( v );
            auto vh = _mm256_extractf128_pd( v , 1 );
            auto vs = _mm_add_pd( vl , vh );
            vs = _mm_hadd_pd( vs , vs );
            return _mm_cvtsd_f64( vs );
        }
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
    inline auto operator*( const mat4x4f32 & a , const mat4x4f32 & b ) noexcept {
        auto tmpA0 = _mm_load_ps( &a.data [ 0 ][ 0 ] + 0x00 );
        auto tmpA1 = _mm_load_ps( &a.data [ 0 ][ 0 ] + 0x04 );
        auto tmpA2 = _mm_load_ps( &a.data [ 0 ][ 0 ] + 0x08 );
        auto tmpA3 = _mm_load_ps( &a.data [ 0 ][ 0 ] + 0x0C );

        auto tmpB01 = _mm256_load_ps( &b.data [ 0 ][ 0 ] + 0x00 );
        auto tmpB23 = _mm256_load_ps( &b.data [ 0 ][ 0 ] + 0x08 );

        _MM_TRANSPOSE4_PS( tmpA0 , tmpA1 , tmpA2 , tmpA3 );

        auto tmpA00 = impl::expand_mm128( tmpA0 );
        auto tmpA11 = impl::expand_mm128( tmpA1 );
        auto tmpA22 = impl::expand_mm128( tmpA2 );
        auto tmpA33 = impl::expand_mm128( tmpA3 );

        auto tmp0 = _mm256_mul_ps( tmpA00 , tmpB01 );
        auto tmp1 = _mm256_mul_ps( tmpA00 , tmpB23 );
        auto tmp2 = _mm256_mul_ps( tmpA11 , tmpB01 );
        auto tmp3 = _mm256_mul_ps( tmpA11 , tmpB23 );
        auto tmp4 = _mm256_mul_ps( tmpA22 , tmpB01 );
        auto tmp5 = _mm256_mul_ps( tmpA22 , tmpB23 );
        auto tmp6 = _mm256_mul_ps( tmpA33 , tmpB01 );
        auto tmp7 = _mm256_mul_ps( tmpA33 , tmpB23 );

        tmp0 = _mm256_hadd_ps( tmp0 , tmp0 );
        tmp0 = _mm256_hadd_ps( tmp0 , tmp0 );
        tmp1 = _mm256_hadd_ps( tmp1 , tmp1 );
        tmp1 = _mm256_hadd_ps( tmp1 , tmp1 );
        tmp2 = _mm256_hadd_ps( tmp2 , tmp2 );
        tmp2 = _mm256_hadd_ps( tmp2 , tmp2 );
        tmp3 = _mm256_hadd_ps( tmp3 , tmp3 );
        tmp3 = _mm256_hadd_ps( tmp3 , tmp3 );
        tmp4 = _mm256_hadd_ps( tmp4 , tmp4 );
        tmp4 = _mm256_hadd_ps( tmp4 , tmp4 );
        tmp5 = _mm256_hadd_ps( tmp5 , tmp5 );
        tmp5 = _mm256_hadd_ps( tmp5 , tmp5 );
        tmp6 = _mm256_hadd_ps( tmp6 , tmp6 );
        tmp6 = _mm256_hadd_ps( tmp6 , tmp6 );
        tmp7 = _mm256_hadd_ps( tmp7 , tmp7 );
        tmp7 = _mm256_hadd_ps( tmp7 , tmp7 );

        mat4x4f32 result {};

        result.data [ 0 ][ 0 ] = _mm_cvtss_f32( _mm256_castps256_ps128( tmp0 ) );
        result.data [ 2 ][ 0 ] = _mm_cvtss_f32( _mm256_castps256_ps128( tmp1 ) );
        result.data [ 0 ][ 1 ] = _mm_cvtss_f32( _mm256_castps256_ps128( tmp2 ) );
        result.data [ 2 ][ 1 ] = _mm_cvtss_f32( _mm256_castps256_ps128( tmp3 ) );
        result.data [ 0 ][ 2 ] = _mm_cvtss_f32( _mm256_castps256_ps128( tmp4 ) );
        result.data [ 2 ][ 2 ] = _mm_cvtss_f32( _mm256_castps256_ps128( tmp5 ) );
        result.data [ 0 ][ 3 ] = _mm_cvtss_f32( _mm256_castps256_ps128( tmp6 ) );
        result.data [ 2 ][ 3 ] = _mm_cvtss_f32( _mm256_castps256_ps128( tmp7 ) );

        result.data [ 1 ][ 0 ] = _mm_cvtss_f32( _mm256_extractf128_ps( tmp0 , 1 ) );
        result.data [ 3 ][ 0 ] = _mm_cvtss_f32( _mm256_extractf128_ps( tmp1 , 1 ) );
        result.data [ 1 ][ 1 ] = _mm_cvtss_f32( _mm256_extractf128_ps( tmp2 , 1 ) );
        result.data [ 3 ][ 1 ] = _mm_cvtss_f32( _mm256_extractf128_ps( tmp3 , 1 ) );
        result.data [ 1 ][ 2 ] = _mm_cvtss_f32( _mm256_extractf128_ps( tmp4 , 1 ) );
        result.data [ 3 ][ 2 ] = _mm_cvtss_f32( _mm256_extractf128_ps( tmp5 , 1 ) );
        result.data [ 1 ][ 3 ] = _mm_cvtss_f32( _mm256_extractf128_ps( tmp6 , 1 ) );
        result.data [ 3 ][ 3 ] = _mm_cvtss_f32( _mm256_extractf128_ps( tmp7 , 1 ) );

        return result;
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
    inline auto operator*( const mat4x4i32& a , const mat4x4i32& b ) noexcept {
        auto tmpB01 = _mm256_load_si256( reinterpret_cast< const __m256i* >( &b.data [ 0 ][ 0 ] + 0x00 ) );
        auto tmpB23 = _mm256_load_si256( reinterpret_cast< const __m256i* >( &b.data [ 0 ][ 0 ] + 0x08 ) );

        auto col0 = _mm_load_si128( reinterpret_cast< const __m128i* >( &a.data [ 0 ][ 0 ] + 0x00 ) );
        auto col1 = _mm_load_si128( reinterpret_cast< const __m128i* >( &a.data [ 0 ][ 0 ] + 0x04 ) );
        auto col2 = _mm_load_si128( reinterpret_cast< const __m128i* >( &a.data [ 0 ][ 0 ] + 0x08 ) );
        auto col3 = _mm_load_si128( reinterpret_cast< const __m128i* >( &a.data [ 0 ][ 0 ] + 0x0C ) );

        auto trans_tmp0 = _mm_unpacklo_epi32( col0 , col1 );
        auto trans_tmp1 = _mm_unpackhi_epi32( col0 , col1 );
        auto trans_tmp2 = _mm_unpacklo_epi32( col2 , col3 );
        auto trans_tmp3 = _mm_unpackhi_epi32( col2 , col3 );

        auto tmpA0 = _mm_unpacklo_epi64( trans_tmp0 , trans_tmp2 );
        auto tmpA1 = _mm_unpackhi_epi64( trans_tmp0 , trans_tmp2 );
        auto tmpA2 = _mm_unpacklo_epi64( trans_tmp1 , trans_tmp3 );
        auto tmpA3 = _mm_unpackhi_epi64( trans_tmp1 , trans_tmp3 );

        auto tmpA00 = _mm256_broadcastsi128_si256( tmpA0 );
        auto tmpA11 = _mm256_broadcastsi128_si256( tmpA1 );
        auto tmpA22 = _mm256_broadcastsi128_si256( tmpA2 );
        auto tmpA33 = _mm256_broadcastsi128_si256( tmpA3 );

        auto tmp0 = _mm256_mullo_epi32( tmpA00 , tmpB01 );
        auto tmp1 = _mm256_mullo_epi32( tmpA00 , tmpB23 );
        auto tmp2 = _mm256_mullo_epi32( tmpA11 , tmpB01 );
        auto tmp3 = _mm256_mullo_epi32( tmpA11 , tmpB23 );
        auto tmp4 = _mm256_mullo_epi32( tmpA22 , tmpB01 );
        auto tmp5 = _mm256_mullo_epi32( tmpA22 , tmpB23 );
        auto tmp6 = _mm256_mullo_epi32( tmpA33 , tmpB01 );
        auto tmp7 = _mm256_mullo_epi32( tmpA33 , tmpB23 );

        tmp0 = _mm256_hadd_epi32( tmp0 , tmp0 );
        tmp0 = _mm256_hadd_epi32( tmp0 , tmp0 );
        tmp1 = _mm256_hadd_epi32( tmp1 , tmp1 );
        tmp1 = _mm256_hadd_epi32( tmp1 , tmp1 );
        tmp2 = _mm256_hadd_epi32( tmp2 , tmp2 );
        tmp2 = _mm256_hadd_epi32( tmp2 , tmp2 );
        tmp3 = _mm256_hadd_epi32( tmp3 , tmp3 );
        tmp3 = _mm256_hadd_epi32( tmp3 , tmp3 );
        tmp4 = _mm256_hadd_epi32( tmp4 , tmp4 );
        tmp4 = _mm256_hadd_epi32( tmp4 , tmp4 );
        tmp5 = _mm256_hadd_epi32( tmp5 , tmp5 );
        tmp5 = _mm256_hadd_epi32( tmp5 , tmp5 );
        tmp6 = _mm256_hadd_epi32( tmp6 , tmp6 );
        tmp6 = _mm256_hadd_epi32( tmp6 , tmp6 );
        tmp7 = _mm256_hadd_epi32( tmp7 , tmp7 );
        tmp7 = _mm256_hadd_epi32( tmp7 , tmp7 );

        mat4x4i32 result {};

        result.data [ 0 ][ 0 ] = _mm_cvtsi128_si32( _mm256_castsi256_si128( tmp0 ) );
        result.data [ 1 ][ 0 ] = _mm_cvtsi128_si32( _mm256_castsi256_si128( tmp2 ) );
        result.data [ 2 ][ 0 ] = _mm_cvtsi128_si32( _mm256_castsi256_si128( tmp1 ) );
        result.data [ 3 ][ 0 ] = _mm_cvtsi128_si32( _mm256_castsi256_si128( tmp3 ) );
        result.data [ 0 ][ 2 ] = _mm_cvtsi128_si32( _mm256_castsi256_si128( tmp4 ) );
        result.data [ 0 ][ 3 ] = _mm_cvtsi128_si32( _mm256_castsi256_si128( tmp6 ) );
        result.data [ 2 ][ 2 ] = _mm_cvtsi128_si32( _mm256_castsi256_si128( tmp5 ) );
        result.data [ 2 ][ 3 ] = _mm_cvtsi128_si32( _mm256_castsi256_si128( tmp7 ) );

        result.data [ 0 ][ 1 ] = _mm_cvtsi128_si32( _mm256_extracti128_si256( tmp0 , 1 ) );
        result.data [ 1 ][ 1 ] = _mm_cvtsi128_si32( _mm256_extracti128_si256( tmp2 , 1 ) );
        result.data [ 2 ][ 1 ] = _mm_cvtsi128_si32( _mm256_extracti128_si256( tmp1 , 1 ) );
        result.data [ 3 ][ 1 ] = _mm_cvtsi128_si32( _mm256_extracti128_si256( tmp3 , 1 ) );
        result.data [ 1 ][ 2 ] = _mm_cvtsi128_si32( _mm256_extracti128_si256( tmp4 , 1 ) );
        result.data [ 1 ][ 3 ] = _mm_cvtsi128_si32( _mm256_extracti128_si256( tmp6 , 1 ) );
        result.data [ 3 ][ 2 ] = _mm_cvtsi128_si32( _mm256_extracti128_si256( tmp5 , 1 ) );
        result.data [ 3 ][ 3 ] = _mm_cvtsi128_si32( _mm256_extracti128_si256( tmp7 , 1 ) );

        return result;
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
    inline auto inverse_mthread( const mat4x4f32& mat ) noexcept -> mat4x4f32 {
        auto tmp00 = __m128{};
        auto tmp01 = __m128{};
        auto tmp02 = __m128{};
        auto tmp03 = __m128{};
        auto tmp04 = __m128{};
        auto tmp05 = __m128{};
        auto tmp06 = __m128{};
        auto tmp07 = __m128{};
        auto tmp08 = __m128{};
        auto tmp09 = __m128{};
        auto tmp10 = __m128{};
        auto tmp11 = __m128{};

        std::atomic<int> syncFlag { 0 };
        auto shuffleIndex = _mm_set_epi32( 3 , 1 , 2 , 0 );
        __m128 scale = _mm_setzero_ps( );
        std::atomic<bool> scaleReady { false };
        mat4x4f32 result;

        std::jthread threadA( [ & ] VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME ) {
            tmp00 = _mm_set_ps( mat.data [ 2 ][ 1 ] , mat.data [ 3 ][ 1 ] , mat.data [ 0 ][ 1 ] , mat.data [ 1 ][ 1 ] );
            tmp01 = _mm_set_ps( mat.data [ 1 ][ 2 ] , mat.data [ 2 ][ 2 ] , mat.data [ 1 ][ 2 ] , mat.data [ 2 ][ 2 ] );
            tmp02 = _mm_set_ps( mat.data [ 0 ][ 3 ] , mat.data [ 0 ][ 3 ] , mat.data [ 3 ][ 3 ] , mat.data [ 3 ][ 3 ] );

            syncFlag.fetch_add( 1 , std::memory_order_release );
            while ( syncFlag.load( std::memory_order_acquire ) != 4 ) {
                std::this_thread::yield( );
            }

            auto tmpA = _mm_sub_ps(
                impl::compute_temp( tmp00 , tmp01 , tmp02 , tmp03 , tmp04 , tmp05 , tmp06 , tmp07 , tmp08 ) ,
                impl::compute_temp( tmp06 , tmp01 , tmp05 , tmp03 , tmp07 , tmp02 , tmp00 , tmp04 , tmp08 ) );
            tmpA = _mm_permutevar_ps( tmpA , shuffleIndex );

            auto tmpE = _mm_set_ps( mat.data [ 3 ][ 0 ] , mat.data [ 2 ][ 0 ] , mat.data [ 1 ][ 0 ] , mat.data [ 0 ][ 0 ] );
            auto tmpF = _mm_mul_ps( tmpE , tmpA );
            auto det = impl::compute_sum( tmpF );
            scale = _mm_set1_ps( 1.f / det );
            scaleReady.store( true , std::memory_order_release );

            tmpA = _mm_mul_ps( tmpA , scale );
            _mm_store_ps( &result.data [ 0 ][ 0 ] + 0x00 , tmpA );
        } );

        std::jthread threadB( [ & ] VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME ) {
            tmp03 = _mm_set_ps( mat.data [ 1 ][ 1 ] , mat.data [ 2 ][ 1 ] , mat.data [ 1 ][ 1 ] , mat.data [ 2 ][ 1 ] );
            tmp04 = _mm_set_ps( mat.data [ 0 ][ 2 ] , mat.data [ 0 ][ 2 ] , mat.data [ 3 ][ 2 ] , mat.data [ 3 ][ 2 ] );
            tmp05 = _mm_set_ps( mat.data [ 2 ][ 3 ] , mat.data [ 3 ][ 3 ] , mat.data [ 0 ][ 3 ] , mat.data [ 1 ][ 3 ] );

            syncFlag.fetch_add( 1 , std::memory_order_release );
            while ( syncFlag.load( std::memory_order_acquire ) != 4 ) {
                std::this_thread::yield( );
            }

            auto tmpB = _mm_sub_ps(
                impl::compute_temp( tmp09 , tmp01 , tmp05 , tmp10 , tmp07 , tmp02 , tmp11 , tmp04 , tmp08 ) ,
                impl::compute_temp( tmp11 , tmp01 , tmp02 , tmp10 , tmp04 , tmp05 , tmp09 , tmp07 , tmp08 )
            );
            tmpB = _mm_permutevar_ps( tmpB , shuffleIndex );

            while ( !scaleReady.load( std::memory_order_acquire ) ) {
                std::this_thread::yield( );
            }

            tmpB = _mm_mul_ps( tmpB , scale );
            _mm_store_ps( &result.data [ 0 ][ 0 ] + 0x04 , tmpB );
        } );

        std::jthread threadC( [ & ] VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME ) {
            tmp06 = _mm_set_ps( mat.data [ 0 ][ 1 ] , mat.data [ 0 ][ 1 ] , mat.data [ 3 ][ 1 ] , mat.data [ 3 ][ 1 ] );
            tmp07 = _mm_set_ps( mat.data [ 2 ][ 2 ] , mat.data [ 3 ][ 2 ] , mat.data [ 0 ][ 2 ] , mat.data [ 1 ][ 2 ] );
            tmp08 = _mm_set_ps( mat.data [ 1 ][ 3 ] , mat.data [ 2 ][ 3 ] , mat.data [ 1 ][ 3 ] , mat.data [ 2 ][ 3 ] );

            syncFlag.fetch_add( 1 , std::memory_order_release );
            while ( syncFlag.load( std::memory_order_acquire ) != 4 ) {
                std::this_thread::yield( );
            }

            auto tmpC = _mm_sub_ps(
                impl::compute_temp( tmp11 , tmp03 , tmp02 , tmp10 , tmp06 , tmp05 , tmp09 , tmp00 , tmp08 ) ,
                impl::compute_temp( tmp09 , tmp03 , tmp05 , tmp10 , tmp00 , tmp02 , tmp11 , tmp06 , tmp08 )
            );
            tmpC = _mm_permutevar_ps( tmpC , shuffleIndex );

            while ( !scaleReady.load( std::memory_order_acquire ) ) {
                std::this_thread::yield( );
            }

            tmpC = _mm_mul_ps( tmpC , scale );
            _mm_store_ps( &result.data [ 0 ][ 0 ] + 0x08 , tmpC );
        } );

        std::jthread threadD( [ & ] VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME ) {
            tmp09 = _mm_set_ps( mat.data [ 0 ][ 0 ] , mat.data [ 0 ][ 0 ] , mat.data [ 3 ][ 0 ] , mat.data [ 3 ][ 0 ] );
            tmp10 = _mm_set_ps( mat.data [ 1 ][ 0 ] , mat.data [ 2 ][ 0 ] , mat.data [ 1 ][ 0 ] , mat.data [ 2 ][ 0 ] );
            tmp11 = _mm_set_ps( mat.data [ 2 ][ 0 ] , mat.data [ 3 ][ 0 ] , mat.data [ 0 ][ 0 ] , mat.data [ 1 ][ 0 ] );

            syncFlag.fetch_add( 1 , std::memory_order_release );
            while ( syncFlag.load( std::memory_order_acquire ) != 4 ) {
                std::this_thread::yield( );
            }

            auto tmpD = _mm_sub_ps(
                impl::compute_temp( tmp09 , tmp03 , tmp07 , tmp10 , tmp00 , tmp04 , tmp11 , tmp06 , tmp01 ) ,
                impl::compute_temp( tmp11 , tmp03 , tmp04 , tmp10 , tmp06 , tmp07 , tmp09 , tmp00 , tmp01 )
            );
            tmpD = _mm_permutevar_ps( tmpD , shuffleIndex );

            while ( !scaleReady.load( std::memory_order_acquire ) ) {
                std::this_thread::yield( );
            }

            tmpD = _mm_mul_ps( tmpD , scale );
            _mm_store_ps( &result.data [ 0 ][ 0 ] + 0x0C , tmpD );
        } );

        return result;
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
    inline auto inverse( const mat4x4f32& mat ) noexcept -> mat4x4f32 {
        auto tmp00 = _mm_set_ps( mat.data[ 2 ][ 1 ] , mat.data[ 3 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 1 ][ 1 ] );
        auto tmp01 = _mm_set_ps( mat.data[ 1 ][ 2 ] , mat.data[ 2 ][ 2 ] , mat.data[ 1 ][ 2 ] , mat.data[ 2 ][ 2 ] );
        auto tmp02 = _mm_set_ps( mat.data[ 0 ][ 3 ] , mat.data[ 0 ][ 3 ] , mat.data[ 3 ][ 3 ] , mat.data[ 3 ][ 3 ] );
        auto tmp03 = _mm_set_ps( mat.data[ 1 ][ 1 ] , mat.data[ 2 ][ 1 ] , mat.data[ 1 ][ 1 ] , mat.data[ 2 ][ 1 ] );
        auto tmp04 = _mm_set_ps( mat.data[ 0 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 3 ][ 2 ] , mat.data[ 3 ][ 2 ] );
        auto tmp05 = _mm_set_ps( mat.data[ 2 ][ 3 ] , mat.data[ 3 ][ 3 ] , mat.data[ 0 ][ 3 ] , mat.data[ 1 ][ 3 ] );
        auto tmp06 = _mm_set_ps( mat.data[ 0 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 3 ][ 1 ] , mat.data[ 3 ][ 1 ] );
        auto tmp07 = _mm_set_ps( mat.data[ 2 ][ 2 ] , mat.data[ 3 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 1 ][ 2 ] );
        auto tmp08 = _mm_set_ps( mat.data[ 1 ][ 3 ] , mat.data[ 2 ][ 3 ] , mat.data[ 1 ][ 3 ] , mat.data[ 2 ][ 3 ] );
        auto tmp09 = _mm_set_ps( mat.data[ 0 ][ 0 ] , mat.data[ 0 ][ 0 ] , mat.data[ 3 ][ 0 ] , mat.data[ 3 ][ 0 ] );
        auto tmp10 = _mm_set_ps( mat.data[ 1 ][ 0 ] , mat.data[ 2 ][ 0 ] , mat.data[ 1 ][ 0 ] , mat.data[ 2 ][ 0 ] );
        auto tmp11 = _mm_set_ps( mat.data[ 2 ][ 0 ] , mat.data[ 3 ][ 0 ] , mat.data[ 0 ][ 0 ] , mat.data[ 1 ][ 0 ] );

        auto tmpA = _mm_sub_ps( impl::compute_temp( tmp00 , tmp01 , tmp02 , tmp03 , tmp04 , tmp05 , tmp06 , tmp07 , tmp08 ) ,
                                impl::compute_temp( tmp06 , tmp01 , tmp05 , tmp03 , tmp07 , tmp02 , tmp00 , tmp04 , tmp08 ) );
        auto tmpB = _mm_sub_ps( impl::compute_temp( tmp09 , tmp01 , tmp05 , tmp10 , tmp07 , tmp02 , tmp11 , tmp04 , tmp08 ) ,
                                impl::compute_temp( tmp11 , tmp01 , tmp02 , tmp10 , tmp04 , tmp05 , tmp09 , tmp07 , tmp08 ) );
        auto tmpC = _mm_sub_ps( impl::compute_temp( tmp11 , tmp03 , tmp02 , tmp10 , tmp06 , tmp05 , tmp09 , tmp00 , tmp08 ) ,
                                impl::compute_temp( tmp09 , tmp03 , tmp05 , tmp10 , tmp00 , tmp02 , tmp11 , tmp06 , tmp08 ) );
        auto tmpD = _mm_sub_ps( impl::compute_temp( tmp09 , tmp03 , tmp07 , tmp10 , tmp00 , tmp04 , tmp11 , tmp06 , tmp01 ) ,
                                impl::compute_temp( tmp11 , tmp03 , tmp04 , tmp10 , tmp06 , tmp07 , tmp09 , tmp00 , tmp01 ) );

        auto shuffle_index = _mm_set_epi32( 3 , 1 , 2 , 0 );

        tmpA = _mm_permutevar_ps( tmpA , shuffle_index ); //[ A00, A01, A02, A03 ]
        tmpB = _mm_permutevar_ps( tmpB , shuffle_index ); //[ A10, A11, A12, A13 ]
        tmpC = _mm_permutevar_ps( tmpC , shuffle_index ); //[ A20, A21, A22, A23 ]
        tmpD = _mm_permutevar_ps( tmpD , shuffle_index ); //[ A30, A31, A32, A33 ]

        auto tmpE = _mm_set_ps( mat.data[ 3 ][ 0 ] , mat.data[ 2 ][ 0 ] , mat.data[ 1 ][ 0 ] , mat.data[ 0 ][ 0 ] );
        auto tmpF = _mm_mul_ps( tmpE , tmpA );

        auto det  = impl::compute_sum( tmpF );
        auto scale = _mm_set1_ps( 1.f / det );

        tmpA = _mm_mul_ps( tmpA , scale );
        tmpB = _mm_mul_ps( tmpB , scale );
        tmpC = _mm_mul_ps( tmpC , scale );
        tmpD = _mm_mul_ps( tmpD , scale );

        mat4x4f32 result;

        _mm_store_ps( &result.data[ 0 ][ 0 ] + 0x00 , tmpA );
        _mm_store_ps( &result.data[ 0 ][ 0 ] + 0x04 , tmpB );
        _mm_store_ps( &result.data[ 0 ][ 0 ] + 0x08 , tmpC );
        _mm_store_ps( &result.data[ 0 ][ 0 ] + 0x0C , tmpD );

        return result;
    }

    /**
     * +------------------------------------------------------------------------------------------------------------------+
     * | [     x00 ,       x01 ,       x02 ,       x03       ]       [     a00 ,       a10 ,       a20 ,       a30     ]  |
     * | [     x10 ,       x11 ,       x12 ,       x13       ]  -->  [     a01 ,       a11 ,       a21 ,       a31     ]  |
     * | [     x20 ,       x21 ,       x22 ,       x23       ]  -->  [     a02 ,       a12 ,       a22 ,       a32     ]  |
     * | [     x30 ,       x31 ,       x32 ,       x33       ]       [     a03 ,       a13 ,       a23 ,       a33     ]  |
     * +------------------------------------------------------------------------------------------------------------------+
     * | A00 = x11 * x22 * x33 + x12 * x23 * x31 + x13 * x21 * x32 - x13 * x22 * x31 - x12 * x21 * x33 - x11 * x23 * x32  |
     * | A02 = x10 * x21 * x33 + x11 * x23 * x30 + x13 * x20 * x31 - x13 * x21 * x30 - x11 * x20 * x33 - x10 * x23 * x31  |
     * | A01 = x13 * x22 * x30 + x12 * x20 * x33 + x10 * x23 * x32 - x10 * x22 * x33 - x12 * x23 * x30 - x13 * x20 * x32  |
     * | A03 = x12 * x21 * x30 + x11 * x20 * x32 + x10 * x22 * x31 - x10 * x21 * x32 - x11 * x22 * x30 - x12 * x20 * x31  |
     * | tpA   t00   t01   t02   t03   t04   t05   t06   t07   t08   t06   t01   t05   t03   t07   t02   t00   t04   t08  |
     * +------------------------------------------------------------------------------------------------------------------+
     * | A10 = x03 * x22 * x31 + x02 * x21 * x33 + x01 * x23 * x32 - x01 * x22 * x33 - x02 * x23 * x31 - x03 * x21 * x32  |
     * | A12 = x03 * x21 * x30 + x01 * x20 * x33 + x00 * x23 * x31 - x00 * x21 * x33 - x01 * x23 * x30 - x03 * x20 * x31  |
     * | A11 = x00 * x22 * x33 + x02 * x23 * x30 + x03 * x20 * x32 - x03 * x22 * x30 - x02 * x20 * x33 - x00 * x23 * x32  |
     * | A13 = x00 * x21 * x32 + x01 * x22 * x30 + x02 * x20 * x31 - x02 * x21 * x30 - x01 * x20 * x32 - x00 * x22 * x31  |
     * | tpB   t09   t01   t05   t10   t07   t02   t11   t04   t08   t11   t01   t02   t10   t04   t05   t09   t07   t08  |
     * +------------------------------------------------------------------------------------------------------------------+
     * | A20 = x01 * x12 * x33 + x02 * x13 * x31 + x03 * x11 * x32 - x03 * x12 * x31 - x02 * x11 * x33 - x01 * x13 * x32  |
     * | A22 = x00 * x11 * x33 + x01 * x13 * x30 + x03 * x10 * x31 - x03 * x11 * x30 - x01 * x10 * x33 - x00 * x13 * x31  |
     * | A21 = x03 * x12 * x30 + x02 * x10 * x33 + x00 * x13 * x32 - x00 * x12 * x33 - x02 * x13 * x30 - x03 * x10 * x32  |
     * | A23 = x02 * x11 * x30 + x01 * x10 * x32 + x00 * x12 * x31 - x00 * x11 * x32 - x01 * x12 * x30 - x02 * x10 * x31  |
     * | tpC   t11   t03   t02   t10   t06   t05   t09   t00   t08   t09   t03   t05   t10   t00   t02   t11   t06   t08  |
     * +------------------------------------------------------------------------------------------------------------------+
     * | A30 = x03 * x12 * x21 + x02 * x11 * x23 + x01 * x13 * x22 - x01 * x12 * x23 - x02 * x13 * x21 - x03 * x11 * x22  |
     * | A32 = x03 * x11 * x20 + x01 * x10 * x23 + x00 * x13 * x21 - x00 * x11 * x23 - x01 * x13 * x20 - x03 * x10 * x21  |
     * | A31 = x00 * x12 * x23 + x02 * x13 * x20 + x03 * x10 * x22 - x03 * x12 * x20 - x02 * x10 * x23 - x00 * x13 * x22  |
     * | A33 = x00 * x11 * x22 + x01 * x12 * x20 + x02 * x10 * x21 - x02 * x11 * x20 - x01 * x10 * x22 - x00 * x12 * x21  |
     * | tpD   t09   t03   t07   t10   t00   t04   t11   t06   t01   t11   t03   t04   t10   t06   t07   t09   t00   t01  |
     * +------------------------------------------------------------------------------------------------------------------+
     */
    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
    inline auto inverse( const mat4x4f64& mat ) noexcept {
        auto tmp00 = _mm256_set_pd( mat.data[ 2 ][ 1 ] , mat.data[ 3 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 1 ][ 1 ] );
        auto tmp01 = _mm256_set_pd( mat.data[ 1 ][ 2 ] , mat.data[ 2 ][ 2 ] , mat.data[ 1 ][ 2 ] , mat.data[ 2 ][ 2 ] );
        auto tmp02 = _mm256_set_pd( mat.data[ 0 ][ 3 ] , mat.data[ 0 ][ 3 ] , mat.data[ 3 ][ 3 ] , mat.data[ 3 ][ 3 ] );
        auto tmp03 = _mm256_set_pd( mat.data[ 1 ][ 1 ] , mat.data[ 2 ][ 1 ] , mat.data[ 1 ][ 1 ] , mat.data[ 2 ][ 1 ] );
        auto tmp04 = _mm256_set_pd( mat.data[ 0 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 3 ][ 2 ] , mat.data[ 3 ][ 2 ] );
        auto tmp05 = _mm256_set_pd( mat.data[ 2 ][ 3 ] , mat.data[ 3 ][ 3 ] , mat.data[ 0 ][ 3 ] , mat.data[ 1 ][ 3 ] );
        auto tmp06 = _mm256_set_pd( mat.data[ 0 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 3 ][ 1 ] , mat.data[ 3 ][ 1 ] );
        auto tmp07 = _mm256_set_pd( mat.data[ 2 ][ 2 ] , mat.data[ 3 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 1 ][ 2 ] );
        auto tmp08 = _mm256_set_pd( mat.data[ 1 ][ 3 ] , mat.data[ 2 ][ 3 ] , mat.data[ 1 ][ 3 ] , mat.data[ 2 ][ 3 ] );
        auto tmp09 = _mm256_set_pd( mat.data[ 0 ][ 0 ] , mat.data[ 0 ][ 0 ] , mat.data[ 3 ][ 0 ] , mat.data[ 3 ][ 0 ] );
        auto tmp10 = _mm256_set_pd( mat.data[ 1 ][ 0 ] , mat.data[ 2 ][ 0 ] , mat.data[ 1 ][ 0 ] , mat.data[ 2 ][ 0 ] );
        auto tmp11 = _mm256_set_pd( mat.data[ 2 ][ 0 ] , mat.data[ 3 ][ 0 ] , mat.data[ 0 ][ 0 ] , mat.data[ 1 ][ 0 ] );

        auto tmpA = _mm256_sub_pd( impl::compute_temp( tmp00 , tmp01 , tmp02 , tmp03 , tmp04 , tmp05 , tmp06 , tmp07 , tmp08 ) ,
                                   impl::compute_temp( tmp06 , tmp01 , tmp05 , tmp03 , tmp07 , tmp02 , tmp00 , tmp04 , tmp08 ) );
        auto tmpB = _mm256_sub_pd( impl::compute_temp( tmp09 , tmp01 , tmp05 , tmp10 , tmp07 , tmp02 , tmp11 , tmp04 , tmp08 ) ,
                                   impl::compute_temp( tmp11 , tmp01 , tmp02 , tmp10 , tmp04 , tmp05 , tmp09 , tmp07 , tmp08 ) );
        auto tmpC = _mm256_sub_pd( impl::compute_temp( tmp11 , tmp03 , tmp02 , tmp10 , tmp06 , tmp05 , tmp09 , tmp00 , tmp08 ) ,
                                   impl::compute_temp( tmp09 , tmp03 , tmp05 , tmp10 , tmp00 , tmp02 , tmp11 , tmp06 , tmp08 ) );
        auto tmpD = _mm256_sub_pd( impl::compute_temp( tmp09 , tmp03 , tmp07 , tmp10 , tmp00 , tmp04 , tmp11 , tmp06 , tmp01 ) ,
                                   impl::compute_temp( tmp11 , tmp03 , tmp04 , tmp10 , tmp06 , tmp07 , tmp09 , tmp00 , tmp01 ) );

        tmpA = _mm256_permute4x64_pd( tmpA , _MM_SHUFFLE( 3 , 1 , 2 , 0 ) ); //[ A00, A01, A02, A03 ]
        tmpB = _mm256_permute4x64_pd( tmpB , _MM_SHUFFLE( 3 , 1 , 2 , 0 ) ); //[ A10, A11, A12, A13 ]
        tmpC = _mm256_permute4x64_pd( tmpC , _MM_SHUFFLE( 3 , 1 , 2 , 0 ) ); //[ A20, A21, A22, A23 ]
        tmpD = _mm256_permute4x64_pd( tmpD , _MM_SHUFFLE( 3 , 1 , 2 , 0 ) ); //[ A30, A31, A32, A33 ]

        auto tmpE = _mm256_set_pd( mat.data[ 3 ][ 0 ] , mat.data[ 2 ][ 0 ] , mat.data[ 1 ][ 0 ] , mat.data[ 0 ][ 0 ] );
        auto tmpF = _mm256_mul_pd( tmpE , tmpA );

        auto det  = impl::compute_sum( tmpF );
        auto scale = _mm256_set1_pd( 1.0 / det );

        tmpA = _mm256_mul_pd( tmpA , scale );
        tmpB = _mm256_mul_pd( tmpB , scale );
        tmpC = _mm256_mul_pd( tmpC , scale );
        tmpD = _mm256_mul_pd( tmpD , scale );

        mat4x4f64 result;

        _mm256_store_pd( &result.data[ 0 ][ 0 ] + 0x00 , tmpA );
        _mm256_store_pd( &result.data[ 0 ][ 0 ] + 0x04 , tmpB );
        _mm256_store_pd( &result.data[ 0 ][ 0 ] + 0x08 , tmpC );
        _mm256_store_pd( &result.data[ 0 ][ 0 ] + 0x0C , tmpD );

        return result;
    }
}
#endif
