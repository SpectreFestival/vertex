
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

#ifndef VERTEX_SIMD3_HPP
#define VERTEX_SIMD3_HPP
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
#include <xmmintrin.h>
#include <iostream>
#include <ostream>

#include "vtxtuple.hpp"

namespace vtx {
    /**
     * @brief       Internal SIMD implementation tmpZails.
     * @namespace   impl
     */
    namespace impl {
        VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_MFMA_NAME )
        inline auto compute_fma(
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
        inline auto compute_fma(
            const __m128i& v1 , const __m128i& v2 , const __m128i& v3 ,
            const __m128i& v4 , const __m128i& v5 , const __m128i& v6 ,
            const __m128i& v7 , const __m128i& v8 , const __m128i& v9
        ) noexcept {
            return
            _mm_add_epi32(
            _mm_add_epi32(
            _mm_mullo_epi32( _mm_mullo_epi32( v1 , v2 ) , v3 ) ,
            _mm_mullo_epi32( _mm_mullo_epi32( v4 , v5 ) , v6 ) ) ,
            _mm_mullo_epi32( _mm_mullo_epi32( v7 , v8 ) , v9 ) ) ;
        }

        VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_MFMA_NAME )
        inline auto compute_fma(
            const __m256d& v1 , const __m256d& v2 , const __m256d& v3 ,
            const __m256d& v4 , const __m256d& v5 , const __m256d& v6 ,
            const __m256d& v7 , const __m256d& v8 , const __m256d& v9
        ) noexcept {
            return
                _mm256_fmadd_pd( _mm256_mul_pd( v7 , v8 ) , v9 ,
                _mm256_fmadd_pd( _mm256_mul_pd( v4 , v5 ) , v6 ,
                  _mm256_mul_pd( _mm256_mul_pd( v1 , v2 ) , v3 ) ) );
        }
    }

    /**
     * A = [[a11, a12, a13, a14],
     *      [a21, a22, a23, a24],
     *      [a31, a32, a33, a34],
     *      [a41, a42, a43, a44]]
     *+-----------------------------------------------------------------------------------------------------------------------+
     *|   a11 * ( a22 * a33 * a44 + a23 * a34 * a42 + a24 * a32 * a43 - a24 * a33 * a42 - a23 * a32 * a44 - a22 * a34 * a43 ) |
     *| - a12 * ( a21 * a33 * a44 + a23 * a34 * a41 + a24 * a31 * a43 - a24 * a33 * a41 - a23 * a31 * a44 - a21 * a34 * a43 ) |
     *| + a13 * ( a21 * a32 * a44 + a22 * a34 * a41 + a24 * a31 * a42 - a24 * a32 * a41 - a22 * a31 * a44 - a21 * a34 * a42 ) |
     *| - a14 * ( a21 * a32 * a43 + a22 * a33 * a41 + a23 * a31 * a42 - a23 * a32 * a41 - a22 * a31 * a43 - a21 * a33 * a42 ) |
     *|   tmpJ    tmpA  tmpB  tmpC  tmpD  tmpE  tmpF  tmpG  tmpH  tmpI  tmpG  tmpB  tmpF  tmpD  tmpH  tmpC  tmpA  tmpE  tmpI  |
     *+-----------------------------------------------------------------------------------------------------------------------+
     */
    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_SSE4_NAME )
    inline auto determinant( const mat4x4f32& mat ) noexcept {
        auto tmpA = _mm_set_ps( mat.data[ 0 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 1 ][ 1 ] );
        auto tmpB = _mm_set_ps( mat.data[ 1 ][ 2 ] , mat.data[ 1 ][ 2 ] , mat.data[ 2 ][ 2 ] , mat.data[ 2 ][ 2 ] );
        auto tmpC = _mm_set_ps( mat.data[ 2 ][ 3 ] , mat.data[ 3 ][ 3 ] , mat.data[ 3 ][ 3 ] , mat.data[ 3 ][ 3 ] );
        auto tmpD = _mm_set_ps( mat.data[ 1 ][ 1 ] , mat.data[ 1 ][ 1 ] , mat.data[ 2 ][ 1 ] , mat.data[ 2 ][ 1 ] );
        auto tmpE = _mm_set_ps( mat.data[ 2 ][ 2 ] , mat.data[ 3 ][ 2 ] , mat.data[ 3 ][ 2 ] , mat.data[ 3 ][ 2 ] );
        auto tmpF = _mm_set_ps( mat.data[ 0 ][ 3 ] , mat.data[ 0 ][ 3 ] , mat.data[ 0 ][ 3 ] , mat.data[ 1 ][ 3 ] );
        auto tmpG = _mm_set_ps( mat.data[ 2 ][ 1 ] , mat.data[ 3 ][ 1 ] , mat.data[ 3 ][ 1 ] , mat.data[ 3 ][ 1 ] );
        auto tmpH = _mm_set_ps( mat.data[ 0 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 1 ][ 2 ] );
        auto tmpI = _mm_set_ps( mat.data[ 1 ][ 3 ] , mat.data[ 1 ][ 3 ] , mat.data[ 2 ][ 3 ] , mat.data[ 2 ][ 3 ] );
        auto tmpJ = _mm_set_ps( mat.data[ 3 ][ 0 ] , mat.data[ 2 ][ 0 ] , mat.data[ 1 ][ 0 ] , mat.data[ 0 ][ 0 ] );
        auto tmpK = _mm_set_ps( -1.f , +1.f , -1.f , +1.f );

        auto tmpL = _mm_mul_ps(
                    _mm_mul_ps( tmpJ , tmpK ),
                    _mm_sub_ps( impl::compute_fma( tmpA , tmpB , tmpC , tmpD , tmpE , tmpF , tmpG , tmpH , tmpI ) ,
                                impl::compute_fma( tmpG , tmpB , tmpF , tmpD , tmpH , tmpC , tmpA , tmpE , tmpI ) ) );
        tmpL = _mm_hadd_ps( tmpL , tmpL );
        tmpL = _mm_hadd_ps( tmpL , tmpL );
        return _mm_cvtss_f32( tmpL );
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
    inline auto determinant( const mat4x4f64& mat ) noexcept {
        auto tmpA = _mm256_set_pd( mat.data[ 0 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 1 ][ 1 ] );
        auto tmpB = _mm256_set_pd( mat.data[ 1 ][ 2 ] , mat.data[ 1 ][ 2 ] , mat.data[ 2 ][ 2 ] , mat.data[ 2 ][ 2 ] );
        auto tmpC = _mm256_set_pd( mat.data[ 2 ][ 3 ] , mat.data[ 3 ][ 3 ] , mat.data[ 3 ][ 3 ] , mat.data[ 3 ][ 3 ] );
        auto tmpD = _mm256_set_pd( mat.data[ 1 ][ 1 ] , mat.data[ 1 ][ 1 ] , mat.data[ 2 ][ 1 ] , mat.data[ 2 ][ 1 ] );
        auto tmpE = _mm256_set_pd( mat.data[ 2 ][ 2 ] , mat.data[ 3 ][ 2 ] , mat.data[ 3 ][ 2 ] , mat.data[ 3 ][ 2 ] );
        auto tmpF = _mm256_set_pd( mat.data[ 0 ][ 3 ] , mat.data[ 0 ][ 3 ] , mat.data[ 0 ][ 3 ] , mat.data[ 1 ][ 3 ] );
        auto tmpG = _mm256_set_pd( mat.data[ 2 ][ 1 ] , mat.data[ 3 ][ 1 ] , mat.data[ 3 ][ 1 ] , mat.data[ 3 ][ 1 ] );
        auto tmpH = _mm256_set_pd( mat.data[ 0 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 1 ][ 2 ] );
        auto tmpI = _mm256_set_pd( mat.data[ 1 ][ 3 ] , mat.data[ 1 ][ 3 ] , mat.data[ 2 ][ 3 ] , mat.data[ 2 ][ 3 ] );
        auto tmpJ = _mm256_set_pd( mat.data[ 3 ][ 0 ] , mat.data[ 2 ][ 0 ] , mat.data[ 1 ][ 0 ] , mat.data[ 0 ][ 0 ] );
        auto tmpK = _mm256_set_pd( -1.0 , +1.0 , -1.0 , +1.0 );

        auto tmpL = _mm256_mul_pd(
                    _mm256_mul_pd( tmpJ , tmpK ),
                    _mm256_sub_pd( impl::compute_fma( tmpA , tmpB , tmpC , tmpD , tmpE , tmpF , tmpG , tmpH , tmpI ) ,
                                   impl::compute_fma( tmpG , tmpB , tmpF , tmpD , tmpH , tmpC , tmpA , tmpE , tmpI ) ) );

        auto tmpM = _mm256_hadd_pd( tmpL , tmpL );
        auto tmpN = _mm256_castpd256_pd128( tmpM );
        auto tmpO = _mm256_extractf128_pd( tmpM , 1 );
        return _mm_cvtsd_f64( _mm_add_sd( tmpN , tmpO ) );
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_SSE4_NAME )
    inline auto determinant( const mat4x4i32& mat ) noexcept {
        auto tmpA = _mm_set_epi32( mat.data[ 0 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 1 ][ 1 ] );
        auto tmpB = _mm_set_epi32( mat.data[ 1 ][ 2 ] , mat.data[ 1 ][ 2 ] , mat.data[ 2 ][ 2 ] , mat.data[ 2 ][ 2 ] );
        auto tmpC = _mm_set_epi32( mat.data[ 2 ][ 3 ] , mat.data[ 3 ][ 3 ] , mat.data[ 3 ][ 3 ] , mat.data[ 3 ][ 3 ] );
        auto tmpD = _mm_set_epi32( mat.data[ 1 ][ 1 ] , mat.data[ 1 ][ 1 ] , mat.data[ 2 ][ 1 ] , mat.data[ 2 ][ 1 ] );
        auto tmpE = _mm_set_epi32( mat.data[ 2 ][ 2 ] , mat.data[ 3 ][ 2 ] , mat.data[ 3 ][ 2 ] , mat.data[ 3 ][ 2 ] );
        auto tmpF = _mm_set_epi32( mat.data[ 0 ][ 3 ] , mat.data[ 0 ][ 3 ] , mat.data[ 0 ][ 3 ] , mat.data[ 1 ][ 3 ] );
        auto tmpG = _mm_set_epi32( mat.data[ 2 ][ 1 ] , mat.data[ 3 ][ 1 ] , mat.data[ 3 ][ 1 ] , mat.data[ 3 ][ 1 ] );
        auto tmpH = _mm_set_epi32( mat.data[ 0 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 1 ][ 2 ] );
        auto tmpI = _mm_set_epi32( mat.data[ 1 ][ 3 ] , mat.data[ 1 ][ 3 ] , mat.data[ 2 ][ 3 ] , mat.data[ 2 ][ 3 ] );
        auto tmpJ = _mm_set_epi32( mat.data[ 3 ][ 0 ] , mat.data[ 2 ][ 0 ] , mat.data[ 1 ][ 0 ] , mat.data[ 0 ][ 0 ] );
        auto tmpK = _mm_set_epi32( -1 , +1 , -1 , +1 );

        auto tmpL = _mm_mullo_epi32(
                    _mm_mullo_epi32( tmpJ , tmpK ),
                    _mm_sub_epi32( impl::compute_fma( tmpA , tmpB , tmpC , tmpD , tmpE , tmpF , tmpG , tmpH , tmpI ) ,
                                   impl::compute_fma( tmpG , tmpB , tmpF , tmpD , tmpH , tmpC , tmpA , tmpE , tmpI ) ) );
        tmpL = _mm_hadd_epi32( tmpL , tmpL );
        tmpL = _mm_hadd_epi32( tmpL , tmpL );
        return _mm_cvtsi128_si32( tmpL );
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
    inline auto determinant( const mat3x3f32& mat ) noexcept {
        auto tmpA = _mm256_setr_ps( mat.data[ 0 ][ 0 ] , mat.data[ 1 ][ 0 ] , mat.data[ 2 ][ 0 ] , mat.data[ 2 ][ 0 ] , mat.data[ 1 ][ 0 ] , mat.data[ 0 ][ 0 ] , 0.f , 0.f );
        auto tmpB = _mm256_setr_ps( mat.data[ 1 ][ 1 ] , mat.data[ 2 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 1 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 2 ][ 1 ] , 0.f , 0.f );
        auto tmpC = _mm256_setr_ps( mat.data[ 2 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 1 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 2 ][ 2 ] , mat.data[ 1 ][ 2 ] , 0.f , 0.f );
        auto tmpD = _mm256_setr_ps( +1.f , +1.f , +1.f , -1.f , -1.f , -1.f , 0.f , 0.f );
        auto tmpE = _mm256_mul_ps(
                    _mm256_mul_ps( tmpA , tmpB ) ,
                    _mm256_mul_ps( tmpC , tmpD ) );
        auto tmpF = _mm256_extractf128_ps( tmpE , 1);
        auto tmpG = _mm256_castps256_ps128( tmpE );
        auto tmpH = _mm_add_ps( tmpG , tmpF );
        tmpH = _mm_hadd_ps( tmpH , tmpH );
        tmpH = _mm_hadd_ps( tmpH , tmpH );
        return _mm_cvtss_f32( tmpH );
    }

    VERTEX_FUNCTION_OPTIMIZATION( VERTEX_OPT_AVX2_NAME )
    inline auto determinant( const mat3x3i32& mat ) noexcept {
        auto tmpA = _mm256_setr_epi32( mat.data[ 0 ][ 0 ] , mat.data[ 1 ][ 0 ] , mat.data[ 2 ][ 0 ] , mat.data[ 2 ][ 0 ] , mat.data[ 1 ][ 0 ] , mat.data[ 0 ][ 0 ] , 0 , 0 );
        auto tmpB = _mm256_setr_epi32( mat.data[ 1 ][ 1 ] , mat.data[ 2 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 1 ][ 1 ] , mat.data[ 0 ][ 1 ] , mat.data[ 2 ][ 1 ] , 0 , 0 );
        auto tmpC = _mm256_setr_epi32( mat.data[ 2 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 1 ][ 2 ] , mat.data[ 0 ][ 2 ] , mat.data[ 2 ][ 2 ] , mat.data[ 1 ][ 2 ] , 0 , 0 );
        auto tmpD = _mm256_setr_epi32( +1 , +1 , +1 , -1 , -1 , -1 , 0 , 0 );
        auto tmpE = _mm256_mullo_epi32(
                    _mm256_mullo_epi32( tmpA , tmpB ) ,
                    _mm256_mullo_epi32( tmpC , tmpD ) );
        auto tmpF = _mm_add_epi32(
              _mm256_extracti128_si256( tmpE , 1 ) ,
              _mm256_castsi256_si128( tmpE )
          );
        auto tmpG = _mm_add_epi32( tmpF, _mm_unpackhi_epi64( tmpF, tmpF ) );
        tmpG = _mm_add_epi32( tmpG, _mm_shuffle_epi32( tmpG, 0x01 ) );
        return _mm_cvtsi128_si32( tmpG );
    }
}
#endif
