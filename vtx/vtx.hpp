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
 * @file      vertex.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     Vertex - Vulkan-Oriented Vertex Input Abstraction Library
 *
 * @defgroup  vtx Vertex Library
 * @brief     Template-based vertex input mapping for Vulkan image engine.
 *
 * @details
 * This library provides compile-time vertex input abstraction with arbitrary
 * dimension vectors and matrices. All includes are centralized through this
 * single header.
 *
 * @note      Define VERTEX_ENABLE_EXTENSIONS to enable SIMD AVX2 optimizations.
 */

#ifndef VERTEX_HPP
#define VERTEX_HPP
#pragma once

/**
 * @defgroup  vtx_basic Basic Types
 * @ingroup   vtx
 * @brief     Core type aliases and utility macros.
 * @{
 */

#include "vtxbasic.hpp"             ///< vi08, vi32, vf32, vf64, arithmetic_t,
                                    ///< VERTEX_FORCEINLINE, VERTEX_CONSTEXPR

/** @} 
 */ // end of vtx_basic

/**
 * @defgroup  vtx_vector Vectors
 * @ingroup   vtx
 * @brief     N-dimensional vector templates.
 * @{
 */

#include "vtxvecx.hpp"              ///< vector_template<T, N>
#include "vtxvec2.hpp"              ///< vec2 = vector_template<T, 2>
#include "vtxvec3.hpp"              ///< vec3 = vector_template<T, 3>
#include "vtxvec4.hpp"              ///< vec4 = vector_template<T, 4>

/** @} 
 */ // end of vtx_vector

/**
 * @defgroup  vtx_matrix Matrices
 * @ingroup   vtx
 * @brief     R x C dimensional matrix templates.
 * @{
 */

#include "vtxmatcxr.hpp"            ///< matrix_template<T, R, C>

#include "vtxmat2x2.hpp"            ///< mat2x2 = matrix_template<T, 2, 2>
#include "vtxmat2x3.hpp"            ///< mat2x3 = matrix_template<T, 2, 3>
#include "vtxmat2x4.hpp"            ///< mat2x4 = matrix_template<T, 2, 4>
#include "vtxmat3x2.hpp"            ///< mat3x2 = matrix_template<T, 3, 2>
#include "vtxmat3x3.hpp"            ///< mat3x3 = matrix_template<T, 3, 3>
#include "vtxmat3x4.hpp"            ///< mat3x4 = matrix_template<T, 3, 4>
#include "vtxmat4x2.hpp"            ///< mat4x2 = matrix_template<T, 4, 2>
#include "vtxmat4x3.hpp"            ///< mat4x3 = matrix_template<T, 4, 3>
#include "vtxmat4x4.hpp"            ///< mat4x4 = matrix_template<T, 4, 4>

/** @} 
 */ // end of vtx_matrix

/**
 * @defgroup  vtx_quat Quaternions
 * @ingroup   vtx
 * @brief     Quaternion template.
 * @{
 */

#include "vtxquat.hpp"              ///< quaternion template

/** @} 
 */ // end of vtx_quat

/**
 * @defgroup  vtx_util Utilities
 * @ingroup   vtx
 * @brief     String utilities, type traits, tuple containers.
 * @{
 */

#include "vtxstring.hpp"            ///< string utilities
#include "vtxtraits.hpp"            ///< type traits
#include "vtxtuple.hpp"             ///< tuple container

/** @} 
 */ // end of vtx_util

/**
 * @defgroup  vtx_ext Extensions
 * @ingroup   vtx
 * @brief     SIMD AVX2 optimizations for add/sub operations.
 * @note      Enabled only when VERTEX_ENABLE_EXTENSIONS is defined.
 * @{
 */

#ifdef VERTEX_ENABLE_EXTENSIONS
#include "vtxsimd.hpp"           ///< AVX2 SIMD optimizations
#endif

/** @} 
 */ // end of vtx_ext

#endif  // VERTEX_HPP