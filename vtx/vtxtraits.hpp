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
* @file      vtxtraits.hpp
* @author    SpectreFestival
* @license   MIT
* @brief     Vertex attribute traits for Vulkan vertex input descriptions.
*
* @defgroup  vtx_util Utilities
* @ingroup   vtx
* @brief     Type traits for mapping Vertex types to Vulkan vertex attributes.
*
* @details
* This header provides vertex_traits<T> which maps Vertex vector/matrix types
* to Vulkan vertex input attribute properties:
* - count:   Number of elements (for matrices, number of columns)
* - format:  Vulkan VkFormat enum value (numeric)
* - offset_step:  Byte offset between consecutive rows/columns
*
* The format values correspond to Vulkan VkFormat enums:
* - VK_FORMAT_R32_SFLOAT = 100, VK_FORMAT_R32G32_SFLOAT = 103, etc.
*
* Usage:
* @code
* using Traits = vertex_traits<vec3f>;
* static_assert(Traits::count == 1);
* static_assert(Traits::format == 106); // VK_FORMAT_R32G32B32_SFLOAT
* static_assert(Traits::offset_step == 12);
* @endcode
*/

#ifndef VERTEX_TRAITS_HPP
#define VERTEX_TRAITS_HPP
#pragma once

#include "vtxbasic.hpp"      ///< arithmetic_t, vi32, vf32, etc.

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

#include <cstdint>           ///< uint32_t

/**
 * @namespace vtx
 * @brief Root namespace for Vertex library.
 */

namespace vtx{
	/**
	 * @brief	Primary template for vertex traits (undefined).
	 * @tparam	Ty Type to query traits for.
	 *
	 * @details
	 * Specializations are provided for all built-in scalar, vector, and matrix types.
	 */
	template <typename Ty>
	struct vertex_traits;

	template<>
	struct vertex_traits<vf64>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 112;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<vf32>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 100;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<vi32>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 99;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<vu32>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 98;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<vi16>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 75;
		static constexpr uint32_t offset_step = 2;
	};

	template<>
	struct vertex_traits<vu16>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 74;
		static constexpr uint32_t offset_step = 2;
	};

	template<>
	struct vertex_traits<vi08>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 14;
		static constexpr uint32_t offset_step = 1;
	};

	template<>
	struct vertex_traits<vu08>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 13;
		static constexpr uint32_t offset_step = 1;
	};

	template<>
	struct vertex_traits<vec2f64>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 115;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<vec2f32>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 103;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<vec2i32>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 102;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<vec2u32>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 101;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<vec2i16>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 82;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<vec2u16>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 81;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<vec2i08>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 21;
		static constexpr uint32_t offset_step = 2;
	};

	template<>
	struct vertex_traits<vec2u08>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 20;
		static constexpr uint32_t offset_step = 2;
	};

	template<>
	struct vertex_traits<vec3f64>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 118;
		static constexpr uint32_t offset_step = 24;
	};

	template<>
	struct vertex_traits<vec3f32>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 106;
		static constexpr uint32_t offset_step = 12;
	};

	template<>
	struct vertex_traits<vec3i32>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 105;
		static constexpr uint32_t offset_step = 12;
	};

	template<>
	struct vertex_traits<vec3u32>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 104;
		static constexpr uint32_t offset_step = 12;
	};

	template<>
	struct vertex_traits<vec3i16>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 89;
		static constexpr uint32_t offset_step = 6;
	};

	template<>
	struct vertex_traits<vec3u16>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 88;
		static constexpr uint32_t offset_step = 6;
	};

	template<>
	struct vertex_traits<vec3i08>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 28;
		static constexpr uint32_t offset_step = 3;
	};

	template<>
	struct vertex_traits<vec3u08>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 27;
		static constexpr uint32_t offset_step = 3;
	};

	template<>
	struct vertex_traits<vec4f64>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 121;
		static constexpr uint32_t offset_step = 32;
	};

	template<>
	struct vertex_traits<vec4f32>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 109;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<vec4i32>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 108;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<vec4u32>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 107;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<vec4i16>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 96;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<vec4u16>{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 95;
		static constexpr uint32_t offset_step = 8;
	};

	template<> struct vertex_traits<vec4i08 >{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 42;
		static constexpr uint32_t offset_step = 4;
	};

	template<> struct vertex_traits<vec4u08 >{
		static constexpr uint32_t count = 1;
		static constexpr uint32_t format = 41;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<mat2x2f64>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 115;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<mat2x3f64>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 115;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<mat2x4f64>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 115;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<mat2x2f32>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 103;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat2x3f32>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 103;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat2x4f32>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 103;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat2x2i32>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 102;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat2x3i32>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 102;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat2x4i32>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 102;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat2x2u32>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 101;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat2x3u32>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 101;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat2x4u32>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 101;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat2x2i16>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 82;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<mat2x3i16>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 82;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<mat2x4i16>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 82;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<mat2x2u16>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 81;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<mat2x3u16>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 81;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<mat2x4u16>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 81;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<mat2x2i08>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 21;
		static constexpr uint32_t offset_step = 2;
	};

	template<>
	struct vertex_traits<mat2x3i08>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 21;
		static constexpr uint32_t offset_step = 2;
	};

	template<>
	struct vertex_traits<mat2x4i08>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 21;
		static constexpr uint32_t offset_step = 2;
	};

	template<>
	struct vertex_traits<mat2x2u08>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 20;
		static constexpr uint32_t offset_step = 2;
	};

	template<>
	struct vertex_traits<mat2x3u08>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 20;
		static constexpr uint32_t offset_step = 2;
	};

	template<>
	struct vertex_traits<mat2x4u08>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 20;
		static constexpr uint32_t offset_step = 2;
	};

	template<>
	struct vertex_traits<mat3x2f64>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 118;
		static constexpr uint32_t offset_step = 24;
	};

	template<>
	struct vertex_traits<mat3x3f64>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 118;
		static constexpr uint32_t offset_step = 24;
	};

	template<>
	struct vertex_traits<mat3x4f64>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 118;
		static constexpr uint32_t offset_step = 24;
	};

	template<>
	struct vertex_traits<mat3x2f32>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 106;
		static constexpr uint32_t offset_step = 12;
	};

	template<>
	struct vertex_traits<mat3x3f32>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 106;
		static constexpr uint32_t offset_step = 12;
	};

	template<>
	struct vertex_traits<mat3x4f32>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 106;
		static constexpr uint32_t offset_step = 12;
	};

	template<>
	struct vertex_traits<mat3x2i32>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 105;
		static constexpr uint32_t offset_step = 12;
	};

	template<>
	struct vertex_traits<mat3x3i32>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 105;
		static constexpr uint32_t offset_step = 12;
	};

	template<>
	struct vertex_traits<mat3x4i32>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 105;
		static constexpr uint32_t offset_step = 12;
	};

	template<>
	struct vertex_traits<mat3x2u32>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 104;
		static constexpr uint32_t offset_step = 12;
	};

	template<>
	struct vertex_traits<mat3x3u32>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 104;
		static constexpr uint32_t offset_step = 12;
	};

	template<>
	struct vertex_traits<mat3x4u32>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 104;
		static constexpr uint32_t offset_step = 12;
	};

	template<>
	struct vertex_traits<mat3x2i16>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 89;
		static constexpr uint32_t offset_step = 6;
	};

	template<>
	struct vertex_traits<mat3x3i16>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 89;
		static constexpr uint32_t offset_step = 6;
	};

	template<>
	struct vertex_traits<mat3x4i16>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 89;
		static constexpr uint32_t offset_step = 6;
	};

	template<>
	struct vertex_traits<mat3x2u16>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 88;
		static constexpr uint32_t offset_step = 6;
	};

	template<>
	struct vertex_traits<mat3x3u16>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 88;
		static constexpr uint32_t offset_step = 6;
	};

	template<>
	struct vertex_traits<mat3x4u16>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 88;
		static constexpr uint32_t offset_step = 6;
	};

	template<>
	struct vertex_traits<mat3x2i08>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 28;
		static constexpr uint32_t offset_step = 3;
	};

	template<>
	struct vertex_traits<mat3x3i08>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 28;
		static constexpr uint32_t offset_step = 3;
	};

	template<>
	struct vertex_traits<mat3x4i08>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 28;
		static constexpr uint32_t offset_step = 3;
	};

	template<>
	struct vertex_traits<mat3x2u08>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 27;
		static constexpr uint32_t offset_step = 3;
	};

	template<>
	struct vertex_traits<mat3x3u08>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 27;
		static constexpr uint32_t offset_step = 3;
	};

	template<>
	struct vertex_traits<mat3x4u08>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 27;
		static constexpr uint32_t offset_step = 3;
	};

	template<>
	struct vertex_traits<mat4x2f64>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 121;
		static constexpr uint32_t offset_step = 32;
	};

	template<>
	struct vertex_traits<mat4x3f64>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 121;
		static constexpr uint32_t offset_step = 32;
	};

	template<>
	struct vertex_traits<mat4x4f64>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 121;
		static constexpr uint32_t offset_step = 32;
	};

	template<>
	struct vertex_traits<mat4x2f32>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 109;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<mat4x3f32>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 109;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<mat4x4f32>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 109;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<mat4x2i32>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 108;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<mat4x3i32>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 108;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<mat4x4i32>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 108;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<mat4x2u32>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 107;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<mat4x3u32>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 107;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<mat4x4u32>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 107;
		static constexpr uint32_t offset_step = 16;
	};

	template<>
	struct vertex_traits<mat4x2i16>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 96;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat4x3i16>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 96;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat4x4i16>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 96;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat4x2u16>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 95;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat4x3u16>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 95;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat4x4u16>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 95;
		static constexpr uint32_t offset_step = 8;
	};

	template<>
	struct vertex_traits<mat4x2i08>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 42;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<mat4x3i08>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 42;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<mat4x4i08>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 42;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<mat4x2u08>{
		static constexpr uint32_t count = 2;
		static constexpr uint32_t format = 41;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<mat4x3u08>{
		static constexpr uint32_t count = 3;
		static constexpr uint32_t format = 41;
		static constexpr uint32_t offset_step = 4;
	};

	template<>
	struct vertex_traits<mat4x4u08>{
		static constexpr uint32_t count = 4;
		static constexpr uint32_t format = 41;
		static constexpr uint32_t offset_step = 4;
	};
}

#endif //VERTEX_TRAITS_HPP
