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

#ifndef VERTEX_VERTEX_HPP_
#define VERTEX_VERTEX_HPP_
#pragma once

// -----------------------------------------------------------------------------
// Vertex - A Vulkan-Oriented Vertex Input Abstraction Library
// -----------------------------------------------------------------------------
// Author:  SpectreFestival
// Project: Vulkan-Based Image Engine
// Purpose: Template-based vertex input mapping with compile-time index
//          generation, supporting arbitrary dimension vectors/matrices
//          beyond 4x4 / vec4.
//
// Key Features:
//   - "vertex" tuple-like container for composable vertex input types
//   - Built-in offset and size computation
//   - vertex_layout type convertible to Vulkan VkVertexInput... structs
//   - SIMD AVX2 optimization for add/sub operations
//   - Unified vector abstraction for both vectors and matrices
//   - Compile-time loop unrolling via index generators
// -----------------------------------------------------------------------------

#include "vtx/vtx.hpp"

namespace vtx {
}  // namespace vertex

#endif  // !VERTEX_VERTEX_HPP_