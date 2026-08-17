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
 * @file      vertex.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     Vertex - Vulkan-Oriented Vertex Input Abstraction Library
 *
 * @defgroup  vtx Vertex Library
 * @brief     Template-based vertex input mapping for Vulkan image engine.
 */
#ifndef VERTEX_VERTEX_HPP
#define VERTEX_VERTEX_HPP
#pragma once

#include "vtxbasic.hpp"

#include "vtxvector.hpp"     ///< vector_template<Ty , N>
#include "vtxvector2.hpp"    ///< vector_template<Ty , 2>
#include "vtxvector3.hpp"    ///< vector_template<Ty , 3>
#include "vtxvector4.hpp"    ///< vector_template<Ty , 4>

#include "vtxmatrix.hpp"     ///< matrix_template<Ty , R , C>
#include "vtxmatrix2x2.hpp"  ///< matrix_template<Ty , 2 , 2>
#include "vtxmatrix2x3.hpp"  ///< matrix_template<Ty , 2 , 3>
#include "vtxmatrix2x4.hpp"  ///< matrix_template<Ty , 2 , 4>
#include "vtxmatrix3x2.hpp"  ///< matrix_template<Ty , 3 , 2>
#include "vtxmatrix3x3.hpp"  ///< matrix_template<Ty , 3 , 3>
#include "vtxmatrix3x4.hpp"  ///< matrix_template<Ty , 3 , 4>
#include "vtxmatrix4x2.hpp"  ///< matrix_template<Ty , 4 , 2>
#include "vtxmatrix4x3.hpp"  ///< matrix_template<Ty , 4 , 3>
#include "vtxmatrix4x4.hpp"  ///< matrix_template<Ty , 4 , 4>

#include "vtxgenerator.hpp"
#include "vtxquat.hpp"

#ifndef VERTEX_DISABLE_SIMD
#include "vtxsimd1.hpp"
#include "vtxsimd2.hpp"
#include "vtxsimd3.hpp"
#endif

#include "vtxtraits.hpp"
#include "vtxtuple.hpp"
#include "vtxstring.hpp"

#endif