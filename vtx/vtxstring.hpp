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
 * @file      vtxstring.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     String formatting and output utilities for Vertex types.
 *
 * @defgroup  vtx_util Utilities
 * @ingroup   vtx
 * @brief     String formatting support for vectors and matrices.
 *
 * @details
 * This header provides:
 * - std::ostream   operator<< for vector_template
 * - std::formatter specialization for vector_template (C++20 format)
 *
 * Output format: [element0, element1, ..., elementN-1]
 *
 * Example:
 * @code
 * vec3f v{1.0f, 2.0f, 3.0f};
 * std::cout << v << std::endl;  // [1, 2, 3]
 * auto s = std::format("{}", v); // "[1, 2, 3]"
 * @endcode
 */

#ifndef VERTEX_STRING_HPP
#define VERTEX_STRING_HPP
#pragma once

#include "vtxvector.hpp"     ///< vector_template<Ty, N>
#include "vtxmatrix.hpp"     ///< matrix_template<Ty, R, C>

#include <string>     ///< std::string
#include <format>     ///< std::format, std::formatter, std::format_to
#include <utility>    ///< std::index_sequence, std::make_index_sequence
#include <ostream>    ///< std::ostream (implicitly via <string>? ensure it's included)
#include <iostream>   ///< For std::ostream definition (though <ostream> is better)
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace vtx {

    /**
     * @brief   Stream insertion operator for vector_template.
     * @tparam  Ty Element type.
     * @tparam  N Vector dimension.
     * @param   os Output stream.
     * @param   vec Vector to output.
     * @return  Reference to the output stream.
     *
     * @details
     * Delegates to std::format for consistent formatting.
     */
    template <arithmetic_t Ty , std::size_t N>
    std::ostream& operator<<( std::ostream& os , const vector_template<Ty , N>& vec ) {
        os << std::format( "{}" , vec );
        return os;
    }

    template <arithmetic_t T , size_t R , size_t C>
    VERTEX_FORCEINLINE std::ostream& operator<<( std::ostream& os , const matrix_template<T , R , C>& mat ){
        constexpr int precision = 6;
        size_t maxLen = 0;

        for ( size_t y = 0; y < R; ++y ){
            for ( size_t x = 0; x < C; ++x ){
                std::ostringstream ss {};
                if constexpr ( std::is_floating_point_v<T> ){
                    ss << std::fixed << std::setprecision( precision ) << mat.data [ x ][ y ];
                } else {
                    ss << mat.data [ x ][ y ];
                }
                maxLen = std::max( { maxLen , ss.str( ).length( ) } );
            }
        }

        for ( size_t y = 0; y < R; ++y ){
            os << "[ ";
            for ( size_t x = 0; x < C; ++x ){
                std::ostringstream ss;
                if constexpr ( std::is_floating_point_v<T> ){
                    ss << std::fixed << std::setprecision( precision ) << mat.data [ x ][ y ];
                } else {
                    ss << mat.data [ x ][ y ];
                }
                os << std::setw( static_cast< int >( maxLen ) ) << ss.str( );
                if ( x + 1 < C ) os << ' ';
            }
            os << " ]\n";
        }
        return os;
    }

} // namespace vtx

/*
 *+-----------------------------------------------------------------------------+
 *| std::formatter specialization for vtx::vector_template                      |
 *+-----------------------------------------------------------------------------+
 */

template <vtx::arithmetic_t Ty, std::size_t N>
struct std::formatter<vtx::vector_template<Ty, N>> {
    /**
     * @brief   Parses format specification (currently no custom specifiers).
     * @param   ctx Parse context.
     * @return  Iterator to the end of the format string.
     */
    constexpr auto parse( format_parse_context& ctx ) {
        return ctx.begin( );
    }

    /**
     * @brief   Formats a vector_template as a string.
     * @param   vec Vector to format.
     * @param   ctx Format context.
     * @return  Iterator to the end of the output.
     *
     * @details
     * Output format: [element0, element1, ..., elementN-1]
     */
    auto format( const vtx::vector_template<Ty, N>& vec, format_context& ctx ) const {
        auto out = format_to( ctx.out( ) , "[" );
        [ & ]<std::size_t... I>( std::index_sequence<I...> ){
            ( ( out = format_to( out , I == 0 ? "{}" : ", {}" , vec.data [ I ] ) ) , ... );
        }( std::make_index_sequence<N>( ) );
        return format_to( out , "]" );
    }
};

#endif // VERTEX_STRING_HPP