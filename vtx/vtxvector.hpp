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
 * @file      vtxvector.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     N-dimensional vector template with compile-time index generation.
 *
 * @defgroup  vtx_vector Vectors
 * @ingroup   vtx
 * @brief     Generic vector template for arbitrary dimensions.
 *
 * @details
 * This header provides:
 * - vector_template<T, N> - N-dimensional vector with random-access iterator
 * - Element-wise arithmetic operators (+, -, *, /)
 * - Compound assignment operators (+=, -=, *=, /=)
 * - Comparison operators (==, !=, <, <=, >, >=) returning vector_template<bool, N>
 * - Logical operators (&&, ||) for per-element boolean operations
 * - dot()          - dot product
 * - cross()        - 2D scalar cross / 3D vector cross
 * - length()       - Euclidean norm
 * - normalize()    - unit vector
 * - distance()     - Euclidean distance
 * - angle()        - angle between two vectors
 * - rotate()       - 2D/3D/4D rotation
 * - reflect()      - reflection vector
 * - refract()      - refraction vector
 * - face_forward() - orientation correction
 * - fma()          - fused multiply-add
 * - vector_cast()  - type conversion and construction
 *
 * @note  All operations are compile-time unrolled via index_sequence
 *        and expanded using fold expressions (C++17).
 */
#ifndef VERTEX_VECTORX_HPP
#define VERTEX_VECTORX_HPP
#pragma once

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <algorithm>
#include <utility>

#include "vtxbasic.hpp"

/**
 * @namespace   vtx
 * @brief       Root namespace for Vertex library.
 */
namespace vtx {
    /**
     * @brief   N-dimensional vector template.
     * @tparam  Ty Arithmetic type (integral or floating-point).
     * @tparam  N Number of dimensions.
     */
    template <arithmetic_t Ty, std::size_t N>
    struct vector_template {
        using iterator_category         = std::random_access_iterator_tag;
        using value_type                = Ty;
        using difference_type           = std::ptrdiff_t;
        using pointer                   = Ty*;
        using reference                 = Ty&;
        using iterator                  = Ty*;
        using const_iterator            = const Ty*;
        using reverse_iterator          = std::reverse_iterator<iterator>;
        using const_reverse_iterator    = std::reverse_iterator<const_iterator>;

        /**
         * @brief   Default constructor - zero-initializes all elements.
         */
        constexpr vector_template( ) noexcept = default;

        /**
         * @brief   Returns an iterator to the first element.
         * @return  Iterator to the beginning of the vector.
         */
        [[nodiscard]] constexpr iterator begin( ) noexcept {
            return data;
        }

        /**
         * @brief   Returns an iterator to the element following the last element.
         * @return  Iterator to the end of the vector.
         */
        [[nodiscard]] constexpr iterator end( ) noexcept {
            return data + N;
        }

        /**
         * @brief   Returns a const iterator to the first element.
         * @return  Const iterator to the beginning of the vector.
         */
        [[nodiscard]] constexpr const_iterator begin( ) const noexcept {
            return data;
        }

        /**
         * @brief   Returns a const iterator to the element following the last element.
         * @return  Const iterator to the end of the vector.
         */
        [[nodiscard]] constexpr const_iterator end( ) const noexcept {
            return data + N;
        }

        /**
         * @brief   Returns a const iterator to the first element.
         * @return  Const iterator to the beginning of the vector.
         */
        [[nodiscard]] constexpr const_iterator cbegin( ) const noexcept {
            return data;
        }

        /**
         * @brief   Returns a const iterator to the element following the last element.
         * @return  Const iterator to the end of the vector.
         */
        [[nodiscard]] constexpr const_iterator cend( ) const noexcept {
            return data + N;
        }

        /**
         * @brief   Returns a reverse iterator to the first element of the reversed vector.
         * @return  Reverse iterator to the beginning of the reversed vector.
         */
        [[nodiscard]] constexpr reverse_iterator rbegin( ) noexcept {
            return std::reverse_iterator( end( ) );
        }

        /**
         * @brief   Returns a reverse iterator to the element following the last element of the reversed vector.
         * @return  Reverse iterator to the end of the reversed vector.
         */
        [[nodiscard]] constexpr reverse_iterator rend( ) noexcept {
            return std::reverse_iterator( begin( ) );
        }

        /**
         * @brief   Returns a const reverse iterator to the first element of the reversed vector.
         * @return  Const reverse iterator to the beginning of the reversed vector.
         */
        [[nodiscard]] constexpr const_reverse_iterator rbegin( ) const noexcept {
            return std::reverse_iterator( end( ) );
        }

        /**
         * @brief   Returns a const reverse iterator to the element following the last element of the reversed vector.
         * @return  Const reverse iterator to the end of the reversed vector.
         */
        [[nodiscard]] constexpr const_reverse_iterator rend( ) const noexcept {
            return std::reverse_iterator( begin( ) );
        }

        /**
         * @brief   Returns a const reverse iterator to the first element of the reversed vector.
         * @return  Const reverse iterator to the beginning of the reversed vector.
         */
        [[nodiscard]] constexpr const_reverse_iterator crbegin( ) const noexcept {
            return std::reverse_iterator( cend( ) );
        }

        /**
         * @brief   Returns a const reverse iterator to the element following the last element of the reversed vector.
         * @return  Const reverse iterator to the end of the reversed vector.
         */
        [[nodiscard]] constexpr const_reverse_iterator crend( ) const noexcept {
            return std::reverse_iterator( cbegin( ) );
        }

        /**
         * @brief   Constructs vector from variadic arguments.
         * @tparam  Args Variadic argument types.
         * @param   args Element values.
         */
        template <typename... Args>
        explicit constexpr vector_template( Args&&... args ) noexcept
            : data{ static_cast<Ty>( args )... }{}

        /**
         * @brief   Constructs vector from an initializer list.
         * @param   vec Initializer list of element values.
         */
        constexpr vector_template( const std::initializer_list<Ty>& vec ) noexcept {
            [ & ]<size_t... I>( std::index_sequence<I...> ) {
                ( ( data [ I ] = vec.begin( ) [ I ] ) , ... );
            }( std::make_index_sequence<N>( ) );
        }

        /**
         * @brief   Accesses element at specified index (unchecked).
         * @param   idx Index of the element.
         * @return  Reference to the element.
         */
        [[nodiscard]] constexpr Ty& operator[]( const std::size_t idx ) noexcept {
            return data [ idx ];
        }

        /**
         * @brief   Accesses element at specified index (unchecked, const).
         * @param   idx Index of the element.
         * @return  Const reference to the element.
         */
        [[nodiscard]] constexpr const Ty& operator[]( const std::size_t idx ) const noexcept {
            return data [ idx ];
        }

        /**
         * @brief   Converts vector to another arithmetic type.
         * @tparam  U Target arithmetic type.
         * @return  vector_template<U, N> with each element casted to type U.
         */
        template <arithmetic_t U>
        explicit operator vector_template<U , N>( ) noexcept {
            return [ & ]<size_t... I>( std::index_sequence<I...> ) {
                return vector_template<U , N>( static_cast< U >( data [ I ] )... );
            }( std::make_index_sequence<N>( ) );
        }

        Ty data [ N ] {};
    };

    /**
     * @brief   Unary plus operator.
     * @param   vec Input vector.
     * @return  Copy of the vector (element-wise unary plus).
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr auto operator+( const vector_template<Ty , N>& vec ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( + vec.data [ I ]... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Unary negation operator.
     * @param   vec Input vector.
     * @return  Vector with each element negated.
     */
    template <arithmetic_t Ty, std::size_t N>
    [[nodiscard]] constexpr auto operator-( const vector_template<Ty , N>& vec ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( - vec.data [ I ]... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Element-wise addition of two vectors.
     * @param   vec1 Left-hand side vector.
     * @param   vec2 Right-hand side vector.
     * @return  vector_template with each element computed as vec1[i] + vec2[i].
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr auto operator+( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( ( vec1.data [ I ] + vec2.data [ I ] )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Element-wise subtraction of two vectors.
     * @param   vec1 Left-hand side vector.
     * @param   vec2 Right-hand side vector.
     * @return  vector_template with each element computed as vec1[i] - vec2[i].
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr auto operator-( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( ( vec1.data [ I ] - vec2.data [ I ] )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Element-wise multiplication of two vectors (Hadamard product).
     * @param   vec1 Left-hand side vector.
     * @param   vec2 Right-hand side vector.
     * @return  vector_template with each element computed as vec1[i] * vec2[i].
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr auto operator*( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( ( vec1.data [ I ] * vec2.data [ I ] )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Element-wise division of two vectors.
     * @param   vec1 Left-hand side vector (dividend).
     * @param   vec2 Right-hand side vector (divisor).
     * @return  vector_template with each element computed as vec1[i] / vec2[i].
     * @note    Division by zero is undefined behavior.
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr auto operator/( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( ( vec1.data [ I ] / vec2.data [ I ] )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Element-wise addition compound assignment.
     * @param   vec1 Left-hand side vector (modified in-place).
     * @param   vec2 Right-hand side vector.
     * @return  Reference to vec1.
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto& operator+=( vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( ( vec1.data [ I ] += vec2.data [ I ] ), ... );
        }( std::make_index_sequence<N>( ) );
        return vec1;
    }

    /**
     * @brief   Element-wise subtraction compound assignment.
     * @param   vec1 Left-hand side vector (modified in-place).
     * @param   vec2 Right-hand side vector.
     * @return  Reference to vec1.
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto& operator-=( vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( ( vec1.data [ I ] -= vec2.data [ I ] ), ... );
        }( std::make_index_sequence<N>( ) );
        return vec1;
    }

    /**
     * @brief   Element-wise multiplication compound assignment (Hadamard).
     * @param   vec1 Left-hand side vector (modified in-place).
     * @param   vec2 Right-hand side vector.
     * @return  Reference to vec1.
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto& operator*=( vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( ( vec1.data [ I ] *= vec2.data [ I ] ), ... );
        }( std::make_index_sequence<N>( ) );
        return vec1;
    }

    /**
     * @brief   Element-wise division compound assignment.
     * @param   vec1 Left-hand side vector (modified in-place, dividend).
     * @param   vec2 Right-hand side vector (divisor).
     * @return  Reference to vec1.
     * @note    Division by zero is undefined behavior.
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto& operator/=( vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( ( vec1.data [ I ] /= vec2.data [ I ] ), ... );
        }( std::make_index_sequence<N>( ) );
        return vec1;
    }

    /**
     * @brief   Scalar multiplication (vector * scalar).
     * @tparam  Ty Vector element type.
     * @tparam  Ts Scalar type (arithmetic).
     * @tparam  N Vector dimension.
     * @param   vec Input vector.
     * @param   scale Scalar multiplier.
     * @return  Vector with each element multiplied by scale.
     */
    template <arithmetic_t Ty , arithmetic_t Ts , std::size_t N>
    [[nodiscard]] constexpr auto operator*( const vector_template<Ty , N>& vec , const Ts& scale ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( ( vec.data [ I ] * scale )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Scalar division (vector / scalar).
     * @tparam  Ty Vector element type.
     * @tparam  Ts Scalar type (arithmetic).
     * @tparam  N Vector dimension.
     * @param   vec Input vector (dividend).
     * @param   scale Scalar divisor.
     * @return  Vector with each element divided by scale.
     * @note    Division by zero is undefined behavior.
     */
    template <arithmetic_t Ty , arithmetic_t Ts , std::size_t N>
    [[nodiscard]] constexpr auto operator/( const vector_template<Ty , N>& vec , const Ts& scale ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( ( vec.data [ I ] / scale )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Scalar multiplication compound assignment (vector *= scalar).
     * @tparam  Ty Vector element type.
     * @tparam  Ts Scalar type (arithmetic).
     * @tparam  N Vector dimension.
     * @param   vec Vector to be modified in-place.
     * @param   scale Scalar multiplier.
     * @return  Reference to vec.
     */
    template <arithmetic_t Ty ,arithmetic_t Ts , std::size_t N>
    constexpr auto& operator*=( vector_template<Ty , N>& vec , const Ts& scale ) noexcept {
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( ( vec.data [ I ] *= scale ), ... );
        }( std::make_index_sequence<N>( ) );
        return vec;
    }

    /**
     * @brief   Scalar division compound assignment (vector /= scalar).
     * @tparam  Ty Vector element type.
     * @tparam  Ts Scalar type (arithmetic).
     * @tparam  N Vector dimension.
     * @param   vec Vector to be modified in-place (dividend).
     * @param   scale Scalar divisor.
     * @return  Reference to vec.
     * @note    Division by zero is undefined behavior.
     */
    template <arithmetic_t Ty ,arithmetic_t Ts , std::size_t N>
    constexpr auto& operator/=( vector_template<Ty , N>& vec , const Ts& scale ) noexcept {
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( ( vec.data [ I ] /= scale ), ... );
        }( std::make_index_sequence<N>( ) );
        return vec;
    }

    /**
     * @brief   Element-wise equality comparison.
     * @param   vec1 Left-hand side vector.
     * @param   vec2 Right-hand side vector.
     * @return  vector_template<bool, N> with each element computed as vec1[i] == vec2[i].
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr vector_template<bool , N> operator==( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<bool , N>( ( vec1.data [ I ] == vec2.data [ I ] )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Element-wise inequality comparison.
     * @param   vec1 Left-hand side vector.
     * @param   vec2 Right-hand side vector.
     * @return  vector_template<bool, N> with each element computed as vec1[i] != vec2[i].
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr vector_template<bool , N> operator!=( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<bool , N>( ( vec1.data [ I ] != vec2.data [ I ] )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Element-wise greater-than-or-equal comparison.
     * @param   vec1 Left-hand side vector.
     * @param   vec2 Right-hand side vector.
     * @return  vector_template<bool, N> with each element computed as vec1[i] >= vec2[i].
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr vector_template<bool , N> operator>=( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<bool , N>( ( vec1.data [ I ] >= vec2.data [ I ] )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Element-wise less-than-or-equal comparison.
     * @param   vec1 Left-hand side vector.
     * @param   vec2 Right-hand side vector.
     * @return  vector_template<bool, N> with each element computed as vec1[i] <= vec2[i].
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr vector_template<bool , N> operator<=( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<bool , N>( ( vec1.data [ I ] <= vec2.data [ I ] )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Element-wise greater-than comparison.
     * @param   vec1 Left-hand side vector.
     * @param   vec2 Right-hand side vector.
     * @return  vector_template<bool, N> with each element computed as vec1[i] > vec2[i].
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr vector_template<bool , N> operator>( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<bool , N>( ( vec1.data [ I ] > vec2.data [ I ] )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Element-wise less-than comparison.
     * @param   vec1 Left-hand side vector.
     * @param   vec2 Right-hand side vector.
     * @return  vector_template<bool, N> with each element computed as vec1[i] < vec2[i].
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr vector_template<bool , N> operator<( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<bool , N>( ( vec1.data [ I ] < vec2.data [ I ] )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Element-wise logical AND.
     * @param   vec1 Left-hand side vector.
     * @param   vec2 Right-hand side vector.
     * @return  vector_template<bool, N> with each element computed as vec1[i] && vec2[i].
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr vector_template<bool , N> operator&&( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<bool , N>( ( vec1.data [ I ] && vec2.data [ I ] )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Element-wise logical OR.
     * @param   vec1 Left-hand side vector.
     * @param   vec2 Right-hand side vector.
     * @return  vector_template<bool, N> with each element computed as vec1[i] || vec2[i].
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr vector_template<bool , N> operator||( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<bool , N>( ( vec1.data [ I ] || vec2.data [ I ] )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Computes the dot product of two vectors.
     * @param   vec1 Left-hand side vector.
     * @param   vec2 Right-hand side vector.
     * @return  Scalar dot product value.
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] VERTEX_CONSTEXPR auto dot( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return ( ( vec1.data [ I ] * vec2.data [ I ] ) + ... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Computes the 2D cross product (scalar value).
     * @param   vec1 First 2D vector.
     * @param   vec2 Second 2D vector.
     * @return  Scalar cross product value (vec1.x * vec2.y - vec1.y * vec2.x).
     */
    template <arithmetic_t Ty>
    [[nodiscard]] VERTEX_CONSTEXPR auto cross( const vector_template<Ty , 2>& vec1 , const vector_template<Ty , 2>& vec2 ) noexcept {
        return vec1.data [ 0 ] * vec2.data [ 1 ] - vec1.data [ 1 ] * vec2.data [ 0 ];
    }

    /**
     * @brief   Computes the 3D cross product (vector perpendicular to both inputs).
     * @param   vec1 First 3D vector.
     * @param   vec2 Second 3D vector.
     * @return  3D vector perpendicular to both vec1 and vec2.
     */
    template <arithmetic_t Ty>
    [[nodiscard]] VERTEX_CONSTEXPR auto cross( const vector_template<Ty , 3>& vec1 , const vector_template<Ty , 3>& vec2 ) noexcept {
        return vector_template<Ty , 3>(
            vec1.data [ 1 ] * vec2.data [ 2 ] - vec1.data [ 2 ] * vec2.data [ 1 ] ,
            vec1.data [ 2 ] * vec2.data [ 0 ] - vec1.data [ 0 ] * vec2.data [ 2 ] ,
            vec1.data [ 0 ] * vec2.data [ 1 ] - vec1.data [ 1 ] * vec2.data [ 0 ]
        );
    }

    /**
     * @brief   Computes the Euclidean norm (length) of a vector.
     * @param   vec Input vector.
     * @return  Length of the vector.
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] VERTEX_CONSTEXPR auto length( const vector_template<Ty , N>& vec ) noexcept {
        return std::sqrt( dot<Ty , N>( vec , vec ) );
    }

    /**
     * @brief   Normalizes a vector to unit length.
     * @param   vec Input vector.
     * @return  Unit vector pointing in the same direction.
     * @note    If the vector is zero-length, the result is undefined (division by zero).
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] VERTEX_CONSTEXPR auto normalize( const vector_template<Ty , N>& vec ) noexcept {
        return vec / length<Ty , N>( vec );
    }

    /**
     * @brief   Computes the Euclidean distance between two vectors.
     * @param   vec1 First vector.
     * @param   vec2 Second vector.
     * @return  Distance between vec1 and vec2.
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] VERTEX_CONSTEXPR auto distance( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        return length<Ty , N>( vec2 - vec1 );
    }

    /**
     * @brief   Computes the angle (in radians) between two vectors.
     * @param   vec1 First vector.
     * @param   vec2 Second vector.
     * @return  Angle in radians between vec1 and vec2, in range [0, π].
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] VERTEX_CONSTEXPR auto angle( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        auto dt = dot<Ty , N>( vec1 , vec2 );
        auto l1 = length<Ty>( vec1 );
        auto l2 = length<Ty>( vec2 );
        auto cs = std::clamp( dt / ( l1 * l2 ) , -Ty { 1.0 } , Ty { 1.0 } );
        return std::acos( cs );
    }

    /**
     * @brief   Rotates a 2D vector around the origin.
     * @tparam  Ty Vector element type (arithmetic).
     * @tparam  Tf Angle type (arithmetic).
     * @param   vec Input 2D vector.
     * @param   angle Rotation angle in radians.
     * @return  Rotated 2D vector.
     */
    template <arithmetic_t Ty , arithmetic_t Tf>
    [[nodiscard]] VERTEX_CONSTEXPR auto rotate( const vector_template<Ty , 2>& vec , const Tf& angle ) noexcept {
        const auto s = std::sin( angle );
        const auto c = std::cos( angle );
        return vector_template<Ty , 2>(
            static_cast< Ty >( c * vec.data[ 0 ] - s * vec.data[ 1 ] ) ,
            static_cast< Ty >( c * vec.data[ 1 ] + s * vec.data[ 0 ] )
        );
    }

    /**
     * @brief   Rotates a 2D vector around a specified pivot point.
     * @tparam  Ty Vector element type (arithmetic).
     * @tparam  Tf Angle type (arithmetic).
     * @param   vec Input 2D vector to rotate.
     * @param   point Pivot point to rotate around.
     * @param   angle Rotation angle in radians.
     * @return  Rotated 2D vector around the pivot point.
     */
    template <arithmetic_t Ty , arithmetic_t Tf>
    [[nodiscard]] VERTEX_CONSTEXPR auto rotate( const vector_template<Ty , 2>& vec , const vector_template<Ty , 2>& point , const Tf angle ) noexcept {
        return rotate<Ty , Tf>( vec - point , angle ) + point;
    }

    /**
     * @brief   Rotates a 3D vector around an arbitrary axis.
     * @tparam  Ty Vector element type (arithmetic).
     * @tparam  Tf Angle type (arithmetic).
     * @param   vec Input 3D vector to rotate.
     * @param   axis Rotation axis (will be normalized internally).
     * @param   angle Rotation angle in radians.
     * @return  Rotated 3D vector around the specified axis.
     */
    template <arithmetic_t Ty , arithmetic_t Tf>
    [[nodiscard]] VERTEX_CONSTEXPR auto rotate( const vector_template<Ty , 3>& vec , const vector_template<Ty , 3>& axis , const Tf angle ) noexcept {
        auto an = normalize<Ty , 3>( axis );
        auto cx = std::cos( angle );
        auto sx = std::sin( angle );
        auto oc = Ty { 1 } - cx;
        auto dt = dot<Ty , 3>( an , vec );
        auto cs = cross<Ty>( an , vec );
        return vec * cx + cs * sx + an * dt * oc;
    }

    /**
     * @brief   Rotates a 3D vector around an arbitrary axis passing through a pivot point.
     * @tparam  Ty Vector element type (arithmetic).
     * @tparam  Tf Angle type (arithmetic).
     * @param   vec Input 3D vector to rotate.
     * @param   point Pivot point on the rotation axis.
     * @param   axis Rotation axis direction (will be normalized internally).
     * @param   angle Rotation angle in radians.
     * @return  Rotated 3D vector around the axis through the pivot point.
     */
    template <arithmetic_t Ty , arithmetic_t Tf>
    [[nodiscard]] VERTEX_CONSTEXPR auto rotate( const vector_template<Ty , 3>& vec , const vector_template<Ty , 3>& point , const vector_template<Ty , 3>& axis , const Tf angle ) noexcept {
        return rotate<Ty , Tf>( vec - point , axis , angle ) + point;
    }

    /**
     * @brief   Rotates a 4D vector in two independent 2D planes.
     * @tparam  Ty Vector element type (arithmetic).
     * @tparam  Tf Angle type (arithmetic).
     * @param   vec Input 4D vector to rotate.
     * @param   a_xy Rotation angle in the XY plane.
     * @param   a_zw Rotation angle in the ZW plane.
     * @return  Rotated 4D vector.
     */
    template <arithmetic_t Ty , arithmetic_t Tf>
    [[nodiscard]] VERTEX_CONSTEXPR auto rotate( const vector_template<Ty , 4>& vec , const Tf a_xy , const Tf a_zw ) noexcept {
        const auto c1 = std::cos( a_xy );
        const auto s1 = std::sin( a_xy );
        const auto c2 = std::cos( a_zw );
        const auto s2 = std::sin( a_zw );
        return vector_template<Ty , 4>(
            static_cast< Ty >( vec.data [ 0 ] * c1 - vec.data [ 1 ] * s1 ) ,
            static_cast< Ty >( vec.data [ 0 ] * s1 + vec.data [ 1 ] * c1 ) ,
            static_cast< Ty >( vec.data [ 2 ] * c2 - vec.data [ 3 ] * s2 ) ,
            static_cast< Ty >( vec.data [ 2 ] * s2 + vec.data [ 3 ] * c2 )
        );
    }

    /**
     * @brief   Computes the reflection vector of an incident direction.
     * @tparam  Ty Vector element type (arithmetic).
     * @tparam  N Vector dimension.
     * @param   i Incident direction vector.
     * @param   n Surface normal vector (should be normalized).
     * @return  Reflected direction vector.
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] VERTEX_CONSTEXPR auto reflect( const vector_template<Ty , N>& i , const vector_template<Ty , N>& n ) noexcept {
        return i - n * ( Ty { 2 } * dot( i , n ) );
    }

    /**
     * @brief   Computes the refraction vector using Snell's law.
     * @tparam  Ty Vector element type (arithmetic).
     * @tparam  N Vector dimension.
     * @param   i Incident direction vector (pointing toward the surface).
     * @param   n Surface normal vector (should be normalized).
     * @param   eta Ratio of indices of refraction (n1 / n2).
     * @return  Refracted direction vector, or zero vector if total internal reflection occurs.
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] VERTEX_CONSTEXPR auto refract( const vector_template<Ty , N>& i , const vector_template<Ty , N>& n , const Ty eta ) noexcept {
        const auto dt = dot( n , i );
        const auto k  = Ty { 1 } - eta * eta * ( Ty { 1 } - dt * dt );
        if ( k < Ty { 0 } ) {
            return vector_template<Ty , N>( );
        }
        return i * eta - n * ( eta * dt + std::sqrt( k ) );
    }

    /**
     * @brief   Orients a normal to face the same direction as a reference direction.
     * @tparam  Ty Vector element type (arithmetic).
     * @tparam  N Vector dimension.
     * @param   n Normal vector to be oriented.
     * @param   i Incident direction vector (pointing toward the surface).
     * @param   nref Reference normal vector.
     * @return  n if nref points toward i, otherwise -n.
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] VERTEX_CONSTEXPR auto face_forward( const vector_template<Ty , N>& n , const vector_template<Ty , N>& i , const vector_template<Ty , N>& nref ) noexcept {
        return dot<Ty , N>( nref , i ) < Ty { 0 } ? n : -n;
    }

    /**
     * @brief   Creates a vector with all elements set to the same scalar value.
     * @tparam  Ty Vector element type (arithmetic).
     * @tparam  N Vector dimension.
     * @param   scale Value to fill all elements with.
     * @return  vector_template<Ty, N> with all elements equal to scale.
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] VERTEX_CONSTEXPR auto vector_cast( const Ty& scale ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty, N>( ( ( void )I , scale )... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Creates a vector from a set of arguments, deducing dimension from argument count.
     * @tparam  Ty Element type of the resulting vector.
     * @param   args Element values (variadic).
     * @return  vector_template<Ty, N> where N = sizeof...(args).
     */
    template <arithmetic_t Ty>
    [[nodiscard]] VERTEX_CONSTEXPR auto vector_cast( auto&&... args ) noexcept {
        return vector_template<Ty, sizeof...( args )>( std::forward<decltype( args )>( args )... );
    }
}

#endif //VERTEX_VECTORX_HPP