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
 * @file      vtxvecx.hpp
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
 * - dot() - dot product
 * - cross() - 2D scalar cross / 3D vector cross
 * - length() - Euclidean norm
 * - normalize() - unit vector
 * - distance() - Euclidean distance
 * - angle() - angle between two vectors
 * - rotate() - 2D/3D/4D rotation
 * - reflect() - reflection vector
 * - refract() - refraction vector
 * - face_forward() - orientation correction
 * - fma() - fused multiply-add
 * - vector_cast() - type conversion and construction
 *
 * @note  All operations are compile-time unrolled via index_sequence
 *        and expanded using fold expressions (C++17).
 */
#ifndef VERTEX_VECTORX_HPP
#define VERTEX_VECTORX_HPP
#pragma once

#include <cmath>            ///< std::sqrt, std::cos, std::sin, std::acos
#include <cstddef>          ///< std::size_t, std::ptrdiff_t
#include <initializer_list> ///< std::initializer_list
#include <algorithm>        ///< std::clamp, std::min, std::max, std::reverse_iterator
#include <utility>          ///< std::index_sequence, std::make_index_sequence, std::forward
#include "vtxbasic.hpp"     ///< arithmetic_t, VERTEX_CONSTEXPR, vtx type aliases

namespace vtx {
    /**
     * @brief   N-dimensional vector template.
     * @tparam  Ty  Arithmetic type (integral or floating-point).
     * @tparam  N   Number of dimensions.
     *
     * @details
     * Provides random-access container semantics with compile-time fixed size.
     * All element-wise operations are unrolled at compile time via index_sequence.
     *
     * Example:
     * @code
     * using vec3f = vector_template<float, 3>;
     * vec3f v{1.0f, 2.0f, 3.0f};
     * auto len = length(v);
     * auto normalized = normalize(v);
     * @endcode
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
         * @brief Default constructor - zero-initializes all elements.
         */
        constexpr vector_template( ) noexcept = default;

        /**
         * @brief Returns an iterator to the first element.
         * @return Iterator to the beginning of the vector.
         */
        [[nodiscard]] constexpr iterator begin() noexcept {
            return data;
        }

        /**
         * @brief Returns an iterator to the element following the last element.
         * @return Iterator to the end of the vector.
         */
        [[nodiscard]] constexpr iterator end() noexcept {
            return data + N;
        }

        /**
         * @brief Returns a const iterator to the first element.
         * @return Const iterator to the beginning of the vector.
         */
        [[nodiscard]] constexpr const_iterator begin() const noexcept {
            return data;
        }

        /**
         * @brief Returns a const iterator to the element following the last element.
         * @return Const iterator to the end of the vector.
         */
        [[nodiscard]] constexpr const_iterator end() const noexcept {
            return data + N;
        }

        /**
         * @brief Returns a const iterator to the first element.
         * @return Const iterator to the beginning of the vector.
         */
        [[nodiscard]] constexpr const_iterator cbegin() const noexcept {
            return data;
        }

        /**
         * @brief Returns a const iterator to the element following the last element.
         * @return Const iterator to the end of the vector.
         */
        [[nodiscard]] constexpr const_iterator cend() const noexcept {
            return data + N;
        }

        /**
         * @brief Returns a reverse iterator to the first element of the reversed vector.
         * @return Reverse iterator to the beginning of the reversed vector.
         */
        [[nodiscard]] constexpr reverse_iterator rbegin() noexcept {
            return std::reverse_iterator(end());
        }

        /**
         * @brief Returns a reverse iterator to the element following the last element of the reversed vector.
         * @return Reverse iterator to the end of the reversed vector.
         */
        [[nodiscard]] constexpr reverse_iterator rend() noexcept {
            return std::reverse_iterator(begin());
        }

        /**
         * @brief Returns a const reverse iterator to the first element of the reversed vector.
         * @return Const reverse iterator to the beginning of the reversed vector.
         */
        [[nodiscard]] constexpr const_reverse_iterator rbegin() const noexcept {
            return std::reverse_iterator(end());
        }

        /**
         * @brief Returns a const reverse iterator to the element following the last element of the reversed vector.
         * @return Const reverse iterator to the end of the reversed vector.
         */
        [[nodiscard]] constexpr const_reverse_iterator rend() const noexcept {
            return std::reverse_iterator(begin());
        }

        /**
         * @brief Returns a const reverse iterator to the first element of the reversed vector.
         * @return Const reverse iterator to the beginning of the reversed vector.
         */
        [[nodiscard]] constexpr const_reverse_iterator crbegin() const noexcept {
            return std::reverse_iterator(cend());
        }

        /**
         * @brief Returns a const reverse iterator to the element following the last element of the reversed vector.
         * @return Const reverse iterator to the end of the reversed vector.
         */
        [[nodiscard]] constexpr const_reverse_iterator crend() const noexcept {
            return std::reverse_iterator(cbegin());
        }

        /**
         * @brief Constructs vector from variadic arguments.
         * @tparam Args Variadic argument types.
         * @param args Element values.
         *
         * @details
         * Example:
         * @code
         * vector_template<float, 3> v{1.0f, 2.0f, 3.0f};
         * @endcode
         */
        template <typename... Args>
        explicit constexpr vector_template( Args&&... args ) noexcept {
            const Ty vals [ N ] = { static_cast<Ty>( args )... };
            [ & ]<size_t... I>( std::index_sequence<I...> ) {
                ( ( data [ I ] = vals [ I ] ), ... );
            }( std::make_index_sequence<N>( ) );
        }

        /**
         * @brief Constructs vector from an initializer list.
         * @param vec Initializer list of element values.
         *
         * @details
         * Example:
         * @code
         * vector_template<float, 3> v{ {1.0f, 2.0f, 3.0f} };
         * @endcode
         */
        explicit constexpr vector_template( const std::initializer_list<Ty>& vec ) noexcept {
            [ & ]<size_t... I>( std::index_sequence<I...> ) {
                ( ( data [ I ] = vec.begin( ) [ I ] ) , ... );
            }( std::make_index_sequence<N>( ) );
        }

        /**
         * @brief Accesses element at specified index (unchecked).
         * @param idx Index of the element.
         * @return Reference to the element.
         */
        [[nodiscard]] constexpr Ty& operator[]( const std::size_t idx ) noexcept {
            return data [ idx ];
        }

        /**
         * @brief Accesses element at specified index (unchecked, const).
         * @param idx Index of the element.
         * @return Const reference to the element.
         */
        [[nodiscard]] constexpr const Ty& operator[]( const std::size_t idx ) const noexcept {
            return data [ idx ];
        }

        /**
         * @brief Converts vector to another arithmetic type.
         * @tparam U Target arithmetic type.
         * @return vector_template<U, N> with each element casted to type U.
         *
         * @details
         * Example:
         * @code
         * vector_template<float, 3> vf{1.0f, 2.0f, 3.0f};
         * auto vi = static_cast<vector_template<int, 3>>(vf);  // {1, 2, 3}
         * @endcode
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
     * @brief Unary plus operator.
     * @param vec Input vector.
     * @return Copy of the vector (element-wise unary plus).
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr auto operator+( const vector_template<Ty , N>& vec ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( +vec.data [ I ]... );
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
            return vector_template<Ty , N>( -vec.data [ I ]... );
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
    [[nodiscard]] constexpr vector_template<bool , N> operator> ( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
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
    [[nodiscard]] constexpr vector_template<bool , N> operator< ( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
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
     *
     * @details
     * In 2D, the cross product returns a scalar representing the signed area
     * of the parallelogram formed by the two vectors.
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
     *
     * @details
     * The resulting vector is orthogonal to both input vectors.
     * Magnitude equals the area of the parallelogram formed by the two vectors.
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
     *
     * @details
     * The angle is computed using: θ = acos(dot(v1, v2) / (|v1| * |v2|)).
     * The result is clamped to [-1, 1] to handle floating-point precision issues.
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] VERTEX_CONSTEXPR auto angle( const vector_template<Ty , N>& vec1 , const vector_template<Ty , N>& vec2 ) noexcept {
        const auto dt = dot<Ty , N>( vec1 , vec2 );
        const auto l1 = length<Ty>( vec1 );
        const auto l2 = length<Ty>( vec2 );
        const auto cs = std::clamp( dt / ( l1 * l2 ) , -Ty { 1.0 } , Ty { 1.0 } );
        return std::acos( cs );
    }

    /**
     * @brief   Rotates a 2D vector around the origin.
     * @tparam  Ty Vector element type (arithmetic).
     * @tparam  Tf Angle type (arithmetic).
     * @param   vec Input 2D vector.
     * @param   angle Rotation angle in radians.
     * @return  Rotated 2D vector.
     *
     * @details
     * Rotates the vector around the origin using the standard 2D rotation matrix:
     * [ cos(θ)  -sin(θ) ]
     * [ sin(θ)   cos(θ) ]
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
     *
     * @details
     * Translates the vector so the pivot is at the origin, rotates,
     * then translates back to the original position.
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
     *
     * @details
     * Implements Rodrigues' rotation formula:
     * v_rot = v * cos(θ) + (axis × v) * sin(θ) + axis * (axis · v) * (1 - cos(θ))
     *
     * The axis is automatically normalized before computation.
     */
    template <arithmetic_t Ty , arithmetic_t Tf>
    [[nodiscard]] VERTEX_CONSTEXPR auto rotate( const vector_template<Ty , 3>& vec , const vector_template<Ty , 3>& axis , const Tf angle ) noexcept {
        const auto an = normalize<Ty , 3>( axis );
        const auto cx = std::cos( angle );
        const auto sx = std::sin( angle );
        const auto oc = Ty { 1 } - cx;
        const auto dt = dot<Ty , 3>( an , vec );
        const auto cs = cross<Ty>( an , vec );
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
     *
     * @details
     * Translates the vector so the pivot is at the origin, applies Rodrigues'
     * rotation formula, then translates back to the original position.
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
     *
     * @details
     * Performs two independent 2D rotations:
     * - XY plane: rotates (x, y) components.
     * - ZW plane: rotates (z, w) components.
     *
     * This is useful for 4D geometry and quaternion-like operations.
     *
     * Rotation matrix:
     * [ cos(θ₁)  -sin(θ₁)  0          0        ]
     * [ sin(θ₁)   cos(θ₁)  0          0        ]
     * [ 0          0         cos(θ₂)  -sin(θ₂) ]
     * [ 0          0         sin(θ₂)   cos(θ₂) ]
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
     *
     * @details
     * Implements the reflection formula: r = i - 2 * (i · n) * n
     * Assumes n is a unit vector. If n is not normalized, the result will be scaled.
     *
     * @note The incident vector i points toward the surface (not away from it).
     *       Both i and n should be unit vectors for physically correct reflection.
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
     *
     * @details
     * Implements Snell's law: η₁ * sin(θ₁) = η₂ * sin(θ₂)
     * 
     * The refracted vector is computed as:
     * r = η * i - n * (η * (i · n) + sqrt(k))
     * where k = 1 - η² * (1 - (i · n)²)
     *
     * If k < 0, total internal reflection occurs and a zero vector is returned.
     *
     * @note Both i and n should be unit vectors for physically correct refraction.
     *       eta = n1 / n2 (incident medium index / transmitted medium index).
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
     *
     * @details
     * Ensures that the returned normal faces the same hemisphere as the incident
     * direction relative to the reference normal.
     *
     * The logic is:
     * - If dot(nref, i) < 0, the reference normal faces away from i, so return n
     * - Otherwise, return -n (flip the normal)
     *
     * This is commonly used in shaders to ensure normals face the correct direction
     * for lighting calculations.
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] VERTEX_CONSTEXPR auto face_forward( const vector_template<Ty , N>& n , const vector_template<Ty , N>& i , const vector_template<Ty , N>& nref ) noexcept {
        return dot<Ty , N>( nref , i ) < Ty { 0 } ? n : -n;
    }

    /**
     * @brief   Fused multiply-add: (a * b) + c.
     * @tparam  Ty Vector element type (arithmetic).
     * @tparam  N Vector dimension.
     * @param   a Multiplicand vector.
     * @param   b Multiplier vector.
     * @param   c Addend vector.
     * @return  Element-wise (a * b + c).
     *
     * @details
     * Performs element-wise fused multiply-add operation.
     * For floating-point types, this can be mapped to hardware FMA instructions
     * by the compiler (when enabled with -mfma or /arch:AVX2).
     *
     * @note This implementation does not guarantee a single hardware FMA instruction;
     *       it is a convenience function that may be optimized by the compiler.
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] VERTEX_CONSTEXPR auto fma( const vector_template<Ty , N>& a , const vector_template<Ty , N>& b , const vector_template<Ty , N>& c ) noexcept {
        return a * b + c;
    }

    /**
     * @brief   Creates a vector with all elements set to the same scalar value.
     * @tparam  Ty Vector element type (arithmetic).
     * @tparam  N Vector dimension.
     * @param   scale Value to fill all elements with.
     * @return  vector_template<Ty, N> with all elements equal to scale.
     *
     * @details
     * Example:
     * @code
     * auto v = vector_cast<float, 4>(2.0f);  // {2.0f, 2.0f, 2.0f, 2.0f}
     * @endcode
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] VERTEX_CONSTEXPR auto vector_cast( const Ty& scale ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty, N>{ ( ( void )I , scale )... };
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Creates a vector from a set of arguments, deducing dimension from argument count.
     * @tparam  Ty Element type of the resulting vector.
     * @param   args Element values (variadic).
     * @return  vector_template<Ty, N> where N = sizeof...(args).
     *
     * @details
     * Example:
     * @code
     * auto v = vector_cast<float>(1.0f, 2.0f, 3.0f);  // vector_template<float, 3>{1, 2, 3}
     * @endcode
     *
     * @note The dimension N is automatically deduced from the number of arguments.
     *       All arguments are forwarded and cast to Ty via the vector_template constructor.
     */
    template <arithmetic_t Ty>
    [[nodiscard]] VERTEX_CONSTEXPR auto vector_cast(auto&&... args) noexcept {
        return vector_template<Ty, sizeof...(args)>( std::forward<decltype(args)>(args)... );
    }
}

#endif //VERTEX_VECTORX_HPP
