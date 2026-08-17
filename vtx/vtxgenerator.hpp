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
 * @file    vtxgenerator.hpp
 * @brief   Generator utilities for creating vectors and matrices
 *
 *          This header provides convenient factory functions for generating vector and
 *          matrix objects with various initialization patterns. It includes functions
 *          for creating zero-filled, one-filled, identity, diagonal, and randomly
 *          initialized vectors and matrices.
 *
 *          The generator functions are designed to be constexpr where possible, providing
 *          compile-time evaluation for performance-critical applications. They leverage
 *          compile-time index sequences to eliminate runtime overhead for element access.
 *
 * @see     vtxmatrix.hpp
 * @see     vtxvector.hpp
 */
#ifndef VERTEX_GENERATOR_HPP
#define VERTEX_GENERATOR_HPP
#pragma once

#include "vtxmatrix.hpp"
#include "vtxvector.hpp"
#include <utility>
#include <array>
#include <initializer_list>
#include <vector>
#include <random>

namespace vtx {

    /**
     * @brief   Constructs a vector object from a raw data pointer
     *
     *          This function template reads N elements from contiguous raw data storage
     *          and constructs a vector<Ty, N> object. It uses compile-time index sequence
     *          expansion to determine the element positions at compile time, ensuring
     *          zero runtime overhead.
     *
     * @tparam  Ty The arithmetic type of vector elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the vector, i.e., the number of elements to read
     * @param   data Reference to a pointer to the raw data array. After function execution,
     *              the pointer will be advanced by N positions
     * @return  constexpr auto A vector<Ty, N> object constructed from the data
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    The data pointer is modified during the call (advanced by N elements).
     *          Ensure sufficient elements are available before calling
     *
     * @warning The caller must guarantee that the memory pointed to by data contains
     *          at least N valid elements of type Ty
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto from_data( const Ty*& data ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( data [ I ]... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Constructs a vector object from a fixed-size C-style array
     *
     *          This function template reads all N elements from a fixed-size raw array
     *          and constructs a vector<Ty, N> object. It uses compile-time index sequence
     *          expansion to access array elements at compile time, ensuring zero runtime
     *          overhead. The array size is deduced from the template parameter N.
     *
     * @tparam  Ty The arithmetic type of vector elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the vector, deduced from the array size
     * @param   data Reference to a fixed-size C-style array of Ty with N elements
     * @return  constexpr auto A vector<Ty, N> object constructed from the array elements
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    Unlike the pointer version, the array reference is not modified
     * @note    The array size N is automatically deduced by the compiler
     *
     * @warning The array must contain exactly N elements; this is enforced by the type system
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto from_data( const Ty* ( &data )[ N ] ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( data [ I ]... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Constructs a vector object from an initializer list
     *
     *          This function template reads N elements from a std::initializer_list
     *          and constructs a vector<Ty, N> object. It uses compile-time index sequence
     *          expansion to access elements at compile time via the begin() iterator,
     *          ensuring zero runtime overhead for index calculations.
     *
     * @tparam  Ty The arithmetic type of vector elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the vector, i.e., the number of elements to read
     * @param   data Reference to a const std::initializer_list of Ty elements
     * @return  constexpr auto A vector<Ty, N> object constructed from the initializer list elements
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    Accesses elements using data.begin()[I] with compile-time indices
     * @note    The initializer_list is not modified by this function
     *
     * @warning The caller must ensure that data.size() >= N; accessing out-of-bounds
     *          elements results in undefined behavior
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto from_data( const std::initializer_list<Ty>& data ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( data.begin( )[ I ]... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Constructs a vector object from a std::array
     *
     *          This function template reads all N elements from a std::array
     *          and constructs a vector<Ty, N> object. It uses compile-time index sequence
     *          expansion to access array elements at compile time via operator[],
     *          ensuring zero runtime overhead. The size N is deduced from the std::array
     *          template parameter and is guaranteed to match the vector dimension.
     *
     * @tparam  Ty The arithmetic type of vector elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the vector, deduced from the std::array size
     * @param   data Reference to a const std::array<Ty, N> containing the source elements
     * @return  constexpr auto A vector<Ty, N> object constructed from the array elements
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    The std::array is not modified by this function
     * @note    The size N is automatically deduced by the compiler from the array type
     *
     * @warning The std::array size is enforced by the type system, so out-of-bounds
     *          access is statically prevented
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto from_data( const std::array<Ty , N>& data ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( data [ I ]... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Constructs a vector object from a std::vector
     *
     *          This function template reads N elements from a std::vector
     *          and constructs a vector<Ty, N> object. It uses compile-time index sequence
     *          expansion to access elements at compile time via operator[],
     *          ensuring zero runtime overhead for index calculations.
     *
     * @tparam  Ty The arithmetic type of vector elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the vector, i.e., the number of elements to read
     * @param   data Reference to a const std::vector<Ty> containing the source elements
     * @return  constexpr auto A vector<Ty, N> object constructed from the vector elements
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    The std::vector is not modified by this function
     * @note    Elements are accessed using data[I] with compile-time indices
     *
     * @warning The caller must ensure that data.size() >= N; accessing out-of-bounds
     *          elements results in undefined behavior
     * @warning The size N is not automatically deduced; the caller must specify it explicitly
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto from_data( const std::vector<Ty>& data ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( data [ I ]... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Constructs a matrix object from a raw data pointer
     *
     *          This function template reads R * C elements from contiguous raw data storage
     *          and constructs a matrix<Ty, R, C> object. It uses compile-time index sequence
     *          expansion to determine the element positions at compile time, ensuring
     *          zero runtime overhead. The total number of elements read is the product
     *          of the matrix dimensions R and C.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  R The number of rows in the matrix
     * @tparam  C The number of columns in the matrix
     * @param   data Reference to a pointer to the raw data array. After function execution,
     *             the pointer will be advanced by R * C positions
     * @return  constexpr auto A matrix<Ty, R, C> object constructed from the data
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    The total number of elements read is R * C, calculated at compile time
     * @note    The data pointer is modified during the call (advanced by R * C elements).
     *          Ensure sufficient elements are available before calling
     *
     * @warning The caller must guarantee that the memory pointed to by data contains
     *          at least R * C valid elements of type Ty
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    constexpr auto from_data( const Ty*& data ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( data [ I ]... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Constructs a matrix object from a fixed-size C-style array
     *
     *          This function template reads all R * C elements from a fixed-size raw array
     *          and constructs a matrix<Ty, R, C> object. It uses compile-time index sequence
     *          expansion to access array elements at compile time, ensuring zero runtime
     *          overhead. The array size is deduced from the template parameter and must
     *          exactly match R * C.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  R The number of rows in the matrix
     * @tparam  C The number of columns in the matrix
     * @param   data Reference to a fixed-size C-style array of Ty with R * C elements
     * @return  constexpr auto A matrix<Ty, R, C> object constructed from the array elements
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    The array size R * C is enforced by the type system
     * @note    Unlike the pointer version, the array reference is not modified
     *
     * @warning The array must contain exactly R * C elements; this is enforced
     *          by the type system at compile time
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    constexpr auto from_data( const Ty* ( &data )[ R * C ] ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( data [ I ]... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Constructs a matrix object from a 2D fixed-size C-style array
     *
     *          This function template reads all R * C elements from a 2D fixed-size raw array
     *          and constructs a matrix<Ty, R, C> object. It uses compile-time index sequence
     *          expansion to access array elements at compile time, converting a linear index
     *          to 2D indices via division and modulo operations. The array dimensions R and C
     *          are deduced from the array type and must match the matrix dimensions exactly.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  R The number of rows in the matrix, deduced from the first array dimension
     * @tparam  C The number of columns in the matrix, deduced from the second array dimension
     * @param   data Reference to a 2D fixed-size C-style array of Ty with dimensions [R][C]
     * @return  constexpr auto A matrix<Ty, R, C> object constructed from the array elements
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    The array dimensions R and C are enforced by the type system
     * @note    The array is not modified by this function
     * @note    Elements are accessed using 2D indices: data[I / C][I % C]
     *
     * @warning The array must have exactly R rows and C columns; this is enforced
     *          by the type system at compile time
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    constexpr auto from_data( const Ty* ( &data )[ R ][ C ] )noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( data [ I / C ][ I % C ]... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Constructs a matrix object from an initializer list
     *
     *          This function template reads R * C elements from a std::initializer_list
     *          and constructs a matrix<Ty, R, C> object. It uses compile-time index sequence
     *          expansion to access elements at compile time via the begin() iterator,
     *          ensuring zero runtime overhead for index calculations. The total number of
     *          elements read is the product of the matrix dimensions R and C.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  R The number of rows in the matrix
     * @tparam  C The number of columns in the matrix
     * @param   data Reference to a const std::initializer_list of Ty elements
     * @return  constexpr auto A matrix<Ty, R, C> object constructed from the initializer list elements
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    Accesses elements using data.begin()[I] with compile-time indices
     * @note    The initializer_list is not modified by this function
     * @note    The total number of elements read is R * C, calculated at compile time
     *
     * @warning The caller must ensure that data.size() >= R * C; accessing out-of-bounds
     *          elements results in undefined behavior
     * @warning The dimensions R and C are not automatically deduced; the caller must
     *          specify them explicitly
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    constexpr auto from_data( const std::initializer_list<Ty>& data ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( data.begin [ I ]... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Constructs a matrix object from a std::array
     *
     *          This function template reads all R * C elements from a std::array
     *          and constructs a matrix<Ty, R, C> object. It uses compile-time index sequence
     *          expansion to access array elements at compile time via operator[],
     *          ensuring zero runtime overhead. The size of the std::array is deduced from
     *          its template parameter and must exactly match R * C.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  R The number of rows in the matrix
     * @tparam  C The number of columns in the matrix
     * @param   data Reference to a const std::array<Ty, R * C> containing the source elements
     * @return  constexpr auto A matrix<Ty, R, C> object constructed from the array elements
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    The std::array is not modified by this function
     * @note    The array size R * C is enforced by the type system
     *
     * @warning The std::array size is enforced by the type system, so out-of-bounds
     *          access is statically prevented
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    constexpr auto from_data( const std::array<Ty , R * C>& data ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( data [ I ]... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Constructs a matrix object from a std::vector
     *
     *          This function template reads R * C elements from a std::vector
     *          and constructs a matrix<Ty, R, C> object. It uses compile-time index sequence
     *          expansion to access elements at compile time via operator[],
     *          ensuring zero runtime overhead for index calculations. The total number of
     *          elements read is the product of the matrix dimensions R and C.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  R The number of rows in the matrix
     * @tparam  C The number of columns in the matrix
     * @param   data Reference to a const std::vector<Ty> containing the source elements
     * @return  constexpr auto A matrix<Ty, R, C> object constructed from the vector elements
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    The std::vector is not modified by this function
     * @note    Elements are accessed using data[I] with compile-time indices
     * @note    The total number of elements read is R * C, calculated at compile time
     *
     * @warning The caller must ensure that data.size() >= R * C; accessing out-of-bounds
     *          elements results in undefined behavior
     * @warning The dimensions R and C are not automatically deduced; the caller must
     *          specify them explicitly
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    constexpr auto from_data( const std::vector<Ty>& data ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( data [ I ]... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Creates a vector of zeros
     *
     *          This function template constructs a vector<Ty, N> object with all elements
     *          zero-initialized. It provides a convenient way to create a zero vector
     *          without explicitly specifying the constructor arguments.
     *
     * @tparam  Ty The arithmetic type of vector elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the vector
     * @return  constexpr auto A vector<Ty, N> object with all elements initialized to zero
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    Uses the default constructor of vector which zero-initializes all elements
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto zeros( ) noexcept {
        return vector_template<Ty , N>( );
    }

    /**
     * @brief   Creates a matrix of zeros
     *
     *          This function template constructs a matrix<Ty, R, C> object with all elements
     *          zero-initialized. It provides a convenient way to create a zero matrix
     *          without explicitly specifying the constructor arguments.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  R The number of rows in the matrix
     * @tparam  C The number of columns in the matrix
     * @return  constexpr auto A matrix<Ty, R, C> object with all elements initialized to zero
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    Uses the default constructor of matrix which zero-initializes all elements
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    constexpr auto zeros( ) noexcept {
        return matrix_template<Ty , R , C>( );
    }

    /**
     * @brief   Creates a vector of ones
     *
     *          This function template constructs a vector<Ty, N> object with all elements
     *          initialized to the value 1. It uses compile-time index sequence expansion
     *          to generate N elements, each set to Ty{1}, ensuring zero runtime overhead.
     *
     * @tparam  Ty The arithmetic type of vector elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the vector
     * @return  constexpr auto A vector<Ty, N> object with all elements initialized to one
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    Uses the comma operator with I to discard the index and produce Ty{1} for each element
     * @note    The expression (void)I, Ty{1} creates N copies of the value 1 at compile time
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto ones( ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( ( (void)I, Ty{ 1 } ) ... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Creates a matrix of ones
     *
     *          This function template constructs a matrix<Ty, R, C> object with all elements
     *          initialized to the value 1. It uses compile-time index sequence expansion
     *          to generate R * C elements, each set to Ty{1}, ensuring zero runtime overhead.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  R The number of rows in the matrix
     * @tparam  C The number of columns in the matrix
     * @return  constexpr auto A matrix<Ty, R, C> object with all elements initialized to one
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    Uses the comma operator with I to discard the index and produce Ty{1} for each element
     * @note    The expression (void)I, Ty{1} creates R * C copies of the value 1 at compile time
     * @note    The total number of elements generated is R * C
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    constexpr auto ones( ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( ( (void)I, Ty{ 1 } ) ... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Creates a vector filled with a specified value
     *
     *          This function template constructs a vector<Ty, N> object with all elements
     *          initialized to the given value. It uses compile-time index sequence expansion
     *          to generate N elements, each set to the value converted to type Ty, ensuring
     *          zero runtime overhead. The noexcept specification depends on whether the
     *          conversion from the value type to Ty can be done without throwing exceptions.
     *
     * @tparam  Ty The arithmetic type of vector elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the vector
     * @param   value The value to fill the vector with (type deduced automatically)
     * @return  constexpr auto A vector<Ty, N> object with all elements initialized to the specified value
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    The noexcept specification is conditional:
     *          - noexcept if the construction Ty{value} is noexcept
     *          - Throws if the conversion from value to Ty can throw
     * @note    Uses the comma operator with I to discard the index and produce static_cast<Ty>(value) for each element
     * @note    The expression (void)I, static_cast<Ty>(value) creates N copies of the converted value at compile time
     *
     * @warning The value type must be convertible to Ty; otherwise, a compilation error occurs
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto full( const auto& value ) noexcept( std::is_nothrow_constructible_v<Ty , decltype( value )> ) {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( ( (void)I, static_cast<Ty>( value ) ) ... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Creates a matrix filled with a specified value
     *
     *          This function template constructs a matrix<Ty, R, C> object with all elements
     *          initialized to the given value. It uses compile-time index sequence expansion
     *          to generate R * C elements, each set to the value converted to type Ty, ensuring
     *          zero runtime overhead. The noexcept specification depends on whether the
     *          conversion from the value type to Ty can be done without throwing exceptions.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  R The number of rows in the matrix
     * @tparam  C The number of columns in the matrix
     * @param   value The value to fill the matrix with (type deduced automatically)
     * @return  constexpr auto A matrix<Ty, R, C> object with all elements initialized to the specified value
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    The noexcept specification is conditional:
     *          - noexcept if the construction Ty{value} is noexcept
     *          - Throws if the conversion from value to Ty can throw
     * @note    Uses the comma operator with I to discard the index and produce static_cast<Ty>(value) for each element
     * @note    The expression (void)I, static_cast<Ty>(value) creates R * C copies of the converted value at compile time
     * @note    The total number of elements generated is R * C
     *
     * @warning The value type must be convertible to Ty; otherwise, a compilation error occurs
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    constexpr auto full( const auto& value ) noexcept( std::is_nothrow_constructible_v<Ty , decltype( value )> ) {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( ( (void)I, static_cast<Ty>( value ) ) ... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Creates an identity matrix
     *
     *          This function template constructs an N x N identity matrix where all diagonal
     *          elements are set to 1 and all off-diagonal elements are set to 0. It uses
     *          compile-time index sequence expansion to set each diagonal element to Ty{1}
     *          in a zero-initialized matrix, ensuring zero runtime overhead.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the square matrix (both rows and columns)
     * @return  constexpr auto An N x N identity matrix<Ty, N, N> object
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    The matrix is first zero-initialized, then diagonal elements are set to 1
     * @note    Uses fold expression with comma operator to set diagonal elements: (result.data[I][I] = Ty{1}), ...
     * @note    The expression sets result.data[0][0], result.data[1][1], ..., result.data[N-1][N-1] to 1
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto eye( ) noexcept {
        matrix_template<Ty , N , N> result{ };
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( (result.data[ I ][ I ] = Ty{ 1 } ), ... );
        }( std::make_index_sequence<N>( ) );
        return result;
    }

    /**
     * @brief   Creates an identity matrix
     *
     *          This function template constructs an N x N identity matrix where all diagonal
     *          elements are set to 1 and all off-diagonal elements are set to 0. It uses
     *          compile-time index sequence expansion to set each diagonal element to Ty{1}
     *          in a zero-initialized matrix, ensuring zero runtime overhead.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the square matrix (both rows and columns)
     * @return  constexpr auto An N x N identity matrix<Ty, N, N> object
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    The matrix is first zero-initialized, then diagonal elements are set to 1
     * @note    Uses fold expression with comma operator to set diagonal elements: (result.data[I][I] = Ty{1}), ...
     * @note    The expression sets result.data[0][0], result.data[1][1], ..., result.data[N-1][N-1] to 1
     * @note    This function is semantically identical to eye() and is provided as an alias
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto identity( ) noexcept {
        matrix_template<Ty , N , N> result{ };
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( (result.data[ I ][ I ] = Ty{ 1 } ), ... );
        }( std::make_index_sequence<N>( ) );
        return result;
    }

    /**
     * @brief   Creates a diagonal matrix from a vector
     *
     *          This function template constructs an N x N diagonal matrix where the diagonal
     *          elements are taken from the input vector, and all off-diagonal elements are
     *          set to 0. It uses compile-time index sequence expansion to copy each vector
     *          element to the corresponding diagonal position, ensuring zero runtime overhead.
     *
     * @tparam  Ty The arithmetic type of matrix and vector elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the square matrix and the size of the input vector
     * @param   data The input vector containing the diagonal elements
     * @return  constexpr auto An N x N diagonal matrix<Ty, N, N> object
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    The matrix is first zero-initialized, then diagonal elements are set from the vector
     * @note    Uses fold expression with comma operator to set diagonal elements: (result.data[I][I] = data.data[I]), ...
     * @note    The expression sets result.data[0][0] = data.data[0], result.data[1][1] = data.data[1], ...
     * @note    The input vector is not modified by this function
     *
     * @warning The vector must have exactly N elements; this is enforced by the type system
     */
    template <arithmetic_t Ty , std::size_t N>
    constexpr auto diag( const vector_template<Ty , N>& data ) noexcept {
        matrix_template<Ty , N , N> result{ };
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( (result.data[ I ][ I ] = data.data[ I ] ) , ... );
        }( std::make_index_sequence<N>( ) );
        return result;
    }

    /**
     * @brief   Creates a diagonal matrix from a vector with type conversion
     *
     *          This function template constructs an N x N diagonal matrix where the diagonal
     *          elements are taken from the input vector and converted to the matrix element
     *          type Ty. All off-diagonal elements are set to 0. It uses compile-time index
     *          sequence expansion to copy and convert each vector element to the corresponding
     *          diagonal position, ensuring zero runtime overhead.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  Tv The arithmetic type of vector elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the square matrix and the size of the input vector
     * @param   data The input vector containing the diagonal elements of type Tv
     * @return  constexpr auto An N x N diagonal matrix<Ty, N, N> object
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    The matrix is first zero-initialized, then diagonal elements are set from the vector
     * @note    Uses fold expression with comma operator to set diagonal elements with explicit conversion
     * @note    The expression sets result.data[I][I] = static_cast<Ty>(data.data[I]) for each I
     * @note    The input vector is not modified by this function
     *
     * @warning The vector must have exactly N elements; this is enforced by the type system
     * @warning The conversion from Tv to Ty must be valid; otherwise, a compilation error occurs
     */
    template <arithmetic_t Ty , arithmetic_t Tv , std::size_t N>
    constexpr auto diag( const vector_template<Tv , N>& data ) noexcept {
        matrix_template<Ty , N , N> result{ };
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( (result.data[ I ][ I ] = static_cast<Ty>( data.data[ I ] ) ), ... );
        }( std::make_index_sequence<N>( ) );
        return result;
    }

    /**
     * @brief   Creates a diagonal matrix from a list of values
     *
     *          This function template constructs a diagonal matrix where the diagonal elements
     *          are taken from the variadic argument list. It forwards the arguments to construct
     *          a vector, then delegates to the diag function that creates a diagonal matrix
     *          from a vector. This provides a convenient syntax for creating diagonal matrices
     *          directly from individual values.
     *
     * @param   args The values to be placed on the diagonal of the matrix
     * @return  constexpr auto A diagonal matrix with the specified values on the diagonal
     *
     * @note    This function is constexpr and can be evaluated at compile time
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    Forwards all arguments to construct a vector, then calls diag(const vector<Ty, N>&)
     * @note    The number of arguments determines the size of the matrix
     * @note    All arguments must be convertible to a common type for vector construction
     *
     * @warning The number of arguments determines the matrix dimension; at least one argument is required
     * @warning All arguments must be of the same type or convertible to a common type
     */
    template <arithmetic_t Ty>
    constexpr auto diag( auto&&...args ) noexcept {
        return diag( vector_template<Ty , sizeof...(args)>( args... ) );
    }

    /**
     * @brief   Creates a vector with random integer values
     *
     *          This function template generates a vector of random integers using a uniform
     *          distribution. It uses compile-time index sequence expansion to generate N
     *          random values, each produced by the random number generator. The random
     *          device is used for seeding to ensure different sequences on each run.
     *
     * @tparam  Ty The arithmetic type of vector elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the vector
     * @param   max The maximum value of the random range (inclusive)
     * @param   min The minimum value of the random range (inclusive)
     * @return  auto A vector<Ty, N> with random integer values in the range [min, max]
     *
     * @note    This function is NOT constexpr as it uses runtime random number generation
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    Uses std::random_device for seeding and std::mt19937 as the generator
     * @note    Uses std::uniform_int_distribution for integer distribution
     * @note    The random number generator is created and seeded on each call
     * @note    The distribution is defined with integer type (default is int)
     *
     * @warning The max and min parameters are swapped in the distribution constructor
     *          (max is passed as the first argument, min as the second)
     * @warning This function generates only integer values due to uniform_int_distribution
     * @warning The random device and generator are constructed on each call, which may
     *          be slow for frequent generation
     */
    template <arithmetic_t Ty , std::size_t N>
    auto rand( const auto& max , const auto& min ) noexcept {
        std::random_device randomSeed;
        std::mt19937 generator( randomSeed( ) );
        std::uniform_int_distribution<> distribution( max , min );
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( ( (void)I , distribution( generator ) ) ... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Creates a matrix with random integer values
     *
     *          This function template generates a matrix of random integers using a uniform
     *          distribution. It uses compile-time index sequence expansion to generate R * C
     *          random values, each produced by the random number generator. The random
     *          device is used for seeding to ensure different sequences on each run.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  R The number of rows in the matrix
     * @tparam  C The number of columns in the matrix
     * @param   max The maximum value of the random range (inclusive)
     * @param   min The minimum value of the random range (inclusive)
     * @return  auto A matrix<Ty, R, C> with random integer values in the range [min, max]
     *
     * @note    This function is NOT constexpr as it uses runtime random number generation
     * @note    Marked as noexcept, guaranteed not to throw exceptions
     * @note    Uses std::random_device for seeding and std::mt19937 as the generator
     * @note    Uses std::uniform_int_distribution for integer distribution
     * @note    The random number generator is created and seeded on each call
     * @note    The distribution is defined with integer type (default is int)
     * @note    The total number of random values generated is R * C
     *
     * @warning The max and min parameters are swapped in the distribution constructor
     *          (max is passed as the first argument, min as the second)
     * @warning This function generates only integer values due to uniform_int_distribution
     * @warning The random device and generator are constructed on each call, which may
     *          be slow for frequent generation
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    auto rand( const auto& max , const auto& min ) noexcept {
        std::random_device randomSeed;
        std::mt19937 generator( randomSeed( ) );
        std::uniform_int_distribution<> distribution( max , min );
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( ( (void)I , distribution( generator ) ) ... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Creates a vector by repeatedly invoking a generator function
     *
     *          This function template constructs a vector by calling a user-provided generator
     *          function N times, with each invocation producing one element. It uses compile-time
     *          index sequence expansion to generate N values, ensuring zero runtime overhead
     *          for the index handling. The generator is forwarded as a universal reference.
     *
     * @tparam  Ty The arithmetic type of vector elements, constrained by arithmetic_t concept
     * @tparam  N The dimension of the vector
     * @param   generator A callable object that returns values convertible to Ty
     * @return  auto A vector<Ty, N> with elements generated by the provided generator
     *
     * @note    This function is NOT constexpr as it may involve runtime generation
     * @note    Marked as noexcept, guaranteed not to throw exceptions (assuming generator does not throw)
     * @note    The generator is captured by reference in the lambda
     * @note    The generator is invoked N times using (void)I, generator() pattern
     * @note    The index parameter I is discarded using the comma operator
     *
     * @warning The generator must return a value convertible to Ty
     * @warning The generator is invoked N times; ensure it has no side effects if undesired
     */
    template <arithmetic_t Ty , std::size_t N>
    auto rand( auto&& generator ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( ( (void)I , generator( ) ) ... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Creates a matrix by repeatedly invoking a generator function
     *
     *          This function template constructs a matrix by calling a user-provided generator
     *          function R * C times, with each invocation producing one element. It uses
     *          compile-time index sequence expansion to generate R * C values, ensuring zero
     *          runtime overhead for the index handling. The generator is forwarded as a
     *          universal reference.
     *
     * @tparam  Ty The arithmetic type of matrix elements, constrained by arithmetic_t concept
     * @tparam  R The number of rows in the matrix
     * @tparam  C The number of columns in the matrix
     * @param   generator A callable object that returns values convertible to Ty
     * @return  auto A matrix<Ty, R, C> with elements generated by the provided generator
     *
     * @note    This function is NOT constexpr as it may involve runtime generation
     * @note    Marked as noexcept, guaranteed not to throw exceptions (assuming generator does not throw)
     * @note    The generator is captured by reference in the lambda
     * @note    The generator is invoked R * C times using (void)I, generator() pattern
     * @note    The index parameter I is discarded using the comma operator
     * @note    The total number of generator invocations is R * C
     *
     * @warning The generator must return a value convertible to Ty
     * @warning The generator is invoked R * C times; ensure it has no side effects if undesired
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    auto rand( auto&& generator ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( ( (void)I , generator( ) ) ... );
        }( std::make_index_sequence<R * C>( ) );
    }
}

#endif