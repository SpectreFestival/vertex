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
 * @file      vtxmatcxr.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     R x C dimensional matrix template with compile-time index generation.
 *
 * @defgroup  vtx_matrix Matrices
 * @ingroup   vtx
 * @brief     Generic matrix template for arbitrary rows and columns.
 *
 * @details
 * This header provides:
 * - matrix_template<T, R, C> - R-row, C-column matrix with random-access iterator
 * - Column-major storage layout (compatible with OpenGL/Vulkan)
 * - Element-wise arithmetic operators (+, -, *, /)
 * - Compound assignment operators (+=, -=, *=, /=)
 * - Matrix multiplication: (L×M) * (M×R) → (L×R)
 * - Matrix-vector multiplication: (R×C) * (C) → (R)
 * - Vector-matrix multiplication: (R) * (R×C) → (C)
 * - Comparison operators returning matrix_template<bool, R, C>
 * - row() / col()          - extract row/column vectors
 * - transpose()            - matrix transposition
 * - trace()                - matrix trace (square matrices only)
 * - determinant()          - 2x2, 3x3, 4x4 determinants
 * - inverse()              - 2x2, 3x3, 4x4 matrix inversion
 * - inverse_transpose()    - inverse-transpose (for normal transformation)
 * - hadamard()             - element-wise product
 * - outer_product()        - vector outer product
 * - translate() / rotate() / scale()    - 4x4 affine transformations
 * - look_at() / perspective() / ortho() - 4x4 view/projection matrices
 *
 * @note  All operations are compile-time unrolled via index_sequence
 *        and expanded using fold expressions (C++17).
 * @note  Storage is column-major: data[col][row], accessible via operator()(row, col).
 */
#ifndef VERTEX_MATRIXCXR_HPP
#define VERTEX_MATRIXCXR_HPP
#pragma once

#include "vtxbasic.hpp"      ///< arithmetic_t, VERTEX_CONSTEXPR, vtx type aliases
#include "vtxvecx.hpp"       ///< vector_template<T, N>, dot(), cross(), normalize()
#include <initializer_list>  ///< std::initializer_list
#include <iterator>          ///< std::reverse_iterator
#include <utility>           ///< std::index_sequence, std::make_index_sequence

namespace vtx {
    /**
     * @brief   R x C dimensional matrix template.
     * @tparam  Ty Arithmetic type (integral or floating-point).
     * @tparam  R Number of rows.
     * @tparam  C Number of columns.
     *
     * @details
     * Provides random-access container semantics with compile-time fixed size.
     * Storage is column-major: data[col][row], compatible with OpenGL/Vulkan.
     * Element access via operator()(row, col) returns data[col][row].
     *
     * All element-wise operations are unrolled at compile time via index_sequence.
     *
     * Example:
     * @code
     * using mat4x4f = matrix_template<float, 4, 4>;
     * mat4x4f m{
     *     1.0f,  2.0f,  3.0f,  4.0f,
     *     5.0f,  6.0f,  7.0f,  8.0f,
     *     9.0f, 10.0f, 11.0f, 12.0f,
     *    13.0f, 14.0f, 15.0f, 16.0f
     * };
     * auto col0 = col(m, 0);
     * auto row0 = row(m, 0);
     * auto trace = trace(m);
     * auto inv = inverse(m);
     * @endcode
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    struct alignas( vertex_alignment<Ty> ) matrix_template {
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
        constexpr matrix_template( ) noexcept = default;

        /**
         * @brief   Returns an iterator to the first element.
         * @return  Iterator to the beginning of the matrix.
         */
        [[nodiscard]] constexpr iterator begin( ) noexcept {
            return std::addressof( data[ 0 ][ 0 ] );
        }

        /**
         * @brief   Returns an iterator to the element following the last element.
         * @return  Iterator to the end of the matrix.
         */
        [[nodiscard]] constexpr iterator end( ) noexcept {
            return std::addressof( data[ 0 ][ 0 ] ) + ( R * C );
        }

        /**
         * @brief   Returns a const iterator to the first element.
         * @return  Const iterator to the beginning of the matrix.
         */
        [[nodiscard]] constexpr const_iterator begin( ) const noexcept {
            return std::addressof( data[ 0 ][ 0 ] );
        }

        /**
         * @brief   Returns a const iterator to the element following the last element.
         * @return  Const iterator to the end of the matrix.
         */
        [[nodiscard]] constexpr const_iterator end( ) const noexcept {
            return std::addressof( data[ 0 ][ 0 ] ) + ( R * C );
        }

        /**
         * @brief   Returns a const iterator to the first element.
         * @return  Const iterator to the beginning of the matrix.
         */
        [[nodiscard]] constexpr const_iterator cbegin( ) const noexcept {
            return std::addressof( data[ 0 ][ 0 ] );
        }

        /**
         * @brief   Returns a const iterator to the element following the last element.
         * @return  Const iterator to the end of the matrix.
         */
        [[nodiscard]] constexpr const_iterator cend( ) const noexcept {
            return std::addressof( data[ 0 ][ 0 ] ) + ( R * C );
        }

        /**
         * @brief   Returns a reverse iterator to the first element of the reversed matrix.
         * @return  Reverse iterator to the beginning of the reversed matrix.
         */
        [[nodiscard]] constexpr reverse_iterator rbegin( ) noexcept {
            return std::reverse_iterator( end( ) );
        }

        /**
         * @brief   Returns a reverse iterator to the element following the last element of the reversed matrix.
         * @return  Reverse iterator to the end of the reversed matrix.
         */
        [[nodiscard]] constexpr reverse_iterator rend( ) noexcept {
            return std::reverse_iterator( begin( ) );
        }

        /**
         * @brief   Returns a const reverse iterator to the first element of the reversed matrix.
         * @return  Const reverse iterator to the beginning of the reversed matrix.
         */
        [[nodiscard]] constexpr const_reverse_iterator rbegin( ) const noexcept {
            return std::reverse_iterator( end( ) );
        }

        /**
         * @brief   Returns a const reverse iterator to the element following the last element of the reversed matrix.
         * @return  Const reverse iterator to the end of the reversed matrix.
         */
        [[nodiscard]] constexpr const_reverse_iterator rend( ) const noexcept {
            return std::reverse_iterator( begin( ) );
        }

        /**
         * @brief   Returns a const reverse iterator to the first element of the reversed matrix.
         * @return  Const reverse iterator to the beginning of the reversed matrix.
         */
        [[nodiscard]] constexpr const_reverse_iterator crbegin( ) const noexcept {
            return std::reverse_iterator( cend( ) );
        }

        /**
         * @brief   Returns a const reverse iterator to the element following the last element of the reversed matrix.
         * @return  Const reverse iterator to the end of the reversed matrix.
         */
        [[nodiscard]] constexpr const_reverse_iterator crend( ) const noexcept {
            return std::reverse_iterator( cbegin( ) );
        }

        /**
         * @brief   Constructs matrix from variadic arguments.
         * @tparam  Args Variadic argument types.
         * @param   args Element values in column-major order.
         *
         * @details
         * Example:
         * @code
         * matrix_template<float, 2, 2> m{1.0f, 2.0f, 3.0f, 4.0f};
         * // data[0][0]=1, data[1][0]=2, data[0][1]=3, data[1][1]=4
         * //    m(0 ,0)=1,    m(1 ,0)=2,    m(0 ,1)=3,    m(1 ,1)=4
         * @endcode
         */
        template <typename... Args>
        explicit constexpr matrix_template( Args&&... args ) noexcept {
            const Ty vals [ R * C ] = { static_cast< Ty >( args )... };
            [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
                ( ( data [ I / R ][ I % R ] = vals [ ( I % R ) * C + ( I / R ) ] ) , ... );
            }( std::make_index_sequence<R * C>( ) );
        }

        /**
         * @brief   Constructs matrix from an initializer list.
         * @param   mat Initializer list of element values in column-major order.
         *
         * @details
         * Example:
         * @code
         * matrix_template<float, 2, 2> m{{1.0f, 2.0f, 3.0f, 4.0f}};
         * // data[0][0]=1, data[1][0]=2, data[0][1]=3, data[1][1]=4
         * @endcode
         */
        constexpr matrix_template( const std::initializer_list<Ty>& mat ) noexcept {
            [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
                ( ( data [ I / R ][ I % R ] = mat.begin( ) [ ( I % R ) * C + ( I / R ) ] ) , ... );
            }( std::make_index_sequence<R * C>( ) );
        }
        
        /**
         * @brief   Accesses element at (row, col).
         * @param   r Row index.
         * @param   c Column index.
         * @return  Reference to the element at data[c][r].
         * @details
         * Maps to column-major storage: returns data[c][r].
         */
        constexpr Ty& operator( )( const std::size_t r, const std::size_t c ) noexcept {
            return data[ c ][ r ];
        }

        /**
         * @brief   Accesses element at (row, col) (const version).
         * @param   r Row index.
         * @param   c Column index.
         * @return  Const reference to the element at data[c][r].
         */
        constexpr const Ty& operator( )( const std::size_t r, const std::size_t c ) const noexcept {
            return data[ c ][ r ];
        }

        /**
         * @brief   Converts matrix to another arithmetic type.
         * @tparam  U Target arithmetic type.
         * @return  matrix_template<U, R, C> with each element casted to type U.
         *
         * @details
         * Example:
         * @code
         * matrix_template<float, 2, 2> mf{1.0f, 2.0f, 3.0f, 4.0f};
         * auto mi = static_cast<matrix_template<int, 2, 2>>(mf);
         * @endcode
         */
        template <typename U>
        explicit constexpr operator matrix_template<U , R , C>( ) const noexcept {
            return [ & ]<size_t... I>( std::index_sequence<I...> ) {
                return matrix_template<U , R , C>( ( static_cast< U >( data [ I % C ][ I / C ] ) )... );
            }( std::make_index_sequence<R * C>( ) );
        }

        Ty data [ C ][ R ] {};
    };

    /**
     * @brief   Unary plus operator.
     * @param   mat Input matrix.
     * @return  Copy of the matrix (element-wise unary plus).
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator+( const matrix_template<Ty , R , C>& mat ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( ( + mat.data [ I % C ][ I / C ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Unary negation operator.
     * @param   mat Input matrix.
     * @return  Matrix with each element negated.
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator-( const matrix_template<Ty , R , C>& mat ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( ( - mat.data [ I % C ][ I / C ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Element-wise addition of two matrices.
     * @param   mat1 Left-hand side matrix.
     * @param   mat2 Right-hand side matrix.
     * @return  matrix_template with each element computed as mat1[i] + mat2[i].
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator+( const matrix_template<Ty , R , C>& mat1 , const matrix_template<Ty , R , C>& mat2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( ( mat1.data [ I % C ][ I / C ] + mat2.data [ I % C ][ I / C ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Element-wise subtraction of two matrices.
     * @param   mat1 Left-hand side matrix.
     * @param   mat2 Right-hand side matrix.
     * @return  matrix_template with each element computed as mat1[i] - mat2[i].
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator-( const matrix_template<Ty , R , C>& mat1 , const matrix_template<Ty , R , C>& mat2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( ( mat1.data [ I % C ][ I / C ] - mat2.data [ I % C ][ I / C ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Matrix multiplication: (L x M) * (M x R) → (L x R).
     * @tparam  Ty Element type.
     * @tparam  L Rows of mat1.
     * @tparam  M Columns of mat1 / Rows of mat2.
     * @tparam  R Columns of mat2.
     * @param   mat1 Left-hand side matrix (L x M).
     * @param   mat2 Right-hand side matrix (M x R).
     * @return  Result matrix (L x R).
     *
     * @details
     * Each element (row, col) is computed as dot product of row from mat1
     * and column from mat2. All loops are unrolled at compile time.
     *
     * Example:
     * @code
     * matrix_template<float, 2, 3> a{...};
     * matrix_template<float, 3, 4> b{...};
     * auto c = a * b;  // matrix_template<float, 2, 4>
     * @endcode
     */
    template<arithmetic_t Ty, std::size_t L, std::size_t M, std::size_t R>
    [[nodiscard]] constexpr auto operator*( const matrix_template<Ty , L , M>& mat1 , const matrix_template<Ty , M , R>& mat2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , L , R>(
                ( [ & ] ( ) -> Ty {
                    constexpr std::size_t row = I / R;
                    constexpr std::size_t col = I % R;
                    return [ & ]<std::size_t... K>( std::index_sequence<K...> ) {
                        return ( ( mat1.data [ K ][ row ] * mat2.data [ col ][ K ] ) + ... );
                    }( std::make_index_sequence<M>( ) );
            }( ) )... );
        }( std::make_index_sequence<L * R>( ) );
    }

    /**
     * @brief   Element-wise addition compound assignment.
     * @param   mat1 Left-hand side matrix (modified in-place).
     * @param   mat2 Right-hand side matrix.
     * @return  Reference to mat1.
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    constexpr auto& operator+=( matrix_template<Ty , R , C>& mat1 , const matrix_template<Ty , R , C>& mat2 ) noexcept {
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( ( mat1.data [ I / R ][ I % R ] += mat2.data [ I / R ][ I % R ] ) , ... );
        }( std::make_index_sequence<R * C>( ) );
        return mat1;
    }

    /**
     * @brief   Element-wise subtraction compound assignment.
     * @param   mat1 Left-hand side matrix (modified in-place).
     * @param   mat2 Right-hand side matrix.
     * @return  Reference to mat1.
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    constexpr auto& operator-=( matrix_template<Ty , R , C>& mat1 , const matrix_template<Ty , R , C>& mat2 ) noexcept {
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( ( mat1.data [ I / R ][ I % R ] -= mat2.data [ I / R ][ I % R ] ) , ... );
        }( std::make_index_sequence<R * C>( ) );
        return mat1;
    }

    /**
     * @brief   Matrix multiplication compound assignment (right-multiply by square matrix).
     * @tparam  Ty Element type.
     * @tparam  R Rows of mat1.
     * @tparam  C Columns of mat1 / dimension of square mat2.
     * @param   mat1 Left-hand side matrix (modified in-place, R x C).
     * @param   mat2 Right-hand side square matrix (C x C).
     * @return  Reference to mat1.
     *
     * @details
     * Performs: mat1 = mat1 * mat2
     * Only supports right-multiplication by a square matrix of dimension C.
     *
     * Example:
     * @code
     * matrix_template<float, 3, 3> a{...};
     * matrix_template<float, 3, 3> b{...};
     * a *= b;  // a = a * b
     * @endcode
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    constexpr auto& operator*=( matrix_template<Ty , R , C>& mat1 , const matrix_template<Ty , C , C>& mat2 ) noexcept {
        return ( mat1 = mat1 * mat2, mat1);
    }

    /**
     * @brief   Scalar multiplication (matrix * scalar).
     * @tparam  Ty Matrix element type.
     * @tparam  Sy Scalar type (arithmetic).
     * @tparam  R Number of rows.
     * @tparam  C Number of columns.
     * @param   mat Input matrix.
     * @param   scale Scalar multiplier.
     * @return  Matrix with each element multiplied by scale.
     */
    template <arithmetic_t Ty , arithmetic_t Sy , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator*( const matrix_template<Ty , R , C>& mat , const Sy& scale ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( ( mat.data [ I % C ][ I / C ] * static_cast< Ty >( scale ) )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Scalar division (matrix / scalar).
     * @tparam  Ty Matrix element type.
     * @tparam  Ts Scalar type (arithmetic).
     * @tparam  R Number of rows.
     * @tparam  C Number of columns.
     * @param   mat Input matrix (dividend).
     * @param   scale Scalar divisor.
     * @return  Matrix with each element divided by scale.
     * @note    Division by zero is undefined behavior.
     */
    template <arithmetic_t Ty , arithmetic_t Ts , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator/( const matrix_template<Ty , R , C>& mat , const Ts& scale ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( ( mat.data [ I % C ][ I / C ] / static_cast< Ty >( scale ) )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Scalar multiplication compound assignment (matrix *= scalar).
     * @tparam  Ty Matrix element type.
     * @tparam  Ts Scalar type (arithmetic).
     * @tparam  R Number of rows.
     * @tparam  C Number of columns.
     * @param   mat Matrix to be modified in-place.
     * @param   scale Scalar multiplier.
     * @return  Reference to mat.
     */
    template <arithmetic_t Ty , arithmetic_t Ts , std::size_t R , std::size_t C>
    constexpr auto& operator*=( matrix_template<Ty , R , C>& mat , const Ts& scale ) noexcept {
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( ( mat.data [ I / R ][ I % R ] *= static_cast< Ty >( scale ) ) , ... );
        }( std::make_index_sequence<R * C>( ) );
        return mat;
    }

    /**
     * @brief   Scalar division compound assignment (matrix /= scalar).
     * @tparam  Ty Matrix element type.
     * @tparam  Ts Scalar type (arithmetic).
     * @tparam  R Number of rows.
     * @tparam  C Number of columns.
     * @param   mat Matrix to be modified in-place (dividend).
     * @param   scale Scalar divisor.
     * @return  Reference to mat.
     * @note    Division by zero is undefined behavior.
     */
    template <arithmetic_t Ty , arithmetic_t Ts , std::size_t R , std::size_t C>
    constexpr auto& operator/=( matrix_template<Ty , R , C>& mat , const Ts& scale ) noexcept {
        [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            ( ( mat.data [ I / R ][ I % R ] /= static_cast< Ty >( scale ) ) , ... );
        }( std::make_index_sequence<R * C>( ) );
        return mat;
    }

    /**
     * @brief   Matrix-vector multiplication: (R x C) * (C) → (R).
     * @tparam  Ty Element type.
     * @tparam  R Number of rows.
     * @tparam  C Number of columns.
     * @param   mat Input matrix (R x C).
     * @param   vec Input vector (C).
     * @return  Result vector (R).
     *
     * @details
     * Each element of the result is the dot product of a row of the matrix
     * and the input vector. All loops are unrolled at compile time.
     *
     * Example:
     * @code
     * matrix_template<float, 3, 3> m{...};
     * vec3f v{1.0f, 2.0f, 3.0f};
     * auto result = m * v;  // vec3f
     * @endcode
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator*( const matrix_template<Ty , R , C>& mat , const vector_template<Ty , C>& vec ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            const auto dot = [ & ] ( std::size_t row ) {
                return [ & ]<std::size_t... K>( std::index_sequence<K...> ) {
                    return ( ( mat.data [ K ][ row ] * vec.data [ K ] ) + ... );
                }( std::make_index_sequence<C>( ) );
            };
            return vector_template<Ty , R>( ( dot( I ) )... );
        }( std::make_index_sequence<R>{} );
    }

    /**
     * @brief   Vector-matrix multiplication: (R) * (R x C) → (C).
     * @tparam  Ty Element type.
     * @tparam  R Number of rows.
     * @tparam  C Number of columns.
     * @param   vec Input vector (R).
     * @param   mat Input matrix (R x C).
     * @return  Result vector (C).
     *
     * @details
     * Each element of the result is the dot product of the input vector
     * and a column of the matrix. All loops are unrolled at compile time.
     *
     * Example:
     * @code
     * vec3f v{1.0f, 2.0f, 3.0f};
     * matrix_template<float, 3, 3> m{...};
     * auto result = v * m;  // vec3f
     * @endcode
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator*( const vector_template<Ty , R>& vec , const matrix_template<Ty , R , C>& mat ) noexcept {
        return [ & ]<std::size_t... J>( std::index_sequence<J...> ) {
            const auto dot = [ & ] ( std::size_t col ) {
                return [ & ]<std::size_t... K>( std::index_sequence<K...> ) {
                        return ( ( vec.data [ K ] * mat.data [ col ][ K ] ) + ... );
                    }( std::make_index_sequence<R>( ) );
                };
            return vector_template<Ty , C>( ( dot( J ) )... );
        }( std::make_index_sequence<C>( ) );
    }

    /**
     * @brief   Element-wise equality comparison.
     * @param   m1 Left-hand side matrix.
     * @param   m2 Right-hand side matrix.
     * @return  matrix_template<bool, R, C> with each element computed as m1[i] == m2[i].
     */
    template <arithmetic_t Ty, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr auto operator==( const matrix_template<Ty , R , C>& m1 , const matrix_template<Ty , R , C>& m2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<bool , R , C>( ( m1.data [ I % R ][ I / R ] == m2.data [ I % R ][ I / R ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Element-wise inequality comparison.
     * @param   m1 Left-hand side matrix.
     * @param   m2 Right-hand side matrix.
     * @return  matrix_template<bool, R, C> with each element computed as m1[i] != m2[i].
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator!=( const matrix_template<Ty , R , C>& m1 , const matrix_template<Ty , R , C>& m2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<bool , R , C>( ( m1.data [ I % R ][ I / R ] != m2.data [ I % R ][ I / R ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Element-wise greater-than-or-equal comparison.
     * @param   m1 Left-hand side matrix.
     * @param   m2 Right-hand side matrix.
     * @return  matrix_template<bool, R, C> with each element computed as m1[i] >= m2[i].
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator>=( const matrix_template<Ty , R , C>& m1 , const matrix_template<Ty , R , C>& m2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<bool , R , C>( ( m1.data [ I % R ][ I / R ] >= m2.data [ I % R ][ I / R ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Element-wise less-than-or-equal comparison.
     * @param   m1 Left-hand side matrix.
     * @param   m2 Right-hand side matrix.
     * @return  matrix_template<bool, R, C> with each element computed as m1[i] <= m2[i].
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator<=( const matrix_template<Ty , R , C>& m1 , const matrix_template<Ty , R , C>& m2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<bool , R , C>( ( m1.data [ I % R ][ I / R ] <= m2.data [ I % R ][ I / R ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Element-wise greater-than comparison.
     * @param   m1 Left-hand side matrix.
     * @param   m2 Right-hand side matrix.
     * @return  matrix_template<bool, R, C> with each element computed as m1[i] > m2[i].
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator>( const matrix_template<Ty , R , C>& m1 , const matrix_template<Ty , R , C>& m2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<bool , R , C>( ( m1.data [ I % R ][ I / R ] > m2.data [ I % R ][ I / R ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Element-wise less-than comparison.
     * @param   m1 Left-hand side matrix.
     * @param   m2 Right-hand side matrix.
     * @return  matrix_template<bool, R, C> with each element computed as m1[i] < m2[i].
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator<( const matrix_template<Ty , R , C>& m1 , const matrix_template<Ty , R , C>& m2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<bool , R , C>( ( m1.data [ I % R ][ I / R ] < m2.data [ I % R ][ I / R ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Element-wise logical AND.
     * @param   m1 Left-hand side matrix.
     * @param   m2 Right-hand side matrix.
     * @return  matrix_template<bool, R, C> with each element computed as m1[i] && m2[i].
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator&&( const matrix_template<Ty , R , C>& m1 , const matrix_template<Ty , R , C>& m2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<bool , R , C>( ( m1.data [ I % R ][ I / R ] && m2.data [ I % R ][ I / R ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Element-wise logical OR.
     * @param   m1 Left-hand side matrix.
     * @param   m2 Right-hand side matrix.
     * @return  matrix_template<bool, R, C> with each element computed as m1[i] || m2[i].
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto operator||( const matrix_template<Ty , R , C>& m1 , const matrix_template<Ty , R , C>& m2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<bool , R , C>( ( m1.data [ I % R ][ I / R ] || m2.data [ I % R ][ I / R ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Extracts a row from the matrix as a vector.
     * @tparam  Ty Element type.
     * @tparam  R Number of rows.
     * @tparam  C Number of columns.
     * @param   mat Input matrix.
     * @param   idx Row index (0 to R-1).
     * @return  vector_template<Ty, C> containing the row elements.
     *
     * @details
     * Example:
     * @code
     * matrix_template<float, 3, 3> m{...};
     * auto row0 = row(m, 0);  // vec3f
     * @endcode
     */
    template <arithmetic_t Ty, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr auto row( const matrix_template<Ty , R , C>& mat , std::size_t idx ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , C>( ( mat.data [ I ][ idx ] )... );
        }( std::make_index_sequence<C>( ) );
    }

    /**
     * @brief   Extracts a column from the matrix as a vector.
     * @tparam  Ty Element type.
     * @tparam  R Number of rows.
     * @tparam  C Number of columns.
     * @param   mat Input matrix.
     * @param   idx Column index (0 to C-1).
     * @return  vector_template<Ty, R> containing the column elements.
     *
     * @details
     * Example:
     * @code
     * matrix_template<float, 3, 3> m{...};
     * auto col0 = col(m, 0);  // vec3f
     * @endcode
     */
    template <arithmetic_t Ty, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr auto col( const matrix_template<Ty , R , C>& mat , std::size_t idx ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , R>( ( mat.data [ idx ][ I ] )... );
        }( std::make_index_sequence<R>( ) );
    }

    /**
     * @brief   Transposes a matrix (R x C → C x R).
     * @tparam  Ty Element type.
     * @tparam  R Number of rows.
     * @tparam  C Number of columns.
     * @param   mat Input matrix.
     * @return  Transposed matrix of type matrix_template<Ty, C, R>.
     *
     * @details
     * Example:
     * @code
     * matrix_template<float, 2, 3> m{...};
     * auto mt = transpose(m);  // matrix_template<float, 3, 2>
     * @endcode
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto transpose( const matrix_template<Ty , R , C>& mat ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , C , R>( ( mat.data [ I / R ][ I % R ] )... );
        }( std::make_index_sequence<C * R>( ) );
    }

    /**
     * @brief   Computes the trace of a square matrix (sum of diagonal elements).
     * @tparam  Ty Element type.
     * @tparam  S Matrix dimension (S x S).
     * @param   mat Input square matrix.
     * @return  Trace value (sum of diagonal elements).
     *
     * @details
     * Example:
     * @code
     * matrix_template<float, 3, 3> m{...};
     * auto tr = trace(m);  // m(0,0) + m(1,1) + m(2,2)
     * @endcode
     */
    template <arithmetic_t Ty , std::size_t S>
    [[nodiscard]] constexpr auto trace( const matrix_template<Ty , S , S>& mat ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return ( ( mat.data [ I ][ I ] ) + ... );
        }( std::make_index_sequence<S>( ) );
    }

    /**
     * @brief   Computes the Hadamard (element-wise) product of two matrices.
     * @tparam  T Element type.
     * @tparam  R Number of rows.
     * @tparam  C Number of columns.
     * @param   mat1 Left-hand side matrix.
     * @param   mat2 Right-hand side matrix.
     * @return  matrix_template<T, R, C> with each element computed as mat1[i][j] * mat2[i][j].
     *
     * @details
     * Also known as the Schur product or entrywise product.
     * Unlike matrix multiplication, this is an element-wise operation.
     *
     * Example:
     * @code
     * matrix_template<float, 2, 2> a{{1, 2}, {3, 4}};
     * matrix_template<float, 2, 2> b{{5, 6}, {7, 8}};
     * auto c = hadamard(a, b);  // {{5, 12}, {21, 32}}
     * @endcode
     */
    template <arithmetic_t T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr auto hadamard( const matrix_template<T , R , C>& mat1 , const matrix_template<T , R , C>& mat2 ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<T , R , C>( ( mat1.data [ I % C ][ I / C ] * mat2.data [ I % C ][ I / C ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Computes the outer product of two vectors.
     * @tparam  Ty Element type.
     * @tparam  R Dimension of the first vector (rows of resulting matrix).
     * @tparam  C Dimension of the second vector (columns of resulting matrix).
     * @param   c Column vector (R x 1).
     * @param   r Row vector (1 x C).
     * @return  matrix_template<Ty, R, C> where result[i][j] = c[i] * r[j].
     *
     * @details
     * The outer product of two vectors produces a matrix:
     * (R x 1) * (1 x C) = (R x C)
     *
     * Example:
     * @code
     * vec3f c{1.0f, 2.0f, 3.0f};
     * vec2f r{4.0f, 5.0f};
     * auto m = outer_product(c, r);  // matrix_template<float, 3, 2>
     * // m(0,0)=4,  m(0,1)=5
     * // m(1,0)=8,  m(1,1)=10
     * // m(2,0)=12, m(2,1)=15
     * @endcode
     */
    template <arithmetic_t Ty , std::size_t R , std::size_t C>
    [[nodiscard]] constexpr auto outer_product( const vector_template<Ty , R>& c , const vector_template<Ty , C>& r ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return matrix_template<Ty , R , C>( ( c.data [ I / C ] * r.data [ I % C ] )... );
        }( std::make_index_sequence<R * C>( ) );
    }

    /**
     * @brief   Computes the determinant of a 2x2 matrix.
     * @tparam  Ty Element type.
     * @param   mat Input 2x2 matrix.
     * @return  Determinant: a*d - b*c
     *
     * @details
     * For matrix [a b; c d], determinant = a*d - b*c.
     */
    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto determinant( const matrix_template<Ty , 2 , 2>& mat ) noexcept{
        return
            mat.data [ 0 ][ 0 ] * mat.data [ 1 ][ 1 ] -
            mat.data [ 1 ][ 0 ] * mat.data [ 0 ][ 1 ];
    }

    /**
     * @brief   Computes the determinant of a 3x3 matrix.
     * @tparam  Ty Element type.
     * @param   mat Input 3x3 matrix.
     * @return  Determinant via Laplace expansion along the first row.
     *
     * @details
     * Uses Sarrus' rule / Laplace expansion along the first row:
     * det = a(ei - fh) - b(di - fg) + c(dh - eg)
     */
    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto determinant( const matrix_template<Ty , 3 , 3>& mat ) noexcept{
        return
            mat.data [ 0 ][ 0 ] * ( mat.data [ 1 ][ 1 ] * mat.data [ 2 ][ 2 ] - mat.data [ 2 ][ 1 ] * mat.data [ 1 ][ 2 ] ) -
            mat.data [ 0 ][ 1 ] * ( mat.data [ 1 ][ 0 ] * mat.data [ 2 ][ 2 ] - mat.data [ 2 ][ 0 ] * mat.data [ 1 ][ 2 ] ) +
            mat.data [ 0 ][ 2 ] * ( mat.data [ 1 ][ 0 ] * mat.data [ 2 ][ 1 ] - mat.data [ 2 ][ 0 ] * mat.data [ 1 ][ 1 ] );
    }

    /**
     * @brief   Computes the determinant of a 4x4 matrix.
     * @tparam  Ty Element type.
     * @param   mat Input 4x4 matrix.
     * @return  Determinant via Laplace expansion along the first row.
     *
     * @details
     * Uses Laplace expansion along the first row with 3x3 minor determinants.
     * All 3x3 determinants are computed inline without function call overhead.
     */
    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto determinant( const matrix_template<Ty , 4 , 4>& mat ) noexcept{
        return
            mat.data [ 0 ][ 0 ] * (
            mat.data [ 1 ][ 1 ] * mat.data [ 2 ][ 2 ] * mat.data [ 3 ][ 3 ] +
            mat.data [ 1 ][ 2 ] * mat.data [ 2 ][ 3 ] * mat.data [ 3 ][ 1 ] +
            mat.data [ 1 ][ 3 ] * mat.data [ 2 ][ 1 ] * mat.data [ 3 ][ 2 ] -
            mat.data [ 1 ][ 3 ] * mat.data [ 2 ][ 2 ] * mat.data [ 3 ][ 1 ] -
            mat.data [ 1 ][ 2 ] * mat.data [ 2 ][ 1 ] * mat.data [ 3 ][ 3 ] -
            mat.data [ 1 ][ 1 ] * mat.data [ 2 ][ 3 ] * mat.data [ 3 ][ 2 ] ) +
            mat.data [ 0 ][ 1 ] * (
            mat.data [ 1 ][ 3 ] * mat.data [ 2 ][ 2 ] * mat.data [ 3 ][ 0 ] +
            mat.data [ 1 ][ 2 ] * mat.data [ 2 ][ 0 ] * mat.data [ 3 ][ 3 ] +
            mat.data [ 1 ][ 0 ] * mat.data [ 2 ][ 3 ] * mat.data [ 3 ][ 2 ] -
            mat.data [ 1 ][ 0 ] * mat.data [ 2 ][ 2 ] * mat.data [ 3 ][ 3 ] -
            mat.data [ 1 ][ 2 ] * mat.data [ 2 ][ 3 ] * mat.data [ 3 ][ 0 ] -
            mat.data [ 1 ][ 3 ] * mat.data [ 2 ][ 0 ] * mat.data [ 3 ][ 2 ] ) +
            mat.data [ 0 ][ 2 ] * (
            mat.data [ 1 ][ 0 ] * mat.data [ 2 ][ 1 ] * mat.data [ 3 ][ 3 ] +
            mat.data [ 1 ][ 1 ] * mat.data [ 2 ][ 3 ] * mat.data [ 3 ][ 0 ] +
            mat.data [ 1 ][ 3 ] * mat.data [ 2 ][ 0 ] * mat.data [ 3 ][ 1 ] -
            mat.data [ 1 ][ 3 ] * mat.data [ 2 ][ 1 ] * mat.data [ 3 ][ 0 ] -
            mat.data [ 1 ][ 1 ] * mat.data [ 2 ][ 0 ] * mat.data [ 3 ][ 3 ] -
            mat.data [ 1 ][ 0 ] * mat.data [ 2 ][ 3 ] * mat.data [ 3 ][ 1 ] ) +
            mat.data [ 0 ][ 3 ] * (
            mat.data [ 1 ][ 2 ] * mat.data [ 2 ][ 1 ] * mat.data [ 3 ][ 0 ] +
            mat.data [ 1 ][ 1 ] * mat.data [ 2 ][ 0 ] * mat.data [ 3 ][ 2 ] +
            mat.data [ 1 ][ 0 ] * mat.data [ 2 ][ 2 ] * mat.data [ 3 ][ 1 ] -
            mat.data [ 1 ][ 0 ] * mat.data [ 2 ][ 1 ] * mat.data [ 3 ][ 2 ] -
            mat.data [ 1 ][ 1 ] * mat.data [ 2 ][ 2 ] * mat.data [ 3 ][ 0 ] -
            mat.data [ 1 ][ 2 ] * mat.data [ 2 ][ 0 ] * mat.data [ 3 ][ 1 ] );
    }

    /**
     * @brief   Computes the inverse of a 2x2 matrix.
     * @tparam  Ty Element type.
     * @param   mat Input 2x2 matrix.
     * @return  Inverse matrix.
     *
     * @details
     * For matrix [a b; c d], inverse = (1/det) * [d  -b; -c  a]
     *
     * @note If determinant is zero, the matrix is singular and division by zero occurs.
     *       Caller should check determinant before calling this function.
     */
    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto inverse( const matrix_template<Ty , 2 , 2>& mat ) noexcept{
        auto dt = determinant<Ty>( mat );
        return matrix_template<Ty , 2 , 2>(
            mat.data [ 1 ][ 1 ] / dt , -mat.data [ 1 ][ 0 ] / dt ,
           -mat.data [ 0 ][ 1 ] / dt ,  mat.data [ 0 ][ 0 ] / dt
        );
    }

    /**
     * @brief   Computes the inverse of a 3x3 matrix using the adjugate method.
     * @tparam  Ty Element type.
     * @param   mat Input 3x3 matrix.
     * @return  Inverse matrix.
     *
     * @details
     * Computes the inverse via the adjugate matrix: A⁻¹ = adj(A) / det(A).
     * The adjugate is the transpose of the cofactor matrix.
     *
     * @note If determinant is zero, the matrix is singular and division by zero occurs.
     *       Caller should check determinant before calling this function.
     */
    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto inverse( const matrix_template<Ty , 3 , 3>& mat ) noexcept{
        auto c00 = +( mat.data [ 1 ][ 1 ] * mat.data [ 2 ][ 2 ] - mat.data [ 1 ][ 2 ] * mat.data [ 2 ][ 1 ] );
        auto c01 = -( mat.data [ 1 ][ 0 ] * mat.data [ 2 ][ 2 ] - mat.data [ 1 ][ 2 ] * mat.data [ 2 ][ 0 ] );
        auto c02 = +( mat.data [ 1 ][ 0 ] * mat.data [ 2 ][ 1 ] - mat.data [ 1 ][ 1 ] * mat.data [ 2 ][ 0 ] );
        auto c10 = -( mat.data [ 0 ][ 1 ] * mat.data [ 2 ][ 2 ] - mat.data [ 0 ][ 2 ] * mat.data [ 2 ][ 1 ] );
        auto c11 = +( mat.data [ 0 ][ 0 ] * mat.data [ 2 ][ 2 ] - mat.data [ 0 ][ 2 ] * mat.data [ 2 ][ 0 ] );
        auto c12 = -( mat.data [ 0 ][ 0 ] * mat.data [ 2 ][ 1 ] - mat.data [ 0 ][ 1 ] * mat.data [ 2 ][ 0 ] );
        auto c20 = +( mat.data [ 0 ][ 1 ] * mat.data [ 1 ][ 2 ] - mat.data [ 0 ][ 2 ] * mat.data [ 1 ][ 1 ] );
        auto c21 = -( mat.data [ 0 ][ 0 ] * mat.data [ 1 ][ 2 ] - mat.data [ 0 ][ 2 ] * mat.data [ 1 ][ 0 ] );
        auto c22 = +( mat.data [ 0 ][ 0 ] * mat.data [ 1 ][ 1 ] - mat.data [ 0 ][ 1 ] * mat.data [ 1 ][ 0 ] );

        auto det = mat.data [ 0 ][ 0 ] * c00 +mat.data [ 0 ][ 1 ] * c01 +mat.data [ 0 ][ 2 ] * c02;
        auto inv = static_cast< Ty >( 1 ) / det;

        return matrix_template<Ty , 3 , 3>(
            c00 * inv , c10 * inv , c20 * inv ,
            c01 * inv , c11 * inv , c21 * inv ,
            c02 * inv , c12 * inv , c22 * inv
        );
    }

    /**
     * @brief   Computes the inverse of a 4x4 matrix using the adjugate method with SIMD-friendly optimization.
     * @tparam  Ty Element type.
     * @param   mat Input 4x4 matrix.
     * @return  Inverse matrix.
     *
     * @details
     * Uses the adjugate matrix method with cofactor computation.
     * The algorithm is optimized by grouping cofactors into 4D vectors,
     * which improves cache locality and enables SIMD vectorization.
     *
     * Steps:
     * 1. Compute all 2x2 cofactor determinants
     * 2. Group cofactors into 4D vectors for each row
     * 3. Compute adjugate matrix columns using vector operations
     * 4. Compute determinant and multiply by inverse scale
     *
     * @note If determinant is zero, the matrix is singular and division by zero occurs.
     *       Caller should check determinant before calling this function.
     */
    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto inverse( const matrix_template<Ty , 4 , 4>& mat ) noexcept {
        auto Coef00 = mat.data [ 2 ][ 2 ] * mat.data [ 3 ][ 3 ] - mat.data [ 3 ][ 2 ] * mat.data [ 2 ][ 3 ];
        auto Coef02 = mat.data [ 1 ][ 2 ] * mat.data [ 3 ][ 3 ] - mat.data [ 3 ][ 2 ] * mat.data [ 1 ][ 3 ];
        auto Coef03 = mat.data [ 1 ][ 2 ] * mat.data [ 2 ][ 3 ] - mat.data [ 2 ][ 2 ] * mat.data [ 1 ][ 3 ];
        auto Coef04 = mat.data [ 2 ][ 1 ] * mat.data [ 3 ][ 3 ] - mat.data [ 3 ][ 1 ] * mat.data [ 2 ][ 3 ];
        auto Coef06 = mat.data [ 1 ][ 1 ] * mat.data [ 3 ][ 3 ] - mat.data [ 3 ][ 1 ] * mat.data [ 1 ][ 3 ];
        auto Coef07 = mat.data [ 1 ][ 1 ] * mat.data [ 2 ][ 3 ] - mat.data [ 2 ][ 1 ] * mat.data [ 1 ][ 3 ];
        auto Coef08 = mat.data [ 2 ][ 1 ] * mat.data [ 3 ][ 2 ] - mat.data [ 3 ][ 1 ] * mat.data [ 2 ][ 2 ];
        auto Coef10 = mat.data [ 1 ][ 1 ] * mat.data [ 3 ][ 2 ] - mat.data [ 3 ][ 1 ] * mat.data [ 1 ][ 2 ];
        auto Coef11 = mat.data [ 1 ][ 1 ] * mat.data [ 2 ][ 2 ] - mat.data [ 2 ][ 1 ] * mat.data [ 1 ][ 2 ];
        auto Coef12 = mat.data [ 2 ][ 0 ] * mat.data [ 3 ][ 3 ] - mat.data [ 3 ][ 0 ] * mat.data [ 2 ][ 3 ];
        auto Coef14 = mat.data [ 1 ][ 0 ] * mat.data [ 3 ][ 3 ] - mat.data [ 3 ][ 0 ] * mat.data [ 1 ][ 3 ];
        auto Coef15 = mat.data [ 1 ][ 0 ] * mat.data [ 2 ][ 3 ] - mat.data [ 2 ][ 0 ] * mat.data [ 1 ][ 3 ];
        auto Coef16 = mat.data [ 2 ][ 0 ] * mat.data [ 3 ][ 2 ] - mat.data [ 3 ][ 0 ] * mat.data [ 2 ][ 2 ];
        auto Coef18 = mat.data [ 1 ][ 0 ] * mat.data [ 3 ][ 2 ] - mat.data [ 3 ][ 0 ] * mat.data [ 1 ][ 2 ];
        auto Coef19 = mat.data [ 1 ][ 0 ] * mat.data [ 2 ][ 2 ] - mat.data [ 2 ][ 0 ] * mat.data [ 1 ][ 2 ];
        auto Coef20 = mat.data [ 2 ][ 0 ] * mat.data [ 3 ][ 1 ] - mat.data [ 3 ][ 0 ] * mat.data [ 2 ][ 1 ];
        auto Coef22 = mat.data [ 1 ][ 0 ] * mat.data [ 3 ][ 1 ] - mat.data [ 3 ][ 0 ] * mat.data [ 1 ][ 1 ];
        auto Coef23 = mat.data [ 1 ][ 0 ] * mat.data [ 2 ][ 1 ] - mat.data [ 2 ][ 0 ] * mat.data [ 1 ][ 1 ];

        vector_template<Ty , 4> Fac0( Coef00 , Coef00 , Coef02 , Coef03 );
        vector_template<Ty , 4> Fac1( Coef04 , Coef04 , Coef06 , Coef07 );
        vector_template<Ty , 4> Fac2( Coef08 , Coef08 , Coef10 , Coef11 );
        vector_template<Ty , 4> Fac3( Coef12 , Coef12 , Coef14 , Coef15 );
        vector_template<Ty , 4> Fac4( Coef16 , Coef16 , Coef18 , Coef19 );
        vector_template<Ty , 4> Fac5( Coef20 , Coef20 , Coef22 , Coef23 );

        vector_template<Ty , 4> Vec0( mat.data [ 1 ][ 0 ] , mat.data [ 0 ][ 0 ] , mat.data [ 0 ][ 0 ] , mat.data [ 0 ][ 0 ] );
        vector_template<Ty , 4> Vec1( mat.data [ 1 ][ 1 ] , mat.data [ 0 ][ 1 ] , mat.data [ 0 ][ 1 ] , mat.data [ 0 ][ 1 ] );
        vector_template<Ty , 4> Vec2( mat.data [ 1 ][ 2 ] , mat.data [ 0 ][ 2 ] , mat.data [ 0 ][ 2 ] , mat.data [ 0 ][ 2 ] );
        vector_template<Ty , 4> Vec3( mat.data [ 1 ][ 3 ] , mat.data [ 0 ][ 3 ] , mat.data [ 0 ][ 3 ] , mat.data [ 0 ][ 3 ] );

        auto Inv0 = Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2;
        auto Inv1 = Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4;
        auto Inv2 = Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5;
        auto Inv3 = Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5;

        matrix_template<Ty , 4 , 4> adj ( 
             Inv0 [ 0 ], -Inv1 [ 0 ],  Inv2 [ 0 ], -Inv3 [ 0 ],
            -Inv0 [ 1 ],  Inv1 [ 1 ], -Inv2 [ 1 ],  Inv3 [ 1 ],
             Inv0 [ 2 ], -Inv1 [ 2 ],  Inv2 [ 2 ], -Inv3 [ 2 ],
            -Inv0 [ 3 ],  Inv1 [ 3 ], -Inv2 [ 3 ],  Inv3 [ 3 ]
        );

        auto det = 
            mat.data [ 0 ][ 0 ] * ( +Inv0 [ 0 ] ) + 
            mat.data [ 0 ][ 1 ] * ( -Inv1 [ 0 ] ) + 
            mat.data [ 0 ][ 2 ] * ( +Inv2 [ 0 ] ) + 
            mat.data [ 0 ][ 3 ] * ( -Inv3 [ 0 ] );
        
        return adj * ( static_cast<Ty>(1) / det );
    }

    /**
     * @brief   Creates a translation matrix and multiplies it with the input matrix.
     * @tparam  Ty Element type.
     * @param   mat Input 4x4 matrix to be transformed.
     * @param   vec Translation vector (x, y, z).
     * @return  Matrix with translation applied: mat * translate(vec).
     *
     * @details
     * Constructs a translation matrix T and returns mat * T.
     * Equivalent to translating the coordinate system by vec.
     *
     * Translation matrix:
     * [1  0  0  x]
     * [0  1  0  y]
     * [0  0  1  z]
     * [0  0  0  1]
     *
     * Example:
     * @code
     * mat4x4f model{...};
     * vec3f translation{1.0f, 2.0f, 3.0f};
     * auto translated = translate(model, translation);
     * @endcode
     */
    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto translate( const matrix_template<Ty , 4 , 4>& mat , const vector_template<Ty , 3>& vec ) noexcept {
        return mat * matrix_template<Ty , 4 , 4>(
            Ty { 1 } , Ty { 0 } , Ty { 0 } , vec.data [ 0 ] ,
            Ty { 0 } , Ty { 1 } , Ty { 0 } , vec.data [ 1 ] ,
            Ty { 0 } , Ty { 0 } , Ty { 1 } , vec.data [ 2 ] ,
            Ty { 0 } , Ty { 0 } , Ty { 0 } , Ty { 1 }
        );
    }

    /**
     * @brief   Creates a rotation matrix around an arbitrary axis and multiplies it with the input matrix.
     * @tparam  Ty Element type.
     * @tparam  U Angle type (arithmetic).
     * @param   mat Input 4x4 matrix to be transformed.
     * @param   angle Rotation angle in radians.
     * @param   axis Rotation axis (will be normalized internally).
     * @return  Matrix with rotation applied: mat * rotate(angle, axis).
     *
     * @details
     * Constructs a rotation matrix using Rodrigues' rotation formula and returns mat * R.
     * The axis is automatically normalized before computation.
     *
     * Rodrigues' rotation formula for matrix form:
     * R = I + sin(θ) * [a]× + (1 - cos(θ)) * (a × (a × I))
     *
     * Where [a]× is the skew-symmetric cross-product matrix of axis a.
     *
     * Example:
     * @code
     * mat4x4f model{...};
     * auto rotated = rotate(model, radians(45.0f), vec3f{0.0f, 1.0f, 0.0f});
     * @endcode
     */
    template <arithmetic_t Ty , arithmetic_t U>
    [[nodiscard]] constexpr auto rotate( const matrix_template<Ty , 4 , 4>& mat , const U& angle , const vector_template<Ty , 3>& axis ) noexcept {
        auto axisLength = length<Ty , 3>( axis );

        auto x = axis.data [ 0 ] / axisLength;
        auto y = axis.data [ 1 ] / axisLength;
        auto z = axis.data [ 2 ] / axisLength;

        auto c = std::cos( angle );
        auto s = std::sin( angle );
        auto t = 1 - c;

        auto tx = t * x;
        auto ty = t * y;
        auto tz = t * z;
        auto sx = s * x;
        auto sy = s * y;
        auto sz = s * z;

        auto rotateMatrix = matrix_template<Ty , 4 , 4>(
            tx * x + c  , ty * x - sz , tz * x + sy , Ty { 0 } ,
            tx * y + sz , ty * y + c  , tz * y - sx , Ty { 0 } ,
            tx * z - sy , ty * z + sx , tz * z + c  , Ty { 0 } ,
            Ty { 0 }    , Ty { 0 }    , Ty { 0 }    , Ty { 1 }
        );
        return mat * rotateMatrix;
    }

    /**
     * @brief   Creates a scaling matrix and multiplies it with the input matrix.
     * @tparam  Ty Element type.
     * @param   mat Input 4x4 matrix to be transformed.
     * @param   vec Scale factors for (x, y, z) axes.
     * @return  Matrix with scaling applied: mat * scale(vec).
     *
     * @details
     * Constructs a diagonal scaling matrix S and returns mat * S.
     *
     * Scaling matrix:
     * [sx  0   0   0]
     * [ 0  sy  0   0]
     * [ 0   0  sz  0]
     * [ 0   0   0   1]
     *
     * Example:
     * @code
     * mat4x4f model{...};
     * vec3f scaleFactors{2.0f, 1.5f, 2.0f};
     * auto scaled = scale(model, scaleFactors);
     * @endcode
     */
    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto scale( const matrix_template<Ty , 4 , 4>& mat , const vector_template<Ty , 3>& vec ) noexcept {
        auto scaleMatrix = matrix_template<Ty , 4 , 4> (
            vec.data [ 0 ] ,        Ty{ 0 }        , Ty{ 0 } , Ty{ 0 },
                   Ty{ 0 } , vec.data [ 1 ] ,        Ty{ 0 } , Ty{ 0 },
                   Ty{ 0 } ,        Ty{ 0 } , vec.data [ 2 ] , Ty{ 0 },
                   Ty{ 0 } ,        Ty{ 0 } ,        Ty{ 0 } , Ty{ 1 }
        );
        return mat * scaleMatrix;
    }

    /**
     * @brief   Constructs a view matrix from eye position, target center, and up vector.
     * @tparam  Ty Element type.
     * @param   eye Camera position.
     * @param   center Target point to look at.
     * @param   up World up vector (usually (0, 1, 0)).
     * @return  4x4 view matrix.
     *
     * @details
     * Constructs a view matrix that transforms world space to view space.
     * The matrix is built from orthonormal basis vectors:
     * - forward: direction from eye to center (z-axis, negated)
     * - right:   cross(forward, up)   (x-axis)
     * - trueUp:  cross(right, forward) (y-axis)
     *
     * The result is a 4x4 matrix in column-major order suitable for Vulkan/OpenGL.
     *
     * Example:
     * @code
     * vec3f eye{0.0f, 5.0f, 10.0f};
     * vec3f center{0.0f, 0.0f, 0.0f};
     * vec3f up{0.0f, 1.0f, 0.0f};
     * auto view = look_at(eye, center, up);
     * @endcode
     */
    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto look_at( const vector_template<Ty , 3>& eye , const vector_template<Ty , 3>& center , const vector_template<Ty , 3>& up ) noexcept {
        auto forward = normalize<Ty , 3>( center - eye );
        auto right = normalize<Ty , 3>( cross<Ty>( forward , up ) );
        auto trueUp = cross<Ty>( right , forward );
        auto x03 = -dot<Ty , 3>(   right , eye );
        auto x13 = -dot<Ty , 3>(  trueUp , eye );
        auto x23 =  dot<Ty , 3>( forward , eye );

        return matrix_template<Ty , 4 , 4>(
            right.data [ 0 ] ,    right.data [ 1 ] ,    right.data [ 2 ] , x03 ,
           trueUp.data [ 0 ] ,   trueUp.data [ 1 ] ,   trueUp.data [ 2 ] , x13 ,
         -forward.data [ 0 ] , -forward.data [ 1 ] , -forward.data [ 2 ] , x23 ,
                    Ty { 0 } ,            Ty { 0 } ,            Ty { 0 } , Ty { 1 }
        );
    }

    /**
     * @brief   Constructs a perspective projection matrix.
     * @tparam  Ty Element type (floating-point).
     * @param   fovy Field of view angle in radians (vertical).
     * @param   aspect Aspect ratio (width / height).
     * @param   znear Near clipping plane distance (> 0).
     * @param   zfar Far clipping plane distance (> znear).
     * @return  4x4 perspective projection matrix.
     *
     * @details
     * Creates a perspective projection matrix mapping the view frustum to
     * the canonical clipping volume. Uses the standard OpenGL/Vulkan style
     * where Z is mapped to [0, 1] for Vulkan or [-1, 1] for OpenGL.
     *
     * Matrix form (Vulkan Z range [0, 1]):
     * [  f/aspect      0            0                              0 ]
     * [  0             f            0                              0 ]
     * [  0             0            zfar/(zfar-znear)              1 ]
     * [  0             0           -znear*zfar/(zfar-znear)        0 ]
     *
     * Where f = 1 / tan(fovy / 2)
     *
     * @note For Vulkan-style depth range, use this as-is.
     *       For OpenGL-style depth range [-1, 1], adjust the third row accordingly.
     *
     * Example:
     * @code
     * auto proj = perspective(radians(45.0f), 16.0f/9.0f, 0.1f, 100.0f);
     * @endcode
     */
    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto perspective( const Ty fovy , const Ty aspect , const Ty znear , const Ty zfar ) noexcept {
        auto tmpVar1 = Ty { 1 } / ( std::tan( fovy / Ty { 2 } ) );
        auto tmpVar2 = zfar / ( zfar - znear );
        return matrix_template<Ty , 4 , 4>(
            tmpVar1 / aspect , Ty { 0 } , Ty { 0 } , Ty { 0 } ,
                    Ty { 0 } ,  tmpVar1 , Ty { 0 } , Ty { 0 } ,
                    Ty { 0 } , Ty { 0 } ,  tmpVar2 , Ty { 0 } ,
                    Ty { 0 } , Ty { 0 } , Ty { 1 } , -znear * tmpVar2
        );
    }


    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto ortho( const Ty left , const Ty right , const Ty bottom , const Ty top , const Ty znear , const Ty zfar ) noexcept {
        auto tmp1 = right - left;
        auto tmp2 = top - bottom;
        auto tmp3 = zfar - znear;

        return matrix_template<Ty , 4 , 4>(
            2 / tmp1 , Ty { 0 } , Ty { 0 } , -( right + left ) / tmp1 ,
            Ty { 0 } , 2 / tmp2 , Ty { 0 } , -( top + bottom ) / tmp2 ,
            Ty { 0 } , Ty { 0 } ,-2 / tmp3 , -( zfar + znear ) / tmp3 ,
            Ty { 0 } , Ty { 0 } , Ty { 0 } , Ty { 1 }
        );
    }

    /**
     * @brief   Extracts the diagonal elements of a square matrix as a vector.
     * @tparam  Ty Element type.
     * @tparam  N Matrix dimension (S x S).
     * @param   mat Input square matrix.
     * @return  vector_template<Ty, S> containing the diagonal elements.
     *
     * @details
     * Example:
     * @code
     * matrix_template<float, 3, 3> m{...};
     * auto d = diagonal(m);  // vec3f{ m(0,0), m(1,1), m(2,2) }
     * @endcode
     */
    template <arithmetic_t Ty , std::size_t N>
    [[nodiscard]] constexpr auto diagonal( const matrix_template<Ty , N , N>& mat ) noexcept {
        return [ & ]<std::size_t... I>( std::index_sequence<I...> ) {
            return vector_template<Ty , N>( mat.data [ I ][ I ]... );
        }( std::make_index_sequence<N>( ) );
    }

    /**
     * @brief   Computes the inverse-transpose of a 2x2 matrix.
     * @tparam  Ty Element type.
     * @param   mat Input 2x2 matrix.
     * @return  Inverse-transpose matrix.
     *
     * @details
     * Used for transforming normals when the transformation matrix has non-uniform
     * scaling. Normals must be transformed by the inverse-transpose to remain
     * perpendicular to surfaces.
     *
     * For a matrix M, the normal transformation matrix is (M⁻¹)ᵀ.
     */
    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto inverse_transpose( const matrix_template<Ty , 2 , 2>& mat ) noexcept {
        return transpose<Ty, 2, 2>( inverse<Ty>( mat ) );
    }

    /**
     * @brief   Computes the inverse-transpose of a 3x3 matrix.
     * @tparam  Ty Element type.
     * @param   mat Input 3x3 matrix.
     * @return  Inverse-transpose matrix.
     *
     * @details
     * Used for transforming normals when the transformation matrix has non-uniform
     * scaling. For a 3x3 matrix M representing rotation/scale, normals are
     * transformed by (M⁻¹)ᵀ.
     */
    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto inverse_transpose( const matrix_template<Ty , 3 , 3>& mat ) noexcept {
        return transpose<Ty, 3, 3>( inverse<Ty>( mat ) );
    }

    /**
     * @brief   Computes the inverse-transpose of a 4x4 matrix.
     * @tparam  Ty Element type.
     * @param   mat Input 4x4 matrix.
     * @return  Inverse-transpose of the upper-left 3x3 submatrix.
     *
     * @details
     * For 4x4 affine transformation matrices, only the upper-left 3x3 rotation/scale
     * part is used for normal transformation. The translation component is ignored.
     *
     * Used for transforming normals in 3D rendering with non-uniform scaling.
     * The 4x4 matrix's upper-left 3x3 submatrix is extracted, inverted, transposed,
     * and returned as a 4x4 matrix with identity translation.
     */
    template <arithmetic_t Ty>
    [[nodiscard]] constexpr auto inverse_transpose( const matrix_template<Ty , 4 , 4>& mat ) noexcept {
        return transpose<Ty, 4, 4>( inverse<Ty>( mat ) );
    }
}

#endif //VERTEX_MATRIXCXR_HPP