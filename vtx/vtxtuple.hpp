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
 * @file      vtxtuple.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     Tuple-like container for composable vertex input types.
 *
 * @defgroup  vtx_util Utilities
 * @ingroup   vtx
 * @brief     Vertex tuple container for Vulkan vertex input assembly.
 */

#ifndef VERTEX_TUPLE_HPP
#define VERTEX_TUPLE_HPP
#pragma once

#include "vtxtraits.hpp"    ///< vertex_traits<T>
#include <vector>           ///< std::vector
#include <tuple>            ///< std::tuple_element_t, std::index_sequence

namespace vtx {
    /**
     * @brief Describes a single vertex attribute for Vulkan vertex input binding.
     *
     * @details
     * This structure contains all information needed to create a
     * VkVertexInputAttributeDescription for a vertex attribute.
     *
     * Fields correspond to:
     * - location:    Shader binding location (location = N in shader)
     * - vkformat:    Vulkan VkFormat enum value (numeric)
     * - count:       Number of elements (1 for scalar, 2/3/4 for vectors,
     *                columns for matrices)
     * - offset:      Byte offset from the start of the vertex struct
     * - offset_step: Byte stride between consecutive elements (for matrices)
     *
     * Example:
     * @code
     * vertex_layout layout{
     *     .location    = 0,
     *     .vkformat    = 106,  // VK_FORMAT_R32G32B32_SFLOAT
     *     .count       = 1,
     *     .offset      = 0,
     *     .offset_step = 12
     * };
     * @endcode
     */
    struct vertex_layout{
        std::uint32_t location;
        std::uint32_t vkformat;
        std::uint32_t count;
        std::uint32_t offset;
        std::uint32_t offset_step;
    };

    /**
     * @brief   Primary template declaration for vertex tuple container.
     * @tparam  I Starting index for element access.
     * @tparam  Ty Variadic list of element types.
     *
     * @details
     * This is the recursive tuple-like container for composable vertex types.
     * Each element is wrapped in vertex_element<I, Ty> to provide type-safe
     * access by index.
     *
     * The recursive inheritance structure:
     * vertex_template<I, T0, T1, T2...>
     *   : vertex_element<I, T0>
     *   , vertex_template<I+1, T1, T2...>
     */
    template <std::size_t I , typename... Ty>
    class vertex_template;

    /**
     * @brief   Type-tagged wrapper for a single vertex element.
     * @tparam  Tag Unique compile-time index for this element.
     * @tparam  Ty Underlying type of the element (arithmetic, vector, or matrix).
     *
     * @details
     * Wraps a value of type Ty with a compile-time tag (index) to enable
     * type-safe element access in the vertex tuple hierarchy.
     *
     * Provides:
     * - Value initialization and assignment from compatible types
     * - Move and copy semantics
     * - Private storage accessible only via get() accessors
     *
     * @note This class is intended to be used as a base class in the
     *       recursive vertex_template inheritance chain.
     *
     * @see vertex_template
     */
    template <std::size_t Tag , typename Ty>
    class vertex_element {
    public:

        /// @brief Default constructor - zero-initializes the wrapped value.
        constexpr vertex_element( ) noexcept = default;

        /**
         * @brief   Constructs a vertex element from a const lvalue.
         * @tparam  U Type of the source value.
         * @param   value Value to be stored (converted to Ty).
         */
        template <typename U>
        explicit constexpr vertex_element( const U& value ) noexcept
            : my_value ( static_cast< Ty >( std::forward<U>( value ) ) )
        {}

        /**
         * @brief   Constructs a vertex element from an rvalue.
         * @tparam  U Type of the source value.
         * @param   value Value to be moved and stored (converted to Ty).
         */
        template <typename U>
        explicit constexpr vertex_element( U&& value ) noexcept
            : my_value { static_cast< Ty >( std::forward<U>( value ) ) }
        {}

        /**
         * @brief   Copy assigns from another vertex_element (different tag/type).
         * @tparam  I Tag of the source element.
         * @tparam  U Type of the source element.
         * @param   elem Source element to copy from.
         * @return  Reference to this element.
         */
        template <std::size_t I , typename U>
        constexpr vertex_element<Tag , Ty>& operator=( const vertex_element<I , U>& elem ) noexcept{
            this->my_value = elem.my_value;
            return *this;
        }

        /**
         * @brief   Move assigns from another vertex_element (different tag/type).
         * @tparam  I Tag of the source element.
         * @tparam  U Type of the source element.
         * @param   elem Source element to move from.
         * @return  Reference to this element.
         */
        template <size_t I , typename U>
        constexpr vertex_element<Tag , Ty>& operator=( vertex_element<I , U>&& elem ) noexcept {
            this->my_value = std::move( elem.my_value );
            return *this;
        }

        template <std::size_t , typename...>
        friend class vertex_template;
    private:
        Ty my_value{};
    };

    /**
     * @brief   Base case specialization of vertex_template for a single element.
     * @tparam  I Index of the single element.
     * @tparam  Ty Type of the single element.
     *
     * @note Static_assert in get() ensures index is correct.
     */
    template <size_t I , typename Ty>
    class vertex_template<I , Ty> {
    public:
        /// @brief Default constructor - zero-initializes the stored element.
        constexpr vertex_template() noexcept = default;

        /**
         * @brief   Copy constructor.
         * @param   other Source vertex_template to copy from.
         */
        constexpr vertex_template( const vertex_template& other ) noexcept
        : my_element( other.my_element )
        {}

        /**
         * @brief   Move constructor.
         * @tparam  U Type of the source vertex_template.
         * @param   other Source vertex_template to move from.
         */
        constexpr vertex_template( vertex_template&& other ) noexcept
        : my_element( std::move( other.my_element ) )
        {}

        constexpr vertex_template& operator=( const vertex_template& other ) noexcept= default;
        constexpr vertex_template& operator=( vertex_template&& other ) noexcept = default;

        /**
         * @brief   Constructs a single-element vertex template from a const lvalue.
         * @tparam  U Type of the argument.
         * @param   arg Value to be stored.
         */
        template <typename U>
        explicit constexpr vertex_template( const U& arg ) noexcept
        : my_element( arg )
        {}

        /**
         * @brief   Constructs a single-element vertex template from an rvalue.
         * @tparam  U Type of the argument.
         * @param   arg Value to be moved and stored.
         */
        template <typename U>
        explicit constexpr vertex_template( U&& arg ) noexcept
        : my_element( std::forward<U>( arg ) )
        {}

        /**
         * @brief   Assigns from a const lvalue.
         * @tparam  U Type of the argument.
         * @param   arg Value to be stored.
         * @return  Reference to this vertex_template.
         */
        template <typename U>
        constexpr vertex_template& operator=( const U& arg ) noexcept{
            this->my_element = arg;
            return *this;
        }

        /**
         * @brief   Assigns from an rvalue.
         * @tparam  U Type of the argument.
         * @param   arg Value to be moved and stored.
         * @return  Reference to this vertex_template.
         */
        template <typename U>
        constexpr vertex_template& operator=( U&& arg ) noexcept{
            this->my_element = std::forward<U>( arg );
            return *this;
        }

        /**
         * @brief   Returns a reference to the element at index Idx (non-const lvalue).
         * @tparam  Idx Index of the element to access (must equal I).
         * @return  Reference to the stored value.
         * @note    Compile-time assertion ensures index correctness.
         */
        template <std::size_t Idx>
        constexpr auto& get( ) & noexcept{
            static_assert( Idx == I , "Index out of range" );
            return my_element.my_value;
        }

        /**
         * @brief   Returns a const reference to the element at index Idx (const lvalue).
         * @tparam  Idx Index of the element to access (must equal I).
         * @return  Const reference to the stored value.
         * @note    Compile-time assertion ensures index correctness.
         */
        template <std::size_t Idx>
        constexpr const auto& get( ) const& noexcept{
            static_assert( Idx == I , "Index out of range" );
            return my_element.my_value;
        }
    private:
        vertex_element<I , Ty> my_element {};
    };

    /**
     * @brief   Recursive specialization of vertex_template for multiple elements.
     * @tparam  I Starting index for this level.
     * @tparam  ThisTy Type of the current element.
     * @tparam  NextTy Type of the next element.
     * @tparam  RestTy Remaining element types.
     */
    template <std::size_t I , typename ThisTy , typename NextTy , typename... RestTy>
    class vertex_template<I , ThisTy , NextTy , RestTy...>
        : private vertex_element<I , ThisTy>
        , private vertex_template<I + 1 , NextTy , RestTy...> {
    public:
        using value_type = ThisTy;
        using basic_type = vertex_template<I + 1 , NextTy , RestTy...>;
        static constexpr size_t type_count = 2 + sizeof...( RestTy );

        /// @brief Default constructor - zero-initializes all elements.
        constexpr vertex_template( ) noexcept = default;

        /**
         * @brief   Constructs a multi-element vertex from variadic arguments.
         * @tparam  Ty Type of the first argument.
         * @tparam  Ny Type of the second argument.
         * @tparam  Ry Types of the remaining arguments.
         * @param   arg0 Value for the first element (ThisTy).
         * @param   arg1 Value for the second element (NextTy).
         * @param   args Values for the remaining elements (RestTy...).
         *
         * @details
         * Forwards arguments to the appropriate base classes:
         * - First argument goes to vertex_element<I, ThisTy>
         * - Remaining arguments go to vertex_template<I+1, NextTy, RestTy...>
         */
        template <typename Ty , typename Ny , typename ...Ry>
        constexpr vertex_template( Ty&& arg0 , Ny&& arg1 , Ry&&... args ) noexcept
        : vertex_element<I , ThisTy>( std::forward<Ty>( arg0 ) ) , basic_type ( std::forward<Ny>( arg1 ), std::forward<Ry>( args )... )
        {}

        /**
         * @brief   Returns a reference to the element at index Idx (non-const lvalue).
         * @tparam  Idx Index of the element to access.
         * @return  Reference to the stored value.
         *
         * @details
         * Recursively searches through the inheritance chain:
         * - If Idx == I, return the current element
         * - Otherwise, delegate to the base class (remaining elements)
         */
        template <size_t Idx>
        constexpr auto& get( ) & noexcept{
            if constexpr ( Idx == I ){
                return static_cast< vertex_element<I , ThisTy>& >( *this ).my_value;
            }else{
                return basic_type::template get<Idx>( );
            }
        }

        /**
         * @brief   Returns a const reference to the element at index Idx (const lvalue).
         * @tparam  Idx Index of the element to access.
         * @return  Const reference to the stored value.
         *
         * @details
         * Recursively searches through the inheritance chain:
         * - If Idx == I, return the current element
         * - Otherwise, delegate to the base class (remaining elements)
         */
        template <size_t Idx>
        constexpr const auto& get( ) const& noexcept{
            if constexpr ( Idx == I ){
                return static_cast< const vertex_element<I , ThisTy>& >( *this ).my_value;
            }else{
                return static_cast< const basic_type& >( *this ).template get<Idx>( );
            }
        }
    };

    /**
     * @brief   Alias for vertex_template starting at index 0.
     * @tparam  Ty Variadic list of element types.
     *
     * @details
     * This is the primary user-facing type for creating vertex tuples.
     * Elements are indexed starting from 0, matching the shader location.
     *
     * Example:
     * @code
     * using MyVertex = vertex<vec3f, vec3f, vec2f>;
     * MyVertex v{pos, normal, uv};
     * auto& pos = get<0>(v);
     * @endcode
     */
    template <typename... Ty>
    using vertex = vertex_template<0 , Ty...>;

    /**
     * @brief   Computes byte offsets of each element in a vertex tuple.
     * @tparam  Ty Element types of the vertex.
     *
     * @details
     * Uses pointer arithmetic to compute the offset of each element
     * relative to the start of the vertex struct.
     */
    template <typename ...Ty>
    struct compute_element_offset{

        /**
         * @brief   Computes offsets from a given vertex instance.
         * @param   vertexs Vertex instance to compute offsets from.
         * @return  Vector of byte offsets for each element.
         */
        std::vector<std::size_t> call( const vertex<Ty...>& vertexs ) noexcept{
            std::vector<std::size_t> offsets( vertex<Ty...>::type_count );
            [ & ]<std::size_t... I>( std::index_sequence<I...> ){
                ( ( offsets [ I ] = reinterpret_cast< const char* >( &get<I>( vertexs ) ) -
                                    reinterpret_cast< const char* >( &vertexs ) ) , ... );
            }( std::index_sequence_for<Ty...>( ) );
            return offsets;
        }

        /**
         * @brief   Computes offsets using a default-constructed vertex.
         * @return  Vector of byte offsets for each element.
         */
        std::vector<size_t> call( ) noexcept{
            return this->call( vertex<Ty...>( ) );
        }
    };

    /**
     * @brief   Returns a reference to the element at index I (non-const).
     * @tparam  I Index of the element to access.
     * @tparam  Ty Element types of the vertex.
     * @param   vt Vertex tuple to access.
     * @return  Reference to the stored value.
     *
     * @details
     * This is the primary user-facing accessor for vertex tuples.
     * It forwards to the internal get() method of vertex_template.
     *
     * Example:
     * @code
     * using Vertex = vertex<vec3f, vec3f, vec2f>;
     * Vertex v{pos, normal, uv};
     * auto& pos = get<0>(v);
     * @endcode
     */
    template <size_t I , typename... Ty>
    constexpr auto& get( vertex<Ty...>& vt ) noexcept{
        return vt.template get<I>( );
    }

    /**
     * @brief   Returns a const reference to the element at index I (const).
     * @tparam  I Index of the element to access.
     * @tparam  Ty Element types of the vertex.
     * @param   vt Vertex tuple to access.
     * @return  Const reference to the stored value.
     */
    template <size_t I , typename... Ty>
    constexpr const auto& get( const vertex<Ty...>& vt ) noexcept{
        return vt.template get<I>( );
    }

    /**
     * @brief   Computes byte offsets of each element from a given vertex instance.
     * @tparam  Ty Element types of the vertex.
     * @param   vt Vertex instance to compute offsets from.
     * @return  Vector of byte offsets for each element.
     *
     * @details
     * Convenience wrapper around compute_element_offset::call().
     *
     * Example:
     * @code
     * using Vertex = vertex<vec3f, vec3f, vec2f>;
     * Vertex v{pos, normal, uv};
     * auto offsets = compute_offset(v);
     * // offsets[0] = 0, offsets[1] = 12, offsets[2] = 24
     * @endcode
     */
    template <typename... Ty>
    std::vector<std::size_t> compute_offset( const vertex<Ty...>& vt ) noexcept{
        return compute_element_offset<Ty...>{}.call( vt );
    }

    /**
     * @brief   Computes byte offsets using a default-constructed vertex.
     * @tparam  Ty Element types of the vertex.
     * @return  Vector of byte offsets for each element.
     *
     * @details
     * Convenience wrapper for compile-time offset computation.
     * Uses a temporary default-constructed vertex instance.
     *
     * Example:
     * @code
     * using Vertex = vertex<vec3f, vec3f, vec2f>;
     * auto offsets = compute_offset<vec3f, vec3f, vec2f>();
     * @endcode
     */
    template <typename... Ty>
    std::vector<std::size_t> compute_offset( ) noexcept{
        return compute_element_offset<Ty...>{}.call( );
    }

    /**
     * @brief   Generates Vulkan vertex input layout descriptions.
     * @tparam  Ty Element types of the vertex.
     * @return  Vector of vertex_layout structures for each vertex attribute.
     *
     * @details
     * Creates a complete Vulkan vertex input layout by:
     * 1. Computing byte offsets of each element
     * 2. Querying vertex_traits for format and count information
     * 3. Assigning sequential shader locations
     *
     * The returned layout can be used to populate:
     * - VkVertexInputBindingDescription
     * - VkVertexInputAttributeDescription
     *
     * Example:
     * @code
     * using Vertex = vertex<vec3f, vec3f, vec2f>;
     * auto layout = MakeVertexLayout<vec3f, vec3f, vec2f>();
     * // layout[0]: location=0, format=106 (R32G32B32_SFLOAT), offset=0
     * // layout[1]: location=1, format=106 (R32G32B32_SFLOAT), offset=12
     * // layout[2]: location=2, format=103 (R32G32_SFLOAT), offset=24
     * @endcode
     *
     * @note    Each unique type gets a consecutive location starting from 0.
     * @see     vertex_layout
     * @see     vertex_traits
     */
    template <typename... Ty>
    std::vector<vertex_layout> MakeVertexLayout( ) noexcept{
        auto offsets = compute_offset<Ty...>( );
        using Types = std::tuple<Ty...>;
        std::vector<vertex_layout> layouts( sizeof...( Ty ) );
        std::uint32_t running_loc = 0;
        [ & ]<std::size_t... I>( std::index_sequence<I...> ){
            ( ( layouts [ I ] = vertex_layout {
                .location       = running_loc,
                .vkformat       = vertex_traits<std::tuple_element_t<I, Types>>::format,
                .count          = vertex_traits<std::tuple_element_t<I, Types>>::count,
                .offset         = static_cast<std::uint32_t>( offsets [ I ] ),
                .offset_step    = vertex_traits<std::tuple_element_t<I, Types>>::offset_step,
            } ,
            running_loc += layouts [ I ].count ) , ... );
        }( std::index_sequence_for<Ty...>( ) );
        return layouts;
    }
}

#endif //VERTEX_TUPLE_HPP
