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
 * @file      vtxquat.hpp
 * @author    SpectreFestival
 * @license   MIT
 * @brief     Quaternion template for 3D rotation representation.
 *
 * @defgroup  vtx_quat Quaternions
 * @ingroup   vtx
 * @brief     Quaternion operations and conversions.
 *
 * @details
 * This header provides:
 * - quat_template<Ty> - quaternion with w, x, y, z components
 * - Arithmetic operators (+, -, *, /) and compound assignments
 * - Quaternion multiplication (Hamilton product)
 * - dot() - quaternion dot product
 * - conjugate() - quaternion conjugate
 * - magnitude() / normalize() - quaternion normalization
 * - inverse() - quaternion inverse
 * - rotate_vector() - rotate a 3D vector using a quaternion
 * - slerp() - spherical linear interpolation
 * - lerp() - linear interpolation (with normalization)
 * - angle_axis() - create quaternion from angle-axis representation
 * - to_matrix() - convert to 3x3 rotation matrix
 * - matrix4_cast() - convert to 4x4 rotation matrix
 * - quat_cast() - convert 3x3 rotation matrix to quaternion
 * 
 * @note Quaternion components are stored as (w, x, y, z).
 *       w is the scalar (real) part, (x, y, z) is the vector (imaginary) part.
 *
 * Example:
 * @code
 * auto q = angle_axis(radians(45.0f), vec3f{0, 1, 0});
 * auto v_rot = rotate_vector(q, vec3f{1, 0, 0});
 * auto m = matrix4_cast(q);
 * @endcode
 */

#ifndef VERTEX_QUAT_HPP
#define VERTEX_QUAT_HPP
#pragma once

#include "vtxbasic.hpp"     ///< arithmetic_t, VERTEX_CONSTEXPR, vtx type aliases
#include "vtxvecx.hpp"      ///< vector_template<T, N>, dot(), normalize(), cross()
#include "vtxmat4x4.hpp"    ///< matrix_template<T, 4, 4>

namespace vtx {
    /**
     * @brief   Quaternion template for 3D rotation representation.
     * @tparam  Ty Arithmetic type (floating-point recommended).
     *
     * @details
     * Quaternion components are stored as (w, x, y, z).
     * - w: scalar (real) part
     * - (x, y, z): vector (imaginary) part
     *
     * For rotation quaternions, w = cos(θ/2) and (x, y, z) = sin(θ/2) * axis.
     *
     * Example:
     * @code
     * quat_template<float> q{1.0f, 0.0f, 0.0f, 0.0f};  // Identity quaternion
     * @endcode
     */
    template <arithmetic_t Ty>
    struct quat_template {
        Ty w{};  ///< Scalar (real) component
        Ty x{};  ///< Vector (imaginary) X component
        Ty y{};  ///< Vector (imaginary) Y component
        Ty z{};  ///< Vector (imaginary) Z component
    };

    /**
     * @brief   Unary plus operator.
     * @param   quat Input quaternion.
     * @return  Copy of the quaternion.
     */
    template <arithmetic_t Ty>
    constexpr quat_template<Ty> operator+( const quat_template<Ty>& quat ) noexcept {
        return quat_template<Ty>(
            +quat.w,
            +quat.x,
            +quat.y,
            +quat.z
        );
    }

    /**
     * @brief   Unary negation operator.
     * @param   quat Input quaternion.
     * @return  Quaternion with all components negated.
     */
    template <arithmetic_t Ty>
    constexpr quat_template<Ty> operator-( const quat_template<Ty>& quat ) noexcept {
        return quat_template<Ty>(
            -quat.w,
            -quat.x,
            -quat.y,
            -quat.z
        );
    }

    /**
     * @brief   Quaternion addition (component-wise).
     * @param   quat1 Left-hand side quaternion.
     * @param   quat2 Right-hand side quaternion.
     * @return  Quaternion with each component computed as quat1[i] + quat2[i].
     */
    template <arithmetic_t Ty>
	constexpr quat_template<Ty> operator+( const quat_template<Ty>& quat1 , const quat_template<Ty>& quat2 ) noexcept {
		return quat_template<Ty>( 
			quat1.w + quat2.w , 
			quat1.x + quat2.x , 
			quat1.y + quat2.y , 
			quat1.z + quat2.z 
		);
	}

    /**
     * @brief   Quaternion subtraction (component-wise).
     * @param   quat1 Left-hand side quaternion.
     * @param   quat2 Right-hand side quaternion.
     * @return  Quaternion with each component computed as quat1[i] - quat2[i].
     */
    template <arithmetic_t Ty>
    constexpr quat_template<Ty> operator-( const quat_template<Ty>& quat1 , const quat_template<Ty>& quat2 ) noexcept {
    	return quat_template<Ty>( 
    		quat1.w - quat2.w , 
    		quat1.x - quat2.x ,  
    		quat1.y - quat2.y , 
    		quat1.z - quat2.z 
    	);
    }
    
    /**
     * @brief   Scalar multiplication (quaternion * scalar).
     * @tparam  Ty Quaternion element type.
     * @tparam  U Scalar type (arithmetic).
     * @param   quat Input quaternion.
     * @param   scale Scalar multiplier.
     * @return  Quaternion with each component multiplied by scale.
     */
    template <arithmetic_t Ty, arithmetic_t U>
    constexpr quat_template<Ty> operator*( const quat_template<Ty>& quat , const U& scale ) noexcept {
    	return quat_template<Ty>( 
    		quat.w * scale , 
    		quat.x * scale , 
    		quat.y * scale , 
    		quat.z * scale 
    	);
    }
    
    /**
     * @brief   Scalar division (quaternion / scalar).
     * @tparam  Ty Quaternion element type.
     * @tparam  U Scalar type (arithmetic).
     * @param   quat Input quaternion (dividend).
     * @param   scale Scalar divisor.
     * @return  Quaternion with each component divided by scale.
     * @note    Division by zero is undefined behavior.
     */
    template <arithmetic_t Ty, arithmetic_t U>
    constexpr quat_template<Ty> operator/( const quat_template<Ty>& quat , const U& scale ) noexcept {
    	return quat_template<Ty>( 
    		quat.w / scale ,
    		quat.x / scale ,
    		quat.y / scale ,
    		quat.z / scale
    	);
    }
    
    /**
     * @brief   Quaternion addition compound assignment.
     * @param   quat1 Left-hand side quaternion (modified in-place).
     * @param   quat2 Right-hand side quaternion.
     * @return  Reference to quat1.
     */
    template <arithmetic_t Ty>
    constexpr quat_template<Ty>& operator+=( quat_template<Ty>& quat1 , const quat_template<Ty>& quat2 ) noexcept {
    	quat1.w += quat2.w;
    	quat1.x += quat2.x;
    	quat1.y += quat2.y; 
    	quat1.z += quat2.z;
    	return quat1;
    }
    
    /**
     * @brief   Quaternion subtraction compound assignment.
     * @param   quat1 Left-hand side quaternion (modified in-place).
     * @param   quat2 Right-hand side quaternion.
     * @return  Reference to quat1.
     */
    template <arithmetic_t Ty>
    constexpr quat_template<Ty>& operator-=( quat_template<Ty>& quat1 , const quat_template<Ty>& quat2 ) noexcept {
    	quat1.w -= quat2.w;
    	quat1.x -= quat2.x;
    	quat1.y -= quat2.y; 
    	quat1.z -= quat2.z;
    	return quat1;
    }
    
    /**
     * @brief   Scalar multiplication compound assignment (quaternion *= scalar).
     * @tparam  Ty Quaternion element type.
     * @tparam  U Scalar type (arithmetic).
     * @param   quat Quaternion to be modified in-place.
     * @param   scale Scalar multiplier.
     * @return  Reference to quat.
     */
    template <arithmetic_t Ty, arithmetic_t U>
    constexpr quat_template<Ty>& operator*=( quat_template<Ty>& quat , const U& scale ) noexcept {
    	quat.w *= scale;
    	quat.x *= scale;
    	quat.y *= scale; 
    	quat.z *= scale;
    	return quat;
    }
    
    /**
     * @brief   Scalar division compound assignment (quaternion /= scalar).
     * @tparam  Ty Quaternion element type.
     * @tparam  U Scalar type (arithmetic).
     * @param   quat Quaternion to be modified in-place (dividend).
     * @param   scale Scalar divisor.
     * @return  Reference to quat.
     * @note    Division by zero is undefined behavior.
     */
    template <arithmetic_t Ty, arithmetic_t U>
    constexpr quat_template<Ty>& operator/=( quat_template<Ty>& quat , const U& scale ) noexcept {
    	quat.w /= scale; 
    	quat.x /= scale; 
    	quat.y /= scale; 
    	quat.z /= scale;
    	return quat;
    }
    
    /**
     * @brief   Quaternion multiplication (Hamilton product).
     * @param   quat1 Left-hand side quaternion.
     * @param   quat2 Right-hand side quaternion.
     * @return  Quaternion product (quat1 * quat2).
     *
     * @details
     * The Hamilton product of two quaternions is used to compose rotations.
     * For unit quaternions, multiplication corresponds to rotation composition.
     *
     * Formula:
     * (w1, v1) * (w2, v2) = (w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2)
     *
     * @note Quaternion multiplication is non-commutative (quat1 * quat2 != quat2 * quat1).
     *       The order represents applying quat2 first, then quat1.
     */
    template <arithmetic_t Ty>
    constexpr quat_template<Ty> operator*( const quat_template<Ty>& quat1 , const quat_template<Ty>& quat2 ) noexcept {
    	return quat_template<Ty>(
    		quat1.w * quat2.w - quat1.x * quat2.x - quat1.y * quat2.y - quat1.z * quat2.z ,
    		quat1.w * quat2.x + quat1.x * quat2.w + quat1.y * quat2.z - quat1.z * quat2.y ,
    		quat1.w * quat2.y - quat1.x * quat2.z + quat1.y * quat2.w + quat1.z * quat2.x ,
    		quat1.w * quat2.z + quat1.x * quat2.y - quat1.y * quat2.x + quat1.z * quat2.w
    	);
    }
    
    /**
     * @brief   Quaternion multiplication compound assignment.
     * @param   quat1 Left-hand side quaternion (modified in-place).
     * @param   quat2 Right-hand side quaternion.
     * @return  Reference to quat1.
     * 
     * @details
     * Performs: quat1 = quat1 * quat2
     * This composes rotations, applying quat2 first, then quat1.
     */
    template <arithmetic_t Ty>
    constexpr quat_template<Ty>& operator*=( quat_template<Ty>& quat1 , const quat_template<Ty>& quat2 ) noexcept {
    	return ( quat1 = quat1 * quat2 , quat1 );
    }

    /**
     * @brief   Computes the dot product of two quaternions.
     * @param   quat1 First quaternion.
     * @param   quat2 Second quaternion.
     * @return  Scalar dot product value.
     *
     * @details
     * The dot product is computed as: w1*w2 + x1*x2 + y1*y2 + z1*z2.
     * For unit quaternions, the dot product represents the cosine of half the angle
     * between the two rotations.
     */
	template <arithmetic_t Ty>
	VERTEX_CONSTEXPR auto dot( const quat_template<Ty>& quat1 , const quat_template<Ty>& quat2 ) noexcept {
    	return
			quat1.w * quat2.w +
			quat1.x * quat2.x +
			quat1.y * quat2.y +
			quat1.z * quat2.z;
    }

    /**
     * @brief   Computes the conjugate of a quaternion.
     * @param   quat Input quaternion.
     * @return  Conjugate quaternion.
     *
     * @details
     * The conjugate of a quaternion is obtained by negating the imaginary (vector) part:
     * conj(q) = (w, -x, -y, -z)
     *
     * For unit quaternions, the conjugate is also the inverse (q⁻¹ = conj(q)).
     * For non-unit quaternions, use inverse() which divides by magnitude squared.
     */
	template <arithmetic_t Ty>
	VERTEX_CONSTEXPR auto conjugate( const quat_template<Ty>& quat ) noexcept {
    	return quat_template<Ty>(
    		+quat.w ,
    		-quat.x ,
    		-quat.y ,
    		-quat.z
    	);
    }

    /**
     * @brief   Computes the magnitude (Euclidean norm) of a quaternion.
     * @param   quat Input quaternion.
     * @return  Magnitude value.
     *
     * @details
     * The magnitude is computed as: sqrt(w² + x² + y² + z²).
     * For unit quaternions (used for rotation), the magnitude is 1.
     */
	template <arithmetic_t Ty>
	VERTEX_CONSTEXPR auto magnitude( const quat_template<Ty>& quat ) noexcept {
    	return std::sqrt(
    		quat.w * quat.w +
    		quat.x * quat.x +
    		quat.y * quat.y +
    		quat.z * quat.z
    	);
    }

    /**
     * @brief   Normalizes a quaternion to unit length.
     * @param   quat Input quaternion.
     * @return  Unit quaternion pointing in the same direction.
     * @note    If the quaternion is zero-length, the result is undefined (division by zero).
     */
    template <arithmetic_t Ty>
    VERTEX_CONSTEXPR auto normalize(const quat_template<Ty>& quat) noexcept {
        return quat / magnitude(quat);
    }

    /**
     * @brief   Computes the inverse of a quaternion.
     * @param   quat Input quaternion.
     * @return  Inverse quaternion.
     *
     * @details
     * For a quaternion q, its inverse is: q⁻¹ = conj(q) / |q|²
     * For unit quaternions, this simplifies to: q⁻¹ = conj(q)
     *
     * @note    If the quaternion is zero-length, the result is undefined (division by zero).
     */
    template <arithmetic_t Ty>
    VERTEX_CONSTEXPR auto inverse(const quat_template<Ty>& quat) noexcept {
        return conjugate<Ty>(quat) / dot<Ty>(quat, quat);
    }

    /**
     * @brief   Rotates a 3D vector using a quaternion.
     * @param   quat Rotation quaternion (should be normalized).
     * @param   vec Input 3D vector.
     * @return  Rotated 3D vector.
     *
     * @details
     * Rotates the vector using the formula: v' = q * v * q⁻¹
     * where v is treated as a pure quaternion (0, x, y, z).
     *
     * The resulting rotation matrix is computed from the quaternion components
     * and multiplied with the vector for efficiency.
     *
     * @note For best performance and accuracy, the quaternion should be normalized.
     */
	template <arithmetic_t Ty>
	VERTEX_CONSTEXPR auto rotate_vector( const quat_template<Ty>& quat , const vector_template<Ty , 3>& vec ) noexcept {
    	auto w2 = quat.w * quat.w;
    	auto x2 = quat.x * quat.x;
    	auto y2 = quat.y * quat.y;
    	auto z2 = quat.z * quat.z;
    	auto xy = quat.x * quat.y;
    	auto xz = quat.x * quat.z;
    	auto yz = quat.y * quat.z;
    	auto wx = quat.w * quat.x;
    	auto wy = quat.w * quat.y;
    	auto wz = quat.w * quat.z;

    	return vector_template<Ty , 3>(
    		( w2 + x2 - y2 - z2 ) * vec.data [ 0 ] + 2 * ( xy - wz ) * vec.data [ 1 ] + 2 * ( xz + wy ) * vec.data [ 2 ] ,
			2 * ( xy + wz ) * vec.data [ 0 ] + ( w2 - x2 + y2 - z2 ) * vec.data [ 1 ] + 2 * ( yz - wx ) * vec.data [ 2 ] ,
			2 * ( xz - wy ) * vec.data [ 0 ] + 2 * ( yz + wx ) * vec.data [ 1 ] + ( w2 - x2 - y2 + z2 ) * vec.data [ 2 ]
		);
    }

    /**
     * @brief   Spherical Linear Interpolation (SLERP) between two quaternions.
     * @tparam  Ty Quaternion element type.
     * @tparam  U Interpolation factor type (arithmetic).
     * @param   quat1 Starting quaternion.
     * @param   quat2 Ending quaternion (taken by value for potential negation).
     * @param   t Interpolation factor in range [0, 1].
     * @return  Interpolated quaternion.
     *
     * @details
     * SLERP provides constant angular velocity interpolation between two rotations.
     * - If dot < 0, negate quat2 to take the shortest path.
     * - If the angle is very small, fall back to normalized linear interpolation (NLERP).
     * - Otherwise, interpolate using spherical trigonometry.
     *
     * The interpolation formula is:
     * slerp(q1, q2, t) = sin((1-t)*θ)/sin(θ) * q1 + sin(t*θ)/sin(θ) * q2
     * where θ = acos(dot(q1, q2)).
     *
     * @note    t should be in the range [0, 1]. Values outside this range may produce
     *          unexpected results.
     */
	template <arithmetic_t Ty , arithmetic_t U>
	VERTEX_CONSTEXPR auto slerp( const quat_template<Ty>& quat1 , quat_template<Ty> quat2 , U t ) noexcept {
    	auto dt = dot<Ty>( quat1 , quat2 );

    	if ( dt < 0 ) {
    		quat2 = -quat2;
    	}

    	if ( dt > 0.999999 ) {
    		return normalize<Ty>( quat1 * ( 1 - t ) + quat2 * t );
    	}

    	auto theta = std::acos( dt );
    	auto sinTheta = std::sin( theta );
    	auto w1Coeff = std::sin( ( 1 - t ) * theta ) / sinTheta;
    	auto w2Coeff = std::sin( t * theta ) / sinTheta;

    	return quat1 * w1Coeff + quat2 * w2Coeff;
    }

    /**
     * @brief   Converts a quaternion to a 3x3 rotation matrix.
     * @tparam  Ty Quaternion element type.
     * @param   quat Input quaternion (should be normalized).
     * @return  3x3 rotation matrix.
     *
     * @details
     * Converts a unit quaternion to its equivalent 3x3 rotation matrix.
     * The matrix is column-major and satisfies: v' = M * v.
     *
     * @note    For best results, the quaternion should be normalized.
     *          The resulting matrix is orthonormal.
     */
	template <arithmetic_t Ty>
	VERTEX_CONSTEXPR auto to_matrix( const quat_template<Ty>& quat ) noexcept {
    	auto w2 = quat.w * quat.w;
    	auto x2 = quat.x * quat.x;
    	auto y2 = quat.y * quat.y;
    	auto z2 = quat.z * quat.z;
    	auto xy = quat.x * quat.y;
    	auto wz = quat.w * quat.z;
    	auto xz = quat.x * quat.z;
    	auto wy = quat.w * quat.y;
    	auto yz = quat.y * quat.z;
    	auto wx = quat.w * quat.x;

    	return matrix_template<Ty , 3 , 3>(
    		w2 + x2 - y2 - z2 , 2 * ( xy - wz ) , 2 * ( xz + wy ) ,
			2 * ( xy + wz ) , w2 - x2 + y2 - z2 , 2 * ( yz - wx ) ,
			2 * ( xz - wy ) , 2 * ( yz + wx ) , w2 - x2 - y2 + z2
		);
    }

    /**
     * @brief   Creates a quaternion from angle-axis representation.
     * @tparam  Ty Quaternion element type.
     * @tparam  Tf Angle type (arithmetic).
     * @param   angle Rotation angle in radians.
     * @param   axis Rotation axis (will be normalized internally).
     * @return  Quaternion representing the rotation.
     *
     * @details
     * Constructs a quaternion from angle θ and axis a:
     * q = (cos(θ/2), sin(θ/2) * a)
     *
     * The axis is automatically normalized before computation.
     *
     * Example:
     * @code
     * auto q = angle_axis(radians(45.0f), vec3f{0.0f, 1.0f, 0.0f});
     * // q represents a 45-degree rotation around the Y-axis
     * @endcode
     */
	template <arithmetic_t Ty , arithmetic_t Tf>
	VERTEX_CONSTEXPR auto angle_axis( const Tf& angle , const vector_template<Ty , 3>& axis ) noexcept {
    	const auto a = normalize( axis );
    	const auto s = static_cast< Ty >( std::sin( angle / Tf { 2 } ) );
    	const auto c = static_cast< Ty >( std::cos( angle / Tf { 2 } ) );
    	return quat_template<Ty>( c , a[0] * s , a[1] * s , a[2] * s );
    }

    /**
     * @brief   Converts a quaternion to a 4x4 rotation matrix with identity translation.
     * @tparam  Ty Quaternion element type.
     * @param   q Input quaternion (should be normalized).
     * @return  4x4 homogeneous rotation matrix.
     *
     * @details
     * Converts a unit quaternion to its equivalent 4x4 affine transformation matrix.
     * The matrix is in column-major format, compatible with OpenGL/Vulkan.
     *
     * The resulting matrix applies rotation only, with zero translation:
     * [ R   0 ]
     * [ 0   1 ]
     *
     * @note    For best results, the quaternion should be normalized.
     *          The resulting matrix is orthonormal.
     */
	template <arithmetic_t Ty>
	VERTEX_CONSTEXPR auto matrix4_cast( const quat_template<Ty>& q ) noexcept {
    	auto w2 = q.w * q.w;
    	auto x2 = q.x * q.x;
    	auto y2 = q.y * q.y;
    	auto z2 = q.z * q.z;
    	auto xy = q.x * q.y;
    	auto xz = q.x * q.z;
    	auto yz = q.y * q.z;
    	auto wx = q.w * q.x;
    	auto wy = q.w * q.y;
    	auto wz = q.w * q.z;
    	return matrix_template<Ty , 4 , 4>(
    		w2 + x2 - y2 - z2 ,   2 * ( xy - wz )   , 2 * ( xz + wy )   , 0 ,
    		2 * ( xy + wz )   ,   w2 - x2 + y2 - z2 , 2 * ( yz - wx )   , 0 ,
    		2 * ( xz - wy )   ,   2 * ( yz + wx )   , w2 - x2 - y2 + z2 , 0 ,
    		0                 ,   0                 ,                 0 , 1
		);
    }

    /**
     * @brief   Converts a 3x3 rotation matrix to a quaternion.
     * @tparam  Ty Element type.
     * @param   m Input 3x3 rotation matrix (must be orthonormal).
     * @return  Quaternion representing the same rotation.
     *
     * @details
     * Uses the trace method with numerical stability improvements.
     * The algorithm selects the largest diagonal element to avoid division by zero
     * and ensure accurate results.
     *
     * Steps:
     * 1. Compute 4 * (component²) values from the matrix diagonal
     * 2. Find the largest to determine which component to compute first
     * 3. Compute the selected component using sqrt(trace + 1) / 2
     * 4. Compute the remaining components using the matrix elements
     *
     * @note    The input matrix should be a valid orthonormal rotation matrix.
     *          If the matrix is not orthonormal, the result may be inaccurate.
     */
	template <arithmetic_t Ty>
	VERTEX_CONSTEXPR auto quat_cast( const matrix_template<Ty , 3 , 3>& m ) noexcept {
        auto fourXSq = m.data [ 0 ][ 0 ] - m.data [ 1 ][ 1 ] - m.data [ 2 ][ 2 ];
        auto fourYSq = m.data [ 1 ][ 1 ] - m.data [ 0 ][ 0 ] - m.data [ 2 ][ 2 ];
        auto fourZSq = m.data [ 2 ][ 2 ] - m.data [ 0 ][ 0 ] - m.data [ 1 ][ 1 ];
        auto fourWSq = m.data [ 0 ][ 0 ] + m.data [ 1 ][ 1 ] + m.data [ 2 ][ 2 ];

    	int idx = 0;
    	auto biggest = fourWSq;
    	if ( fourXSq > biggest ) { biggest = fourXSq; idx = 1; }
    	if ( fourYSq > biggest ) { biggest = fourYSq; idx = 2; }
    	if ( fourZSq > biggest ) { biggest = fourZSq; idx = 3; }

    	auto val = std::sqrt( biggest + Ty { 1 } ) * Ty { 0.5 };
    	auto mul = Ty { 0.25 } / val;

    	switch ( idx ) {
    		case 0: return quat_template<Ty>( val , ( m.data [ 1 ][ 2 ] - m.data [ 2 ][ 1 ] ) * mul , ( m.data [ 2 ][ 0 ] - m.data [ 0 ][ 2 ] ) * mul , ( m.data [ 0 ][ 1 ] - m.data [ 1 ][ 0 ] ) * mul );
    		case 1: return quat_template<Ty>( ( m.data [ 1 ][ 2 ] - m.data [ 2 ][ 1 ] ) * mul , val , ( m.data [ 0 ][ 1 ] + m.data [ 1 ][ 0 ] ) * mul , ( m.data [ 2 ][ 0 ] + m.data [ 0 ][ 2 ] ) * mul );
    		case 2: return quat_template<Ty>( ( m.data [ 2 ][ 0 ] - m.data [ 0 ][ 2 ] ) * mul , ( m.data [ 0 ][ 1 ] + m.data [ 1 ][ 0 ] ) * mul , val , ( m.data [ 1 ][ 2 ] + m.data [ 2 ][ 1 ] ) * mul );
    	   default: return quat_template<Ty>( ( m.data [ 0 ][ 1 ] - m.data [ 1 ][ 0 ] ) * mul , ( m.data [ 2 ][ 0 ] + m.data [ 0 ][ 2 ] ) * mul , ( m.data [ 1 ][ 2 ] + m.data [ 2 ][ 1 ] ) * mul , val );
    	}
    }

    /**
     * @brief   Linear interpolation (LERP) between two quaternions.
     * @tparam  Ty Quaternion element type.
     * @tparam  U Interpolation factor type (arithmetic).
     * @param   quat1 Starting quaternion.
     * @param   quat2 Ending quaternion.
     * @param   t Interpolation factor in range [0, 1].
     * @return  Interpolated quaternion (normalized).
     *
     * @details
     * Performs component-wise linear interpolation followed by normalization (NLERP).
     * This is faster than SLERP but does not maintain constant angular velocity.
     *
     * The interpolation formula is:
     * lerp(q1, q2, t) = normalize(q1 * (1-t) + q2 * t)
     *
     * @note    t should be in the range [0, 1]. Values outside this range may produce
     *          unexpected results.
     *
     * @see     slerp() for constant angular velocity interpolation.
     */
	template <arithmetic_t Ty , arithmetic_t U>
	VERTEX_CONSTEXPR quat_template<Ty> lerp( const quat_template<Ty>& quat1 , const quat_template<Ty>& quat2 , const U& t ) noexcept {
    	return normalize<Ty>( quat1 * ( Ty { 1 } - t ) + quat2 * t );
    }
}

#endif //VERTEX_QUAT_HPP
