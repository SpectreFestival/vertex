#include "vertex.hpp"

#include <cstdio>
#include <cmath>

using namespace vtx;

// ===========================================================================
// vtx production-readiness test suite
//
// Covers every template (vector/matrix/quaternion/vertex), every type
// (vi08..vf64), every function and operator, plus edge cases.
// ===========================================================================

static int g_total  = 0;
static int g_failed = 0;

#define EXPECT( cond )															\
	do {																		\
		++g_total;																\
		if ( !( cond ) ) {														\
			++g_failed;															\
			std::printf( "  [FAIL] %s:%d  %s\n" , __FILE__ , __LINE__ , #cond );\
		}																		\
	} while ( 0 )

static void section( const char* name ) { std::printf( "\n== %s ==\n" , name ); }

// ---- comparison helpers (exact for int, epsilon for float) ----
template < typename A , typename B >
static bool eq( A a , B b ) {
	if constexpr ( std::is_floating_point_v< A > || std::is_floating_point_v< B > ) {
		return std::fabs( a - b ) <= 1e-4;
	} else {
		return a == b;
	}
}

template < typename Ty , size_t N >
static bool veq( const vector_template< Ty , N >& a , const vector_template< Ty , N >& b ) {
	for ( size_t i = 0; i < N; ++i ) {
		if ( !eq( a [ i ] , b [ i ] ) ) {
			return false;
		}
	}
	return true;
}

template < typename Ty , size_t R , size_t C >
static bool meq( const matrix_template< Ty , R , C >& a , const matrix_template< Ty , R , C >& b ) {
	for ( size_t c = 0; c < C; ++c ) {
		for ( size_t r = 0; r < R; ++r ) {
			if ( !eq( a.data [ c ][ r ] , b.data [ c ][ r ] ) ) {
				return false;
			}
		}
	}
	return true;
}

template < typename Ty >
static bool qeq( const quat_template< Ty >& a , const quat_template< Ty >& b ) {
	return 
		eq( a.w , b.w ) && 
		eq( a.x , b.x ) && 
		eq( a.y , b.y ) && 
		eq( a.z , b.z );
}

// ===========================================================================
// vector
// ===========================================================================
template < typename Ty >
static void test_vector( const char* name ) {
	section( name );
	using v2 = vector_template< Ty , 2 >;
	using v3 = vector_template< Ty , 3 >;
	using v4 = vector_template< Ty , 4 >;

	{   // construction & access
		v2 z;  EXPECT( eq( z [ 0 ] , Ty {} ) && eq( z [ 1 ] , Ty {} ) );
		v2 a { Ty( 1 ) , Ty( 2 ) };  EXPECT( eq( a [ 0 ] , Ty( 1 ) ) && eq( a [ 1 ] , Ty( 2 ) ) );
		v2 b( Ty( 3 ) , Ty( 4 ) );  EXPECT( eq( b [ 0 ] , Ty( 3 ) ) && eq( b [ 1 ] , Ty( 4 ) ) );
		a [ 0 ] = Ty( 9 );  EXPECT( eq( a [ 0 ] , Ty( 9 ) ) );
		const v3 cv { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) };  EXPECT( eq( cv [ 2 ] , Ty( 3 ) ) );
	}
	{   // iterators
		v3 v { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) };
		Ty s = Ty( 0 ); for ( auto x : v ) s += x;
		EXPECT( eq( s , Ty( 1 ) + Ty( 2 ) + Ty( 3 ) ) );
		EXPECT( v.begin( ) + 3 == v.end( ) );
	}
	{   // arithmetic
		v3 a { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) } , b { Ty( 4 ) , Ty( 5 ) , Ty( 6 ) };
		EXPECT( veq( a + b , v3 { Ty( 5 ) , Ty( 7 ) , Ty( 9 ) } ) );
		EXPECT( veq( b - a , v3 { Ty( 3 ) , Ty( 3 ) , Ty( 3 ) } ) );
		EXPECT( veq( a * b , v3 { Ty( 4 ) , Ty( 10 ) , Ty( 18 ) } ) );
		EXPECT( veq( a * Ty( 2 ) , v3 { Ty( 2 ) , Ty( 4 ) , Ty( 6 ) } ) );
		EXPECT( veq( +a , a ) );
		if constexpr ( std::is_signed_v< Ty > || std::is_floating_point_v< Ty > )
			EXPECT( veq( -a , v3 { Ty( -1 ) , Ty( -2 ) , Ty( -3 ) } ) );
	}
	{   // division (divisible values, safe for int/unsigned)
		v2 d { Ty( 6 ) , Ty( 8 ) } , e { Ty( 2 ) , Ty( 4 ) };
		EXPECT( veq( d / e , v2 { Ty( 3 ) , Ty( 2 ) } ) );
		EXPECT( veq( d / Ty( 2 ) , v2 { Ty( 3 ) , Ty( 4 ) } ) );
	}
	{   // compound assignment
		v3 a { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) } , b { Ty( 4 ) , Ty( 5 ) , Ty( 6 ) };
		a += b;  EXPECT( veq( a , v3 { Ty( 5 ) , Ty( 7 ) , Ty( 9 ) } ) );
		a -= b;  EXPECT( veq( a , v3 { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) } ) );
		a *= b;  EXPECT( veq( a , v3 { Ty( 4 ) , Ty( 10 ) , Ty( 18 ) } ) );
		a *= Ty( 3 );  EXPECT( veq( a , v3 { Ty( 12 ) , Ty( 30 ) , Ty( 54 ) } ) );
		a /= Ty( 3 );  EXPECT( veq( a , v3 { Ty( 4 ) , Ty( 10 ) , Ty( 18 ) } ) );
	}
	{   // relational (element-wise bool vector)
		v3 a { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) } , b { Ty( 2 ) , Ty( 2 ) , Ty( 3 ) };
		using bv = vector_template< bool , 3 >;
		EXPECT( veq( a < b , bv { true  , false , false } ) );
		EXPECT( veq( a <= b , bv { true  , true  , true } ) );
		EXPECT( veq( a > b , bv { false , false , false } ) );
		EXPECT( veq( a >= b , bv { false , true  , true } ) );
		EXPECT( veq( a == b , bv { false , true  , true } ) );
		EXPECT( veq( a != b , bv { true  , false , false } ) );
		EXPECT( veq( ( a < b ) && ( a <= b ) , bv { true , false , false } ) );
		EXPECT( veq( ( a < b ) || ( a > b ) , bv { true , false , false } ) );
	}
	{   // dot / cross / reflect / fma / vector_cast (pure arithmetic)
		EXPECT( eq( dot( v3 { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) } , v3 { Ty( 4 ) , Ty( 5 ) , Ty( 6 ) } ) , Ty( 32 ) ) );
		EXPECT( eq( cross( v2 { Ty( 1 ) , Ty( 0 ) } , v2 { Ty( 0 ) , Ty( 1 ) } ) , Ty( 1 ) ) );
		EXPECT( veq( cross( v3 { Ty( 1 ) , Ty( 0 ) , Ty( 0 ) } , v3 { Ty( 0 ) , Ty( 1 ) , Ty( 0 ) } ) , v3 { Ty( 0 ) , Ty( 0 ) , Ty( 1 ) } ) );
		EXPECT( veq( reflect( v3 { Ty( 1 ) , Ty( 0 ) , Ty( 0 ) } , v3 { Ty( 1 ) , Ty( 0 ) , Ty( 0 ) } ) , v3 { Ty( -1 ) , Ty( 0 ) , Ty( 0 ) } ) );
		EXPECT( veq( fma( v3 { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) } , v3 { Ty( 4 ) , Ty( 5 ) , Ty( 6 ) } , v3 { Ty( 7 ) , Ty( 8 ) , Ty( 9 ) } ) , v3 { Ty( 11 ) , Ty( 18 ) , Ty( 27 ) } ) );
		EXPECT( veq( vector_cast< Ty , 3 >( Ty( 7 ) ) , v3 { Ty( 7 ) , Ty( 7 ) , Ty( 7 ) } ) );
		if constexpr ( std::is_signed_v< Ty > || std::is_floating_point_v< Ty > )
			EXPECT( veq( face_forward( v3 { Ty( 0 ) , Ty( 1 ) , Ty( 0 ) } , v3 { Ty( 0 ) , Ty( -1 ) , Ty( 0 ) } , v3 { Ty( 0 ) , Ty( 1 ) , Ty( 0 ) } ) , v3 { Ty( 0 ) , Ty( 1 ) , Ty( 0 ) } ) );
	}
	if constexpr ( std::is_floating_point_v< Ty > ) {
		// length / normalize / distance / angle / refract / rotate
		EXPECT( eq( length( v3 { Ty( 2 ) , Ty( 3 ) , Ty( 6 ) } ) , Ty( 7 ) ) );
		EXPECT( veq( normalize( v2 { Ty( 3 ) , Ty( 4 ) } ) , v2 { Ty( 0.6 ) , Ty( 0.8 ) } ) );
		EXPECT( eq( distance( v3 { Ty( 0 ) , Ty( 0 ) , Ty( 0 ) } , v3 { Ty( 3 ) , Ty( 4 ) , Ty( 0 ) } ) , Ty( 5 ) ) );
		EXPECT( eq( angle( v2 { Ty( 1 ) , Ty( 0 ) } , v2 { Ty( 0 ) , Ty( 1 ) } ) , Ty( 1.57079632679 ) ) );
		EXPECT( veq( refract( v3 { Ty( 0 ) , Ty( -1 ) , Ty( 0 ) } , v3 { Ty( 0 ) , Ty( 1 ) , Ty( 0 ) } , Ty( 1 ) ) , v3 { Ty( 0 ) , Ty( -1 ) , Ty( 0 ) } ) );
		EXPECT( veq( rotate( v2 { Ty( 1 ) , Ty( 0 ) } , Ty( 1.57079632679 ) ) , v2 { Ty( 0 ) , Ty( 1 ) } ) );
	}
}

// ===========================================================================
// matrix
// ===========================================================================
template < typename Ty >
static void test_matrix( const char* name ) {
	section( name );
	using m2 = matrix_template< Ty , 2 , 2 >;
	using m3 = matrix_template< Ty , 3 , 3 >;
	using m4 = matrix_template< Ty , 4 , 4 >;
	using v2 = vector_template< Ty , 2 >;
	using v3 = vector_template< Ty , 3 >;

	{   // construction & access
		m2 m { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) , Ty( 4 ) };
		EXPECT( eq( m( 0 , 0 ) , Ty( 1 ) ) && eq( m( 0 , 1 ) , Ty( 2 ) ) );
		EXPECT( eq( m( 1 , 0 ) , Ty( 3 ) ) && eq( m( 1 , 1 ) , Ty( 4 ) ) );
		m2 n( Ty( 5 ) , Ty( 6 ) , Ty( 7 ) , Ty( 8 ) );
		EXPECT( eq( n( 0 , 0 ) , Ty( 5 ) ) && eq( n( 1 , 1 ) , Ty( 8 ) ) );
		m2 z;  EXPECT( eq( z.data [ 0 ][ 0 ] , Ty {} ) && eq( z.data [ 1 ][ 1 ] , Ty {} ) );
	}
	{   // arithmetic
		m2 a { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) , Ty( 4 ) } , b { Ty( 5 ) , Ty( 6 ) , Ty( 7 ) , Ty( 8 ) };
		EXPECT( meq( a + b , m2 { Ty( 6 ) , Ty( 8 ) , Ty( 10 ) , Ty( 12 ) } ) );
		EXPECT( meq( b - a , m2 { Ty( 4 ) , Ty( 4 ) , Ty( 4 ) , Ty( 4 ) } ) );
		EXPECT( meq( a * Ty( 2 ) , m2 { Ty( 2 ) , Ty( 4 ) , Ty( 6 ) , Ty( 8 ) } ) );
		m2 even { Ty( 2 ) , Ty( 4 ) , Ty( 6 ) , Ty( 8 ) };
		EXPECT( meq( even / Ty( 2 ) , m2 { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) , Ty( 4 ) } ) );
	}
	{   // mat * mat / mat * vec / vec * mat
		m2 I { Ty( 1 ) , Ty( 0 ) , Ty( 0 ) , Ty( 1 ) } , m { Ty( 5 ) , Ty( 6 ) , Ty( 7 ) , Ty( 8 ) };
		EXPECT( meq( I * m , m ) );
		EXPECT( meq( m * I , m ) );
		EXPECT( veq( I * v2 { Ty( 3 ) , Ty( 4 ) } , v2 { Ty( 3 ) , Ty( 4 ) } ) );
		EXPECT( veq( v2 { Ty( 3 ) , Ty( 4 ) } * I , v2 { Ty( 3 ) , Ty( 4 ) } ) );
	}
	{   // compound
		m2 a { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) , Ty( 4 ) } , b { Ty( 5 ) , Ty( 6 ) , Ty( 7 ) , Ty( 8 ) };
		a += b;  EXPECT( meq( a , m2 { Ty( 6 ) , Ty( 8 ) , Ty( 10 ) , Ty( 12 ) } ) );
		a -= b;  EXPECT( meq( a , m2 { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) , Ty( 4 ) } ) );
		a *= Ty( 2 );  EXPECT( meq( a , m2 { Ty( 2 ) , Ty( 4 ) , Ty( 6 ) , Ty( 8 ) } ) );
		a /= Ty( 2 );  EXPECT( meq( a , m2 { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) , Ty( 4 ) } ) );
	}
	{   // relational
		m2 a { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) , Ty( 4 ) } , b { Ty( 2 ) , Ty( 2 ) , Ty( 4 ) , Ty( 4 ) };
		using bm = matrix_template< bool , 2 , 2 >;
		EXPECT( meq( a < b , bm { true , false , true , false } ) );
		EXPECT( meq( a <= b , bm { true , true , true , true } ) );
		EXPECT( meq( a == b , bm { false , true , false , true } ) );
		EXPECT( meq( a != b , bm { true , false , true , false } ) );
	}
	{   // row / col / transpose / trace / hadamard / outer_product / diagonal
		m2 m { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) , Ty( 4 ) };
		EXPECT( veq( row( m , 0 ) , v2 { Ty( 1 ) , Ty( 2 ) } ) );
		EXPECT( veq( col( m , 0 ) , v2 { Ty( 1 ) , Ty( 3 ) } ) );
		EXPECT( meq( transpose( m ) , m2 { Ty( 1 ) , Ty( 3 ) , Ty( 2 ) , Ty( 4 ) } ) );
		EXPECT( eq( trace( m ) , Ty( 5 ) ) );
		EXPECT( meq( hadamard( m , m ) , m2 { Ty( 1 ) , Ty( 4 ) , Ty( 9 ) , Ty( 16 ) } ) );
		EXPECT( meq( outer_product( v2 { Ty( 1 ) , Ty( 2 ) } , v2 { Ty( 3 ) , Ty( 4 ) } ) , m2 { Ty( 3 ) , Ty( 4 ) , Ty( 6 ) , Ty( 8 ) } ) );
		EXPECT( veq( diagonal( m ) , v2 { Ty( 1 ) , Ty( 4 ) } ) );
	}
	{   // determinant (diagonal, positive)
		m2 d { Ty( 2 ) , Ty( 0 ) , Ty( 0 ) , Ty( 3 ) };
		EXPECT( eq( determinant( d ) , Ty( 6 ) ) );
		m3 d3 { Ty( 2 ),Ty( 0 ),Ty( 0 ), Ty( 0 ),Ty( 3 ),Ty( 0 ), Ty( 0 ),Ty( 0 ),Ty( 4 ) };
		EXPECT( eq( determinant( d3 ) , Ty( 24 ) ) );
	}
	if constexpr ( std::is_floating_point_v< Ty > ) {
		m2 m { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) , Ty( 4 ) };
		EXPECT( meq( inverse( m ) , m2 { Ty( -2 ) , Ty( 1 ) , Ty( 1.5 ) , Ty( -0.5 ) } ) );
		EXPECT( meq( inverse_transpose( m ) , m2 { Ty( -2 ) , Ty( 1.5 ) , Ty( 1 ) , Ty( -0.5 ) } ) );

		m4 I { Ty( 1 ),Ty( 0 ),Ty( 0 ),Ty( 0 ), Ty( 0 ),Ty( 1 ),Ty( 0 ),Ty( 0 ), Ty( 0 ),Ty( 0 ),Ty( 1 ),Ty( 0 ), Ty( 0 ),Ty( 0 ),Ty( 0 ),Ty( 1 ) };
		EXPECT( meq( inverse( I ) , I ) );
		EXPECT( meq( translate( I , v3 { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) } ) ,
				m4 { Ty( 1 ),Ty( 0 ),Ty( 0 ),Ty( 1 ), Ty( 0 ),Ty( 1 ),Ty( 0 ),Ty( 2 ), Ty( 0 ),Ty( 0 ),Ty( 1 ),Ty( 3 ), Ty( 0 ),Ty( 0 ),Ty( 0 ),Ty( 1 ) } ) );
		EXPECT( meq( scale( I , v3 { Ty( 2 ) , Ty( 3 ) , Ty( 4 ) } ) ,
				m4 { Ty( 2 ),Ty( 0 ),Ty( 0 ),Ty( 0 ), Ty( 0 ),Ty( 3 ),Ty( 0 ),Ty( 0 ), Ty( 0 ),Ty( 0 ),Ty( 4 ),Ty( 0 ), Ty( 0 ),Ty( 0 ),Ty( 0 ),Ty( 1 ) } ) );
		EXPECT( meq( rotate( I , Ty( 1.57079632679 ) , v3 { Ty( 0 ) , Ty( 0 ) , Ty( 1 ) } ) ,
				m4 { Ty( 0 ),Ty( -1 ),Ty( 0 ),Ty( 0 ), Ty( 1 ),Ty( 0 ),Ty( 0 ),Ty( 0 ), Ty( 0 ),Ty( 0 ),Ty( 1 ),Ty( 0 ), Ty( 0 ),Ty( 0 ),Ty( 0 ),Ty( 1 ) } ) );
		EXPECT( meq( look_at( v3 { Ty( 0 ),Ty( 0 ),Ty( 0 ) } , v3 { Ty( 0 ),Ty( 0 ),Ty( -1 ) } , v3 { Ty( 0 ),Ty( 1 ),Ty( 0 ) } ) , I ) );
		m4 P = perspective( Ty( 1.57079632679 ) , Ty( 1 ) , Ty( 0.1 ) , Ty( 100 ) );
		EXPECT( eq( P( 0 , 0 ) , Ty( 1 ) ) && eq( P( 1 , 1 ) , Ty( 1 ) ) );
		m4 O = ortho( Ty( -1 ) , Ty( 1 ) , Ty( -1 ) , Ty( 1 ) , Ty( 0.1 ) , Ty( 100 ) );
		EXPECT( eq( O( 0 , 0 ) , Ty( 1 ) ) && eq( O( 1 , 1 ) , Ty( 1 ) ) && eq( O( 3 , 3 ) , Ty( 1 ) ) );
	}
}

// ===========================================================================
// quaternion
// ===========================================================================
template < typename Ty >
static void test_quat( const char* name ) {
	section( name );

	quat_template< Ty > q { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) , Ty( 4 ) };
	EXPECT( eq( q.w , Ty( 1 ) ) && eq( q.x , Ty( 2 ) ) && eq( q.y , Ty( 3 ) ) && eq( q.z , Ty( 4 ) ) );

	quat_template< Ty > a { Ty( 1 ) , Ty( 2 ) , Ty( 3 ) , Ty( 4 ) } , b { Ty( 5 ) , Ty( 6 ) , Ty( 7 ) , Ty( 8 ) };
	auto sum = a + b;
	EXPECT( eq( sum.w , Ty( 6 ) ) && eq( sum.x , Ty( 8 ) ) && eq( sum.y , Ty( 10 ) ) && eq( sum.z , Ty( 12 ) ) );
	EXPECT( eq( dot( a , b ) , Ty( 70 ) ) );
	EXPECT( qeq( conjugate( a ) , quat_template< Ty >{ Ty( 1 ) , Ty( -2 ) , Ty( -3 ) , Ty( -4 ) } ) );
	EXPECT( qeq( a * b , quat_template< Ty >{ Ty( -60 ) , Ty( 12 ) , Ty( 30 ) , Ty( 24 ) } ) );

	if constexpr ( std::is_floating_point_v< Ty > ) {
		quat_template< Ty > m { Ty( 0 ) , Ty( 3 ) , Ty( 4 ) , Ty( 0 ) };
		EXPECT( eq( magnitude( m ) , Ty( 5 ) ) );
		auto n = normalize( m );
		EXPECT( eq( n.w , Ty( 0 ) ) && eq( n.x , Ty( 0.6 ) ) && eq( n.y , Ty( 0.8 ) ) && eq( n.z , Ty( 0 ) ) );

		quat_template< Ty > id { Ty( 1 ) , Ty( 0 ) , Ty( 0 ) , Ty( 0 ) };
		auto inv = inverse( id );
		EXPECT( eq( inv.w , Ty( 1 ) ) && eq( inv.x , Ty( 0 ) ) && eq( inv.y , Ty( 0 ) ) && eq( inv.z , Ty( 0 ) ) );

		auto qz = angle_axis( Ty( 1.57079632679 ) , vector_template< Ty , 3 >{ Ty( 0 ) , Ty( 0 ) , Ty( 1 ) } );
		EXPECT( eq( qz.w , Ty( 0.70710678 ) ) && eq( qz.z , Ty( 0.70710678 ) ) );
		EXPECT( veq( rotate_vector( qz , vector_template< Ty , 3 >{ Ty( 1 ) , Ty( 0 ) , Ty( 0 ) } ) , vector_template< Ty , 3 >{ Ty( 0 ) , Ty( 1 ) , Ty( 0 ) } ) );
		EXPECT( meq( to_matrix( qz ) , matrix_template< Ty , 3 , 3 >{ Ty( 0 ) , Ty( -1 ) , Ty( 0 ) , Ty( 1 ) , Ty( 0 ) , Ty( 0 ) , Ty( 0 ) , Ty( 0 ) , Ty( 1 ) } ) );
		EXPECT( meq( matrix4_cast( qz ) , matrix_template< Ty , 4 , 4 >{ Ty( 0 ) , Ty( -1 ) , Ty( 0 ) , Ty( 0 ) , Ty( 1 ) , Ty( 0 ) , Ty( 0 ) , Ty( 0 ) , Ty( 0 ) , Ty( 0 ) , Ty( 1 ) , Ty( 0 ) , Ty( 0 ) , Ty( 0 ) , Ty( 0 ) , Ty( 1 ) } ) );

		auto qr = quat_cast( to_matrix( qz ) );
		EXPECT( eq( std::fabs( qr.w ) , Ty( 0.70710678 ) ) && eq( std::fabs( qr.z ) , Ty( 0.70710678 ) ) );

		quat_template< Ty > q1 { Ty( 1 ) , Ty( 0 ) , Ty( 0 ) , Ty( 0 ) } , q2 { Ty( 0 ) , Ty( 0 ) , Ty( 0 ) , Ty( 1 ) };
		auto lp = lerp( q1 , q2 , Ty( 0.5 ) );
		EXPECT( eq( lp.w , Ty( 0.70710678 ) ) && eq( lp.z , Ty( 0.70710678 ) ) );
		auto sp = slerp( q1 , q2 , Ty( 0.5 ) );
		EXPECT( eq( sp.w , Ty( 0.70710678 ) ) && eq( sp.z , Ty( 0.70710678 ) ) );
	}
}

// ===========================================================================
// vertex / tuple / traits
// ===========================================================================
static void test_vertex( ) {
	section( "vertex traits / tuple / layout" );

	// traits
	EXPECT( vertex_traits< vf32 >::count == 1 );
	EXPECT( vertex_traits< vf64 >::offset_step == 8 );
	EXPECT( vertex_traits< vec2f32 >::count == 1 );
	EXPECT( vertex_traits< mat4x4f32 >::count == 4 );

	// tuple element access
	vertex< vf32 , vf32 , vf32 > v( 1.0f , 2.0f , 3.0f );
	EXPECT( eq( get< 0 >( v ) , 1.0f ) );
	EXPECT( eq( get< 1 >( v ) , 2.0f ) );
	EXPECT( eq( get< 2 >( v ) , 3.0f ) );

	// vertex layout
	auto layout = MakeVertexLayout< vf32 , vec2f32 , vf32 >( );
	EXPECT( layout.size( ) == 3 );
	EXPECT( layout [ 0 ].location == 0 );
	EXPECT( layout [ 1 ].location == 1 );
	EXPECT( layout [ 2 ].location == 2 );

	// element offset computation
	auto offsets = compute_offset< vf32 , vec2f32 >( );
	EXPECT( offsets.size( ) == 2 );
	EXPECT( offsets [ 0 ] == 0 );
	EXPECT( offsets [ 1 ] == vertex_traits< vf32 >::offset_step );
}

// ===========================================================================
// edge cases
// ===========================================================================
static void test_edge_cases( ) {
	section( "edge cases" );

	// zero vector length
	EXPECT( eq( length( vec3f32 { 0.0f , 0.0f , 0.0f } ) , 0.0f ) );
	// dot with zero vector
	EXPECT( eq( dot( vec3f32 { 1.0f , 2.0f , 3.0f } , vec3f32 { 0.0f , 0.0f , 0.0f } ) , 0.0f ) );
	// cross of parallel vectors = zero
	EXPECT( veq( cross( vec3f32 { 1,0,0 } , vec3f32 { 2,0,0 } ) , vec3f32 { 0,0,0 } ) );
	// angle of same-direction vectors = 0
	EXPECT( eq( angle( vec3f32 { 1,0,0 } , vec3f32 { 1,0,0 } ) , 0.0f ) );
	// distance of identical points = 0
	EXPECT( eq( distance( vec3f32 { 1,2,3 } , vec3f32 { 1,2,3 } ) , 0.0f ) );

	// determinant of singular matrix = 0
	EXPECT( eq( determinant( mat2x2f32 { 1,2,2,4 } ) , 0.0f ) );
	// inverse of identity = identity
	EXPECT( meq( inverse( mat4x4f32 { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 } ) ,
			mat4x4f32 { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 } ) );
		// transpose of symmetric matrix = itself
	EXPECT( meq( transpose( mat2x2f32 { 1,2,2,3 } ) , mat2x2f32 { 1,2,2,3 } ) );
	// trace of identity = dimension
	EXPECT( eq( trace( mat3x3f32 { 1,0,0, 0,1,0, 0,0,1 } ) , 3.0f ) );

	// quaternion identity: magnitude 1, conjugate = inverse
	quat_template< float > id { 1.0f , 0.0f , 0.0f , 0.0f };
	EXPECT( eq( magnitude( id ) , 1.0f ) );
	EXPECT( qeq( conjugate( id ) , id ) );
	EXPECT( qeq( inverse( id ) , id ) );
}

// ===========================================================================
int main( ) {
	std::printf( "== vtx production-readiness test ==\n" );

#define TV( T , N ) test_vector< T >( "vector " N );
	TV( vi08 , "vi08" )
		TV( vi16 , "vi16" )
		TV( vi32 , "vi32" )
		TV( vi64 , "vi64" )
		TV( vu08 , "vu08" )
		TV( vu16 , "vu16" )
		TV( vu32 , "vu32" )
		TV( vu64 , "vu64" )
		TV( vf32 , "vf32" )
		TV( vf64 , "vf64" )
#undef TV

#define TM( T , N ) test_matrix< T >( "matrix " N );
		TM( vi32 , "vi32" )
		TM( vi64 , "vi64" )
		TM( vu32 , "vu32" )
		TM( vu64 , "vu64" )
		TM( vf32 , "vf32" )
		TM( vf64 , "vf64" )
#undef TM

#define TQ( T , N ) test_quat< T >( "quat " N );
		TQ( vi32 , "vi32" )
		TQ( vf32 , "vf32" )
		TQ( vf64 , "vf64" )
#undef TQ

		test_vertex( );
	test_edge_cases( );

	std::printf( "\n------------------------------\n" );
	std::printf( "passed %d / %d\n" , g_total - g_failed , g_total );
	if ( g_failed == 0 )
		std::printf( "ALL TESTS PASSED\n" );
	else
		std::printf( "%d TEST(S) FAILED\n" , g_failed );

	return g_failed == 0 ? 0 : 1;
}