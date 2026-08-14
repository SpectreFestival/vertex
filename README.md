# Vertex

A header-only **C++20** math library that maps composable C++ types to Vulkan
vertex input descriptions at compile time.

`vtx` combines a unified vector / matrix / quaternion toolkit with a
`vertex<Ty...>` container that auto-generates `VkVertexInputAttributeDescription`
layouts for your shader. Everything is `constexpr` where possible and
compile-time unrolled with `std::index_sequence` and fold expressions.

```cpp
#include "vertex.hpp"
using namespace vtx;

using Vertex = vertex<vec3f, vec3f, vec2f>;   // position, normal, uv

Vertex v{ vec3f{0, 0, 0}, vec3f{0, 1, 0}, vec2f{0, 0} };

auto layout = MakeVertexLayout<vec3f, vec3f, vec2f>();
// layout[0] = { location=0, format=106, count=1, offset=0,  offset_step=12 }
// layout[1] = { location=1, format=106, count=1, offset=12, offset_step=12 }
// layout[2] = { location=2, format=103, count=1, offset=24, offset_step=8  }
```

---

## Table of contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Getting started](#getting-started)
4. [Type system](#type-system)
5. [Vectors](#vectors)
6. [Matrices](#matrices)
7. [Quaternions](#quaternions)
8. [Vertex input mapping](#vertex-input-mapping)
9. [String formatting](#string-formatting)
10. [SIMD extensions](#simd-extensions)
11. [Building and running the tests](#building-and-running-the-tests)
12. [Project structure](#project-structure)
13. [Notes and caveats](#notes-and-caveats)
14. [License](#license)

---

## Features

- **Header-only** — a single include, no build step, no binary to link.
- **Compile-time everything** — dimensions are template parameters; every
  element-wise loop is unrolled via `std::index_sequence` + fold expressions.
- **Arbitrary dimensions** — `vector_template<T, N>` and
  `matrix_template<T, R, C>` accept any `N`, `R`, `C` (not limited to 4x4).
- **Full arithmetic coverage** — every template instantiates over `vi08..vi64`,
  `vu08..vu64`, `vf32`, `vf64`.
- **Vulkan vertex input** — `vertex_traits<T>`, `compute_offset` and
  `MakeVertexLayout` turn a `vertex<Ty...>` into a ready-to-use
  `VkVertexInputAttributeDescription` sequence.
- **Optional AVX2 SIMD** — accelerated matrix add/sub and 4x4 inverse when
  `VERTEX_ENABLE_EXTENSIONS` is defined.
- **Quaternion rotation math** — `slerp`, `angle_axis`, matrix conversion,
  `rotate_vector`.
- **Formatting** — `std::ostream` and `std::format` support for vectors.
- **Zero dependencies** — standard library only.

## Requirements

- A C++20 compiler: MSVC v14.3+ (VS 2022+), GCC 10+, or Clang 12+.
- No external dependencies.
- Optional SIMD path additionally requires AVX2 (`-mavx2` / `/arch:AVX2`).

## Getting started

### As a header-only library

Add the repository root to your include path and include the entry header:

```cpp
#include "vertex.hpp"
```

All symbols live in the `vtx` namespace. You can also include the aggregator
directly via `#include "vtx/vtx.hpp"`.

### With CMake

Either `add_subdirectory` or `FetchContent`:

```cmake
FetchContent_Declare(vertex GIT_REPOSITORY https://github.com/<you>/vertex)
FetchContent_MakeAvailable(vertex)

target_link_libraries(my_app PRIVATE vertex)
```

> **Note:** the bundled `CMakeLists.txt` currently builds only the test
> executable (`vertex_test`). An installable `vertex::vertex` INTERFACE target
> is on the roadmap.

## Type system

### Scalars (`vtx/vtxbasic.hpp`)

| Alias  | C++ type  | Alias  | C++ type  |
|--------|-----------|--------|-----------|
| `vi08` | `int8_t`  | `vu08` | `uint8_t` |
| `vi16` | `int16_t` | `vu16` | `uint16_t`|
| `vi32` | `int32_t` | `vu32` | `uint32_t`|
| `vi64` | `int64_t` | `vu64` | `uint64_t`|
| `vf32` | `float`   | `vf64` | `double`  |

The concept `arithmetic_t<T>` constrains templates to any integral or
floating-point type.

### Vectors (`vtx/vtxvec2/3/4.hpp`)

`vector_template<Ty, N>` is the primary template; `vec2<Ty>`, `vec3<Ty>`,
`vec4<Ty>` are aliases. Concrete aliases for every arithmetic type:

| Base      | `vec2`      | `vec3`      | `vec4`      |
|-----------|-------------|-------------|-------------|
| `*i08`    | `vec2i08`   | `vec3i08`   | `vec4i08`   |
| `*i16`    | `vec2i16`   | `vec3i16`   | `vec4i16`   |
| `*i32`    | `vec2i32`   | `vec3i32`   | `vec4i32`   |
| `*i64`    | `vec2i64`   | `vec3i64`   | `vec4i64`   |
| `*u08`    | `vec2u08`   | `vec3u08`   | `vec4u08`   |
| `*u16`    | `vec2u16`   | `vec3u16`   | `vec4u16`   |
| `*u32`    | `vec2u32`   | `vec3u32`   | `vec4u32`   |
| `*u64`    | `vec2u64`   | `vec3u64`   | `vec4u64`   |
| `*f32`    | `vec2f32`   | `vec3f32`   | `vec4f32`   |
| `*f64`    | `vec2f64`   | `vec3f64`   | `vec4f64`   |

Convenience defaults: `vecNf` = `vecNf32`, `vecNd` = `vecNf64`, `vecNi` =
`vecNi32`, `vecNu` = `vecNu32` (e.g. `vec3f`, `vec3d`, `vec3i`, `vec3u`).

### Matrices (`vtx/vtxmat*.hpp`)

`matrix_template<Ty, R, C>` (R rows, C columns) is the primary template, with
`mat2x2`..`mat4x4` and non-square `mat2x3`, `mat2x4`, `mat3x2`, `mat3x4`,
`mat4x2`, `mat4x3` aliases. Each follows the same concrete-alias scheme, e.g.
`mat4x4i08`, `mat4x4i16`, ..., `mat4x4f32`, `mat4x4f64`, plus defaults
`mat4x4f`, `mat4x4d`, `mat4x4i`, `mat4x4u`.

### Quaternions (`vtx/vtxquat.hpp`)

`quat_template<Ty>` is the only quaternion type (there is no short alias).
Components are stored as `(w, x, y, z)`.

## Vectors

### Class `vector_template<Ty, N>`

```cpp
template <arithmetic_t Ty, std::size_t N>
struct vector_template {
    Ty data[N]{};                       // contiguous storage

    // construction
    constexpr vector_template() = default;                 // zero-initialized
    explicit constexpr vector_template(Args&&... args);    // variadic
    explicit constexpr vector_template(std::initializer_list<Ty>);

    // element access
    constexpr Ty&       operator[](std::size_t idx);
    constexpr const Ty& operator[](std::size_t idx) const;

    // conversion to another element type
    template <arithmetic_t U> explicit operator vector_template<U, N>();

    // iterators (random-access): begin/end/cbegin/cend/rbegin/rend/crbegin/crend
};
```

### Operators

| Expression | Result | Notes |
|------------|--------|-------|
| `+a`, `-a` | `vec` | element-wise unary |
| `a + b`, `a - b` | `vec` | element-wise |
| `a * b` | `vec` | Hadamard (element-wise) product |
| `a / b` | `vec` | element-wise |
| `a * s`, `a / s` | `vec` | scalar multiply / divide |
| `a += b`, `a -= b`, `a *= b`, `a /= b` | `vec&` | compound |
| `a *= s`, `a /= s` | `vec&` | compound scalar |
| `a == b`, `a != b` | `vector_template<bool, N>` | element-wise |
| `a < b`, `a <= b`, `a > b`, `a >= b` | `vector_template<bool, N>` | element-wise |
| `a && b`, `a || b` | `vector_template<bool, N>` | element-wise logical |

### Free functions

| Function | Description |
|----------|-------------|
| `dot(a, b)` | Dot product (scalar). |
| `cross(a, b)` | 2D: scalar cross; 3D: vector cross. |
| `length(v)` | Euclidean norm `sqrt(dot(v, v))`. |
| `normalize(v)` | Unit vector (undefined for zero vector). |
| `distance(a, b)` | `length(b - a)`. |
| `angle(a, b)` | Angle in radians, clamped to `[0, pi]`. |
| `reflect(i, n)` | Reflection `i - 2 * dot(i, n) * n`. |
| `refract(i, n, eta)` | Snell's law; returns zero vector on total internal reflection. |
| `face_forward(n, i, nref)` | Flip normal to face the incident direction. |
| `fma(a, b, c)` | Fused multiply-add `a * b + c`. |
| `vector_cast<Ty, N>(s)` | Fill an N-vector with scalar `s`. |
| `vector_cast<Ty>(args...)` | Build a vector, dimension deduced from arg count. |
| `rotate(v2, angle)` | Rotate a 2D vector around the origin. |
| `rotate(v2, point, angle)` | Rotate a 2D vector around a pivot. |
| `rotate(v3, axis, angle)` | Rodrigues rotation around an axis (normalized internally). |
| `rotate(v3, point, axis, angle)` | Rodrigues rotation around an axis through a pivot. |
| `rotate(v4, a_xy, a_zw)` | Rotate a 4D vector in the XY and ZW planes. |

### Example

```cpp
vec3f a{ 1.0f, 2.0f, 3.0f };
vec3f b{ 4.0f, 5.0f, 6.0f };

auto c  = a + b;              // { 5, 7, 9 }
auto s  = a * 2.0f;           // { 2, 4, 6 }
auto dp = dot(a, b);          // 32
auto cp = cross(a, b);        // { -3, 6, -3 }
auto n  = normalize(a);
auto l  = length(a);          // sqrt(14)
auto t  = angle(a, b);        // radians

auto mask = a < b;            // vector_template<bool, 3> { true, true, true }
```

## Matrices

### Class `matrix_template<Ty, R, C>`

```cpp
template <arithmetic_t Ty, std::size_t R, std::size_t C>
struct matrix_template {
    Ty data[C][R]{};                      // column-major storage

    constexpr Ty&       operator()(std::size_t row, std::size_t col);   // data[col][row]
    constexpr const Ty& operator()(std::size_t row, std::size_t col) const;

    template <typename U> explicit operator matrix_template<U, R, C>() const;

    // iterators (random-access) over all R*C elements
};
```

Storage is **column-major** (`data[col][row]`), matching the OpenGL/Vulkan
convention. Element access is `m(row, col)`. The variadic / initializer-list
constructors accept values in row-major order and transpose them into
column-major storage.

### Operators

| Expression | Result | Notes |
|------------|--------|-------|
| `+m`, `-m` | `mat` | element-wise unary |
| `m + n`, `m - n` | `mat` | element-wise (same shape) |
| `m * n` | `mat` | matrix product `(L x M) * (M x N) -> (L x N)` |
| `m * v` | `vec` | matrix-vector `(R x C) * (C) -> (R)` |
| `v * m` | `vec` | vector-matrix `(R) * (R x C) -> (C)` |
| `m * s`, `m / s` | `mat` | scalar multiply / divide |
| `m += n`, `m -= n`, `m *= n`, `m /= n` | `mat&` | compound |
| `m *= s`, `m /= s` | `mat&` | compound scalar |
| `m == n`, `m != n` | `matrix_template<bool, R, C>` | element-wise |
| `m < n`, `m <= n`, `m > n`, `m >= n` | `matrix_template<bool, R, C>` | element-wise |
| `m && n`, `m || n` | `matrix_template<bool, R, C>` | element-wise logical |

### Free functions

| Function | Description |
|----------|-------------|
| `row(m, i)` | Extract row `i` as `vector_template<Ty, C>`. |
| `col(m, i)` | Extract column `i` as `vector_template<Ty, R>`. |
| `transpose(m)` | `(R x C) -> (C x R)`. |
| `trace(m)` | Sum of diagonal (square matrices). |
| `hadamard(a, b)` | Element-wise product. |
| `outer_product(c, r)` | `(R x 1) * (1 x C) -> (R x C)`. |
| `diagonal(m)` | Diagonal as `vector_template<Ty, S>` (square). |
| `determinant(m)` | For 2x2, 3x3, 4x4. |
| `inverse(m)` | For 2x2, 3x3, 4x4 (undefined for singular). |
| `inverse_transpose(m)` | For 2x2, 3x3, 4x4 (normal transform). |
| `translate(m, v3)` | `m * T(v)`. |
| `scale(m, v3)` | `m * S(v)`. |
| `rotate(m, angle, v3axis)` | `m * R(angle, axis)`. |
| `look_at(eye, center, up)` | View matrix. |
| `perspective(fovy, aspect, znear, zfar)` | Perspective projection. |
| `ortho(l, r, b, t, znear, zfar)` | Orthographic projection. |

### Example

```cpp
mat4x4f id{ 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };

auto model = translate(id, vec3f{ 1.0f, 2.0f, 3.0f });
auto rot   = rotate(id, 1.57079632679f, vec3f{ 0, 0, 1 });
auto inv   = inverse(model);
auto t     = transpose(rot);
auto tr    = trace(model);        // 4 (identity trace)
auto det   = determinant(model);  // 1

auto view = look_at(vec3f{0,5,10}, vec3f{0,0,0}, vec3f{0,1,0});
auto proj = perspective(1.57079632679f, 16.0f / 9.0f, 0.1f, 100.0f);
```

## Quaternions

### Class `quat_template<Ty>`

```cpp
template <arithmetic_t Ty>
struct quat_template {
    Ty w{};   // scalar (real) part
    Ty x{};   // imaginary X
    Ty y{};   // imaginary Y
    Ty z{};   // imaginary Z
};
```

### Operators

| Expression | Result | Notes |
|------------|--------|-------|
| `+q`, `-q` | `quat` | component-wise unary |
| `q + r`, `q - r` | `quat` | component-wise |
| `q * r` | `quat` | Hamilton product (non-commutative) |
| `q * s`, `q / s` | `quat` | scalar |
| `q += r`, `q -= r`, `q *= r`, `q /= r` | `quat&` | compound |

### Free functions

| Function | Description |
|----------|-------------|
| `dot(q, r)` | Dot product of the four components. |
| `conjugate(q)` | `(w, -x, -y, -z)`. |
| `magnitude(q)` | `sqrt(w*w + x*x + y*y + z*z)`. |
| `normalize(q)` | Unit quaternion. |
| `inverse(q)` | `conjugate(q) / dot(q, q)`. |
| `rotate_vector(q, v3)` | Rotate a 3D vector `q * v * q^-1`. |
| `slerp(q1, q2, t)` | Spherical interpolation (constant angular velocity). |
| `lerp(q1, q2, t)` | Normalized linear interpolation (NLERP). |
| `angle_axis(angle, axis)` | Build from angle (radians) + axis. |
| `to_matrix(q)` | Convert to `matrix_template<Ty, 3, 3>`. |
| `matrix4_cast(q)` | Convert to `matrix_template<Ty, 4, 4>`. |
| `quat_cast(m3)` | Convert a 3x3 rotation matrix back to a quaternion. |

### Example

```cpp
quat_template<float> q = angle_axis(1.57079632679f, vec3f{ 0, 0, 1 });

vec3f rotated = rotate_vector(q, vec3f{ 1, 0, 0 });   // { 0, 1, 0 }
mat3x3f m3    = to_matrix(q);
mat4x4f m4    = matrix4_cast(q);
auto   back   = quat_cast(m3);

auto mid = slerp(quat_template<float>{1,0,0,0}, q, 0.5f);
```

> Floating-point types are recommended: `angle_axis`, `slerp`, `magnitude`,
> `normalize`, `inverse` use `sqrt`/`sin`/`cos`/`acos`.

## Vertex input mapping

This is the library's core feature. A `vertex<Ty...>` is a recursive,
tuple-like container (implemented as a compile-time inheritance chain of
`vertex_element<I, Ty>`) that describes a vertex as a sequence of attributes.

### Core types

```cpp
// Vulkan attribute description, directly mappable to VkVertexInputAttributeDescription
struct vertex_layout {
    std::uint32_t location;      // shader binding location (location = N)
    std::uint32_t vkformat;      // Vulkan VkFormat numeric value
    std::uint32_t count;         // 1 for scalar/vector, C for matrices
    std::uint32_t offset;        // byte offset from the start of the struct
    std::uint32_t offset_step;   // byte stride between consecutive elements
};

template <typename... Ty> using vertex = vertex_template<0, Ty...>;

// Compile-time traits for any scalar/vector/matrix type:
//   vertex_traits<T>::count        -> 1, or number of columns for matrices
//   vertex_traits<T>::format       -> Vulkan VkFormat numeric value
//   vertex_traits<T>::offset_step  -> byte size of one element / column
```

### `VkFormat` mapping

Base scalar formats and the vector/matrix formats that derive from them:

| Type    | `format` | Vulkan enum                 | Type    | `format` | Vulkan enum                 |
|---------|----------|-----------------------------|---------|----------|-----------------------------|
| `vf32`  | 100      | `VK_FORMAT_R32_SFLOAT`      | `vf64`  | 112      | `VK_FORMAT_R64_SFLOAT`      |
| `vi32`  | 99       | `VK_FORMAT_R32_SINT`        | `vu32`  | 98       | `VK_FORMAT_R32_UINT`        |
| `vi16`  | 75       | `VK_FORMAT_R16_SINT`        | `vu16`  | 74       | `VK_FORMAT_R16_UINT`        |
| `vi08`  | 14       | `VK_FORMAT_R8_SINT`         | `vu08`  | 13       | `VK_FORMAT_R8_UINT`         |
| `vec2f32` | 103   | `VK_FORMAT_R32G32_SFLOAT`   | `vec2f64`| 115    | `VK_FORMAT_R64G64_SFLOAT`   |
| `vec3f32` | 106   | `VK_FORMAT_R32G32B32_SFLOAT`| `vec3f64`| 118    | `VK_FORMAT_R64G64B64_SFLOAT`|
| `vec4f32` | 109   | `VK_FORMAT_R32G32B32A32_SFLOAT` | `vec4f64` | 121 | `VK_FORMAT_R64G64B64A64_SFLOAT` |

The full pattern covers every scalar type at each vector width (2/3/4
components). A matrix `matRxC<T>` reports the format of its column vector
`vecR<T>`, `count == C` (number of columns), and `offset_step == C * sizeof(T)`
bytes per column.

### Generating a layout

```cpp
using Vertex = vertex<vec3f, vec3f, vec2f>;   // position, normal, uv

// 1. Element access, like std::get
Vertex v{ vec3f{0,0,0}, vec3f{0,1,0}, vec2f{0,0} };
auto& position = get<0>(v);

// 2. Byte offsets
auto offsets = compute_offset<vec3f, vec3f, vec2f>();   // { 0, 12, 24 }
auto offsets2 = compute_offset(v);                       // same, from an instance

// 3. Full layout, ready for VkVertexInputAttributeDescription
auto layout = MakeVertexLayout<vec3f, vec3f, vec2f>();
for (auto const& a : layout) {
    // a.location, a.vkformat, a.count, a.offset, a.offset_step
}
```

`MakeVertexLayout` assigns locations sequentially starting at 0, advancing by
`count` per attribute (so a `mat4x4` consumes four consecutive locations). The
result is a `std::vector<vertex_layout>` that maps one-to-one onto a
`VkVertexInputAttributeDescription` array.

### `vertex_traits` examples

```cpp
static_assert(vertex_traits<vf32>::count       == 1);
static_assert(vertex_traits<vf32>::format      == 100);
static_assert(vertex_traits<vf32>::offset_step == 4);
static_assert(vertex_traits<vec3f>::count      == 1);
static_assert(vertex_traits<vec3f>::format     == 106);
static_assert(vertex_traits<mat4x4f>::count    == 4);   // four columns -> four locations
```

## String formatting

`vtx/vtxstring.hpp` provides `operator<<` and a `std::formatter`
specialization for `vector_template`. Output format is `[e0, e1, ..., eN-1]`.

```cpp
vec3f v{ 1.0f, 2.0f, 3.0f };
std::cout << v;                        // [1, 2, 3]
auto s = std::format("{}", v);         // "[1, 2, 3]"
```

## SIMD extensions

`vtx/vtxsimd.hpp` is opt-in, guarded by the `VERTEX_ENABLE_EXTENSIONS` macro
(defined **before** including `vertex.hpp`). When enabled it accelerates, using
AVX2 intrinsics:

- matrix addition (`vf32` / `vf64`)
- matrix subtraction (`vf32` / `vf64`)
- 4x4 matrix inverse (`vf32` / `vf64`, experimental, additionally gated by
  `VERTEX_ENABLE_TESTEXT`)

```cpp
#define VERTEX_ENABLE_EXTENSIONS
#include "vertex.hpp"
```

Requirements and caveats:

- Requires AVX2 and the matching compiler flag (`-mavx2` / `/arch:AVX2`).
- Matrix dimensions must align to the SIMD lane width: `vf32` needs a total
  element count that is a multiple of 8, `vf64` a multiple of 4.
- When the macro is not defined, scalar fallbacks are used throughout.

## Building and running the tests

`main.cpp` is a dependency-free, macro-based test harness (`EXPECT` macro)
covering every template and type, including edge cases.

```bash
cmake -S . -B build
cmake --build build
./build/vertex_test          # or build/Debug/vertex_test.exe on MSVC
```

A passing run prints `ALL TESTS PASSED` and exits with code 0.

## Project structure

```
vertex.hpp            Entry header (aggregates everything)
vtx/
  vtxbasic.hpp        Scalar aliases, arithmetic_t concept, FORCEINLINE/CONSTEXPR macros
  vtxvecx.hpp         vector_template<T, N> + vector free functions
  vtxvec2.hpp         vec2 aliases
  vtxvec3.hpp         vec3 aliases
  vtxvec4.hpp         vec4 aliases
  vtxmatcxr.hpp       matrix_template<T, R, C> + matrix free functions
  vtxmat2x2.hpp       mat2x2 aliases
  vtxmat2x3.hpp       mat2x3 aliases
  vtxmat2x4.hpp       mat2x4 aliases
  vtxmat3x2.hpp       mat3x2 aliases
  vtxmat3x3.hpp       mat3x3 aliases
  vtxmat3x4.hpp       mat3x4 aliases
  vtxmat4x2.hpp       mat4x2 aliases
  vtxmat4x3.hpp       mat4x3 aliases
  vtxmat4x4.hpp       mat4x4 aliases
  vtxquat.hpp         quat_template<T> + quaternion operations
  vtxtraits.hpp       vertex_traits<T> -> Vulkan format / count / offset_step
  vtxtuple.hpp        vertex<Ty...>, get, compute_offset, MakeVertexLayout
  vtxstring.hpp       ostream / std::format support for vectors
  vtxsimd.hpp         AVX2 matrix add/sub + 4x4 inverse (optional)
main.cpp              Production-readiness test suite
CMakeLists.txt        Test build
```

## Notes and caveats

- `normalize` on a zero-length vector / quaternion, and `inverse` on a singular
  matrix, are undefined (division by zero). Check `length` / `determinant`
  first when inputs are untrusted.
- `refract` returns the zero vector on total internal reflection (`k < 0`).
- Division on integral types truncates (C++ integer semantics).
- Comparison operators (`<`, `==`, ...) return `vector_template<bool, N>` /
  `matrix_template<bool, R, C>`, not a scalar `bool`. Combine with `&&` / `||`
  for per-element masking.
- Quaternion functions using `sqrt`/`sin`/`cos` are intended for floating-point
  types.

## License

[MIT](LICENSE) — Copyright (c) 2026 SpectreFestival.
