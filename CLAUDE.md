# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**Vertex** — a header-only **C++20** math library (namespace `vtx`) with a
Vulkan vertex-input abstraction layer. Provides vectors / matrices /
quaternions, plus a `vertex<Ty...>` container that auto-generates
`VkVertexInputAttributeDescription` layouts at compile time.

Users include the single entry header `vertex.hpp`.

## Build (MSVC, Windows)

- Primary build: Visual Studio solution `vtxtext.slnx` / project `vtxtext.vcxproj`.
  - Configuration: **Release x64**, toolset **v145**, language **C++20** (`stdcpp20`).
- A minimal `CMakeLists.txt` also exists (builds the `vertex_test` executable).

### Benchmark

`bench.cpp` is a Google Benchmark suite comparing **vtx / glm / eigen** across
all vector and matrix operations, printing a `vtx/glm` and `vtx/eigen` ratio
table. Run the built `x64\Release\vtxtext.exe`.

- Links prebuilt Google Benchmark via `#pragma comment(lib, ...)`:
  `benchmark.lib` + `benchmark_main.lib`, plus `shlwapi.lib` / `advapi32.lib`.
- `GLM_FORCE_INTRINSICS` is defined (glm SIMD enabled).
- Eigen is included from `./Eigen/Dense`.
- Every benchmark perturbs **all** elements of its input each iteration to
  prevent the compiler from constant-folding the computation (a single-element
  perturbation lets the compiler fold scalar reductions like `determinant` /
  `trace` / `dot` into a constant formula).

## Directory layout

```
vertex.hpp          Entry header (aggregates vtx/)
vtx/                Library headers (namespace vtx)
  vtxbasic.hpp      Scalar aliases, arithmetic_t concept, macros
  vtxvecx.hpp       vector_template<T, N> + vector free functions
  vtxvec2/3/4.hpp   vec2/3/4 aliases
  vtxmatcxr.hpp     matrix_template<T, R, C> + matrix free functions
  vtxmat2x2..4x4    mat aliases
  vtxquat.hpp       quat_template<T> + quaternion ops
  vtxtraits.hpp     vertex_traits<T> (Vulkan format/count/offset_step)
  vtxtuple.hpp      vertex<Ty...>, get, compute_offset, MakeVertexLayout
  vtxstring.hpp     ostream / std::format support
  vtxsimd.hpp       AVX2 add/sub + 4x4 inverse (adjugate/cofactor, opt-in)
bench.cpp           Benchmark: vtx vs glm vs eigen
main.cpp            (currently a stub; benchmark has its own main())
benchmark/ glm/ Eigen/   Vendored third-party sources
```

## Conventions and gotchas

- **Pure ASCII source only.** This machine builds with MSVC code page 936
  (GBK). Non-ASCII characters in `.cpp`/`.hpp` (Chinese comments, arrows, `π`,
  etc.) trigger warning C4819 and can corrupt parsing. Use English comments.
- **Column-major matrices.** `matrix_template<T, R, C>` stores `data[col][row]`
  and is indexed `m(row, col)`. The variadic constructor takes values in
  row-major order.
- **Header-only.** No compiled library; everything is in headers.
- **C++20 required** (concepts, `std::format`, fold expressions, `constexpr`).
- Define **`VERTEX_ENABLE_EXTENSIONS`** before including `vertex.hpp` to enable
  the AVX2 SIMD path (`vtxsimd.hpp`); otherwise scalar fallbacks are used.

### SIMD 4x4 inverse (`vtxsimd.hpp`)

The SIMD inverse uses the **adjugate (cofactor) method**, not LU. Two
correctness rules that are easy to get wrong:

- **Determinant needs a full horizontal sum.** A single `_mm256_hadd_pd` (or
  `_mm_hadd_ps`) only sums adjacent pairs; summing 4 doubles/lanes correctly
  requires `extractf128 + add + hadd` (or two `hadd` for SSE). A single `hadd`
  + broadcast silently drops half the determinant terms.
- **Do NOT transpose when storing.** `data[col][row]` column-major storage
  already performs the adjugate transpose (`inv[i][j] = C[j][i]`). Store each
  cofactor row directly as a column; the classic `unpacklo/hi +
  permute2f128` transpose dance scrambles the result.

## Performance notes (from the benchmark)

Where vtx currently stands vs glm/eigen:

- **vtx wins**: integer matrices (Eigen's integer `mat*mat` is 20–40x slower —
  no small-size specialization), `vec4` float/double (Eigen's 16-byte-aligned
  SSE has overhead for 4 lanes), f64 matrices, 4x4 f32 inverse.
- **vtx loses**:
  1. **Float small vectors (vec2/vec3)** — ~2 ns vs Eigen ~0.6 ns. Root cause:
     `operator+`/`dot`/`length` wrap the work in a `lambda + index_sequence +
     fold` and route it through the variadic constructor; MSVC does not
     scalarize this for float. Fix direction: write `data` directly.
  2. **Scalar reductions (trace/dot)** — same root cause.
  3. **determinant** — vtx uses a 72-mul Laplace expansion; Eigen uses the
     30-mul "Martin Costabel" trick (`bruteforce_det4_helper`).

Highest-leverage optimization: fix the float vector scalarization first (it
unblocks vec2/vec3, dot, length, trace at once).
