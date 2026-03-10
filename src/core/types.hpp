/**
 * @file types.hpp
 * @brief Basic type definitions and tensor aliases for mpfem
 * 
 * This file provides the foundational type definitions used throughout
 * the mpfem library, including:
 * - Index types for mesh entities and DoFs
 * - Tensor type aliases using Eigen
 * - Physical constants and dimension utilities
 */

#ifndef MPFEM_CORE_TYPES_HPP
#define MPFEM_CORE_TYPES_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mpfem {

// ============================================================
// Index Types
// ============================================================

/// Global index type for mesh entities and DoFs
using Index = int32_t;

/// Local index type for element-local operations
using LocalIndex = int16_t;

/// Size type for containers
using SizeType = size_t;

/// Invalid index constant
constexpr Index INVALID_INDEX = -1;

// ============================================================
// Scalar Types
// ============================================================

/// Default floating-point type (double precision)
using Scalar = double;

/// Complex scalar type
using ComplexScalar = std::complex<double>;

// ============================================================
// Vector Types (Dynamic Size)
// ============================================================

/// Dynamic-size vector
using VectorX = Eigen::VectorX<Scalar>;

/// Dynamic-size complex vector
using VectorXc = Eigen::VectorX<ComplexScalar>;

// ============================================================
// Fixed-Size Vector Types (Dimension Templates)
// ============================================================

/// Fixed-size vector for dim-dimensional space
template <int dim>
using Vector = Eigen::Vector<Scalar, dim>;

/// 1D vector
using Vector1 = Vector<1>;
/// 2D vector
using Vector2 = Vector<2>;
/// 3D vector
using Vector3 = Vector<3>;

/// Point in dim-dimensional space (alias for Vector)
template <int dim>
using Point = Vector<dim>;

// ============================================================
// Matrix Types (Dynamic Size)
// ============================================================

/// Dynamic-size matrix
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

/// Dynamic-size complex matrix
using MatrixXc = Eigen::Matrix<ComplexScalar, Eigen::Dynamic, Eigen::Dynamic>;

// ============================================================
// Fixed-Size Matrix Types (Dimension Templates)
// ============================================================

/// Fixed-size matrix
template <int rows, int cols>
using Matrix = Eigen::Matrix<Scalar, rows, cols>;

/// Square matrix in dim-dimensional space
template <int dim>
using SquareMatrix = Matrix<dim, dim>;

/// 2x2 matrix
using Matrix2 = SquareMatrix<2>;
/// 3x3 matrix
using Matrix3 = SquareMatrix<3>;

// ============================================================
// Tensor Types (Rank-3 and Rank-4)
// ============================================================

/**
 * @brief Rank-3 tensor storage
 * 
 * Stored as a 1D array with row-major ordering.
 * Dimensions: (dim1, dim2, dim3)
 */
template <int dim1, int dim2, int dim3>
struct Tensor3 {
    std::array<Scalar, dim1 * dim2 * dim3> data{};

    Scalar& operator()(int i, int j, int k) {
        return data[i * dim2 * dim3 + j * dim3 + k];
    }

    const Scalar& operator()(int i, int j, int k) const {
        return data[i * dim2 * dim3 + j * dim3 + k];
    }

    void setZero() { data.fill(0.0); }
};

/**
 * @brief Rank-4 tensor storage
 * 
 * Stored as a 1D array with row-major ordering.
 * Dimensions: (dim1, dim2, dim3, dim4)
 * Used for stiffness tensors in solid mechanics.
 */
template <int dim1, int dim2, int dim3, int dim4>
struct Tensor4 {
    std::array<Scalar, dim1 * dim2 * dim3 * dim4> data{};

    Scalar& operator()(int i, int j, int k, int l) {
        return data[((i * dim2 + j) * dim3 + k) * dim4 + l];
    }

    const Scalar& operator()(int i, int j, int k, int l) const {
        return data[((i * dim2 + j) * dim3 + k) * dim4 + l];
    }

    void setZero() { data.fill(0.0); }
};

/// 3x3x3 Rank-3 tensor
using Tensor3_3d = Tensor3<3, 3, 3>;
/// 3x3x3x3 Rank-4 tensor (stiffness tensor)
using Tensor4_3d = Tensor4<3, 3, 3, 3>;

// ============================================================
// Sparse Matrix Types
// ============================================================

/// Sparse matrix in CSR format
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, Index>;

/// Sparse vector
using SparseVector = Eigen::SparseVector<Scalar, Eigen::RowMajor, Index>;

// ============================================================
// Physical Constants
// ============================================================

namespace constants {

/// Boltzmann constant [J/K]
constexpr Scalar BOLTZMANN = 1.380649e-23;

/// Stefan-Boltzmann constant [W/(m^2·K^4)]
constexpr Scalar STEFAN_BOLTZMANN = 5.670374419e-8;

/// Vacuum permittivity [F/m]
constexpr Scalar EPSILON_0 = 8.8541878128e-12;

/// Vacuum permeability [H/m]
constexpr Scalar MU_0 = 1.25663706212e-6;

/// Absolute zero [K]
constexpr Scalar ABSOLUTE_ZERO = 0.0;

/// Reference temperature [K] (20°C)
constexpr Scalar T_REF = 293.15;

}  // namespace constants

// ============================================================
// Utility Type Aliases
// ============================================================

/// Span of indices
using IndexSpan = std::span<const Index>;

/// Span of scalars
using ScalarSpan = std::span<const Scalar>;

/// Array of indices
using IndexArray = std::vector<Index>;

/// Array of scalars
using ScalarArray = std::vector<Scalar>;

// ============================================================
// Dimension Utilities
// ============================================================

/// Spatial dimension enumeration
enum class Dimension : int {
    Dim1 = 1,
    Dim2 = 2,
    Dim3 = 3
};

/// Convert Dimension enum to int
constexpr int to_int(Dimension dim) {
    return static_cast<int>(dim);
}

// ============================================================
// Geometry Type Enumeration
// ============================================================

/// Geometry/element shape types
enum class GeometryType : int {
    Point = 0,
    Segment = 1,    ///< Line segment (1D)
    Triangle = 2,   ///< Triangle (2D)
    Quadrilateral = 3, ///< Quadrilateral (2D)
    Tetrahedron = 4,   ///< Tetrahedron (3D)
    Hexahedron = 5,    ///< Hexahedron (3D)
    Wedge = 6,         ///< Prism/Wedge (3D)
    Pyramid = 7        ///< Pyramid (3D)
};

/// Get dimension of geometry type
constexpr int geometry_dimension(GeometryType type) {
    switch (type) {
        case GeometryType::Point:
            return 0;
        case GeometryType::Segment:
            return 1;
        case GeometryType::Triangle:
        case GeometryType::Quadrilateral:
            return 2;
        case GeometryType::Tetrahedron:
        case GeometryType::Hexahedron:
        case GeometryType::Wedge:
        case GeometryType::Pyramid:
            return 3;
        default:
            return -1;
    }
}

/// Get number of vertices for geometry type (linear elements)
constexpr int num_vertices(GeometryType type) {
    switch (type) {
        case GeometryType::Point:
            return 1;
        case GeometryType::Segment:
            return 2;
        case GeometryType::Triangle:
            return 3;
        case GeometryType::Quadrilateral:
            return 4;
        case GeometryType::Tetrahedron:
            return 4;
        case GeometryType::Hexahedron:
            return 8;
        case GeometryType::Wedge:
            return 6;
        case GeometryType::Pyramid:
            return 5;
        default:
            return 0;
    }
}

/// Get string name for geometry type
inline const char* geometry_name(GeometryType type) {
    switch (type) {
        case GeometryType::Point:
            return "Point";
        case GeometryType::Segment:
            return "Segment";
        case GeometryType::Triangle:
            return "Triangle";
        case GeometryType::Quadrilateral:
            return "Quadrilateral";
        case GeometryType::Tetrahedron:
            return "Tetrahedron";
        case GeometryType::Hexahedron:
            return "Hexahedron";
        case GeometryType::Wedge:
            return "Wedge";
        case GeometryType::Pyramid:
            return "Pyramid";
        default:
            return "Unknown";
    }
}

}  // namespace mpfem

#endif  // MPFEM_CORE_TYPES_HPP
