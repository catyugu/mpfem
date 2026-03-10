/**
 * @file types.hpp
 * @brief Core type definitions for mpfem
 */

#ifndef MPFEM_CORE_TYPES_HPP
#define MPFEM_CORE_TYPES_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace mpfem {

// ============================================================
// Index Types
// ============================================================

using Index = std::int64_t;
using SizeType = std::size_t;

using IndexArray = std::vector<Index>;
using ScalarArray = std::vector<double>;

// ============================================================
// Scalar Type
// ============================================================

using Scalar = double;

// ============================================================
// Eigen-based Matrix and Vector Types
// ============================================================

template <int Rows, int Cols>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols>;

template <int Size>
using Vector = Eigen::Matrix<Scalar, Size, 1>;

template <int Size>
using SquareMatrix = Eigen::Matrix<Scalar, Size, Size>;

// Dynamic size versions
using DynamicVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using DynamicMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

// Sparse matrix type
using SparseMatrix = Eigen::SparseMatrix<Scalar>;

// Common fixed-size types
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;
using Matrix2 = SquareMatrix<2>;
using Matrix3 = SquareMatrix<3>;

// Point type (3D coordinates, but can represent 2D by ignoring z)
template <int Dim>
using Point = Vector<Dim>;

// ============================================================
// Tensor Template
// ============================================================

template <int Rank, int Dim>
struct Tensor;

/// Rank-1 Tensor (Vector)
template <int Dim>
struct Tensor<1, Dim> : public Vector<Dim> {
    using Base = Vector<Dim>;
    using Base::Base;

    Tensor() : Base() { Base::setZero(); }

    static Tensor Zero() {
        Tensor t;
        t.Base::setZero();
        return t;
    }
};

/// Rank-2 Tensor (Matrix)
template <int Dim>
struct Tensor<2, Dim> : public SquareMatrix<Dim> {
    using Base = SquareMatrix<Dim>;
    using Base::Base;

    Tensor() : Base() { Base::setZero(); }

    static Tensor Identity() {
        Tensor t;
        t.Base::setIdentity();
        return t;
    }

    static Tensor Zero() {
        Tensor t;
        t.Base::setZero();
        return t;
    }
};

/// Rank-4 Tensor (for elasticity tensor)
template <int Dim>
struct Tensor<4, Dim> {
    Scalar data[Dim][Dim][Dim][Dim];

    Tensor() {
        for (int i = 0; i < Dim; ++i)
            for (int j = 0; j < Dim; ++j)
                for (int k = 0; k < Dim; ++k)
                    for (int l = 0; l < Dim; ++l)
                        data[i][j][k][l] = 0.0;
    }

    Scalar& operator()(int i, int j, int k, int l) {
        return data[i][j][k][l];
    }

    Scalar operator()(int i, int j, int k, int l) const {
        return data[i][j][k][l];
    }

    static Tensor Zero() {
        return Tensor();
    }
};

// ============================================================
// Geometry Type Enumeration
// ============================================================

enum class GeometryType {
    Invalid = -1,
    Point = 0,
    Segment = 1,
    Triangle = 2,
    Quadrilateral = 3,
    Tetrahedron = 4,
    Hexahedron = 5,
    Wedge = 6,
    Pyramid = 7
};

// ============================================================
// Update Flags for FEValues
// ============================================================

enum class UpdateFlags : int {
    None = 0,
    UpdateJxW = 1 << 0,
    UpdateGradients = 1 << 1,
    UpdateValues = 1 << 2,
    UpdateQuadraturePoints = 1 << 3,
    UpdateNormals = 1 << 4,
    UpdateDefault = UpdateValues | UpdateGradients | UpdateJxW,
    UpdateAll = UpdateValues | UpdateGradients | UpdateJxW | UpdateQuadraturePoints | UpdateNormals
};

inline UpdateFlags operator|(UpdateFlags a, UpdateFlags b) {
    return static_cast<UpdateFlags>(static_cast<int>(a) | static_cast<int>(b));
}

inline UpdateFlags operator&(UpdateFlags a, UpdateFlags b) {
    return static_cast<UpdateFlags>(static_cast<int>(a) & static_cast<int>(b));
}

inline bool has_flag(UpdateFlags flags, UpdateFlags flag) {
    return (flags & flag) == flag;
}

// ============================================================
// Constants
// ============================================================

constexpr Index InvalidIndex = -1;

}  // namespace mpfem

#endif  // MPFEM_CORE_TYPES_HPP