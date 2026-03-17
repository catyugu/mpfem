#ifndef MPFEM_TYPES_HPP
#define MPFEM_TYPES_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <span>

namespace mpfem {

// =============================================================================
// Basic types
// =============================================================================

using Index = std::int32_t;
using LocalIndex = std::int16_t;
using Real = double;

/// Invalid index constant
inline constexpr Index InvalidIndex = -1;

// =============================================================================
// Eigen type aliases
// =============================================================================

// Dense vector types
using VectorX = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Vector = VectorX;  // Alias for convenience
using Vector2 = Eigen::Matrix<Real, 2, 1>;
using Vector3 = Eigen::Matrix<Real, 3, 1>;

// Dense matrix types
using MatrixX = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using Matrix = MatrixX;  // Alias for convenience
using Matrix2 = Eigen::Matrix<Real, 2, 2>;
using Matrix3 = Eigen::Matrix<Real, 3, 3>;
using Matrix32 = Eigen::Matrix<Real, 3, 2>;  // Jacobian for 2D element in 3D space
using Matrix23 = Eigen::Matrix<Real, 2, 3>;

// Sparse matrix type (Eigen-based, for general use)
using EigenSparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;

// Triplet for sparse matrix construction
using Triplet = Eigen::Triplet<Real>;

// =============================================================================
// Geometry related types
// =============================================================================

/// Maximum spatial dimension
inline constexpr LocalIndex MaxDim = 3;

/// Maximum vector field dimension (for displacement, velocity, etc.)
inline constexpr LocalIndex MaxVectorDim = 3;

// =============================================================================
// Element-related constants (for fixed-size stack arrays)
// =============================================================================

/// Maximum number of nodes per volume element (for fixed-size arrays)
/// hex27: 3x3x3 = 27 nodes for second-order hexahedron
inline constexpr LocalIndex MaxNodesPerElement = 27;

/// Maximum number of nodes per boundary element
/// quad9: 3x3 = 9 nodes for second-order quadrilateral
inline constexpr LocalIndex MaxNodesPerBdrElement = 9;

/// Maximum number of vertices per element (first-order)
inline constexpr LocalIndex MaxVerticesPerElement = 8;  // hex8

/// Maximum DOFs per volume element (scalar field)
inline constexpr LocalIndex MaxDofsPerElement = MaxNodesPerElement;

/// Maximum DOFs per boundary element (scalar field)
inline constexpr LocalIndex MaxDofsPerBdrElement = MaxNodesPerBdrElement;

/// Maximum DOFs per volume element (vector field, e.g., displacement)
inline constexpr LocalIndex MaxVectorDofsPerElement = MaxDofsPerElement * MaxVectorDim;

/// Maximum DOFs per boundary element (vector field)
inline constexpr LocalIndex MaxVectorDofsPerBdrElement = MaxDofsPerBdrElement * MaxVectorDim;

/// Maximum quadrature points per element
inline constexpr LocalIndex MaxQuadraturePoints = 27;

/// Maximum strain/stress components (3D: 6 independent components)
inline constexpr LocalIndex MaxStrainComponents = 6;

/// Coordinate array type
using CoordArray = std::array<Real, MaxDim>;

/// Node index array for an element
using NodeIndices = std::vector<Index>;

/// Fixed-size node array (for stack allocation)
template <LocalIndex N>
using FixedNodeArray = std::array<Index, N>;

// =============================================================================
// Quadrature related types
// =============================================================================

/// Integration point in reference coordinates
struct IntegrationPoint {
    Real xi = 0.0;      ///< Reference coordinate (first parametric direction)
    Real eta = 0.0;     ///< Reference coordinate (second parametric direction)
    Real zeta = 0.0;    ///< Reference coordinate (third parametric direction)
    Real weight = 0.0;  ///< Quadrature weight
    
    IntegrationPoint() = default;
    IntegrationPoint(Real x, Real w) : xi(x), weight(w) {}
    IntegrationPoint(Real x, Real y, Real w) : xi(x), eta(y), weight(w) {}
    IntegrationPoint(Real x, Real y, Real z, Real w) : xi(x), eta(y), zeta(z), weight(w) {}
};

// Note: QuadratureRule is defined in fe/quadrature.hpp

// =============================================================================
// Element matrix types
// =============================================================================

/// Element stiffness matrix (local matrix)
using ElementMatrix = MatrixX;

/// Element vector (local load vector)
using ElementVector = VectorX;

// =============================================================================
// Boundary types
// =============================================================================

/// Boundary type enumeration
enum class BoundaryType : std::uint8_t {
    Interior,   ///< Internal boundary (shared by multiple elements)
    Exterior    ///< External boundary (boundary of the domain)
};

/// Physical dimension enumeration
enum class Dim : std::uint8_t {
    D1 = 1,
    D2 = 2,
    D3 = 3
};

// =============================================================================
// Utility functions
// =============================================================================

/// Convert dimension enum to integer
inline constexpr std::size_t dimToInt(Dim d) {
    return static_cast<std::size_t>(d);
}

/// Get number of components for a given dimension
inline constexpr std::size_t numComponents(Dim dim) {
    return dimToInt(dim);
}

}  // namespace mpfem

// =============================================================================
// Eigen vector output for debugging
// =============================================================================

namespace Eigen {
template <typename Derived>
std::ostream& operator<<(std::ostream& os, const MatrixBase<Derived>& m) {
    os << "[";
    for (Index i = 0; i < m.rows(); ++i) {
        if (i > 0) os << "; ";
        for (Index j = 0; j < m.cols(); ++j) {
            if (j > 0) os << " ";
            os << m(i, j);
        }
    }
    os << "]";
    return os;
}
}  // namespace Eigen

#endif  // MPFEM_TYPES_HPP
