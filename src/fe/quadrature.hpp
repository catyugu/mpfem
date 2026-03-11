#ifndef MPFEM_QUADRATURE_HPP
#define MPFEM_QUADRATURE_HPP

#include "mesh/geometry.hpp"
#include "core/types.hpp"
#include <vector>
#include <array>
#include <cmath>

namespace mpfem {

/**
 * @brief Quadrature rule for numerical integration on reference elements.
 * 
 * Provides Gauss-Legendre quadrature points and weights for various
 * element geometries.
 */
class QuadratureRule {
public:
    /// Default constructor
    QuadratureRule() = default;

    /// Construct from integration points
    explicit QuadratureRule(std::vector<IntegrationPoint> points)
        : points_(std::move(points)) {}

    /// Construct from separate arrays
    QuadratureRule(const std::vector<Real>& xi,
                   const std::vector<Real>& weights,
                   int dim = 1) {
        points_.reserve(xi.size());
        for (size_t i = 0; i < xi.size(); ++i) {
            points_.push_back(IntegrationPoint(xi[i], 0.0, 0.0, weights[i]));
        }
    }

    // -------------------------------------------------------------------------
    // Access
    // -------------------------------------------------------------------------

    /// Get number of integration points
    int size() const { return static_cast<int>(points_.size()); }

    /// Get integration point by index
    const IntegrationPoint& operator[](int i) const { return points_[i]; }
    IntegrationPoint& operator[](int i) { return points_[i]; }

    /// Get all points
    const std::vector<IntegrationPoint>& points() const { return points_; }

    /// Get points (mutable)
    std::vector<IntegrationPoint>& points() { return points_; }

    // -------------------------------------------------------------------------
    // Iterator support
    // -------------------------------------------------------------------------

    auto begin() const { return points_.begin(); }
    auto end() const { return points_.end(); }
    auto begin() { return points_.begin(); }
    auto end() { return points_.end(); }

private:
    std::vector<IntegrationPoint> points_;
};

// =============================================================================
// Gauss-Legendre Quadrature (1D)
// =============================================================================

namespace gauss {

/**
 * @brief Get 1D Gauss-Legendre points and weights on [-1, 1].
 * @param order Number of points (1-10 supported)
 */
std::pair<std::vector<Real>, std::vector<Real>> get1D(int order);

/**
 * @brief Get Gauss-Legendre points on [-1, 1].
 */
std::vector<Real> getPoints(int order);

/**
 * @brief Get Gauss-Legendre weights on [-1, 1].
 */
std::vector<Real> getWeights(int order);

}  // namespace gauss

// =============================================================================
// Quadrature for specific geometries
// =============================================================================

namespace quadrature {

/**
 * @brief Get quadrature rule for a segment.
 * @param order Number of points in 1D rule
 */
QuadratureRule getSegment(int order);

/**
 * @brief Get quadrature rule for a triangle.
 * @param order Integration order (1-5 supported)
 * 
 * Uses Dunavant rules for triangles.
 */
QuadratureRule getTriangle(int order);

/**
 * @brief Get quadrature rule for a square (tensor product).
 * @param order Number of points per direction
 */
QuadratureRule getSquare(int order);

/**
 * @brief Get quadrature rule for a tetrahedron.
 * @param order Integration order (1-5 supported)
 */
QuadratureRule getTetrahedron(int order);

/**
 * @brief Get quadrature rule for a cube (tensor product).
 * @param order Number of points per direction
 */
QuadratureRule getCube(int order);

/**
 * @brief Get quadrature rule for any geometry.
 * @param geom Geometry type
 * @param order Integration order
 */
QuadratureRule get(Geometry geom, int order);

}  // namespace quadrature

// =============================================================================
// Triangle quadrature rules (Dunavant)
// =============================================================================

namespace dunavant {

/// Triangle quadrature point count for each order
inline constexpr int triangleOrderToPoints[] = {1, 3, 4, 6, 7, 12, 13, 16, 19, 25, 27, 33, 37, 42, 48, 52};

/**
 * @brief Get Dunavant triangle quadrature rule.
 * @param order Order (1-15 supported)
 */
QuadratureRule getTriangle(int order);

}  // namespace dunavant

// =============================================================================
// Tetrahedron quadrature rules
// =============================================================================

namespace tet_quadrature {

/**
 * @brief Get tetrahedron quadrature rule.
 * @param order Order (1-5 supported)
 */
QuadratureRule getTetrahedron(int order);

}  // namespace tet_quadrature

// =============================================================================
// Inline implementations for gauss
// =============================================================================

namespace gauss {

inline std::vector<Real> getPoints(int order) {
    return get1D(order).first;
}

inline std::vector<Real> getWeights(int order) {
    return get1D(order).second;
}

}  // namespace gauss

// =============================================================================
// Inline implementations for quadrature
// =============================================================================

namespace quadrature {

inline QuadratureRule getSegment(int order) {
    auto [xi, w] = gauss::get1D(order);
    QuadratureRule rule;
    for (size_t i = 0; i < xi.size(); ++i) {
        rule.points().push_back(IntegrationPoint(xi[i], 0.0, 0.0, w[i]));
    }
    return rule;
}

inline QuadratureRule getSquare(int order) {
    auto [xi1d, w1d] = gauss::get1D(order);
    QuadratureRule rule;
    for (size_t i = 0; i < xi1d.size(); ++i) {
        for (size_t j = 0; j < xi1d.size(); ++j) {
            rule.points().push_back(IntegrationPoint(
                xi1d[i], xi1d[j], 0.0, w1d[i] * w1d[j]));
        }
    }
    return rule;
}

inline QuadratureRule getCube(int order) {
    auto [xi1d, w1d] = gauss::get1D(order);
    QuadratureRule rule;
    for (size_t i = 0; i < xi1d.size(); ++i) {
        for (size_t j = 0; j < xi1d.size(); ++j) {
            for (size_t k = 0; k < xi1d.size(); ++k) {
                rule.points().push_back(IntegrationPoint(
                    xi1d[i], xi1d[j], xi1d[k], w1d[i] * w1d[j] * w1d[k]));
            }
        }
    }
    return rule;
}

inline QuadratureRule get(Geometry geom, int order) {
    switch (geom) {
        case Geometry::Segment:
            return getSegment(order);
        case Geometry::Triangle:
            return getTriangle(order);
        case Geometry::Square:
            return getSquare(order);
        case Geometry::Tetrahedron:
            return getTetrahedron(order);
        case Geometry::Cube:
            return getCube(order);
        default:
            return QuadratureRule();
    }
}

}  // namespace quadrature

}  // namespace mpfem

#endif  // MPFEM_QUADRATURE_HPP