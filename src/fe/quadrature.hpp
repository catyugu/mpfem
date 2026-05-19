#ifndef MPFEM_QUADRATURE_HPP
#define MPFEM_QUADRATURE_HPP

#include "core/geometry.hpp"
#include "core/types.hpp"
#include <vector>

namespace mpfem {

    class QuadratureRule {
    public:
        QuadratureRule() = default;

        explicit QuadratureRule(std::vector<IntegrationPoint> points)
            : points_(std::move(points)) { }

        int size() const { return static_cast<int>(points_.size()); }
        const IntegrationPoint& operator[](int i) const { return points_[i]; }
        IntegrationPoint& operator[](int i) { return points_[i]; }

        const std::vector<IntegrationPoint>& points() const { return points_; }
        std::vector<IntegrationPoint>& points() { return points_; }

        auto begin() const { return points_.begin(); }
        auto end() const { return points_.end(); }
        auto begin() { return points_.begin(); }
        auto end() { return points_.end(); }

    private:
        std::vector<IntegrationPoint> points_;
    };

    namespace quadrature {
        QuadratureRule get(Geometry geom, int order);
        QuadratureRule getSegment(int order);
        QuadratureRule getTriangle(int order);
        QuadratureRule getSquare(int order);
        QuadratureRule getTetrahedron(int order);
        QuadratureRule getCube(int order);
    } // namespace quadrature

} // namespace mpfem

#endif // MPFEM_QUADRATURE_HPP