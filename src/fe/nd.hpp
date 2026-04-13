#ifndef MPFEM_ND_HPP
#define MPFEM_ND_HPP

#include "finite_element.hpp"

namespace mpfem {

    class NDFiniteElement final : public FiniteElement {
    public:
        NDFiniteElement(Geometry geom, int order);

        BasisType basisType() const override { return BasisType::ND; }
        MapType mapType() const override { return MapType::COVARIANT_PIOLA; }
        Geometry geometry() const override { return geom_; }
        int order() const override { return order_; }
        int numDofs() const override { return numDofs_; }
        int vdim() const override { return 1; }
        DofLayout dofLayout() const override;

        void evalShape(const Vector3& xi, ShapeMatrix& shape) const override;
        void evalDerivatives(const Vector3& xi, DerivMatrix& derivatives) const override;
        std::vector<Vector3> interpolationPoints() const override;
        std::vector<int> edgeDofs(int edgeIdx) const override;
        std::vector<int> faceDofs(int faceIdx) const override;

    private:
        Geometry geom_ = Geometry::Invalid;
        int order_ = 1;
        int numDofs_ = 0;
    };

} // namespace mpfem

#endif // MPFEM_ND_HPP
