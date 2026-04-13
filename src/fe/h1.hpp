#ifndef MPFEM_H1_HPP
#define MPFEM_H1_HPP

#include "finite_element.hpp"

namespace mpfem {

    class H1FiniteElement final : public FiniteElement {
    public:
        H1FiniteElement(Geometry geom, int order, int vdim = 1);

        BasisType basisType() const override { return BasisType::H1; }
        MapType mapType() const override { return MapType::VALUE; }
        Geometry geometry() const override { return geom_; }
        int order() const override { return order_; }
        int numDofs() const override;
        int vdim() const override { return vdim_; }
        DofLayout dofLayout() const override;

        void evalShape(const Vector3& xi, ShapeMatrix& shape) const override;
        void evalDerivatives(const Vector3& xi, DerivMatrix& derivatives) const override;
        std::vector<Vector3> interpolationPoints() const override;
        std::vector<int> vertexDofs(int vertexIdx) const override;
        std::vector<int> edgeDofs(int edgeIdx) const override;
        std::vector<int> faceDofs(int faceIdx) const override;
        std::vector<int> cellDofs(int cellIdx) const override;

    private:
        Geometry geom_ = Geometry::Invalid;
        int order_;
        int numDofs_;
        int vdim_ = 1;
    };

} // namespace mpfem

#endif // MPFEM_H1_HPP
