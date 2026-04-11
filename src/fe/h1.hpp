#ifndef MPFEM_H1_HPP
#define MPFEM_H1_HPP

#include "finite_element.hpp"

namespace mpfem {

    class H1FiniteElement final : public FiniteElement {
    public:
        H1FiniteElement(Geometry geom, int order);

        BasisType basisType() const override { return BasisType::H1; }
        Geometry geometry() const override { return geom_; }
        int order() const override { return order_; }
        int numDofs() const override;
        int vdim() const override { return 1; }
        DofLayout dofLayout() const override;

        void evalShape(const Vector3& xi, Matrix& shape) const override;
        void evalDerivatives(const Vector3& xi, Matrix& derivatives) const override;
        std::vector<Vector3> interpolationPoints() const override;
        std::vector<int> faceDofs(int faceIdx) const override;

    private:
        Geometry geom_ = Geometry::Invalid;
        int order_;
    };

} // namespace mpfem

#endif // MPFEM_H1_HPP
