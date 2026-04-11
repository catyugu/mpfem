#ifndef MPFEM_H1_HPP
#define MPFEM_H1_HPP

#include "finite_element.hpp"

#include <algorithm>

namespace mpfem {

    class H1SegmentShape : public FiniteElement {
    public:
        explicit H1SegmentShape(int order);

        BasisType basisType() const override { return BasisType::H1; }
        Geometry geometry() const override { return Geometry::Segment; }
        int order() const override { return order_; }
        int numDofs() const override { return order_ + 1; }
        int vdim() const override { return 1; }

        int dofsPerVertex() const override { return 1; }
        int dofsPerEdge() const override { return std::max(0, order_ - 1); }
        int dofsPerFace() const override { return 0; }
        int dofsPerVolume() const override { return 0; }

        void evalShape(const Vector3& xi, Matrix& shape) const override;
        void evalDerivatives(const Vector3& xi, Matrix& derivatives) const override;
        std::vector<Vector3> dofCoords() const override;
        std::vector<int> faceDofs(int faceIdx) const override;

    private:
        void evalValuesImpl(const Vector3& xi, std::span<Real> values) const;
        void evalGradsImpl(const Vector3& xi, std::span<Vector3> grads) const;

        int order_;
    };

    class H1TriangleShape : public FiniteElement {
    public:
        explicit H1TriangleShape(int order);

        BasisType basisType() const override { return BasisType::H1; }
        Geometry geometry() const override { return Geometry::Triangle; }
        int order() const override { return order_; }
        int numDofs() const override;
        int vdim() const override { return 1; }

        int dofsPerVertex() const override { return 1; }
        int dofsPerEdge() const override { return std::max(0, order_ - 1); }
        int dofsPerFace() const override { return 0; }
        int dofsPerVolume() const override { return 0; }

        void evalShape(const Vector3& xi, Matrix& shape) const override;
        void evalDerivatives(const Vector3& xi, Matrix& derivatives) const override;
        std::vector<Vector3> dofCoords() const override;
        std::vector<int> faceDofs(int faceIdx) const override;

    private:
        void evalValuesImpl(const Vector3& xi, std::span<Real> values) const;
        void evalGradsImpl(const Vector3& xi, std::span<Vector3> grads) const;

        int order_;
    };

    class H1SquareShape : public FiniteElement {
    public:
        explicit H1SquareShape(int order);

        BasisType basisType() const override { return BasisType::H1; }
        Geometry geometry() const override { return Geometry::Square; }
        int order() const override { return order_; }
        int numDofs() const override { return (order_ + 1) * (order_ + 1); }
        int vdim() const override { return 1; }

        int dofsPerVertex() const override { return 1; }
        int dofsPerEdge() const override { return std::max(0, order_ - 1); }
        int dofsPerFace() const override { return 0; }
        int dofsPerVolume() const override { return order_ > 1 ? 1 : 0; }

        void evalShape(const Vector3& xi, Matrix& shape) const override;
        void evalDerivatives(const Vector3& xi, Matrix& derivatives) const override;
        std::vector<Vector3> dofCoords() const override;
        std::vector<int> faceDofs(int faceIdx) const override;

    private:
        void evalValuesImpl(const Vector3& xi, std::span<Real> values) const;
        void evalGradsImpl(const Vector3& xi, std::span<Vector3> grads) const;

        int order_;
    };

    class H1TetrahedronShape : public FiniteElement {
    public:
        explicit H1TetrahedronShape(int order);

        BasisType basisType() const override { return BasisType::H1; }
        Geometry geometry() const override { return Geometry::Tetrahedron; }
        int order() const override { return order_; }
        int numDofs() const override;
        int vdim() const override { return 1; }

        int dofsPerVertex() const override { return 1; }
        int dofsPerEdge() const override { return std::max(0, order_ - 1); }
        int dofsPerFace() const override { return 0; }
        int dofsPerVolume() const override { return 0; }

        void evalShape(const Vector3& xi, Matrix& shape) const override;
        void evalDerivatives(const Vector3& xi, Matrix& derivatives) const override;
        std::vector<Vector3> dofCoords() const override;
        std::vector<int> faceDofs(int faceIdx) const override;

    private:
        void evalValuesImpl(const Vector3& xi, std::span<Real> values) const;
        void evalGradsImpl(const Vector3& xi, std::span<Vector3> grads) const;

        int order_;
    };

    class H1CubeShape : public FiniteElement {
    public:
        explicit H1CubeShape(int order);

        BasisType basisType() const override { return BasisType::H1; }
        Geometry geometry() const override { return Geometry::Cube; }
        int order() const override { return order_; }
        int numDofs() const override { return (order_ + 1) * (order_ + 1) * (order_ + 1); }
        int vdim() const override { return 1; }

        int dofsPerVertex() const override { return 1; }
        int dofsPerEdge() const override { return std::max(0, order_ - 1); }
        int dofsPerFace() const override { return order_ > 1 ? 1 : 0; }
        int dofsPerVolume() const override { return order_ > 1 ? 1 : 0; }

        void evalShape(const Vector3& xi, Matrix& shape) const override;
        void evalDerivatives(const Vector3& xi, Matrix& derivatives) const override;
        std::vector<Vector3> dofCoords() const override;
        std::vector<int> faceDofs(int faceIdx) const override;

    private:
        void evalValuesImpl(const Vector3& xi, std::span<Real> values) const;
        void evalGradsImpl(const Vector3& xi, std::span<Vector3> grads) const;

        int order_;
    };

} // namespace mpfem

#endif // MPFEM_H1_HPP
