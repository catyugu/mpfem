#ifndef MPFEM_SHAPE_EVALUATOR_HPP
#define MPFEM_SHAPE_EVALUATOR_HPP

#include "core/exception.hpp"
#include "core/types.hpp"
#include "fe/element_transform.hpp"
#include "fe/reference_element.hpp"

#include <cmath>
#include <span>

namespace mpfem {

    class ShapeEvaluator {
    public:
        static void evalPhysShape(MapType mapType,
            const ElementTransform& trans,
            const ShapeMatrix& refShape,
            ShapeMatrix& physShape,
            std::span<const int> orientation = {})
        {
            const int nd = refShape.rows();
            const int dim = trans.dim();

            if (mapType == MapType::VALUE) {
                physShape.resize(nd, refShape.cols());
                physShape = refShape;
                applyOrientation(physShape, orientation);
                return;
            }

            if (mapType == MapType::COVARIANT_PIOLA) {
                physShape.setZero(nd, 3);
                const Matrix3& invJT = trans.invJacobianT();
                for (int i = 0; i < nd; ++i) {
                    for (int d = 0; d < 3; ++d) {
                        Real val = 0.0;
                        for (int k = 0; k < dim; ++k) {
                            val += invJT(d, k) * refShape(i, k);
                        }
                        physShape(i, d) = val;
                    }
                }
                applyOrientation(physShape, orientation);
                return;
            }

            if (mapType == MapType::CONTRAVARIANT_PIOLA) {
                const Real detJ = trans.detJ();
                if (std::abs(detJ) <= 1e-15) {
                    MPFEM_THROW(Exception, "ShapeEvaluator::evalPhysShape singular Jacobian for CONTRAVARIANT_PIOLA");
                }
                const Matrix3& J = trans.jacobian();
                physShape.setZero(nd, 3);
                for (int i = 0; i < nd; ++i) {
                    for (int d = 0; d < 3; ++d) {
                        Real val = 0.0;
                        for (int k = 0; k < dim; ++k) {
                            val += J(d, k) * refShape(i, k);
                        }
                        physShape(i, d) = val / detJ;
                    }
                }
                applyOrientation(physShape, orientation);
                return;
            }

            MPFEM_THROW(Exception, "ShapeEvaluator::evalPhysShape unsupported map type");
        }

        static void evalPhysDerivatives(MapType mapType,
            const ElementTransform& trans,
            const DerivMatrix& refDerivatives,
            DerivMatrix& physDerivatives)
        {
            if (mapType != MapType::VALUE) {
                MPFEM_THROW(NotImplementedException, "ShapeEvaluator::evalPhysDerivatives supports VALUE map type only");
            }

            const int nd = refDerivatives.rows();
            const int dim = trans.dim();
            const Matrix3& invJT = trans.invJacobianT();

            physDerivatives.setZero(nd, 3);
            for (int i = 0; i < nd; ++i) {
                for (int d = 0; d < 3; ++d) {
                    Real val = 0.0;
                    for (int k = 0; k < dim; ++k) {
                        val += invJT(d, k) * refDerivatives(i, k);
                    }
                    physDerivatives(i, d) = val;
                }
            }
        }

    private:
        static void applyOrientation(ShapeMatrix& values, std::span<const int> orientation)
        {
            if (orientation.empty()) {
                return;
            }
            const int nd = values.rows();
            if (static_cast<int>(orientation.size()) < nd) {
                MPFEM_THROW(ArgumentException, "ShapeEvaluator::applyOrientation orientation size is smaller than number of DOFs");
            }
            for (int i = 0; i < nd; ++i) {
                const int sign = orientation[i] < 0 ? -1 : 1;
                values.row(i) *= static_cast<Real>(sign);
            }
        }
    };

} // namespace mpfem

#endif // MPFEM_SHAPE_EVALUATOR_HPP
