#include "coefficient.hpp"
#include "element_transform.hpp"

namespace mpfem {

// =============================================================================
// DomainMappedCoefficient<Coefficient> implementation
// =============================================================================

void DomainMappedCoefficient<Coefficient>::eval(ElementTransform& trans, Real& result, Real t) const {
    const Coefficient* coef = get(static_cast<int>(trans.attribute()));
    if (coef) {
        coef->eval(trans, result, t);
    } else {
        result = 0.0;
    }
}

// =============================================================================
// DomainMappedCoefficient<VectorCoefficient> implementation
// =============================================================================

void DomainMappedCoefficient<VectorCoefficient>::eval(ElementTransform& trans, Vector3& result, Real t) const {
    const VectorCoefficient* coef = get(static_cast<int>(trans.attribute()));
    if (coef) {
        coef->eval(trans, result, t);
    } else {
        result.setZero();
    }
}

// =============================================================================
// DomainMappedCoefficient<MatrixCoefficient> implementation
// =============================================================================

void DomainMappedCoefficient<MatrixCoefficient>::eval(ElementTransform& trans, Matrix3& result, Real t) const {
    const MatrixCoefficient* coef = get(static_cast<int>(trans.attribute()));
    if (coef) {
        coef->eval(trans, result, t);
    } else {
        result.setZero();
    }
}

}  // namespace mpfem
