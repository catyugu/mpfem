#include "coefficient.hpp"
#include "element_transform.hpp"
#include <stdexcept>
#include <sstream>

namespace mpfem {

template<>
void DomainMappedCoefficient<Coefficient>::eval(ElementTransform& trans, Real& result, Real t) const {
    if (const Coefficient* coef = get(static_cast<int>(trans.attribute()))) {
        coef->eval(trans, result, t);
    } else {
        throw std::runtime_error("No scalar coefficient for domain " + 
            std::to_string(trans.attribute()));
    }
}

template<>
void DomainMappedCoefficient<VectorCoefficient>::eval(ElementTransform& trans, Vector3& result, Real t) const {
    if (const VectorCoefficient* coef = get(static_cast<int>(trans.attribute()))) {
        coef->eval(trans, result, t);
    } else {
        throw std::runtime_error("No vector coefficient for domain " + 
            std::to_string(trans.attribute()));
    }
}

template<>
void DomainMappedCoefficient<MatrixCoefficient>::eval(ElementTransform& trans, Matrix3& result, Real t) const {
    if (const MatrixCoefficient* coef = get(static_cast<int>(trans.attribute()))) {
        coef->eval(trans, result, t);
    } else {
        throw std::runtime_error("No matrix coefficient for domain " + 
            std::to_string(trans.attribute()));
    }
}

}  // namespace mpfem