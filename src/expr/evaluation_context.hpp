#ifndef MPFEM_EXPR_EVALUATION_CONTEXT_HPP
#define MPFEM_EXPR_EVALUATION_CONTEXT_HPP

#include "core/tensor_shape.hpp"
#include "core/tensor_value.hpp"
#include "core/types.hpp"

#include <span>

namespace mpfem {

    struct EvaluationContext {
        Real time = Real(0);
        int domainId = -1;
        Index elementId = InvalidIndex;
        std::span<const Vector3> physicalPoints;
        std::span<const Vector3> referencePoints;
        std::span<const Matrix> invJacobianTransposes;
    };

} // namespace mpfem

#endif // MPFEM_EXPR_EVALUATION_CONTEXT_HPP
