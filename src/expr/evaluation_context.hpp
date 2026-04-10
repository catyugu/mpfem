#ifndef MPFEM_EXPR_EVALUATION_CONTEXT_HPP
#define MPFEM_EXPR_EVALUATION_CONTEXT_HPP

#include "core/tensor_shape.hpp"
#include "core/tensor.hpp"
#include "core/types.hpp"

#include <span>

namespace mpfem {

    class ElementTransform;

    struct EvaluationContext {
        Real time = Real(0);
        int domainId = -1;
        Index elementId = InvalidIndex;
        std::span<const Vector3> physicalPoints;
        std::span<const Vector3> referencePoints;
        std::span<ElementTransform* const> transforms;
    };

} // namespace mpfem

#endif // MPFEM_EXPR_EVALUATION_CONTEXT_HPP
