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
        std::span<const Matrix3> invJacobianTransposes;
    };

    class ExternalDataProvider {
    public:
        virtual ~ExternalDataProvider() = default;
        virtual TensorShape shape() const = 0;
        virtual void evaluateBatch(const EvaluationContext& ctx, std::span<TensorValue> dest) const = 0;
    };

} // namespace mpfem

#endif // MPFEM_EXPR_EVALUATION_CONTEXT_HPP
