#ifndef MPFEM_COEFFICIENT_HPP
#define MPFEM_COEFFICIENT_HPP

#include "core/types.hpp"
#include <cstdint>
#include <functional>
#include <memory>

namespace mpfem {

    class ElementTransform;
    class GridFunction;

    // =============================================================================
    // Base classes (kept for type safety in integrators)
    // =============================================================================

    class Coefficient {
    public:
        virtual ~Coefficient() = default;
        virtual void eval(ElementTransform& trans, Real& result, Real t = 0.0) const = 0;
    };

    class VectorCoefficient {
    public:
        virtual ~VectorCoefficient() = default;
        virtual void eval(ElementTransform& trans, Vector3& result, Real t = 0.0) const = 0;
    };

    class MatrixCoefficient {
    public:
        virtual ~MatrixCoefficient() = default;
        virtual void eval(ElementTransform& trans, Matrix3& result, Real t = 0.0) const = 0;
    };

    // =============================================================================
    // Function-based coefficients (non-template)
    // =============================================================================

    class FunctionCoefficient : public Coefficient {
    public:
        using Func = std::function<void(ElementTransform&, Real&, Real)>;

        explicit FunctionCoefficient(Func f)
            : func_(std::move(f)) { }

        void eval(ElementTransform& trans, Real& result, Real t) const override { func_(trans, result, t); }

    private:
        Func func_;
    };

    class VectorFunctionCoefficient : public VectorCoefficient {
    public:
        using Func = std::function<void(ElementTransform&, Vector3&, Real)>;

        explicit VectorFunctionCoefficient(Func f)
            : func_(std::move(f)) { }

        void eval(ElementTransform& trans, Vector3& result, Real t) const override { func_(trans, result, t); }

    private:
        Func func_;
    };

    class MatrixFunctionCoefficient : public MatrixCoefficient {
    public:
        using Func = std::function<void(ElementTransform&, Matrix3&, Real)>;

        explicit MatrixFunctionCoefficient(Func f)
            : func_(std::move(f)) { }

        void eval(ElementTransform& trans, Matrix3& result, Real t) const override { func_(trans, result, t); }

    private:
        Func func_;
    };

    // =============================================================================
    // Convenience functions for creating coefficients
    // =============================================================================

    inline std::unique_ptr<Coefficient> constantCoefficient(Real value)
    {
        return std::make_unique<FunctionCoefficient>(
            [value](ElementTransform&, Real& r, Real) { r = value; });
    }

    inline std::unique_ptr<VectorCoefficient> constantVectorCoefficient(Real x, Real y, Real z)
    {
        return std::make_unique<VectorFunctionCoefficient>(
            [x, y, z](ElementTransform&, Vector3& r, Real) { r << x, y, z; });
    }

    inline std::unique_ptr<MatrixCoefficient> constantMatrixCoefficient(const Matrix3& mat)
    {
        return std::make_unique<MatrixFunctionCoefficient>(
            [mat](ElementTransform&, Matrix3& r, Real) { r = mat; });
    }

} // namespace mpfem

#endif // MPFEM_COEFFICIENT_HPP