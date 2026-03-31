#ifndef MPFEM_COEFFICIENT_HPP
#define MPFEM_COEFFICIENT_HPP

#include "core/types.hpp"
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>

namespace mpfem {

class ElementTransform;
class GridFunction;

inline constexpr std::uint64_t DynamicCoefficientTag = std::numeric_limits<std::uint64_t>::max();

inline std::uint64_t combineTag(std::uint64_t seed, std::uint64_t value) {
    if (seed == DynamicCoefficientTag || value == DynamicCoefficientTag) {
        return DynamicCoefficientTag;
    }
    return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
}

inline std::uint64_t hashRealTag(Real value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "Real size mismatch in hashRealTag");
    std::memcpy(&bits, &value, sizeof(value));
    return 1469598103934665603ull ^ bits;
}

// =============================================================================
// Base classes
// =============================================================================

class Coefficient {
public:
    virtual ~Coefficient() = default;
    virtual void eval(ElementTransform& trans, Real& result, Real t = 0.0) const = 0;
    virtual std::uint64_t stateTag() const { return DynamicCoefficientTag; }
};

class VectorCoefficient {
public:
    virtual ~VectorCoefficient() = default;
    virtual void eval(ElementTransform& trans, Vector3& result, Real t = 0.0) const = 0;
    virtual std::uint64_t stateTag() const { return DynamicCoefficientTag; }
};

class MatrixCoefficient {
public:
    virtual ~MatrixCoefficient() = default;
    virtual void eval(ElementTransform& trans, Matrix3& result, Real t = 0.0) const = 0;
    virtual std::uint64_t stateTag() const { return DynamicCoefficientTag; }
};

// =============================================================================
// Lambda-based coefficients
// =============================================================================

class ScalarCoefficient : public Coefficient {
public:
    using Func = std::function<void(ElementTransform&, Real&, Real)>;
    using TagFunc = std::function<std::uint64_t()>;

    explicit ScalarCoefficient(Func f,
                               TagFunc tagFunc = [] { return DynamicCoefficientTag; })
        : func_(std::move(f)), tagFunc_(std::move(tagFunc)) {}

    ScalarCoefficient(Func f, std::uint64_t fixedTag)
        : func_(std::move(f)), tagFunc_([fixedTag] { return fixedTag; }) {}

    void eval(ElementTransform& trans, Real& result, Real t) const override { func_(trans, result, t); }
    std::uint64_t stateTag() const override { return tagFunc_(); }

private:
    Func func_;
    TagFunc tagFunc_;
};

/**
 * @brief Product coefficient: (c1 * c2) at each evaluation point
 */
class ProductCoefficient : public Coefficient {
public:
    ProductCoefficient(const Coefficient* c1, const Coefficient* c2) : c1_(c1), c2_(c2) {}

    void eval(ElementTransform& trans, Real& result, Real t) const override {
        Real v1 = 0.0;
        Real v2 = 0.0;
        c1_->eval(trans, v1, t);
        c2_->eval(trans, v2, t);
        result = v1 * v2;
    }

    std::uint64_t stateTag() const override {
        if (!c1_ || !c2_) {
            return DynamicCoefficientTag;
        }
        return combineTag(c1_->stateTag(), c2_->stateTag());
    }

private:
    const Coefficient* c1_ = nullptr;
    const Coefficient* c2_ = nullptr;
};

class VectorFunctionCoefficient : public VectorCoefficient {
public:
    using Func = std::function<void(ElementTransform&, Vector3&, Real)>;
    using TagFunc = std::function<std::uint64_t()>;

    explicit VectorFunctionCoefficient(Func f,
                                       TagFunc tagFunc = [] { return DynamicCoefficientTag; })
        : func_(std::move(f)), tagFunc_(std::move(tagFunc)) {}

    VectorFunctionCoefficient(Func f, std::uint64_t fixedTag)
        : func_(std::move(f)), tagFunc_([fixedTag] { return fixedTag; }) {}

    void eval(ElementTransform& trans, Vector3& result, Real t) const override { func_(trans, result, t); }
    std::uint64_t stateTag() const override { return tagFunc_(); }

private:
    Func func_;
    TagFunc tagFunc_;
};

class MatrixFunctionCoefficient : public MatrixCoefficient {
public:
    using Func = std::function<void(ElementTransform&, Matrix3&, Real)>;
    using TagFunc = std::function<std::uint64_t()>;

    explicit MatrixFunctionCoefficient(Func f,
                                       TagFunc tagFunc = [] { return DynamicCoefficientTag; })
        : func_(std::move(f)), tagFunc_(std::move(tagFunc)) {}

    MatrixFunctionCoefficient(Func f, std::uint64_t fixedTag)
        : func_(std::move(f)), tagFunc_([fixedTag] { return fixedTag; }) {}

    void eval(ElementTransform& trans, Matrix3& result, Real t) const override { func_(trans, result, t); }
    std::uint64_t stateTag() const override { return tagFunc_(); }

private:
    Func func_;
    TagFunc tagFunc_;
};

// =============================================================================
// Convenience functions for creating constant coefficients
// =============================================================================

inline std::unique_ptr<Coefficient> constantCoefficient(Real value) {
    const std::uint64_t tag = hashRealTag(value);
    return std::make_unique<ScalarCoefficient>(
        [value](ElementTransform&, Real& r, Real) { r = value; },
        tag);
}

inline std::unique_ptr<VectorCoefficient> constantVectorCoefficient(Real x, Real y, Real z) {
    std::uint64_t tag = hashRealTag(x);
    tag = combineTag(tag, hashRealTag(y));
    tag = combineTag(tag, hashRealTag(z));
    return std::make_unique<VectorFunctionCoefficient>(
        [x, y, z](ElementTransform&, Vector3& r, Real) { r << x, y, z; },
        tag);
}

inline std::unique_ptr<MatrixCoefficient> constantMatrixCoefficient(const Matrix3& mat) {
    std::uint64_t tag = 1469598103934665603ull;
    for (int r = 0; r < mat.rows(); ++r) {
        for (int c = 0; c < mat.cols(); ++c) {
            tag = combineTag(tag, hashRealTag(mat(r, c)));
        }
    }
    return std::make_unique<MatrixFunctionCoefficient>(
        [mat](ElementTransform&, Matrix3& r, Real) { r = mat; },
        tag);
}

}  // namespace mpfem

#endif  // MPFEM_COEFFICIENT_HPP
