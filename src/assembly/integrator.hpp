#ifndef MPFEM_INTEGRATOR_HPP
#define MPFEM_INTEGRATOR_HPP

#include "core/types.hpp"
#include "fe/element_transform.hpp"
#include "fe/facet_element_transform.hpp"
#include "fe/reference_element.hpp"
#include "fe/coefficient.hpp"
#include <memory>

namespace mpfem {

// =============================================================================
// Base integrator interface - provides coefficient evaluation helpers
// =============================================================================

/**
 * @brief Integrator base class
 * 
 * Ownership policy: Integrators do NOT own coefficients.
 * - Coefficient lifecycle is managed by caller (solver, Problem, etc.)
 * - Integrators hold non-owning raw pointer references
 */
class IntegratorBase {
protected:
    const Coefficient* q_ = nullptr;  ///< Coefficient pointer (non-owning)
    
    /// Evaluate scalar coefficient at integration point
    void evalCoef(ElementTransform& t, Real& result) const {
        if (q_) {
            q_->eval(t, result);
        } else {
            result = 1.0;
        }
    }
    
    void evalCoef(FacetElementTransform& t, Real& result) const {
        if (q_) {
            q_->eval(t, result);
        } else {
            result = 1.0;
        }
    }

public:
    IntegratorBase() = default;
    
    /// Constructor: coefficient pointer (non-owning, caller manages lifecycle)
    explicit IntegratorBase(const Coefficient* q) : q_(q) {}
    
    virtual ~IntegratorBase() = default;
    
    // Non-copyable
    IntegratorBase(const IntegratorBase&) = delete;
    IntegratorBase& operator=(const IntegratorBase&) = delete;
    
    // Movable
    IntegratorBase(IntegratorBase&& other) noexcept : q_(other.q_) {
        other.q_ = nullptr;
    }
    IntegratorBase& operator=(IntegratorBase&& other) noexcept {
        if (this != &other) {
            q_ = other.q_;
            other.q_ = nullptr;
        }
        return *this;
    }
};

// =============================================================================
// Domain bilinear integrator base - scalar field
// =============================================================================

class DomainBilinearIntegrator : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    
    /// Assemble element matrix: ∫ coef * L(φ_i) * L(φ_j) dΩ
    /// Output elmat is nd x nd matrix
    virtual void assembleElementMatrix(const ReferenceElement& ref, 
                                        ElementTransform& trans, 
                                        Matrix& elmat) const = 0;
};

// =============================================================================
// Face bilinear integrator base - scalar field
// =============================================================================

class FaceBilinearIntegrator : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    
    /// Assemble face matrix: ∫ coef * L(φ_i) * L(φ_j) dΓ
    /// Output elmat is nd x nd matrix
    virtual void assembleFaceMatrix(const ReferenceElement& ref,
                                     FacetElementTransform& trans,
                                     Matrix& elmat) const = 0;
};

// =============================================================================
// Domain linear integrator base - scalar field
// =============================================================================

class DomainLinearIntegrator : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    
    /// Assemble element vector: ∫ coef * L(φ_i) dΩ
    /// Output elvec is nd dimensional vector
    virtual void assembleElementVector(const ReferenceElement& ref,
                                        ElementTransform& trans,
                                        Vector& elvec) const = 0;
};

// =============================================================================
// Face linear integrator base - scalar field
// =============================================================================

class FaceLinearIntegrator : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    
    /// Assemble face vector: ∫ coef * L(φ_i) dΓ
    /// Output elvec is nd dimensional vector
    virtual void assembleFaceVector(const ReferenceElement& ref,
                                     FacetElementTransform& trans,
                                     Vector& elvec) const = 0;
};

// =============================================================================
// Vector field integrator bases (for displacement, velocity, etc.)
// =============================================================================

/// Vector field domain bilinear integrator base
class VectorDomainBilinearIntegrator : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    
    /// Assemble element matrix (vector field version)
    /// vdim is provided by FESpace, used internally by integrator
    virtual void assembleElementMatrix(const ReferenceElement& ref,
                                        ElementTransform& trans,
                                        Matrix& elmat,
                                        int vdim) const = 0;
};

/// Vector field domain linear integrator base
class VectorDomainLinearIntegrator : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    
    /// Assemble element vector (vector field version)
    virtual void assembleElementVector(const ReferenceElement& ref,
                                        ElementTransform& trans,
                                        Vector& elvec,
                                        int vdim) const = 0;
};

// =============================================================================
// Matrix coefficient integrator base
// =============================================================================

/**
 * @brief Base class for integrators that use matrix coefficients
 * 
 * Used for anisotropic diffusion, etc.
 */
class MatrixCoefficientIntegratorBase {
protected:
    const MatrixCoefficient* matCoef_ = nullptr;
    
    /// Evaluate matrix coefficient at integration point
    void evalMatCoef(ElementTransform& t, Matrix3& result) const {
        if (matCoef_) {
            matCoef_->eval(t, result);
        } else {
            result = Matrix3::Identity();
        }
    }
    
    void evalMatCoef(FacetElementTransform& t, Matrix3& result) const {
        if (matCoef_) {
            matCoef_->eval(t, result);
        } else {
            result = Matrix3::Identity();
        }
    }

public:
    MatrixCoefficientIntegratorBase() = default;
    explicit MatrixCoefficientIntegratorBase(const MatrixCoefficient* m) : matCoef_(m) {}
    
    virtual ~MatrixCoefficientIntegratorBase() = default;
    
    // Non-copyable, movable
    MatrixCoefficientIntegratorBase(const MatrixCoefficientIntegratorBase&) = delete;
    MatrixCoefficientIntegratorBase& operator=(const MatrixCoefficientIntegratorBase&) = delete;
    MatrixCoefficientIntegratorBase(MatrixCoefficientIntegratorBase&&) = default;
    MatrixCoefficientIntegratorBase& operator=(MatrixCoefficientIntegratorBase&&) = default;
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATOR_HPP
