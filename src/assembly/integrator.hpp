#ifndef MPFEM_INTEGRATOR_HPP
#define MPFEM_INTEGRATOR_HPP

#include "core/types.hpp"
#include "fe/element_transform.hpp"
#include "fe/facet_element_transform.hpp"
#include "fe/reference_element.hpp"
#include "fe/coefficient.hpp"
#include <memory>
#include <vector>

namespace mpfem {

// Forward declarations
class FESpace;
class GridFunction;

/**
 * @file integrator.hpp
 * @brief Base classes for finite element integrators.
 * 
 * Integrators compute local (element-level) matrices and vectors that
 * contribute to the global system. They implement the weak form of
 * partial differential equations.
 * 
 * Resource Management:
 * - All Coefficients are managed via std::shared_ptr for clear ownership
 * - Use setCoefficient() to set with shared ownership
 * 
 * Design inspired by MFEM's BilinearFormIntegrator and LinearFormIntegrator.
 */

// =============================================================================
// Bilinear Form Integrator (for stiffness matrices)
// =============================================================================

/**
 * @brief Abstract base class for bilinear form integrators.
 * 
 * Bilinear form integrators compute element-level matrices:
 *   A_ij = integral( coeff * L(phi_i) * R(phi_j) ) over element
 */
class BilinearFormIntegrator {
public:
    BilinearFormIntegrator() = default;
    
    /// Construct with coefficient
    explicit BilinearFormIntegrator(std::shared_ptr<Coefficient> q) : q_(std::move(q)) {}
    
    virtual ~BilinearFormIntegrator() = default;
    
    /// Set coefficient (takes shared ownership)
    void setCoefficient(std::shared_ptr<Coefficient> q) { q_ = std::move(q); }
    
    /// Get the coefficient (may be nullptr)
    Coefficient* coefficient() const { return q_.get(); }
    
    /// Check if coefficient is set
    bool hasCoefficient() const { return q_ != nullptr; }
    
    /// Evaluate coefficient safely (returns 1.0 if not set)
    Real evalCoefficient(ElementTransform& trans) const {
        return q_ ? q_->eval(trans) : 1.0;
    }
    
    virtual void assembleElementMatrix(const ReferenceElement& refElem,
                                       ElementTransform& trans,
                                       Matrix& elmat) const = 0;
    
    virtual void assembleFaceMatrix(const ReferenceElement& refElem,
                                    FacetElementTransform& trans,
                                    Matrix& elmat) const {
        (void)refElem; (void)trans; (void)elmat;
    }
    
    virtual const char* name() const = 0;
    
protected:
    std::shared_ptr<Coefficient> q_;
};

// =============================================================================
// Linear Form Integrator (for load vectors)
// =============================================================================

/**
 * @brief Abstract base class for linear form integrators.
 */
class LinearFormIntegrator {
public:
    LinearFormIntegrator() = default;
    explicit LinearFormIntegrator(std::shared_ptr<Coefficient> q) : q_(std::move(q)) {}
    virtual ~LinearFormIntegrator() = default;
    
    void setCoefficient(std::shared_ptr<Coefficient> q) { q_ = std::move(q); }
    
    Coefficient* coefficient() const { return q_.get(); }
    bool hasCoefficient() const { return q_ != nullptr; }
    Real evalCoefficient(ElementTransform& trans) const {
        return q_ ? q_->eval(trans) : 1.0;
    }
    
    virtual void assembleElementVector(const ReferenceElement& refElem,
                                       ElementTransform& trans,
                                       Vector& elvec) const = 0;
    
    virtual void assembleFaceVector(const ReferenceElement& refElem,
                                    FacetElementTransform& trans,
                                    Vector& elvec) const {
        (void)refElem; (void)trans; (void)elvec;
    }
    
    virtual const char* name() const = 0;
    
protected:
    std::shared_ptr<Coefficient> q_;
};

// =============================================================================
// Vector-valued Integrators
// =============================================================================

/**
 * @brief Base class for vector field bilinear integrators.
 */
class VectorBilinearFormIntegrator {
public:
    VectorBilinearFormIntegrator() = default;
    explicit VectorBilinearFormIntegrator(std::shared_ptr<Coefficient> q) : q_(std::move(q)) {}
    virtual ~VectorBilinearFormIntegrator() = default;
    
    void setCoefficient(std::shared_ptr<Coefficient> q) { q_ = std::move(q); }
    
    Coefficient* coefficient() const { return q_.get(); }
    bool hasCoefficient() const { return q_ != nullptr; }
    Real evalCoefficient(ElementTransform& trans) const {
        return q_ ? q_->eval(trans) : 1.0;
    }
    
    virtual void assembleElementMatrix(const ReferenceElement& refElem,
                                       ElementTransform& trans,
                                       int vdim,
                                       Matrix& elmat) const = 0;
    
    virtual const char* name() const = 0;
    
protected:
    std::shared_ptr<Coefficient> q_;
};

/**
 * @brief Base class for vector field linear integrators.
 */
class VectorLinearFormIntegrator {
public:
    VectorLinearFormIntegrator() = default;
    explicit VectorLinearFormIntegrator(std::shared_ptr<Coefficient> q) : q_(std::move(q)) {}
    virtual ~VectorLinearFormIntegrator() = default;
    
    void setCoefficient(std::shared_ptr<Coefficient> q) { q_ = std::move(q); }
    
    Coefficient* coefficient() const { return q_.get(); }
    bool hasCoefficient() const { return q_ != nullptr; }
    
    virtual void assembleElementVector(const ReferenceElement& refElem,
                                       ElementTransform& trans,
                                       int vdim,
                                       Vector& elvec) const = 0;
    
    virtual const char* name() const = 0;
    
protected:
    std::shared_ptr<Coefficient> q_;
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATOR_HPP