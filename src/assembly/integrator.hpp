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
 * 
 * where L and R are differential operators, and phi_i, phi_j are shape functions.
 * 
 * Common examples:
 * - DiffusionIntegrator: A_ij = integral( k * grad(phi_i) . grad(phi_j) )
 * - MassIntegrator: A_ij = integral( rho * phi_i * phi_j )
 * - ConvectionIntegrator: A_ij = integral( (b . grad(phi_j)) * phi_i )
 */
class BilinearFormIntegrator {
public:
    BilinearFormIntegrator() = default;
    explicit BilinearFormIntegrator(Coefficient* q) : q_(q) {}
    virtual ~BilinearFormIntegrator() = default;
    
    /// Set the coefficient
    void setCoefficient(Coefficient* q) { q_ = q; }
    
    /// Get the coefficient
    Coefficient* coefficient() const { return q_; }
    
    /**
     * @brief Compute the element matrix for a given element.
     * 
     * @param refElem Reference element (contains shape functions and quadrature).
     * @param trans Element transformation (provides Jacobian and weight).
     * @param elmat Output element matrix (ndofs x ndofs).
     */
    virtual void assembleElementMatrix(const ReferenceElement& refElem,
                                       ElementTransform& trans,
                                       Matrix& elmat) const = 0;
    
    /**
     * @brief Compute boundary face matrix (for boundary integrals).
     * 
     * Used for boundary conditions like Robin BC or Neumann BC.
     */
    virtual void assembleFaceMatrix(const ReferenceElement& refElem,
                                    FacetElementTransform& trans,
                                    Matrix& elmat) const {
        // Default: not implemented, do nothing
        (void)refElem;
        (void)trans;
        (void)elmat;
    }
    
    /**
     * @brief Get integrator name for debugging.
     */
    virtual const char* name() const = 0;
    
protected:
    Coefficient* q_ = nullptr;  ///< Coefficient (material property)
};

// =============================================================================
// Linear Form Integrator (for load vectors)
// =============================================================================

/**
 * @brief Abstract base class for linear form integrators.
 * 
 * Linear form integrators compute element-level vectors:
 *   b_i = integral( f * phi_i ) over element
 * 
 * Common examples:
 * - DomainLFIntegrator: b_i = integral( f * phi_i ) for source terms
 * - BoundaryLFIntegrator: b_i = integral( g * phi_i ) over boundary
 * - GradientLFIntegrator: b_i = integral( F . grad(phi_i) )
 */
class LinearFormIntegrator {
public:
    LinearFormIntegrator() = default;
    explicit LinearFormIntegrator(Coefficient* q) : q_(q) {}
    virtual ~LinearFormIntegrator() = default;
    
    /// Set the coefficient
    void setCoefficient(Coefficient* q) { q_ = q; }
    
    /// Get the coefficient
    Coefficient* coefficient() const { return q_; }
    
    /**
     * @brief Compute the element vector for a given element.
     * 
     * @param refElem Reference element (contains shape functions and quadrature).
     * @param trans Element transformation (provides Jacobian and weight).
     * @param elvec Output element vector (ndofs).
     */
    virtual void assembleElementVector(const ReferenceElement& refElem,
                                       ElementTransform& trans,
                                       Vector& elvec) const = 0;
    
    /**
     * @brief Compute boundary face vector (for boundary integrals).
     */
    virtual void assembleFaceVector(const ReferenceElement& refElem,
                                    FacetElementTransform& trans,
                                    Vector& elvec) const {
        // Default: not implemented, do nothing
        (void)refElem;
        (void)trans;
        (void)elvec;
    }
    virtual const char* name() const = 0;
    
protected:
    Coefficient* q_ = nullptr;  ///< Coefficient (source term)
};

// =============================================================================
// Mixed Form Integrator (for coupled equations)
// =============================================================================

/**
 * @brief Abstract base class for mixed form integrators.
 * 
 * Mixed form integrators compute coupling matrices between different
 * fields (e.g., displacement-velocity coupling in fluid-structure interaction).
 * 
 * A_ij = integral( coeff * L1(phi_i^test) * L2(phi_j^trial) )
 */
class MixedFormIntegrator {
public:
    MixedFormIntegrator() = default;
    explicit MixedFormIntegrator(Coefficient* q) : q_(q) {}
    virtual ~MixedFormIntegrator() = default;
    
    /// Set the coefficient
    void setCoefficient(Coefficient* q) { q_ = q; }
    
    /**
     * @brief Compute the mixed element matrix.
     * 
     * @param testRefElem Reference element for test space.
     * @param trialRefElem Reference element for trial space.
     * @param trans Element transformation.
     * @param elmat Output matrix (test_ndofs x trial_ndofs).
     */
    virtual void assembleElementMatrix(const ReferenceElement& testRefElem,
                                       const ReferenceElement& trialRefElem,
                                       ElementTransform& trans,
                                       Matrix& elmat) const = 0;
    
    virtual const char* name() const = 0;
    
protected:
    Coefficient* q_ = nullptr;
};

// =============================================================================
// Vector-valued Integrators
// =============================================================================

/**
 * @brief Base class for vector field bilinear integrators.
 * 
 * Used for vector fields like displacement, velocity.
 */
class VectorBilinearFormIntegrator {
public:
    VectorBilinearFormIntegrator() = default;
    explicit VectorBilinearFormIntegrator(Coefficient* q) : q_(q) {}
    virtual ~VectorBilinearFormIntegrator() = default;
    
    void setCoefficient(Coefficient* q) { q_ = q; }
    
    /**
     * @brief Compute the element matrix for a vector field.
     * 
     * @param refElem Reference element.
     * @param trans Element transformation.
     * @param vdim Vector dimension (number of components).
     * @param elmat Output element matrix (vdim*ndofs x vdim*ndofs).
     */
    virtual void assembleElementMatrix(const ReferenceElement& refElem,
                                       ElementTransform& trans,
                                       int vdim,
                                       Matrix& elmat) const = 0;
    
    virtual const char* name() const = 0;
    
protected:
    Coefficient* q_ = nullptr;
};

/**
 * @brief Base class for vector field linear integrators.
 */
class VectorLinearFormIntegrator {
public:
    VectorLinearFormIntegrator() = default;
    explicit VectorLinearFormIntegrator(Coefficient* q) : q_(q) {}
    virtual ~VectorLinearFormIntegrator() = default;
    
    void setCoefficient(Coefficient* q) { q_ = q; }
    
    /**
     * @brief Compute the element vector for a vector field.
     * 
     * @param refElem Reference element.
     * @param trans Element transformation.
     * @param vdim Vector dimension.
     * @param elvec Output element vector (vdim * ndofs).
     */
    virtual void assembleElementVector(const ReferenceElement& refElem,
                                       ElementTransform& trans,
                                       int vdim,
                                       Vector& elvec) const = 0;
    
    virtual const char* name() const = 0;
    
protected:
    Coefficient* q_ = nullptr;
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATOR_HPP
