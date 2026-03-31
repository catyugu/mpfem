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
// Abstract interfaces for bilinear integrators
// =============================================================================

class DomainBilinearIntegratorBase {
public:
    virtual ~DomainBilinearIntegratorBase() = default;
    
    // Returns the vdim for this integrator (scalar integrators return 1)
    virtual int vdim() const = 0;
    
    // elmat is pre-sized to vdim()*ndof() x vdim()*ndof() by caller
    // Scalar integrators fill only the diagonal block (0:nd, 0:nd)
    // Vector integrators fill the full matrix
    virtual void assembleElementMatrix(const ReferenceElement& ref,
                                        ElementTransform& trans,
                                        Matrix& elmat) const = 0;
};

class FaceBilinearIntegratorBase {
public:
    virtual ~FaceBilinearIntegratorBase() = default;
    virtual void assembleFaceMatrix(const ReferenceElement& ref,
                                     FacetElementTransform& trans,
                                     Matrix& elmat) const = 0;
};

// =============================================================================
// Abstract interfaces for linear integrators
// =============================================================================

class DomainLinearIntegratorBase {
public:
    virtual ~DomainLinearIntegratorBase() = default;
    
    // Returns the vdim for this integrator (scalar integrators return 1)
    virtual int vdim() const = 0;
    
    // elvec is pre-sized to vdim()*ndof() by caller
    // Scalar integrators fill only segment (0:nd)
    // Vector integrators fill the full vector
    virtual void assembleElementVector(const ReferenceElement& ref,
                                        ElementTransform& trans,
                                        Vector& elvec) const = 0;
};

class FaceLinearIntegratorBase {
public:
    virtual ~FaceLinearIntegratorBase() = default;
    virtual void assembleFaceVector(const ReferenceElement& ref,
                                     FacetElementTransform& trans,
                                     Vector& elvec) const = 0;
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATOR_HPP
