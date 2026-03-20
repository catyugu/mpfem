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

// =============================================================================
// Vector field integrator interfaces
// =============================================================================

class VectorDomainBilinearIntegrator {
public:
    virtual ~VectorDomainBilinearIntegrator() = default;
    virtual void assembleElementMatrix(const ReferenceElement& ref,
                                        ElementTransform& trans,
                                        Matrix& elmat,
                                        int vdim) const = 0;
};

class VectorDomainLinearIntegrator {
public:
    virtual ~VectorDomainLinearIntegrator() = default;
    virtual void assembleElementVector(const ReferenceElement& ref,
                                        ElementTransform& trans,
                                        Vector& elvec,
                                        int vdim) const = 0;
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATOR_HPP