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
// Abstract interfaces (non-template for polymorphic storage)
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
// Template implementations with coefficient storage
// =============================================================================

template<typename CoefType>
class DomainBilinearIntegrator : public DomainBilinearIntegratorBase {
protected:
    const CoefType* coef_ = nullptr;
    
    DomainBilinearIntegrator() = default;
    explicit DomainBilinearIntegrator(const CoefType* c) : coef_(c) {}
};

template<typename CoefType>
class FaceBilinearIntegrator : public FaceBilinearIntegratorBase {
protected:
    const CoefType* coef_ = nullptr;
    
    FaceBilinearIntegrator() = default;
    explicit FaceBilinearIntegrator(const CoefType* c) : coef_(c) {}
};

template<typename CoefType>
class DomainLinearIntegrator : public DomainLinearIntegratorBase {
protected:
    const CoefType* coef_ = nullptr;
    
    DomainLinearIntegrator() = default;
    explicit DomainLinearIntegrator(const CoefType* c) : coef_(c) {}
};

template<typename CoefType>
class FaceLinearIntegrator : public FaceLinearIntegratorBase {
protected:
    const CoefType* coef_ = nullptr;
    
    FaceLinearIntegrator() = default;
    explicit FaceLinearIntegrator(const CoefType* c) : coef_(c) {}
};

// =============================================================================
// Vector field integrator bases
// =============================================================================

class VectorDomainBilinearIntegrator {
protected:
    const Coefficient* coef_ = nullptr;
    
    VectorDomainBilinearIntegrator() = default;
    explicit VectorDomainBilinearIntegrator(const Coefficient* c) : coef_(c) {}
    
public:
    virtual ~VectorDomainBilinearIntegrator() = default;
    virtual void assembleElementMatrix(const ReferenceElement& ref,
                                        ElementTransform& trans,
                                        Matrix& elmat,
                                        int vdim) const = 0;
};

class VectorDomainLinearIntegrator {
protected:
    const Coefficient* coef_ = nullptr;
    
    VectorDomainLinearIntegrator() = default;
    explicit VectorDomainLinearIntegrator(const Coefficient* c) : coef_(c) {}
    
public:
    virtual ~VectorDomainLinearIntegrator() = default;
    virtual void assembleElementVector(const ReferenceElement& ref,
                                        ElementTransform& trans,
                                        Vector& elvec,
                                        int vdim) const = 0;
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATOR_HPP