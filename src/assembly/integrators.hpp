#ifndef MPFEM_INTEGRATORS_HPP
#define MPFEM_INTEGRATORS_HPP

#include "integrator.hpp"
#include "fe/facet_element_transform.hpp"
#include <cmath>

namespace mpfem {

/**
 * @file integrators.hpp
 * @brief Concrete implementations of common integrators.
 */

// =============================================================================
// Diffusion Integrator
// =============================================================================

class DiffusionIntegrator : public BilinearFormIntegrator {
public:
    DiffusionIntegrator() = default;
    explicit DiffusionIntegrator(std::shared_ptr<Coefficient> q) 
        : BilinearFormIntegrator(std::move(q)) {}
    
    void assembleElementMatrix(const ReferenceElement& refElem,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
    
    const char* name() const override { return "DiffusionIntegrator"; }
};

// =============================================================================
// Mass Integrator
// =============================================================================

class MassIntegrator : public BilinearFormIntegrator {
public:
    MassIntegrator() = default;
    explicit MassIntegrator(std::shared_ptr<Coefficient> q) 
        : BilinearFormIntegrator(std::move(q)) {}
    
    void assembleElementMatrix(const ReferenceElement& refElem,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
    
    const char* name() const override { return "MassIntegrator"; }
};

// =============================================================================
// Domain Load Integrator
// =============================================================================

class DomainLFIntegrator : public LinearFormIntegrator {
public:
    DomainLFIntegrator() = default;
    explicit DomainLFIntegrator(std::shared_ptr<Coefficient> f) 
        : LinearFormIntegrator(std::move(f)) {}
    
    void assembleElementVector(const ReferenceElement& refElem,
                               ElementTransform& trans,
                               Vector& elvec) const override;
    
    const char* name() const override { return "DomainLFIntegrator"; }
};

// =============================================================================
// Boundary Load Integrator
// =============================================================================

class BoundaryLFIntegrator : public LinearFormIntegrator {
public:
    BoundaryLFIntegrator() = default;
    explicit BoundaryLFIntegrator(std::shared_ptr<Coefficient> g) 
        : LinearFormIntegrator(std::move(g)) {}
    
    void assembleFaceVector(const ReferenceElement& refElem,
                            FacetElementTransform& trans,
                            Vector& elvec) const override;
    
    const char* name() const override { return "BoundaryLFIntegrator"; }
};

// =============================================================================
// Convection Boundary Integrator (Robin BC)
// =============================================================================

class ConvectionBoundaryIntegrator : public BilinearFormIntegrator {
public:
    ConvectionBoundaryIntegrator() = default;
    
    ConvectionBoundaryIntegrator(std::shared_ptr<Coefficient> h, 
                                  std::shared_ptr<Coefficient> Tinf = nullptr)
        : BilinearFormIntegrator(std::move(h)), Tinf_(std::move(Tinf)) {}
    
    void setAmbientTemperature(std::shared_ptr<Coefficient> Tinf) { Tinf_ = std::move(Tinf); }
    Coefficient* ambientTemperature() const { return Tinf_.get(); }
    
    void assembleFaceMatrix(const ReferenceElement& refElem,
                            FacetElementTransform& trans,
                            Matrix& elmat) const override;
    
    void assembleFaceVector(const ReferenceElement& refElem,
                            FacetElementTransform& trans,
                            Vector& elvec) const;
    
    const char* name() const override { return "ConvectionBoundaryIntegrator"; }
    
private:
    std::shared_ptr<Coefficient> Tinf_;
};

// =============================================================================
// Vector Mass Integrator
// =============================================================================

class VectorMassIntegrator : public VectorBilinearFormIntegrator {
public:
    VectorMassIntegrator() = default;
    explicit VectorMassIntegrator(std::shared_ptr<Coefficient> rho) 
        : VectorBilinearFormIntegrator(std::move(rho)) {}
    
    void assembleElementMatrix(const ReferenceElement& refElem,
                               ElementTransform& trans,
                               int vdim,
                               Matrix& elmat) const override;
    
    const char* name() const override { return "VectorMassIntegrator"; }
};

// =============================================================================
// Inline Implementations
// =============================================================================

inline void DiffusionIntegrator::assembleElementMatrix(const ReferenceElement& refElem,
                                                        ElementTransform& trans,
                                                        Matrix& elmat) const {
    const int nd = refElem.numDofs();
    const int dim = refElem.dim();
    
    elmat.setZero(nd, nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        ShapeValues sv = refElem.evalShape(ip);
        
        std::vector<Vector3> physGrad(nd);
        for (int i = 0; i < nd; ++i) {
            trans.transformGradient(sv.gradients[i], physGrad[i]);
        }
        
        Real coeff = evalCoefficient(trans);
        
        for (int i = 0; i < nd; ++i) {
            for (int j = 0; j < nd; ++j) {
                Real dot = 0.0;
                for (int d = 0; d < dim; ++d) {
                    dot += physGrad[i][d] * physGrad[j][d];
                }
                elmat(i, j) += w * coeff * dot;
            }
        }
    }
}

inline void MassIntegrator::assembleElementMatrix(const ReferenceElement& refElem,
                                                   ElementTransform& trans,
                                                   Matrix& elmat) const {
    const int nd = refElem.numDofs();
    
    elmat.setZero(nd, nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        std::vector<Real> phi = refElem.evalShape(ip).values;
        
        Real coeff = evalCoefficient(trans);
        
        for (int i = 0; i < nd; ++i) {
            for (int j = 0; j < nd; ++j) {
                elmat(i, j) += w * coeff * phi[i] * phi[j];
            }
        }
    }
}

inline void DomainLFIntegrator::assembleElementVector(const ReferenceElement& refElem,
                                                       ElementTransform& trans,
                                                       Vector& elvec) const {
    const int nd = refElem.numDofs();
    
    elvec.setZero(nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        std::vector<Real> phi = refElem.evalShape(ip).values;
        
        Real f = evalCoefficient(trans);
        
        for (int i = 0; i < nd; ++i) {
            elvec(i) += w * f * phi[i];
        }
    }
}

inline void BoundaryLFIntegrator::assembleFaceVector(const ReferenceElement& refElem,
                                                      FacetElementTransform& trans,
                                                      Vector& elvec) const {
    const int nd = refElem.numDofs();
    
    elvec.setZero(nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        std::vector<Real> phi = refElem.evalShape(ip).values;
        
        Real g = evalCoefficient(trans);
        
        for (int i = 0; i < nd; ++i) {
            elvec(i) += w * g * phi[i];
        }
    }
}

inline void ConvectionBoundaryIntegrator::assembleFaceMatrix(const ReferenceElement& refElem,
                                                              FacetElementTransform& trans,
                                                              Matrix& elmat) const {
    const int nd = refElem.numDofs();
    
    elmat.setZero(nd, nd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        std::vector<Real> phi = refElem.evalShape(ip).values;
        
        Real h = evalCoefficient(trans);
        
        for (int i = 0; i < nd; ++i) {
            for (int j = 0; j < nd; ++j) {
                elmat(i, j) += w * h * phi[i] * phi[j];
            }
        }
    }
}

inline void ConvectionBoundaryIntegrator::assembleFaceVector(const ReferenceElement& refElem,
                                                              FacetElementTransform& trans,
                                                              Vector& elvec) const {
    const int nd = refElem.numDofs();
    
    elvec.setZero(nd);
    
    if (!Tinf_) return;
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        std::vector<Real> phi = refElem.evalShape(ip).values;
        
        Real h = evalCoefficient(trans);
        Real Tinf = Tinf_->eval(trans);
        
        for (int i = 0; i < nd; ++i) {
            elvec(i) += w * h * Tinf * phi[i];
        }
    }
}

inline void VectorMassIntegrator::assembleElementMatrix(const ReferenceElement& refElem,
                                                         ElementTransform& trans,
                                                         int vdim,
                                                         Matrix& elmat) const {
    const int nd = refElem.numDofs();
    const int nvd = nd * vdim;
    
    elmat.setZero(nvd, nvd);
    
    const QuadratureRule& rule = refElem.quadrature();
    
    for (const auto& ip : rule) {
        trans.setIntegrationPoint(ip);
        
        Real w = ip.weight * trans.weight();
        
        std::vector<Real> phi = refElem.evalShape(ip).values;
        
        Real rho = evalCoefficient(trans);
        
        for (int c = 0; c < vdim; ++c) {
            for (int i = 0; i < nd; ++i) {
                for (int j = 0; j < nd; ++j) {
                    elmat(c * nd + i, c * nd + j) += w * rho * phi[i] * phi[j];
                }
            }
        }
    }
}

}  // namespace mpfem

#endif  // MPFEM_INTEGRATORS_HPP