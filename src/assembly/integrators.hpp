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

/**
 * @brief Integrator for diffusion term: ∫ σ ∇φᵢ · ∇φⱼ dΩ
 * 
 * Vectorized implementation using Eigen matrix operations.
 */
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

/**
 * @brief Integrator for mass term: ∫ ρ φᵢ φⱼ dΩ
 * 
 * Vectorized implementation using outer product.
 */
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

/**
 * @brief Integrator for domain load: ∫ f φᵢ dΩ
 * 
 * Vectorized implementation.
 */
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

/**
 * @brief Integrator for boundary load (Neumann BC): ∫ g φᵢ dΓ
 * 
 * Vectorized implementation.
 */
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

/**
 * @brief Integrator for convection boundary condition (Robin BC).
 * 
 * Bilinear term: ∫ h φᵢ φⱼ dΓ
 * Linear term: ∫ h T∞ φᵢ dΓ
 * 
 * Vectorized implementation.
 */
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

/**
 * @brief Integrator for vector mass term: ∫ ρ φᵢ · φⱼ dΩ
 * 
 * For vector fields (e.g., displacement), forms block diagonal mass matrix.
 * Vectorized implementation.
 */
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

}  // namespace mpfem

#endif  // MPFEM_INTEGRATORS_HPP
