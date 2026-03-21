#ifndef MPFEM_INTEGRATORS_HPP
#define MPFEM_INTEGRATORS_HPP

#include "integrator.hpp"
#include "fe/facet_element_transform.hpp"
#include "fe/grid_function.hpp"

namespace mpfem {

// =============================================================================
// Diffusion integrator - matrix coefficient
// =============================================================================

/**
 * @brief Diffusion integrator: ∫ (∇φᵢ)ᵀ · D · ∇φⱼ dΩ
 * 
 * D is a 3x3 matrix coefficient (diffusivity/conductivity tensor).
 * - For isotropic: D = σ * I (use diagonalMatrixCoefficient)
 * - For anisotropic: D is full tensor
 */
class DiffusionIntegrator : public DomainBilinearIntegratorBase {
public:
    DiffusionIntegrator() = default;
    explicit DiffusionIntegrator(const MatrixCoefficient* c) : coef_(c) {}
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
private:
    const MatrixCoefficient* coef_ = nullptr;
};

// =============================================================================
// Mass integrator
// =============================================================================

/**
 * @brief Mass integrator: ∫ ρ φᵢ φⱼ dΩ
 */
class MassIntegrator : public DomainBilinearIntegratorBase {
public:
    MassIntegrator() = default;
    explicit MassIntegrator(const Coefficient* c) : coef_(c) {}
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
private:
    const Coefficient* coef_ = nullptr;
};

// =============================================================================
// Domain load integrator
// =============================================================================

/**
 * @brief Domain load integrator: ∫ f φᵢ dΩ
 */
class DomainLFIntegrator : public DomainLinearIntegratorBase {
public:
    DomainLFIntegrator() = default;
    explicit DomainLFIntegrator(const Coefficient* c) : coef_(c) {}
    
    void assembleElementVector(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Vector& elvec) const override;
private:
    const Coefficient* coef_ = nullptr;
};

// =============================================================================
// Boundary load integrator
// =============================================================================

/**
 * @brief Boundary load integrator: ∫ g φᵢ dΓ
 */
class BoundaryLFIntegrator : public FaceLinearIntegratorBase {
public:
    BoundaryLFIntegrator() = default;
    explicit BoundaryLFIntegrator(const Coefficient* c) : coef_(c) {}
    
    void assembleFaceVector(const ReferenceElement& ref,
                            FacetElementTransform& trans,
                            Vector& elvec) const override;
private:
    const Coefficient* coef_ = nullptr;
};

// =============================================================================
// Convection integrators (Robin BC)
// =============================================================================

/**
 * @brief Convection mass integrator (Robin BC matrix part): ∫ h φᵢ φⱼ dΓ
 */
class ConvectionMassIntegrator : public FaceBilinearIntegratorBase {
public:
    ConvectionMassIntegrator() = default;
    explicit ConvectionMassIntegrator(const Coefficient* c) : coef_(c) {}
    
    void assembleFaceMatrix(const ReferenceElement& ref,
                            FacetElementTransform& trans,
                            Matrix& elmat) const override;
private:
    const Coefficient* coef_ = nullptr;
};

/**
 * @brief Convection load integrator (Robin BC vector part): ∫ h Tinf φᵢ dΓ
 */
class ConvectionLFIntegrator : public FaceLinearIntegratorBase {
public:
    ConvectionLFIntegrator() = default;
    
    ConvectionLFIntegrator(const Coefficient* h, const Coefficient* Tinf)
        : coef_(h), Tinf_(Tinf) {}
    
    void setAmbientTemperature(const Coefficient* Tinf) { Tinf_ = Tinf; }
    
    void assembleFaceVector(const ReferenceElement& ref,
                            FacetElementTransform& trans,
                            Vector& elvec) const override;
    
private:
    const Coefficient* coef_ = nullptr;
    const Coefficient* Tinf_ = nullptr;
};

// =============================================================================
// Elasticity integrators
// =============================================================================

/**
 * @brief Elasticity integrator: ∫ (λ div(u) div(v) + 2μ ε(u):ε(v)) dΩ
 */
class ElasticityIntegrator : public VectorDomainBilinearIntegrator {
public:
    ElasticityIntegrator(const Coefficient* E, const Coefficient* nu)
        : E_(E), nu_(nu) {}
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat,
                               int vdim) const override;
    
private:
    const Coefficient* E_ = nullptr;
    const Coefficient* nu_ = nullptr;
};

/**
 * @brief Thermal load integrator: ∫ (3K α_T (T - T_ref) div(v)) dΩ
 */
class ThermalLoadIntegrator : public VectorDomainLinearIntegrator {
public:
    ThermalLoadIntegrator(const Coefficient* E, const Coefficient* nu,
                          const Coefficient* alphaT)
        : E_(E), nu_(nu), alphaT_(alphaT) {}
    
    void assembleElementVector(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Vector& elvec,
                               int vdim) const override;
    
private:
    const Coefficient* E_ = nullptr;
    const Coefficient* nu_ = nullptr;
    const Coefficient* alphaT_ = nullptr;
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATORS_HPP