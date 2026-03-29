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
 */
class DiffusionIntegrator : public DomainBilinearIntegratorBase {
public:
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
 * @brief Thermal load integrator: ∫ σ_thermal : ε(v) dΩ
 * 
 * Computes anisotropic thermal stress: σ_thermal = C : α_tensor : (T - T_ref)
 * - C is the 6×6 elasticity tensor (computed from E and nu)
 * - α_tensor is the 3×3 thermal expansion coefficient matrix
 * - ε_thermal in Voigt notation = {α₁₁, α₂₂, α₃₃, 2α₁₂, 2α₁₃, 2α₂₃} · (T - T_ref)
 */
class ThermalLoadIntegrator : public VectorDomainLinearIntegrator {
public:
    ThermalLoadIntegrator(const Coefficient* E,
                          const Coefficient* nu,
                          const MatrixCoefficient* alphaT)
        : E_(E), nu_(nu), alphaT_(alphaT) {}
    
    void assembleElementVector(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Vector& elvec,
                               int vdim) const override;
    
private:
    const Coefficient* E_ = nullptr;
    const Coefficient* nu_ = nullptr;
    const MatrixCoefficient* alphaT_ = nullptr;
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATORS_HPP