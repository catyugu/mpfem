#ifndef MPFEM_INTEGRATORS_HPP
#define MPFEM_INTEGRATORS_HPP

#include "integrator.hpp"
#include "core/exception.hpp"

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
    explicit DiffusionIntegrator(const MatrixCoefficient* c) : coef_(c) {
        if (!coef_) MPFEM_THROW(ArgumentException, "DiffusionIntegrator requires non-null MatrixCoefficient");
    }
    
    int vdim() const override { return 1; }
    
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
    explicit MassIntegrator(const Coefficient* c) : coef_(c) {
        if (!coef_) MPFEM_THROW(ArgumentException, "MassIntegrator requires non-null Coefficient");
    }
    
    int vdim() const override { return 1; }
    
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
    explicit DomainLFIntegrator(const Coefficient* c) : coef_(c) {
        if (!coef_) MPFEM_THROW(ArgumentException, "DomainLFIntegrator requires non-null Coefficient");
    }
    
    int vdim() const override { return 1; }
    
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
    explicit BoundaryLFIntegrator(const Coefficient* c) : coef_(c) {
        if (!coef_) MPFEM_THROW(ArgumentException, "BoundaryLFIntegrator requires non-null Coefficient");
    }
    
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
    explicit ConvectionMassIntegrator(const Coefficient* c) : coef_(c) {
        if (!coef_) MPFEM_THROW(ArgumentException, "ConvectionMassIntegrator requires non-null Coefficient");
    }
    
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
        : coef_(h), Tinf_(Tinf) {
        if (!coef_) MPFEM_THROW(ArgumentException, "ConvectionLFIntegrator requires non-null h Coefficient");
        if (!Tinf_) MPFEM_THROW(ArgumentException, "ConvectionLFIntegrator requires non-null Tinf Coefficient");
    }
    
    void setAmbientTemperature(const Coefficient* Tinf) {
        if (!Tinf_) MPFEM_THROW(ArgumentException, "ConvectionLFIntegrator::setAmbientTemperature requires non-null Coefficient");
        Tinf_ = Tinf;
    }
    
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
class ElasticityIntegrator : public DomainBilinearIntegratorBase {
public:
    ElasticityIntegrator(const Coefficient* E, const Coefficient* nu, int vdim)
        : E_(E), nu_(nu), vdim_(vdim) {
        if (!E_) MPFEM_THROW(ArgumentException, "ElasticityIntegrator requires non-null E Coefficient");
        if (!nu_) MPFEM_THROW(ArgumentException, "ElasticityIntegrator requires non-null nu Coefficient");
        if (vdim_ <= 0) MPFEM_THROW(ArgumentException, "ElasticityIntegrator requires vdim > 0");
    }
    
    int vdim() const override { return vdim_; }
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
    
private:
    const Coefficient* E_ = nullptr;
    const Coefficient* nu_ = nullptr;
    int vdim_ = 1;
};

/**
 * @brief Thermal load integrator: ∫ σ_thermal : ε(v) dΩ
 * 
 * Computes anisotropic thermal stress: σ_thermal = C : α_tensor : (T - T_ref)
 * - C is the 6×6 elasticity tensor (computed from E and nu)
 * - α_tensor is the 3×3 thermal expansion coefficient matrix
 * - ε_thermal in Voigt notation = {α₁₁, α₂₂, α₃₃, 2α₁₂, 2α₁₃, 2α₂₃} · (T - T_ref)
 */
class ThermalLoadIntegrator : public DomainLinearIntegratorBase {
public:
    ThermalLoadIntegrator(const Coefficient* E,
                          const Coefficient* nu,
                          const MatrixCoefficient* alphaT,
                          int vdim)
        : E_(E), nu_(nu), alphaT_(alphaT), vdim_(vdim) {
        if (!E_) MPFEM_THROW(ArgumentException, "ThermalLoadIntegrator requires non-null E Coefficient");
        if (!nu_) MPFEM_THROW(ArgumentException, "ThermalLoadIntegrator requires non-null nu Coefficient");
        if (!alphaT_) MPFEM_THROW(ArgumentException, "ThermalLoadIntegrator requires non-null alphaT MatrixCoefficient");
        if (vdim_ <= 0) MPFEM_THROW(ArgumentException, "ThermalLoadIntegrator requires vdim > 0");
    }
    
    int vdim() const override { return vdim_; }
    
    void assembleElementVector(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Vector& elvec) const override;
    
private:
    const Coefficient* E_ = nullptr;
    const Coefficient* nu_ = nullptr;
    const MatrixCoefficient* alphaT_ = nullptr;
    int vdim_ = 1;
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATORS_HPP
