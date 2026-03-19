#ifndef MPFEM_INTEGRATORS_HPP
#define MPFEM_INTEGRATORS_HPP

#include "integrator.hpp"
#include "fe/facet_element_transform.hpp"
#include "fe/grid_function.hpp"

namespace mpfem {

// =============================================================================
// Domain integrators (bilinear) - scalar field
// =============================================================================

/**
 * @brief Diffusion integrator: ∫ σ ∇φᵢ · ∇φⱼ dΩ
 * 
 * For isotropic diffusion with scalar coefficient.
 * For anisotropic diffusion, use AnisotropicDiffusionIntegrator.
 */
class DiffusionIntegrator : public DomainBilinearIntegrator {
public:
    using DomainBilinearIntegrator::DomainBilinearIntegrator;
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
};

/**
 * @brief Anisotropic diffusion integrator: ∫ (∇φᵢ)ᵀ · D · ∇φⱼ dΩ
 * 
 * D is a 3x3 matrix coefficient (diffusivity tensor).
 * For isotropic case, D = σ * I (identity matrix).
 */
class AnisotropicDiffusionIntegrator : public DomainBilinearIntegrator, 
                                        public MatrixCoefficientIntegratorBase {
public:
    using MatrixCoefficientIntegratorBase::MatrixCoefficientIntegratorBase;
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
};

/**
 * @brief Mass integrator: ∫ ρ φᵢ φⱼ dΩ
 */
class MassIntegrator : public DomainBilinearIntegrator {
public:
    using DomainBilinearIntegrator::DomainBilinearIntegrator;
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
};

// =============================================================================
// Domain integrators (linear) - scalar field
// =============================================================================

/**
 * @brief Domain load integrator: ∫ f φᵢ dΩ
 */
class DomainLFIntegrator : public DomainLinearIntegrator {
public:
    using DomainLinearIntegrator::DomainLinearIntegrator;
    
    void assembleElementVector(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Vector& elvec) const override;
};

// =============================================================================
// Face integrators (linear) - scalar field
// =============================================================================

/**
 * @brief Boundary load integrator: ∫ g φᵢ dΓ
 */
class BoundaryLFIntegrator : public FaceLinearIntegrator {
public:
    using FaceLinearIntegrator::FaceLinearIntegrator;
    
    void assembleFaceVector(const ReferenceElement& ref,
                            FacetElementTransform& trans,
                            Vector& elvec) const override;
};

// =============================================================================
// Face integrators (bilinear) - scalar field
// =============================================================================

/**
 * @brief Convection mass integrator (Robin BC matrix part): ∫ h φᵢ φⱼ dΓ
 */
class ConvectionMassIntegrator : public FaceBilinearIntegrator {
public:
    using FaceBilinearIntegrator::FaceBilinearIntegrator;
    
    void assembleFaceMatrix(const ReferenceElement& ref,
                            FacetElementTransform& trans,
                            Matrix& elmat) const override;
};

/**
 * @brief Convection load integrator (Robin BC vector part): ∫ h Tinf φᵢ dΓ
 */
class ConvectionLFIntegrator : public FaceLinearIntegrator {
public:
    ConvectionLFIntegrator() = default;
    
    /// Constructor with convection coefficient and ambient temperature coefficient (non-owning)
    ConvectionLFIntegrator(const Coefficient* h, const Coefficient* Tinf)
        : FaceLinearIntegrator(h), Tinf_(Tinf) {}
    
    void setAmbientTemperature(const Coefficient* Tinf) { Tinf_ = Tinf; }
    
    void assembleFaceVector(const ReferenceElement& ref,
                            FacetElementTransform& trans,
                            Vector& elvec) const override;
    
private:
    const Coefficient* Tinf_ = nullptr;  ///< Ambient temperature coefficient (non-owning)
};

// =============================================================================
// Elasticity integrators - vector field specific
// =============================================================================

/**
 * @brief Elasticity integrator: ∫ (λ div(u) div(v) + 2μ ε(u):ε(v)) dΩ
 * 
 * Output elmat is (nd*vdim) x (nd*vdim) matrix.
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
 * 
 * Output elvec is (nd*vdim) dimensional vector.
 * Note: alphaT coefficient should return thermal strain alpha_T * (T - Tref)
 *       e.g., use a lambda-based ScalarCoefficient.
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
