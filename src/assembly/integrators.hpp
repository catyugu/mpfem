#ifndef MPFEM_INTEGRATORS_HPP
#define MPFEM_INTEGRATORS_HPP

#include "integrator.hpp"
#include "core/exception.hpp"
#include "expr/variable_graph.hpp"

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
    explicit DiffusionIntegrator(const VariableNode* c) : coef_(c) {
        if (!coef_) MPFEM_THROW(ArgumentException, "DiffusionIntegrator requires non-null matrix node");
    }
    
    int vdim() const override { return 1; }
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
private:
    const VariableNode* coef_ = nullptr;
};

// =============================================================================
// Mass integrator
// =============================================================================

/**
 * @brief Mass integrator: ∫ ρ φᵢ φⱼ dΩ
 */
class MassIntegrator : public DomainBilinearIntegratorBase {
public:
    explicit MassIntegrator(const VariableNode* c) : coef_(c) {
        if (!coef_) MPFEM_THROW(ArgumentException, "MassIntegrator requires non-null scalar node");
    }
    
    int vdim() const override { return 1; }
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
private:
    const VariableNode* coef_ = nullptr;
};

// =============================================================================
// Domain load integrator
// =============================================================================

/**
 * @brief Domain load integrator: ∫ f φᵢ dΩ
 */
class DomainLFIntegrator : public DomainLinearIntegratorBase {
public:
    explicit DomainLFIntegrator(const VariableNode* c) : coef_(c) {
        if (!coef_) MPFEM_THROW(ArgumentException, "DomainLFIntegrator requires non-null scalar node");
    }
    
    int vdim() const override { return 1; }
    
    void assembleElementVector(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Vector& elvec) const override;
private:
    const VariableNode* coef_ = nullptr;
};

// =============================================================================
// Boundary load integrator
// =============================================================================

/**
 * @brief Boundary load integrator: ∫ g φᵢ dΓ
 */
class BoundaryLFIntegrator : public FaceLinearIntegratorBase {
public:
    explicit BoundaryLFIntegrator(const VariableNode* c) : coef_(c) {
        if (!coef_) MPFEM_THROW(ArgumentException, "BoundaryLFIntegrator requires non-null scalar node");
    }
    
    void assembleFaceVector(const ReferenceElement& ref,
                            ElementTransform& trans,
                            Vector& elvec) const override;
private:
    const VariableNode* coef_ = nullptr;
};

// =============================================================================
// Convection integrators (Robin BC)
// =============================================================================

/**
 * @brief Convection mass integrator (Robin BC matrix part): ∫ h φᵢ φⱼ dΓ
 */
class ConvectionMassIntegrator : public FaceBilinearIntegratorBase {
public:
    explicit ConvectionMassIntegrator(const VariableNode* c) : coef_(c) {
        if (!coef_) MPFEM_THROW(ArgumentException, "ConvectionMassIntegrator requires non-null scalar node");
    }
    
    void assembleFaceMatrix(const ReferenceElement& ref,
                            ElementTransform& trans,
                            Matrix& elmat) const override;
private:
    const VariableNode* coef_ = nullptr;
};

/**
 * @brief Convection load integrator (Robin BC vector part): ∫ h Tinf φᵢ dΓ
 */
class ConvectionLFIntegrator : public FaceLinearIntegratorBase {
public:
    ConvectionLFIntegrator(const VariableNode* h, const VariableNode* Tinf)
        : coef_(h), Tinf_(Tinf) {
        if (!coef_) MPFEM_THROW(ArgumentException, "ConvectionLFIntegrator requires non-null h node");
        if (!Tinf_) MPFEM_THROW(ArgumentException, "ConvectionLFIntegrator requires non-null Tinf node");
    }
    
    void assembleFaceVector(const ReferenceElement& ref,
                            ElementTransform& trans,
                            Vector& elvec) const override;
    
private:
    const VariableNode* coef_ = nullptr;
    const VariableNode* Tinf_ = nullptr;
};

// =============================================================================
// Elasticity integrators
// =============================================================================

/**
 * @brief Elasticity integrator: ∫ (λ div(u) div(v) + 2μ ε(u):ε(v)) dΩ
 */
class ElasticityIntegrator : public DomainBilinearIntegratorBase {
public:
    ElasticityIntegrator(const VariableNode* E, const VariableNode* nu, int vdim)
        : E_(E), nu_(nu), vdim_(vdim) {
        if (!E_) MPFEM_THROW(ArgumentException, "ElasticityIntegrator requires non-null E node");
        if (!nu_) MPFEM_THROW(ArgumentException, "ElasticityIntegrator requires non-null nu node");
        if (vdim_ <= 0) MPFEM_THROW(ArgumentException, "ElasticityIntegrator requires vdim > 0");
    }
    
    int vdim() const override { return vdim_; }
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
    
private:
    const VariableNode* E_ = nullptr;
    const VariableNode* nu_ = nullptr;
    int vdim_ = 1;
};

/**
 * @brief Strain load integrator: ∫ σ : ε(v) dΩ
 *
 * Accepts a 3x3 stress tensor coefficient and assembles B^T * sigma(Voigt).
 * This integrator is independent from how stress is produced.
 */
class StrainLoadIntegrator : public DomainLinearIntegratorBase {
public:
    StrainLoadIntegrator(const VariableNode* stress, int vdim)
        : stress_(stress), vdim_(vdim) {
        if (!stress_) MPFEM_THROW(ArgumentException, "StrainLoadIntegrator requires non-null stress node");
        if (vdim_ <= 0) MPFEM_THROW(ArgumentException, "StrainLoadIntegrator requires vdim > 0");
    }
    
    int vdim() const override { return vdim_; }
    
    void assembleElementVector(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Vector& elvec) const override;
    
private:
    const VariableNode* stress_ = nullptr;
    int vdim_ = 1;
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATORS_HPP
