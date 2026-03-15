#ifndef MPFEM_INTEGRATORS_HPP
#define MPFEM_INTEGRATORS_HPP

#include "integrator.hpp"
#include "fe/facet_element_transform.hpp"
#include "fe/grid_function.hpp"

namespace mpfem {

// =============================================================================
// 域积分器（双线性型）- 标量场
// =============================================================================

/// 扩散积分器: ∫ σ ∇φᵢ · ∇φⱼ dΩ
class DiffusionIntegrator : public DomainBilinearIntegrator {
public:
    using DomainBilinearIntegrator::DomainBilinearIntegrator;
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
};

/// 质量积分器: ∫ ρ φᵢ φⱼ dΩ
class MassIntegrator : public DomainBilinearIntegrator {
public:
    using DomainBilinearIntegrator::DomainBilinearIntegrator;
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
};

// =============================================================================
// 域积分器（线性型）- 标量场
// =============================================================================

/// 域载荷积分器: ∫ f φᵢ dΩ
class DomainLFIntegrator : public DomainLinearIntegrator {
public:
    using DomainLinearIntegrator::DomainLinearIntegrator;
    
    void assembleElementVector(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Vector& elvec) const override;
};

// =============================================================================
// 边界积分器（线性型）- 标量场
// =============================================================================

/// 边界载荷积分器: ∫ g φᵢ dΓ
class BoundaryLFIntegrator : public FaceLinearIntegrator {
public:
    using FaceLinearIntegrator::FaceLinearIntegrator;
    
    void assembleFaceVector(const ReferenceElement& ref,
                            FacetElementTransform& trans,
                            Vector& elvec) const override;
};

// =============================================================================
// 边界积分器（双线性型）- 标量场
// =============================================================================

/// 对流边界质量积分器 (Robin BC矩阵部分): ∫ h φᵢ φⱼ dΓ
class ConvectionMassIntegrator : public FaceBilinearIntegrator {
public:
    using FaceBilinearIntegrator::FaceBilinearIntegrator;
    
    void assembleFaceMatrix(const ReferenceElement& ref,
                            FacetElementTransform& trans,
                            Matrix& elmat) const override;
};

/// 对流边界载荷积分器 (Robin BC向量部分): ∫ h Tinf φᵢ dΓ
class ConvectionLFIntegrator : public FaceLinearIntegrator {
public:
    ConvectionLFIntegrator() = default;
    
    ConvectionLFIntegrator(const Coefficient* h, const Coefficient* Tinf)
        : FaceLinearIntegrator(h), Tinf_(Tinf) {}
    
    ConvectionLFIntegrator(std::unique_ptr<Coefficient> h, std::unique_ptr<Coefficient> Tinf)
        : FaceLinearIntegrator(std::move(h)), ownedTinf_(std::move(Tinf)), Tinf_(ownedTinf_.get()) {}
    
    void setAmbientTemperature(const Coefficient* Tinf) { Tinf_ = Tinf; }
    
    void assembleFaceVector(const ReferenceElement& ref,
                            FacetElementTransform& trans,
                            Vector& elvec) const override;
    
private:
    std::unique_ptr<Coefficient> ownedTinf_;
    const Coefficient* Tinf_ = nullptr;
};

// =============================================================================
// 弹性力学积分器 - 向量场专用
// =============================================================================

/// 弹性积分器: ∫ (λ div(u) div(v) + 2μ ε(u):ε(v)) dΩ
/// 输出 elmat 为 (nd*vdim) x (nd*vdim) 矩阵
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

/// 热膨胀载荷积分器: ∫ (3K α_T (T - T_ref) div(v)) dΩ
/// 输出 elvec 为 (nd*vdim) 维向量
class ThermalLoadIntegrator : public VectorDomainLinearIntegrator {
public:
    ThermalLoadIntegrator(const Coefficient* E, const Coefficient* nu,
                          const Coefficient* alphaT, const GridFunction* T, Real Tref)
        : E_(E), nu_(nu), alphaT_(alphaT), T_(T), Tref_(Tref) {}
    
    void assembleElementVector(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Vector& elvec,
                               int vdim) const override;
    
private:
    const Coefficient* E_ = nullptr;
    const Coefficient* nu_ = nullptr;
    const Coefficient* alphaT_ = nullptr;
    const GridFunction* T_ = nullptr;
    Real Tref_;
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATORS_HPP