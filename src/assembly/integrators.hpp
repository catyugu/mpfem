#ifndef MPFEM_INTEGRATORS_HPP
#define MPFEM_INTEGRATORS_HPP

#include "integrator.hpp"
#include "fe/facet_element_transform.hpp"

namespace mpfem {

/// 扩散积分器: ∫ σ ∇φᵢ · ∇φⱼ dΩ
class DiffusionIntegrator : public BilinearFormIntegrator {
public:
    DiffusionIntegrator() = default;
    explicit DiffusionIntegrator(const Coefficient* q) : BilinearFormIntegrator(q) {}
    explicit DiffusionIntegrator(std::unique_ptr<Coefficient> q) 
        : BilinearFormIntegrator(std::move(q)) {}
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
};

/// 质量积分器: ∫ ρ φᵢ φⱼ dΩ
class MassIntegrator : public BilinearFormIntegrator {
public:
    MassIntegrator() = default;
    explicit MassIntegrator(const Coefficient* q) : BilinearFormIntegrator(q) {}
    explicit MassIntegrator(std::unique_ptr<Coefficient> q) 
        : BilinearFormIntegrator(std::move(q)) {}
    
    void assembleElementMatrix(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Matrix& elmat) const override;
};

/// 域载荷积分器: ∫ f φᵢ dΩ
class DomainLFIntegrator : public LinearFormIntegrator {
public:
    DomainLFIntegrator() = default;
    explicit DomainLFIntegrator(const Coefficient* f) : LinearFormIntegrator(f) {}
    explicit DomainLFIntegrator(std::unique_ptr<Coefficient> f) 
        : LinearFormIntegrator(std::move(f)) {}
    
    void assembleElementVector(const ReferenceElement& ref,
                               ElementTransform& trans,
                               Vector& elvec) const override;
};

/// 边界载荷积分器: ∫ g φᵢ dΓ
class BoundaryLFIntegrator : public LinearFormIntegrator {
public:
    BoundaryLFIntegrator() = default;
    explicit BoundaryLFIntegrator(const Coefficient* g) : LinearFormIntegrator(g) {}
    explicit BoundaryLFIntegrator(std::unique_ptr<Coefficient> g) 
        : LinearFormIntegrator(std::move(g)) {}
    
    void assembleElementVector(const ReferenceElement&, ElementTransform&, Vector& elvec) const override {
        elvec.setZero(0);
    }
    
    void assembleFaceVector(const ReferenceElement& ref,
                            FacetElementTransform& trans,
                            Vector& elvec) const override;
};

/// 对流边界积分器 (Robin BC): ∫ h φᵢ φⱼ dΓ
class ConvectionBoundaryIntegrator : public BilinearFormIntegrator {
public:
    ConvectionBoundaryIntegrator() = default;
    
    /// 非拥有引用版本
    ConvectionBoundaryIntegrator(const Coefficient* h, const Coefficient* Tinf = nullptr)
        : BilinearFormIntegrator(h), Tinf_(Tinf) {}
    
    /// 拥有版本（h 系数）
    explicit ConvectionBoundaryIntegrator(std::unique_ptr<Coefficient> h)
        : BilinearFormIntegrator(std::move(h)) {}
    
    /// 拥有版本（h 和 Tinf 系数）
    ConvectionBoundaryIntegrator(std::unique_ptr<Coefficient> h, std::unique_ptr<Coefficient> Tinf)
        : BilinearFormIntegrator(std::move(h)), ownedTinf_(std::move(Tinf)), Tinf_(ownedTinf_.get()) {}
    
    void setAmbientTemperature(const Coefficient* Tinf) { Tinf_ = Tinf; }
    void setOwnedAmbientTemperature(std::unique_ptr<Coefficient> Tinf) {
        ownedTinf_ = std::move(Tinf);
        Tinf_ = ownedTinf_.get();
    }
    
    void assembleElementMatrix(const ReferenceElement&, ElementTransform&, Matrix& elmat) const override {
        elmat.setZero(0, 0);
    }
    
    void assembleFaceMatrix(const ReferenceElement& ref,
                            FacetElementTransform& trans,
                            Matrix& elmat) const override;
    
    void assembleFaceVector(const ReferenceElement& ref,
                            FacetElementTransform& trans,
                            Vector& elvec) const;
    
private:
    std::unique_ptr<Coefficient> ownedTinf_;  ///< 持有的环境温度系数
    const Coefficient* Tinf_ = nullptr;       ///< 环境温度系数指针
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATORS_HPP