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
// 基础积分器接口 - 只提供系数评估
// =============================================================================

class IntegratorBase {
protected:
    std::unique_ptr<Coefficient> ownedQ_;  ///< 持有的系数（可选）
    const Coefficient* q_ = nullptr;       ///< 系数指针（拥有或非拥有）
    
    Real evalCoef(ElementTransform& t) const { return q_ ? q_->eval(t) : 1.0; }
    Real evalCoef(FacetElementTransform& t) const { return q_ ? q_->eval(t) : 1.0; }

public:
    IntegratorBase() = default;
    explicit IntegratorBase(const Coefficient* q) : q_(q) {}
    explicit IntegratorBase(std::unique_ptr<Coefficient> q)
        : ownedQ_(std::move(q)), q_(ownedQ_.get()) {}
    
    virtual ~IntegratorBase() = default;
    
    IntegratorBase(const IntegratorBase&) = delete;
    IntegratorBase& operator=(const IntegratorBase&) = delete;
    IntegratorBase(IntegratorBase&&) = default;
    IntegratorBase& operator=(IntegratorBase&&) = default;
};

// =============================================================================
// 双线性型域积分器基类 - 标量场
// =============================================================================

class DomainBilinearIntegrator : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    
    /// 组装单元矩阵: ∫ coef * L(φ_i) * L(φ_j) dΩ
    /// 输出 elmat 为 nd x nd 矩阵
    virtual void assembleElementMatrix(const ReferenceElement& ref, 
                                        ElementTransform& trans, 
                                        Matrix& elmat) const = 0;
};

// =============================================================================
// 双线性型边界积分器基类 - 标量场
// =============================================================================

class FaceBilinearIntegrator : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    
    /// 组装边界矩阵: ∫ coef * L(φ_i) * L(φ_j) dΓ
    /// 输出 elmat 为 nd x nd 矩阵
    virtual void assembleFaceMatrix(const ReferenceElement& ref,
                                     FacetElementTransform& trans,
                                     Matrix& elmat) const = 0;
};

// =============================================================================
// 线性型域积分器基类 - 标量场
// =============================================================================

class DomainLinearIntegrator : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    
    /// 组装单元向量: ∫ coef * L(φ_i) dΩ
    /// 输出 elvec 为 nd 维向量
    virtual void assembleElementVector(const ReferenceElement& ref,
                                        ElementTransform& trans,
                                        Vector& elvec) const = 0;
};

// =============================================================================
// 线性型边界积分器基类 - 标量场
// =============================================================================

class FaceLinearIntegrator : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    
    /// 组装边界向量: ∫ coef * L(φ_i) dΓ
    /// 输出 elvec 为 nd 维向量
    virtual void assembleFaceVector(const ReferenceElement& ref,
                                     FacetElementTransform& trans,
                                     Vector& elvec) const = 0;
};

// =============================================================================
// 向量场积分器基类（用于位移场等）
// =============================================================================

/// 向量场域积分器基类
class VectorDomainBilinearIntegrator : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    
    /// 组装单元矩阵（向量场版本）
    /// vdim 由 FESpace 提供，积分器内部使用
    virtual void assembleElementMatrix(const ReferenceElement& ref,
                                        ElementTransform& trans,
                                        Matrix& elmat,
                                        int vdim) const = 0;
};

/// 向量场域线性积分器基类
class VectorDomainLinearIntegrator : public IntegratorBase {
public:
    using IntegratorBase::IntegratorBase;
    
    /// 组装单元向量（向量场版本）
    virtual void assembleElementVector(const ReferenceElement& ref,
                                        ElementTransform& trans,
                                        Vector& elvec,
                                        int vdim) const = 0;
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATOR_HPP