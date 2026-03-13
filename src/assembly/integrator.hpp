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
// 双线性型积分器基类
// =============================================================================

class BilinearFormIntegrator {
public:
    BilinearFormIntegrator() = default;
    
    /// 非拥有引用版本
    explicit BilinearFormIntegrator(const Coefficient* q) : q_(q) {}
    
    /// 拥有版本（解决内存泄漏）
    explicit BilinearFormIntegrator(std::unique_ptr<Coefficient> q)
        : ownedQ_(std::move(q)), q_(ownedQ_.get()) {}
    
    virtual ~BilinearFormIntegrator() = default;
    
    // 禁止拷贝，允许移动
    BilinearFormIntegrator(const BilinearFormIntegrator&) = delete;
    BilinearFormIntegrator& operator=(const BilinearFormIntegrator&) = delete;
    BilinearFormIntegrator(BilinearFormIntegrator&&) = default;
    BilinearFormIntegrator& operator=(BilinearFormIntegrator&&) = default;
    
    virtual void assembleElementMatrix(const ReferenceElement& ref, 
                                        ElementTransform& trans, 
                                        Matrix& elmat) const = 0;
    virtual void assembleFaceMatrix(const ReferenceElement& ref,
                                     FacetElementTransform& trans,
                                     Matrix& elmat) const {}
    
protected:
    std::unique_ptr<Coefficient> ownedQ_;  ///< 持有的系数（可选）
    const Coefficient* q_ = nullptr;       ///< 系数指针（拥有或非拥有）
    Real evalCoef(ElementTransform& t) const { return q_ ? q_->eval(t) : 1.0; }
};

// =============================================================================
// 线性型积分器基类
// =============================================================================

class LinearFormIntegrator {
public:
    LinearFormIntegrator() = default;
    
    /// 非拥有引用版本
    explicit LinearFormIntegrator(const Coefficient* q) : q_(q) {}
    
    /// 拥有版本（解决内存泄漏）
    explicit LinearFormIntegrator(std::unique_ptr<Coefficient> q)
        : ownedQ_(std::move(q)), q_(ownedQ_.get()) {}
    
    virtual ~LinearFormIntegrator() = default;
    
    // 禁止拷贝，允许移动
    LinearFormIntegrator(const LinearFormIntegrator&) = delete;
    LinearFormIntegrator& operator=(const LinearFormIntegrator&) = delete;
    LinearFormIntegrator(LinearFormIntegrator&&) = default;
    LinearFormIntegrator& operator=(LinearFormIntegrator&&) = default;
    
    virtual void assembleElementVector(const ReferenceElement& ref,
                                        ElementTransform& trans,
                                        Vector& elvec) const = 0;
    virtual void assembleFaceVector(const ReferenceElement& ref,
                                     FacetElementTransform& trans,
                                     Vector& elvec) const {}
    
protected:
    std::unique_ptr<Coefficient> ownedQ_;  ///< 持有的系数（可选）
    const Coefficient* q_ = nullptr;       ///< 系数指针（拥有或非拥有）
    Real evalCoef(ElementTransform& t) const { return q_ ? q_->eval(t) : 1.0; }
};

}  // namespace mpfem

#endif  // MPFEM_INTEGRATOR_HPP