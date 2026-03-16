#ifndef MPFEM_COUPLING_JOULE_HEATING_HPP
#define MPFEM_COUPLING_JOULE_HEATING_HPP

#include "fe/coefficient.hpp"
#include "fe/grid_function.hpp"
#include "core/types.hpp"
#include <memory>
#include <set>

namespace mpfem {

/**
 * @file joule_heating.hpp
 * @brief 焦耳热耦合管理器
 * 
 * 设计原则：耦合逻辑独立于单场求解器。
 * 该模块管理电场和热场之间的焦耳热耦合。
 */

/**
 * @brief 焦耳热耦合管理器
 * 
 * 管理电势场到热源场的耦合：
 * Q = sigma * |grad V|^2
 */
class JouleHeatingCoupling {
public:
    /// 设置电势场（非拥有指针）
    void setPotentialField(const GridFunction* V) { potential_ = V; }
    
    /// 设置电导率系数（非拥有指针）
    void setConductivity(const Coefficient* sigma) { conductivity_ = sigma; }
    
    /// 限制耦合到指定域（空集表示所有域）
    void setDomains(const std::set<int>& domains) { domains_ = domains; }
    
    /// 获取焦耳热系数（延迟创建）
    const Coefficient* getHeatSource() {
        if (!jouleHeat_) {
            jouleHeat_ = std::make_unique<JouleHeatCoefficient>();
        }
        jouleHeat_->setPotential(potential_);
        jouleHeat_->setConductivity(conductivity_);
        jouleHeat_->setDomains(domains_);
        return jouleHeat_.get();
    }
    
private:
    const GridFunction* potential_ = nullptr;
    const Coefficient* conductivity_ = nullptr;
    std::set<int> domains_;
    std::unique_ptr<JouleHeatCoefficient> jouleHeat_;
};

}  // namespace mpfem

#endif  // MPFEM_COUPLING_JOULE_HEATING_HPP