#ifndef MPFEM_COUPLING_MANAGER_HPP
#define MPFEM_COUPLING_MANAGER_HPP

#include "physics/electrostatics_solver.hpp"
#include "physics/heat_transfer_solver.hpp"
#include "physics/structural_solver.hpp"
#include "assembly/integrators.hpp"
#include "fe/coefficient.hpp"
#include "model/field_kind.hpp"
#include "core/logger.hpp"
#include <memory>
#include <set>
#include <map>
#include <string>

namespace mpfem {

struct CouplingResult {
    bool converged = false;
    int iterations = 0;
    Real residual = 0.0;
};

/**
 * @brief 电-热-结构耦合管理器
 * 
 * 设计原则：
 * - 耦合逻辑集中在此，不在单场求解器中
 * - 求解器引用为非拥有指针，生命周期由外部管理
 * - 系数以字符串为索引统一管理，支持域选择
 */
class CouplingManager {
public:
    CouplingManager() = default;
    
    // =========================================================================
    // 求解器设置（非拥有指针）
    // =========================================================================
    
    void setElectrostaticsSolver(ElectrostaticsSolver* s) { esSolver_ = s; }
    void setHeatTransferSolver(HeatTransferSolver* s) { htSolver_ = s; }
    void setStructuralSolver(StructuralSolver* s) { stSolver_ = s; }
    
    /// 获取场的解（非拥有指针）
    const GridFunction* getField(const std::string& name) const;
    
    // =========================================================================
    // 系数管理（以字符串为索引，支持域选择）
    // =========================================================================
    
    /// 设置系数（指定域）
    void setCoefficient(const std::string& name, 
                        const std::set<int>& domains,
                        const Coefficient* coef);
    
    /// 设置系数（所有域）
    void setCoefficient(const std::string& name, const Coefficient* coef);
    
    /// 获取系数
    const Coefficient* getCoefficient(const std::string& name) const;
    
    /// 创建并持有系数（返回指针供外部使用）
    template<typename T, typename... Args>
    T* createCoefficient(const std::string& name, Args&&... args) {
        static_assert(std::is_base_of_v<Coefficient, T>, "T must derive from Coefficient");
        auto coef = std::make_unique<T>(std::forward<Args>(args)...);
        T* ptr = coef.get();
        ownedCoefficients_[name] = std::move(coef);
        return ptr;
    }
    
    // =========================================================================
    // 求解参数
    // =========================================================================
    
    void setTolerance(Real tol) { tol_ = tol; }
    void setMaxIterations(int n) { maxIter_ = n; }
    void setCouplingMethod(CouplingMethod method) { method_ = method; }
    
    // =========================================================================
    // 求解接口
    // =========================================================================
    
    /// 执行耦合求解
    CouplingResult solve();

private:
    Real computeError();
    
    // 求解器引用（非拥有）
    ElectrostaticsSolver* esSolver_ = nullptr;
    HeatTransferSolver* htSolver_ = nullptr;
    StructuralSolver* stSolver_ = nullptr;
    
    // 拥有的系数（以字符串为索引）
    std::map<std::string, std::unique_ptr<Coefficient>> ownedCoefficients_;
    
    // 非拥有的系数引用（域映射）
    std::map<std::string, DomainMappedCoefficient> coefficients_;
    
    // 迭代状态
    Vector prevT_;
    int maxIter_ = 20;
    Real tol_ = 1e-6;
    CouplingMethod method_ = CouplingMethod::Picard;
};

}  // namespace mpfem

#endif  // MPFEM_COUPLING_MANAGER_HPP
