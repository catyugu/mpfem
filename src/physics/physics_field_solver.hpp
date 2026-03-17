#ifndef MPFEM_PHYSICS_FIELD_SOLVER_HPP
#define MPFEM_PHYSICS_FIELD_SOLVER_HPP

#include "model/field_kind.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include "fe/coefficient.hpp"
#include "mesh/mesh.hpp"
#include "assembly/assembler.hpp"
#include "solver/solver_config.hpp"
#include "solver/solver_factory.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief 物理场求解器基类
 * 
 * 包含所有单场求解器的共同成员和接口。
 */
class PhysicsFieldSolver {
public:
    virtual ~PhysicsFieldSolver() = default;
    
    virtual FieldKind fieldKind() const = 0;
    virtual std::string fieldName() const = 0;
    
    virtual void assemble() = 0;
    virtual bool solve() = 0;
    
    virtual const GridFunction& field() const = 0;
    virtual GridFunction& field() = 0;
    
    const FESpace& feSpace() const { return *fes_; }
    Index numDofs() const { return fes_ ? fes_->numDofs() : 0; }
    const Mesh& mesh() const { return *mesh_; }
    
    void setOrder(int o) { order_ = o; }
    
    /// 设置求解器配置
    void setSolverConfig(const SolverConfig& config) { 
        solverConfig_ = config; 
    }
    
    int iterations() const { return iter_; }
    Real residual() const { return res_; }
    
protected:
    /// 创建求解器实例
    void createSolver() {
        solver_ = SolverFactory::create(solverConfig_);
    }
    
    // 配置参数
    int order_ = 1;
    SolverConfig solverConfig_;  // 统一使用SolverConfig
    int iter_ = 0;
    Real res_ = 0.0;
    
    // 共同成员变量
    const Mesh* mesh_ = nullptr;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<BilinearFormAssembler> matAsm_;
    std::unique_ptr<LinearFormAssembler> vecAsm_;
    std::unique_ptr<LinearSolver> solver_;
};

}  // namespace mpfem

#endif  // MPFEM_PHYSICS_FIELD_SOLVER_HPP