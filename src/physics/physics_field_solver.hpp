#ifndef MPFEM_PHYSICS_FIELD_SOLVER_HPP
#define MPFEM_PHYSICS_FIELD_SOLVER_HPP

#include "model/field_kind.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include "fe/coefficient.hpp"
#include "mesh/mesh.hpp"
#include "assembly/assembler.hpp"
#include "solver/linear_solver.hpp"
#include <memory>
#include <string>

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
    void setSolver(const std::string& type, int maxIter = 1000, Real tol = 1e-10) {
        solverType_ = type; maxIter_ = maxIter; tol_ = tol;
    }
    
    int iterations() const { return iter_; }
    Real residual() const { return res_; }
    
protected:
    // 配置参数
    int order_ = 1;
    std::string solverType_ = "sparse_lu";
    int maxIter_ = 1000;
    Real tol_ = 1e-10;
    int iter_ = 0;
    Real res_ = 0.0;
    
    // 共同成员变量
    const Mesh* mesh_ = nullptr;
    std::unique_ptr<FECollection> fec_;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<BilinearFormAssembler> matAsm_;
    std::unique_ptr<LinearFormAssembler> vecAsm_;
    std::unique_ptr<LinearSolver> solver_;
};

}  // namespace mpfem

#endif  // MPFEM_PHYSICS_FIELD_SOLVER_HPP
