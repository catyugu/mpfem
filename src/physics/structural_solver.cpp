#include "structural_solver.hpp"
#include "assembly/integrators.hpp"
#include "assembly/dirichlet_bc.hpp"
#include "solver/solver_factory.hpp"
#include "fe/element_transform.hpp"
#include "fe/grid_function.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool StructuralSolver::initialize(const Mesh& mesh) {
    mesh_ = &mesh;
    
    // 创建向量H1单元 (vdim=3)
    fec_ = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, fec_.get(), 3);  // 3D位移
    
    u_ = std::make_unique<GridFunction>(fes_.get());
    u_->setZero();
    
    matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    matAsm_->computeSparsityPattern();
    
    solver_ = SolverFactory::create(solverType_, maxIter_, tol_);
    
    LOG_INFO << "StructuralSolver: " << fes_->numDofs() << " DOFs";
    return true;
}

void StructuralSolver::assemble() {
    if (!fes_) return;
    
    if (!E_ || !nu_) {
        LOG_ERROR << "StructuralSolver: material not set";
        return;
    }
    
    matAsm_->clear();
    vecAsm_->clear();
    matAsm_->clearIntegrators();
    vecAsm_->clearIntegrators();
    
    // 添加弹性积分器（向量场）
    auto elasticity = std::make_unique<ElasticityIntegrator>(E_, nu_);
    matAsm_->addDomainIntegrator(std::move(elasticity));
    matAsm_->assemble();
    
    // 添加外部线性积分器（用于耦合载荷）
    for (auto& integrator : linearIntegrators_) {
        vecAsm_->addDomainIntegrator(std::move(integrator));
    }
    linearIntegrators_.clear();
    
    vecAsm_->assemble();
    
    // 应用位移边界条件
    applyDirichletBC(matAsm_->matrix(), vecAsm_->vector(), u_->values(),
                     *fes_, *mesh_, displacementBCs_, 3);
    
    // 应用分量边界条件
    if (!componentBCs_.empty()) {
        applyDirichletBCComponent(matAsm_->matrix(), vecAsm_->vector(), u_->values(),
                                  *fes_, *mesh_, componentBCs_, 3);
    }
    
    matAsm_->finalize();
}

bool StructuralSolver::solve() {
    if (!matAsm_ || !vecAsm_) return false;
    
    bool success = solver_->solve(matAsm_->matrix(), u_->values(), vecAsm_->vector());
    
    if (success) {
        iter_ = solver_->iterations();
        res_ = solver_->residual();
        LOG_INFO << "StructuralSolver: displacement norm = " << u_->values().norm();
    }
    
    return success;
}

void StructuralSolver::computeStressStrain() {
    // TODO: 实现应力/应变后处理
}

}  // namespace mpfem