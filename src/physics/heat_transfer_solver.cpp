#include "heat_transfer_solver.hpp"
#include "assembly/integrators.hpp"
#include "solver/solver_factory.hpp"
#include "core/logger.hpp"

namespace mpfem {

bool HeatTransferSolver::initialize(const Mesh& mesh, 
                                     const PWConstCoefficient& conductivity) {
    mesh_ = &mesh;
    
    fec_ = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, fec_.get());
    T_ = std::make_unique<GridFunction>(fes_.get());
    T_->values().setConstant(293.15);
    
    kInternal_ = conductivity;
    k_ = &kInternal_;
    
    matAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    vecAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    matAsm_->computeSparsityPattern();
    
    solver_ = SolverFactory::create(solverType_, maxIter_, tol_);
    
    LOG_INFO << "HeatTransferSolver: " << fes_->numDofs() << " DOFs";
    return true;
}

void HeatTransferSolver::assemble() {
    matAsm_->clear();
    vecAsm_->clear();
    matAsm_->clearIntegrators();
    vecAsm_->clearIntegrators();
    
    auto diff = std::make_unique<DiffusionIntegrator>(k_);
    matAsm_->addDomainIntegrator(std::move(diff));
    
    for (const auto& [bid, bc] : convBCs_) {
        auto conv = std::make_unique<ConvectionBoundaryIntegrator>(new ConstantCoefficient(bc.h));
        matAsm_->addBoundaryIntegrator(std::move(conv), bid);
        
        auto rhsInt = std::make_unique<BoundaryLFIntegrator>(new ConstantCoefficient(bc.h * bc.Tinf));
        vecAsm_->addBoundaryIntegrator(std::move(rhsInt), bid);
    }
    
    matAsm_->assemble();
    
    if (heatSource_) {
        auto src = std::make_unique<DomainLFIntegrator>(heatSource_);
        vecAsm_->addDomainIntegrator(std::move(src));
    }
    
    vecAsm_->assemble();
    applyBCs();
    matAsm_->finalize();
}

void HeatTransferSolver::applyBCs() {
    std::map<Index, Real> dofVals;
    
    for (const auto& [bid, val] : bcValues_) {
        for (Index b = 0; b < mesh_->numBdrElements(); ++b) {
            if (mesh_->bdrElement(b).attribute() == bid) {
                std::vector<Index> dofs;
                fes_->getBdrElementDofs(b, dofs);
                for (Index d : dofs) {
                    if (d != InvalidIndex && dofVals.find(d) == dofVals.end()) {
                        dofVals[d] = val;
                    }
                }
            }
        }
    }
    
    matAsm_->matrix().eliminateRows(dofVals, vecAsm_->vector());
    for (const auto& [d, v] : dofVals) T_->values()(d) = v;
}

bool HeatTransferSolver::solve() {
    if (!solver_) return false;
    bool ok = solver_->solve(matAsm_->matrix(), T_->values(), vecAsm_->vector());
    if (ok) {
        iter_ = solver_->iterations();
        res_ = solver_->residual();
        LOG_INFO << "HeatTransfer converged: iter=" << iter_ << " res=" << res_;
    }
    return ok;
}

}  // namespace mpfem
