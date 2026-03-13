#include "physics/heat_transfer_solver.hpp"
#include "core/exception.hpp"
#include <cmath>

namespace mpfem {

// =============================================================================
// HeatTransferSolver Implementation
// =============================================================================

bool HeatTransferSolver::initialize(const Mesh& mesh, 
                                     const PWConstCoefficient& thermalConductivity) {
    mesh_ = &mesh;
    
    // Create FE collection and space
    fec_ = std::make_unique<FECollection>(order_, FECollection::Type::H1);
    fes_ = std::make_unique<FESpace>(&mesh, fec_.get());
    
    // Create solution field
    T_ = std::make_unique<GridFunction>(fes_.get());
    T_->setZero();
    
    // Set initial temperature to ambient (293.15 K = 20°C)
    // This is important for temperature-dependent material properties
    T_->values().setConstant(293.15);
    
    // Store thermal conductivity (copy)
    thermalConductivity_ = std::make_shared<PWConstCoefficient>(thermalConductivity);
    
    // Create assemblers
    bilinearAsm_ = std::make_unique<BilinearFormAssembler>(fes_.get());
    linearAsm_ = std::make_unique<LinearFormAssembler>(fes_.get());
    
    // Create linear solver
    solver_ = SolverFactory::create(solverType_, maxIterations_, tolerance_, printLevel_);
    
    LOG_INFO << "HeatTransferSolver initialized: " << fes_->numDofs() << " DOFs, initial T = 293.15 K";
    
    return true;
}

void HeatTransferSolver::setJouleHeating(const GridFunction* V, 
                                          const Coefficient* sigma) {
    auto jouleHeat = std::make_shared<JouleHeatCoefficient>(V, sigma);
    heatSource_ = jouleHeat;
}

void HeatTransferSolver::assemble() {
    if (!fes_ || !mesh_) {
        MPFEM_THROW(Exception, "HeatTransferSolver not initialized");
    }
    
    LOG_INFO << "HeatTransfer::assemble() - clearing previous assembly...";
    
    // Clear previous assembly
    bilinearAsm_->clear();
    linearAsm_->clear();
    
    LOG_INFO << "HeatTransfer::assemble() - adding diffusion integrator...";
    
    // 1. Add diffusion integrator with thermal conductivity coefficient
    auto diffInteg = std::make_unique<DiffusionIntegrator>(thermalConductivity_);
    bilinearAsm_->addDomainIntegrator(std::move(diffInteg));
    
    LOG_INFO << "HeatTransfer::assemble() - adding convection BCs...";
    
    // 2. Add convection boundary integrators
    for (const auto& [boundaryId, bc] : convectionBCs_) {
        LOG_DEBUG << "Adding convection BC for boundary " << boundaryId;
        auto convInteg = std::make_unique<ConvectionBoundaryIntegrator>(bc.h, bc.Tinf);
        bilinearAsm_->addBoundaryIntegrator(std::move(convInteg), boundaryId);
        
        // Also add the RHS contribution (h * Tinf)
        auto rhsCoef = std::make_shared<ProductCoefficient>(bc.h, bc.Tinf);
        auto rhsInteg = std::make_unique<BoundaryLFIntegrator>(rhsCoef);
        linearAsm_->addBoundaryIntegrator(std::move(rhsInteg), boundaryId);
    }
    
    // Assemble stiffness matrix and boundary contributions
    LOG_INFO << "HeatTransfer: assembling bilinear form...";
    bilinearAsm_->assemble();
    LOG_INFO << "HeatTransfer: matrix assembled, size=" << bilinearAsm_->rows();
    
    LOG_INFO << "HeatTransfer: adding heat source integrator...";
    
    // 3. Assemble heat source (if any)
    if (heatSource_) {
        auto sourceInteg = std::make_unique<DomainLFIntegrator>(heatSource_);
        linearAsm_->addDomainIntegrator(std::move(sourceInteg));
    }
    
    LOG_INFO << "HeatTransfer: assembling linear form...";
    
    // Assemble RHS
    linearAsm_->assemble();
    
    LOG_INFO << "RHS norm: " << linearAsm_->vector().norm();
    
    LOG_INFO << "HeatTransfer: applying Dirichlet BCs...";
    
    // 4. Apply Dirichlet boundary conditions
    applyDirichletBCs();
    
    LOG_INFO << "HeatTransfer: finalizing matrix...";
    
    // Finalize matrix
    bilinearAsm_->finalize();
    
    LOG_INFO << "HeatTransferSolver assembled: matrix " 
              << bilinearAsm_->rows() << "x" << bilinearAsm_->cols();
}

void HeatTransferSolver::applyDirichletBCs() {
    if (dirichletBCs_.empty()) return;
    
    // Collect all constrained DOFs and their values
    std::map<Index, Real> dofValues;
    
    for (const auto& [boundaryId, coef] : dirichletBCs_) {
        // Find boundary elements with this attribute
        for (Index b = 0; b < mesh_->numBdrElements(); ++b) {
            const Element& bdrElem = mesh_->bdrElement(b);
            if (static_cast<int>(bdrElem.attribute()) == boundaryId) {
                // Get DOFs on this boundary element
                std::vector<Index> dofs;
                fes_->getBdrElementDofs(b, dofs);
                
                // Get boundary element center for coefficient evaluation
                FacetElementTransform bdrTrans(mesh_, b);
                Real xi[3] = {0.0, 0.0, 0.0};
                bdrTrans.setIntegrationPoint(xi);
                
                Real value = coef->eval(bdrTrans);
                
                for (Index dof : dofs) {
                    if (dof != InvalidIndex && dofValues.find(dof) == dofValues.end()) {
                        dofValues[dof] = value;
                    }
                }
            }
        }
    }
    
    // Apply elimination
    bilinearAsm_->matrix().eliminateRows(dofValues, linearAsm_->vector());
    
    // Set solution values for constrained DOFs
    for (const auto& [dof, value] : dofValues) {
        T_->values()(dof) = value;
    }
    
    LOG_DEBUG << "Applied " << dofValues.size() << " Dirichlet BCs for temperature";
}

bool HeatTransferSolver::solve() {
    if (!solver_) {
        LOG_ERROR << "HeatTransferSolver: solver not configured";
        return false;
    }
    
    // Solve the system
    bool success = solver_->solve(bilinearAsm_->matrix(), 
                                  T_->values(), 
                                  linearAsm_->vector());
    
    if (success) {
        iterations_ = solver_->iterations();
        residual_ = solver_->residual();
        LOG_INFO << "HeatTransferSolver converged in " << iterations_ 
                 << " iterations, residual = " << residual_;
    } else {
        LOG_ERROR << "HeatTransferSolver failed to converge";
    }
    
    return success;
}

void HeatTransferSolver::updateThermalConductivity() {
    // For temperature-dependent thermal conductivity
    // This would update the coefficient based on the temperature field
    // For now, thermal conductivity is assumed constant
}

// =============================================================================
// JouleHeatCoefficient Implementation
// =============================================================================

Real JouleHeatCoefficient::eval(ElementTransform& trans) const {
    if (!V_ || !sigma_) return 0.0;
    
    Index elemIdx = trans.elementIndex();
    
    // Get integration point coordinates in reference element
    const IntegrationPoint& ip = trans.integrationPoint();
    Real xi[3] = {ip.xi, ip.eta, ip.zeta};
    
    // Compute gradient of V at integration point
    Vector3 gradV = V_->gradient(elemIdx, xi, trans);
    
    // Get conductivity at this point
    Real sigma = sigma_->eval(trans);
    
    // Q = σ|∇V|² (Joule heating)
    Real gradMag2 = gradV.norm() * gradV.norm();
    Real Q = sigma * gradMag2;
    
    return Q;
}

}  // namespace mpfem