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
#include "physics/field_values.hpp"
#include "core/logger.hpp"
#include <memory>

namespace mpfem {

/**
 * @brief Physics field solver base class
 * 
 * Unified design:
 * - fieldValues_ is owned by base class (shared by all solvers)
 * - field() is implemented in base class via fieldId()
 * - solve() has common logic in base class
 * - assemble() remains pure virtual (solver-specific)
 */
class PhysicsFieldSolver {
public:
    virtual ~PhysicsFieldSolver() = default;
    
    /// Get field kind
    virtual FieldKind fieldKind() const = 0;
    
    /// Get field name for logging
    virtual std::string fieldName() const = 0;
    
    /// Get field ID for FieldValues lookup
    virtual FieldId fieldId() const = 0;
    
    /// Assemble system matrix and vector
    virtual void assemble() = 0;
    
    /// Solve the linear system
    bool solve() {
        if (!solver_ || !matAsm_ || !vecAsm_) return false;
        bool ok = solver_->solve(matAsm_->matrix(), field().values(), vecAsm_->vector());
        if (ok) {
            LOG_INFO << fieldName() << " solver converged!";
        }
        return ok;
    }
    
    /// Get field (implemented in base class)
    const GridFunction& field() const { return fieldValues_->current(fieldId()); }
    GridFunction& field() { return fieldValues_->current(fieldId()); }
    
    const FESpace& feSpace() const { return *fes_; }
    Index numDofs() const { return fes_ ? fes_->numDofs() : 0; }
    const Mesh& mesh() const { return *mesh_; }
    
    void setOrder(int o) { order_ = o; }
    void setSolverConfig(const SolverConfig& config) { solverConfig_ = config; }
    
    int iterations() const { return iter_; }
    Real residual() const { return res_; }

protected:
    void createSolver() { solver_ = SolverFactory::create(solverConfig_); }
    
    void clearAssemblers() {
        if (matAsm_) { matAsm_->clear(); matAsm_->clearIntegrators(); }
        if (vecAsm_) { vecAsm_->clear(); vecAsm_->clearIntegrators(); }
    }
    
    // Configuration
    int order_ = 1;
    SolverConfig solverConfig_;
    int iter_ = 0;
    Real res_ = 0.0;
    
    // Common members
    const Mesh* mesh_ = nullptr;
    FieldValues* fieldValues_ = nullptr;  ///< Non-owning reference (owned by Problem)
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<BilinearFormAssembler> matAsm_;
    std::unique_ptr<LinearFormAssembler> vecAsm_;
    std::unique_ptr<LinearSolver> solver_;
};

}  // namespace mpfem

#endif  // MPFEM_PHYSICS_FIELD_SOLVER_HPP