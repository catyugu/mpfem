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

class PhysicsFieldSolver {
public:
    virtual ~PhysicsFieldSolver() = default;
    
    virtual FieldKind fieldKind() const = 0;
    virtual std::string fieldName() const = 0;
    virtual FieldId fieldId() const = 0;
    virtual void assemble() = 0;
    
    bool solve() {
        if (!solver_ || !matAsm_ || !vecAsm_) return false;
        bool ok = solver_->solve(matAsm_->matrix(), field().values(), vecAsm_->vector());
        if (ok) {
            LOG_INFO << fieldName() << " solver converged!";
        }
        return ok;
    }
    
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
    
    int order_ = 1;
    SolverConfig solverConfig_;
    int iter_ = 0;
    Real res_ = 0.0;
    
    const Mesh* mesh_ = nullptr;
    FieldValues* fieldValues_ = nullptr;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<BilinearFormAssembler> matAsm_;
    std::unique_ptr<LinearFormAssembler> vecAsm_;
    std::unique_ptr<LinearSolver> solver_;
};

}  // namespace mpfem

#endif  // MPFEM_PHYSICS_FIELD_SOLVER_HPP
