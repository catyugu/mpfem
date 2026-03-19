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
#include <memory>

namespace mpfem {

/**
 * @brief Physics field solver base class
 * 
 * Contains common members and interfaces for all single-field solvers.
 * 
 * Key design changes:
 * - Solvers do NOT own GridFunction objects
 * - Solvers hold a reference to FieldValues
 * - FieldValues owns all field data and manages lifecycle
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
    
    /// Set solver configuration
    void setSolverConfig(const SolverConfig& config) { 
        solverConfig_ = config; 
    }
    
    int iterations() const { return iter_; }
    Real residual() const { return res_; }
    
protected:
    /// Create solver instance
    void createSolver() {
        solver_ = SolverFactory::create(solverConfig_);
    }
    
    // Configuration
    int order_ = 1;
    SolverConfig solverConfig_;
    int iter_ = 0;
    Real res_ = 0.0;
    
    // Common members
    const Mesh* mesh_ = nullptr;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<BilinearFormAssembler> matAsm_;
    std::unique_ptr<LinearFormAssembler> vecAsm_;
    std::unique_ptr<LinearSolver> solver_;
};

}  // namespace mpfem

#endif  // MPFEM_PHYSICS_FIELD_SOLVER_HPP
