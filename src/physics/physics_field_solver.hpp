#ifndef MPFEM_PHYSICS_FIELD_SOLVER_HPP
#define MPFEM_PHYSICS_FIELD_SOLVER_HPP

#include "model/field_kind.hpp"
#include "model/case_definition.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include "fe/coefficient.hpp"
#include "mesh/mesh.hpp"
#include "core/logger.hpp"
#include <memory>
#include <string>

namespace mpfem {

/**
 * @file physics_field_solver.hpp
 * @brief Base class for physics field solvers.
 * 
 * Provides a common interface for different physics field solvers
 * (electrostatics, heat transfer, solid mechanics).
 */

/**
 * @brief Abstract base class for physics field solvers.
 */
class PhysicsFieldSolver {
public:
    virtual ~PhysicsFieldSolver() = default;
    
    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    
    /// Set polynomial order for the field
    virtual void setOrder(int order) { order_ = order; }
    
    /// Get polynomial order
    int order() const { return order_; }
    
    /// Get the field kind
    virtual FieldKind fieldKind() const = 0;
    
    /// Get the field name
    virtual std::string fieldName() const = 0;
    
    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------
    
    /**
     * @brief Initialize the solver with mesh and material properties.
     * @param mesh The computational mesh
     * @param conductivity Piecewise constant conductivity coefficient
     * @return true if initialization successful
     */
    virtual bool initialize(const Mesh& mesh, 
                           const PWConstCoefficient& conductivity) = 0;
    
    // -------------------------------------------------------------------------
    // Boundary conditions
    // -------------------------------------------------------------------------
    
    /**
     * @brief Add a Dirichlet boundary condition.
     * @param boundaryId Boundary attribute ID
     * @param value The prescribed value
     */
    virtual void addDirichletBC(int boundaryId, Real value) = 0;
    
    /**
     * @brief Add a Dirichlet boundary condition with coefficient.
     * @param boundaryId Boundary attribute ID
     * @param coef The prescribed value as a coefficient
     */
    virtual void addDirichletBC(int boundaryId, Coefficient* coef) = 0;
    
    /**
     * @brief Clear all boundary conditions.
     */
    virtual void clearBoundaryConditions() = 0;
    
    // -------------------------------------------------------------------------
    // Assembly and solve
    // -------------------------------------------------------------------------
    
    /**
     * @brief Assemble the system matrix and right-hand side.
     */
    virtual void assemble() = 0;
    
    /**
     * @brief Solve the linear system.
     * @return true if solved successfully
     */
    virtual bool solve() = 0;
    
    // -------------------------------------------------------------------------
    // Results access
    // -------------------------------------------------------------------------
    
    /// Get the field solution
    virtual const GridFunction& field() const = 0;
    
    /// Get the field solution (mutable)
    virtual GridFunction& field() = 0;
    
    /// Get the FE space
    virtual const FESpace& feSpace() const = 0;
    
    /// Get number of DOFs
    virtual Index numDofs() const = 0;
    
    /// Get min field value
    virtual Real minValue() const = 0;
    
    /// Get max field value
    virtual Real maxValue() const = 0;
    
    // -------------------------------------------------------------------------
    // Solver configuration
    // -------------------------------------------------------------------------
    
    /// Set solver type from string
    virtual void setSolverType(const std::string& type) {
        solverType_ = type;
    }
    
    /// Set solver max iterations
    virtual void setMaxIterations(int iter) {
        maxIterations_ = iter;
    }
    
    /// Set solver tolerance
    virtual void setTolerance(Real tol) {
        tolerance_ = tol;
    }
    
    /// Set solver print level
    virtual void setPrintLevel(int level) {
        printLevel_ = level;
    }
    
    /// Get solver iterations
    virtual int iterations() const { return iterations_; }
    
    /// Get solver residual
    virtual Real residual() const { return residual_; }
    
protected:
    int order_ = 1;
    std::string solverType_ = "sparse_lu";
    int maxIterations_ = 1000;
    Real tolerance_ = 1e-10;
    int printLevel_ = 0;
    int iterations_ = 0;
    Real residual_ = 0.0;
};

}  // namespace mpfem

#endif  // MPFEM_PHYSICS_FIELD_SOLVER_HPP
