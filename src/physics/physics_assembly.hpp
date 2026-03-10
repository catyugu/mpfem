/**
 * @file physics_assembly.hpp
 * @brief Base class for physics assembly
 */

#ifndef MPFEM_PHYSICS_PHYSICS_ASSEMBLY_HPP
#define MPFEM_PHYSICS_PHYSICS_ASSEMBLY_HPP

#include "core/types.hpp"
#include "mesh/mesh.hpp"
#include "dof/dof_handler.hpp"
#include "material/material_database.hpp"
#include "linalg/solver_base.hpp"
#include <memory>
#include <functional>

namespace mpfem {

/**
 * @brief Result of physics solve
 */
struct PhysicsResult {
    SolverStatus status;
    int iterations;
    Scalar residual;
    DynamicVector solution;
    
    bool success() const { return status == SolverStatus::Success; }
};

/**
 * @brief Base class for single physics field assembly and solve
 */
class PhysicsAssembly {
public:
    virtual ~PhysicsAssembly() = default;
    
    /**
     * @brief Initialize the physics
     * @param mesh The mesh
     * @param dof_handler DoF handler for this field
     * @param mat_db Material database
     */
    virtual void initialize(const Mesh* mesh,
                           const DoFHandler* dof_handler,
                           const MaterialDB* mat_db) {
        mesh_ = mesh;
        dof_handler_ = dof_handler;
        mat_db_ = mat_db;
    }
    
    /**
     * @brief Assemble the system matrix (stiffness matrix)
     * @param K Output sparse matrix
     */
    virtual void assemble_stiffness(SparseMatrix& K) = 0;
    
    /**
     * @brief Assemble the right-hand side vector
     * @param f Output vector
     */
    virtual void assemble_rhs(DynamicVector& f) = 0;
    
    /**
     * @brief Apply Dirichlet boundary conditions
     * @param K System matrix (modified in place)
     * @param f RHS vector (modified in place)
     */
    virtual void apply_boundary_conditions(SparseMatrix& K, DynamicVector& f) = 0;
    
    /**
     * @brief Solve the linear system
     * @param solver Solver to use
     * @return Physics result with solution
     */
    virtual PhysicsResult solve(SolverBase& solver) {
        PhysicsResult result;
        
        // Assemble
        SparseMatrix K;
        DynamicVector f;
        assemble_stiffness(K);
        assemble_rhs(f);
        apply_boundary_conditions(K, f);
        
        // Solve
        result.solution.resize(dof_handler_->n_dofs());
        result.status = solver.solve(K, f, result.solution);
        result.iterations = solver.iterations();
        result.residual = solver.residual();
        
        return result;
    }
    
    /**
     * @brief Get the field name
     */
    virtual std::string field_name() const = 0;
    
    /**
     * @brief Get number of components
     */
    virtual int n_components() const { return 1; }
    
    /**
     * @brief Set external field value (for coupling)
     * @param field_name Name of the external field
     * @param values Field values at nodes
     */
    virtual void set_external_field(const std::string& field_name,
                                   const DynamicVector& values) {
        // Default: do nothing
        (void)field_name;
        (void)values;
    }
    
protected:
    const Mesh* mesh_ = nullptr;
    const DoFHandler* dof_handler_ = nullptr;
    const MaterialDB* mat_db_ = nullptr;
};

} // namespace mpfem

#endif // MPFEM_PHYSICS_PHYSICS_ASSEMBLY_HPP
