/**
 * @file assembly.hpp
 * @brief Assembly module - unified header
 * 
 * This module provides the core assembly functionality for finite element
 * computations, bridging the gap between reference element operations
 * and global system assembly.
 * 
 * Key components:
 * - FEValues: Evaluates finite element quantities at quadrature points
 * - BilinearForm: Assembles stiffness matrices from weak forms
 * - LinearForm: Assembles right-hand side vectors
 * - DofMap: Utilities for local-to-global DoF mapping
 * 
 * Usage example (Poisson equation):
 * @code
 * // Setup
 * Mesh mesh = read_mesh("mesh.mphtxt");
 * FESpace fe_space(&mesh, "Lagrange1", 1);
 * DoFHandler dof_handler;
 * dof_handler.initialize(&fe_space);
 * dof_handler.distribute_dofs();
 * dof_handler.add_dirichlet_bc(1, 0.0);  // Boundary 1: u = 0
 * dof_handler.add_dirichlet_bc(2, 1.0);  // Boundary 2: u = 1
 * dof_handler.apply_boundary_conditions();
 * 
 * // Assembly
 * SparseMatrix K;
 * DynamicVector F;
 * 
 * BilinearForm bilinear(&dof_handler);
 * bilinear.assemble(BilinearForms::laplacian(1.0), K);
 * 
 * LinearForm linear(&dof_handler);
 * linear.assemble(LinearForms::source(0.0), F);  // No source
 * 
 * // Apply BCs
 * DofMap::apply_dirichlet_bc_simple(K, F, &dof_handler);
 * 
 * // Solve
 * DirectSolver solver;
 * DynamicVector u;
 * solver.solve(K, F, u);
 * @endcode
 */

#ifndef MPFEM_ASSEMBLY_ASSEMBLY_HPP
#define MPFEM_ASSEMBLY_ASSEMBLY_HPP

// Core assembly components
#include "fe_values.hpp"
#include "bilinear_form.hpp"
#include "linear_form.hpp"
#include "dof_map.hpp"

namespace mpfem {

/**
 * @brief Assembly namespace containing assembly utilities
 */
namespace assembly {
    // Re-export key types for convenience
    using FEValues = mpfem::FEValues;
    using BilinearForm = mpfem::BilinearForm;
    using LinearForm = mpfem::LinearForm;
    using DofMap = mpfem::DofMap;
    
    // Update flags
    using UpdateFlags = mpfem::UpdateFlags;
    constexpr UpdateFlags UpdateJxW = UpdateFlags::UpdateJxW;
    constexpr UpdateFlags UpdateGradients = UpdateFlags::UpdateGradients;
    constexpr UpdateFlags UpdateValues = UpdateFlags::UpdateValues;
    constexpr UpdateFlags UpdateQuadraturePoints = UpdateFlags::UpdateQuadraturePoints;
    constexpr UpdateFlags UpdateNormals = UpdateFlags::UpdateNormals;
    constexpr UpdateFlags UpdateDefault = UpdateFlags::UpdateDefault;
    constexpr UpdateFlags UpdateAll = UpdateFlags::UpdateAll;
    
    // Predefined assemblers
    using BilinearForms::laplacian;
    using BilinearForms::laplacian_anisotropic;
    using BilinearForms::mass;
    using BilinearForms::elasticity;
    using BilinearForms::convection_bc;
    
    using LinearForms::source;
    using LinearForms::source_function;
    using LinearForms::neumann_flux;
    using LinearForms::convection_rhs;
    using LinearForms::thermal_strain;
}

}  // namespace mpfem

#endif  // MPFEM_ASSEMBLY_ASSEMBLY_HPP
