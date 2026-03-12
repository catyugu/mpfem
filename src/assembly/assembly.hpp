#ifndef MPFEM_ASSEMBLY_HPP
#define MPFEM_ASSEMBLY_HPP

/**
 * @file assembly.hpp
 * @brief Assembly module for finite element matrices and vectors.
 * 
 * This module provides:
 * - Integrators: Compute element-level matrices and vectors
 * - Assembler: Assembles global system from element contributions
 * - Boundary condition handlers: Apply Dirichlet and Neumann BCs
 */

#include "assembly/integrator.hpp"
#include "assembly/integrators.hpp"
#include "assembly/assembler.hpp"

#endif  // MPFEM_ASSEMBLY_HPP
