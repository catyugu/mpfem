#pragma once

// Config
#include "mpfem/config/problem_config_loader.hpp"

// Core
#include "mpfem/core/problem_definition.hpp"
#include "mpfem/core/types.hpp"
#include "mpfem/core/eigen_types.hpp"
#include "mpfem/core/logger.hpp"

// Mesh
#include "mpfem/mesh/element.hpp"
#include "mpfem/mesh/mesh.hpp"
#include "mpfem/mesh/boundary_topology.hpp"
#include "mpfem/mesh/comsol_mesh_reader.hpp"

// Material
#include "mpfem/material/material.hpp"

// FEM
#include "mpfem/fem/fe.hpp"
#include "mpfem/fem/fe_space.hpp"
#include "mpfem/fem/coefficient.hpp"
#include "mpfem/fem/grid_function.hpp"
#include "mpfem/fem/integrator.hpp"
#include "mpfem/fem/integrators.hpp"
#include "mpfem/fem/bilinear_form.hpp"

// Physics
#include "mpfem/physics/physics_model.hpp"
#include "mpfem/physics/coupled_physics_model.hpp"

// Solver
#include "mpfem/solver/linear_solver.hpp"
#include "mpfem/solver/physics_solver.hpp"
#include "mpfem/solver/coupled_solver.hpp"

// IO
#include "mpfem/io/vtk_writer.hpp"
