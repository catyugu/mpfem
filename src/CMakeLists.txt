set(MPFEM_SOURCES
  mesh/comsol_mesh_reader.cpp
  mesh/boundary_topology.cpp
  material/comsol_material_reader.cpp
  config/problem_config_loader.cpp
  core/logger.cpp
  fem/fe.cpp
  fem/fe_space.cpp
  fem/coefficient.cpp
  fem/grid_function.cpp
  fem/integrators.cpp
  fem/bilinear_form.cpp
  solver/linear_solver.cpp
  solver/physics_solver.cpp
  solver/coupled_solver.cpp
  io/vtk_writer.cpp
  ../external/tinyxml2/tinyxml2.cpp
)

add_library(mpfem ${MPFEM_SOURCES})
add_library(mpfem::mpfem ALIAS mpfem)

target_include_directories(mpfem
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/tinyxml2
)

# Link Eigen (header-only)
target_link_libraries(mpfem PUBLIC Eigen3::Eigen)

mpfem_set_default_compiler_options(mpfem)
