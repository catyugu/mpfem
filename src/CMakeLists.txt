# ============================================================
# mpfem Library Source Files
# ============================================================

# Core module
set(MPFEM_CORE_SOURCES
    core/logger.cpp
)
set(MPFEM_CORE_HEADERS
    core/types.hpp
    core/logger.hpp
    core/exception.hpp
)

# Mesh module
set(MPFEM_MESH_SOURCES
    mesh/mesh.cpp
    mesh/mphtxt_reader.cpp
)
set(MPFEM_MESH_HEADERS
    mesh/mesh.hpp
    mesh/element.hpp
    mesh/connectivity.hpp
    mesh/mphtxt_reader.hpp
    mesh/geometry.hpp
)

# FEM module
set(MPFEM_FEM_SOURCES
    fem/fe_h1.cpp
    fem/fe_collection.cpp
    fem/quadrature.cpp
    fem/element_transformation.cpp
)
set(MPFEM_FEM_HEADERS
    fem/fe_base.hpp
    fem/fe_h1.hpp
    fem/fe_collection.hpp
    fem/quadrature.hpp
    fem/integration_rule.hpp
    fem/element_transformation.hpp
)

# DoF module
set(MPFEM_DOF_SOURCES
    dof/dof_handler.cpp
    dof/fe_space.cpp
)
set(MPFEM_DOF_HEADERS
    dof/dof_handler.hpp
    dof/dof_table.hpp
    dof/fe_space.hpp
)

# Assembly module
set(MPFEM_ASSEMBLY_SOURCES
    assembly/fe_values.cpp
    assembly/bilinear_form.cpp
    assembly/linear_form.cpp
    assembly/system_assembly.cpp
)
set(MPFEM_ASSEMBLY_HEADERS
    assembly/fe_values.hpp
    assembly/bilinear_form.hpp
    assembly/linear_form.hpp
    assembly/system_assembly.hpp
)

# Linalg module
set(MPFEM_LINALG_SOURCES
    linalg/direct_solver.cpp
    linalg/iterative_solver.cpp
)
set(MPFEM_LINALG_HEADERS
    linalg/solver_base.hpp
    linalg/direct_solver.hpp
    linalg/iterative_solver.hpp
    linalg/pardiso_solver.hpp
)

# Physics module
set(MPFEM_PHYSICS_SOURCES
    physics/coefficient.cpp
    physics/material.cpp
    physics/physics_base.cpp
    physics/electrostatics.cpp
    physics/heat_transfer.cpp
    physics/solid_mechanics.cpp
    physics/multiphysics.cpp
)
set(MPFEM_PHYSICS_HEADERS
    physics/coefficient.hpp
    physics/material.hpp
    physics/physics_base.hpp
    physics/electrostatics.hpp
    physics/heat_transfer.hpp
    physics/solid_mechanics.hpp
    physics/multiphysics.hpp
)

# IO module
set(MPFEM_IO_SOURCES
    io/variable_system.cpp
    io/xml_reader.cpp
    io/material_reader.cpp
    io/result_writer.cpp
)
set(MPFEM_IO_HEADERS
    io/variable_system.hpp
    io/xml_reader.hpp
    io/material_reader.hpp
    io/result_writer.hpp
)

# Combine all sources and headers
set(MPFEM_ALL_SOURCES
    ${MPFEM_CORE_SOURCES}
    ${MPFEM_MESH_SOURCES}
    ${MPFEM_FEM_SOURCES}
    ${MPFEM_DOF_SOURCES}
    ${MPFEM_ASSEMBLY_SOURCES}
    ${MPFEM_LINALG_SOURCES}
    ${MPFEM_PHYSICS_SOURCES}
    ${MPFEM_IO_SOURCES}
)

set(MPFEM_ALL_HEADERS
    ${MPFEM_CORE_HEADERS}
    ${MPFEM_MESH_HEADERS}
    ${MPFEM_FEM_HEADERS}
    ${MPFEM_DOF_HEADERS}
    ${MPFEM_ASSEMBLY_HEADERS}
    ${MPFEM_LINALG_HEADERS}
    ${MPFEM_PHYSICS_HEADERS}
    ${MPFEM_IO_HEADERS}
)

# ============================================================
# Library Definition
# ============================================================
add_library(mpfem STATIC
    ${MPFEM_ALL_SOURCES}
    ${MPFEM_ALL_HEADERS}
)

add_library(mpfem::mpfem ALIAS mpfem)

target_include_directories(mpfem
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/mpfem>
)

target_link_libraries(mpfem
    PUBLIC
        Eigen3::Eigen
        tinyxml2::tinyxml2
)

target_compile_features(mpfem PUBLIC cxx_std_20)

# Platform-specific settings
if(MSVC)
    target_compile_options(mpfem PRIVATE /W4 /WX-)
else()
    target_compile_options(mpfem PRIVATE -Wall -Wextra -Wpedantic)
endif()
