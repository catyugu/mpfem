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

# Assembly module (to be implemented)
set(MPFEM_ASSEMBLY_SOURCES
)
set(MPFEM_ASSEMBLY_HEADERS
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

# Combine all sources and headers
set(MPFEM_ALL_SOURCES
    ${MPFEM_CORE_SOURCES}
    ${MPFEM_MESH_SOURCES}
    ${MPFEM_FEM_SOURCES}
    ${MPFEM_DOF_SOURCES}
    ${MPFEM_ASSEMBLY_SOURCES}
    ${MPFEM_LINALG_SOURCES}
)

set(MPFEM_ALL_HEADERS
    ${MPFEM_CORE_HEADERS}
    ${MPFEM_MESH_HEADERS}
    ${MPFEM_FEM_HEADERS}
    ${MPFEM_DOF_HEADERS}
    ${MPFEM_ASSEMBLY_HEADERS}
    ${MPFEM_LINALG_HEADERS}
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