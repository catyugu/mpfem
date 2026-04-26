# =============================================================================
# Targets.cmake - Library target definitions (Optimized for Build Speed)
# =============================================================================

# =============================================================================
# Helper function for creating mpfem library targets
# =============================================================================

function(mpfem_add_library name)
    set(options HEADER_ONLY)
    set(oneValueArgs)
    set(multiValueArgs SOURCES DEPENDS PUBLIC_LINK PRIVATE_LINK)

    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(ARG_HEADER_ONLY)
        add_library(${name} INTERFACE)
        target_include_directories(${name}
            INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:include>
        )

        if(ARG_PUBLIC_LINK)
            target_link_libraries(${name} INTERFACE ${ARG_PUBLIC_LINK})
        endif()

        if(ARG_PRIVATE_LINK)
            message(WARNING "PRIVATE_LINK ignored for INTERFACE library ${name}")
        endif()
    else()
        add_library(${name} ${ARG_SOURCES})
        target_include_directories(${name}
            PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:include>
        )

        if(ARG_PUBLIC_LINK)
            target_link_libraries(${name} PUBLIC ${ARG_PUBLIC_LINK})
        endif()

        if(ARG_PRIVATE_LINK)
            target_link_libraries(${name} PRIVATE ${ARG_PRIVATE_LINK})
        endif()
    endif()

    if(ARG_DEPENDS)
        add_dependencies(${name} ${ARG_DEPENDS})
    endif()
endfunction()

macro(mpfem_create_alias name)
    add_library(mpfem::${name} ALIAS mpfem_${name})
endmacro()

# =============================================================================
# 1. Core library (Bottom of the dependency tree)
# =============================================================================
mpfem_add_library(mpfem_core
    SOURCES
    src/core/logger.cpp
    PUBLIC_LINK
    Eigen3::Eigen
)

if(MPFEM_MKL_FOUND)
    target_compile_definitions(mpfem_core PUBLIC MPFEM_USE_MKL)
    target_link_libraries(mpfem_core PUBLIC MKL::MKL)
    message(STATUS "MKL enabled for PARDISO solver")
endif()

if(MPFEM_UMFPACK_FOUND)
    target_compile_definitions(mpfem_core PUBLIC MPFEM_USE_UMFPACK)

    if(TARGET SuiteSparse::UMFPACK)
        target_link_libraries(mpfem_core PUBLIC SuiteSparse::UMFPACK)
    else()
        target_link_libraries(mpfem_core PUBLIC ${UMFPACK_LIBRARIES})
    endif()

    message(STATUS "SuiteSparse::UMFPACK solver enabled")
endif()

if(MPFEM_OPENMP_FOUND)
    target_link_libraries(mpfem_core PUBLIC OpenMP::OpenMP_CXX)
endif()

target_precompile_headers(mpfem_core PUBLIC
    <vector>
    <memory>
    <string>
    <Eigen/Core>
    <Eigen/SparseCore>
)

# =============================================================================
# 2. Base Modules (Depend only on Core)
# =============================================================================
mpfem_add_library(mpfem_mesh
    SOURCES
    src/mesh/mesh.cpp
    PUBLIC_LINK
    mpfem_core
)

mpfem_add_library(mpfem_expr
    SOURCES
    src/expr/unit_parser.cpp
    src/expr/expression_parser.cpp
    src/expr/variable_graph.cpp
    PUBLIC_LINK
    mpfem_core
)

mpfem_add_library(mpfem_solver
    SOURCES
    src/solver/linear_operator.cpp
    src/solver/eigen_solver.cpp
    src/solver/pardiso_solver.cpp
    src/solver/umfpack_solver.cpp
    src/solver/solver_factory.cpp
    src/solver/solver_config.cpp
    PUBLIC_LINK
    mpfem_core
)

# =============================================================================
# 3. Intermediate Modules
# =============================================================================
mpfem_add_library(mpfem_io
    SOURCES
    src/io/case_xml_reader.cpp
    src/io/material_xml_reader.cpp
    src/io/material_database.cpp
    src/io/problem_input_loader.cpp
    src/io/result_exporter.cpp
    src/io/mphtxt_reader.cpp
    PUBLIC_LINK
    mpfem_expr
    tinyxml2::tinyxml2
    PRIVATE_LINK
    mpfem_mesh # IO needs mesh internally to read/write formats
)

mpfem_add_library(mpfem_fe
    SOURCES
    src/fe/quadrature.cpp
    src/fe/element_transform.cpp
    src/fe/finite_element.cpp
    src/fe/geometry_mapping.cpp
    src/fe/h1.cpp
    PUBLIC_LINK
    mpfem_mesh
    PRIVATE_LINK
    basix
)

mpfem_add_library(mpfem_field
    SOURCES
    src/field/fe_space.cpp
    src/field/grid_function.cpp
    PUBLIC_LINK
    mpfem_fe
    mpfem_mesh
)

mpfem_add_library(mpfem_assembly
    SOURCES
    src/assembly/assembler.cpp
    src/assembly/integrators.cpp
    PUBLIC_LINK
    mpfem_field
    mpfem_expr
)

# =============================================================================
# 4. High-Level Modules (Physics & Problem)
# =============================================================================
mpfem_add_library(mpfem_physics
    SOURCES
    src/physics/electrostatics_solver.cpp
    src/physics/heat_transfer_solver.cpp
    src/physics/structural_solver.cpp
    PUBLIC_LINK
    mpfem_assembly
    mpfem_solver
    mpfem_expr
)

mpfem_add_library(mpfem_problem
    SOURCES
    src/problem/problem.cpp
    src/problem/transient_problem.cpp
    src/problem/physics_problem_builder.cpp
    src/problem/time/time_integrator.cpp
    src/problem/time/bdf1_integrator.cpp
    src/problem/time/bdf2_integrator.cpp
    PUBLIC_LINK
    mpfem_physics
    PRIVATE_LINK
    mpfem_io
    mpfem_expr
)

# =============================================================================
# Create simple aliases
# =============================================================================
mpfem_create_alias(core)
mpfem_create_alias(expr)
mpfem_create_alias(mesh)
mpfem_create_alias(io)
mpfem_create_alias(fe)
mpfem_create_alias(field)
mpfem_create_alias(assembly)
mpfem_create_alias(solver)
mpfem_create_alias(physics)
mpfem_create_alias(problem)