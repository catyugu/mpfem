# =============================================================================
# Targets.cmake - Library target definitions
# =============================================================================

include(CompilerOptions)

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
        # For INTERFACE libraries, use INTERFACE keyword for link libraries
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
        mpfem_set_default_compiler_options(${name})
        # For regular libraries, use PUBLIC/PRIVATE keywords
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

# =============================================================================
# Create simple aliases for all libraries
# =============================================================================
macro(mpfem_create_alias name)
    add_library(mpfem::${name} ALIAS mpfem_${name})
endmacro()

# =============================================================================
# Core library
# =============================================================================

mpfem_add_library(mpfem_core
    SOURCES
        src/core/logger.cpp
    PUBLIC_LINK
        Eigen3::Eigen
)

# --- MKL support (PARDISO solver + optional BLAS acceleration) ---
if(MPFEM_MKL_FOUND)
    target_compile_definitions(mpfem_core PUBLIC MPFEM_USE_MKL)
    target_link_libraries(mpfem_core PUBLIC MKL::MKL)
    message(STATUS "MKL enabled for PARDISO solver")
endif()

# --- OpenBLAS support (BLAS acceleration for Eigen) ---
if(MPFEM_OPENBLAS_FOUND)
    target_compile_definitions(mpfem_core PUBLIC MPFEM_USE_OPENBLAS)
    target_compile_definitions(mpfem_core PUBLIC EIGEN_USE_BLAS)
    if(TARGET OpenBLAS::OpenBLAS)
        target_link_libraries(mpfem_core PUBLIC OpenBLAS::OpenBLAS)
    elseif(TARGET PkgConfig::OpenBLAS)
        target_link_libraries(mpfem_core PUBLIC PkgConfig::OpenBLAS)
    else()
        target_include_directories(mpfem_core PUBLIC ${OpenBLAS_INCLUDE_DIRS})
        target_link_libraries(mpfem_core PUBLIC ${OpenBLAS_LIBRARIES})
    endif()
    message(STATUS "OpenBLAS acceleration enabled for Eigen")
endif()

# --- SuiteSparse support (UMFPACK solver) ---
if(MPFEM_SUITESPARSE_FOUND)
    target_compile_definitions(mpfem_core PUBLIC MPFEM_USE_SUITESPARSE)
    if(TARGET SuiteSparse::UMFPACK)
        target_link_libraries(mpfem_core PUBLIC SuiteSparse::UMFPACK)
    elseif(TARGET UMFPACK::UMFPACK)
        target_link_libraries(mpfem_core PUBLIC UMFPACK::UMFPACK)
    else()
        target_link_libraries(mpfem_core PUBLIC ${UMFPACK_LIBRARIES})
    endif()
    message(STATUS "SuiteSparse/UMFPACK solver enabled")
endif()

# --- OpenMP support ---
if(MPFEM_OPENMP_FOUND)
    target_link_libraries(mpfem_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# Mesh library
# =============================================================================

mpfem_add_library(mpfem_mesh
    SOURCES
        src/mesh/mesh.cpp
    PUBLIC_LINK
        Eigen3::Eigen
        mpfem_core
)

# =============================================================================
# Expression runtime library
# =============================================================================

mpfem_add_library(mpfem_expr
    SOURCES
        src/expr/unit_parser.cpp
        src/expr/expression_parser.cpp
        src/expr/variable_graph.cpp
    PUBLIC_LINK
        Eigen3::Eigen
        mpfem_core
)

# =============================================================================
# IO library
# =============================================================================

mpfem_add_library(mpfem_io
    SOURCES
        src/io/case_xml_reader.cpp
        src/io/material_xml_reader.cpp
        src/model/material_database.cpp
        src/io/problem_input_loader.cpp
        src/io/result_exporter.cpp
        src/io/mphtxt_reader.cpp
    PUBLIC_LINK
        Eigen3::Eigen
        mpfem_core
        mpfem_expr
        tinyxml2::tinyxml2
)

# =============================================================================
# FE library
# =============================================================================

mpfem_add_library(mpfem_fe
    SOURCES
        src/fe/quadrature.cpp
        src/fe/element_transform.cpp
        src/fe/finite_element.cpp
        src/fe/geometry_mapping.cpp
        src/fe/h1.cpp
        src/fe/grid_function.cpp
        src/fe/fe_space.cpp
    PUBLIC_LINK
        Eigen3::Eigen
        mpfem_core
        mpfem_mesh
)

# =============================================================================
# Solver library (header-only)
# =============================================================================

mpfem_add_library(mpfem_solver
    HEADER_ONLY
    PUBLIC_LINK
        Eigen3::Eigen
        mpfem_core
)

# =============================================================================
# Assembly library
# =============================================================================

mpfem_add_library(mpfem_assembly
    SOURCES
        src/assembly/assembler.cpp
        src/assembly/integrators.cpp
    PUBLIC_LINK
        Eigen3::Eigen
        mpfem_core
        mpfem_mesh
        mpfem_fe
        mpfem_solver
)

# =============================================================================
# Problem library (header-only) - 纯数据基类
# =============================================================================

mpfem_add_library(mpfem_problem
    SOURCES
        src/problem/problem.cpp
        src/problem/physics_problem_builder.cpp
    PUBLIC_LINK
        Eigen3::Eigen
        mpfem_core
        mpfem_expr
        mpfem_mesh
        mpfem_fe
        mpfem_io
        mpfem_assembly
        mpfem_physics
)

# =============================================================================
# Physics library
# =============================================================================

mpfem_add_library(mpfem_physics
    SOURCES
        src/physics/electrostatics_solver.cpp
        src/physics/heat_transfer_solver.cpp
        src/physics/structural_solver.cpp
        src/problem/transient_problem.cpp
        src/time/time_integrator.cpp
        src/time/bdf1_integrator.cpp
        src/time/bdf2_integrator.cpp
    PUBLIC_LINK
        Eigen3::Eigen
        mpfem_core
        mpfem_mesh
        mpfem_fe
        mpfem_assembly
        mpfem_solver
        mpfem_io
)

# =============================================================================
# Create simple aliases (mpfem::core, mpfem::mesh, etc.)
# =============================================================================

mpfem_create_alias(core)
mpfem_create_alias(expr)
mpfem_create_alias(mesh)
mpfem_create_alias(io)
mpfem_create_alias(fe)
mpfem_create_alias(assembly)
mpfem_create_alias(solver)
mpfem_create_alias(problem)
mpfem_create_alias(physics)
