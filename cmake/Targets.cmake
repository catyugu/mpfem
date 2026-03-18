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

# MKL support
if(MPFEM_MKL_FOUND)
    target_compile_definitions(mpfem_core PUBLIC MPFEM_USE_MKL)
    target_link_libraries(mpfem_core PUBLIC MKL::MKL)
    # Note: Not using EIGEN_USE_BLAS to avoid MKL BLAS configuration issues
    # MKL is only used for PARDISO solver
    message(STATUS "MKL enabled for PARDISO solver")
endif()

# OpenMP support
if(MPFEM_OPENMP_FOUND)
    target_link_libraries(mpfem_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# Model library (header-only)
# =============================================================================

mpfem_add_library(mpfem_model
    HEADER_ONLY
    PUBLIC_LINK
        Eigen3::Eigen
        mpfem_core
)

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
# IO library
# =============================================================================

mpfem_add_library(mpfem_io
    SOURCES
        src/io/value_parser.cpp
        src/io/case_xml_reader.cpp
        src/io/material_xml_reader.cpp
        src/io/result_exporter.cpp
        src/io/mphtxt_reader.cpp
    PUBLIC_LINK
        Eigen3::Eigen
        mpfem_core
        mpfem_model
        tinyxml2::tinyxml2
)

# =============================================================================
# FE library
# =============================================================================

mpfem_add_library(mpfem_fe
    SOURCES
        src/fe/quadrature.cpp
        src/fe/element_transform.cpp
        src/fe/facet_element_transform.cpp
        src/fe/grid_function.cpp
        src/fe/coefficient.cpp
        src/fe/shape_function.cpp
    PUBLIC_LINK
        Eigen3::Eigen
        mpfem_core
        mpfem_mesh
        mpfem_model
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
# Coupling library (header-only)
# =============================================================================

mpfem_add_library(mpfem_coupling
    HEADER_ONLY
    PUBLIC_LINK
        Eigen3::Eigen
        mpfem_core
        mpfem_fe
)

# =============================================================================
# Physics library
# =============================================================================

mpfem_add_library(mpfem_physics
    SOURCES
        src/physics/electrostatics_solver.cpp
        src/physics/heat_transfer_solver.cpp
        src/physics/structural_solver.cpp
        src/physics/coupling_manager.cpp
        src/physics/physics_problem_builder.cpp
    PUBLIC_LINK
        Eigen3::Eigen
        mpfem_core
        mpfem_mesh
        mpfem_fe
        mpfem_assembly
        mpfem_solver
        mpfem_coupling
        mpfem_io
        mpfem_model
)

# =============================================================================
# Create simple aliases (mpfem::core, mpfem::mesh, etc.)
# =============================================================================

mpfem_create_alias(core)
mpfem_create_alias(model)
mpfem_create_alias(mesh)
mpfem_create_alias(io)
mpfem_create_alias(fe)
mpfem_create_alias(assembly)
mpfem_create_alias(solver)
mpfem_create_alias(coupling)
mpfem_create_alias(physics)