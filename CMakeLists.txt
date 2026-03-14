cmake_minimum_required(VERSION 3.20)

project(mpfem
    VERSION 0.1.0
    DESCRIPTION "Multi-Physics Finite Element Method Library"
    LANGUAGES CXX
)

# =============================================================================
# Options
# =============================================================================
option(MPFEM_BUILD_TESTS "Build unit tests" ON)
option(MPFEM_BUILD_EXAMPLES "Build examples" ON)
option(MPFEM_USE_OPENMP "Use OpenMP for parallelization" ON)
option(MPFEM_USE_MKL "Use Intel MKL for PARDISO solver" OFF)
option(MPFEM_USE_UMFPACK "Use UMFPACK solver" OFF)

# =============================================================================
# C++ Standard
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# CMake modules
# =============================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerOptions)
include(CPM)

# =============================================================================
# Dependencies
# =============================================================================

# Eigen3 (required)
find_package(Eigen3 REQUIRED)

# OpenMP (optional)
if(MPFEM_USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# MKL PARDISO (optional)
if(MPFEM_USE_MKL)
    find_package(MKL COMPONENTS PARDISO REQUIRED)
endif()

# UMFPACK (optional)
if(MPFEM_USE_UMFPACK)
    find_package(SuiteSparse COMPONENTS UMFPACK REQUIRED)
endif()

# tinyxml2 (required for XML parsing)
CPMAddPackage(
    NAME tinyxml2
    GITHUB_REPOSITORY leethomason/tinyxml2
    GIT_TAG 11.0.0
    OPTIONS
        "BUILD_TESTING OFF"
)

# GoogleTest (optional, for testing)
if(MPFEM_BUILD_TESTS)
    enable_testing()
    CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        GIT_TAG v1.14.0
        OPTIONS
            "BUILD_GMOCK OFF"
            "INSTALL_GTEST OFF"
    )
endif()

# =============================================================================
# Library targets
# =============================================================================

# Main library
add_library(mpfem_core
    src/core/logger.cpp
)

add_library(mpfem_model INTERFACE)

add_library(mpfem_io
    src/io/value_parser.cpp
    src/io/case_xml_reader.cpp
    src/io/material_xml_reader.cpp
    src/io/result_exporter.cpp
)

add_library(mpfem_fe
    src/fe/quadrature.cpp
    src/fe/element_transform.cpp
    src/fe/facet_element_transform.cpp
    src/fe/grid_function.cpp
    src/fe/coefficient.cpp
    src/fe/shape_function.cpp
)

add_library(mpfem_mesh
    src/mesh/mesh.cpp
    src/mesh/io/mphtxt_reader.cpp
)

# Assembly module
add_library(mpfem_assembly
    src/assembly/assembler.cpp
    src/assembly/integrators.cpp
)

# Solver module (header-only for now)
add_library(mpfem_solver INTERFACE)

# Coupling module (header-only)
add_library(mpfem_coupling INTERFACE)

# Physics module
add_library(mpfem_physics
    src/physics/electrostatics_solver.cpp
    src/physics/heat_transfer_solver.cpp
    src/physics/structural_solver.cpp
    src/physics/coupling_manager.cpp
)

# Library aliases for convenience
add_library(mpfem::core ALIAS mpfem_core)
add_library(mpfem::model ALIAS mpfem_model)
add_library(mpfem::io ALIAS mpfem_io)
add_library(mpfem::fe ALIAS mpfem_fe)
add_library(mpfem::mesh ALIAS mpfem_mesh)
add_library(mpfem::assembly ALIAS mpfem_assembly)
add_library(mpfem::solver ALIAS mpfem_solver)
add_library(mpfem::coupling ALIAS mpfem_coupling)
add_library(mpfem::physics ALIAS mpfem_physics)

# =============================================================================
# Target properties
# =============================================================================

# mpfem_core
target_include_directories(mpfem_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(mpfem_core
    PUBLIC Eigen3::Eigen
)
if(MPFEM_USE_OPENMP)
    target_link_libraries(mpfem_core PUBLIC OpenMP::OpenMP_CXX)
endif()
mpfem_set_default_compiler_options(mpfem_core)

# mpfem_model (header-only)
target_include_directories(mpfem_model
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(mpfem_model
    INTERFACE Eigen3::Eigen
    INTERFACE mpfem_core
)

# mpfem_io
target_include_directories(mpfem_io
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(mpfem_io
    PUBLIC Eigen3::Eigen
    PUBLIC mpfem_core
    PUBLIC mpfem_model
    PUBLIC tinyxml2::tinyxml2
)
mpfem_set_default_compiler_options(mpfem_io)

# mpfem_fe
target_include_directories(mpfem_fe
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(mpfem_fe
    PUBLIC Eigen3::Eigen
    PUBLIC mpfem_core
    PUBLIC mpfem_mesh
    PUBLIC mpfem_model
)
mpfem_set_default_compiler_options(mpfem_fe)

# mpfem_mesh
target_include_directories(mpfem_mesh
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(mpfem_mesh
    PUBLIC Eigen3::Eigen
    PUBLIC mpfem_core
)
mpfem_set_default_compiler_options(mpfem_mesh)

# mpfem_assembly
target_include_directories(mpfem_assembly
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(mpfem_assembly
    PUBLIC Eigen3::Eigen
    PUBLIC mpfem_core
    PUBLIC mpfem_mesh
    PUBLIC mpfem_fe
    PUBLIC mpfem_solver
)
mpfem_set_default_compiler_options(mpfem_assembly)

# mpfem_solver (header-only)
target_include_directories(mpfem_solver
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(mpfem_solver
    INTERFACE Eigen3::Eigen
    INTERFACE mpfem_core
)

# mpfem_coupling (header-only)
target_include_directories(mpfem_coupling
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(mpfem_coupling
    INTERFACE Eigen3::Eigen
    INTERFACE mpfem_core
    INTERFACE mpfem_fe
)

# mpfem_physics
target_include_directories(mpfem_physics
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(mpfem_physics
    PUBLIC Eigen3::Eigen
    PUBLIC mpfem_core
    PUBLIC mpfem_mesh
    PUBLIC mpfem_fe
    PUBLIC mpfem_assembly
    PUBLIC mpfem_solver
    PUBLIC mpfem_coupling
    PUBLIC mpfem_io
    PUBLIC mpfem_model
)
mpfem_set_default_compiler_options(mpfem_physics)

# =============================================================================
# Compile definitions
# =============================================================================
if(MPFEM_USE_MKL)
    target_compile_definitions(mpfem_core PUBLIC MPFEM_USE_MKL)
    target_link_libraries(mpfem_core PUBLIC MKL::MKL)
endif()

if(MPFEM_USE_UMFPACK)
    target_compile_definitions(mpfem_core PUBLIC MPFEM_USE_UMFPACK)
    target_link_libraries(mpfem_core PUBLIC SuiteSparse::UMFPACK)
endif()

# =============================================================================
# Tests
# =============================================================================
if(MPFEM_BUILD_TESTS AND googletest_ADDED)
    add_subdirectory(tests)
endif()

# =============================================================================
# Examples
# =============================================================================
if(MPFEM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()