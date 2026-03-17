# =============================================================================
# Dependencies.cmake - External dependency management
# =============================================================================
#
# This module manages all external dependencies using a consistent approach:
# - System libraries: find_package with custom Find modules
# - Header-only libraries: CPM for downloading
#
# Output variables:
#   MPFEM_OPENBLAS_FOUND   - OpenBLAS available
#   MPFEM_OPENMP_FOUND     - OpenMP available
#   MPFEM_SUPERLU_FOUND    - SuperLU available
#   MPFEM_SUITESPARSE_FOUND - SuiteSparse (UMFPACK) available
#   MPFEM_CHOLMOD_FOUND    - CHOLMOD available
#
# =============================================================================

include(CPM)

# =============================================================================
# 1. Core dependencies (required)
# =============================================================================

# Eigen3 (required)
find_package(Eigen3 REQUIRED)

# =============================================================================
# 2. Optional: OpenBLAS for Eigen BLAS acceleration
# =============================================================================

option(MPFEM_USE_OPENBLAS "Use OpenBLAS for Eigen BLAS acceleration" ON)
if(MPFEM_USE_OPENBLAS)
    find_package(OpenBLAS QUIET)
    if(OpenBLAS_FOUND)
        message(STATUS "OpenBLAS found: ${OpenBLAS_LIBRARIES}")
        set(MPFEM_OPENBLAS_FOUND TRUE)
    else()
        message(STATUS "OpenBLAS not found, using Eigen's built-in backend")
        set(MPFEM_OPENBLAS_FOUND FALSE)
    endif()
endif()

# =============================================================================
# 3. Optional: OpenMP for parallelization
# =============================================================================

option(MPFEM_USE_OPENMP "Use OpenMP for parallelization" ON)
if(MPFEM_USE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_FOUND)
        message(STATUS "OpenMP found")
        set(MPFEM_OPENMP_FOUND TRUE)
    else()
        message(STATUS "OpenMP not found, parallelization disabled")
        set(MPFEM_OPENMP_FOUND FALSE)
    endif()
endif()

# =============================================================================
# 4. Optional: Linear solver libraries
# =============================================================================

# SuperLU (optional)
option(MPFEM_USE_SUPERLU "Use SuperLU solver" ON)
if(MPFEM_USE_SUPERLU)
    find_package(SuperLU QUIET)
    if(SuperLU_FOUND)
        message(STATUS "SuperLU found: ${SuperLU_LIBRARIES}")
        set(MPFEM_SUPERLU_FOUND TRUE)
    else()
        message(STATUS "SuperLU not found")
        set(MPFEM_SUPERLU_FOUND FALSE)
    endif()
endif()

# SuiteSparse (UMFPACK, CHOLMOD, etc.)
option(MPFEM_USE_SUITESPARSE "Use SuiteSparse solvers (UMFPACK, CHOLMOD)" ON)
if(MPFEM_USE_SUITESPARSE)
    find_package(SuiteSparse QUIET)
    
    if(SuiteSparse_FOUND OR UMFPACK_FOUND)
        message(STATUS "SuiteSparse found")
        set(MPFEM_SUITESPARSE_FOUND TRUE)
        set(MPFEM_UMFPACK_FOUND TRUE)
    else()
        message(STATUS "SuiteSparse not found")
        set(MPFEM_SUITESPARSE_FOUND FALSE)
        set(MPFEM_UMFPACK_FOUND FALSE)
    endif()
    
    if(CHOLMOD_FOUND)
        message(STATUS "CHOLMOD found")
        set(MPFEM_CHOLMOD_FOUND TRUE)
    else()
        set(MPFEM_CHOLMOD_FOUND FALSE)
    endif()
endif()

# =============================================================================
# 5. Build dependencies (downloaded via CPM)
# =============================================================================

option(MPFEM_BUILD_TESTS "Build unit tests" ON)
option(MPFEM_BUILD_EXAMPLES "Build examples" ON)

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
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== mpfem Dependency Summary ===")
message(STATUS "Eigen3:          FOUND")
message(STATUS "OpenBLAS:        ${MPFEM_OPENBLAS_FOUND}")
message(STATUS "OpenMP:          ${MPFEM_OPENMP_FOUND}")
message(STATUS "SuperLU:         ${MPFEM_SUPERLU_FOUND}")
message(STATUS "UMFPACK:         ${MPFEM_UMFPACK_FOUND}")
message(STATUS "CHOLMOD:         ${MPFEM_CHOLMOD_FOUND}")
message(STATUS "Build tests:     ${MPFEM_BUILD_TESTS}")
message(STATUS "Build examples:  ${MPFEM_BUILD_EXAMPLES}")
message(STATUS "================================")
message(STATUS "")