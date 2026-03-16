# =============================================================================
# Dependencies.cmake - External dependency management
# =============================================================================

include(CPM)

# =============================================================================
# Required dependencies
# =============================================================================

# Eigen3 (required)
find_package(Eigen3 REQUIRED)

# OpenBLAS (optional, for Eigen acceleration)
option(MPFEM_USE_OPENBLAS "Use OpenBLAS for Eigen BLAS acceleration" ON)
if(MPFEM_USE_OPENBLAS)
    # Use pkg-config to find OpenBLAS
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OpenBLAS openblas)
    endif()
    
    if(NOT OpenBLAS_FOUND)
        # Standard find for OpenBLAS
        find_library(OPENBLAS_LIBRARY NAMES openblas)
        find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h)
        
        if(OPENBLAS_LIBRARY AND OPENBLAS_INCLUDE_DIR)
            set(OpenBLAS_FOUND TRUE)
            set(OpenBLAS_LIBRARIES ${OPENBLAS_LIBRARY})
            set(OpenBLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIR})
        endif()
    endif()
    
    if(OpenBLAS_FOUND)
        message(STATUS "OpenBLAS found: ${OpenBLAS_LIBRARIES}")
        set(MPFEM_OPENBLAS_FOUND TRUE)
    else()
        message(STATUS "OpenBLAS not found, using Eigen's built-in backend")
        set(MPFEM_OPENBLAS_FOUND FALSE)
    endif()
endif()

# OpenMP (optional)
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
# Linear solver libraries
# =============================================================================

# SuperLU (optional)
option(MPFEM_USE_SUPERLU "Use SuperLU solver" ON)
if(MPFEM_USE_SUPERLU)
    # Try standard find methods first
    find_library(SUPERLU_LIBRARY NAMES superlu)
    find_path(SUPERLU_INCLUDE_DIR NAMES superlu/slu_Cnames.h slu_Cnames.h)
    
    if(SUPERLU_LIBRARY AND SUPERLU_INCLUDE_DIR)
        message(STATUS "SuperLU found: ${SUPERLU_LIBRARY}")
        set(MPFEM_SUPERLU_FOUND TRUE)
    else()
        message(STATUS "SuperLU not found")
        set(MPFEM_SUPERLU_FOUND FALSE)
    endif()
endif()

# SuiteSparse (UMFPACK, CHOLMOD, etc.)
option(MPFEM_USE_SUITESPARSE "Use SuiteSparse solvers (UMFPACK, CHOLMOD)" ON)
if(MPFEM_USE_SUITESPARSE)
    # Try SuiteSparse config first
    find_package(SuiteSparse QUIET)
    
    if(SuiteSparse_FOUND)
        message(STATUS "SuiteSparse found via config")
        set(MPFEM_SUITESPARSE_FOUND TRUE)
        set(MPFEM_UMFPACK_FOUND TRUE)
        set(MPFEM_CHOLMOD_FOUND TRUE)
    else()
        # Standard find for UMFPACK
        find_library(UMFPACK_LIBRARY NAMES umfpack)
        find_library(AMD_LIBRARY NAMES amd)
        find_library(CHOLMOD_LIBRARY NAMES cholmod)
        find_library(COLAMD_LIBRARY NAMES colamd)
        find_library(SUITESPARSE_CONFIG_LIBRARY NAMES suitesparseconfig)
        find_path(UMFPACK_INCLUDE_DIR NAMES umfpack.h suitesparse/umfpack.h)
        
        if(UMFPACK_LIBRARY AND UMFPACK_INCLUDE_DIR)
            message(STATUS "UMFPACK found: ${UMFPACK_LIBRARY}")
            set(MPFEM_SUITESPARSE_FOUND TRUE)
            set(MPFEM_UMFPACK_FOUND TRUE)
            set(UMFPACK_LIBRARIES 
                ${UMFPACK_LIBRARY} 
                ${AMD_LIBRARY} 
                ${CHOLMOD_LIBRARY} 
                ${COLAMD_LIBRARY} 
                ${SUITESPARSE_CONFIG_LIBRARY})
            
            # Check for CHOLMOD separately
            if(CHOLMOD_LIBRARY)
                message(STATUS "CHOLMOD found: ${CHOLMOD_LIBRARY}")
                set(MPFEM_CHOLMOD_FOUND TRUE)
            else()
                set(MPFEM_CHOLMOD_FOUND FALSE)
            endif()
        else()
            message(STATUS "SuiteSparse not found")
            set(MPFEM_SUITESPARSE_FOUND FALSE)
            set(MPFEM_UMFPACK_FOUND FALSE)
            set(MPFEM_CHOLMOD_FOUND FALSE)
        endif()
    endif()
endif()

# =============================================================================
# Build dependencies (downloaded via CPM)
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
