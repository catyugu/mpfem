# =============================================================================
# Dependencies.cmake - External dependency management
# =============================================================================
#
# This module manages all external dependencies using a consistent approach:
# - System libraries: find_package with standard paths
# - Header-only libraries: CPM for downloading
#
# Output variables:
#   MPFEM_MKL_FOUND        - Intel MKL available
#   MPFEM_OPENBLAS_FOUND   - OpenBLAS available
#   MPFEM_SUITESPARSE_FOUND - SuiteSparse (UMFPACK) available
#   MPFEM_OPENMP_FOUND     - OpenMP available
#
# =============================================================================

include(CPM)

# =============================================================================
# 1. Core dependencies (required)
# =============================================================================

# Eigen3 (required)
find_package(Eigen3 REQUIRED)

# =============================================================================
# 2. Linear algebra backends (optional, priority: MKL > OpenBLAS)
# =============================================================================

# --- Intel MKL ---
option(MPFEM_USE_MKL "Use Intel MKL for BLAS/LAPACK and PARDISO solver" ON)
# If using LLVM style compiler, abandon MKL since it may not be compatible
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    message(WARNING "Intel MKL may not be compatible with Clang/AppleClang. Disabling MKL support.")
    set(MPFEM_USE_MKL OFF)
endif()
if(MPFEM_USE_MKL)
    # Avoid loading two different OpenMP runtimes in one process.
    # MKL's default intel_thread layer brings libiomp5, while clang-cl/OpenMP
    # often links libomp, which causes OMP Error #15 at runtime.
    set(MPFEM_MKL_THREADING "sequential" CACHE STRING "MKL threading layer (sequential|intel_thread|tbb_thread)")
    set_property(CACHE MPFEM_MKL_THREADING PROPERTY STRINGS sequential intel_thread tbb_thread)
    set(MKL_THREADING "${MPFEM_MKL_THREADING}" CACHE STRING "MKL threading layer" FORCE)

    # Set MKL_ROOT from environment if available
    if(DEFINED ENV{MKLROOT})
        set(MKL_ROOT "$ENV{MKLROOT}")
    elseif(DEFINED ENV{MKL_DIR})
        set(MKL_ROOT "$ENV{MKL_DIR}")
    endif()
    
    # Prefer MKL CMake config if available
    if(MKL_ROOT AND EXISTS "${MKL_ROOT}/lib/cmake/mkl/MKLConfig.cmake")
        set(MKL_DIR "${MKL_ROOT}/lib/cmake/mkl")
        find_package(MKL CONFIG QUIET)
    else()
        # Try system-wide search
        find_package(MKL QUIET)
    endif()
    
    if(MKL_FOUND)
        set(MPFEM_MKL_FOUND TRUE)
        message(STATUS "Intel MKL found: ${MKL_ROOT} (threading=${MKL_THREADING})")
    else()
        message(STATUS "Intel MKL not found")
        set(MPFEM_MKL_FOUND FALSE)
    endif()
else()
    set(MPFEM_MKL_FOUND FALSE)
endif()

# --- OpenBLAS (fallback for BLAS acceleration) ---
# Only enable if MKL is not available
# Note: We avoid OpenBLAS's CMake config because it forces Fortran detection
option(MPFEM_USE_OPENBLAS "Use OpenBLAS for BLAS acceleration" ON)
if(MPFEM_USE_OPENBLAS AND NOT MPFEM_MKL_FOUND)
    # Manual search for OpenBLAS (avoid Fortran detection in config file)
    find_path(OpenBLAS_INCLUDE_DIR
        NAMES openblas_config.h cblas.h
        PATHS
            $ENV{OpenBLAS_DIR}
            $ENV{OpenBLAS_ROOT}
            /usr/include
            /usr/local/include
            /opt/openblas/include
    )
    
    find_library(OpenBLAS_LIBRARY
        NAMES openblas libopenblas
        PATHS
            $ENV{OpenBLAS_DIR}
            $ENV{OpenBLAS_ROOT}
            /usr/lib
            /usr/local/lib
            /opt/openblas/lib
    )
    
    if(OpenBLAS_INCLUDE_DIR AND OpenBLAS_LIBRARY)
        set(OpenBLAS_FOUND TRUE)
        set(OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})
        set(OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY})
        
        # Create imported target
        add_library(OpenBLAS::OpenBLAS STATIC IMPORTED)
        set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIR}"
            IMPORTED_LOCATION "${OpenBLAS_LIBRARY}"
        )
    endif()
    
    if(OpenBLAS_FOUND)
        set(MPFEM_OPENBLAS_FOUND TRUE)
        message(STATUS "OpenBLAS found: ${OpenBLAS_LIBRARY}")
    else()
        message(STATUS "OpenBLAS not found")
        set(MPFEM_OPENBLAS_FOUND FALSE)
    endif()
else()
    set(MPFEM_OPENBLAS_FOUND FALSE)
endif()

# =============================================================================
# 3. Direct solvers (optional)
# =============================================================================

# --- SuiteSparse (UMFPACK) ---
option(MPFEM_USE_SUITESPARSE "Use SuiteSparse (UMFPACK) direct solver" ON)
if(MPFEM_USE_SUITESPARSE)
    # Try to find SuiteSparse via CMake config
    find_package(SuiteSparse QUIET)
    if(NOT SuiteSparse_FOUND)
        # Try individual components
        find_package(UMFPACK QUIET)
        if(UMFPACK_FOUND)
            set(SuiteSparse_FOUND TRUE)
        endif()
    endif()
    
    if(SuiteSparse_FOUND OR UMFPACK_FOUND)
        set(MPFEM_SUITESPARSE_FOUND TRUE)
        message(STATUS "SuiteSparse/UMFPACK found")
    else()
        message(STATUS "SuiteSparse/UMFPACK not found")
        set(MPFEM_SUITESPARSE_FOUND FALSE)
    endif()
else()
    set(MPFEM_SUITESPARSE_FOUND FALSE)
endif()

# =============================================================================
# 4. OpenMP for parallelization
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
message(STATUS "Eigen3:           FOUND")
message(STATUS "Intel MKL:        ${MPFEM_MKL_FOUND}")
message(STATUS "OpenBLAS:         ${MPFEM_OPENBLAS_FOUND}")
message(STATUS "SuiteSparse:      ${MPFEM_SUITESPARSE_FOUND}")
message(STATUS "OpenMP:           ${MPFEM_OPENMP_FOUND}")
message(STATUS "Build tests:      ${MPFEM_BUILD_TESTS}")
message(STATUS "Build examples:   ${MPFEM_BUILD_EXAMPLES}")
message(STATUS "================================")
message(STATUS "")
