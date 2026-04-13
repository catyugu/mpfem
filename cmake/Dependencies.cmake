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
#   MPFEM_UMFPACK_FOUND    - SuiteSparse::UMFPACK available
#   MPFEM_OPENMP_FOUND     - OpenMP available
#
# =============================================================================

include(CPM)

# =============================================================================
# 1. Core dependencies (required)
# =============================================================================

# Eigen3 (required)
find_package(Eigen3 CONFIG REQUIRED)

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
    set(MKL_LINK "static")          # 静态链接
    set(MKL_INTERFACE "lp64")       # 默认整数接口（最通用）
    set(MKL_THREADING "sequential") # 单线程（无 OpenMP 依赖，最稳定）

    if(DEFINED ENV{MKLROOT})
        set(MKL_ROOT "$ENV{MKLROOT}")
    elseif(DEFINED ENV{MKL_DIR})
        set(MKL_ROOT "$ENV{MKL_DIR}")
    endif()
    
    # Prefer MKL CMake config if available
    if(MKL_ROOT AND EXISTS "${MKL_ROOT}/lib/cmake/mkl/MKLConfig.cmake")
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

# =============================================================================
# 3. Direct solvers (optional)
# =============================================================================

# ---  UMFPACK ---
option(MPFEM_USE_UMFPACK "Use UMFPACK direct solver" ON)
if(MPFEM_USE_UMFPACK)
    # Try to find SuiteSparse
    find_package(UMFPACK CONFIG QUIET)
    
    if(UMFPACK_FOUND)
        set(MPFEM_UMFPACK_FOUND TRUE)
        message(STATUS "UMFPACK found")
    else()
        message(STATUS "UMFPACK not found")
        set(MPFEM_UMFPACK_FOUND FALSE)
    endif()
else()
    set(MPFEM_UMFPACK_FOUND FALSE)
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
message(STATUS "UMFPACK:          ${MPFEM_UMFPACK_FOUND}")
message(STATUS "OpenMP:           ${MPFEM_OPENMP_FOUND}")
message(STATUS "Build tests:      ${MPFEM_BUILD_TESTS}")
message(STATUS "Build examples:   ${MPFEM_BUILD_EXAMPLES}")
message(STATUS "================================")
message(STATUS "")
