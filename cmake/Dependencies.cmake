# =============================================================================
# Dependencies.cmake - External dependency management
# =============================================================================
#
# This module manages all external dependencies using a consistent approach:
# - System libraries: find_package with custom Find modules
# - Header-only libraries: CPM for downloading
#
# Output variables:
#   MPFEM_MKL_FOUND       - Intel MKL available
#   MPFEM_OPENMP_FOUND    - OpenMP available
#
# =============================================================================

include(CPM)

# =============================================================================
# 1. Core dependencies (required)
# =============================================================================

# Eigen3 (required)
find_package(Eigen3 REQUIRED)

# =============================================================================
# 2. Optional: Intel MKL
# =============================================================================

option(MPFEM_USE_MKL "Use Intel MKL for BLAS/LAPACK and PARDISO solver" ON)
if(MPFEM_USE_MKL)
    # Try to find MKL via environment variable or standard paths
    if(DEFINED ENV{MKLROOT})
        set(MKL_ROOT "$ENV{MKLROOT}")
    elseif(DEFINED ENV{MKL_DIR})
        set(MKL_ROOT "$ENV{MKL_DIR}")
    elseif(EXISTS "E:/env/cpp/intel/oneAPI/mkl/latest")
        set(MKL_ROOT "E:/env/cpp/intel/oneAPI/mkl/latest")
    endif()
    
    if(MKL_ROOT)
        set(MKL_INCLUDE_DIRS "${MKL_ROOT}/include")
        set(MKL_LIB_DIR "${MKL_ROOT}/lib")
        
        # Find MKL libraries (sequential version for simplicity)
        find_library(MKL_CORE_LIB mkl_core PATHS "${MKL_LIB_DIR}" NO_DEFAULT_PATH)
        find_library(MKL_INTEL_LP64_LIB mkl_intel_lp64 PATHS "${MKL_LIB_DIR}" NO_DEFAULT_PATH)
        find_library(MKL_SEQUENTIAL_LIB mkl_sequential PATHS "${MKL_LIB_DIR}" NO_DEFAULT_PATH)
        
        if(MKL_CORE_LIB AND MKL_INTEL_LP64_LIB AND MKL_SEQUENTIAL_LIB)
            set(MKL_LIBRARIES 
                ${MKL_INTEL_LP64_LIB}
                ${MKL_SEQUENTIAL_LIB}
                ${MKL_CORE_LIB}
            )
            
            # Create imported target
            add_library(MKL::MKL INTERFACE IMPORTED)
            target_include_directories(MKL::MKL INTERFACE "${MKL_INCLUDE_DIRS}")
            target_link_libraries(MKL::MKL INTERFACE ${MKL_LIBRARIES})
            
            set(MKL_FOUND TRUE)
            set(MPFEM_MKL_FOUND TRUE)
            message(STATUS "Intel MKL found: ${MKL_ROOT}")
            message(STATUS "  MKL libraries: ${MKL_LIBRARIES}")
        else()
            message(STATUS "Intel MKL libraries not found in: ${MKL_LIB_DIR}")
            message(STATUS "  mkl_core: ${MKL_CORE_LIB}")
            message(STATUS "  mkl_intel_lp64: ${MKL_INTEL_LP64_LIB}")
            message(STATUS "  mkl_sequential: ${MKL_SEQUENTIAL_LIB}")
            set(MPFEM_MKL_FOUND FALSE)
        endif()
    else()
        # Try system-wide search
        find_package(MKL QUIET)
        if(MKL_FOUND)
            set(MPFEM_MKL_FOUND TRUE)
            message(STATUS "Intel MKL found (system)")
        else()
            message(STATUS "Intel MKL not found")
            set(MPFEM_MKL_FOUND FALSE)
        endif()
    endif()
else()
    set(MPFEM_MKL_FOUND FALSE)
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
# 4. Build dependencies (downloaded via CPM)
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
message(STATUS "Intel MKL:       ${MPFEM_MKL_FOUND}")
message(STATUS "OpenMP:          ${MPFEM_OPENMP_FOUND}")
message(STATUS "Build tests:     ${MPFEM_BUILD_TESTS}")
message(STATUS "Build examples:  ${MPFEM_BUILD_EXAMPLES}")
message(STATUS "================================")
message(STATUS "")