# =============================================================================
# Dependencies.cmake - External dependency management
# =============================================================================
#
# This module manages all external dependencies using a consistent approach:
# - System libraries: find_package with standard paths
# - Header-only libraries: CPM for downloading
#
# Output variables:
# MPFEM_MKL_FOUND        - Intel MKL available
# MPFEM_UMFPACK_FOUND    - SuiteSparse::UMFPACK available
# MPFEM_OPENMP_FOUND     - OpenMP available
#
# =============================================================================

include(CPM)

# --- Intel MKL ---
option(MPFEM_USE_MKL "Use Intel MKL for BLAS/LAPACK and PARDISO solver" ON)

if(MPFEM_USE_MKL)
    set(MKL_LINK "sdl" CACHE STRING "MKL link type (sdl|static|dynamic)")
    set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading runtime")
    set(MKL_INTERFACE "lp64" CACHE STRING "MKL index interface (lp64|ilp64)")
    find_package(MKL QUIET)

    if(TARGET MKL::MKL)
        set(MPFEM_MKL_FOUND TRUE CACHE INTERNAL "" FORCE)

        if(TARGET MKL::mkl_rt)
            get_target_property(_mkl_rt_loc MKL::mkl_rt LOCATION)
            get_filename_component(MKL_BIN_DIR "${_mkl_rt_loc}" DIRECTORY)
            message(STATUS "MKL FOUND!")
        else()
            message(WARNING
                "oneMKL was found without MKL::mkl_rt; runtime DLLs will not "
                "be copied automatically")
        endif()
    else()
        message(WARNING
            "USE_MKL=ON but oneMKL was not found; disabling Pardiso and "
            "falling back to Eigen EigenSparseLU")
    endif()
else()
    set(MPFEM_MKL_FOUND FALSE)
endif()

# ---  UMFPACK ---
option(MPFEM_USE_UMFPACK "Use UMFPACK direct solver" ON)

if(MPFEM_USE_UMFPACK)
    # Try to find SuiteSparse
    find_package(UMFPACK QUIET)

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

option(MPFEM_BUILD_TESTS "Build unit tests" ON)
option(MPFEM_BUILD_EXAMPLES "Build examples" ON)

CPMAddPackage(
    GITLAB_REPOSITORY libeigen/eigen
    GIT_TAG 5.0.0
    OPTIONS
    "EIGEN_BUILD_DOC OFF"
    "EIGEN_BUILD_TESTING OFF"
    "EIGEN_BUILD_PKGCONFIG OFF"
)

# tinyxml2 (required for XML parsing)
CPMAddPackage(
    NAME tinyxml2
    GITHUB_REPOSITORY leethomason/tinyxml2
    GIT_TAG 11.0.0
    OPTIONS
    "BUILD_TESTING OFF"
)

# FEniCS Basix (finite element basis evaluation)
CPMAddPackage(
    NAME basix

    # 关键：直接下载 cpp 目录，不是整个项目！
    GITHUB_REPOSITORY FEniCS/basix
    GIT_TAG v0.9.0
    SOURCE_SUBDIR cpp
    OPTIONS
    "CMAKE_POLICY_VERSION_MINIMUM 3.21"
    "BUILD_SHARED_LIBS OFF"
)

# GoogleTest (optional, for testing)
if(MPFEM_BUILD_TESTS)
    enable_testing()
    CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        GIT_TAG v1.15.2
        OPTIONS
        " BUILD_GMOCK OFF "
        " INSTALL_GTEST OFF "
    )
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS " ")
message(STATUS " === mpfem Dependency Summary === ")
message(STATUS " Eigen3: FOUND ")
message(STATUS " Intel MKL: ${MPFEM_MKL_FOUND} ")
message(STATUS " UMFPACK: ${MPFEM_UMFPACK_FOUND} ")
message(STATUS " OpenMP: ${MPFEM_OPENMP_FOUND} ")
message(STATUS " Build tests: ${MPFEM_BUILD_TESTS} ")
message(STATUS " Build examples: ${MPFEM_BUILD_EXAMPLES} ")
message(STATUS " ================================ ")
message(STATUS " ")

# =============================================================================
# Suppress warnings from external (CPM) packages
# =============================================================================
if(MSVC)
    if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
        set(EXTERNAL_WARNING_FLAGS "-w")
    else()
        set(EXTERNAL_WARNING_FLAGS /WX- /W0)
    endif()
else()
    set(EXTERNAL_WARNING_FLAGS "-w")
endif()

# Suppress warnings for known CPM targets
# Note: Some targets may not exist yet at this point (created by CPMAddPackage)
set(CPM_KNOWN_TARGETS basix tinyxml2)

if(MPFEM_BUILD_TESTS)
    list(APPEND CPM_KNOWN_TARGETS gtest gtest_main gmock gmock_main)
endif()

foreach(CPM_TARGET IN LISTS CPM_KNOWN_TARGETS)
    if(TARGET ${CPM_TARGET})
        target_compile_options(${CPM_TARGET} PRIVATE ${EXTERNAL_WARNING_FLAGS})
    endif()
endforeach()
