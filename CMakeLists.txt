# =============================================================================
# mpfem - Multi-Physics Finite Element Method Library
# =============================================================================

cmake_minimum_required(VERSION 3.20)


if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain")
    message(STATUS "Using VCPKG_ROOT: $ENV{VCPKG_ROOT}")
endif()

project(mpfem
    VERSION 0.1.0
    DESCRIPTION "Multi-Physics Finite Element Method Library"
    LANGUAGES CXX C
)

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

# =============================================================================
# Dependencies
# =============================================================================

include(Dependencies)

# =============================================================================
# Library targets
# =============================================================================

include(Targets)

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

