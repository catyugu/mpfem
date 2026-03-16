# =============================================================================
# mpfem - Multi-Physics Finite Element Method Library
# =============================================================================

cmake_minimum_required(VERSION 3.20)

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

# =============================================================================
# Installation
# =============================================================================

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
install(DIRECTORY src/ DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Install cmake files
install(FILES
    cmake/mpfemConfig.cmake.in
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpfem
)

# Configure package config
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mpfemConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/mpfemConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpfem
)

# Write package version
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/mpfemConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/mpfemConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/mpfemConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpfem
)
