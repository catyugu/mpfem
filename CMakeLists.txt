cmake_minimum_required(VERSION 3.20)

project(mpfem
    VERSION 0.1.0
    DESCRIPTION "Multi-Physics Finite Element Method Library"
    LANGUAGES CXX
)

# ============================================================
# Options
# ============================================================
option(MPFEM_BUILD_TESTS "Build tests" ON)
option(MPFEM_BUILD_EXAMPLES "Build examples" ON)
option(MPFEM_ENABLE_PARDISO "Enable PARDISO solver (requires MKL)" OFF)

# ============================================================
# C++ Standard and Compiler Options
# ============================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/CompilerOptions.cmake)

# ============================================================
# Dependencies (managed by CPM)
# ============================================================
include(cmake/CPM.cmake)

# Eigen - Linear algebra library
CPMAddPackage(
    NAME Eigen3
    GITHUB_REPOSITORY libigl/eigen
    GIT_TAG 3.4.0
    OPTIONS
        "BUILD_TESTING OFF"
        "BUILD_DOC OFF"
        "EIGEN_BUILD_DOC OFF"
)

# tinyxml2 - XML parser (use local version)
add_subdirectory(external/tinyxml2)

# Google Test (optional, for testing)
if(MPFEM_BUILD_TESTS)
    CPMAddPackage(
        NAME GTest
        GITHUB_REPOSITORY google/googletest
        GIT_TAG v1.14.0
        OPTIONS
            "BUILD_GMOCK OFF"
            "INSTALL_GTEST OFF"
    )
    enable_testing()
endif()

# ============================================================
# Main Library Target
# ============================================================
add_subdirectory(src)

# ============================================================
# Tests
# ============================================================
if(MPFEM_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ============================================================
# Examples
# ============================================================
if(MPFEM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================
# Install
# ============================================================
include(GNUInstallDirs)

install(TARGETS mpfem
    EXPORT mpfem-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mpfem
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT mpfem-targets
    FILE mpfem-targets.cmake
    NAMESPACE mpfem::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpfem
)

# ============================================================
# Print Configuration Summary
# ============================================================
message(STATUS "")
message(STATUS "=== mpfem Configuration Summary ===")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build Tests:    ${MPFEM_BUILD_TESTS}")
message(STATUS "  Build Examples: ${MPFEM_BUILD_EXAMPLES}")
message(STATUS "  Enable PARDISO: ${MPFEM_ENABLE_PARDISO}")
message(STATUS "====================================")
message(STATUS "")
