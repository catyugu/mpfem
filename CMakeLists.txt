cmake_minimum_required(VERSION 3.21)

project(mpfem LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerOptions)

option(MPFEM_BUILD_TESTS "Build mpfem tests" ON)
option(MPFEM_BUILD_EXAMPLES "Build mpfem examples" ON)

# Fetch Eigen (header-only library)
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen)

add_subdirectory(src)

if(MPFEM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(MPFEM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
