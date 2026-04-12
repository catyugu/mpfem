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



if(MSVC)
    set(MSVC_IGNORE_WARNINGS
        /wd4244  # 'return': 从__int64/Eigen::Index转int，可能丢失数据
        /wd4267  # 'return': 从size_t转int，可能丢失数据
        /wd4819  # 文件含不能在代码页936表示的字符（UTF-8问题）
        /wd4324  # 结构因对齐被填充（无害）
    )

# 全局加给所有目标
    add_compile_options(${MSVC_IGNORE_WARNINGS})
    add_compile_options(/W4 /permissive- /utf-8 /wd4819 /bigobj /MP /INCREMENTAL)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(MINGW)
    add_compile_options(-Wa,-mbig-obj)
endif()

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

