# =============================================================================
# FindOpenBLAS.cmake - Find OpenBLAS library
# =============================================================================
#
# This module finds OpenBLAS and sets the following variables:
#
#   OpenBLAS_FOUND        - True if OpenBLAS was found
#   OpenBLAS_INCLUDE_DIRS - Include directories
#   OpenBLAS_LIBRARIES    - Libraries to link
#   OpenBLAS_VERSION      - Version string
#
# This module respects:
#   OpenBLAS_ROOT         - Preferred installation prefix
#
# =============================================================================

include(FindPackageHandleStandardArgs)

# Try pkg-config first
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PC_OpenBLAS QUIET openblas)
endif()

# Find include directory
find_path(OpenBLAS_INCLUDE_DIR
    NAMES cblas.h openblas/cblas.h
    HINTS
        ${OpenBLAS_ROOT}
        ENV OpenBLAS_ROOT
        ${PC_OpenBLAS_INCLUDE_DIRS}
    PATH_SUFFIXES include include/openblas
)

# Find library
find_library(OpenBLAS_LIBRARY
    NAMES openblas
    HINTS
        ${OpenBLAS_ROOT}
        ENV OpenBLAS_ROOT
        ${PC_OpenBLAS_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
)

# Extract version from header if available
if(OpenBLAS_INCLUDE_DIR)
    set(_version_regex "^#define[ \t]+OPENBLAS_VERSION[ \t]+\"([^\"]+)\"")
    file(STRINGS "${OpenBLAS_INCLUDE_DIR}/openblas_config.h" _version_line
        REGEX "${_version_regex}" LIMIT_COUNT 1)
    if(_version_line)
        string(REGEX REPLACE "${_version_regex}" "\\1" OpenBLAS_VERSION "${_version_line}")
    endif()
endif()

# Handle standard arguments
find_package_handle_standard_args(OpenBLAS
    REQUIRED_VARS OpenBLAS_LIBRARY OpenBLAS_INCLUDE_DIR
    VERSION_VAR OpenBLAS_VERSION
)

# Set output variables
if(OpenBLAS_FOUND)
    set(OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY})
    set(OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})
    
    # Create imported target
    if(NOT TARGET OpenBLAS::OpenBLAS)
        add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)
        set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
            IMPORTED_LOCATION "${OpenBLAS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIR}"
        )
    endif()
    
    mark_as_advanced(OpenBLAS_INCLUDE_DIR OpenBLAS_LIBRARY)
endif()
