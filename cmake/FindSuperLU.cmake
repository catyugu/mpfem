# =============================================================================
# FindSuperLU.cmake - Find SuperLU library
# =============================================================================
#
# This module finds SuperLU and sets the following variables:
#
#   SuperLU_FOUND        - True if SuperLU was found
#   SuperLU_INCLUDE_DIRS - Include directories
#   SuperLU_LIBRARIES    - Libraries to link
#
# This module respects:
#   SuperLU_ROOT         - Preferred installation prefix
#
# =============================================================================

include(FindPackageHandleStandardArgs)

# Try pkg-config first
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PC_SuperLU QUIET superlu)
endif()

# Find include directory
# Eigen's SuperLUSupport uses #include <slu_Cnames.h>
# The file is typically at: <prefix>/include/superlu/slu_Cnames.h
find_path(SuperLU_INCLUDE_DIR
    NAMES slu_Cnames.h
    HINTS
        ${SuperLU_ROOT}
        ENV SuperLU_ROOT
        ${PC_SuperLU_INCLUDE_DIRS}
    PATH_SUFFIXES 
        superlu
        include/superlu
)

# Find library
find_library(SuperLU_LIBRARY
    NAMES superlu
    HINTS
        ${SuperLU_ROOT}
        ENV SuperLU_ROOT
        ${PC_SuperLU_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
)

# Handle standard arguments
find_package_handle_standard_args(SuperLU
    REQUIRED_VARS SuperLU_LIBRARY SuperLU_INCLUDE_DIR
)

# Set output variables and create imported target
if(SuperLU_FOUND)
    set(SuperLU_LIBRARIES ${SuperLU_LIBRARY})
    set(SuperLU_INCLUDE_DIRS ${SuperLU_INCLUDE_DIR})
    
    if(NOT TARGET SuperLU::SuperLU)
        add_library(SuperLU::SuperLU UNKNOWN IMPORTED)
        set_target_properties(SuperLU::SuperLU PROPERTIES
            IMPORTED_LOCATION "${SuperLU_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${SuperLU_INCLUDE_DIR}"
        )
    endif()
    
    mark_as_advanced(SuperLU_INCLUDE_DIR SuperLU_LIBRARY)
endif()