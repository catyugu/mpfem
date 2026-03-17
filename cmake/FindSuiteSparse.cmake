# =============================================================================
# FindSuiteSparse.cmake - Find SuiteSparse libraries (UMFPACK, CHOLMOD, etc.)
# =============================================================================
#
# This module finds SuiteSparse components and sets the following variables:
#
#   SuiteSparse_FOUND        - True if SuiteSparse was found
#   SuiteSparse_INCLUDE_DIRS - Include directories
#   SuiteSparse_LIBRARIES    - Libraries to link
#
# Component variables (set based on SuiteSparse_FOUND or individual find):
#   UMFPACK_FOUND            - True if UMFPACK was found
#   UMFPACK_LIBRARIES        - UMFPACK libraries
#   CHOLMOD_FOUND            - True if CHOLMOD was found
#   CHOLMOD_LIBRARIES        - CHOLMOD libraries
#
# This module respects:
#   SuiteSparse_ROOT         - Preferred installation prefix
#
# =============================================================================

include(FindPackageHandleStandardArgs)

# Try SuiteSparse config first (modern installations)
find_package(SuiteSparse QUIET CONFIG)
if(SuiteSparse_FOUND)
    # Config-based find succeeded, extract component info
    if(TARGET SuiteSparse::UMFPACK)
        set(UMFPACK_FOUND TRUE)
        get_target_property(UMFPACK_LIBRARIES SuiteSparse::UMFPACK IMPORTED_LOCATION)
    endif()
    if(TARGET SuiteSparse::CHOLMOD)
        set(CHOLMOD_FOUND TRUE)
        get_target_property(CHOLMOD_LIBRARIES SuiteSparse::CHOLMOD IMPORTED_LOCATION)
    endif()
    return()
endif()

# Fallback to manual find

# Common include directory
find_path(SuiteSparse_INCLUDE_DIR
    NAMES umfpack.h suitesparse/umfpack.h
    HINTS
        ${SuiteSparse_ROOT}
        ENV SuiteSparse_ROOT
    PATH_SUFFIXES include include/suitesparse
)

# Find UMFPACK and its dependencies
find_library(UMFPACK_LIBRARY NAMES umfpack
    HINTS ${SuiteSparse_ROOT} ENV SuiteSparse_ROOT
    PATH_SUFFIXES lib lib64)
find_library(AMD_LIBRARY NAMES amd
    HINTS ${SuiteSparse_ROOT} ENV SuiteSparse_ROOT
    PATH_SUFFIXES lib lib64)
find_library(CHOLMOD_LIBRARY NAMES cholmod
    HINTS ${SuiteSparse_ROOT} ENV SuiteSparse_ROOT
    PATH_SUFFIXES lib lib64)
find_library(COLAMD_LIBRARY NAMES colamd
    HINTS ${SuiteSparse_ROOT} ENV SuiteSparse_ROOT
    PATH_SUFFIXES lib lib64)
find_library(SUITESPARSE_CONFIG_LIBRARY NAMES suitesparseconfig
    HINTS ${SuiteSparse_ROOT} ENV SuiteSparse_ROOT
    PATH_SUFFIXES lib lib64)

# Determine which components are available
if(UMFPACK_LIBRARY AND AMD_LIBRARY AND SuiteSparse_INCLUDE_DIR)
    set(UMFPACK_FOUND TRUE)
    set(UMFPACK_LIBRARIES 
        ${UMFPACK_LIBRARY} 
        ${AMD_LIBRARY}
        ${CHOLMOD_LIBRARY}
        ${COLAMD_LIBRARY}
        ${SUITESPARSE_CONFIG_LIBRARY})
endif()

if(CHOLMOD_LIBRARY)
    set(CHOLMOD_FOUND TRUE)
    set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY})
endif()

# Overall SuiteSparse found if UMFPACK is found
if(UMFPACK_FOUND)
    set(SuiteSparse_FOUND TRUE)
    set(SuiteSparse_INCLUDE_DIRS ${SuiteSparse_INCLUDE_DIR})
    set(SuiteSparse_LIBRARIES ${UMFPACK_LIBRARIES})
endif()

# Create imported targets
if(UMFPACK_FOUND AND NOT TARGET SuiteSparse::UMFPACK)
    add_library(SuiteSparse::UMFPACK UNKNOWN IMPORTED)
    set_target_properties(SuiteSparse::UMFPACK PROPERTIES
        IMPORTED_LOCATION "${UMFPACK_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SuiteSparse_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${AMD_LIBRARY};${SUITESPARSE_CONFIG_LIBRARY}"
    )
endif()

if(CHOLMOD_FOUND AND NOT TARGET SuiteSparse::CHOLMOD)
    add_library(SuiteSparse::CHOLMOD UNKNOWN IMPORTED)
    set_target_properties(SuiteSparse::CHOLMOD PROPERTIES
        IMPORTED_LOCATION "${CHOLMOD_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SuiteSparse_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${AMD_LIBRARY};${COLAMD_LIBRARY};${SUITESPARSE_CONFIG_LIBRARY}"
    )
endif()

mark_as_advanced(
    SuiteSparse_INCLUDE_DIR
    UMFPACK_LIBRARY
    AMD_LIBRARY
    CHOLMOD_LIBRARY
    COLAMD_LIBRARY
    SUITESPARSE_CONFIG_LIBRARY
)
