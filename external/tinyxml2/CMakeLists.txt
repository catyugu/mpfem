# TinyXML2 CMake configuration

cmake_minimum_required(VERSION 3.14)

project(tinyxml2 VERSION 10.0.0 LANGUAGES CXX)

# Create library
add_library(tinyxml2 STATIC
    tinyxml2.cpp
    tinyxml2.h
)

add_library(tinyxml2::tinyxml2 ALIAS tinyxml2)

target_include_directories(tinyxml2
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(tinyxml2 PUBLIC cxx_std_11)

# Set folder for IDE organization
set_target_properties(tinyxml2 PROPERTIES
    FOLDER "external"
)

# Install rules
include(GNUInstallDirs)

install(TARGETS tinyxml2
    EXPORT tinyxml2-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES tinyxml2.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT tinyxml2-targets
    FILE tinyxml2-targets.cmake
    NAMESPACE tinyxml2::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinyxml2
)
