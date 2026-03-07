set(MPFEM_SOURCES
  mesh/comsol_mesh_reader.cpp
  material/comsol_material_reader.cpp
  config/problem_config_loader.cpp
  ../external/mfem/general/tinyxml2.cpp
)

add_library(mpfem ${MPFEM_SOURCES})
add_library(mpfem::mpfem ALIAS mpfem)

target_include_directories(mpfem
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/mfem/general
)

if(TARGET mfem)
  target_link_libraries(mpfem PUBLIC mfem)
endif()

mpfem_set_default_compiler_options(mpfem)
