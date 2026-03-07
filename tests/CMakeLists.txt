add_executable(mpfem_busbar_case_test busbar_case_test.cpp)
target_link_libraries(mpfem_busbar_case_test PRIVATE mpfem::mpfem)
mpfem_set_default_compiler_options(mpfem_busbar_case_test)

add_test(
  NAME mpfem.busbar.case
  COMMAND mpfem_busbar_case_test ${CMAKE_SOURCE_DIR}
)
