# Tests CMakeLists.txt

add_compile_definitions(MPFEM_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")

foreach(test_src IN LISTS TEST_SOURCES)
    get_filename_component(test_name "${test_src}" NAME_WE)
    add_executable(${test_name} "${test_src}")
    target_link_libraries(${test_name} PRIVATE mpfem::problem GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
