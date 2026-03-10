# ============================================================
# mpfem Tests
# ============================================================

# Test sources will be added as modules are implemented
# Each test file is a separate executable

# Helper function to create test executable
function(mpfem_add_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME}
        PRIVATE
            mpfem
            GTest::gtest
            GTest::gtest_main
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
    # Set working directory to project root for test case access
    set_tests_properties(${TEST_NAME} PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endfunction()

# Mesh test
mpfem_add_test(test_mesh)

# FEM test
mpfem_add_test(test_fem)

# DoF test
mpfem_add_test(test_dof)

# Tests will be uncommented as modules are implemented
# mpfem_add_test(test_types)
# mpfem_add_test(test_fe)
# mpfem_add_test(test_dof)
# mpfem_add_test(test_assembly)
# mpfem_add_test(test_solver)
# mpfem_add_test(test_physics)
