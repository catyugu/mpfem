# Tests CMakeLists.txt

# Define project root directory for test data files
add_compile_definitions(MPFEM_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# =============================================================================
# Helper function to create test executable
# =============================================================================
function(mpfem_add_test name main_target)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name}
        PRIVATE ${main_target}
        PRIVATE GTest::gtest_main
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# =============================================================================
# Mesh Tests
# =============================================================================
mpfem_add_test(mpfem_test_mesh mpfem::mesh test_mesh_read.cpp)

# =============================================================================
# Quadrature Tests
# =============================================================================
mpfem_add_test(mpfem_test_quadrature mpfem::fe test_quadrature.cpp)

# =============================================================================
# Shape Function Tests
# =============================================================================
mpfem_add_test(mpfem_test_shape_function mpfem::fe test_shape_function.cpp)

# =============================================================================
# Element Transform Tests
# =============================================================================
mpfem_add_test(mpfem_test_element_transform mpfem::fe test_element_transform.cpp)

# =============================================================================
# FE Space Tests
# =============================================================================
mpfem_add_test(mpfem_test_fe_space mpfem::fe test_fe_space.cpp)

# =============================================================================
# IO Tests
# =============================================================================
mpfem_add_test(mpfem_test_value_parser mpfem::io test_value_parser.cpp)
mpfem_add_test(mpfem_test_case_xml_reader mpfem::io test_case_xml_reader.cpp)
mpfem_add_test(mpfem_test_material_xml_reader mpfem::io test_material_xml_reader.cpp)

# =============================================================================
# Grid Function Tests
# =============================================================================
mpfem_add_test(mpfem_test_grid_function mpfem::fe test_grid_function.cpp)

# =============================================================================
# Curved Element Tests (Second-order geometry)
# =============================================================================
mpfem_add_test(mpfem_test_curved_element mpfem::fe test_curved_element.cpp)

# =============================================================================
# Quadratic FE Space Tests
# =============================================================================
mpfem_add_test(mpfem_test_fe_space_quadratic mpfem::fe test_fe_space_quadratic.cpp)

# =============================================================================
# Coefficient Tests (includes order 1 and order 2 tests)
# =============================================================================
mpfem_add_test(mpfem_test_coefficient mpfem::fe test_coefficient.cpp)

# =============================================================================
# Integrator Tests (Assembly module)
# =============================================================================
mpfem_add_test(mpfem_test_integrator mpfem::assembly test_integrator.cpp)

# =============================================================================
# Electrostatics Solver Tests (Physics module)
# =============================================================================
mpfem_add_test(mpfem_test_electrostatics_solver mpfem::physics test_electrostatics_solver.cpp)
