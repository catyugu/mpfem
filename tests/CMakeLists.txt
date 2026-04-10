# Tests CMakeLists.txt

# Define project root directory for test data files
add_compile_definitions(MPFEM_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# =============================================================================
# Helper function to create test executable
# =============================================================================
function(mpfem_add_test name main_target)
    add_executable(${name} ${main_target})
    target_link_libraries(${name}
        PRIVATE ${ARGN}
        PRIVATE GTest::gtest_main
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# =============================================================================
# Mesh Tests
# =============================================================================
mpfem_add_test(mpfem_test_mesh test_mesh_read.cpp  mpfem::mesh mpfem::io)

# =============================================================================
# Quadrature Tests
# =============================================================================
mpfem_add_test(mpfem_test_quadrature test_quadrature.cpp mpfem::fe )

# =============================================================================
# Shape Function Tests
# =============================================================================
mpfem_add_test(mpfem_test_shape_function test_shape_function.cpp mpfem::fe)

# =============================================================================
# Element Transform Tests
# =============================================================================
mpfem_add_test(mpfem_test_element_transform test_element_transform.cpp mpfem::fe)

# =============================================================================
# FE Space Tests
# =============================================================================
mpfem_add_test(mpfem_test_fe_space test_fe_space.cpp mpfem::fe)

# =============================================================================
# IO Tests
# =============================================================================
mpfem_add_test(mpfem_test_case_xml_reader test_case_xml_reader.cpp mpfem::io)
mpfem_add_test(mpfem_test_material_xml_reader test_material_xml_reader.cpp mpfem::io)

# =============================================================================
# Grid Function Tests
# =============================================================================
mpfem_add_test(mpfem_test_grid_function test_grid_function.cpp mpfem::fe)

# =============================================================================
# Curved Element Tests (Second-order geometry)
# =============================================================================
mpfem_add_test(mpfem_test_curved_element test_curved_element.cpp mpfem::fe)

# =============================================================================
# Quadratic FE Space Tests
# =============================================================================
mpfem_add_test(mpfem_test_fe_space_quadratic test_fe_space_quadratic.cpp mpfem::fe mpfem::io)

# =============================================================================
# Variable Tests (includes order 1 and order 2 tests)
# =============================================================================
mpfem_add_test(mpfem_test_variable test_variable.cpp mpfem::problem)

# =============================================================================
# Integrator Tests (Assembly module)
# =============================================================================
mpfem_add_test(mpfem_test_integrator test_integrator.cpp mpfem::assembly)

# =============================================================================
# Electrostatics Solver Tests (Physics module)
# =============================================================================
mpfem_add_test(mpfem_test_electrostatics_solver test_electrostatics_solver.cpp mpfem::problem)

# =============================================================================
# Literal Tests (BracketLiteral - expr module)
# =============================================================================
mpfem_add_test(mpfem_test_literal test_literal.cpp mpfem::expr)
