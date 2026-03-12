# Tests CMakeLists.txt

# =============================================================================
# Mesh Tests
# =============================================================================
add_executable(mpfem_test_mesh
    test_mesh_read.cpp
)

target_link_libraries(mpfem_test_mesh
    PRIVATE mpfem::mesh
    PRIVATE GTest::gtest_main
)

add_test(NAME MeshRead COMMAND mpfem_test_mesh)

# =============================================================================
# Quadrature Tests
# =============================================================================
add_executable(mpfem_test_quadrature
    test_quadrature.cpp
)

target_link_libraries(mpfem_test_quadrature
    PRIVATE mpfem::fe
    PRIVATE GTest::gtest_main
)

add_test(NAME Quadrature COMMAND mpfem_test_quadrature)

# =============================================================================
# Shape Function Tests
# =============================================================================
add_executable(mpfem_test_shape_function
    test_shape_function.cpp
)

target_link_libraries(mpfem_test_shape_function
    PRIVATE mpfem::fe
    PRIVATE GTest::gtest_main
)

add_test(NAME ShapeFunction COMMAND mpfem_test_shape_function)

# =============================================================================
# Element Transform Tests
# =============================================================================
add_executable(mpfem_test_element_transform
    test_element_transform.cpp
)

target_link_libraries(mpfem_test_element_transform
    PRIVATE mpfem::mesh
    PRIVATE mpfem::fe
    PRIVATE GTest::gtest_main
)

add_test(NAME ElementTransform COMMAND mpfem_test_element_transform)

# =============================================================================
# FE Space Tests
# =============================================================================
add_executable(mpfem_test_fe_space
    test_fe_space.cpp
)

target_link_libraries(mpfem_test_fe_space
    PRIVATE mpfem::mesh
    PRIVATE mpfem::fe
    PRIVATE GTest::gtest_main
)

add_test(NAME FESpace COMMAND mpfem_test_fe_space)

# =============================================================================
# IO Tests
# =============================================================================
add_executable(mpfem_test_value_parser
    test_value_parser.cpp
)

target_link_libraries(mpfem_test_value_parser
    PRIVATE mpfem::io
    PRIVATE GTest::gtest_main
)

add_test(NAME ValueParser COMMAND mpfem_test_value_parser)

add_executable(mpfem_test_case_xml_reader
    test_case_xml_reader.cpp
)

target_link_libraries(mpfem_test_case_xml_reader
    PRIVATE mpfem::io
    PRIVATE mpfem::mesh
    PRIVATE GTest::gtest_main
)

add_test(NAME CaseXmlReader COMMAND mpfem_test_case_xml_reader)

add_executable(mpfem_test_material_xml_reader
    test_material_xml_reader.cpp
)

target_link_libraries(mpfem_test_material_xml_reader
    PRIVATE mpfem::io
    PRIVATE GTest::gtest_main
)

add_test(NAME MaterialXmlReader COMMAND mpfem_test_material_xml_reader)

# =============================================================================
# Grid Function Tests
# =============================================================================
add_executable(mpfem_test_grid_function
    test_grid_function.cpp
)

target_link_libraries(mpfem_test_grid_function
    PRIVATE mpfem::fe
    PRIVATE GTest::gtest_main
)

add_test(NAME GridFunction COMMAND mpfem_test_grid_function)

# =============================================================================
# FE Values Tests
# =============================================================================
add_executable(mpfem_test_fe_values
    test_fe_values.cpp
)

target_link_libraries(mpfem_test_fe_values
    PRIVATE mpfem::fe
    PRIVATE GTest::gtest_main
)

add_test(NAME FEValues COMMAND mpfem_test_fe_values)