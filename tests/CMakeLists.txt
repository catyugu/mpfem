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