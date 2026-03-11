# Tests CMakeLists.txt

add_executable(mpfem_test_mesh
    test_mesh_read.cpp
)

target_link_libraries(mpfem_test_mesh
    PRIVATE mpfem::mesh
    PRIVATE GTest::gtest_main
)

add_test(NAME MeshRead COMMAND mpfem_test_mesh)

# Add more tests as needed
