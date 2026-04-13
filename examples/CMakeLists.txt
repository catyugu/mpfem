# Examples CMakeLists.txt

add_executable(busbar_example
    busbar_example.cpp
)

target_link_libraries(busbar_example
    PRIVATE mpfem::problem
    PRIVATE mpfem::io
    PRIVATE mpfem::mesh
    PRIVATE mpfem::fe
)
