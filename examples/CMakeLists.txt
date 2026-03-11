# Examples CMakeLists.txt

add_executable(busbar_example
    busbar_example.cpp
)

target_link_libraries(busbar_example
    PRIVATE mpfem::mesh
    PRIVATE mpfem::fe
)

# Add more examples as needed
