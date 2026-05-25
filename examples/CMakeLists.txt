# Examples CMakeLists.txt

add_executable(fem_solver
    fem_solver.cpp
)

target_link_libraries(fem_solver
    PRIVATE mpfem::problem
    PRIVATE mpfem::io
    PRIVATE mpfem::mesh
    PRIVATE mpfem::fe
)
