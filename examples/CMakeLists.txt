# ============================================================
# mpfem Examples
# ============================================================

# Examples will be added as the library is implemented

# Helper function to create example executable
function(mpfem_add_example EXAMPLE_NAME)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}.cpp)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE mpfem)
endfunction()

# Examples will be uncommented as modules are implemented
# mpfem_add_example(run_busbar)
