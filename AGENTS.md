# PROJECT KNOWLEDGE BASE

## OVERVIEW

Multi-Physics Finite Element Method (MPFEM) library for electrostatics, heat transfer, and structural mechanics. C++20 with Eigen for linear algebra. Supports steady/transient analysis with coupled physics (Joule heating, thermal expansion).

## STRUCTURE

```bash
./
├── src/
│   ├── core/          # Logger, exceptions, types, tensor, sparse_matrix
│   ├── mesh/          # Mesh class, element/face/edge data
│   ├── expr/          # Expression parser (Pratt), VariableNode, unit handling
│   ├── fe/            # Finite elements: H1, ND, quadrature, transforms
│   ├── field/          # FE spaces, grid functions, field values
│   ├── assembly/       # BilinearFormAssembler, integrators, Dirichlet BC
│   ├── solver/         # LinearOperator base, factory, MKL/UMFPACK/Eigen solvers
│   ├── io/             # XML readers, material DB, COMSOL/VTK export
│   ├── physics/        # PhysicsFieldSolver base, electrostatics/heat/structural
│   └── problem/        # Problem class, transient, physics builder
├── tests/             # 17 gtest test files
├── examples/          # busbar_example.cpp
├── cases/             # busbar_steady/, busbar_steady_order2/, busbar_large/, busbar_transient/
└── cmake/             # Dependencies.cmake, Targets.cmake, CPM.cmake
```

## CONVENTIONS AND MANDATORY RULES

- **Namespace**: `mpfem::`
- **Limited OOP**: No hierarchical inheritance. Use OOP features like inheritance and polymorphism only for sharing interface. Use DOP for most cases.
- **PIMPL pattern**: PIMPL is considerable for decoupling and for reducing compilation dependency.
- **No dynamic_cast**: Always use virtual `configure()` method for parameter injection
- **No Backward Compatibility**: Strictly forbid anything remained for backward-compatibility.
- **No Cross Dependency**: Any two files or modules/libs shall not rely on each other. Dependencies shall only happen on single direction.

## COMMANDS

```bash
# Build
conda activate numerical
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run busbar example
conda activate numerical
build/examples/busbar_example.exe ./cases/busbar_steady_order2
```

## BUILD CONFIG

- C++20, MSVC `/W4 /WX /permissive- /utf-8 /bigobj`, Clang `-Werror -Wall -Wextra -Wpedantic`
- Precompiled headers: `<vector>`, `<memory>`, `<string>`, `Eigen/Core`, `Eigen/SparseCore`

## OTHER DOCUMENTS

- [RULES](doc/RULES.md): The rules. MUST BE FOLLOWED STRICTLY.
- [CASES](doc/CASES.md): The cases description for validation.
- [VALIDATION](doc/VALIDATION.md): The validation workflow and standard.
- [WORK](doc/WORK.md): The current work tasks.
