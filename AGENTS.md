# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-23
**Commit:** ace1ef3 (refactor)
**Branch:** refactor

## OVERVIEW

Multi-Physics Finite Element Method (MPFEM) library for electrostatics, heat transfer, and structural mechanics. C++20 with Eigen for linear algebra. Supports steady/transient analysis with coupled physics (Joule heating, thermal expansion).

## STRUCTURE

```
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
- **Class style**: Concrete implementations `final`, abstract bases with virtual destructors
- **PIMPL pattern**: PIMPL is great for decoupling and for reducing compilation dependency.
- **Conditional compilation**: `#ifdef MPFEM_USE_MKL`, `#ifdef MPFEM_USE_UMFPACK`
- **No dynamic_cast**: Virtual `configure()` method for parameter injection
- **No Backward Compatibility**: Strictly forbid anything remained for backward-compatibility.
- **No Cross Dependency**: Any two files or modules/libs shall not rely on each other. Dependencies shall only happen on single direction.

## COMMANDS

```bash
# MSVC build
cmd /c "call E:\env\cpp\VS14\Common7\Tools\VsDevCmd.bat & cmake -S . -B build-msvc & cmake --build build-msvc --parallel --config Release"

# Run busbar example
build-msvc/examples/Release/busbar_example.exe ./cases/busbar_steady_order2
```

## BUILD CONFIG

- C++20, MSVC `/W4 /WX /permissive- /utf-8 /bigobj`, Clang `-Werror -Wall -Wextra -Wpedantic`
- VCPKG for dependency management (Eigen3, MKL, SuiteSparse, tinyxml2)
- Precompiled headers: `<vector>`, `<memory>`, `<string>`, `Eigen/Core`, `Eigen/SparseCore`

## OTHER DOCUMENTS

- [RULES](doc/RULES.md): The rules. MUST BE FOLLOWED STRICTLY.
- [CASES](doc/CASES.md): The cases description for validation.
- [VALIDATION](doc/VALIDATION.md): The validation workflow and standard.
- [WORK](doc/WORK.md): The current work tasks.