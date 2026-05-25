# Repository Guidelines

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
├── examples/          # fem_solver.cpp
├── cases/             # busbar_steady/, busbar_steady_order2/, busbar_large/, busbar_transient/
└── cmake/             # Dependencies.cmake, Targets.cmake, CPM.cmake
```

## Coding Style & Naming Conventions

- **Namespace**: `mpfem::`
- **Limited OOP**: No hierarchical inheritance. Use OOP features like inheritance and polymorphism only for sharing interface. Use DOP for most cases.
- **PIMPL pattern**: PIMPL is considerable for decoupling and for reducing compilation dependency.
- **No dynamic_cast**: Always use virtual `configure()` method for parameter injection
- **No Backward Compatibility**: Strictly forbid anything remained for backward-compatibility.
- **No Cross Dependency**: Any two files or modules/libs shall not rely on each other. Dependencies shall only happen on single direction.

## Activity Tracking (Required)

- Summaries should include what changed, files touched, and any notable decisions.
- Use the scratchpad tool for follow-ups or TODOs discovered during work.

## Build, Test, and Development Commands

- **Build Config**: C++20, MSVC `/W4 /WX /permissive- /utf-8 /bigobj`, Clang `-Werror -Wall -Wextra -Wpedantic`

```bash
# Build
conda activate numerical
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run busbar example
conda activate numerical
build/examples/fem_solver.exe ./cases/busbar_steady_order2

# Run ctest
ctest --test-dir build
```

## Testing Guidelines

- Enforce TDD for every behavior change: follow `red -> green -> refactor`.
- Start by establishing a verifiable baseline: run the relevant existing tests before edits, and record the exact command + outcome in the PR/commit notes.
- Add or update a failing test first that reproduces the bug or captures the new requirement; implement code only after the test fails for the expected reason.
- Keep tests green after implementation and after any refactor; do not merge with skipped failing tests.
- Every bug fix must include a regression test that fails before the fix and passes after it.
- Prefer behavior-focused assertions (pipeline run successfully, results are expected...).

## Commit & Pull Request Guidelines

- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`) and keep messages imperative.
- PRs: include a short summary, exact test command(s) run, and call out any changes to on-disk memory formats or `qmd` behavior.

## OTHER DOCUMENTS

- [RULES](doc/RULES.md): The rules. MUST BE FOLLOWED STRICTLY.
- [CASES](doc/CASES.md): The cases description for validation.
- [VALIDATION](doc/VALIDATION.md): The validation workflow and standard.
