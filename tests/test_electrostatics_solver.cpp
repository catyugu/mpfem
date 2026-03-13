#include <gtest/gtest.h>
#include "physics/electrostatics_solver.hpp"
#include "physics/physics_problem_builder.hpp"
#include "mesh/mesh.hpp"
#include "fe/coefficient.hpp"
#include "core/logger.hpp"
#include <cmath>
#include <fstream>

using namespace mpfem;

// =============================================================================
// Integration Test: Busbar Case
// =============================================================================

class BusbarElectrostaticsTest : public ::testing::Test {
protected:
    void SetUp() override {
        caseDir_ = MPFEM_PROJECT_ROOT + std::string("/cases/busbar");
    }
    
    std::string caseDir_;
};

// =============================================================================
// Convergence and Performance Tests
// =============================================================================

TEST_F(BusbarElectrostaticsTest, SolverConvergence) {
    PhysicsProblemSetup setup = PhysicsProblemBuilder::build(caseDir_);
    auto& solver = setup.electrostatics;
    
    solver->assemble();
    solver->solve();
    
    int iterations = solver->iterations();
    Real residual = solver->residual();
    
    LOG_INFO << "Solver converged in " << iterations << " iterations, residual = " << residual;
    
    // Direct solver should converge in 0 iterations reported
    // Iterative solver should converge within tolerance
    EXPECT_LT(residual, 1e-6);
    
    // Verify potential range
    Real minV = solver->minValue();
    Real maxV = solver->maxValue();
    
    LOG_INFO << "Potential range: [" << minV << ", " << maxV << "]";
    
    // Potential should be between 0 and 0.02V
    EXPECT_GE(minV, -1e-6) << "Minimum potential should be >= 0";
    EXPECT_LE(maxV, 0.02 + 1e-6) << "Maximum potential should be <= 0.02V";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}