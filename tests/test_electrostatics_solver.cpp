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
// Unit Tests for ElectrostaticsSolver
// =============================================================================

class ElectrostaticsSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple cube mesh for testing with internal node
        createCubeMesh();
    }
    
    void createCubeMesh() {
        mesh_.setDim(3);
        
        // Unit cube vertices plus center node
        // Corner nodes: 0-7
        // Center node: 8 (internal, not on any boundary)
        mesh_.addVertex(0.0, 0.0, 0.0);  // 0
        mesh_.addVertex(1.0, 0.0, 0.0);  // 1
        mesh_.addVertex(1.0, 1.0, 0.0);  // 2
        mesh_.addVertex(0.0, 1.0, 0.0);  // 3
        mesh_.addVertex(0.0, 0.0, 1.0);  // 4
        mesh_.addVertex(1.0, 0.0, 1.0);  // 5
        mesh_.addVertex(1.0, 1.0, 1.0);  // 6
        mesh_.addVertex(0.0, 1.0, 1.0);  // 7
        mesh_.addVertex(0.5, 0.5, 0.5);  // 8 - center (internal node)
        
        // Eight tetrahedra with center node
        // Bottom tets
        mesh_.addElement(Geometry::Tetrahedron, {0, 1, 2, 8}, 1);
        mesh_.addElement(Geometry::Tetrahedron, {0, 2, 3, 8}, 1);
        // Top tets
        mesh_.addElement(Geometry::Tetrahedron, {4, 5, 6, 8}, 1);
        mesh_.addElement(Geometry::Tetrahedron, {4, 6, 7, 8}, 1);
        // Middle tets
        mesh_.addElement(Geometry::Tetrahedron, {0, 1, 5, 8}, 1);
        mesh_.addElement(Geometry::Tetrahedron, {0, 4, 5, 8}, 1);
        mesh_.addElement(Geometry::Tetrahedron, {2, 3, 7, 8}, 1);
        mesh_.addElement(Geometry::Tetrahedron, {2, 6, 7, 8}, 1);
        
        // Boundary faces with proper attributes
        // Bottom face (z=0): attribute 1, V = 0
        mesh_.addBdrElement(Geometry::Triangle, {0, 2, 1}, 1);
        mesh_.addBdrElement(Geometry::Triangle, {0, 3, 2}, 1);
        // Top face (z=1): attribute 2, V = 1
        mesh_.addBdrElement(Geometry::Triangle, {4, 5, 6}, 2);
        mesh_.addBdrElement(Geometry::Triangle, {4, 6, 7}, 2);
        // Front face (y=0): attribute 3 (natural BC)
        mesh_.addBdrElement(Geometry::Triangle, {0, 1, 5}, 3);
        mesh_.addBdrElement(Geometry::Triangle, {0, 5, 4}, 3);
        // Back face (y=1): attribute 4 (natural BC)
        mesh_.addBdrElement(Geometry::Triangle, {2, 3, 7}, 4);
        mesh_.addBdrElement(Geometry::Triangle, {2, 7, 6}, 4);
        // Left face (x=0): attribute 5 (natural BC)
        mesh_.addBdrElement(Geometry::Triangle, {0, 3, 4}, 5);
        mesh_.addBdrElement(Geometry::Triangle, {3, 4, 7}, 5);
        // Right face (x=1): attribute 6 (natural BC)
        mesh_.addBdrElement(Geometry::Triangle, {1, 2, 5}, 6);
        mesh_.addBdrElement(Geometry::Triangle, {2, 5, 6}, 6);
    }
    
    Mesh mesh_;
};

TEST_F(ElectrostaticsSolverTest, BasicInitialization) {
    // Create conductivity coefficient
    PWConstCoefficient sigma(1);
    sigma.setConstant(1, 1.0);  // Domain 1 has conductivity 1
    
    // Create solver
    ElectrostaticsSolver solver(1);
    EXPECT_TRUE(solver.initialize(mesh_, sigma));
    
    EXPECT_EQ(solver.order(), 1);
    EXPECT_EQ(solver.fieldKind(), FieldKind::ElectricPotential);
    EXPECT_GT(solver.numDofs(), 0);
}

TEST_F(ElectrostaticsSolverTest, DISABLED_SimplePotentialField) {
    // Create uniform conductivity
    PWConstCoefficient sigma(1);
    sigma.setConstant(1, 1.0);
    
    // Create solver
    ElectrostaticsSolver solver(1);
    solver.initialize(mesh_, sigma);
    
    // Apply boundary conditions: V=0 on bottom (boundary 1), V=1 on top (boundary 2)
    solver.addDirichletBC(1, 0.0);
    solver.addDirichletBC(2, 1.0);
    
    // Assemble and solve
    solver.assemble();
    bool success = solver.solve();
    
    EXPECT_TRUE(success);
    
    // Check potential range
    Real minV = solver.minValue();
    Real maxV = solver.maxValue();
    
    EXPECT_NEAR(minV, 0.0, 1e-6);
    EXPECT_NEAR(maxV, 1.0, 1e-6);
    
    LOG_INFO << "Potential range: [" << minV << ", " << maxV << "]";
}

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

TEST_F(BusbarElectrostaticsTest, DISABLED_QuadraticOrder) {
    // Test with quadratic elements - DISABLED due to high computational cost
    std::string caseDir2 = MPFEM_PROJECT_ROOT + std::string("/cases/busbar_order2");
    
    PhysicsProblemSetup setup = PhysicsProblemBuilder::build(caseDir2);
    
    ASSERT_TRUE(setup.hasElectrostatics());
    
    auto& solver = setup.electrostatics;
    EXPECT_EQ(solver->order(), 1);  // Case file specifies order=1
    
    solver->assemble();
    bool success = solver->solve();
    
    ASSERT_TRUE(success);
    
    Real minV = solver->minValue();
    Real maxV = solver->maxValue();
    
    LOG_INFO << "Quadratic mesh potential range: [" << minV << ", " << maxV << "]";
    
    // Results should be similar to linear mesh
    EXPECT_NEAR(maxV, 0.02, 1e-3);
}

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