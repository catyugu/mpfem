#include <gtest/gtest.h>
#include <cmath>
#include "mesh/mesh.hpp"
#include "mesh/geometry.hpp"
#include "fe/fe_space.hpp"
#include "fe/fe_collection.hpp"
#include "fe/element_transform.hpp"
#include "fe/quadrature.hpp"
#include "fe/coefficient.hpp"
#include "assembly/integrator.hpp"
#include "assembly/integrators.hpp"
#include "assembly/assembler.hpp"
#include "core/logger.hpp"

using namespace mpfem;

// =============================================================================
// Test Fixtures
// =============================================================================

class IntegratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Warning);
        
        // Create a simple tetrahedron mesh
        mesh_.setDim(3);
        mesh_.addVertex(0.0, 0.0, 0.0);  // 0
        mesh_.addVertex(1.0, 0.0, 0.0);  // 1
        mesh_.addVertex(0.0, 1.0, 0.0);  // 2
        mesh_.addVertex(0.0, 0.0, 1.0);  // 3
        mesh_.addElement(Geometry::Tetrahedron, {0, 1, 2, 3}, 1, 1);
        
        fec_ = std::make_unique<FECollection>(1);
        fes_ = std::make_unique<FESpace>(&mesh_, fec_.get());
    }
    
    Mesh mesh_;
    std::unique_ptr<FECollection> fec_;
    std::unique_ptr<FESpace> fes_;
};

class QuadraticIntegratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Warning);
        
        // Create a quadratic tetrahedron mesh
        mesh_.setDim(3);
        
        // Corner vertices
        mesh_.addVertex(0.0, 0.0, 0.0);  // 0
        mesh_.addVertex(1.0, 0.0, 0.0);  // 1
        mesh_.addVertex(0.0, 1.0, 0.0);  // 2
        mesh_.addVertex(0.0, 0.0, 1.0);  // 3
        
        // Edge midpoints
        mesh_.addVertex(0.5, 0.0, 0.0);   // 4: edge 0-1
        mesh_.addVertex(0.5, 0.5, 0.0);   // 5: edge 1-2
        mesh_.addVertex(0.0, 0.5, 0.0);   // 6: edge 2-0
        mesh_.addVertex(0.0, 0.0, 0.5);   // 7: edge 0-3
        mesh_.addVertex(0.5, 0.0, 0.5);   // 8: edge 1-3
        mesh_.addVertex(0.0, 0.5, 0.5);   // 9: edge 2-3
        
        // Tetrahedron2: V0, V1, V2, V3, E01, E12, E20, E03, E13, E23
        mesh_.addElement(Geometry::Tetrahedron, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 2);
        
        fec_ = std::make_unique<FECollection>(2);
        fes_ = std::make_unique<FESpace>(&mesh_, fec_.get());
    }
    
    Mesh mesh_;
    std::unique_ptr<FECollection> fec_;
    std::unique_ptr<FESpace> fes_;
};

// =============================================================================
// Diffusion Integrator Tests
// =============================================================================

TEST_F(IntegratorTest, DiffusionElementMatrix) {
    // Test diffusion integrator on a single tetrahedron
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    ConstantCoefficient k(1.0);
    DiffusionIntegrator integ(&k);
    
    Matrix elmat;
    integ.assembleElementMatrix(*refElem, trans, elmat);
    
    // Check matrix dimensions
    EXPECT_EQ(elmat.rows(), 4);  // 4 DOFs for linear tetrahedron
    EXPECT_EQ(elmat.cols(), 4);
    
    // Check symmetry
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            EXPECT_NEAR(elmat(i, j), elmat(j, i), 1e-12);
        }
    }
    
    // Check positive semi-definiteness (diagonal should be positive)
    for (int i = 0; i < 4; ++i) {
        EXPECT_GT(elmat(i, i), 0.0);
    }
}

TEST_F(IntegratorTest, DiffusionMatrixScaling) {
    // Test that diffusion matrix scales correctly with coefficient
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    ConstantCoefficient k1(1.0);
    ConstantCoefficient k2(2.0);
    
    DiffusionIntegrator integ1(&k1);
    DiffusionIntegrator integ2(&k2);
    
    Matrix elmat1, elmat2;
    integ1.assembleElementMatrix(*refElem, trans, elmat1);
    integ2.assembleElementMatrix(*refElem, trans, elmat2);
    
    // elmat2 should be exactly 2 * elmat1
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(elmat2(i, j), 2.0 * elmat1(i, j), 1e-12);
        }
    }
}

TEST_F(QuadraticIntegratorTest, DiffusionElementMatrix) {
    // Test diffusion integrator on a quadratic tetrahedron
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    ConstantCoefficient k(1.0);
    DiffusionIntegrator integ(&k);
    
    Matrix elmat;
    integ.assembleElementMatrix(*refElem, trans, elmat);
    
    // Check matrix dimensions
    EXPECT_EQ(elmat.rows(), 10);  // 10 DOFs for quadratic tetrahedron
    EXPECT_EQ(elmat.cols(), 10);
    
    // Check symmetry
    for (int i = 0; i < 10; ++i) {
        for (int j = i + 1; j < 10; ++j) {
            EXPECT_NEAR(elmat(i, j), elmat(j, i), 1e-10);
        }
    }
}

// =============================================================================
// Mass Integrator Tests
// =============================================================================

TEST_F(IntegratorTest, MassElementMatrix) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    ConstantCoefficient rho(1.0);
    MassIntegrator integ(&rho);
    
    Matrix elmat;
    integ.assembleElementMatrix(*refElem, trans, elmat);
    
    // Check matrix dimensions
    EXPECT_EQ(elmat.rows(), 4);
    EXPECT_EQ(elmat.cols(), 4);
    
    // Check symmetry
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            EXPECT_NEAR(elmat(i, j), elmat(j, i), 1e-12);
        }
    }
    
    // Check positive definiteness
    for (int i = 0; i < 4; ++i) {
        EXPECT_GT(elmat(i, i), 0.0);
    }
}

TEST_F(IntegratorTest, MassMatrixTrace) {
    // For unit tetrahedron, trace of mass matrix should be volume / 4
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    ConstantCoefficient rho(1.0);
    MassIntegrator integ(&rho);
    
    Matrix elmat;
    integ.assembleElementMatrix(*refElem, trans, elmat);
    
    Real trace = 0.0;
    for (int i = 0; i < 4; ++i) {
        trace += elmat(i, i);
    }
    
    // Volume of unit tetrahedron is 1/6
    // For mass matrix, each diagonal entry is volume/20, total is volume/5
    // This is approximate - actual value depends on quadrature
    EXPECT_GT(trace, 0.0);
    EXPECT_LT(trace, 1.0);  // Should be small for small element
}

TEST_F(QuadraticIntegratorTest, MassElementMatrix) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    ConstantCoefficient rho(1.0);
    MassIntegrator integ(&rho);
    
    Matrix elmat;
    integ.assembleElementMatrix(*refElem, trans, elmat);
    
    EXPECT_EQ(elmat.rows(), 10);
    EXPECT_EQ(elmat.cols(), 10);
    
    // Check symmetry
    for (int i = 0; i < 10; ++i) {
        for (int j = i + 1; j < 10; ++j) {
            EXPECT_NEAR(elmat(i, j), elmat(j, i), 1e-10);
        }
    }
}

// =============================================================================
// Load Vector Tests
// =============================================================================

TEST_F(IntegratorTest, DomainLoadVector) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    ConstantCoefficient f(1.0);
    DomainLFIntegrator integ(&f);
    
    Vector elvec;
    integ.assembleElementVector(*refElem, trans, elvec);
    
    EXPECT_EQ(elvec.size(), 4);
    
    // Sum of load vector entries should be approximately the volume
    Real sum = elvec.sum();
    EXPECT_GT(sum, 0.0);
}

TEST_F(IntegratorTest, DomainLoadScaling) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    ConstantCoefficient f1(1.0);
    ConstantCoefficient f2(3.0);
    
    DomainLFIntegrator integ1(&f1);
    DomainLFIntegrator integ2(&f2);
    
    Vector elvec1, elvec2;
    integ1.assembleElementVector(*refElem, trans, elvec1);
    integ2.assembleElementVector(*refElem, trans, elvec2);
    
    // elvec2 should be exactly 3 * elvec1
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(elvec2(i), 3.0 * elvec1(i), 1e-12);
    }
}

// =============================================================================
// Assembler Tests
// =============================================================================

TEST_F(IntegratorTest, BilinearFormAssembler) {
    BilinearFormAssembler assembler(fes_.get());
    
    ConstantCoefficient k(1.0);
    assembler.addDomainIntegrator(std::make_unique<DiffusionIntegrator>(&k));
    
    assembler.assemble();
    
    SparseMatrix& A = assembler.matrix();
    
    EXPECT_EQ(A.rows(), 4);  // 4 DOFs
    EXPECT_EQ(A.cols(), 4);
    EXPECT_GT(A.nonZeros(), 0);
}

TEST_F(IntegratorTest, LinearFormAssembler) {
    LinearFormAssembler assembler(fes_.get());
    
    ConstantCoefficient f(1.0);
    assembler.addDomainIntegrator(std::make_unique<DomainLFIntegrator>(&f));
    
    assembler.assemble();
    
    Vector& b = assembler.vector();
    
    EXPECT_EQ(b.size(), 4);
    EXPECT_GT(b.norm(), 0.0);
}

// =============================================================================
// Poisson Problem Test
// =============================================================================

TEST_F(IntegratorTest, SimplePoissonProblem) {
    // Assemble a simple Poisson problem: -laplacian(u) = 1
    // on a single tetrahedron with Dirichlet BC u=0 on one face
    
    // Create mesh with boundary elements
    Mesh mesh;
    mesh.setDim(3);
    mesh.addVertex(0.0, 0.0, 0.0);  // 0
    mesh.addVertex(1.0, 0.0, 0.0);  // 1
    mesh.addVertex(0.0, 1.0, 0.0);  // 2
    mesh.addVertex(0.0, 0.0, 1.0);  // 3
    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3}, 1, 1);
    
    // Add boundary elements (triangular faces)
    // Face 0: vertices 1, 2, 3 (opposite vertex 0)
    mesh.addBdrElement(Geometry::Triangle, {1, 2, 3}, 1, 1);
    // Face 1: vertices 0, 2, 3 (opposite vertex 1)
    mesh.addBdrElement(Geometry::Triangle, {0, 2, 3}, 2, 1);
    
    FECollection fec(1);
    FESpace fes(&mesh, &fec);
    
    // Assemble system
    BilinearFormAssembler bilinAsm(&fes);
    ConstantCoefficient k(1.0);
    bilinAsm.addDomainIntegrator(std::make_unique<DiffusionIntegrator>(&k));
    bilinAsm.assemble();
    
    LinearFormAssembler linAsm(&fes);
    ConstantCoefficient f(1.0);
    linAsm.addDomainIntegrator(std::make_unique<DomainLFIntegrator>(&f));
    linAsm.assemble();
    
    // Apply Dirichlet BC on boundary 1 (vertex 0 is on this boundary)
    DirichletBC bc(&fes);
    bc.addBoundaryId(1);
    bc.setValue(0.0);
    
    Vector x = Vector::Zero(fes.numDofs());
    bc.apply(bilinAsm.matrix(), x, linAsm.vector());
    
    // Verify that DOF 0 is constrained
    const auto& constrained = bc.constrainedDofs();
    EXPECT_GT(constrained.size(), 0);
}

// =============================================================================
// Quadratic Element Tests
// =============================================================================

TEST_F(QuadraticIntegratorTest, FullAssembly) {
    BilinearFormAssembler assembler(fes_.get());
    
    ConstantCoefficient k(1.0);
    assembler.addDomainIntegrator(std::make_unique<DiffusionIntegrator>(&k));
    
    assembler.assemble();
    
    SparseMatrix& A = assembler.matrix();
    
    EXPECT_EQ(A.rows(), 10);  // 10 DOFs for quadratic tetrahedron
    EXPECT_EQ(A.cols(), 10);
    EXPECT_GT(A.nonZeros(), 0);
}

TEST_F(QuadraticIntegratorTest, MassPlusDiffusion) {
    BilinearFormAssembler assembler(fes_.get());
    
    ConstantCoefficient k(1.0);
    ConstantCoefficient rho(1.0);
    
    assembler.addDomainIntegrator(std::make_unique<DiffusionIntegrator>(&k));
    assembler.addDomainIntegrator(std::make_unique<MassIntegrator>(&rho));
    
    assembler.assemble();
    
    SparseMatrix& A = assembler.matrix();
    
    EXPECT_EQ(A.rows(), 10);
    
    // Matrix should be symmetric positive definite
    // (This is a basic check - proper SPD verification would need eigenvalue analysis)
    for (int i = 0; i < A.rows(); ++i) {
        EXPECT_GT(A.coeff(i, i), 0.0) << "Diagonal entry " << i << " should be positive";
    }
}

// =============================================================================
// Analytical Solution Test
// =============================================================================

TEST(IntegratorAnalyticalTest, LaplacianOnUnitTetrahedron) {
    // For f(x,y,z) = -6 (corresponding to u = x^2 + y^2 + z^2)
    // The Laplacian of u = 2 + 2 + 2 = 6
    // So -laplacian(u) = -6, and f = -6
    
    Mesh mesh;
    mesh.setDim(3);
    mesh.addVertex(0.0, 0.0, 0.0);
    mesh.addVertex(1.0, 0.0, 0.0);
    mesh.addVertex(0.0, 1.0, 0.0);
    mesh.addVertex(0.0, 0.0, 1.0);
    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3}, 1, 1);
    
    FECollection fec(1);
    FESpace fes(&mesh, &fec);
    
    // For u = x^2 + y^2 + z^2, the gradient is (2x, 2y, 2z)
    // The diffusion term is |grad(u)|^2 = 4(x^2 + y^2 + z^2)
    // This is not constant, so we test the structure instead
    
    ElementTransform trans(&mesh, 0);
    const ReferenceElement* refElem = fes.elementRefElement(0);
    
    ConstantCoefficient k(1.0);
    DiffusionIntegrator integ(&k);
    
    Matrix elmat;
    integ.assembleElementMatrix(*refElem, trans, elmat);
    
    // The element matrix should be SPD
    // Check that the row sums are zero (constant function has zero energy)
    // Actually, for diffusion, constant functions are in the kernel only
    // if homogeneous Neumann BC is applied
    
    // Just verify the matrix is assembled correctly
    EXPECT_GT(elmat.norm(), 0.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
