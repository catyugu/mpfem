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
        
        mesh_.setDim(3);
        mesh_.addVertex(0.0, 0.0, 0.0);
        mesh_.addVertex(1.0, 0.0, 0.0);
        mesh_.addVertex(0.0, 1.0, 0.0);
        mesh_.addVertex(0.0, 0.0, 1.0);
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
        
        mesh_.setDim(3);
        
        // Corner vertices
        mesh_.addVertex(0.0, 0.0, 0.0);
        mesh_.addVertex(1.0, 0.0, 0.0);
        mesh_.addVertex(0.0, 1.0, 0.0);
        mesh_.addVertex(0.0, 0.0, 1.0);
        
        // Edge midpoints
        mesh_.addVertex(0.5, 0.0, 0.0);
        mesh_.addVertex(0.5, 0.5, 0.0);
        mesh_.addVertex(0.0, 0.5, 0.0);
        mesh_.addVertex(0.0, 0.0, 0.5);
        mesh_.addVertex(0.5, 0.0, 0.5);
        mesh_.addVertex(0.0, 0.5, 0.5);
        
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
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    auto k = std::make_shared<ConstantCoefficient>(1.0);
    DiffusionIntegrator integ(k);
    
    Matrix elmat;
    integ.assembleElementMatrix(*refElem, trans, elmat);
    
    EXPECT_EQ(elmat.rows(), 4);
    EXPECT_EQ(elmat.cols(), 4);
    
    // Check symmetry
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            EXPECT_NEAR(elmat(i, j), elmat(j, i), 1e-12);
        }
    }
    
    // Check diagonal positive
    for (int i = 0; i < 4; ++i) {
        EXPECT_GT(elmat(i, i), 0.0);
    }
}

TEST_F(IntegratorTest, DiffusionMatrixScaling) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    auto k1 = std::make_shared<ConstantCoefficient>(1.0);
    auto k2 = std::make_shared<ConstantCoefficient>(2.0);
    
    DiffusionIntegrator integ1(k1);
    DiffusionIntegrator integ2(k2);
    
    Matrix elmat1, elmat2;
    integ1.assembleElementMatrix(*refElem, trans, elmat1);
    integ2.assembleElementMatrix(*refElem, trans, elmat2);
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(elmat2(i, j), 2.0 * elmat1(i, j), 1e-12);
        }
    }
}

TEST_F(QuadraticIntegratorTest, DiffusionElementMatrix) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    auto k = std::make_shared<ConstantCoefficient>(1.0);
    DiffusionIntegrator integ(k);
    
    Matrix elmat;
    integ.assembleElementMatrix(*refElem, trans, elmat);
    
    EXPECT_EQ(elmat.rows(), 10);
    EXPECT_EQ(elmat.cols(), 10);
    
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
    
    auto rho = std::make_shared<ConstantCoefficient>(1.0);
    MassIntegrator integ(rho);
    
    Matrix elmat;
    integ.assembleElementMatrix(*refElem, trans, elmat);
    
    EXPECT_EQ(elmat.rows(), 4);
    EXPECT_EQ(elmat.cols(), 4);
    
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            EXPECT_NEAR(elmat(i, j), elmat(j, i), 1e-12);
        }
    }
    
    for (int i = 0; i < 4; ++i) {
        EXPECT_GT(elmat(i, i), 0.0);
    }
}

TEST_F(QuadraticIntegratorTest, MassElementMatrix) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    auto rho = std::make_shared<ConstantCoefficient>(1.0);
    MassIntegrator integ(rho);
    
    Matrix elmat;
    integ.assembleElementMatrix(*refElem, trans, elmat);
    
    EXPECT_EQ(elmat.rows(), 10);
    EXPECT_EQ(elmat.cols(), 10);
    
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
    
    auto f = std::make_shared<ConstantCoefficient>(1.0);
    DomainLFIntegrator integ(f);
    
    Vector elvec;
    integ.assembleElementVector(*refElem, trans, elvec);
    
    EXPECT_EQ(elvec.size(), 4);
    EXPECT_GT(elvec.sum(), 0.0);
}

TEST_F(IntegratorTest, DomainLoadScaling) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    auto f1 = std::make_shared<ConstantCoefficient>(1.0);
    auto f2 = std::make_shared<ConstantCoefficient>(3.0);
    
    DomainLFIntegrator integ1(f1);
    DomainLFIntegrator integ2(f2);
    
    Vector elvec1, elvec2;
    integ1.assembleElementVector(*refElem, trans, elvec1);
    integ2.assembleElementVector(*refElem, trans, elvec2);
    
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(elvec2(i), 3.0 * elvec1(i), 1e-12);
    }
}

// =============================================================================
// Assembler Tests
// =============================================================================

TEST_F(IntegratorTest, BilinearFormAssembler) {
    BilinearFormAssembler assembler(fes_.get());
    
    auto k = std::make_shared<ConstantCoefficient>(1.0);
    auto integ = std::make_unique<DiffusionIntegrator>(k);
    assembler.addDomainIntegrator(std::move(integ));
    
    assembler.assemble();
    
    SparseMatrix& A = assembler.matrix();
    
    EXPECT_EQ(A.rows(), 4);
    EXPECT_EQ(A.cols(), 4);
    EXPECT_GT(A.nonZeros(), 0);
}

TEST_F(IntegratorTest, LinearFormAssembler) {
    LinearFormAssembler assembler(fes_.get());
    
    auto f = std::make_shared<ConstantCoefficient>(1.0);
    auto integ = std::make_unique<DomainLFIntegrator>(f);
    assembler.addDomainIntegrator(std::move(integ));
    
    assembler.assemble();
    
    Vector& b = assembler.vector();
    
    EXPECT_EQ(b.size(), 4);
    EXPECT_GT(b.norm(), 0.0);
}

// =============================================================================
// Dirichlet BC Tests
// =============================================================================

TEST_F(IntegratorTest, DirichletBCElimination) {
    Mesh mesh;
    mesh.setDim(3);
    mesh.addVertex(0.0, 0.0, 0.0);
    mesh.addVertex(1.0, 0.0, 0.0);
    mesh.addVertex(0.0, 1.0, 0.0);
    mesh.addVertex(0.0, 0.0, 1.0);
    mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3}, 1, 1);
    mesh.addBdrElement(Geometry::Triangle, {1, 2, 3}, 1, 1);
    mesh.addBdrElement(Geometry::Triangle, {0, 2, 3}, 2, 1);
    
    FECollection fec(1);
    FESpace fes(&mesh, &fec);
    
    auto k = std::make_shared<ConstantCoefficient>(1.0);
    auto f = std::make_shared<ConstantCoefficient>(1.0);
    
    BilinearFormAssembler bilinAsm(&fes);
    bilinAsm.addDomainIntegrator(std::make_unique<DiffusionIntegrator>(k));
    bilinAsm.assemble();
    
    LinearFormAssembler linAsm(&fes);
    linAsm.addDomainIntegrator(std::make_unique<DomainLFIntegrator>(f));
    linAsm.assemble();
    
    DirichletBC bc(&fes);
    bc.addBoundaryId(1);
    bc.setValue(0.0);
    
    Vector x = Vector::Zero(fes.numDofs());
    bc.apply(bilinAsm.matrix(), x, linAsm.vector());
    
    const auto& constrained = bc.constrainedDofs();
    EXPECT_GT(constrained.size(), 0);
}

// =============================================================================
// Quadratic Assembly Tests
// =============================================================================

TEST_F(QuadraticIntegratorTest, FullAssembly) {
    BilinearFormAssembler assembler(fes_.get());
    
    auto k = std::make_shared<ConstantCoefficient>(1.0);
    assembler.addDomainIntegrator(std::make_unique<DiffusionIntegrator>(k));
    
    assembler.assemble();
    
    SparseMatrix& A = assembler.matrix();
    
    EXPECT_EQ(A.rows(), 10);
    EXPECT_EQ(A.cols(), 10);
    EXPECT_GT(A.nonZeros(), 0);
}

TEST_F(QuadraticIntegratorTest, MassPlusDiffusion) {
    BilinearFormAssembler assembler(fes_.get());
    
    auto k = std::make_shared<ConstantCoefficient>(1.0);
    auto rho = std::make_shared<ConstantCoefficient>(1.0);
    
    assembler.addDomainIntegrator(std::make_unique<DiffusionIntegrator>(k));
    assembler.addDomainIntegrator(std::make_unique<MassIntegrator>(rho));
    
    assembler.assemble();
    
    SparseMatrix& A = assembler.matrix();
    
    EXPECT_EQ(A.rows(), 10);
    
    for (int i = 0; i < A.rows(); ++i) {
        EXPECT_GT(A.coeff(i, i), 0.0);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}