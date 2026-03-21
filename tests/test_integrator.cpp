#include <gtest/gtest.h>
#include <cmath>
#include "mesh/mesh.hpp"
#include "mesh/geometry.hpp"
#include "fe/fe_space.hpp"
#include "fe/fe_collection.hpp"
#include "fe/element_transform.hpp"
#include "fe/quadrature.hpp"
#include "fe/coefficient.hpp"
#include "assembly/integrators.hpp"
#include "assembly/assembler.hpp"
#include "core/logger.hpp"

using namespace mpfem;

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
        
        fes_ = std::make_unique<FESpace>(&mesh_, std::make_unique<FECollection>(1));
        
        // Use new coefficient helper functions
        k1_ = constantCoefficient(1.0);
        k2_ = constantCoefficient(2.0);
        
        // Matrix coefficients for diffusion integrator
        mat1_ = diagonalMatrixCoefficient(1.0);
        mat2_ = diagonalMatrixCoefficient(2.0);
    }
    
    Mesh mesh_;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<Coefficient> k1_;
    std::unique_ptr<Coefficient> k2_;
    std::unique_ptr<MatrixCoefficient> mat1_;
    std::unique_ptr<MatrixCoefficient> mat2_;
};

TEST_F(IntegratorTest, DiffusionElementMatrix) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    DiffusionIntegrator integ(mat1_.get());
    
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
    
    DiffusionIntegrator integ1(mat1_.get());
    DiffusionIntegrator integ2(mat2_.get());
    
    Matrix elmat1, elmat2;
    integ1.assembleElementMatrix(*refElem, trans, elmat1);
    integ2.assembleElementMatrix(*refElem, trans, elmat2);
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(elmat2(i, j), 2.0 * elmat1(i, j), 1e-12);
        }
    }
}

TEST_F(IntegratorTest, AnisotropicDiffusion) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    // Create anisotropic matrix coefficient
    Matrix3 D = Matrix3::Identity() * 2.0;  // Diagonal = 2
    auto matCoef = constantMatrixCoefficient(D);
    
    // DiffusionIntegrator now handles both isotropic and anisotropic cases
    DiffusionIntegrator integ(matCoef.get());
    
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
    
    // Diagonal should be positive
    for (int i = 0; i < 4; ++i) {
        EXPECT_GT(elmat(i, i), 0.0);
    }
}

TEST_F(IntegratorTest, MassElementMatrix) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    MassIntegrator integ(k1_.get());
    
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

TEST_F(IntegratorTest, DomainLoadVector) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    DomainLFIntegrator integ(k1_.get());
    
    Vector elvec;
    integ.assembleElementVector(*refElem, trans, elvec);
    
    EXPECT_EQ(elvec.size(), 4);
    EXPECT_GT(elvec.sum(), 0.0);
}

TEST_F(IntegratorTest, BilinearFormAssembler) {
    BilinearFormAssembler assembler(fes_.get());
    
    assembler.addDomainIntegrator(std::make_unique<DiffusionIntegrator>(mat1_.get()));
    assembler.assemble();
    
    SparseMatrix& A = assembler.matrix();
    
    EXPECT_EQ(A.rows(), 4);
    EXPECT_EQ(A.cols(), 4);
    EXPECT_GT(A.nonZeros(), 0);
}

TEST_F(IntegratorTest, LinearFormAssembler) {
    LinearFormAssembler assembler(fes_.get());
    
    assembler.addDomainIntegrator(std::make_unique<DomainLFIntegrator>(k1_.get()));
    assembler.assemble();
    
    Vector& b = assembler.vector();
    
    EXPECT_EQ(b.size(), 4);
    EXPECT_GT(b.norm(), 0.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
