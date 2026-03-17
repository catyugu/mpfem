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
    }
    
    Mesh mesh_;
    std::unique_ptr<FESpace> fes_;
    ConstantCoefficient k1_{1.0};
    ConstantCoefficient k2_{2.0};
};

TEST_F(IntegratorTest, DiffusionElementMatrix) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    DiffusionIntegrator integ(&k1_);
    
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
    
    DiffusionIntegrator integ1(&k1_);
    DiffusionIntegrator integ2(&k2_);
    
    Matrix elmat1, elmat2;
    integ1.assembleElementMatrix(*refElem, trans, elmat1);
    integ2.assembleElementMatrix(*refElem, trans, elmat2);
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(elmat2(i, j), 2.0 * elmat1(i, j), 1e-12);
        }
    }
}

TEST_F(IntegratorTest, MassElementMatrix) {
    ElementTransform trans(&mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);
    
    MassIntegrator integ(&k1_);
    
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
    
    DomainLFIntegrator integ(&k1_);
    
    Vector elvec;
    integ.assembleElementVector(*refElem, trans, elvec);
    
    EXPECT_EQ(elvec.size(), 4);
    EXPECT_GT(elvec.sum(), 0.0);
}

TEST_F(IntegratorTest, BilinearFormAssembler) {
    BilinearFormAssembler assembler(fes_.get());
    
    assembler.addDomainIntegrator(std::make_unique<DiffusionIntegrator>(&k1_));
    assembler.assemble();
    
    SparseMatrix& A = assembler.matrix();
    
    EXPECT_EQ(A.rows(), 4);
    EXPECT_EQ(A.cols(), 4);
    EXPECT_GT(A.nonZeros(), 0);
}

TEST_F(IntegratorTest, LinearFormAssembler) {
    LinearFormAssembler assembler(fes_.get());
    
    assembler.addDomainIntegrator(std::make_unique<DomainLFIntegrator>(&k1_));
    assembler.assemble();
    
    Vector& b = assembler.vector();
    
    EXPECT_EQ(b.size(), 4);
    EXPECT_GT(b.norm(), 0.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
