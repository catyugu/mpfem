#include "assembly/assembler.hpp"
#include "assembly/element_binding.hpp"
#include "assembly/integrators.hpp"
#include "core/logger.hpp"
#include "core/types.hpp"
#include "expr/variable_graph.hpp"
#include "fe/element_transform.hpp"
#include "fe/fe_collection.hpp"
#include "fe/fe_space.hpp"
#include "fe/quadrature.hpp"
#include "mesh/geometry.hpp"
#include <cmath>
#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

using namespace mpfem;

namespace {

    class ScalarConstantNode final : public VariableNode {
    public:
        explicit ScalarConstantNode(Real value)
            : value_(value) { }

        void evaluateBatch(const EvaluationContext& ctx, std::span<Tensor> dest) const override
        {
            const size_t n = ctx.physicalPoints.empty() ? dest.size() : ctx.physicalPoints.size();
            if (dest.size() != n) {
                throw std::runtime_error("ScalarConstantNode destination size mismatch");
            }
            for (size_t i = 0; i < dest.size(); ++i) {
                dest[i] = Tensor::scalar(value_);
            }
        }

    private:
        Real value_ = 0.0;
    };

    class MatrixConstantNode final : public VariableNode {
    public:
        explicit MatrixConstantNode(const Matrix3& value)
            : value_(value) { }

        void evaluateBatch(const EvaluationContext& ctx, std::span<Tensor> dest) const override
        {
            const size_t n = ctx.physicalPoints.empty() ? dest.size() : ctx.physicalPoints.size();
            if (dest.size() != n) {
                throw std::runtime_error("MatrixConstantNode destination size mismatch");
            }
            for (size_t i = 0; i < n; ++i) {
                dest[i] = Tensor::matrix(3, 3, value_);
            }
        }

    private:
        Matrix3 value_ = Matrix3::Zero();
    };

} // namespace

class IntegratorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        Logger::setLevel(LogLevel::Warning);

        mesh_.setDim(3);
        mesh_.addVertex(0.0, 0.0, 0.0);
        mesh_.addVertex(1.0, 0.0, 0.0);
        mesh_.addVertex(0.0, 1.0, 0.0);
        mesh_.addVertex(0.0, 0.0, 1.0);
        mesh_.addElement(Geometry::Tetrahedron, {0, 1, 2, 3}, 1, 1);
        mesh_.buildTopology();

        fes_ = std::make_unique<FESpace>(&mesh_, std::make_unique<FECollection>(1));

        k1_ = std::make_unique<ScalarConstantNode>(1.0);
        k2_ = std::make_unique<ScalarConstantNode>(2.0);

        // Initialize matrix coefficients for diffusion tests
        Matrix3 D1 = Matrix3::Identity() * 1.0;
        Matrix3 D2 = Matrix3::Identity() * 2.0;
        mat1_ = std::make_unique<MatrixConstantNode>(D1);
        mat2_ = std::make_unique<MatrixConstantNode>(D2);
    }

    Mesh mesh_;
    std::unique_ptr<FESpace> fes_;
    std::unique_ptr<VariableNode> k1_;
    std::unique_ptr<VariableNode> k2_;
    std::unique_ptr<VariableNode> mat1_;
    std::unique_ptr<VariableNode> mat2_;
};

TEST_F(IntegratorTest, DiffusionElementMatrix)
{
    ElementTransform trans;
    bindElementToTransform(trans, mesh_, 0);
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

TEST_F(IntegratorTest, DiffusionMatrixScaling)
{
    ElementTransform trans;
    bindElementToTransform(trans, mesh_, 0);
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

TEST_F(IntegratorTest, AnisotropicDiffusion)
{
    ElementTransform trans;
    bindElementToTransform(trans, mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);

    // Create anisotropic matrix coefficient
    Matrix3 D = Matrix3::Identity() * 2.0; // Diagonal = 2
    auto matCoef = std::make_unique<MatrixConstantNode>(D);

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

TEST_F(IntegratorTest, MassElementMatrix)
{
    ElementTransform trans;
    bindElementToTransform(trans, mesh_, 0);
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

TEST_F(IntegratorTest, DomainLoadVector)
{
    ElementTransform trans;
    bindElementToTransform(trans, mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);

    DomainLFIntegrator integ(k1_.get());

    Vector elvec;
    integ.assembleElementVector(*refElem, trans, elvec);

    EXPECT_EQ(elvec.size(), 4);
    EXPECT_GT(elvec.sum(), 0.0);
}

TEST_F(IntegratorTest, StrainLoadVectorScaling)
{
    ElementTransform trans;
    bindElementToTransform(trans, mesh_, 0);
    const ReferenceElement* refElem = fes_->elementRefElement(0);

    Matrix3 sigma1 = Matrix3::Zero();
    sigma1(0, 0) = 1.0;
    sigma1(1, 1) = 2.0;
    sigma1(2, 2) = 3.0;
    sigma1(0, 1) = sigma1(1, 0) = 0.5;
    auto stress1 = std::make_unique<MatrixConstantNode>(sigma1);

    Matrix3 sigma2 = 2.0 * sigma1;
    auto stress2 = std::make_unique<MatrixConstantNode>(sigma2);

    StrainLoadIntegrator integ1(stress1.get(), 3);
    StrainLoadIntegrator integ2(stress2.get(), 3);

    Vector elvec1, elvec2;
    integ1.assembleElementVector(*refElem, trans, elvec1);
    integ2.assembleElementVector(*refElem, trans, elvec2);

    EXPECT_EQ(elvec1.size(), 12);
    EXPECT_EQ(elvec2.size(), 12);
    EXPECT_GT(elvec1.norm(), 0.0);

    for (int i = 0; i < elvec1.size(); ++i) {
        EXPECT_NEAR(elvec2(i), 2.0 * elvec1(i), 1e-12);
    }
}

TEST_F(IntegratorTest, BilinearFormAssembler)
{
    BilinearFormAssembler assembler(fes_.get());

    assembler.addDomainIntegrator(std::make_unique<DiffusionIntegrator>(mat1_.get()));
    assembler.assemble();

    SparseMatrix& A = assembler.matrix();

    EXPECT_EQ(A.rows(), 4);
    EXPECT_EQ(A.cols(), 4);
    EXPECT_GT(A.nonZeros(), 0);
}

TEST_F(IntegratorTest, LinearFormAssembler)
{
    LinearFormAssembler assembler(fes_.get());

    assembler.addDomainIntegrator(std::make_unique<DomainLFIntegrator>(k1_.get()));
    assembler.assemble();

    Vector& b = assembler.vector();

    EXPECT_EQ(b.size(), 4);
    EXPECT_GT(b.norm(), 0.0);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
