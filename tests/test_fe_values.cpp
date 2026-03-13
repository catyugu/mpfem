#include <gtest/gtest.h>
#include "fe/fe_values.hpp"
#include "fe/grid_function.hpp"
#include "fe/fe_space.hpp"
#include "fe/fe_collection.hpp"
#include "mesh/mesh.hpp"
#include "core/logger.hpp"

using namespace mpfem;

class FEValuesTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Info);
        mesh_ = createSimpleTetMesh();
        fec_ = std::make_unique<FECollection>(1);
        fes_ = std::make_unique<FESpace>(&mesh_, fec_.get(), 1);
    }
    
    Mesh createSimpleTetMesh() {
        Mesh mesh;
        mesh.setDim(3);
        mesh.addVertex(0.0, 0.0, 0.0);
        mesh.addVertex(1.0, 0.0, 0.0);
        mesh.addVertex(0.0, 1.0, 0.0);
        mesh.addVertex(0.0, 0.0, 1.0);
        mesh.addElement(Geometry::Tetrahedron, {0, 1, 2, 3}, 1);
        return mesh;
    }
    
    Mesh mesh_;
    std::unique_ptr<FECollection> fec_;
    std::unique_ptr<FESpace> fes_;
};

TEST_F(FEValuesTest, FieldRegistration) {
    FEValues feValues;
    GridFunction gf(fes_.get());
    
    feValues.registerField(FieldKind::Temperature, &gf);
    
    EXPECT_EQ(feValues.numFields(), 1);
}

TEST_F(FEValuesTest, FieldAccess) {
    FEValues feValues;
    GridFunction gf(fes_.get());
    gf.setConstant(293.15);
    
    feValues.registerField(FieldKind::Temperature, &gf);
    
    const GridFunction* retrieved = feValues.field(FieldKind::Temperature);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_DOUBLE_EQ(retrieved->l2Norm(), gf.l2Norm());
}

TEST_F(FEValuesTest, GetValue) {
    FEValues feValues;
    GridFunction gf(fes_.get());
    
    Eigen::VectorXd values(4);
    values << 300.0, 310.0, 320.0, 330.0;
    gf.setValues(values);
    
    feValues.registerField(FieldKind::Temperature, &gf);
    
    Real xi[] = {0.25, 0.25, 0.25};
    Real T = feValues.getValue(FieldKind::Temperature, 0, xi);
    
    EXPECT_NEAR(T, 315.0, 1e-10);
}

TEST_F(FEValuesTest, Clear) {
    FEValues feValues;
    GridFunction gf(fes_.get());
    
    feValues.registerField(FieldKind::Temperature, &gf);
    EXPECT_EQ(feValues.numFields(), 1);
    
    feValues.clear();
    EXPECT_EQ(feValues.numFields(), 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}