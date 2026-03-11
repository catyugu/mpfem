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
        fesVector_ = std::make_unique<FESpace>(&mesh_, fec_.get(), 3);
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
    std::unique_ptr<FESpace> fesVector_;
};

TEST_F(FEValuesTest, TimeManagement) {
    FEValues feValues;
    
    EXPECT_DOUBLE_EQ(feValues.time(), 0.0);
    
    feValues.setTime(1.5);
    EXPECT_DOUBLE_EQ(feValues.time(), 1.5);
}

TEST_F(FEValuesTest, FieldRegistration) {
    FEValues feValues;
    GridFunction gf(fes_.get());
    
    feValues.registerField(FieldKind::Temperature, &gf);
    
    EXPECT_TRUE(feValues.hasField(FieldKind::Temperature));
    EXPECT_FALSE(feValues.hasField(FieldKind::ElectricPotential));
    EXPECT_EQ(feValues.numFields(), 1);
}

TEST_F(FEValuesTest, FieldRegistrationByName) {
    FEValues feValues;
    GridFunction gf(fes_.get());
    
    feValues.registerField("my_field", &gf);
    
    EXPECT_TRUE(feValues.hasField("my_field"));
    EXPECT_FALSE(feValues.hasField("other_field"));
}

TEST_F(FEValuesTest, FieldAccess) {
    FEValues feValues;
    GridFunction gf(fes_.get());
    gf.setConstant(293.15);
    
    feValues.registerField(FieldKind::Temperature, &gf);
    
    GridFunction* retrieved = feValues.field(FieldKind::Temperature);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_DOUBLE_EQ(retrieved->l2Norm(), gf.l2Norm());
}

TEST_F(FEValuesTest, GetValue) {
    FEValues feValues;
    GridFunction gf(fes_.get());
    
    // Set temperature values at vertices
    Eigen::VectorXd values(4);
    values << 300.0, 310.0, 320.0, 330.0;
    gf.setValues(values);
    
    feValues.registerField(FieldKind::Temperature, &gf);
    
    // Evaluate at center (barycentric coords 0.25 each)
    IntegrationPoint ip(0.25, 0.25, 0.25, 0.0);
    Real T = feValues.getValue(FieldKind::Temperature, 0, ip);
    
    // Expected: 0.25*(300+310+320+330) = 315
    EXPECT_NEAR(T, 315.0, 1e-10);
}

TEST_F(FEValuesTest, ConvenienceMethods) {
    FEValues feValues;
    
    // Register electric potential
    GridFunction potential(fes_.get());
    potential.setConstant(5.0);  // 5V everywhere
    feValues.registerField(FieldKind::ElectricPotential, &potential);
    
    // Register temperature
    GridFunction temperature(fes_.get());
    temperature.setConstant(293.15);  // 293.15K everywhere
    feValues.registerField(FieldKind::Temperature, &temperature);
    
    // Test convenience methods
    IntegrationPoint ip(0.25, 0.25, 0.25, 0.0);
    
    Real V = feValues.electricPotential(0, ip);
    EXPECT_DOUBLE_EQ(V, 5.0);
    
    Real T = feValues.temperature(0, ip);
    EXPECT_DOUBLE_EQ(T, 293.15);
}

TEST_F(FEValuesTest, Clear) {
    FEValues feValues;
    GridFunction gf(fes_.get());
    
    feValues.registerField(FieldKind::Temperature, &gf);
    EXPECT_EQ(feValues.numFields(), 1);
    
    feValues.clear();
    EXPECT_EQ(feValues.numFields(), 0);
    EXPECT_FALSE(feValues.hasField(FieldKind::Temperature));
}

TEST_F(FEValuesTest, MultipleFields) {
    FEValues feValues;
    
    GridFunction potential(fes_.get());
    GridFunction temperature(fes_.get());
    GridFunction displacement(fesVector_.get());
    
    potential.setConstant(10.0);
    temperature.setConstant(350.0);
    displacement.setConstant(0.001);
    
    feValues.registerField(FieldKind::ElectricPotential, &potential);
    feValues.registerField(FieldKind::Temperature, &temperature);
    feValues.registerField(FieldKind::Displacement, &displacement);
    
    EXPECT_EQ(feValues.numFields(), 3);
    
    IntegrationPoint ip(0.25, 0.25, 0.25, 0.0);
    
    EXPECT_DOUBLE_EQ(feValues.electricPotential(0, ip), 10.0);
    EXPECT_DOUBLE_EQ(feValues.temperature(0, ip), 350.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
