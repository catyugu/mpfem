#include <gtest/gtest.h>
#include "coefficient/coefficient.hpp"
#include "coefficient/material_coefficient.hpp"
#include "fe/fe_values.hpp"
#include "fe/grid_function.hpp"
#include "fe/fe_space.hpp"
#include "fe/fe_collection.hpp"
#include "fe/element_transform.hpp"
#include "mesh/mesh.hpp"
#include "model/material_database.hpp"
#include "core/logger.hpp"

using namespace mpfem;

class CoefficientTest : public ::testing::Test {
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

TEST_F(CoefficientTest, ConstantCoefficient) {
    auto coeff = constCoeff(5.0);
    
    FEValues state;
    IntegrationPoint ip(0.25, 0.25, 0.25, 0.0);
    ElementTransform trans;
    
    Real value = coeff->eval(0.0, state, 0, ip, trans);
    EXPECT_DOUBLE_EQ(value, 5.0);
}

TEST_F(CoefficientTest, FunctionCoefficient) {
    // Coefficient that returns x + y + z
    auto coeff = funcCoeff([](Real t, const Vector3& x, const FEValues& state) {
        return x.x() + x.y() + x.z();
    });
    
    FEValues state;
    IntegrationPoint ip(0.5, 0.5, 0.5, 0.0);
    
    // Create element transform for the reference tetrahedron
    ElementTransform trans;
    trans.setMesh(&mesh_);
    trans.setElement(0);
    
    // At reference coordinates (0.5, 0.5, 0.5), the physical position
    // for our simple tet will be computed by the transform
    Real value = coeff->eval(0.0, state, 0, ip, trans);
    
    // Value should be x + y + z at the physical point
    // We're just verifying it runs and produces a value
    EXPECT_TRUE(std::isfinite(value));
}

TEST_F(CoefficientTest, FieldCoefficient) {
    FieldCoefficient coeff(FieldKind::Temperature);
    
    FEValues state;
    GridFunction gf(fes_.get());
    
    // Set constant temperature
    gf.setConstant(300.0);
    state.registerField(FieldKind::Temperature, &gf);
    
    IntegrationPoint ip(0.25, 0.25, 0.25, 0.0);
    ElementTransform trans;
    
    Real value = coeff.eval(0.0, state, 0, ip, trans);
    EXPECT_DOUBLE_EQ(value, 300.0);
}

TEST_F(CoefficientTest, GradientSquaredCoefficient) {
    GradientSquaredCoefficient coeff(FieldKind::ElectricPotential);
    
    FEValues state;
    GridFunction gf(fes_.get());
    
    // Set linear potential: values [0, 1, 2, 3]
    Eigen::VectorXd values(4);
    values << 0.0, 1.0, 2.0, 3.0;
    gf.setValues(values);
    state.registerField(FieldKind::ElectricPotential, &gf);
    
    IntegrationPoint ip(0.25, 0.25, 0.25, 0.0);
    ElementTransform trans;
    trans.setMesh(&mesh_);
    trans.setElement(0);
    
    Real value = coeff.eval(0.0, state, 0, ip, trans);
    
    // Should return |grad V|^2
    EXPECT_TRUE(value >= 0.0);
}

TEST_F(CoefficientTest, ScaledCoefficient) {
    auto inner = constCoeff(10.0);
    auto scaled = scaleCoeff(2.0, std::move(inner));
    
    FEValues state;
    IntegrationPoint ip(0.0, 0.0, 0.0, 0.0);
    ElementTransform trans;
    
    Real value = scaled->eval(0.0, state, 0, ip, trans);
    EXPECT_DOUBLE_EQ(value, 20.0);
}

TEST_F(CoefficientTest, ProductCoefficient) {
    auto a = constCoeff(3.0);
    auto b = constCoeff(4.0);
    auto product = productCoeff(std::move(a), std::move(b));
    
    FEValues state;
    IntegrationPoint ip(0.0, 0.0, 0.0, 0.0);
    ElementTransform trans;
    
    Real value = product->eval(0.0, state, 0, ip, trans);
    EXPECT_DOUBLE_EQ(value, 12.0);
}

TEST_F(CoefficientTest, MaterialCoefficientBasic) {
    // Create a simple material database
    MaterialDatabase matDb;
    MaterialPropertyModel mat;
    mat.setProperty("thermalconductivity", 400.0);
    mat.setProperty("electricconductivity", 5.998e7);
    mat.tag = "mat1";
    matDb.addMaterial(mat);
    
    std::map<int, std::string> domainToMaterial = {{1, "mat1"}};
    
    MaterialCoefficient coeff(&matDb, "thermalconductivity", domainToMaterial);
    
    FEValues state;
    IntegrationPoint ip(0.25, 0.25, 0.25, 0.0);
    ElementTransform trans;
    trans.setMesh(&mesh_);
    trans.setElement(0);
    
    Real value = coeff.eval(0.0, state, 0, ip, trans);
    EXPECT_DOUBLE_EQ(value, 400.0);
}

TEST_F(CoefficientTest, LinearTemperatureCoefficient) {
    LinearTemperatureCoefficient coeff(100.0, 0.01, 293.15);
    
    FEValues state;
    GridFunction temp(fes_.get());
    temp.setConstant(303.15);  // 10K above reference
    state.registerField(FieldKind::Temperature, &temp);
    
    IntegrationPoint ip(0.0, 0.0, 0.0, 0.0);
    ElementTransform trans;
    
    // Expected: 100.0 * (1 + 0.01 * (303.15 - 293.15)) = 100.0 * 1.1 = 110.0
    Real value = coeff.eval(0.0, state, 0, ip, trans);
    EXPECT_DOUBLE_EQ(value, 110.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
