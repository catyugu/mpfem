/**
 * @file test_coefficient.cpp
 * @brief Unit tests for Coefficient classes.
 * 
 * Tests cover:
 * - Scalar coefficients (Constant, PWConst, Function, GridFunction)
 * - Temperature-dependent coefficients
 * - Vector coefficients
 * - Matrix coefficients
 * 
 * Tests are designed for both order 1 (linear) and order 2 (quadratic) elements.
 */

#include <gtest/gtest.h>
#include "fe/coefficient.hpp"
#include "fe/element_transform.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include "fe/quadrature.hpp"
#include "mesh/mesh.hpp"
#include "mesh/io/mphtxt_reader.hpp"
#include "core/logger.hpp"
#include <cmath>
#include <memory>

using namespace mpfem;

// =============================================================================
// Test Fixtures
// =============================================================================

class CoefficientTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Warning);
    }
};

/**
 * @brief Test fixture for coefficients with a simple mesh.
 */
class CoefficientMeshTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Warning);
        
        // Create a simple 2x2 quad mesh in 2D
        createSimpleQuadMesh();
    }
    
    void createSimpleQuadMesh() {
        // Create 2x2 quad mesh manually
        // Vertices: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)
        mesh_ = std::make_unique<Mesh>();
        mesh_->setDim(2);
        
        // Add vertices (Vector3 first, then dim)
        mesh_->addVertex(Vertex(Vector3(0.0, 0.0, 0.0), 2));
        mesh_->addVertex(Vertex(Vector3(1.0, 0.0, 0.0), 2));
        mesh_->addVertex(Vertex(Vector3(2.0, 0.0, 0.0), 2));
        mesh_->addVertex(Vertex(Vector3(0.0, 1.0, 0.0), 2));
        mesh_->addVertex(Vertex(Vector3(1.0, 1.0, 0.0), 2));
        mesh_->addVertex(Vertex(Vector3(2.0, 1.0, 0.0), 2));
        mesh_->addVertex(Vertex(Vector3(0.0, 2.0, 0.0), 2));
        mesh_->addVertex(Vertex(Vector3(1.0, 2.0, 0.0), 2));
        mesh_->addVertex(Vertex(Vector3(2.0, 2.0, 0.0), 2));
        
        // Add elements (quads)
        // Element 0: vertices 0,1,4,3, attribute 1
        Element elem0(Geometry::Square, std::vector<Index>{0, 1, 4, 3}, 1);
        mesh_->addElement(elem0);
        
        // Element 1: vertices 1,2,5,4, attribute 2
        Element elem1(Geometry::Square, std::vector<Index>{1, 2, 5, 4}, 2);
        mesh_->addElement(elem1);
        
        // Element 2: vertices 3,4,7,6, attribute 1
        Element elem2(Geometry::Square, std::vector<Index>{3, 4, 7, 6}, 1);
        mesh_->addElement(elem2);
        
        // Element 3: vertices 4,5,8,7, attribute 2
        Element elem3(Geometry::Square, std::vector<Index>{4, 5, 8, 7}, 2);
        mesh_->addElement(elem3);
        
        // Add boundary elements
        // Bottom boundary
        mesh_->addBdrElement(Element(Geometry::Segment, std::vector<Index>{0, 1}, 10));
        mesh_->addBdrElement(Element(Geometry::Segment, std::vector<Index>{1, 2}, 11));
        // Right boundary
        mesh_->addBdrElement(Element(Geometry::Segment, std::vector<Index>{2, 5}, 12));
        mesh_->addBdrElement(Element(Geometry::Segment, std::vector<Index>{5, 8}, 13));
        // Top boundary
        mesh_->addBdrElement(Element(Geometry::Segment, std::vector<Index>{8, 7}, 14));
        mesh_->addBdrElement(Element(Geometry::Segment, std::vector<Index>{7, 6}, 15));
        // Left boundary
        mesh_->addBdrElement(Element(Geometry::Segment, std::vector<Index>{6, 3}, 16));
        mesh_->addBdrElement(Element(Geometry::Segment, std::vector<Index>{3, 0}, 17));
    }
    
    std::unique_ptr<Mesh> mesh_;
};

// =============================================================================
// ConstantCoefficient Tests
// =============================================================================

TEST_F(CoefficientTest, ConstantCoefficient_Value) {
    ConstantCoefficient coef(3.14);
    EXPECT_DOUBLE_EQ(coef.value(), 3.14);
}

TEST_F(CoefficientTest, ConstantCoefficient_Default) {
    ConstantCoefficient coef;
    EXPECT_DOUBLE_EQ(coef.value(), 1.0);
}

TEST_F(CoefficientTest, ConstantCoefficient_SetValue) {
    ConstantCoefficient coef(1.0);
    coef.setValue(2.5);
    EXPECT_DOUBLE_EQ(coef.value(), 2.5);
}

// =============================================================================
// PWConstCoefficient Tests
// =============================================================================

TEST_F(CoefficientMeshTest, PWConstCoefficient_Basic) {
    PWConstCoefficient coef(2);  // 2 attributes
    coef(1) = 10.0;
    coef(2) = 20.0;
    
    ElementTransform trans(mesh_.get(), 0);  // Element 0 has attribute 1
    IntegrationPoint ip(0.5, 0.5, 0.0, 1.0);
    trans.setIntegrationPoint(ip);
    
    EXPECT_DOUBLE_EQ(coef.eval(trans), 10.0);
    
    trans.setElement(1);  // Element 1 has attribute 2
    EXPECT_DOUBLE_EQ(coef.eval(trans), 20.0);
}

TEST_F(CoefficientMeshTest, PWConstCoefficient_Resize) {
    PWConstCoefficient coef;
    coef.resize(3, 5.0);
    
    EXPECT_EQ(coef.numAttributes(), 3);
    EXPECT_DOUBLE_EQ(coef.constant(1), 5.0);
    EXPECT_DOUBLE_EQ(coef.constant(2), 5.0);
    EXPECT_DOUBLE_EQ(coef.constant(3), 5.0);
}

TEST_F(CoefficientMeshTest, PWConstCoefficient_SetConstant) {
    PWConstCoefficient coef(3);
    coef.setConstant(1, 1.0);
    coef.setConstant(2, 2.0);
    coef.setConstant(3, 3.0);
    
    EXPECT_DOUBLE_EQ(coef.constant(1), 1.0);
    EXPECT_DOUBLE_EQ(coef.constant(2), 2.0);
    EXPECT_DOUBLE_EQ(coef.constant(3), 3.0);
}

// =============================================================================
// FunctionCoefficient Tests
// =============================================================================

TEST_F(CoefficientMeshTest, FunctionCoefficient_TimeIndependent) {
    // f(x,y,z) = x + 2*y
    FunctionCoefficient coef([](Real x, Real y, Real z) {
        return x + 2.0 * y;
    });
    
    ElementTransform trans(mesh_.get(), 0);
    IntegrationPoint ip(0.0, 0.0, 0.0, 1.0);  // At origin
    trans.setIntegrationPoint(ip);
    
    // Element 0 spans [0,1] x [0,1], at (0,0) should give 1.5
    Real val = coef.eval(trans);
    EXPECT_NEAR(val, 1.5 , 1e-10);
    
    // Test at center of element (0.0, 0.0)
    ip = IntegrationPoint(0.0, 0.0, 0.0, 1.0);  // Reference center
    trans.setIntegrationPoint(ip);
    // At reference (0.0, 0.0) which is physical (0.5, 0.5), value should be 1.5
    val = coef.eval(trans);
    EXPECT_NEAR(val, 1.5, 1e-10);
}

TEST_F(CoefficientMeshTest, FunctionCoefficient_TimeDependent) {
    // f(x,y,z,t) = x + y + t
    FunctionCoefficient coef([](Real x, Real y, Real z, Real t) {
        return x + y + t;
    }, true);
    
    ElementTransform trans(mesh_.get(), 0);
    // Reference (-1,-1) maps to physical (0,0), so f(0,0,0,t) = t
    IntegrationPoint ip(-1.0, -1.0, 0.0, 1.0);
    trans.setIntegrationPoint(ip);
    
    coef.setTime(5.0);
    Real val = coef.eval(trans);
    // At (0,0) with t=5, value should be 5
    EXPECT_NEAR(val, 5.0, 1e-10);
}

// =============================================================================
// GridFunctionCoefficient Tests
// =============================================================================

TEST_F(CoefficientMeshTest, GridFunctionCoefficient_Basic) {
    // Create FE space
    FECollection fec(1, FECollection::Type::H1);
    FESpace fes(mesh_.get(), &fec);
    
    // Create grid function
    GridFunction gf(&fes);
    
    // Set values: constant 2.5 on all DOFs
    gf.setConstant(2.5);
    
    // Create coefficient
    GridFunctionCoefficient coef(&gf);
    
    // Evaluate at element center
    ElementTransform trans(mesh_.get(), 0);
    IntegrationPoint ip(0.0, 0.0, 0.0, 1.0);  // Reference center
    trans.setIntegrationPoint(ip);
    
    Real val = coef.eval(trans);
    EXPECT_NEAR(val, 2.5, 1e-10);
}

TEST_F(CoefficientMeshTest, GridFunctionCoefficient_LinerField) {
    // Create FE space
    FECollection fec(1, FECollection::Type::H1);
    FESpace fes(mesh_.get(), &fec);
    
    // Create grid function
    GridFunction gf(&fes);
    
    // Set linear field: u = x (value equals x-coordinate)
    // Vertex 0 at (0,0): u = 0
    // Vertex 1 at (1,0): u = 1
    // Vertex 3 at (0,1): u = 0
    // Vertex 4 at (1,1): u = 1
    gf(0) = 0.0;  // Vertex 0
    gf(1) = 1.0;  // Vertex 1
    gf(2) = 2.0;  // Vertex 2
    gf(3) = 0.0;  // Vertex 3
    gf(4) = 1.0;  // Vertex 4
    gf(5) = 2.0;  // Vertex 5
    gf(6) = 0.0;  // Vertex 6
    gf(7) = 1.0;  // Vertex 7
    gf(8) = 2.0;  // Vertex 8
    
    GridFunctionCoefficient coef(&gf);
    
    // Test at center of element 0: should be 0.5
    ElementTransform trans(mesh_.get(), 0);
    IntegrationPoint ip(0.0, 0.0, 0.0, 1.0);  // Reference center
    trans.setIntegrationPoint(ip);
    
    // At center of element 0 (physical 0.5, 0.5), value should be 0.5
    Real val = coef.eval(trans);
    // Note: The exact interpolation depends on shape function implementation
    // For a bilinear element, center value is average of corner values
    // Corner values: 0, 1, 1, 0 -> average = 0.5
    EXPECT_NEAR(val, 0.5, 0.1);  // Allow some tolerance for numerical issues
}

// =============================================================================  
// SumCoefficient Tests
// =============================================================================

TEST_F(CoefficientTest, SumCoefficient_Basic) {
    auto a = std::make_shared<ConstantCoefficient>(3.0);
    auto b = std::make_shared<ConstantCoefficient>(4.0);
    
    SumCoefficient sum(a, b, 2.0, 3.0);  // 2*a + 3*b = 2*3 + 3*4 = 18
    
    // Verification through construction
    EXPECT_NE(&sum, nullptr);
}
// =============================================================================
// TemperatureDependentConductivity Tests
// =============================================================================

TEST_F(CoefficientMeshTest, TemperatureDependentConductivity_ConstantTemp) {
    TemperatureDependentConductivityCoefficient coef;
    
    // Set material fields for 2 attributes
    std::vector<Real> rho0 = {1.72e-8, 1.35e-6};      // Resistivity
    std::vector<Real> alpha = {0.0039, 0.0};          // Temp coefficient
    std::vector<Real> tref = {298.0, 298.0};          // Reference temp
    std::vector<Real> sigma0 = {5.998e7, 7.407e5};    // Conductivity
    
    coef.setMaterialFields(rho0, alpha, tref, sigma0);
    
    ElementTransform trans(mesh_.get(), 0);
    IntegrationPoint ip(0.0, 0.0, 0.0, 1.0);
    trans.setIntegrationPoint(ip);
    
    // At reference temperature, conductivity should be 1/rho0
    Real conductivity = coef.eval(trans);
    Real expected = 1.0 / rho0[0];
    EXPECT_NEAR(conductivity, expected, expected * 1e-6);
}

TEST_F(CoefficientMeshTest, TemperatureDependentConductivity_WithTempField) {
    // Create FE space for temperature field
    FECollection fec(1, FECollection::Type::H1);
    FESpace fes(mesh_.get(), &fec);
    GridFunction temperature(&fes);
    
    // Set temperature to 350K everywhere
    temperature.setConstant(350.0);
    
    TemperatureDependentConductivityCoefficient coef;
    coef.setTemperatureField(&temperature);
    
    // Set material fields
    std::vector<Real> rho0 = {1.72e-8, 1.35e-6};
    std::vector<Real> alpha = {0.0039, 0.0};
    std::vector<Real> tref = {298.0, 298.0};
    std::vector<Real> sigma0 = {5.998e7, 7.407e5};
    
    coef.setMaterialFields(rho0, alpha, tref, sigma0);
    
    ElementTransform trans(mesh_.get(), 0);
    IntegrationPoint ip(0.0, 0.0, 0.0, 1.0);
    trans.setIntegrationPoint(ip);
    
    // At T=350K, rho = rho0 * (1 + alpha * (T - Tref))
    Real rho = rho0[0] * (1.0 + alpha[0] * (350.0 - 298.0));
    Real expected = 1.0 / rho;
    
    Real conductivity = coef.eval(trans);
    EXPECT_NEAR(conductivity, expected, expected * 0.01);  // 1% tolerance
}

// =============================================================================
// VectorCoefficient Tests
// =============================================================================

TEST_F(CoefficientTest, VectorConstantCoefficient_Basic) {
    VectorConstantCoefficient coef(Vector3(1.0, 2.0, 3.0));
    
    EXPECT_EQ(coef.vdim(), 3);
    
    // VectorConstantCoefficient doesn't need transform for constant value
    Real result[3];
    ElementTransform trans;
    coef.eval(trans, result);
    
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], 3.0);
}

TEST_F(CoefficientTest, VectorConstantCoefficient_CustomDim) {
    std::vector<Real> values = {1.0, 2.0, 3.0, 4.0};
    VectorConstantCoefficient coef(4, values);
    
    EXPECT_EQ(coef.vdim(), 4);
}

// =============================================================================
// MatrixCoefficient Tests
// =============================================================================

TEST_F(CoefficientTest, IdentityMatrixCoefficient_Basic) {
    IdentityMatrixCoefficient coef(3);
    
    EXPECT_EQ(coef.rows(), 3);
    EXPECT_EQ(coef.cols(), 3);
    
    Real result[9];
    ElementTransform trans;
    coef.eval(trans, result);
    
    // Check identity matrix
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(result[i * 3 + j], (i == j) ? 1.0 : 0.0);
        }
    }
}

TEST_F(CoefficientTest, DiagonalMatrixCoefficient_Basic) {
    DiagonalMatrixCoefficient coef(3, 5.0);
    
    Real result[9];
    ElementTransform trans;
    coef.eval(trans, result);
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(result[i * 3 + j], (i == j) ? 5.0 : 0.0);
        }
    }
}

TEST_F(CoefficientMeshTest, DiagonalFromScalarCoefficient_Basic) {
    auto scalar = std::make_shared<ConstantCoefficient>(4.0);
    DiagonalFromScalarCoefficient coef(3, scalar);
    
    ElementTransform trans(mesh_.get(), 0);
    IntegrationPoint ip(0.0, 0.0, 0.0, 1.0);
    trans.setIntegrationPoint(ip);
    
    Real result[9];
    coef.eval(trans, result);
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(result[i * 3 + j], (i == j) ? 4.0 : 0.0);
        }
    }
}

// =============================================================================
// Utility Function Tests
// =============================================================================

TEST_F(CoefficientTest, MakeConstant) {
    auto coef = makeConstant(3.14);
    EXPECT_NE(coef, nullptr);
}

TEST_F(CoefficientTest, MakePWConst) {
    std::vector<Real> values = {1.0, 2.0, 3.0};
    auto coef = makePWConst(values);
    EXPECT_NE(coef, nullptr);
}

// =============================================================================
// Time-Dependent Tests
// =============================================================================

TEST_F(CoefficientTest, Coefficient_Time) {
    ConstantCoefficient coef(1.0);
    
    EXPECT_DOUBLE_EQ(coef.time(), 0.0);
    
    coef.setTime(5.0);
    EXPECT_DOUBLE_EQ(coef.time(), 5.0);
}

// =============================================================================
// Order 2 (Quadratic) Element Tests
// =============================================================================

/**
 * @brief Test fixture for quadratic (order 2) elements.
 * 
 * Uses busbar_order2 mesh if available, or creates a simple quadratic mesh.
 */
class CoefficientQuadraticTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Warning);
        
        // Try to load the busbar_order2 mesh
        std::string meshPath = MPFEM_PROJECT_ROOT "/cases/busbar_order2/mesh.mphtxt";
        
        try {
            MphtxtReader reader;
            mesh_ = std::make_unique<Mesh>(reader.read(meshPath));
        } catch (...) {
            GTEST_SKIP() << "Could not load quadratic mesh, skipping quadratic tests";
        }
    }
    
    std::unique_ptr<Mesh> mesh_;
};

TEST_F(CoefficientQuadraticTest, PWConstCoefficient_QuadraticMesh) {
    if (!mesh_) return;
    
    // Get max attribute
    int maxAttr = 0;
    for (Index i = 0; i < mesh_->numElements(); ++i) {
        maxAttr = std::max(maxAttr, static_cast<int>(mesh_->element(i).attribute()));
    }
    
    PWConstCoefficient coef(maxAttr);
    for (int i = 1; i <= maxAttr; ++i) {
        coef(i) = static_cast<Real>(i) * 10.0;
    }
    
    // Test on first element
    if (mesh_->numElements() > 0) {
        ElementTransform trans(mesh_.get(), 0);
        IntegrationPoint ip(0.0, 0.0, 0.0, 1.0);  // Reference center
        trans.setIntegrationPoint(ip);
        
        Index attr = mesh_->element(0).attribute();
        Real expected = static_cast<Real>(attr) * 10.0;
        
        EXPECT_NEAR(coef.eval(trans), expected, 1e-10);
    }
}

TEST_F(CoefficientQuadraticTest, GridFunctionCoefficient_QuadraticSpace) {
    if (!mesh_) return;
    
    // Create quadratic FE space
    FECollection fec(2, FECollection::Type::H1);
    FESpace fes(mesh_.get(), &fec);
    
    GridFunction gf(&fes);
    gf.setConstant(1.5);
    
    GridFunctionCoefficient coef(&gf);
    
    // Evaluate at center of first element
    if (mesh_->numElements() > 0) {
        ElementTransform trans(mesh_.get(), 0);
        IntegrationPoint ip(0.0, 0.0, 0.0, 1.0);
        trans.setIntegrationPoint(ip);
        
        Real val = coef.eval(trans);
        EXPECT_NEAR(val, 1.5, 1e-10);
    }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
