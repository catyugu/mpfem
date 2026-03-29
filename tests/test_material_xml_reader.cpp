#include <gtest/gtest.h>
#include "io/material_xml_reader.hpp"
#include "core/logger.hpp"

using namespace mpfem;

class MaterialXmlReaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Warning);
    }
    
    // Helper to get test data path
    static std::string dataPath(const std::string& relativePath) {
#ifdef MPFEM_PROJECT_ROOT
        return std::string(MPFEM_PROJECT_ROOT) + "/" + relativePath;
#else
        return relativePath;
#endif
    }
};

TEST_F(MaterialXmlReaderTest, ReadBusbarMaterials) {
    MaterialDatabase database;
    
    ASSERT_NO_THROW({
        MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);
    });

    // Should have 2 materials
    EXPECT_EQ(database.size(), 2);

    // Check mat1 (Copper)
    const MaterialPropertyModel* copper = database.getMaterial("mat1");
    ASSERT_NE(copper, nullptr);
    
    // Check key properties for Copper
    // Electric conductivity is now an expression (temperature-dependent)
    EXPECT_TRUE(copper->hasMatrix("electricconductivity"));
    EXPECT_TRUE(copper->hasMatrixExpression("electricconductivity"));
    
    // Thermal conductivity is a constant matrix
    EXPECT_TRUE(copper->hasMatrix("thermalconductivity"));
    EXPECT_FALSE(copper->hasMatrixExpression("thermalconductivity"));

    // Check mat2 (Titanium)
    const MaterialPropertyModel* titanium = database.getMaterial("mat2");
    ASSERT_NE(titanium, nullptr);
    
    // Titanium has constant electric conductivity (no expression)
    EXPECT_TRUE(titanium->hasMatrix("electricconductivity"));
    EXPECT_FALSE(titanium->hasMatrixExpression("electricconductivity"));
    EXPECT_TRUE(titanium->hasMatrix("thermalconductivity"));
}

TEST_F(MaterialXmlReaderTest, MaterialPropertyAccess) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);

    const MaterialPropertyModel* copper = database.getMaterial("mat1");
    ASSERT_NE(copper, nullptr);

    // Test getScalarProperty
    auto E = copper->getScalarProperty("E");
    EXPECT_TRUE(E.has_value());
    EXPECT_GT(E.value(), 0.0);

    auto nu = copper->getScalarProperty("nu");
    EXPECT_TRUE(nu.has_value());
    EXPECT_GT(nu.value(), 0.0);
    EXPECT_LT(nu.value(), 1.0);  // Poisson ratio should be between 0 and 1

    // Test non-existent property returns nullopt
    auto nonexistent = copper->getScalarProperty("nonexistent_property");
    EXPECT_FALSE(nonexistent.has_value());
}

TEST_F(MaterialXmlReaderTest, TemperatureDependentConductivity) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);

    const MaterialPropertyModel* copper = database.getMaterial("mat1");
    ASSERT_NE(copper, nullptr);

    // Copper has temperature-dependent conductivity
    EXPECT_TRUE(copper->hasMatrixExpression("electricconductivity"));

    // Evaluate at different temperatures
    std::map<std::string, double> vars;
    
    vars["T"] = 293.15;
    auto sigma_293 = copper->evaluateMatrix("electricconductivity", vars);
    EXPECT_TRUE(sigma_293.has_value());
    
    vars["T"] = 373.15;
    auto sigma_373 = copper->evaluateMatrix("electricconductivity", vars);
    EXPECT_TRUE(sigma_373.has_value());
    
    // Check that matrices are diagonal (isotropic case)
    const auto& mat293 = sigma_293.value();
    EXPECT_GT(mat293(0, 0), 0.0);
    EXPECT_GT(mat293(1, 1), 0.0);
    EXPECT_GT(mat293(2, 2), 0.0);
    EXPECT_NEAR(mat293(0, 0), mat293(1, 1), 1e-10);
    EXPECT_NEAR(mat293(0, 0), mat293(2, 2), 1e-10);
    
    // Conductivity should decrease with temperature for metals
    EXPECT_GT(sigma_373.value()(0, 0), 0.0);
    EXPECT_GT(mat293(0, 0), sigma_373.value()(0, 0));
}

TEST_F(MaterialXmlReaderTest, MaterialNotFound) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);

    const MaterialPropertyModel* notFound = database.getMaterial("nonexistent");
    EXPECT_EQ(notFound, nullptr);

    EXPECT_FALSE(database.hasMaterial("nonexistent"));
    EXPECT_TRUE(database.hasMaterial("mat1"));
    EXPECT_TRUE(database.hasMaterial("mat2"));
}

TEST_F(MaterialXmlReaderTest, MaterialTags) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);

    auto tags = database.getMaterialTags();
    EXPECT_EQ(tags.size(), 2);
    
    // Check that both tags are present
    bool hasMat1 = false, hasMat2 = false;
    for (const auto& tag : tags) {
        if (tag == "mat1") hasMat1 = true;
        if (tag == "mat2") hasMat2 = true;
    }
    EXPECT_TRUE(hasMat1);
    EXPECT_TRUE(hasMat2);
}

TEST_F(MaterialXmlReaderTest, ReadOrder2Materials) {
    MaterialDatabase database;
    
    ASSERT_NO_THROW({
        MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady_order2/material.xml"), database);
    });

    // Should have same materials as order 1 case
    EXPECT_EQ(database.size(), 2);
    EXPECT_TRUE(database.hasMaterial("mat1"));
    EXPECT_TRUE(database.hasMaterial("mat2"));
}

TEST_F(MaterialXmlReaderTest, ConstantConductivity) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);

    // Titanium has constant (non-temperature-dependent) conductivity
    const MaterialPropertyModel* titanium = database.getMaterial("mat2");
    ASSERT_NE(titanium, nullptr);

    EXPECT_TRUE(titanium->hasMatrix("electricconductivity"));
    EXPECT_FALSE(titanium->hasMatrixExpression("electricconductivity"));
    
    // Evaluate without temperature variable (should still work)
    std::map<std::string, double> emptyVars;
    auto sigma = titanium->evaluateMatrix("electricconductivity", emptyVars);
    EXPECT_TRUE(sigma.has_value());
    
    // Check diagonal matrix
    const auto& mat = sigma.value();
    EXPECT_GT(mat(0, 0), 0.0);
    EXPECT_NEAR(mat(0, 0), mat(1, 1), 1e-10);
    EXPECT_NEAR(mat(0, 0), mat(2, 2), 1e-10);
}
