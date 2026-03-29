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
    // Electric conductivity is an expression (temperature-dependent)
    EXPECT_TRUE(copper->hasMatrix("electricconductivity"));
    
    // Thermal conductivity is a constant matrix
    EXPECT_TRUE(copper->hasMatrix("thermalconductivity"));
    // Note: hasMatrix() returns true for both expressions and constants

    // Check mat2 (Titanium)
    const MaterialPropertyModel* titanium = database.getMaterial("mat2");
    ASSERT_NE(titanium, nullptr);
    
    // Titanium has constant electric conductivity (no expression)
    EXPECT_TRUE(titanium->hasMatrix("electricconductivity"));
    EXPECT_TRUE(titanium->hasMatrix("thermalconductivity"));
}

TEST_F(MaterialXmlReaderTest, MaterialPropertyAccess) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);

    const MaterialPropertyModel* copper = database.getMaterial("mat1");
    ASSERT_NE(copper, nullptr);

    // Test getScalar - returns value directly, throws if not found
    double E = copper->getScalar("E");
    EXPECT_GT(E, 0.0);

    double nu = copper->getScalar("nu");
    EXPECT_GT(nu, 0.0);
    EXPECT_LT(nu, 1.0);  // Poisson ratio should be between 0 and 1

    // Test non-existent property throws
    EXPECT_THROW(copper->getScalar("nonexistent_property"), ArgumentException);
}

TEST_F(MaterialXmlReaderTest, TemperatureDependentConductivity) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);

    const MaterialPropertyModel* copper = database.getMaterial("mat1");
    ASSERT_NE(copper, nullptr);

    // Copper has temperature-dependent conductivity (expression)
    EXPECT_TRUE(copper->hasMatrix("electricconductivity"));

    // Evaluate at different temperatures
    std::map<std::string, double> vars;
    
    vars["T"] = 293.15;
    Matrix3 sigma_293 = copper->getMatrix("electricconductivity", vars);
    
    vars["T"] = 373.15;
    Matrix3 sigma_373 = copper->getMatrix("electricconductivity", vars);
    
    // Check that matrices are diagonal (isotropic case)
    EXPECT_GT(sigma_293(0, 0), 0.0);
    EXPECT_GT(sigma_293(1, 1), 0.0);
    EXPECT_GT(sigma_293(2, 2), 0.0);
    EXPECT_NEAR(sigma_293(0, 0), sigma_293(1, 1), 1e-10);
    EXPECT_NEAR(sigma_293(0, 0), sigma_293(2, 2), 1e-10);
    
    // Conductivity should decrease with temperature for metals
    EXPECT_GT(sigma_373(0, 0), 0.0);
    EXPECT_GT(sigma_293(0, 0), sigma_373(0, 0));
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
    
    // Evaluate without temperature variable (should still work for constant)
    std::map<std::string, double> emptyVars;
    Matrix3 sigma = titanium->getMatrix("electricconductivity", emptyVars);
    
    // Check diagonal matrix
    EXPECT_GT(sigma(0, 0), 0.0);
    EXPECT_NEAR(sigma(0, 0), sigma(1, 1), 1e-10);
    EXPECT_NEAR(sigma(0, 0), sigma(2, 2), 1e-10);
}