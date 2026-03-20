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
        MaterialXmlReader::readFromFile(dataPath("cases/busbar/material.xml"), database);
    });

    // Should have 2 materials
    EXPECT_EQ(database.size(), 2);

    // Check mat1 (Copper)
    const MaterialPropertyModel* copper = database.getMaterial("mat1");
    ASSERT_NE(copper, nullptr);
    
    // Check key properties for Copper (conductivities are now Matrix3)
    EXPECT_TRUE(copper->electricConductivity.has_value());
    EXPECT_TRUE(copper->thermalConductivity.has_value());
    EXPECT_GT(copper->youngModulus.value_or(0.0), 0.0);
    EXPECT_GT(copper->poissonRatio.value_or(0.0), 0.0);
    EXPECT_GT(copper->density.value_or(0.0), 0.0);
    
    // Check temperature-dependent resistivity parameters
    EXPECT_GT(copper->rho0.value_or(0.0), 0.0);
    EXPECT_GT(copper->alpha.value_or(0.0), 0.0);

    // Check mat2 (Titanium)
    const MaterialPropertyModel* titanium = database.getMaterial("mat2");
    ASSERT_NE(titanium, nullptr);
    
    // Check key properties for Titanium
    EXPECT_TRUE(titanium->electricConductivity.has_value());
    EXPECT_TRUE(titanium->thermalConductivity.has_value());
    EXPECT_GT(titanium->youngModulus.value_or(0.0), 0.0);
    EXPECT_GT(titanium->poissonRatio.value_or(0.0), 0.0);
}

TEST_F(MaterialXmlReaderTest, MaterialPropertyAccess) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar/material.xml"), database);

    const MaterialPropertyModel* copper = database.getMaterial("mat1");
    ASSERT_NE(copper, nullptr);

    // Test getProperty (now returns std::optional)
    auto E = copper->getProperty("E");
    EXPECT_TRUE(E.has_value());
    EXPECT_GT(E.value(), 0.0);

    auto nu = copper->getProperty("nu");
    EXPECT_TRUE(nu.has_value());
    EXPECT_GT(nu.value(), 0.0);
    EXPECT_LT(nu.value(), 1.0);  // Poisson ratio should be between 0 and 1

    // Test non-existent property returns nullopt
    auto nonexistent = copper->getProperty("nonexistent_property");
    EXPECT_FALSE(nonexistent.has_value());
}

TEST_F(MaterialXmlReaderTest, TemperatureDependentConductivity) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar/material.xml"), database);

    const MaterialPropertyModel* copper = database.getMaterial("mat1");
    ASSERT_NE(copper, nullptr);

    // Test temperature-dependent conductivity (now returns std::optional<Matrix3>)
    auto sigma_293 = copper->getElectricConductivityMatrix(293.15);
    auto sigma_373 = copper->getElectricConductivityMatrix(373.15);

    // Conductivity should decrease with temperature for metals
    EXPECT_TRUE(sigma_293.has_value());
    EXPECT_TRUE(sigma_373.has_value());
    
    // Check that matrix is diagonal (isotropic case)
    const auto& mat293 = sigma_293.value();
    EXPECT_GT(mat293(0, 0), 0.0);
    EXPECT_GT(mat293(1, 1), 0.0);
    EXPECT_GT(mat293(2, 2), 0.0);
    EXPECT_NEAR(mat293(0, 0), mat293(1, 1), 1e-10);
    EXPECT_NEAR(mat293(0, 0), mat293(2, 2), 1e-10);
}

TEST_F(MaterialXmlReaderTest, MaterialNotFound) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar/material.xml"), database);

    const MaterialPropertyModel* notFound = database.getMaterial("nonexistent");
    EXPECT_EQ(notFound, nullptr);

    EXPECT_FALSE(database.hasMaterial("nonexistent"));
    EXPECT_TRUE(database.hasMaterial("mat1"));
    EXPECT_TRUE(database.hasMaterial("mat2"));
}

TEST_F(MaterialXmlReaderTest, MaterialTags) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar/material.xml"), database);

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
        MaterialXmlReader::readFromFile(dataPath("cases/busbar_order2/material.xml"), database);
    });

    // Should have same materials as order 1 case
    EXPECT_EQ(database.size(), 2);
    EXPECT_TRUE(database.hasMaterial("mat1"));
    EXPECT_TRUE(database.hasMaterial("mat2"));
}
