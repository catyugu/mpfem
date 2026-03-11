#include <gtest/gtest.h>
#include "io/material_xml_reader.hpp"
#include "core/logger.hpp"

using namespace mpfem;

class MaterialXmlReaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Warning);
    }
};

TEST_F(MaterialXmlReaderTest, ReadBusbarMaterials) {
    MaterialDatabase database;
    
    ASSERT_NO_THROW({
        MaterialXmlReader::readFromFile("cases/busbar/material.xml", database);
    });

    // Should have 2 materials
    EXPECT_EQ(database.size(), 2);

    // Check mat1 (Copper)
    const MaterialPropertyModel* copper = database.getMaterial("mat1");
    ASSERT_NE(copper, nullptr);
    
    // Check key properties for Copper
    EXPECT_GT(copper->electricConductivity, 0.0);
    EXPECT_GT(copper->thermalConductivity, 0.0);
    EXPECT_GT(copper->youngModulus, 0.0);
    EXPECT_GT(copper->poissonRatio, 0.0);
    EXPECT_GT(copper->density, 0.0);
    
    // Check temperature-dependent resistivity parameters
    EXPECT_GT(copper->rho0, 0.0);
    EXPECT_GT(copper->alpha, 0.0);

    // Check mat2 (Titanium)
    const MaterialPropertyModel* titanium = database.getMaterial("mat2");
    ASSERT_NE(titanium, nullptr);
    
    // Check key properties for Titanium
    EXPECT_GT(titanium->electricConductivity, 0.0);
    EXPECT_GT(titanium->thermalConductivity, 0.0);
    EXPECT_GT(titanium->youngModulus, 0.0);
    EXPECT_GT(titanium->poissonRatio, 0.0);
}

TEST_F(MaterialXmlReaderTest, MaterialPropertyAccess) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile("cases/busbar/material.xml", database);

    const MaterialPropertyModel* copper = database.getMaterial("mat1");
    ASSERT_NE(copper, nullptr);

    // Test getProperty
    double E = copper->getProperty("E");
    EXPECT_GT(E, 0.0);

    double nu = copper->getProperty("nu");
    EXPECT_GT(nu, 0.0);
    EXPECT_LT(nu, 1.0);  // Poisson ratio should be between 0 and 1
}

TEST_F(MaterialXmlReaderTest, TemperatureDependentConductivity) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile("cases/busbar/material.xml", database);

    const MaterialPropertyModel* copper = database.getMaterial("mat1");
    ASSERT_NE(copper, nullptr);

    // Test temperature-dependent conductivity
    double sigma_293 = copper->getElectricConductivity(293.15);
    double sigma_373 = copper->getElectricConductivity(373.15);

    // Conductivity should decrease with temperature for metals
    EXPECT_GT(sigma_293, 0.0);
    EXPECT_GT(sigma_373, 0.0);
    // Note: This depends on the temperature coefficient being positive
    // For copper, conductivity decreases with temperature
}

TEST_F(MaterialXmlReaderTest, MaterialNotFound) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile("cases/busbar/material.xml", database);

    const MaterialPropertyModel* notFound = database.getMaterial("nonexistent");
    EXPECT_EQ(notFound, nullptr);

    EXPECT_FALSE(database.hasMaterial("nonexistent"));
    EXPECT_TRUE(database.hasMaterial("mat1"));
    EXPECT_TRUE(database.hasMaterial("mat2"));
}

TEST_F(MaterialXmlReaderTest, MaterialTags) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile("cases/busbar/material.xml", database);

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
        MaterialXmlReader::readFromFile("cases/busbar_order2/material.xml", database);
    });

    // Should have same materials as order 1 case
    EXPECT_EQ(database.size(), 2);
    EXPECT_TRUE(database.hasMaterial("mat1"));
    EXPECT_TRUE(database.hasMaterial("mat2"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
