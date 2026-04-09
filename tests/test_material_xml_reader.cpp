#include <gtest/gtest.h>

#include "core/logger.hpp"
#include "core/types.hpp"
#include "io/material_xml_reader.hpp"

using namespace mpfem;

class MaterialXmlReaderTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        Logger::setLevel(LogLevel::Warning);
    }

    static std::string dataPath(const std::string& relativePath)
    {
#ifdef MPFEM_PROJECT_ROOT
        return std::string(MPFEM_PROJECT_ROOT) + "/" + relativePath;
#else
        return relativePath;
#endif
    }

    static void buildSimpleDomainIndex(MaterialDatabase& database)
    {
        std::vector<MaterialAssignment> assignments;
        assignments.push_back(MaterialAssignment {std::set<int> {1}, "mat1"});
        assignments.push_back(MaterialAssignment {std::set<int> {2}, "mat2"});
        database.buildDomainIndex(assignments);
    }
};

TEST_F(MaterialXmlReaderTest, ReadBusbarMaterials)
{
    MaterialDatabase database;

    ASSERT_NO_THROW({
        MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);
    });

    EXPECT_EQ(database.size(), 2);

    buildSimpleDomainIndex(database);
    ASSERT_EQ(database.domainIds().size(), 2);
    EXPECT_EQ(database.domainIds()[0], 1);
    EXPECT_EQ(database.domainIds()[1], 2);

    EXPECT_NO_THROW({
        static_cast<void>(database.matrixExpressionByDomain(1, "electricconductivity"));
        static_cast<void>(database.matrixExpressionByDomain(1, "thermalconductivity"));
        static_cast<void>(database.matrixExpressionByDomain(2, "electricconductivity"));
        static_cast<void>(database.matrixExpressionByDomain(2, "thermalconductivity"));
    });
}

TEST_F(MaterialXmlReaderTest, DomainScalarPropertyExpressionStrings)
{
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);
    buildSimpleDomainIndex(database);

    // MaterialDatabase now only stores and returns expression strings
    // Actual evaluation is done through VariableManager
    const std::string& eExpr = database.scalarExpressionByDomain(1, "E");
    const std::string& nuExpr = database.scalarExpressionByDomain(1, "nu");

    // Just verify the expression strings are non-empty
    EXPECT_FALSE(eExpr.empty());
    EXPECT_FALSE(nuExpr.empty());

    // E and nu are independent scalar properties in the current material.xml
    EXPECT_EQ(eExpr.find("nu"), std::string::npos);
}

TEST_F(MaterialXmlReaderTest, DomainMatrixPropertyExpressionStrings)
{
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);
    buildSimpleDomainIndex(database);

    // MaterialDatabase now only stores and returns expression strings
    const std::string& sigmaExpr = database.matrixExpressionByDomain(1, "electricconductivity");

    EXPECT_FALSE(sigmaExpr.empty());
    // Temperature-dependent conductivity should reference T
    EXPECT_NE(sigmaExpr.find("T"), std::string::npos);
}

TEST_F(MaterialXmlReaderTest, InvalidDomainOrPropertyThrows)
{
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);
    buildSimpleDomainIndex(database);

    EXPECT_THROW(static_cast<void>(database.scalarExpressionByDomain(999, "E")), ArgumentException);
    EXPECT_THROW(static_cast<void>(database.matrixExpressionByDomain(999, "electricconductivity")), ArgumentException);
    EXPECT_THROW(static_cast<void>(database.matrixExpressionByDomain(2, "nonexistent_matrix")), ArgumentException);
}

TEST_F(MaterialXmlReaderTest, ReadOrder2Materials)
{
    MaterialDatabase database;

    ASSERT_NO_THROW({
        MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady_order2/material.xml"), database);
    });

    EXPECT_EQ(database.size(), 2);

    buildSimpleDomainIndex(database);
    EXPECT_NO_THROW(static_cast<void>(database.matrixExpressionByDomain(1, "electricconductivity")));
    EXPECT_NO_THROW(static_cast<void>(database.matrixExpressionByDomain(2, "thermalconductivity")));
}

TEST_F(MaterialXmlReaderTest, ConstantConductivityExpressionString)
{
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);
    buildSimpleDomainIndex(database);

    // Domain 2 has constant (non-temperature-dependent) electric conductivity
    const std::string& sigmaExpr = database.matrixExpressionByDomain(2, "electricconductivity");

    EXPECT_FALSE(sigmaExpr.empty());
    // Constant expression should not reference T
    EXPECT_EQ(sigmaExpr.find("T"), std::string::npos);
}