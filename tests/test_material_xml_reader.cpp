#include <gtest/gtest.h>

#include "io/material_xml_reader.hpp"
#include "expr/expression_parser.hpp"
#include "core/logger.hpp"

using namespace mpfem;

class MaterialXmlReaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Warning);
    }

    static std::string dataPath(const std::string& relativePath) {
#ifdef MPFEM_PROJECT_ROOT
        return std::string(MPFEM_PROJECT_ROOT) + "/" + relativePath;
#else
        return relativePath;
#endif
    }

    static void buildSimpleDomainIndex(MaterialDatabase& database) {
        std::vector<MaterialAssignment> assignments;
        assignments.push_back(MaterialAssignment{std::set<int>{1}, "mat1"});
        assignments.push_back(MaterialAssignment{std::set<int>{2}, "mat2"});
        database.buildDomainIndex(assignments);
    }
};

TEST_F(MaterialXmlReaderTest, ReadBusbarMaterials) {
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

TEST_F(MaterialXmlReaderTest, DomainScalarPropertyAccess) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);
    buildSimpleDomainIndex(database);

    const std::string& eExpr = database.scalarExpressionByDomain(1, "E");
    const std::string& nuExpr = database.scalarExpressionByDomain(1, "nu");

    ExpressionParser parser;
    const auto eProgram = parser.compileScalar(eExpr);
    const auto nuProgram = parser.compileScalar(nuExpr);
    const double E = eProgram.evaluate({});
    const double nu = nuProgram.evaluate({});

    EXPECT_GT(E, 0.0);
    EXPECT_GT(nu, 0.0);
    EXPECT_LT(nu, 1.0);

    EXPECT_THROW(static_cast<void>(database.scalarExpressionByDomain(1, "nonexistent_property")), ArgumentException);
}

TEST_F(MaterialXmlReaderTest, TemperatureDependentConductivityByDomain) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);
    buildSimpleDomainIndex(database);

    const std::string& sigmaExpr = database.matrixExpressionByDomain(1, "electricconductivity");

    std::unordered_map<std::string, double> vars;
    vars["T"] = 293.15;
    ExpressionParser parser;
    const auto sigmaProgram = parser.compileMatrix(sigmaExpr);
    Matrix3 sigma293 = sigmaProgram.evaluate(vars);

    vars["T"] = 373.15;
    Matrix3 sigma373 = sigmaProgram.evaluate(vars);

    EXPECT_GT(sigma293(0, 0), 0.0);
    EXPECT_GT(sigma293(1, 1), 0.0);
    EXPECT_GT(sigma293(2, 2), 0.0);
    EXPECT_NEAR(sigma293(0, 0), sigma293(1, 1), 1e-10);
    EXPECT_NEAR(sigma293(0, 0), sigma293(2, 2), 1e-10);

    EXPECT_GT(sigma373(0, 0), 0.0);
    EXPECT_GT(sigma293(0, 0), sigma373(0, 0));
}

TEST_F(MaterialXmlReaderTest, InvalidDomainOrPropertyThrows) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);
    buildSimpleDomainIndex(database);

    EXPECT_THROW(static_cast<void>(database.scalarExpressionByDomain(999, "E")), ArgumentException);
    EXPECT_THROW(static_cast<void>(database.matrixExpressionByDomain(999, "electricconductivity")), ArgumentException);
    EXPECT_THROW(static_cast<void>(database.matrixExpressionByDomain(2, "nonexistent_matrix")), ArgumentException);
}

TEST_F(MaterialXmlReaderTest, ReadOrder2Materials) {
    MaterialDatabase database;

    ASSERT_NO_THROW({
        MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady_order2/material.xml"), database);
    });

    EXPECT_EQ(database.size(), 2);

    buildSimpleDomainIndex(database);
    EXPECT_NO_THROW(static_cast<void>(database.matrixExpressionByDomain(1, "electricconductivity")));
    EXPECT_NO_THROW(static_cast<void>(database.matrixExpressionByDomain(2, "thermalconductivity")));
}

TEST_F(MaterialXmlReaderTest, ConstantConductivityByDomain) {
    MaterialDatabase database;
    MaterialXmlReader::readFromFile(dataPath("cases/busbar_steady/material.xml"), database);
    buildSimpleDomainIndex(database);

    const std::string& sigmaExpr = database.matrixExpressionByDomain(2, "electricconductivity");

    ExpressionParser parser;
    const auto sigmaProgram = parser.compileMatrix(sigmaExpr);
    Matrix3 sigma = sigmaProgram.evaluate({});

    EXPECT_GT(sigma(0, 0), 0.0);
    EXPECT_NEAR(sigma(0, 0), sigma(1, 1), 1e-10);
    EXPECT_NEAR(sigma(0, 0), sigma(2, 2), 1e-10);
}
