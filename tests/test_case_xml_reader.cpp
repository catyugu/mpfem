#include "core/logger.hpp"
#include "io/case_xml_reader.hpp"
#include <gtest/gtest.h>

using namespace mpfem;

class CaseXmlReaderTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        Logger::setLevel(LogLevel::Warning);
    }

    // Helper to get test data path
    static std::string dataPath(const std::string& relativePath)
    {
#ifdef MPFEM_PROJECT_ROOT
        return std::string(MPFEM_PROJECT_ROOT) + "/" + relativePath;
#else
        return relativePath;
#endif
    }
};

TEST_F(CaseXmlReaderTest, ReadBusbarCase)
{
    CaseDefinition caseDef;

    // Read the actual case file
    ASSERT_NO_THROW({
        CaseXmlReader::readFromFile(dataPath("cases/busbar_steady/case.xml"), caseDef);
    });

    // Verify case name
    EXPECT_EQ(caseDef.caseName, "busbar");

    // Verify study type
    EXPECT_EQ(caseDef.studyType, "steady");

    // Verify paths
    EXPECT_EQ(caseDef.meshPath, "mesh.mphtxt");
    EXPECT_EQ(caseDef.materialsPath, "material.xml");
    EXPECT_EQ(caseDef.comsolResultPath, "result.txt");

    // Verify variables
    EXPECT_GE(caseDef.variables.size(), 4);

    bool foundVtot = false;
    bool foundL = false;
    bool foundHtc = false;

    for (const auto& v : caseDef.variables) {
        if (v.name == "Vtot") {
            foundVtot = true;
            EXPECT_EQ(v.valueText, "20[mV]");
        }
        if (v.name == "L") {
            foundL = true;
            EXPECT_EQ(v.valueText, "9[cm]");
        }
        if (v.name == "htc") {
            foundHtc = true;
            EXPECT_EQ(v.valueText, "5[W/m^2/K]");
        }
    }
    EXPECT_TRUE(foundVtot);
    EXPECT_TRUE(foundL);
    EXPECT_TRUE(foundHtc);

    // Verify material assignments
    EXPECT_GE(caseDef.materialAssignments.size(), 2);

    // Verify physics definitions (should have 3: electrostatics, heat_transfer, solid_mechanics)
    EXPECT_EQ(caseDef.physics.size(), 3);

    // Check electrostatics physics
    ASSERT_TRUE(caseDef.physics.count("electrostatics") > 0);
    const auto& electrostatics = caseDef.physics.at("electrostatics");
    EXPECT_EQ(electrostatics.order, 1);
    ASSERT_NE(electrostatics.solver, nullptr);
    EXPECT_EQ(electrostatics.solver->type, OperatorType::CG);
    ASSERT_NE(electrostatics.solver->preconditioner, nullptr);
    EXPECT_EQ(electrostatics.solver->preconditioner->type, OperatorType::Diagonal);
    EXPECT_GE(electrostatics.boundaries.size(), 2); // At least voltage and ground

    // Check heat transfer physics
    ASSERT_TRUE(caseDef.physics.count("heat_transfer") > 0);
    const auto& heatTransfer = caseDef.physics.at("heat_transfer");
    EXPECT_EQ(heatTransfer.order, 1);
    ASSERT_NE(heatTransfer.solver, nullptr);
    EXPECT_EQ(heatTransfer.solver->type, OperatorType::CG);
    ASSERT_NE(heatTransfer.solver->preconditioner, nullptr);
    EXPECT_EQ(heatTransfer.solver->preconditioner->type, OperatorType::AdditiveSchwarz);
    ASSERT_NE(heatTransfer.solver->preconditioner->localSolver, nullptr);
    EXPECT_EQ(heatTransfer.solver->preconditioner->localSolver->type, OperatorType::Diagonal);
    EXPECT_GE(heatTransfer.boundaries.size(), 1); // At least convection

    // Check solid mechanics physics
    ASSERT_TRUE(caseDef.physics.count("solid_mechanics") > 0);
    const auto& solidMechanics = caseDef.physics.at("solid_mechanics");
    EXPECT_EQ(solidMechanics.order, 1);
    ASSERT_NE(solidMechanics.solver, nullptr);
    EXPECT_EQ(solidMechanics.solver->type, OperatorType::Umfpack);
    EXPECT_EQ(solidMechanics.solver->preconditioner, nullptr);
    EXPECT_GE(solidMechanics.boundaries.size(), 1); // At least fixed constraint

    // Verify coupled physics definitions
    EXPECT_GE(caseDef.coupledPhysicsDefinitions.size(), 2);

    // Verify coupling config
    EXPECT_GT(caseDef.couplingConfig.maxIterations, 0);
}

TEST_F(CaseXmlReaderTest, ReadBusbarOrder2Case)
{
    CaseDefinition caseDef;

    ASSERT_NO_THROW({
        CaseXmlReader::readFromFile(dataPath("cases/busbar_steady_order2/case.xml"), caseDef);
    });

    // Should have order 2 for all physics
    for (const auto& [kind, physics] : caseDef.physics) {
        EXPECT_EQ(physics.order, 2);
    }
}

TEST_F(CaseXmlReaderTest, VariableLookup)
{
    CaseDefinition caseDef;
    CaseXmlReader::readFromFile(dataPath("cases/busbar_steady/case.xml"), caseDef);

    EXPECT_EQ(caseDef.getVariableExpression("Vtot"), "20[mV]");
    EXPECT_EQ(caseDef.getVariableExpression("L"), "9[cm]");
    EXPECT_EQ(caseDef.getVariableExpression("htc"), "5[W/m^2/K]");

    // Non-existent variable returns empty string
    EXPECT_EQ(caseDef.getVariableExpression("nonexistent"), "");
}

TEST_F(CaseXmlReaderTest, BoundaryConditionParsing)
{
    CaseDefinition caseDef;
    CaseXmlReader::readFromFile(dataPath("cases/busbar_steady/case.xml"), caseDef);

    // Find the electrostatics physics and check voltage boundary
    if (caseDef.physics.count("electrostatics") > 0) {
        const auto& physics = caseDef.physics.at("electrostatics");
        bool foundVoltage43 = false;
        for (const auto& bc : physics.boundaries) {
            if (bc.type == "Voltage" && bc.ids.count(43) > 0) {
                foundVoltage43 = true;
                EXPECT_FALSE(bc.parameters.at("value").empty());
            }
        }
        EXPECT_TRUE(foundVoltage43);
    }
}

TEST_F(CaseXmlReaderTest, MaterialAssignmentParsing)
{
    CaseDefinition caseDef;
    CaseXmlReader::readFromFile(dataPath("cases/busbar_steady/case.xml"), caseDef);

    // Domain 1 should have mat1 (Copper)
    bool foundCopper = false;
    for (const auto& assign : caseDef.materialAssignments) {
        if (assign.domainIds.count(1) > 0) {
            foundCopper = true;
            EXPECT_EQ(assign.materialTag, "mat1");
        }
    }
    EXPECT_TRUE(foundCopper);

    // Domains 2-7 should have mat2 (Titanium)
    bool foundTitanium = false;
    for (const auto& assign : caseDef.materialAssignments) {
        if (assign.domainIds.count(2) > 0) {
            foundTitanium = true;
            EXPECT_EQ(assign.materialTag, "mat2");
        }
    }
    EXPECT_TRUE(foundTitanium);
}
