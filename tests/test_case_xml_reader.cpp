#include <gtest/gtest.h>
#include "io/case_xml_reader.hpp"
#include "core/logger.hpp"

using namespace mpfem;

class CaseXmlReaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::setLevel(LogLevel::Warning);
    }
};

TEST_F(CaseXmlReaderTest, ReadBusbarCase) {
    CaseDefinition caseDef;
    
    // Read the actual case file (relative to project root)
    ASSERT_NO_THROW({
        CaseXmlReader::readFromFile("cases/busbar/case.xml", caseDef);
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
            EXPECT_DOUBLE_EQ(v.siValue, 0.02);
        }
        if (v.name == "L") {
            foundL = true;
            EXPECT_DOUBLE_EQ(v.siValue, 0.09);
        }
        if (v.name == "htc") {
            foundHtc = true;
            EXPECT_DOUBLE_EQ(v.siValue, 5.0);
        }
    }
    EXPECT_TRUE(foundVtot);
    EXPECT_TRUE(foundL);
    EXPECT_TRUE(foundHtc);

    // Verify material assignments
    EXPECT_GE(caseDef.materialAssignments.size(), 2);

    // Verify physics definitions (should have 3: electrostatics, heat_transfer, solid_mechanics)
    EXPECT_EQ(caseDef.physicsDefinitions.size(), 3);

    // Check electrostatics physics
    bool foundElectrostatics = false;
    for (const auto& physics : caseDef.physicsDefinitions) {
        if (physics.kind == "electrostatics") {
            foundElectrostatics = true;
            EXPECT_EQ(physics.order, 1);
            EXPECT_GE(physics.boundaries.size(), 2);  // At least voltage and ground
        }
    }
    EXPECT_TRUE(foundElectrostatics);

    // Check heat transfer physics
    bool foundHeatTransfer = false;
    for (const auto& physics : caseDef.physicsDefinitions) {
        if (physics.kind == "heat_transfer") {
            foundHeatTransfer = true;
            EXPECT_EQ(physics.order, 1);
            EXPECT_GE(physics.boundaries.size(), 1);  // At least convection
        }
    }
    EXPECT_TRUE(foundHeatTransfer);

    // Check solid mechanics physics
    bool foundSolidMechanics = false;
    for (const auto& physics : caseDef.physicsDefinitions) {
        if (physics.kind == "solid_mechanics") {
            foundSolidMechanics = true;
            EXPECT_EQ(physics.order, 1);
            EXPECT_GE(physics.boundaries.size(), 1);  // At least fixed constraint
        }
    }
    EXPECT_TRUE(foundSolidMechanics);

    // Verify coupled physics definitions
    EXPECT_GE(caseDef.coupledPhysicsDefinitions.size(), 2);

    // Verify coupling config
    EXPECT_EQ(caseDef.couplingConfig.method, CouplingMethod::Picard);
    EXPECT_GT(caseDef.couplingConfig.maxIterations, 0);
}

TEST_F(CaseXmlReaderTest, ReadBusbarOrder2Case) {
    CaseDefinition caseDef;
    
    ASSERT_NO_THROW({
        CaseXmlReader::readFromFile("cases/busbar_order2/case.xml", caseDef);
    });

    // Should have order 2 for all physics
    for (const auto& physics : caseDef.physicsDefinitions) {
        EXPECT_EQ(physics.order, 2);
    }
}

TEST_F(CaseXmlReaderTest, GetVariableMap) {
    CaseDefinition caseDef;
    CaseXmlReader::readFromFile("cases/busbar/case.xml", caseDef);

    auto varMap = caseDef.getVariableMap();
    
    EXPECT_DOUBLE_EQ(varMap["Vtot"], 0.02);
    EXPECT_DOUBLE_EQ(varMap["L"], 0.09);
    EXPECT_DOUBLE_EQ(varMap["htc"], 5.0);
}

TEST_F(CaseXmlReaderTest, BoundaryConditionParsing) {
    CaseDefinition caseDef;
    CaseXmlReader::readFromFile("cases/busbar/case.xml", caseDef);

    // Find the electrostatics physics and check voltage boundary
    for (const auto& physics : caseDef.physicsDefinitions) {
        if (physics.kind == "electrostatics") {
            bool foundVoltage43 = false;
            for (const auto& bc : physics.boundaries) {
                if (bc.kind == "voltage" && bc.ids.count(43) > 0) {
                    foundVoltage43 = true;
                    // Check that value parameter exists
                    EXPECT_TRUE(bc.params.count("value") > 0);
                }
            }
            EXPECT_TRUE(foundVoltage43);
        }
    }
}

TEST_F(CaseXmlReaderTest, MaterialAssignmentParsing) {
    CaseDefinition caseDef;
    CaseXmlReader::readFromFile("cases/busbar/case.xml", caseDef);

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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
