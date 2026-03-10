/**
 * @file test_config.cpp
 * @brief Tests for case configuration and material parsing
 */

#include <gtest/gtest.h>
#include "config/case_parser.hpp"
#include "config/case_config.hpp"
#include "material/material_database.hpp"
#include "core/logger.hpp"
#include <filesystem>

namespace fs = std::filesystem;

namespace mpfem {
namespace test {

class ConfigTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        Logger::instance().set_level(LogLevel::INFO);
    }
    
    std::string get_cases_dir() {
        // Find cases directory relative to current working directory
        return "cases/busbar";
    }
};

TEST_F(ConfigTest, ParseCaseXml) {
    CaseParser parser;
    std::string case_file = "cases/busbar/case.xml";
    
    ASSERT_TRUE(fs::exists(case_file)) << "Case file not found: " << case_file;
    
    CaseConfig config = parser.parse_case(case_file, "cases/busbar");
    
    EXPECT_TRUE(parser.success()) << "Parse failed: " << parser.error();
    EXPECT_EQ(config.name, "busbar");
    EXPECT_EQ(config.study_type, "steady");
}

TEST_F(ConfigTest, ParseVariables) {
    CaseParser parser;
    CaseConfig config = parser.parse_case("cases/busbar/case.xml", "cases/busbar");
    
    EXPECT_TRUE(parser.success());
    
    // Check variables
    EXPECT_TRUE(config.has_variable("Vtot"));
    EXPECT_TRUE(config.has_variable("htc"));
    EXPECT_TRUE(config.has_variable("L"));
    
    // Check variable values
    EXPECT_DOUBLE_EQ(config.get_variable("Vtot"), 0.02);
    EXPECT_DOUBLE_EQ(config.get_variable("htc"), 5.0);
    EXPECT_DOUBLE_EQ(config.get_variable("L"), 0.09);
}

TEST_F(ConfigTest, ParseMaterialAssignments) {
    CaseParser parser;
    CaseConfig config = parser.parse_case("cases/busbar/case.xml", "cases/busbar");
    
    EXPECT_TRUE(parser.success());
    ASSERT_EQ(config.material_assignments.size(), 2);
    
    // First assignment: domain 1 -> mat1
    EXPECT_EQ(config.material_assignments[0].material, "mat1");
    ASSERT_EQ(config.material_assignments[0].domains.size(), 1);
    EXPECT_EQ(config.material_assignments[0].domains[0], 1);
    
    // Second assignment: domains 2-7 -> mat2
    EXPECT_EQ(config.material_assignments[1].material, "mat2");
    ASSERT_EQ(config.material_assignments[1].domains.size(), 6);
    EXPECT_EQ(config.material_assignments[1].domains[0], 2);
    EXPECT_EQ(config.material_assignments[1].domains[5], 7);
}

TEST_F(ConfigTest, ParsePhysics) {
    CaseParser parser;
    CaseConfig config = parser.parse_case("cases/busbar/case.xml", "cases/busbar");
    
    EXPECT_TRUE(parser.success());
    ASSERT_EQ(config.physics.size(), 3);
    
    // Check electrostatics
    const PhysicsConfig* elec = config.get_physics("electrostatics");
    ASSERT_NE(elec, nullptr);
    EXPECT_EQ(elec->order, 1);
    EXPECT_EQ(elec->solver.type, "direct");
    
    // Check boundary conditions
    ASSERT_GE(elec->boundaries.size(), 3);
    
    // Find voltage BC
    bool found_voltage_bc = false;
    for (const auto& bc : elec->boundaries) {
        if (bc.kind == "voltage") {
            found_voltage_bc = true;
            EXPECT_FALSE(bc.ids.empty());
            EXPECT_TRUE(bc.params.count("value") > 0);
        }
    }
    EXPECT_TRUE(found_voltage_bc);
    
    // Check heat transfer
    const PhysicsConfig* heat = config.get_physics("heat_transfer");
    ASSERT_NE(heat, nullptr);
    EXPECT_EQ(heat->order, 1);
    
    // Check solid mechanics
    const PhysicsConfig* mech = config.get_physics("solid_mechanics");
    ASSERT_NE(mech, nullptr);
    EXPECT_EQ(mech->order, 1);
}

TEST_F(ConfigTest, ParseCoupling) {
    CaseParser parser;
    CaseConfig config = parser.parse_case("cases/busbar/case.xml", "cases/busbar");
    
    EXPECT_TRUE(parser.success());
    
    // Check coupled physics
    ASSERT_EQ(config.coupled_physics.size(), 2);
    
    // Joule heating
    EXPECT_EQ(config.coupled_physics[0].kind, "joule_heating");
    ASSERT_EQ(config.coupled_physics[0].physics.size(), 2);
    
    // Thermal expansion
    EXPECT_EQ(config.coupled_physics[1].kind, "thermal_expansion");
    
    // Check coupling solver config
    EXPECT_EQ(config.coupling.method, "picard");
    EXPECT_EQ(config.coupling.max_iter, 20);
}

TEST_F(ConfigTest, ParseMaterialXml) {
    CaseParser parser;
    std::string mat_file = "cases/busbar/material.xml";
    
    ASSERT_TRUE(fs::exists(mat_file)) << "Material file not found: " << mat_file;
    
    MaterialDatabase db = parser.parse_materials(mat_file);
    
    EXPECT_TRUE(parser.success());
    EXPECT_EQ(db.materials.size(), 2);
    
    // Check mat1 (Copper)
    const MaterialConfig* copper = db.get("mat1");
    ASSERT_NE(copper, nullptr);
    EXPECT_EQ(copper->label, "Copper");
    
    // Check mat2 (Titanium)
    const MaterialConfig* titanium = db.get("mat2");
    ASSERT_NE(titanium, nullptr);
    EXPECT_EQ(titanium->label, "Titanium beta-21S");
}

TEST_F(ConfigTest, MaterialProperties) {
    CaseParser parser;
    MaterialDatabase config_db = parser.parse_materials("cases/busbar/material.xml");
    
    EXPECT_TRUE(parser.success());
    
    // Build material database
    MaterialDB mat_db;
    mat_db.build(config_db);
    
    // Get copper material
    const Material* copper = mat_db.get("mat1");
    ASSERT_NE(copper, nullptr);
    
    // Check Young's modulus
    Scalar E = copper->get_youngs_modulus();
    EXPECT_DOUBLE_EQ(E, 110e9);
    
    // Check Poisson's ratio
    Scalar nu = copper->get_poissons_ratio();
    EXPECT_DOUBLE_EQ(nu, 0.35);
    
    // Check density
    MaterialEvaluator evaluator;
    Scalar rho = copper->get_density();
    EXPECT_DOUBLE_EQ(rho, 8960.0);
    
    // Check heat capacity
    Scalar cp = copper->get_heat_capacity();
    EXPECT_DOUBLE_EQ(cp, 385.0);
}

TEST_F(ConfigTest, TemperatureDependentResistivity) {
    CaseParser parser;
    MaterialDatabase config_db = parser.parse_materials("cases/busbar/material.xml");
    
    EXPECT_TRUE(parser.success());
    
    MaterialDB mat_db;
    mat_db.build(config_db);
    
    const Material* copper = mat_db.get("mat1");
    ASSERT_NE(copper, nullptr);
    
    MaterialEvaluator evaluator;
    
    // At reference temperature (298 K)
    evaluator.set_temperature(298.0);
    Scalar sigma_298 = copper->get_scalar("electricconductivity", evaluator);
    // From linearized resistivity: sigma = 1/rho0 = 1/1.72e-8 ≈ 5.814e7 S/m
    EXPECT_NEAR(sigma_298, 1.0/1.72e-8, 1e6);
    
    // At higher temperature
    evaluator.set_temperature(323.0);
    Scalar sigma_323 = copper->get_scalar("electricconductivity", evaluator);
    // Conductivity should decrease with temperature
    EXPECT_LT(sigma_323, sigma_298);
}

TEST_F(ConfigTest, ThermalConductivity) {
    CaseParser parser;
    MaterialDatabase config_db = parser.parse_materials("cases/busbar/material.xml");
    
    EXPECT_TRUE(parser.success());
    
    MaterialDB mat_db;
    mat_db.build(config_db);
    
    const Material* copper = mat_db.get("mat1");
    const Material* titanium = mat_db.get("mat2");
    
    ASSERT_NE(copper, nullptr);
    ASSERT_NE(titanium, nullptr);
    
    MaterialEvaluator evaluator;
    
    // Copper thermal conductivity
    Tensor<2, 3> k_cu = copper->get_thermal_conductivity(evaluator);
    EXPECT_NEAR(k_cu(0, 0), 400.0, 1e-10);
    EXPECT_NEAR(k_cu(1, 1), 400.0, 1e-10);
    
    // Titanium thermal conductivity
    Tensor<2, 3> k_ti = titanium->get_thermal_conductivity(evaluator);
    EXPECT_NEAR(k_ti(0, 0), 7.5, 1e-10);
}

TEST_F(ConfigTest, ThermalExpansion) {
    CaseParser parser;
    MaterialDatabase config_db = parser.parse_materials("cases/busbar/material.xml");
    
    EXPECT_TRUE(parser.success());
    
    MaterialDB mat_db;
    mat_db.build(config_db);
    
    const Material* copper = mat_db.get("mat1");
    ASSERT_NE(copper, nullptr);
    
    Tensor<2, 3> alpha = copper->get_thermal_expansion();
    EXPECT_NEAR(alpha(0, 0), 17e-6, 1e-12);
    EXPECT_NEAR(alpha(1, 1), 17e-6, 1e-12);
}

TEST_F(ConfigTest, ParseIdList) {
    // Test single ID
    auto ids1 = CaseParser::parse_id_list("1");
    ASSERT_EQ(ids1.size(), 1);
    EXPECT_EQ(ids1[0], 1);
    
    // Test multiple IDs
    auto ids2 = CaseParser::parse_id_list("1,2,3,5");
    ASSERT_EQ(ids2.size(), 4);
    EXPECT_EQ(ids2[0], 1);
    EXPECT_EQ(ids2[1], 2);
    EXPECT_EQ(ids2[2], 3);
    EXPECT_EQ(ids2[3], 5);
    
    // Test range
    auto ids3 = CaseParser::parse_id_list("1-7");
    ASSERT_EQ(ids3.size(), 7);
    EXPECT_EQ(ids3[0], 1);
    EXPECT_EQ(ids3[6], 7);
    
    // Test mixed
    auto ids4 = CaseParser::parse_id_list("1-3,5,7-9");
    ASSERT_EQ(ids4.size(), 7);
    EXPECT_EQ(ids4[0], 1);
    EXPECT_EQ(ids4[2], 3);
    EXPECT_EQ(ids4[3], 5);
    EXPECT_EQ(ids4[4], 7);
    EXPECT_EQ(ids4[6], 9);
}

TEST_F(ConfigTest, ParseValueWithUnit) {
    EXPECT_DOUBLE_EQ(CaseParser::parse_value_with_unit("9[cm]"), 0.09);
    EXPECT_DOUBLE_EQ(CaseParser::parse_value_with_unit("6[mm]"), 0.006);
    EXPECT_DOUBLE_EQ(CaseParser::parse_value_with_unit("20[mV]"), 0.02);
    EXPECT_DOUBLE_EQ(CaseParser::parse_value_with_unit("110[GPa]"), 110e9);
    EXPECT_DOUBLE_EQ(CaseParser::parse_value_with_unit("298[K]"), 298.0);
    EXPECT_DOUBLE_EQ(CaseParser::parse_value_with_unit("0.35"), 0.35);
}

TEST_F(ConfigTest, ParseTensor) {
    auto tensor = CaseParser::parse_tensor("{'1','0','0','0','1','0','0','0','1'}");
    ASSERT_EQ(tensor.size(), 9);
    EXPECT_DOUBLE_EQ(tensor[0], 1.0);
    EXPECT_DOUBLE_EQ(tensor[4], 1.0);
    EXPECT_DOUBLE_EQ(tensor[8], 1.0);
    EXPECT_DOUBLE_EQ(tensor[1], 0.0);
}

}  // namespace test
}  // namespace mpfem
