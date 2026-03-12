/**
 * @file busbar_example.cpp
 * @brief Example demonstrating electrostatics solving for the busbar test case.
 * 
 * This example shows how to:
 * 1. Read a mesh from mphtxt format
 * 2. Read case definition from XML
 * 3. Build and solve electrostatics problem
 * 4. Export results to VTU format for visualization
 */

#include "mesh/mesh.hpp"
#include "mesh/io/mphtxt_reader.hpp"
#include "mesh/mesh_topology.hpp"
#include "fe/fe_collection.hpp"
#include "fe/fe_space.hpp"
#include "physics/physics_problem_builder.hpp"
#include "io/case_xml_reader.hpp"
#include "io/material_xml_reader.hpp"
#include "io/result_exporter.hpp"
#include "core/logger.hpp"
#include <filesystem>

using namespace mpfem;

int main(int argc, char* argv[]) {
    Logger::setLevel(LogLevel::Info);
    
    // Default case directory
    std::string caseDir = "cases/busbar";
    
    // Override from command line if provided
    if (argc > 1) {
        caseDir = argv[1];
    }
    
    LOG_INFO << "=== Busbar Electrostatics Example ===";
    LOG_INFO << "Case directory: " << caseDir;
    
    try {
        // Build physics problem from case directory
        PhysicsProblemSetup setup = PhysicsProblemBuilder::build(caseDir);
        
        if (!setup.hasElectrostatics()) {
            LOG_ERROR << "No electrostatics physics found in case";
            return 1;
        }
        
        auto& solver = setup.electrostatics;
        
        // Assemble and solve
        LOG_INFO << "Assembling system...";
        solver->assemble();
        
        LOG_INFO << "Solving...";
        bool success = solver->solve();
        
        if (!success) {
            LOG_ERROR << "Solver failed to converge";
            return 1;
        }
        
        // Print results
        Real minV = solver->minValue();
        Real maxV = solver->maxValue();
        
        LOG_INFO << "=== Results ===";
        LOG_INFO << "Potential range: [" << minV << ", " << maxV << "] V";
        LOG_INFO << "Expected range: [0, 0.02] V";
        
        // Verify results
        if (minV < -1e-6) {
            LOG_WARN << "Minimum potential is negative: " << minV;
        }
        if (std::abs(maxV - 0.02) > 0.001) {
            LOG_WARN << "Maximum potential differs from expected: " << maxV << " vs 0.02";
        } else {
            LOG_INFO << "Potential range is correct!";
        }
        
        LOG_INFO << "Solver iterations: " << solver->iterations();
        LOG_INFO << "Solver residual: " << solver->residual();
        
        // Export results to VTU
        std::filesystem::create_directories("results");
        std::string outputPath = "results/busbar_electrostatics.vtu";
        
        // Prepare field data
        const GridFunction& V = solver->field();
        FieldResult potentialField;
        potentialField.name = "V";
        potentialField.unit = "V";
        potentialField.nodalValues.resize(V.numDofs());
        for (Index i = 0; i < V.numDofs(); ++i) {
            potentialField.nodalValues[i] = V.values()(i);
        }
        
        std::vector<FieldResult> fields = {potentialField};
        ResultExporter::exportVtu(outputPath, *setup.mesh, fields);
        
        LOG_INFO << "Results exported to: " << outputPath;
        LOG_INFO << "=== Example completed successfully! ===";
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR << "Error: " << e.what();
        return 1;
    }
}
