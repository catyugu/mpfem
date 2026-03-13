/**
 * @file busbar_example.cpp
 * @brief Example demonstrating electro-thermal coupling for the busbar test case.
 * 
 * This example shows how to:
 * 1. Read a mesh from mphtxt format
 * 2. Read case definition from XML
 * 3. Build and solve coupled electro-thermal problem
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
    
    LOG_INFO << "=== Busbar Electro-Thermal Example ===";
    LOG_INFO << "Case directory: " << caseDir;
    
    try {
        // Build physics problem from case directory
        PhysicsProblemSetup setup = PhysicsProblemBuilder::build(caseDir);
        
        // Check if we have coupled problem
        if (setup.couplingManager) {
            LOG_INFO << "Running coupled electro-thermal solve...";
            
            // First, solve electrostatics alone to get initial Joule heating
            LOG_INFO << "Step 1: Solving electrostatics with constant conductivity...";
            setup.electrostatics->assemble();
            bool success = setup.electrostatics->solve();
            if (!success) {
                LOG_ERROR << "Initial electrostatics solve failed";
                return 1;
            }
            
            // Then solve heat transfer with Joule heating
            LOG_INFO << "Step 2: Solving heat transfer with Joule heating...";
            setup.heatTransfer->setJouleHeating(
                &setup.electrostatics->field(),
                dynamic_cast<const PWConstCoefficient*>(setup.electrostatics->conductivity())
            );
            setup.heatTransfer->assemble();
            success = setup.heatTransfer->solve();
            if (!success) {
                LOG_ERROR << "Heat transfer solve failed";
                return 1;
            }
            
            LOG_INFO << "Temperature range: [" << setup.heatTransfer->minValue() 
                     << ", " << setup.heatTransfer->maxValue() << "] K";
            
            // Run coupled iteration
            LOG_INFO << "Step 3: Running coupled iteration...";
            CouplingResult result = setup.couplingManager->solve();
            
            if (!result.converged) {
                LOG_WARN << "Coupling did not converge after " << result.iterations 
                         << " iterations, residual = " << result.residual;
            } else {
                LOG_INFO << "Coupling converged in " << result.iterations << " iterations";
            }
            
            // Print results
            if (setup.hasElectrostatics()) {
                Real minV = setup.electrostatics->minValue();
                Real maxV = setup.electrostatics->maxValue();
                LOG_INFO << "Potential range: [" << minV << ", " << maxV << "] V";
            }
            
            if (setup.hasHeatTransfer()) {
                Real minT = setup.heatTransfer->minValue();
                Real maxT = setup.heatTransfer->maxValue();
                LOG_INFO << "Temperature range: [" << minT << ", " << maxT << "] K";
                LOG_INFO << "Temperature range: [" << (minT - 273.15) << ", " 
                         << (maxT - 273.15) << "] °C";
            }
            
        } else if (setup.hasElectrostatics()) {
            // Single physics solve
            LOG_INFO << "Running single physics electrostatics solve...";
            
            auto& solver = setup.electrostatics;
            
            LOG_INFO << "Assembling system...";
            solver->assemble();
            
            LOG_INFO << "Solving...";
            bool success = solver->solve();
            
            if (!success) {
                LOG_ERROR << "Solver failed to converge";
                return 1;
            }
            
            Real minV = solver->minValue();
            Real maxV = solver->maxValue();
            LOG_INFO << "Potential range: [" << minV << ", " << maxV << "] V";
        }
        
        // Export results to VTU
        std::filesystem::create_directories("results");
        std::string outputPath = "results/busbar_results.vtu";
        
        std::vector<FieldResult> fields;
        
        if (setup.hasElectrostatics()) {
            const GridFunction& V = setup.electrostatics->field();
            FieldResult potentialField;
            potentialField.name = "V";
            potentialField.unit = "V";
            potentialField.nodalValues.resize(V.numDofs());
            for (Index i = 0; i < V.numDofs(); ++i) {
                potentialField.nodalValues[i] = V.values()(i);
            }
            fields.push_back(potentialField);
        }
        
        if (setup.hasHeatTransfer()) {
            const GridFunction& T = setup.heatTransfer->field();
            FieldResult tempField;
            tempField.name = "T";
            tempField.unit = "K";
            tempField.nodalValues.resize(T.numDofs());
            for (Index i = 0; i < T.numDofs(); ++i) {
                tempField.nodalValues[i] = T.values()(i);
            }
            fields.push_back(tempField);
            
            // Also add temperature in Celsius
            FieldResult tempCField;
            tempCField.name = "T_C";
            tempCField.unit = "degC";
            tempCField.nodalValues.resize(T.numDofs());
            for (Index i = 0; i < T.numDofs(); ++i) {
                tempCField.nodalValues[i] = T.values()(i) - 273.15;
            }
            fields.push_back(tempCField);
        }
        
        ResultExporter::exportVtu(outputPath, *setup.mesh, fields);
        
        LOG_INFO << "Results exported to: " << outputPath;
        LOG_INFO << "=== Example completed successfully! ===";
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR << "Error: " << e.what();
        return 1;
    }
}
