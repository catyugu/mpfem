#include "mesh/mesh.hpp"
#include "mesh/io/mphtxt_reader.hpp"
#include "fe/fe_collection.hpp"
#include "fe/fe_space.hpp"
#include "physics/physics_problem_builder.hpp"
#include "io/result_exporter.hpp"
#include "core/logger.hpp"
#include <filesystem>

using namespace mpfem;

int main(int argc, char* argv[]) {
    Logger::setLevel(LogLevel::Info);
    
    std::string caseDir = "cases/busbar";
    if (argc > 1) caseDir = argv[1];
    
    LOG_INFO << "=== Busbar Electro-Thermal Example ===";
    LOG_INFO << "Case directory: " << caseDir;
    
    try {
        PhysicsProblemSetup setup = PhysicsProblemBuilder::build(caseDir);
        
        if (setup.couplingManager) {
            LOG_INFO << "Running coupled electro-thermal solve...";
            CouplingResult result = setup.couplingManager->solve();
            
            if (!result.converged) {
                LOG_WARN << "Coupling did not converge after " << result.iterations << " iterations";
            } else {
                LOG_INFO << "Coupling converged in " << result.iterations << " iterations";
            }
        } else if (setup.hasElectrostatics()) {
            LOG_INFO << "Running single physics electrostatics solve...";
            setup.electrostatics->assemble();
            bool success = setup.electrostatics->solve();
            if (!success) {
                LOG_ERROR << "Solver failed to converge";
                return 1;
            }
        }
        
        // Print results
        if (setup.hasElectrostatics()) {
            const auto& V = setup.electrostatics->field().values();
            LOG_INFO << "Potential range: [" << V.minCoeff() 
                     << ", " << V.maxCoeff() << "] V";
        }
        if (setup.hasHeatTransfer()) {
            const auto& T = setup.heatTransfer->field().values();
            Real minT = T.minCoeff();
            Real maxT = T.maxCoeff();
            LOG_INFO << "Temperature range: [" << minT << ", " << maxT << "] K";
            LOG_INFO << "Temperature range: [" << (minT - 273.15) << ", " 
                     << (maxT - 273.15) << "] C";
        }
        
        // Export results
        std::filesystem::create_directories("results");
        std::string outputPath = "results/busbar_results.vtu";
        
        std::vector<FieldResult> fields;
        
        if (setup.hasElectrostatics()) {
            const GridFunction& V = setup.electrostatics->field();
            FieldResult f;
            f.name = "V";
            f.unit = "V";
            f.nodalValues.resize(V.numDofs());
            for (Index i = 0; i < V.numDofs(); ++i)
                f.nodalValues[i] = V.values()(i);
            fields.push_back(f);
        }
        
        if (setup.hasHeatTransfer()) {
            const GridFunction& T = setup.heatTransfer->field();
            FieldResult f;
            f.name = "T";
            f.unit = "K";
            f.nodalValues.resize(T.numDofs());
            for (Index i = 0; i < T.numDofs(); ++i)
                f.nodalValues[i] = T.values()(i);
            fields.push_back(f);
        }
        
        // Add displacement field placeholder (zeros) for comparison
        // TODO: Implement solid mechanics solver
        {
            FieldResult f;
            f.name = "disp";
            f.unit = "m";
            f.nodalValues.resize(setup.mesh->numVertices(), 0.0);
            fields.push_back(f);
        }
        
        ResultExporter::exportVtu(outputPath, *setup.mesh, fields);
        LOG_INFO << "Results exported to: " << outputPath;
        
        // Export COMSOL-format result file for comparison
        std::string comsolOutput = "results/mpfem_result.txt";
        ResultExporter::exportComsolText(comsolOutput, *setup.mesh, fields,
            "Electric potential, Temperature, Displacement magnitude");
        LOG_INFO << "COMSOL format results exported to: " << comsolOutput;
        
        LOG_INFO << "=== Example completed successfully! ===";
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR << "Error: " << e.what();
        return 1;
    }
}