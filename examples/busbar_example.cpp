#include "mesh/mesh.hpp"
#include "io/mphtxt_reader.hpp"
#include "problem/physics_problem_builder.hpp"
#include "problem/transient_problem.hpp"
#include "problem/steady_problem.hpp"
#include "io/result_exporter.hpp"
#include "core/logger.hpp"
#include <filesystem>

using namespace mpfem;

int main(int argc, char* argv[]) {
    Logger::setLevel(LogLevel::Info);
    
    std::string caseDir = "cases/busbar_steady";
    if (argc > 1) caseDir = argv[1];
    
    LOG_INFO << "=== Busbar Electro-Thermal Example ===";
    LOG_INFO << "Case directory: " << caseDir;
    
    try {
        auto setup = PhysicsProblemBuilder::build(caseDir);

        if (setup->isTransient()) {
            LOG_INFO << "Running transient solve...";
            auto& transProb = static_cast<TransientProblem&>(*setup);
            TransientResult result = transProb.solve();
            if (!result.converged) {
                LOG_ERROR << "Transient solve did not converge after " << result.timeSteps << " steps";
                return 1;
            }
            LOG_INFO << "Transient solve converged in " << result.timeSteps << " time steps";

            // Export results
            std::filesystem::create_directories("results");
            ResultExporter::exportVtu(result.snapshots, *setup->mesh, "results/busbar_transient.vtu");
            ResultExporter::exportComsolText(result.snapshots, result.times, *setup->mesh, "results/mpfem_result.txt");
            LOG_INFO << "Exported results";
        } else {
            LOG_INFO << "Running coupled electro-thermal solve...";
            auto& steadyProb = static_cast<SteadyProblem&>(*setup);
            SteadyResult result = steadyProb.solve();
            if (!result.converged) {
                LOG_WARN << "Coupling did not converge after " << result.iterations << " iterations";
            } else {
                LOG_INFO << "Coupling converged in " << result.iterations << " iterations";
            }

            // Print summary
            if (result.fields.hasField(FieldId::ElectricPotential)) {
                const auto& V = result.fields.current(FieldId::ElectricPotential);
                LOG_INFO << "Potential range: [" << V.values().minCoeff()
                         << ", " << V.values().maxCoeff() << "] V";
            }
            if (result.fields.hasField(FieldId::Temperature)) {
                const auto& T = result.fields.current(FieldId::Temperature);
                Real minT = T.values().minCoeff();
                Real maxT = T.values().maxCoeff();
                LOG_INFO << "Temperature range: [" << minT << ", " << maxT << "] K";
                LOG_INFO << "Temperature range: [" << (minT - 273.15) << ", "
                         << (maxT - 273.15) << "] C";
            }
            if (result.fields.hasField(FieldId::Displacement)) {
                const auto& u = result.fields.current(FieldId::Displacement);
                Real maxDisp = 0.0;
                for (Index i = 0; i < u.numDofs() / 3; ++i) {
                    Real dx = u(i * 3);
                    Real dy = u(i * 3 + 1);
                    Real dz = u(i * 3 + 2);
                    maxDisp = std::max(maxDisp, std::sqrt(dx*dx + dy*dy + dz*dz));
                }
                LOG_INFO << "Max displacement magnitude: " << maxDisp << " m";
            }

            // Export results
            std::filesystem::create_directories("results");
            ResultExporter::exportVtu(result.fields, *setup->mesh, "results/busbar_steady.vtu");
            ResultExporter::exportComsolText(result.fields, *setup->mesh, "results/mpfem_result.txt");
            LOG_INFO << "Exported results";
        }
        
        LOG_INFO << "=== Example completed successfully! ===";
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR << "Error: " << e.what();
        return 1;
    }
}
