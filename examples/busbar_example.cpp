#include "mesh/mesh.hpp"
#include "io/mphtxt_reader.hpp"
#include "fe/fe_collection.hpp"
#include "fe/fe_space.hpp"
#include "problem/physics_problem_builder.hpp"
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
        auto setup = PhysicsProblemBuilder::build(caseDir);

        if (setup->isCoupled()) {
            LOG_INFO << "Running coupled electro-thermal solve...";
            CouplingResult result = setup->solve();

            if (!result.converged) {
                LOG_WARN << "Coupling did not converge after " << result.iterations << " iterations";
            } else {
                LOG_INFO << "Coupling converged in " << result.iterations << " iterations";
            }
        } else if (setup->hasElectrostatics()) {
            LOG_INFO << "Running single physics electrostatics solve...";
            setup->electrostatics->assemble();
            bool success = setup->electrostatics->solve();
            if (!success) {
                LOG_ERROR << "Solver failed to converge";
                return 1;
            }
        }
        
        // 打印结果
        if (setup->hasElectrostatics()) {
            const auto& V = setup->electrostatics->field().values();
            LOG_INFO << "Potential range: [" << V.minCoeff() 
                     << ", " << V.maxCoeff() << "] V";
        }
        if (setup->hasHeatTransfer()) {
            const auto& T = setup->heatTransfer->field().values();
            Real minT = T.minCoeff();
            Real maxT = T.maxCoeff();
            LOG_INFO << "Temperature range: [" << minT << ", " << maxT << "] K";
            LOG_INFO << "Temperature range: [" << (minT - 273.15) << ", " 
                     << (maxT - 273.15) << "] C";
        }
        if (setup->hasStructural()) {
            const auto& u = setup->structural->field().values();
            // 计算位移幅值
            Index numNodes = setup->mesh->numVertices();
            Real maxDisp = 0.0;
            for (Index i = 0; i < numNodes; ++i) {
                Real dx = u(i * 3);
                Real dy = u(i * 3 + 1);
                Real dz = u(i * 3 + 2);
                Real mag = std::sqrt(dx*dx + dy*dy + dz*dz);
                maxDisp = std::max(maxDisp, mag);
            }
            LOG_INFO << "Max displacement magnitude: " << maxDisp << " m";
        }
        
        // 导出结果
        std::filesystem::create_directories("results");
        std::string outputPath = "results/busbar_results.vtu";
        
        // 检查是否为高阶网格
        Index numCorners = setup->mesh->numCornerVertices();
        Index numVerts = setup->mesh->numVertices();
        bool isHighOrder = (numCorners < numVerts);
        if (isHighOrder) {
            LOG_INFO << "High-order mesh detected: " << numVerts << " vertices, "
                     << numCorners << " corner vertices";
        }
        
        std::vector<FieldResult> scalarFields;
        std::map<std::string, std::vector<Vector3>> vectorFields;
        
        if (setup->hasElectrostatics()) {
            const GridFunction& V = setup->electrostatics->field();
            FieldResult f;
            f.name = "V";
            f.unit = "V";
            // 对于高阶网格，投影到角点
            Eigen::VectorXd cornerV = V.projectToCorners(*setup->mesh);
            f.nodalValues.assign(cornerV.data(), cornerV.data() + cornerV.size());
            scalarFields.push_back(f);
        }
        
        if (setup->hasHeatTransfer()) {
            const GridFunction& T = setup->heatTransfer->field();
            FieldResult f;
            f.name = "T";
            f.unit = "K";
            // 对于高阶网格，投影到角点
            Eigen::VectorXd cornerT = T.projectToCorners(*setup->mesh);
            f.nodalValues.assign(cornerT.data(), cornerT.data() + cornerT.size());
            scalarFields.push_back(f);
        }
        
        // 添加位移场（导出为向量场）
        if (setup->hasStructural()) {
            const GridFunction& u = setup->structural->field();
            
            // 对于高阶网格，先投影到角点
            Eigen::VectorXd cornerU = u.projectToCorners(*setup->mesh);
            
            // 导出位移向量场
            Index numExport = numCorners;
            std::vector<Vector3> dispVec(numExport);
            for (Index i = 0; i < numExport; ++i) {
                dispVec[i].x() = cornerU(i * 3);
                dispVec[i].y() = cornerU(i * 3 + 1);
                dispVec[i].z() = cornerU(i * 3 + 2);
            }
            vectorFields["displacement"] = dispVec;
            
            // 同时导出位移幅值方便调试
            FieldResult fDispMag;
            fDispMag.name = "disp_magnitude";
            fDispMag.unit = "m";
            fDispMag.nodalValues.resize(numExport);
            for (Index i = 0; i < numExport; ++i) {
                Real dx = cornerU(i * 3);
                Real dy = cornerU(i * 3 + 1);
                Real dz = cornerU(i * 3 + 2);
                fDispMag.nodalValues[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
            }
            scalarFields.push_back(fDispMag);
        } else {
            // 位移场占位符（如果求解器未启用）
            FieldResult f;
            f.name = "disp_magnitude";
            f.unit = "m";
            f.nodalValues.resize(numCorners, 0.0);
            scalarFields.push_back(f);
        }
        
        ResultExporter::exportVtuWithVectors(outputPath, *setup->mesh, scalarFields, vectorFields);
        LOG_INFO << "Exported VTU results to: " << outputPath;
        
        // 导出COMSOL格式结果文件用于比较
        std::string comsolOutput = "results/mpfem_result.txt";
        ResultExporter::exportComsolText(comsolOutput, *setup->mesh, scalarFields,
            "Electric potential, Temperature, Displacement magnitude");
        LOG_INFO << "Exported results to: " << comsolOutput;
        
        LOG_INFO << "=== Example completed successfully! ===";
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR << "Error: " << e.what();
        return 1;
    }
}