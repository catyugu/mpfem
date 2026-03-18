#ifndef MPFEM_PHYSICS_PROBLEM_BUILDER_HPP
#define MPFEM_PHYSICS_PROBLEM_BUILDER_HPP

#include "electrostatics_solver.hpp"
#include "heat_transfer_solver.hpp"
#include "structural_solver.hpp"
#include "assembly/integrators.hpp"
#include "model/case_definition.hpp"
#include "model/material_database.hpp"
#include "io/case_xml_reader.hpp"
#include "io/material_xml_reader.hpp"
#include "mesh/mesh.hpp"
#include "io/mphtxt_reader.hpp"
#include "core/logger.hpp"
#include <memory>
#include <map>
#include <string>

namespace mpfem
{

    /**
     * @brief 耦合求解结果
     */
    struct CouplingResult {
        bool converged = false;
        int iterations = 0;
        Real residual = 0.0;
    };

    /**
     * @brief 物理问题设置结果
     *
     * 所有权策略：
     * - 求解器由本结构持有所有权
     * - 所有系数统一存储在 coefficients_ 中
     * - 求解器内部持有域映射系数
     */
    struct PhysicsProblemSetup
    {
        std::string caseName;
        std::unique_ptr<Mesh> mesh;
        MaterialDatabase materials;
        CaseDefinition caseDef;
        std::map<int, std::string> domainMaterial;

        // 求解器（拥有所有权）
        std::unique_ptr<ElectrostaticsSolver> electrostatics;
        std::unique_ptr<HeatTransferSolver> heatTransfer;
        std::unique_ptr<StructuralSolver> structural;

        // 所有系数统一存储（包括边界条件和耦合系数）
        std::map<std::string, std::unique_ptr<Coefficient>> coefficients;
        std::map<std::string, std::unique_ptr<VectorCoefficient>> vectorCoefficients;

        // 耦合迭代参数
        int couplingMaxIter_ = 15;
        Real couplingTol_ = 1e-6;
        Vector prevT_;  ///< 前一步温度场，用于收敛判断

        bool hasElectrostatics() const { return electrostatics != nullptr; }
        bool hasHeatTransfer() const { return heatTransfer != nullptr; }
        bool hasStructural() const { return structural != nullptr; }
        bool hasJouleHeating() const { return hasElectrostatics() && hasHeatTransfer(); }
        bool hasThermalExpansion() const { return hasHeatTransfer() && hasStructural(); }
        bool isCoupled() const { return hasJouleHeating() || hasThermalExpansion(); }

        /**
         * @brief 执行耦合或单场求解
         */
        CouplingResult solve();

    private:
        Real computeCouplingError();
    };

    class PhysicsProblemBuilder
    {
    public:
        static PhysicsProblemSetup build(const std::string &caseDir);

    private:
        static void buildSolvers(PhysicsProblemSetup &setup);
        static void buildElectrostatics(PhysicsProblemSetup &setup,
                                        const PhysicsDefinition &physics);
        static void buildHeatTransfer(PhysicsProblemSetup &setup,
                                      const PhysicsDefinition &physics);
        static void buildStructural(PhysicsProblemSetup &setup,
                                    const PhysicsDefinition &physics);
        static void setupCoupling(PhysicsProblemSetup &setup);
        static Real parseValue(const std::map<std::string, std::string> &params,
                               const std::string &key,
                               const CaseDefinition &caseDef,
                               Real defaultVal = 0.0);
    };

} // namespace mpfem

#endif // MPFEM_PHYSICS_PROBLEM_BUILDER_HPP
