#ifndef MPFEM_PHYSICS_PROBLEM_BUILDER_HPP
#define MPFEM_PHYSICS_PROBLEM_BUILDER_HPP

#include "steady_problem.hpp"
#include "transient_problem.hpp"
#include "io/case_xml_reader.hpp"
#include "io/material_xml_reader.hpp"
#include "io/mphtxt_reader.hpp"
#include <memory>
#include <string>

namespace mpfem
{

/**
 * @brief 物理问题构建器
 * 
 * 根据 case.xml 和 material.xml 构建问题实例。
 * 目前默认返回稳态问题。
 */
class PhysicsProblemBuilder
{
public:
    /// 构建问题
    static std::unique_ptr<Problem> build(const std::string &caseDir);

private:
    static void buildSolvers(Problem &problem);
    static void buildElectrostatics(Problem &problem, const PhysicsDefinition &physics);
    static void buildHeatTransfer(Problem &problem, const PhysicsDefinition &physics);
    static void buildStructural(Problem &problem, const PhysicsDefinition &physics);
    static void setupCoupling(Problem &problem);
    
    static Real parseValue(const std::map<std::string, std::string> &params,
                           const std::string &key,
                           const CaseDefinition &caseDef,
                           Real defaultVal = 0.0);
    
    /// Get initial condition value for a physics kind, returns default if not found
    static double getInitialCondition(const CaseDefinition &caseDef, const std::string &fieldKind, double defaultVal);
};

} // namespace mpfem

#endif // MPFEM_PHYSICS_PROBLEM_BUILDER_HPP
