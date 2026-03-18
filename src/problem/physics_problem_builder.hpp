#ifndef MPFEM_PHYSICS_PROBLEM_BUILDER_HPP
#define MPFEM_PHYSICS_PROBLEM_BUILDER_HPP

#include "steady_problem.hpp"
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
    static std::unique_ptr<SteadyProblem> build(const std::string &caseDir);

private:
    static void buildSolvers(SteadyProblem &problem);
    static void buildElectrostatics(SteadyProblem &problem, const PhysicsDefinition &physics);
    static void buildHeatTransfer(SteadyProblem &problem, const PhysicsDefinition &physics);
    static void buildStructural(SteadyProblem &problem, const PhysicsDefinition &physics);
    static void setupCoupling(SteadyProblem &problem);
    
    static Real parseValue(const std::map<std::string, std::string> &params,
                           const std::string &key,
                           const CaseDefinition &caseDef,
                           Real defaultVal = 0.0);
};

} // namespace mpfem

#endif // MPFEM_PHYSICS_PROBLEM_BUILDER_HPP
