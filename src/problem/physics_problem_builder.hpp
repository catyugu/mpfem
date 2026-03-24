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

class Problem;

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
    static void setupCoupling(Problem &problem);
    static void buildElectrostatics(Problem &problem, const CaseDefinition::Physics &physics);
    static void buildHeatTransfer(Problem &problem, const CaseDefinition::Physics &physics);
    static void buildStructural(Problem &problem, const CaseDefinition::Physics &physics);
};

} // namespace mpfem

#endif // MPFEM_PHYSICS_PROBLEM_BUILDER_HPP
