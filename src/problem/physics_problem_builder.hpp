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
namespace PhysicsProblemBuilder {
    /// 构建问题
    std::unique_ptr<Problem> build(const std::string &caseDir);

    void buildSolvers(Problem &problem);
    void setupCoupling(Problem &problem);
    void buildElectrostatics(Problem &problem, const CaseDefinition::Physics &physics);
    void buildHeatTransfer(Problem &problem, const CaseDefinition::Physics &physics);
    void buildStructural(Problem &problem, const CaseDefinition::Physics &physics);
} // namespace PhysicsProblemBuilder

} // namespace mpfem

#endif // MPFEM_PHYSICS_PROBLEM_BUILDER_HPP
