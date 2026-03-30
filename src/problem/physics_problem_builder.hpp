#ifndef MPFEM_PHYSICS_PROBLEM_BUILDER_HPP
#define MPFEM_PHYSICS_PROBLEM_BUILDER_HPP

#include "problem.hpp"
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
namespace PhysicsProblemBuilder {
    /// 构建问题
    std::unique_ptr<Problem> build(const std::string &caseDir);
} // namespace PhysicsProblemBuilder

} // namespace mpfem

#endif // MPFEM_PHYSICS_PROBLEM_BUILDER_HPP
