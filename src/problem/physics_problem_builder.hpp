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
 * @brief Interface for physics-specific problem building.
 * 
 * Provides explicit dispatch for building different physics solvers
 * based on the physics kind string.
 */
class IPhysicsBuilder {
public:
    virtual ~IPhysicsBuilder() = default;
    
    /// Build the physics solver and attach to problem
    virtual void build(Problem& problem, const CaseDefinition::Physics& physics) = 0;
    
    /// Return the physics kind this builder handles (e.g., "electrostatics")
    virtual const char* kind() const = 0;
};

/**
 * @brief Builder for electrostatics physics solver
 */
class ElectrostaticsBuilder : public IPhysicsBuilder {
public:
    const char* kind() const override { return "electrostatics"; }
    void build(Problem& problem, const CaseDefinition::Physics& physics) override;
};

/**
 * @brief Builder for heat transfer physics solver
 */
class HeatTransferBuilder : public IPhysicsBuilder {
public:
    const char* kind() const override { return "heat_transfer"; }
    void build(Problem& problem, const CaseDefinition::Physics& physics) override;
};

/**
 * @brief Builder for structural mechanics physics solver
 */
class StructuralBuilder : public IPhysicsBuilder {
public:
    const char* kind() const override { return "solid_mechanics"; }
    void build(Problem& problem, const CaseDefinition::Physics& physics) override;
};

/**
 * @brief Factory to create physics builder by kind string
 */
inline std::unique_ptr<IPhysicsBuilder> createPhysicsBuilder(const std::string& kind) {
    if (kind == "electrostatics") return std::make_unique<ElectrostaticsBuilder>();
    if (kind == "heat_transfer") return std::make_unique<HeatTransferBuilder>();
    if (kind == "solid_mechanics") return std::make_unique<StructuralBuilder>();
    return nullptr;
}

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
};

} // namespace mpfem

#endif // MPFEM_PHYSICS_PROBLEM_BUILDER_HPP
