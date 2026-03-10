/**
 * @file electrostatics.hpp
 * @brief Electrostatics physics assembly
 * 
 * Solves: -∇·(σ∇V) = 0
 * with Dirichlet BC (voltage) and Neumann BC (insulation)
 */

#ifndef MPFEM_PHYSICS_ELECTROSTATICS_HPP
#define MPFEM_PHYSICS_ELECTROSTATICS_HPP

#include "physics_assembly.hpp"
#include "config/case_config.hpp"
#include "core/logger.hpp"
#include "dof/field_registry.hpp"
#include <unordered_map>

namespace mpfem {

/**
 * @brief Electrostatics field assembly
 */
class ElectrostaticsAssembly : public PhysicsAssembly {
public:
    ElectrostaticsAssembly() = default;
    
    void initialize(const Mesh* mesh,
                   const FieldSpace* field,
                   const MaterialDB* mat_db,
                   const PhysicsConfig& config);
    
    void assemble_stiffness(SparseMatrix& K) override;
    void assemble_rhs(DynamicVector& f) override;
    void apply_boundary_conditions(SparseMatrix& K, DynamicVector& f) override;
    
    std::string field_name() const override { return "electrostatics"; }
    
    void set_field_registry(const FieldRegistry* registry) {
        field_registry_ = registry;
    }
    
    void set_domain_materials(const std::unordered_map<Index, std::string>& mapping) {
        domain_material_map_ = mapping;
    }
    
    /**
     * @brief Compute Joule heating source term
     * Q = σ|E|² = σ|∇V|²
     * @param registry Field registry containing electric potential
     * @return Joule heating values at each node
     */
    std::vector<Scalar> compute_joule_heating(const FieldRegistry& registry) const;

private:
    std::vector<BoundaryConditionConfig> bcs_;
    std::unordered_map<Index, std::string> domain_material_map_;
    const FieldRegistry* field_registry_ = nullptr;
};

} // namespace mpfem

#endif // MPFEM_PHYSICS_ELECTROSTATICS_HPP