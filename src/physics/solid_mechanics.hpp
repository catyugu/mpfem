/**
 * @file solid_mechanics.hpp
 * @brief Solid mechanics physics assembly
 * 
 * Solves: -∇·σ = 0 (equilibrium)
 * with displacement boundary conditions
 */

#ifndef MPFEM_PHYSICS_SOLID_MECHANICS_HPP
#define MPFEM_PHYSICS_SOLID_MECHANICS_HPP

#include "physics_assembly.hpp"
#include "config/case_config.hpp"
#include "core/logger.hpp"
#include <unordered_map>

namespace mpfem {

/**
 * @brief Solid mechanics field assembly (linear elasticity)
 */
class SolidMechanicsAssembly : public PhysicsAssembly {
public:
    SolidMechanicsAssembly() : dim_(3) {}
    
    void initialize(const Mesh* mesh,
                   const FieldSpace* field,
                   const MaterialDB* mat_db,
                   const PhysicsConfig& config);
    
    void assemble_stiffness(SparseMatrix& K) override;
    void assemble_rhs(DynamicVector& f) override;
    void apply_boundary_conditions(SparseMatrix& K, DynamicVector& f) override;
    
    std::string field_name() const override { return "solid_mechanics"; }
    int n_components() const override { return dim_; }
    
    void set_external_field(const std::string& field_name,
                           const DynamicVector& values) override {
        if (field_name == "temperature" || field_name == "T") {
            temperature_field_ = &values;
        }
    }
    
    void set_reference_temperature(Scalar T_ref) {
        T_ref_ = T_ref;
    }

private:
    int dim_;
    std::vector<BoundaryConditionConfig> bcs_;
    std::unordered_map<Index, std::string> domain_material_map_;
    const DynamicVector* temperature_field_ = nullptr;
    Scalar T_ref_ = 293.15;
};

} // namespace mpfem

#endif // MPFEM_PHYSICS_SOLID_MECHANICS_HPP
