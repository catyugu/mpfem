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
#include "assembly/bilinear_form.hpp"
#include "assembly/linear_form.hpp"
#include "config/case_config.hpp"
#include "core/logger.hpp"
#include <unordered_map>

namespace mpfem {

/**
 * @brief Solid mechanics field assembly (linear elasticity)
 * 
 * Solves the equilibrium equation:
 * -∇·σ = 0
 * 
 * where σ = D : ε, D is the elasticity tensor, ε is the strain.
 * For thermal expansion: ε_total = ε_mechanical + ε_thermal
 */
class SolidMechanicsAssembly : public PhysicsAssembly {
public:
    SolidMechanicsAssembly() : dim_(3) {}
    
    /**
     * @brief Initialize with physics config
     */
    void initialize(const Mesh* mesh,
                   const DoFHandler* dof_handler,
                   const MaterialDB* mat_db,
                   const PhysicsConfig& config);
    
    void assemble_stiffness(SparseMatrix& K) override;
    void assemble_rhs(DynamicVector& f) override;
    void apply_boundary_conditions(SparseMatrix& K, DynamicVector& f) override;
    
    std::string field_name() const override { return "solid_mechanics"; }
    int n_components() const override { return dim_; }
    
    /**
     * @brief Set temperature field for thermal expansion
     */
    void set_external_field(const std::string& field_name,
                           const DynamicVector& values) override {
        if (field_name == "temperature" || field_name == "T") {
            temperature_field_ = &values;
        }
    }
    
    /**
     * @brief Set reference temperature for thermal expansion
     */
    void set_reference_temperature(Scalar T_ref) {
        T_ref_ = T_ref;
    }

private:
    int dim_;
    std::vector<BoundaryConditionConfig> bcs_;
    std::unordered_map<Index, std::string> domain_material_map_;
    const DynamicVector* temperature_field_ = nullptr;
    Scalar T_ref_ = 293.15;  // Reference temperature for thermal expansion
    
    /**
     * @brief Compute elasticity tensor D for isotropic material
     */
    Tensor<4, 3> compute_elasticity_tensor(Scalar E, Scalar nu) const;
};

} // namespace mpfem

#endif // MPFEM_PHYSICS_SOLID_MECHANICS_HPP
