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
#include "assembly/bilinear_form.hpp"
#include "assembly/linear_form.hpp"
#include "config/case_config.hpp"
#include "core/logger.hpp"
#include <unordered_map>

namespace mpfem {

/**
 * @brief Electrostatics field assembly
 * 
 * Solves the steady-state electric potential distribution:
 * -∇·(σ∇V) = 0
 * 
 * where σ is the electric conductivity.
 */
class ElectrostaticsAssembly : public PhysicsAssembly {
public:
    ElectrostaticsAssembly() = default;
    
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
    
    std::string field_name() const override { return "electrostatics"; }
    
    /**
     * @brief Set temperature field for temperature-dependent conductivity
     */
    void set_external_field(const std::string& field_name,
                           const DynamicVector& values) override {
        if (field_name == "temperature" || field_name == "T") {
            temperature_field_ = &values;
        }
    }
    
    /**
     * @brief Set domain material mapping
     */
    void set_domain_materials(const std::unordered_map<Index, std::string>& mapping) {
        domain_material_map_ = mapping;
    }
    
    /**
     * @brief Get electric field gradient at nodes
     * @param solution Electric potential solution
     * @return Gradient vector (3 components per node)
     */
    DynamicVector get_field_gradient(const DynamicVector& solution) const;
    
private:
    std::vector<BoundaryConditionConfig> bcs_;
    std::unordered_map<Index, std::string> domain_material_map_;
    const DynamicVector* temperature_field_ = nullptr;
    
    // Precomputed conductivity values per domain
    std::unordered_map<Index, Tensor<2, 3>> conductivity_;
    
    void compute_conductivities();
};

} // namespace mpfem

#endif // MPFEM_PHYSICS_ELECTROSTATICS_HPP
