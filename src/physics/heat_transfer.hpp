/**
 * @file heat_transfer.hpp
 * @brief Heat transfer physics assembly
 * 
 * Solves: -∇·(k∇T) = Q
 * with convection and insulation BCs
 */

#ifndef MPFEM_PHYSICS_HEAT_TRANSFER_HPP
#define MPFEM_PHYSICS_HEAT_TRANSFER_HPP

#include "physics_assembly.hpp"
#include "assembly/bilinear_form.hpp"
#include "assembly/linear_form.hpp"
#include "config/case_config.hpp"
#include "core/logger.hpp"
#include <unordered_map>

namespace mpfem {

/**
 * @brief Heat transfer field assembly
 * 
 * Solves the steady-state heat conduction:
 * -∇·(k∇T) = Q
 * 
 * where k is the thermal conductivity and Q is the heat source.
 */
class HeatTransferAssembly : public PhysicsAssembly {
public:
    HeatTransferAssembly() = default;
    
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
    
    std::string field_name() const override { return "heat_transfer"; }
    
    /**
     * @brief Set Joule heating source
     * @param source Heat source values per cell
     */
    void set_heat_source(const DynamicVector& source) {
        heat_source_ = &source;
    }
    
    /**
     * @brief Set external heat source from coupled physics
     */
    void set_external_field(const std::string& field_name,
                           const DynamicVector& values) override {
        if (field_name == "joule_heating" || field_name == "heat_source") {
            heat_source_ = &values;
        }
    }

private:
    std::vector<BoundaryConditionConfig> bcs_;
    std::unordered_map<Index, std::string> domain_material_map_;
    const DynamicVector* heat_source_ = nullptr;
    
    // Coupled fields
    std::unordered_map<Index, Scalar> domain_heat_source_;
};

} // namespace mpfem

#endif // MPFEM_PHYSICS_HEAT_TRANSFER_HPP
