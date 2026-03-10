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
#include "config/case_config.hpp"
#include "core/logger.hpp"
#include <unordered_map>

namespace mpfem {

/**
 * @brief Heat transfer field assembly
 */
class HeatTransferAssembly : public PhysicsAssembly {
public:
    HeatTransferAssembly() = default;
    
    void initialize(const Mesh* mesh,
                   const FieldSpace* field,
                   const MaterialDB* mat_db,
                   const PhysicsConfig& config);
    
    void assemble_stiffness(SparseMatrix& K) override;
    void assemble_rhs(DynamicVector& f) override;
    void apply_boundary_conditions(SparseMatrix& K, DynamicVector& f) override;
    
    std::string field_name() const override { return "heat_transfer"; }
    
    void set_heat_source(const DynamicVector& source) {
        heat_source_ = &source;
    }
    
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
};

} // namespace mpfem

#endif // MPFEM_PHYSICS_HEAT_TRANSFER_HPP
