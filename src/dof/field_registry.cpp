/**
 * @file field_registry.cpp
 * @brief Field registry implementation
 */

#include "field_registry.hpp"
#include "core/logger.hpp"

namespace mpfem {

FieldSpace* FieldRegistry::create_field(const FieldID& name,
                                        const Mesh* mesh,
                                        int n_components,
                                        int order) {
    if (has_field(name)) {
        MPFEM_WARN("Field '" << name << "' already exists, returning existing field");
        return get_field(name);
    }
    
    auto field = std::make_shared<FieldSpace>(name, mesh, n_components, order);
    fields_[name] = field;
    
    MPFEM_INFO("Registered field '" << name << "'");
    return field.get();
}

bool FieldRegistry::register_field(FieldSpacePtr field) {
    if (!field) return false;
    
    const FieldID& name = field->name();
    if (has_field(name)) {
        MPFEM_WARN("Field '" << name << "' already registered");
        return false;
    }
    
    fields_[name] = field;
    return true;
}

bool FieldRegistry::unregister_field(const FieldID& name) {
    auto it = fields_.find(name);
    if (it == fields_.end()) {
        return false;
    }
    
    fields_.erase(it);
    MPFEM_INFO("Unregistered field '" << name << "'");
    return true;
}

FieldSpace* FieldRegistry::get_field(const FieldID& name) {
    auto it = fields_.find(name);
    if (it != fields_.end()) {
        return it->second.get();
    }
    return nullptr;
}

const FieldSpace* FieldRegistry::get_field(const FieldID& name) const {
    auto it = fields_.find(name);
    if (it != fields_.end()) {
        return it->second.get();
    }
    return nullptr;
}

std::vector<FieldID> FieldRegistry::field_names() const {
    std::vector<FieldID> names;
    names.reserve(fields_.size());
    for (const auto& [name, field] : fields_) {
        names.push_back(name);
    }
    return names;
}

void FieldRegistry::get_field_values_at_qpoints(const FieldID& field_name,
                                                Index cell_id,
                                                const std::vector<Point<3>>& qpoints,
                                                std::vector<Scalar>& values) const {
    values.clear();
    
    const FieldSpace* field = get_field(field_name);
    if (!field) {
        MPFEM_WARN("Field '" << field_name << "' not found");
        return;
    }
    
    // Get cell DoFs
    std::vector<Index> cell_dofs;
    field->get_cell_dofs(cell_id, cell_dofs);
    
    if (cell_dofs.empty()) {
        values.resize(qpoints.size(), 0.0);
        return;
    }
    
    // For scalar fields, interpolate using shape functions
    // This is a simplified implementation - assumes linear Lagrange elements
    // TODO: Use proper FE shape functions
    
    const DynamicVector& sol = field->solution();
    
    // Simple averaging for now
    Scalar avg = 0.0;
    int count = 0;
    for (Index dof : cell_dofs) {
        if (dof < sol.size()) {
            avg += sol[dof];
            ++count;
        }
    }
    if (count > 0) avg /= count;
    
    values.resize(qpoints.size(), avg);
}

void FieldRegistry::get_field_gradients_at_qpoints(const FieldID& field_name,
                                                   Index cell_id,
                                                   const std::vector<Point<3>>& qpoints,
                                                   std::vector<Tensor<1, 3>>& gradients) const {
    gradients.clear();
    
    const FieldSpace* field = get_field(field_name);
    if (!field) {
        MPFEM_WARN("Field '" << field_name << "' not found");
        return;
    }
    
    // Get cell DoFs
    std::vector<Index> cell_dofs;
    field->get_cell_dofs(cell_id, cell_dofs);
    
    if (cell_dofs.empty()) {
        gradients.resize(qpoints.size(), Tensor<1, 3>::Zero());
        return;
    }
    
    // TODO: Implement proper gradient computation using FE shape functions
    // For now, return zero gradients
    gradients.resize(qpoints.size(), Tensor<1, 3>::Zero());
}

void FieldRegistry::initialize_all_solutions() {
    for (auto& [name, field] : fields_) {
        field->initialize_solution();
    }
    MPFEM_INFO("Initialized " << fields_.size() << " field solutions");
}

} // namespace mpfem
