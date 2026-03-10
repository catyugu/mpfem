/**
 * @file field_registry.hpp
 * @brief Field registry - central manager for all physical fields
 * 
 * This class provides name-based access to all fields in the simulation.
 * It enables physics modules to access other fields without direct coupling.
 */

#ifndef MPFEM_DOF_FIELD_REGISTRY_HPP
#define MPFEM_DOF_FIELD_REGISTRY_HPP

#include "field_space.hpp"
#include "core/types.hpp"
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace mpfem {

/**
 * @brief Central registry for all physical fields
 * 
 * The FieldRegistry is the single point of truth for field access.
 * Physics modules query fields by name, enabling loose coupling.
 * 
 * Usage:
 * @code
 * FieldRegistry registry;
 * 
 * // Register fields
 * registry.create_field("electric_potential", mesh, 1);
 * registry.create_field("temperature", mesh, 1);
 * registry.create_field("displacement", mesh, 3);
 * 
 * // In physics assembly, access other fields by name
 * const FieldSpace* temp_field = registry.get_field("temperature");
 * double T = temp_field->value_at_node(node_id);
 * @endcode
 */
class FieldRegistry {
public:
    FieldRegistry() = default;
    ~FieldRegistry() = default;
    
    // Non-copyable
    FieldRegistry(const FieldRegistry&) = delete;
    FieldRegistry& operator=(const FieldRegistry&) = delete;
    
    // ============================================================
    // Field Registration
    // ============================================================
    
    /**
     * @brief Create and register a new field
     * @param name Unique field name
     * @param mesh The mesh
     * @param n_components Number of components
     * @param order Polynomial order
     * @return Pointer to the created field
     */
    FieldSpace* create_field(const FieldID& name, 
                             const Mesh* mesh,
                             int n_components,
                             int order = 1);
    
    /**
     * @brief Register an existing field
     * @param field The field to register
     * @return True if registration succeeded
     */
    bool register_field(FieldSpacePtr field);
    
    /**
     * @brief Unregister a field
     * @param name Field name
     * @return True if field was removed
     */
    bool unregister_field(const FieldID& name);
    
    // ============================================================
    // Field Access
    // ============================================================
    
    /**
     * @brief Get a field by name
     * @param name Field name
     * @return Pointer to field, or nullptr if not found
     */
    FieldSpace* get_field(const FieldID& name);
    const FieldSpace* get_field(const FieldID& name) const;
    
    /**
     * @brief Check if a field exists
     */
    bool has_field(const FieldID& name) const {
        return fields_.find(name) != fields_.end();
    }
    
    /**
     * @brief Get all field names
     */
    std::vector<FieldID> field_names() const;
    
    /**
     * @brief Get number of registered fields
     */
    size_t n_fields() const { return fields_.size(); }
    
    // ============================================================
    // Field Value Queries (Convenience Methods)
    // ============================================================
    
    /**
     * @brief Get scalar field value at a node
     * @param field_name Field name
     * @param node_id Node index
     * @return Field value, or 0 if field not found
     */
    Scalar scalar_value(const FieldID& field_name, Index node_id) const {
        const FieldSpace* field = get_field(field_name);
        if (field && field->type() == FieldType::Scalar) {
            return field->value_at_node(node_id);
        }
        return 0.0;
    }
    
    /**
     * @brief Get vector field value at a node
     * @param field_name Field name
     * @param node_id Node index
     * @return Vector value
     */
    Tensor<1, 3> vector_value(const FieldID& field_name, Index node_id) const {
        const FieldSpace* field = get_field(field_name);
        if (field && field->type() == FieldType::Vector) {
            return field->vector_at_node(node_id);
        }
        return Tensor<1, 3>::Zero();
    }
    
    /**
     * @brief Get field values at all quadrature points of a cell
     * @param field_name Field name
     * @param cell_id Cell index
     * @param qpoints Quadrature point coordinates (reference space)
     * @param values Output values at each quadrature point
     * @note This performs interpolation from nodal values
     */
    void get_field_values_at_qpoints(const FieldID& field_name,
                                     Index cell_id,
                                     const std::vector<Point<3>>& qpoints,
                                     std::vector<Scalar>& values) const;
    
    /**
     * @brief Get field gradients at all quadrature points of a cell
     */
    void get_field_gradients_at_qpoints(const FieldID& field_name,
                                        Index cell_id,
                                        const std::vector<Point<3>>& qpoints,
                                        std::vector<Tensor<1, 3>>& gradients) const;
    
    // ============================================================
    // Solution Management
    // ============================================================
    
    /**
     * @brief Initialize all field solutions to zero
     */
    void initialize_all_solutions();
    
    /**
     * @brief Clear all fields
     */
    void clear() {
        fields_.clear();
    }
    
private:
    std::unordered_map<FieldID, FieldSpacePtr> fields_;
};

/**
 * @brief Global field registry (singleton pattern for convenience)
 * 
 * Use this when you need a single global registry across the simulation.
 * For more complex scenarios, create your own FieldRegistry instance.
 */
class GlobalFieldRegistry {
public:
    static FieldRegistry& instance() {
        static FieldRegistry registry;
        return registry;
    }
    
    GlobalFieldRegistry() = delete;
};

} // namespace mpfem

#endif // MPFEM_DOF_FIELD_REGISTRY_HPP
