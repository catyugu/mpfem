#ifndef MPFEM_FE_VALUES_HPP
#define MPFEM_FE_VALUES_HPP

#include "fe/grid_function.hpp"
#include "fe/quadrature.hpp"
#include "model/field_kind.hpp"
#include "core/types.hpp"
#include <map>
#include <string>
#include <memory>

namespace mpfem {

// Forward declaration
class ElementTransform;

/**
 * @brief Multi-field state manager for finite element simulation.
 * 
 * FEValues manages multiple GridFunctions (representing different physics fields)
 * and provides a unified interface for accessing field values and gradients
 * at integration points. This abstraction simplifies coupled physics implementations.
 */
class FEValues {
public:
    /// Default constructor
    FEValues() = default;

    // -------------------------------------------------------------------------
    // Time management
    // -------------------------------------------------------------------------

    /// Get current time
    Real time() const { return time_; }

    /// Set current time
    void setTime(Real t) { time_ = t; }

    // -------------------------------------------------------------------------
    // Field registration
    // -------------------------------------------------------------------------

    /**
     * @brief Register a field by FieldKind.
     * @param kind Field kind identifier.
     * @param gf Pointer to GridFunction (non-owning).
     */
    void registerField(FieldKind kind, GridFunction* gf);

    /**
     * @brief Register a field by name.
     * @param name Field name.
     * @param gf Pointer to GridFunction (non-owning).
     */
    void registerField(const std::string& name, GridFunction* gf);

    /**
     * @brief Check if a field is registered.
     */
    bool hasField(FieldKind kind) const;

    /**
     * @brief Check if a field is registered by name.
     */
    bool hasField(const std::string& name) const;

    // -------------------------------------------------------------------------
    // Field access
    // -------------------------------------------------------------------------

    /**
     * @brief Get field by FieldKind.
     * @return Pointer to GridFunction, nullptr if not found.
     */
    GridFunction* field(FieldKind kind);
    const GridFunction* field(FieldKind kind) const;

    /**
     * @brief Get field by name.
     */
    GridFunction* field(const std::string& name);
    const GridFunction* field(const std::string& name) const;

    /**
     * @brief Get all registered fields.
     */
    const std::map<FieldKind, GridFunction*>& fieldsByKind() const { return fields_; }
    const std::map<std::string, GridFunction*>& fieldsByName() const { return fieldsByName_; }

    // -------------------------------------------------------------------------
    // Field value evaluation
    // -------------------------------------------------------------------------

    /**
     * @brief Get scalar field value at integration point.
     * @param kind Field kind.
     * @param elemIdx Element index.
     * @param ip Integration point.
     * @return Field value.
     */
    Real getValue(FieldKind kind, Index elemIdx, const IntegrationPoint& ip) const;

    /**
     * @brief Get scalar field value at reference coordinates.
     */
    Real getValue(FieldKind kind, Index elemIdx, const Real* xi) const;

    /**
     * @brief Get vector field value at integration point.
     * @return Vector value (size depends on field vdim).
     */
    Vector3 getVectorValue(FieldKind kind, Index elemIdx, const IntegrationPoint& ip) const;

    /**
     * @brief Get field gradient at integration point.
     * @param kind Field kind.
     * @param elemIdx Element index.
     * @param ip Integration point.
     * @param trans Element transformation.
     * @return Gradient in physical coordinates.
     */
    Vector3 getGradient(FieldKind kind, Index elemIdx, 
                        const IntegrationPoint& ip, 
                        const ElementTransform& trans) const;

    /**
     * @brief Get field gradient at reference coordinates.
     */
    Vector3 getGradient(FieldKind kind, Index elemIdx, 
                        const Real* xi, 
                        const ElementTransform& trans) const;

    // -------------------------------------------------------------------------
    // Convenience methods for common fields
    // -------------------------------------------------------------------------

    /// Get electric potential value
    Real electricPotential(Index elemIdx, const IntegrationPoint& ip) const {
        return getValue(FieldKind::ElectricPotential, elemIdx, ip);
    }

    /// Get electric field (negative gradient of potential)
    Vector3 electricField(Index elemIdx, const IntegrationPoint& ip,
                          const ElementTransform& trans) const {
        return -getGradient(FieldKind::ElectricPotential, elemIdx, ip, trans);
    }

    /// Get temperature value
    Real temperature(Index elemIdx, const IntegrationPoint& ip) const {
        return getValue(FieldKind::Temperature, elemIdx, ip);
    }

    /// Get temperature gradient
    Vector3 temperatureGradient(Index elemIdx, const IntegrationPoint& ip,
                                const ElementTransform& trans) const {
        return getGradient(FieldKind::Temperature, elemIdx, ip, trans);
    }

    /// Get displacement vector
    Vector3 displacement(Index elemIdx, const IntegrationPoint& ip) const {
        return getVectorValue(FieldKind::Displacement, elemIdx, ip);
    }

    // -------------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------------

    /// Clear all registered fields
    void clear() {
        fields_.clear();
        fieldsByName_.clear();
    }

    /// Get number of registered fields
    size_t numFields() const { return fields_.size(); }

private:
    Real time_ = 0.0;
    std::map<FieldKind, GridFunction*> fields_;
    std::map<std::string, GridFunction*> fieldsByName_;
};

}  // namespace mpfem

#endif  // MPFEM_FE_VALUES_HPP
