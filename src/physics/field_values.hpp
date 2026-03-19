#ifndef MPFEM_FIELD_VALUES_HPP
#define MPFEM_FIELD_VALUES_HPP

#include "core/types.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include <map>
#include <memory>
#include <deque>
#include <string>

namespace mpfem {

// =============================================================================
// Field identifiers
// =============================================================================

/// Standard field identifiers for multi-physics
enum class FieldId {
    Temperature,        ///< Temperature field (scalar)
    ElectricPotential,  ///< Electric potential field (scalar)
    Displacement,       ///< Displacement field (vector)
    // User-defined fields can be added via string interface
};

// =============================================================================
// FieldValues - Unified field value manager
// =============================================================================

/**
 * @brief Manages GridFunction lifecycle and history for multi-physics simulation
 * 
 * Key responsibilities:
 * - Owns all GridFunction objects for temperature, electric potential, displacement
 * - Supports transient simulation with history field storage
 * - Provides clean interface for coupled solvers to access field values
 * 
 * Ownership model:
 * - FieldValues owns all GridFunction objects
 * - Solvers hold references to FieldValues, not owning GridFunctions
 * - This enables clean coupling and transient support
 */
class FieldValues {
public:
    // =========================================================================
    // Constructors
    // =========================================================================
    
    FieldValues() = default;
    
    /// Constructor with history support for transient problems
    explicit FieldValues(int maxHistorySteps) : maxHistorySteps_(maxHistorySteps) {}
    
    // Non-copyable, movable
    FieldValues(const FieldValues&) = delete;
    FieldValues& operator=(const FieldValues&) = delete;
    FieldValues(FieldValues&&) = default;
    FieldValues& operator=(FieldValues&&) = default;
    
    // =========================================================================
    // Field creation
    // =========================================================================
    
    /// Create a scalar field
    void createScalarField(FieldId id, const FESpace* fes, Real initVal = 0.0) {
        auto& entry = fields_[id];
        entry.field = std::make_unique<GridFunction>(fes, initVal);
        entry.isVector = false;
    }
    
    /// Create a vector field
    void createVectorField(FieldId id, const FESpace* fes, int vdim) {
        auto& entry = fields_[id];
        auto gf = std::make_unique<GridFunction>(fes);
        gf->values().setZero();
        entry.field = std::move(gf);
        entry.isVector = true;
        entry.vdim = vdim;
    }
    
    /// Create field by name (for custom fields)
    void createScalarField(const std::string& name, const FESpace* fes, Real initVal = 0.0) {
        auto& entry = namedFields_[name];
        entry.field = std::make_unique<GridFunction>(fes, initVal);
        entry.isVector = false;
    }
    
    void createVectorField(const std::string& name, const FESpace* fes, int vdim) {
        auto& entry = namedFields_[name];
        auto gf = std::make_unique<GridFunction>(fes);
        gf->values().setZero();
        entry.field = std::move(gf);
        entry.isVector = true;
        entry.vdim = vdim;
    }
    
    // =========================================================================
    // Field access - current time step
    // =========================================================================
    
    /// Get current field value (mutable)
    GridFunction& current(FieldId id) {
        return *fields_.at(id).field;
    }
    
    /// Get current field value (const)
    const GridFunction& current(FieldId id) const {
        return *fields_.at(id).field;
    }
    
    /// Get field by name (mutable)
    GridFunction& current(const std::string& name) {
        return *namedFields_.at(name).field;
    }
    
    /// Get field by name (const)
    const GridFunction& current(const std::string& name) const {
        return *namedFields_.at(name).field;
    }
    
    /// Check if field exists
    bool hasField(FieldId id) const {
        return fields_.find(id) != fields_.end();
    }
    
    bool hasField(const std::string& name) const {
        return namedFields_.find(name) != namedFields_.end();
    }
    
    // =========================================================================
    // Transient support - history fields
    // =========================================================================
    
    /// Get history field (n steps back, n=1 means previous time step)
    GridFunction& history(FieldId id, int stepsBack = 1) {
        auto it = fields_.find(id);
        if (it == fields_.end() || it->second.history.size() < static_cast<size_t>(stepsBack)) {
            static GridFunction empty;
            return empty;  // Return empty reference if not available
        }
        // History is stored in reverse order: index 0 = most recent
        return *it->second.history[stepsBack - 1];
    }
    
    const GridFunction& history(FieldId id, int stepsBack = 1) const {
        auto it = fields_.find(id);
        if (it == fields_.end() || it->second.history.size() < static_cast<size_t>(stepsBack)) {
            static const GridFunction empty;
            return empty;
        }
        return *it->second.history[stepsBack - 1];
    }
    
    /// Advance time step: save current to history
    void advanceTime() {
        for (auto& [id, entry] : fields_) {
            if (maxHistorySteps_ > 0 && entry.field) {
                // Create a copy of current field for history
                auto histField = std::make_unique<GridFunction>();
                *histField = *entry.field;  // Copy values
                
                entry.history.push_front(std::move(histField));
                
                // Trim to max history steps
                while (entry.history.size() > static_cast<size_t>(maxHistorySteps_)) {
                    entry.history.pop_back();
                }
            }
        }
        for (auto& [name, entry] : namedFields_) {
            if (maxHistorySteps_ > 0 && entry.field) {
                auto histField = std::make_unique<GridFunction>();
                *histField = *entry.field;
                
                entry.history.push_front(std::move(histField));
                
                while (entry.history.size() > static_cast<size_t>(maxHistorySteps_)) {
                    entry.history.pop_back();
                }
            }
        }
    }
    
    /// Set max history steps (for transient)
    void setMaxHistorySteps(int steps) {
        maxHistorySteps_ = steps;
    }
    
    int maxHistorySteps() const { return maxHistorySteps_; }
    
    /// Clear all history (keep current fields)
    void clearHistory() {
        for (auto& [id, entry] : fields_) {
            entry.history.clear();
        }
        for (auto& [name, entry] : namedFields_) {
            entry.history.clear();
        }
    }
    
    // =========================================================================
    // Utility methods
    // =========================================================================
    
    /// Get number of fields
    size_t numFields() const { return fields_.size() + namedFields_.size(); }
    
    /// Clear all fields
    void clear() {
        fields_.clear();
        namedFields_.clear();
    }

private:
    // Internal field entry
    struct FieldEntry {
        std::unique_ptr<GridFunction> field;
        std::deque<std::unique_ptr<GridFunction>> history;
        bool isVector = false;
        int vdim = 1;
    };
    
    std::map<FieldId, FieldEntry> fields_;
    std::map<std::string, FieldEntry> namedFields_;
    int maxHistorySteps_ = 0;  ///< 0 = steady state, >0 = transient
};

}  // namespace mpfem

#endif  // MPFEM_FIELD_VALUES_HPP
