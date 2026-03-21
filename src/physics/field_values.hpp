#ifndef MPFEM_FIELD_VALUES_HPP
#define MPFEM_FIELD_VALUES_HPP

#include "core/types.hpp"
#include "core/exception.hpp"
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
};

inline std::string toString(FieldId id) {
    switch (id) {
        case FieldId::Temperature: return "Temperature";
        case FieldId::ElectricPotential: return "ElectricPotential";
        case FieldId::Displacement: return "Displacement";
    }
    return "Unknown";
}

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
    FieldValues() = default;
    
    explicit FieldValues(int maxHistorySteps) : maxHistorySteps_(maxHistorySteps) {}
    
    FieldValues(const FieldValues&) = delete;
    FieldValues& operator=(const FieldValues&) = delete;
    FieldValues(FieldValues&&) = default;
    FieldValues& operator=(FieldValues&&) = default;
    
    void createScalarField(FieldId id, const FESpace* fes, Real initVal = 0.0) {
        auto& entry = fields_[id];
        entry.field = std::make_unique<GridFunction>(fes, initVal);
        entry.isVector = false;
    }
    
    void createVectorField(FieldId id, const FESpace* fes, int vdim) {
        auto& entry = fields_[id];
        auto gf = std::make_unique<GridFunction>(fes);
        gf->values().setZero();
        entry.field = std::move(gf);
        entry.isVector = true;
        entry.vdim = vdim;
    }
    
    GridFunction& current(FieldId id) {
        return *fields_.at(id).field;
    }
    
    const GridFunction& current(FieldId id) const {
        return *fields_.at(id).field;
    }
    
    bool hasField(FieldId id) const {
        return fields_.find(id) != fields_.end();
    }
    
    GridFunction& history(FieldId id, int stepsBack = 1) {
        auto it = fields_.find(id);
        MPFEM_ASSERT(it != fields_.end(), 
            "Field not found: " + toString(id));
        MPFEM_ASSERT(it->second.history.size() >= static_cast<size_t>(stepsBack),
            "History not available for field: " + toString(id) + 
            ", requested " + std::to_string(stepsBack) + " steps back, available: " +
            std::to_string(static_cast<int>(it->second.history.size())));
        return *it->second.history[stepsBack - 1];
    }
    
    const GridFunction& history(FieldId id, int stepsBack = 1) const {
        auto it = fields_.find(id);
        MPFEM_ASSERT(it != fields_.end(), 
            "Field not found: " + toString(id));
        MPFEM_ASSERT(it->second.history.size() >= static_cast<size_t>(stepsBack),
            "History not available for field: " + toString(id));
        return *it->second.history[stepsBack - 1];
    }
    
    void advanceTime() {
        for (auto& [id, entry] : fields_) {
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
    
    void setMaxHistorySteps(int steps) {
        maxHistorySteps_ = steps;
    }
    
    int maxHistorySteps() const { return maxHistorySteps_; }
    
    void clearHistory() {
        for (auto& [id, entry] : fields_) {
            entry.history.clear();
        }
    }
    
    size_t numFields() const { return fields_.size(); }
    
    void clear() {
        fields_.clear();
    }

private:
    struct FieldEntry {
        std::unique_ptr<GridFunction> field;
        std::deque<std::unique_ptr<GridFunction>> history;
        bool isVector = false;
        int vdim = 1;
    };
    
    std::map<FieldId, FieldEntry> fields_;
    int maxHistorySteps_ = 0;
};

}  // namespace mpfem

#endif  // MPFEM_FIELD_VALUES_HPP
