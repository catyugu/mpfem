#ifndef MPFEM_FIELD_VALUES_HPP
#define MPFEM_FIELD_VALUES_HPP

#include "core/types.hpp"
#include "core/exception.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include <map>
#include <memory>
#include <vector>
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
    
    // Copyable for result storage - deep copies GridFunctions
    FieldValues(const FieldValues& other) {
        maxHistorySteps_ = other.maxHistorySteps_;
        for (const auto& [id, entry] : other.fields_) {
            auto& newEntry = fields_[id];
            newEntry.isVector = entry.isVector;
            newEntry.vdim = entry.vdim;
            newEntry.maxHistory_ = entry.maxHistory_;
            newEntry.historyHead = entry.historyHead;
            newEntry.historyCount = entry.historyCount;
            if (entry.field) {
                newEntry.field = std::make_unique<GridFunction>(*entry.field);
            }
            // Copy history buffer (ring buffer)
            newEntry.historyBuffer.reserve(entry.historyBuffer.size());
            for (const auto& hist : entry.historyBuffer) {
                newEntry.historyBuffer.push_back(std::make_unique<GridFunction>(*hist));
            }
        }
    }
    
    FieldValues& operator=(const FieldValues& other) {
        if (this != &other) {
            fields_.clear();
            maxHistorySteps_ = other.maxHistorySteps_;
            for (const auto& [id, entry] : other.fields_) {
                auto& newEntry = fields_[id];
                newEntry.isVector = entry.isVector;
                newEntry.vdim = entry.vdim;
                newEntry.maxHistory_ = entry.maxHistory_;
                newEntry.historyHead = entry.historyHead;
                newEntry.historyCount = entry.historyCount;
                if (entry.field) {
                    newEntry.field = std::make_unique<GridFunction>(*entry.field);
                }
                // Copy history buffer (ring buffer)
                newEntry.historyBuffer.reserve(entry.historyBuffer.size());
                for (const auto& hist : entry.historyBuffer) {
                    newEntry.historyBuffer.push_back(std::make_unique<GridFunction>(*hist));
                }
            }
        }
        return *this;
    }
    
    FieldValues(FieldValues&&) = default;
    FieldValues& operator=(FieldValues&&) = default;
    
    void createScalarField(FieldId id, const FESpace* fes, Real initVal = 0.0) {
        auto& entry = fields_[id];
        entry.field = std::make_unique<GridFunction>(fes, initVal);
        entry.isVector = false;
        entry.maxHistory_ = maxHistorySteps_;
        allocateHistoryBuffer(entry, fes);
    }
    
    void createVectorField(FieldId id, const FESpace* fes, int vdim) {
        auto& entry = fields_[id];
        auto gf = std::make_unique<GridFunction>(fes);
        gf->values().setZero();
        entry.field = std::move(gf);
        entry.isVector = true;
        entry.vdim = vdim;
        entry.maxHistory_ = maxHistorySteps_;
        allocateHistoryBuffer(entry, fes);
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
        MPFEM_ASSERT(it->second.historyBuffer.size() >= static_cast<size_t>(stepsBack),
            "History not available for field: " + toString(id) + 
            ", requested " + std::to_string(stepsBack) + " steps back, available: " +
            std::to_string(static_cast<int>(it->second.historyBuffer.size())));
        // Ring buffer: compute physical index from logical stepsBack
        int idx = (it->second.historyHead - stepsBack + it->second.maxHistory_) 
                   % it->second.maxHistory_;
        return *it->second.historyBuffer[idx];
    }
    
    const GridFunction& history(FieldId id, int stepsBack = 1) const {
        auto it = fields_.find(id);
        MPFEM_ASSERT(it != fields_.end(), 
            "Field not found: " + toString(id));
        MPFEM_ASSERT(it->second.historyBuffer.size() >= static_cast<size_t>(stepsBack),
            "History not available for field: " + toString(id));
        int idx = (it->second.historyHead - stepsBack + it->second.maxHistory_) 
                   % it->second.maxHistory_;
        return *it->second.historyBuffer[idx];
    }
    
    void advanceTime() {
        for (auto& [id, entry] : fields_) {
            if (entry.maxHistory_ > 0 && entry.field) {
                // Copy current field to ring buffer at head position
                *entry.historyBuffer[entry.historyHead] = *entry.field;
                // Advance head (circular)
                entry.historyHead = (entry.historyHead + 1) % entry.maxHistory_;
                entry.historyCount = std::min(entry.historyCount + 1, entry.maxHistory_);
            }
        }
    }
    
    void setMaxHistorySteps(int steps) {
        maxHistorySteps_ = steps;
        // Reallocate history buffers for all existing fields
        for (auto& [id, entry] : fields_) {
            if (entry.field) {
                entry.maxHistory_ = steps;
                allocateHistoryBuffer(entry, entry.field->fes());
            }
        }
    }
    
    int maxHistorySteps() const { return maxHistorySteps_; }
    
    void clearHistory() {
        for (auto& [id, entry] : fields_) {
            entry.historyHead = 0;
            entry.historyCount = 0;
        }
    }
    
    size_t numFields() const { return fields_.size(); }
    
    void clear() {
        fields_.clear();
    }

private:
    struct FieldEntry {
        std::unique_ptr<GridFunction> field;
        std::vector<std::unique_ptr<GridFunction>> historyBuffer;  // Pre-allocated ring buffer
        int historyHead = 0;    // Current write position in ring buffer
        int historyCount = 0;   // Number of valid history entries
        int maxHistory_ = 0;     // Max history depth for this field
        bool isVector = false;
        int vdim = 1;
    };
    
    void allocateHistoryBuffer(FieldEntry& entry, const FESpace* fes) {
        entry.historyBuffer.clear();
        entry.historyBuffer.reserve(entry.maxHistory_);
        for (int i = 0; i < entry.maxHistory_; ++i) {
            entry.historyBuffer.push_back(std::make_unique<GridFunction>(fes));
        }
        entry.historyHead = 0;
        entry.historyCount = 0;
    }
    
    std::map<FieldId, FieldEntry> fields_;
    int maxHistorySteps_ = 0;
};

}  // namespace mpfem

#endif  // MPFEM_FIELD_VALUES_HPP
