#ifndef MPFEM_FIELD_VALUES_HPP
#define MPFEM_FIELD_VALUES_HPP

#include "core/exception.hpp"
#include "core/tensor_shape.hpp"
#include "core/types.hpp"
#include "fe/fe_space.hpp"
#include "fe/grid_function.hpp"
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mpfem {

    // =============================================================================
    // Field identifiers - string-based for extensibility
    // =============================================================================

    // Field names are plain strings: "Temperature", "ElectricPotential", "Displacement".
    // No enum needed - new physics fields can be added without modifying this header.

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

        explicit FieldValues(int maxHistorySteps) : maxHistorySteps_(maxHistorySteps) { }

        // Rule of Zero: default copy/move operations automatically deep-copy via GridFunction value semantics
        FieldValues(const FieldValues&) = default;
        FieldValues& operator=(const FieldValues&) = default;
        FieldValues(FieldValues&&) = default;
        FieldValues& operator=(FieldValues&&) = default;

        /// Unified field creation with explicit TensorShape
        void createField(std::string_view id, const FESpace* fes, TensorShape shape, Real initVal = 0.0)
        {
            auto& entry = fields_[std::string(id)];
            entry.current = GridFunction(fes, initVal);
            entry.shape_ = shape;
            entry.maxHistory = maxHistorySteps_;
            allocateHistory(entry, fes);
        }

        GridFunction& current(std::string_view id)
        {
            return fields_.at(std::string(id)).current;
        }

        const GridFunction& current(std::string_view id) const
        {
            return fields_.at(std::string(id)).current;
        }

        bool hasField(std::string_view id) const
        {
            return fields_.find(std::string(id)) != fields_.end();
        }

        GridFunction& history(std::string_view id, int stepsBack = 1)
        {
            auto it = fields_.find(std::string(id));
            MPFEM_ASSERT(it != fields_.end(),
                "Field not found: " + std::string(id));
            MPFEM_ASSERT(it->second.history.size() >= static_cast<size_t>(stepsBack),
                "History not available for field: " + std::string(id) + ", requested " + std::to_string(stepsBack) + " steps back, available: " + std::to_string(static_cast<int>(it->second.history.size())));
            // Ring buffer: compute physical index from logical stepsBack
            int idx = (it->second.historyHead - stepsBack + it->second.maxHistory)
                % it->second.maxHistory;
            return it->second.history[idx];
        }

        const GridFunction& history(std::string_view id, int stepsBack = 1) const
        {
            auto it = fields_.find(std::string(id));
            MPFEM_ASSERT(it != fields_.end(),
                "Field not found: " + std::string(id));
            MPFEM_ASSERT(it->second.history.size() >= static_cast<size_t>(stepsBack),
                "History not available for field: " + std::string(id));
            int idx = (it->second.historyHead - stepsBack + it->second.maxHistory)
                % it->second.maxHistory;
            return it->second.history[idx];
        }

        void advanceTime()
        {
            for (auto& [id, entry] : fields_) {
                if (entry.maxHistory > 0) {
                    // Copy current field to ring buffer at head position
                    entry.history[entry.historyHead] = entry.current;
                    // Advance head (circular)
                    entry.historyHead = (entry.historyHead + 1) % entry.maxHistory;
                    entry.historyCount = std::min(entry.historyCount + 1, entry.maxHistory);
                }
            }
        }

        void setMaxHistorySteps(int steps)
        {
            maxHistorySteps_ = steps;
            // Reallocate history buffers for all existing fields
            for (auto& [id, entry] : fields_) {
                entry.maxHistory = steps;
                allocateHistory(entry, entry.current.fes());
            }
        }

        int maxHistorySteps() const { return maxHistorySteps_; }

        void clearHistory()
        {
            for (auto& [id, entry] : fields_) {
                entry.historyHead = 0;
                entry.historyCount = 0;
            }
        }

        size_t numFields() const { return fields_.size(); }

        void clear()
        {
            fields_.clear();
        }

    private:
        struct FieldState {
            GridFunction current;
            std::vector<GridFunction> history; // Ring buffer with value semantics
            int historyHead = 0;
            int historyCount = 0;
            int maxHistory = 0;
            TensorShape shape_; // Replaces isVector and vdim
        };

        void allocateHistory(FieldState& entry, const FESpace* fes)
        {
            entry.history.clear();
            entry.history.resize(entry.maxHistory, GridFunction(fes));
            entry.historyHead = 0;
            entry.historyCount = 0;
        }

        std::unordered_map<std::string, FieldState> fields_;
        int maxHistorySteps_ = 0;
    };

} // namespace mpfem

#endif // MPFEM_FIELD_VALUES_HPP
