#ifndef MPFEM_RESULT_EXPORTER_HPP
#define MPFEM_RESULT_EXPORTER_HPP

#include "core/types.hpp"
#include "field/field_values.hpp"

#include <string>
#include <vector>

namespace mpfem {

    class Mesh;

    /**
     * @brief Unified result exporter.
     *
     * Single interface for exporting both steady and transient results.
     */
    class ResultExporter {
    public:
        // COMSOL text format
        static void exportComsolText(const FieldValues& fields, const Mesh& mesh,
            const std::string& filename, Real time = -1);
        static void exportComsolText(const std::vector<FieldValues>& snapshots,
            const std::vector<Real>& times,
            const Mesh& mesh,
            const std::string& filename);

        // VTU format
        static void exportVtu(const FieldValues& fields, const Mesh& mesh,
            const std::string& filename);
        static void exportVtu(const std::vector<FieldValues>& snapshots, const Mesh& mesh,
            const std::string& filename);

    private:
        static std::string getCurrentTimestamp();
        static void exportComsolTextImpl(const FieldValues& fields, const Mesh& mesh,
            const std::string& filename, Real time = -1);
        static void exportVtuImpl(const FieldValues& fields, const Mesh& mesh,
            const std::string& filename);
    };

} // namespace mpfem

#endif // MPFEM_RESULT_EXPORTER_HPP
