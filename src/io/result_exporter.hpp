#ifndef MPFEM_RESULT_EXPORTER_HPP
#define MPFEM_RESULT_EXPORTER_HPP

#include "mesh/mesh.hpp"
#include "core/types.hpp"
#include "physics/field_values.hpp"

#include <string>
#include <vector>

namespace mpfem {

// Forward declarations
struct SteadyResult;
struct TransientResult;

/**
 * @brief Unified result exporter.
 * 
 * Single interface for exporting both steady and transient results.
 */
class ResultExporter {
public:
    // COMSOL text format
    static void exportComsolText(const SteadyResult& result, const Mesh& mesh, 
                                 const std::string& filename);
    static void exportComsolText(const TransientResult& result, const Mesh& mesh,
                                 const std::string& filename);
    
    // VTU format
    static void exportVtu(const SteadyResult& result, const Mesh& mesh,
                           const std::string& filename);
    static void exportVtu(const TransientResult& result, const Mesh& mesh,
                           const std::string& filename);

private:
    static std::string getCurrentTimestamp();
    static void exportComsolTextImpl(const FieldValues& fields, const Mesh& mesh, 
                                      const std::string& filename, Real time = -1);
    static void exportVtuImpl(const FieldValues& fields, const Mesh& mesh,
                               const std::string& filename);
};

}  // namespace mpfem

#endif  // MPFEM_RESULT_EXPORTER_HPP
