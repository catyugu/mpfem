#ifndef MPFEM_RESULT_EXPORTER_HPP
#define MPFEM_RESULT_EXPORTER_HPP

#include "mesh/mesh.hpp"
#include "core/types.hpp"

#include <string>
#include <vector>
#include <map>

namespace mpfem {

/**
 * @brief Field result data for export.
 */
struct FieldResult {
    std::string name;                    ///< Field name (e.g., "V", "T", "disp")
    std::string unit;                    ///< Unit string (e.g., "V", "K", "m")
    std::vector<double> nodalValues;     ///< One value per node
};

/**
 * @brief Exports simulation results to various formats.
 */
class ResultExporter {
public:
    /**
     * @brief Export results to COMSOL-compatible text format.
     * 
     * Format:
     * % Header comments
     * % Nodes: N
     * % Expressions: M
     * x y z field1 field2 ...
     * 
     * @param filename Output file path.
     * @param mesh The mesh (for coordinates).
     * @param fields Vector of field results to export.
     * @param description Optional description string.
     */
    static void exportComsolText(const std::string& filename,
                                 const Mesh& mesh,
                                 const std::vector<FieldResult>& fields,
                                 const std::string& description = "");

    /**
     * @brief Export results to VTK VTU format (XML-based).
     * 
     * Creates an unstructured grid VTU file suitable for ParaView.
     * 
     * @param filename Output file path (.vtu).
     * @param mesh The mesh.
     * @param fields Vector of field results to export.
     */
    static void exportVtu(const std::string& filename,
                          const Mesh& mesh,
                          const std::vector<FieldResult>& fields);

    /**
     * @brief Export vector field results to VTU format.
     * 
     * @param filename Output file path (.vtu).
     * @param mesh The mesh.
     * @param scalarFields Vector of scalar field results.
     * @param vectorFields Map of vector field name to components (x, y, z values per node).
     */
    static void exportVtuWithVectors(const std::string& filename,
                                     const Mesh& mesh,
                                     const std::vector<FieldResult>& scalarFields,
                                     const std::map<std::string, std::vector<Vector3>>& vectorFields);

private:
    static void writeVtuHeader(std::ofstream& file);
    static void writeVtuFooter(std::ofstream& file);
    static std::string getCurrentTimestamp();
};

}  // namespace mpfem

#endif  // MPFEM_RESULT_EXPORTER_HPP
