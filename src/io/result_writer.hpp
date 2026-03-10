/**
 * @file result_writer.hpp
 * @brief Result output writer for VTU and COMSOL formats
 */

#ifndef MPFEM_IO_RESULT_WRITER_HPP
#define MPFEM_IO_RESULT_WRITER_HPP

#include "core/types.hpp"
#include <string>
#include <vector>
#include <map>
#include <fstream>

namespace mpfem {

// Forward declarations
class Mesh;
class DynamicVector;

/**
 * @brief Field data for output
 */
struct FieldData {
    std::string name;                   ///< Field name
    std::string unit;                   ///< Unit string
    const DynamicVector* data;          ///< Field values at nodes
    int n_components = 1;               ///< Number of components (1=scalar, 3=vector)
    
    FieldData() = default;
    FieldData(const std::string& n, const DynamicVector* d, int nc = 1)
        : name(n), data(d), n_components(nc) {}
};

/**
 * @brief Result writer for various output formats
 */
class ResultWriter {
public:
    ResultWriter();
    ~ResultWriter() = default;
    
    /**
     * @brief Write results in VTU format (VTK Unstructured Grid)
     * @param filename Output file path
     * @param mesh Mesh object
     * @param fields Map of field name to field data
     */
    bool write_vtu(const std::string& filename,
                   const Mesh& mesh,
                   const std::map<std::string, FieldData>& fields);
    
    /**
     * @brief Write single field in VTU format
     */
    bool write_vtu(const std::string& filename,
                   const Mesh& mesh,
                   const DynamicVector& solution,
                   const std::string& field_name,
                   int n_components = 1);
    
    /**
     * @brief Write results in COMSOL text format
     * @param filename Output file path
     * @param mesh Mesh object
     * @param fields Map of field name to field data
     * @param model_name Model name for header
     */
    bool write_comsol(const std::string& filename,
                      const Mesh& mesh,
                      const std::map<std::string, FieldData>& fields,
                      const std::string& model_name = "mpfem");
    
    /**
     * @brief Write single field in COMSOL format
     */
    bool write_comsol(const std::string& filename,
                      const Mesh& mesh,
                      const DynamicVector& solution,
                      const std::string& field_name,
                      const std::string& unit = "");
    
    /**
     * @brief Set precision for floating point output
     */
    void set_precision(int prec) { precision_ = prec; }

private:
    int precision_ = 15;
    
    /**
     * @brief Write VTK cell data for different element types
     */
    void write_vtu_cells(std::ofstream& ofs, const Mesh& mesh);
    
    /**
     * @brief Get VTK cell type for element type
     */
    int get_vtk_cell_type(int n_nodes, int dim) const;
};

} // namespace mpfem

#endif // MPFEM_IO_RESULT_WRITER_HPP
