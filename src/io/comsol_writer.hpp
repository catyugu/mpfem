/**
 * @file comsol_writer.hpp
 * @brief COMSOL-compatible result file writer
 * 
 * Writes simulation results in a format compatible with COMSOL
 * result files for easy comparison and verification.
 */

#ifndef MPFEM_IO_COMSOL_WRITER_HPP
#define MPFEM_IO_COMSOL_WRITER_HPP

#include "core/types.hpp"
#include "mesh/mesh.hpp"
#include "dof/dof_handler.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace mpfem {

/**
 * @brief Field data for COMSOL output
 */
struct ComsolField {
    std::string name;           ///< Field name with unit (e.g., "V (V)", "T (K)")
    std::string description;    ///< Field description
    
    /// Field values at each node [num_nodes]
    std::vector<Scalar> values;
    
    ComsolField(const std::string& n, const std::string& desc = "")
        : name(n), description(desc) {}
};

/**
 * @brief COMSOL-compatible result file writer
 * 
 * Writes results in the same format as COMSOL exported text files,
 * enabling direct comparison with COMSOL simulation results.
 * 
 * Usage:
 * @code
 * ComsolWriter writer;
 * writer.open("result.txt");
 * writer.write_header(mesh, fields);
 * writer.write_data(mesh, fields);
 * writer.close();
 * @endcode
 */
class ComsolWriter {
public:
    ComsolWriter();
    ~ComsolWriter();
    
    /**
     * @brief Open a COMSOL result file for writing
     * @param filename Output file path
     * @return true on success
     */
    bool open(const std::string& filename);
    
    /**
     * @brief Close the file
     */
    void close();
    
    /**
     * @brief Check if file is open
     */
    bool is_open() const { return file_.is_open(); }
    
    /**
     * @brief Set model name for header
     */
    void set_model_name(const std::string& name) { model_name_ = name; }
    
    /**
     * @brief Set COMSOL version string
     */
    void set_version(const std::string& version) { version_ = version; }
    
    /**
     * @brief Set length unit
     */
    void set_length_unit(const std::string& unit) { length_unit_ = unit; }
    
    /**
     * @brief Write header section
     * @param mesh The mesh
     * @param fields Vector of fields to write
     */
    void write_header(const Mesh& mesh, const std::vector<ComsolField>& fields);
    
    /**
     * @brief Write data section (coordinates and field values)
     * @param mesh The mesh
     * @param fields Vector of fields to write
     */
    void write_data(const Mesh& mesh, const std::vector<ComsolField>& fields);
    
    /**
     * @brief Write a complete result file in one call
     * @param filename Output file path
     * @param mesh The mesh
     * @param fields Vector of fields to write
     * @param model_name Optional model name
     */
    static void write(const std::string& filename,
                      const Mesh& mesh,
                      const std::vector<ComsolField>& fields,
                      const std::string& model_name = "mpfem");
    
    /**
     * @brief Create ComsolField from solution vector
     * @param name Field name with unit
     * @param solution Global solution vector
     * @param dof_handler DoF handler
     * @param type Field type (scalar or vector magnitude)
     */
    static ComsolField create_field(const std::string& name,
                                    const DynamicVector& solution,
                                    const DoFHandler& dof_handler,
                                    bool is_vector = false);
    
    /**
     * @brief Create ComsolField for vector displacement magnitude
     * @param name Field name with unit
     * @param solution Global solution vector (displacement components)
     * @param dof_handler DoF handler
     * @param dim Spatial dimension
     */
    static ComsolField create_displacement_magnitude(
        const std::string& name,
        const DynamicVector& solution,
        const DoFHandler& dof_handler,
        int dim);
    
private:
    std::ofstream file_;
    std::string model_name_ = "mpfem";
    std::string version_ = "mpfem 1.0";
    std::string length_unit_ = "m";
    
    /// Get current date/time string
    static std::string get_current_datetime();
};

}  // namespace mpfem

#endif  // MPFEM_IO_COMSOL_WRITER_HPP
