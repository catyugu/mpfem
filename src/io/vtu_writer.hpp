/**
 * @file vtu_writer.hpp
 * @brief VTU (VTK Unstructured Grid) file writer
 * 
 * Writes simulation results in VTK XML format for visualization
 * with ParaView, VisIt, and other VTK-compatible tools.
 */

#ifndef MPFEM_IO_VTU_WRITER_HPP
#define MPFEM_IO_VTU_WRITER_HPP

#include "core/types.hpp"
#include "mesh/mesh.hpp"
#include "dof/dof_handler.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

namespace mpfem {

/**
 * @brief Field data for VTU output
 */
struct VtuField {
    std::string name;
    enum class Type {
        Scalar,     ///< Single component (e.g., temperature, voltage)
        Vector,     ///< 3 components (e.g., displacement, velocity)
        Tensor      ///< 9 components (3x3 matrix)
    } type;
    
    /// Field values at each node
    /// For Scalar: [num_nodes]
    /// For Vector: [num_nodes * 3] (interleaved: x0,y0,z0,x1,y1,z1,...)
    /// For Tensor: [num_nodes * 9] (row-major)
    std::vector<Scalar> values;
    
    VtuField(const std::string& n, Type t) : name(n), type(t) {}
};

/**
 * @brief VTU file writer for unstructured meshes
 * 
 * Writes mesh geometry and solution fields in VTK XML format.
 * Supports all element types supported by mpfem.
 * 
 * Usage:
 * @code
 * VTUWriter writer;
 * writer.open("output.vtu");
 * writer.write_mesh(mesh);
 * writer.write_field("temperature", temperature_solution, dof_handler, VtuField::Type::Scalar);
 * writer.write_field("displacement", displacement_solution, dof_handler, VtuField::Type::Vector);
 * writer.close();
 * @endcode
 */
class VTUWriter {
public:
    VTUWriter();
    ~VTUWriter();
    
    /**
     * @brief Open a VTU file for writing
     * @param filename Output file path
     * @return true on success
     */
    bool open(const std::string& filename);
    
    /**
     * @brief Close the VTU file
     */
    void close();
    
    /**
     * @brief Check if file is open
     */
    bool is_open() const { return file_.is_open(); }
    
    /**
     * @brief Write mesh geometry
     * @param mesh The mesh to write
     * 
     * Writes all vertices and cells (volume elements) to the file.
     * Must be called before write_field().
     */
    void write_mesh(const Mesh& mesh);
    
    /**
     * @brief Write a scalar field defined at nodes
     * @param name Field name
     * @param values Field values at each node [num_nodes]
     */
    void write_point_data_scalar(const std::string& name, const std::vector<Scalar>& values);
    
    /**
     * @brief Write a vector field defined at nodes
     * @param name Field name
     * @param values Field values [num_nodes * 3] (interleaved)
     */
    void write_point_data_vector(const std::string& name, const std::vector<Scalar>& values);
    
    /**
     * @brief Write solution from DoFHandler
     * @param name Field name
     * @param solution Global solution vector
     * @param dof_handler DoF handler
     * @param type Field type (Scalar or Vector)
     * 
     * Extracts nodal values from the solution vector and writes them.
     * For Lagrange elements, DoFs correspond directly to nodes.
     */
    void write_field(const std::string& name,
                     const DynamicVector& solution,
                     const DoFHandler& dof_handler,
                     VtuField::Type type = VtuField::Type::Scalar);
    
    /**
     * @brief Write multiple fields
     * @param fields Vector of fields to write
     */
    void write_fields(const std::vector<VtuField>& fields);
    
    /**
     * @brief Write cell data (per-element values)
     * @param name Data name
     * @param values Per-cell values [num_cells]
     */
    void write_cell_data_scalar(const std::string& name, const std::vector<Scalar>& values);
    
    /**
     * @brief Write domain IDs as cell data
     * @param mesh The mesh
     * 
     * Writes the domain ID for each cell, useful for visualization
     * of material regions.
     */
    void write_domain_ids(const Mesh& mesh);
    
    /**
     * @brief Set binary output mode
     * @param binary If true, use base64-encoded binary data (smaller files)
     *               If false, use ASCII (human-readable, larger files)
     */
    void set_binary_mode(bool binary) { binary_mode_ = binary; }
    
    /**
     * @brief Write a complete VTU file in one call
     * @param filename Output file path
     * @param mesh The mesh
     * @param fields Solution fields
     */
    static void write(const std::string& filename,
                      const Mesh& mesh,
                      const std::vector<VtuField>& fields);
    
private:
    std::ofstream file_;
    bool binary_mode_ = false;
    
    // Mesh data cached during write_mesh
    SizeType num_points_ = 0;
    SizeType num_cells_ = 0;
    
    // VTK cell type mapping
    static int vtk_cell_type(ElementType type);
    
    // Write points section
    void write_points(const Mesh& mesh);
    
    // Write cells section
    void write_cells(const Mesh& mesh);
    
    // Base64 encoding for binary mode
    static std::string base64_encode(const std::vector<unsigned char>& data);
    
    // Write data array (ASCII or binary)
    void write_data_array(const std::string& name, 
                          const std::vector<Scalar>& values,
                          int num_components = 1);
};

}  // namespace mpfem

#endif  // MPFEM_IO_VTU_WRITER_HPP
