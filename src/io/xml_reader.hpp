/**
 * @file xml_reader.hpp
 * @brief XML configuration reader for case files
 */

#ifndef MPFEM_IO_XML_READER_HPP
#define MPFEM_IO_XML_READER_HPP

#include "variable_system.hpp"
#include "core/types.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace tinyxml2 {
    class XMLElement;
    class XMLDocument;
}

namespace mpfem {

/**
 * @brief Boundary condition configuration
 */
struct BoundaryConditionConfig {
    std::string kind;                           ///< BC type (voltage, convection, etc.)
    std::vector<Index> ids;                     ///< Boundary IDs
    std::map<std::string, std::string> params;  ///< Parameters (value, h, T_inf, etc.)
};

/**
 * @brief Source term configuration
 */
struct SourceConfig {
    std::string kind;                           ///< Source type
    std::vector<Index> domains;                 ///< Domain IDs
    std::string value;                          ///< Source value expression
};

/**
 * @brief Solver configuration
 */
struct SolverConfig {
    std::string type = "direct";                ///< Solver type (direct, cg_gs, etc.)
    int max_iter = 1000;                        ///< Maximum iterations
    Scalar tolerance = 1e-10;                   ///< Convergence tolerance
    int print_level = 0;                        ///< Output level
};

/**
 * @brief Physics field configuration
 */
struct PhysicsConfig {
    std::string kind;                           ///< Physics kind (electrostatics, etc.)
    int order = 1;                              ///< Polynomial order
    SolverConfig solver;
    std::vector<BoundaryConditionConfig> boundaries;
    std::vector<SourceConfig> sources;
};

/**
 * @brief Coupled physics configuration
 */
struct CouplingConfig {
    std::string name;                           ///< Coupling name
    std::string kind;                           ///< Coupling kind (joule_heating, etc.)
    std::vector<std::string> physics;           ///< Coupled physics names
    std::vector<Index> domains;                 ///< Domain IDs where coupling applies
};

/**
 * @brief Nonlinear solver configuration
 */
struct NonlinearConfig {
    std::string method = "picard";              ///< Iteration method (picard, newton)
    int max_iter = 20;                          ///< Maximum iterations
    Scalar tolerance = 1e-8;                    ///< Convergence tolerance
    Scalar relaxation = 1.0;                    ///< Under-relaxation factor
};

/**
 * @brief Material assignment configuration
 */
struct MaterialAssignment {
    std::vector<Index> domains;                 ///< Domain IDs
    std::string material_tag;                   ///< Material tag
};

/**
 * @brief Complete case configuration
 */
struct CaseConfig {
    // Paths
    std::string mesh_path;
    std::string material_path;
    std::string result_path;
    
    // Study type
    std::string study_type = "steady";          ///< steady, transient, eigenvalue
    
    // Variables
    VariableSystem variables;
    
    // Materials
    std::vector<MaterialAssignment> material_assignments;
    
    // Physics fields
    std::vector<PhysicsConfig> physics;
    std::map<std::string, size_t> physics_index;  ///< name -> index in physics vector
    
    // Couplings
    std::vector<CouplingConfig> couplings;
    
    // Nonlinear solver settings
    NonlinearConfig nonlinear;
    
    // Case name
    std::string name;
};

/**
 * @brief XML configuration file reader
 * 
 * Reads case.xml files and produces a CaseConfig structure.
 */
class XmlReader {
public:
    XmlReader();
    ~XmlReader();
    
    /**
     * @brief Read case configuration from XML file
     * @param filename Path to case.xml
     * @return Parsed configuration
     */
    CaseConfig read(const std::string& filename);
    
    /**
     * @brief Get last error message
     */
    const std::string& error() const { return error_; }
    
private:
    std::unique_ptr<tinyxml2::XMLDocument> doc_;
    std::string error_;
    
    void parse_paths(tinyxml2::XMLElement* elem, CaseConfig& config);
    void parse_variables(tinyxml2::XMLElement* elem, CaseConfig& config);
    void parse_materials(tinyxml2::XMLElement* elem, CaseConfig& config);
    void parse_physics(tinyxml2::XMLElement* elem, CaseConfig& config);
    void parse_couplings(tinyxml2::XMLElement* elem, CaseConfig& config);
    
    BoundaryConditionConfig parse_boundary(tinyxml2::XMLElement* elem);
    SourceConfig parse_source(tinyxml2::XMLElement* elem);
    SolverConfig parse_solver(tinyxml2::XMLElement* elem);
    
    /**
     * @brief Parse ID list like "1-7,9-14,16-42" into vector of IDs
     */
    std::vector<Index> parse_id_list(const std::string& id_str) const;
    
    /**
     * @brief Parse domain list (supports ranges)
     */
    std::vector<Index> parse_domain_list(const std::string& domain_str) const;
};

} // namespace mpfem

#endif // MPFEM_IO_XML_READER_HPP
