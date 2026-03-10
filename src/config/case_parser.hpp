/**
 * @file case_parser.hpp
 * @brief Parser for case.xml and material.xml configuration files
 */

#ifndef MPFEM_CONFIG_CASE_PARSER_HPP
#define MPFEM_CONFIG_CASE_PARSER_HPP

#include "case_config.hpp"
#include <string>
#include <memory>

// Forward declaration
namespace tinyxml2 {
    class XMLDocument;
    class XMLElement;
}

namespace mpfem {

/**
 * @brief Parser for simulation case files
 * 
 * Parses case.xml files that define simulation setup including
 * variables, materials, physics, boundary conditions, and coupling.
 */
class CaseParser {
public:
    CaseParser();
    ~CaseParser();
    
    /**
     * @brief Parse case.xml file
     * @param filename Path to case.xml
     * @param base_dir Base directory for resolving relative paths
     * @return Parsed configuration
     */
    CaseConfig parse_case(const std::string& filename,
                          const std::string& base_dir = "");
    
    /**
     * @brief Parse material.xml file
     * @param filename Path to material.xml
     * @return Parsed material database
     */
    MaterialDatabase parse_materials(const std::string& filename);
    
    /**
     * @brief Evaluate an expression with variable substitution
     * @param expr Expression string (e.g., "Vtot * 2", "293.15[K]")
     * @param config Case configuration with variables
     * @return Evaluated value
     */
    static Scalar evaluate_expression(const std::string& expr,
                                       const CaseConfig& config);
    
    /**
     * @brief Parse a value with unit (e.g., "9[cm]" -> 0.09)
     * @param value_str Value string with optional unit
     * @return Value in SI units
     */
    static Scalar parse_value_with_unit(const std::string& value_str);
    
    /**
     * @brief Parse a list of IDs (e.g., "1-7" or "1,2,3,5")
     * @param id_str ID string
     * @return List of IDs
     */
    static std::vector<Index> parse_id_list(const std::string& id_str);
    
    /**
     * @brief Parse a tensor string (e.g., "{'1','0','0','0','1','0','0','0','1'}")
     * @param tensor_str Tensor string from COMSOL format
     * @return Vector of 9 values (row-major 3x3)
     */
    static std::vector<Scalar> parse_tensor(const std::string& tensor_str);
    
    /**
     * @brief Get last error message
     */
    const std::string& error() const { return error_; }
    
    /**
     * @brief Check if last parse was successful
     */
    bool success() const { return success_; }

private:
    std::unique_ptr<tinyxml2::XMLDocument> doc_;
    std::string error_;
    bool success_ = false;
    
    // Parse helper methods
    void parse_paths(tinyxml2::XMLElement* elem, CaseConfig& config,
                     const std::string& base_dir);
    
    void parse_variables(tinyxml2::XMLElement* elem, CaseConfig& config);
    
    void parse_material_assignments(tinyxml2::XMLElement* elem, CaseConfig& config);
    
    void parse_physics(tinyxml2::XMLElement* elem, CaseConfig& config);
    
    void parse_boundary_condition(tinyxml2::XMLElement* elem,
                                   PhysicsConfig& physics);
    
    void parse_source(tinyxml2::XMLElement* elem, PhysicsConfig& physics);
    
    void parse_solver(tinyxml2::XMLElement* elem, SolverConfig& solver);
    
    void parse_coupled_physics(tinyxml2::XMLElement* elem, CaseConfig& config);
    
    void parse_coupling(tinyxml2::XMLElement* elem, CaseConfig& config);
    
    void parse_material(tinyxml2::XMLElement* elem, MaterialDatabase& db);
    
    void parse_property_group(tinyxml2::XMLElement* elem, MaterialConfig& mat);
};

}  // namespace mpfem

#endif  // MPFEM_CONFIG_CASE_PARSER_HPP
