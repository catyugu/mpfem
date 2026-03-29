#ifndef MPFEM_MATERIAL_DATABASE_HPP
#define MPFEM_MATERIAL_DATABASE_HPP

#include "core/types.hpp"
#include "core/exception.hpp"
#include "io/exprtk_expression_parser.hpp"
#include "fe/element_transform.hpp"
#include "fe/coefficient.hpp"
#include <string>
#include <map>
#include <vector>
#include <optional>

namespace mpfem {

/**
 * @brief Material property model with unified storage for constants and expressions.
 * 
 * Design principles:
 * - Type-level distinction: scalar vs matrix properties via separate storage
 * - Properties can be either constants or expressions (evaluated at runtime)
 * - Expressions support variables like T (temperature), V (voltage), u (displacement)
 * - Evaluation is deferred to runtime using ExpressionParser
 * - Single unified access API: getScalar() / getMatrix() throw if not found
 */
struct MaterialPropertyModel {
    std::string tag;
    std::string label;

    // =======================================================================
    // Storage (private detail - use access methods)
    // =======================================================================
    
    /// Scalar properties (constants)
    std::map<std::string, double> scalarProperties;

    /// Matrix properties (constants)  
    std::map<std::string, Matrix3> matrixProperties;

    /// Scalar expressions (evaluated at runtime with variables)
    std::map<std::string, std::string> scalarExpressions;

    /// Matrix expressions (evaluated at runtime with variables)
    std::map<std::string, std::string> matrixExpressions;

    // =======================================================================
    // Unified property access (THROWS if not found)
    // Expression takes precedence over constant.
    // =======================================================================
    
    /**
     * @brief Get scalar property value (constant or expression).
     * @param name Property name
     * @param variables Variable values for expression evaluation
     * @return Evaluated scalar value
     * @throws ArgumentException if property not found (neither expression nor constant)
     */
    double getScalar(const std::string& name,
                     const std::map<std::string, double>& variables = {}) const {
        // Check expressions first
        auto exprIt = scalarExpressions.find(name);
        if (exprIt != scalarExpressions.end()) {
            return ExpressionParser::instance().evaluateScalar(exprIt->second, variables);
        }
        // Fall back to constant
        auto constIt = scalarProperties.find(name);
        if (constIt != scalarProperties.end()) {
            return constIt->second;
        }
        MPFEM_THROW(ArgumentException, "Scalar property '" + name + "' not found in material '" + tag + "'");
    }

    /**
     * @brief Get matrix property value (constant or expression).
     * @param name Property name
     * @param variables Variable values for expression evaluation
     * @return Evaluated matrix value
     * @throws ArgumentException if property not found (neither expression nor constant)
     */
    Matrix3 getMatrix(const std::string& name,
                      const std::map<std::string, double>& variables = {}) const {
        // Check expressions first
        auto exprIt = matrixExpressions.find(name);
        if (exprIt != matrixExpressions.end()) {
            return ExpressionParser::instance().evaluateMatrix(exprIt->second, variables);
        }
        // Fall back to constant
        auto constIt = matrixProperties.find(name);
        if (constIt != matrixProperties.end()) {
            return constIt->second;
        }
        MPFEM_THROW(ArgumentException, "Matrix property '" + name + "' not found in material '" + tag + "'");
    }

    // =======================================================================
    // Property existence queries (for optional properties)
    // =======================================================================
    
    bool hasScalar(const std::string& name) const {
        return scalarExpressions.count(name) > 0 || scalarProperties.count(name) > 0;
    }

    bool hasMatrix(const std::string& name) const {
        return matrixExpressions.count(name) > 0 || matrixProperties.count(name) > 0;
    }

    // =======================================================================
    // Factory methods for creating coefficients (for optional properties)
    // Returns nullptr if property not found.
    // Expression takes precedence if both exist.
    // =======================================================================
    
    /// Create a scalar coefficient from expression or constant property
    /// Returns nullptr if property not found (expression or constant)
    template<typename Func>
    std::unique_ptr<Coefficient> createScalarCoefficient(const std::string& name, Func&& getVars) const {
        if (hasScalar(name)) {
            return std::make_unique<ScalarCoefficient>(
                [this, name, getVars](ElementTransform& trans, Real& result, Real) {
                    auto vars = getVars(trans);
                    result = this->getScalar(name, vars);
                });
        }
        return nullptr;
    }

    /// Create a matrix coefficient from expression or constant property
    /// Returns nullptr if property not found (expression or constant)
    template<typename Func>
    std::unique_ptr<MatrixCoefficient> createMatrixCoefficient(const std::string& name, Func&& getVars) const {
        if (hasMatrix(name)) {
            return std::make_unique<MatrixFunctionCoefficient>(
                [this, name, getVars](ElementTransform& trans, Matrix3& result, Real) {
                    auto vars = getVars(trans);
                    result = this->getMatrix(name, vars);
                });
        }
        return nullptr;
    }

    // =======================================================================
    // Legacy setters (for testing/manual material construction)
    // =======================================================================
    
    void setScalarProperty(const std::string& name, double value) {
        scalarProperties[name] = value;
    }
    
    void setMatrixProperty(const std::string& name, const Matrix3& mat) {
        matrixProperties[name] = mat;
    }

    void setScalarExpression(const std::string& name, const std::string& expr) {
        scalarExpressions[name] = expr;
    }

    void setMatrixExpression(const std::string& name, const std::string& expr) {
        matrixExpressions[name] = expr;
    }
};

/**
 * @brief Database of material properties.
 */
class MaterialDatabase {
public:
    void addMaterial(const MaterialPropertyModel& material) {
        materials_[material.tag] = material;
    }

    const MaterialPropertyModel* getMaterial(const std::string& tag) const {
        auto it = materials_.find(tag);
        return it != materials_.end() ? &it->second : nullptr;
    }

    bool hasMaterial(const std::string& tag) const {
        return materials_.find(tag) != materials_.end();
    }

    std::vector<std::string> getMaterialTags() const {
        std::vector<std::string> tags;
        for (const auto& [tag, _] : materials_) tags.push_back(tag);
        return tags;
    }

    size_t size() const { return materials_.size(); }
    void clear() { materials_.clear(); }

    auto begin() const { return materials_.begin(); }
    auto end() const { return materials_.end(); }

private:
    std::map<std::string, MaterialPropertyModel> materials_;
};

}  // namespace mpfem

#endif  // MPFEM_MATERIAL_DATABASE_HPP