#ifndef MPFEM_MATERIAL_DATABASE_HPP
#define MPFEM_MATERIAL_DATABASE_HPP

#include "core/types.hpp"
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
 */
struct MaterialPropertyModel {
    std::string tag;
    std::string label;

    // =======================================================================
    // Scalar properties (constants)
    // =======================================================================
    std::map<std::string, double> scalarProperties;

    // =======================================================================
    // Matrix properties (constants)  
    // =======================================================================
    std::map<std::string, Matrix3> matrixProperties;

    // =======================================================================
    // Scalar expressions (evaluated at runtime with variables)
    // =======================================================================
    std::map<std::string, std::string> scalarExpressions;

    // =======================================================================
    // Matrix expressions (evaluated at runtime with variables)
    // =======================================================================
    std::map<std::string, std::string> matrixExpressions;

    // =======================================================================
    // Generic property access
    // =======================================================================
    
    std::optional<double> getScalarProperty(const std::string& name) const {
        auto it = scalarProperties.find(name);
        return it != scalarProperties.end() ? std::optional<double>{it->second} : std::nullopt;
    }

    void setScalarProperty(const std::string& name, double value) {
        scalarProperties[name] = value;
    }
    
    std::optional<Matrix3> getMatrixProperty(const std::string& name) const {
        auto it = matrixProperties.find(name);
        return it != matrixProperties.end() ? std::optional<Matrix3>{it->second} : std::nullopt;
    }
    
    void setMatrixProperty(const std::string& name, const Matrix3& mat) {
        matrixProperties[name] = mat;
    }

    // =======================================================================
    // Expression access
    // =======================================================================
    
    bool hasScalarExpression(const std::string& name) const {
        return scalarExpressions.find(name) != scalarExpressions.end();
    }

    bool hasMatrixExpression(const std::string& name) const {
        return matrixExpressions.find(name) != matrixExpressions.end();
    }

    void setScalarExpression(const std::string& name, const std::string& expr) {
        scalarExpressions[name] = expr;
    }

    void setMatrixExpression(const std::string& name, const std::string& expr) {
        matrixExpressions[name] = expr;
    }

    // =======================================================================
    // Property evaluation with variables
    // Returns evaluated value for the given variables map
    // =======================================================================
    
    /// Evaluate a scalar property (constant or expression)
    /// Returns nullopt if property not found
    std::optional<double> evaluateScalar(const std::string& name,
                                        const std::map<std::string, double>& variables = {}) const {
        // Check expressions first
        auto exprIt = scalarExpressions.find(name);
        if (exprIt != scalarExpressions.end()) {
            double result = ExpressionParser::instance().evaluateScalar(exprIt->second, variables);
            return result;
        }
        // Fall back to constant
        return getScalarProperty(name);
    }

    /// Evaluate a matrix property (constant or expression)
    /// Returns nullopt if property not found
    std::optional<Matrix3> evaluateMatrix(const std::string& name,
                                        const std::map<std::string, double>& variables = {}) const {
        // Check expressions first
        auto exprIt = matrixExpressions.find(name);
        if (exprIt != matrixExpressions.end()) {
            Matrix3 result = ExpressionParser::instance().evaluateMatrix(exprIt->second, variables);
            return result;
        }
        // Fall back to constant
        return getMatrixProperty(name);
    }

    /// Check if a property exists (either constant or expression)
    bool hasScalar(const std::string& name) const {
        return hasScalarExpression(name) || scalarProperties.count(name) > 0;
    }

    bool hasMatrix(const std::string& name) const {
        return hasMatrixExpression(name) || matrixProperties.count(name) > 0;
    }

    // =======================================================================
    // Factory methods for creating coefficients
    // The getVars callback takes ElementTransform& and returns variables map
    // =======================================================================
    
    /// Create a scalar coefficient from expression or constant property
    /// Expression takes precedence if both exist
    template<typename Func>
    std::unique_ptr<Coefficient> createScalarCoefficient(const std::string& name, Func&& getVars) const {
        if (hasScalarExpression(name)) {
            return std::make_unique<ScalarCoefficient>(
                [this, name, getVars](ElementTransform& trans, Real& result, Real) {
                    auto vars = getVars(trans);
                    result = evaluateScalar(name, vars).value_or(0.0);
                });
        }
        auto value = getScalarProperty(name);
        if (value.has_value()) {
            return constantCoefficient(value.value());
        }
        return nullptr;
    }

    /// Create a matrix coefficient from expression or constant property
    /// Expression takes precedence if both exist
    template<typename Func>
    std::unique_ptr<MatrixCoefficient> createMatrixCoefficient(const std::string& name, Func&& getVars) const {
        if (hasMatrixExpression(name)) {
            return std::make_unique<MatrixFunctionCoefficient>(
                [this, name, getVars](ElementTransform& trans, Matrix3& result, Real) {
                    auto vars = getVars(trans);
                    result = evaluateMatrix(name, vars).value_or(Matrix3::Identity());
                });
        }
        auto value = getMatrixProperty(name);
        if (value.has_value()) {
            return constantMatrixCoefficient(value.value());
        }
        return nullptr;
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
