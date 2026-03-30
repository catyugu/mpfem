#ifndef MPFEM_MATERIAL_DATABASE_HPP
#define MPFEM_MATERIAL_DATABASE_HPP

#include "core/types.hpp"
#include "core/exception.hpp"
#include "io/exprtk_expression_parser.hpp"
#include "fe/element_transform.hpp"
#include "fe/coefficient.hpp"
#include <string>
#include <map>
#include <memory>

namespace mpfem {

/**
 * @brief Material property model with unified storage for all properties as expressions.
 * 
 * All properties stored as expression strings. Constants like "0.35" are just
 * expressions with no variables. Unit handling is automatic via ExpressionParser.
 */
struct MaterialPropertyModel {
    std::string tag;
    std::string label;

    // Unified storage - ALL properties as expression strings
    std::unordered_map<std::string, std::string> scalarExpressions_;
    std::unordered_map<std::string, std::string> matrixExpressions_;

    // Get scalar value - evaluates expression with optional variables
    double getScalar(const std::string& name,
                    const std::map<std::string, double>& variables = {}) const {
        auto it = scalarExpressions_.find(name);
        if (it != scalarExpressions_.end()) {
            return ExpressionParser::instance().evaluate(it->second, variables);
        }
        MPFEM_THROW(ArgumentException, "Scalar property '" + name + "' not found");
    }

    // Get matrix value - evaluates expression with optional variables
    Matrix3 getMatrix(const std::string& name,
                    const std::map<std::string, double>& variables = {}) const {
        auto it = matrixExpressions_.find(name);
        if (it != matrixExpressions_.end()) {
            return ExpressionParser::instance().evaluateMatrix(it->second, variables);
        }
        MPFEM_THROW(ArgumentException, "Matrix property '" + name + "' not found");
    }

    bool hasScalar(const std::string& name) const {
        return scalarExpressions_.count(name) > 0;
    }

    bool hasMatrix(const std::string& name) const {
        return matrixExpressions_.count(name) > 0;
    }

    // Factory methods for creating coefficients
    template<typename Func>
    std::unique_ptr<Coefficient> createScalarCoefficient(const std::string& name, Func&& getVars) const {
        if (!hasScalar(name)) return nullptr;
        return std::make_unique<ScalarCoefficient>(
            [this, name, getVars](ElementTransform& trans, Real& result, Real) {
                result = this->getScalar(name, getVars(trans));
            });
    }

    template<typename Func>
    std::unique_ptr<MatrixCoefficient> createMatrixCoefficient(const std::string& name, Func&& getVars) const {
        if (!hasMatrix(name)) return nullptr;
        return std::make_unique<MatrixFunctionCoefficient>(
            [this, name, getVars](ElementTransform& trans, Matrix3& result, Real) {
                result = this->getMatrix(name, getVars(trans));
            });
    }

    void setScalar(const std::string& name, const std::string& expr) {
        scalarExpressions_[name] = expr;
    }

    void setMatrix(const std::string& name, const std::string& expr) {
        matrixExpressions_[name] = expr;
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
    std::unordered_map<std::string, MaterialPropertyModel> materials_;
};

} // namespace mpfem

#endif
