#ifndef MPFEM_MATERIAL_DATABASE_HPP
#define MPFEM_MATERIAL_DATABASE_HPP

#include "core/types.hpp"
#include "core/exception.hpp"
#include "io/exprtk_expression_parser.hpp"
#include <string>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>

namespace mpfem {

/**
 * @brief Material property model with unified storage for all properties as expressions.
 * 
 * All properties stored as expression strings. Constants like "0.35" are just
 * expressions with no variables. Unit handling is automatic via ExpressionParser.
 */
class MaterialPropertyModel {
public:
    MaterialPropertyModel() = default;

    const std::string& tag() const { return tag_; }
    const std::string& label() const { return label_; }
    void setTag(std::string tag) { tag_ = std::move(tag); }
    void setLabel(std::string label) { label_ = std::move(label); }

    // Get scalar value - evaluates expression with optional variables
    double getScalar(const std::string& name,
                    const std::map<std::string, double>& variables = {}) const {
        return ExpressionParser::instance().evaluate(scalarExpression(name), variables);
    }

    // Get matrix value - evaluates expression with optional variables
    Matrix3 getMatrix(const std::string& name,
                    const std::map<std::string, double>& variables = {}) const {
        return ExpressionParser::instance().evaluateMatrix(matrixExpression(name), variables);
    }

    const std::string& scalarExpression(const std::string& name) const {
        auto it = scalarExpressions_.find(name);
        if (it != scalarExpressions_.end()) {
            return it->second;
        }
        MPFEM_THROW(ArgumentException, "Scalar property '" + name + "' not found");
    }

    const std::string& matrixExpression(const std::string& name) const {
        auto it = matrixExpressions_.find(name);
        if (it != matrixExpressions_.end()) {
            return it->second;
        }
        MPFEM_THROW(ArgumentException, "Matrix property '" + name + "' not found");
    }

    bool hasScalar(const std::string& name) const {
        return scalarExpressions_.count(name) > 0;
    }

    bool hasMatrix(const std::string& name) const {
        return matrixExpressions_.count(name) > 0;
    }

    void setScalar(const std::string& name, const std::string& expr) {
        scalarExpressions_[name] = expr;
    }

    void setMatrix(const std::string& name, const std::string& expr) {
        matrixExpressions_[name] = expr;
    }

private:
    std::string tag_;
    std::string label_;
    std::unordered_map<std::string, std::string> scalarExpressions_;
    std::unordered_map<std::string, std::string> matrixExpressions_;
};

/**
 * @brief Database of material properties.
 */
class MaterialDatabase {
public:
    void addMaterial(const MaterialPropertyModel& material) {
        materials_[material.tag()] = material;
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
