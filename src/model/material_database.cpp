#include "model/material_database.hpp"

#include "core/exception.hpp"
#include "io/exprtk_expression_parser.hpp"

#include <unordered_map>
#include <utility>

namespace mpfem {

struct MaterialPropertyModel::Impl {
    std::string tag;
    std::string label;
    std::unordered_map<std::string, std::string> scalarExpressions;
    std::unordered_map<std::string, std::string> matrixExpressions;
};

MaterialPropertyModel::MaterialPropertyModel()
    : impl_(std::make_unique<Impl>()) {}

MaterialPropertyModel::~MaterialPropertyModel() = default;

MaterialPropertyModel::MaterialPropertyModel(const MaterialPropertyModel& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

MaterialPropertyModel::MaterialPropertyModel(MaterialPropertyModel&& other) noexcept = default;

MaterialPropertyModel& MaterialPropertyModel::operator=(const MaterialPropertyModel& other) {
    if (this != &other) {
        *impl_ = *other.impl_;
    }
    return *this;
}

MaterialPropertyModel& MaterialPropertyModel::operator=(MaterialPropertyModel&& other) noexcept = default;

const std::string& MaterialPropertyModel::tag() const {
    return impl_->tag;
}

const std::string& MaterialPropertyModel::label() const {
    return impl_->label;
}

void MaterialPropertyModel::setTag(std::string tag) {
    impl_->tag = std::move(tag);
}

void MaterialPropertyModel::setLabel(std::string label) {
    impl_->label = std::move(label);
}

double MaterialPropertyModel::getScalar(
    const std::string& name,
    const std::map<std::string, double>& variables) const {
    return ExpressionParser::instance().evaluate(scalarExpression(name), variables);
}

Matrix3 MaterialPropertyModel::getMatrix(
    const std::string& name,
    const std::map<std::string, double>& variables) const {
    return ExpressionParser::instance().evaluateMatrix(matrixExpression(name), variables);
}

const std::string& MaterialPropertyModel::scalarExpression(const std::string& name) const {
    auto it = impl_->scalarExpressions.find(name);
    if (it != impl_->scalarExpressions.end()) {
        return it->second;
    }
    MPFEM_THROW(ArgumentException, "Scalar property '" + name + "' not found");
}

const std::string& MaterialPropertyModel::matrixExpression(const std::string& name) const {
    auto it = impl_->matrixExpressions.find(name);
    if (it != impl_->matrixExpressions.end()) {
        return it->second;
    }
    MPFEM_THROW(ArgumentException, "Matrix property '" + name + "' not found");
}

bool MaterialPropertyModel::hasScalar(const std::string& name) const {
    return impl_->scalarExpressions.find(name) != impl_->scalarExpressions.end();
}

bool MaterialPropertyModel::hasMatrix(const std::string& name) const {
    return impl_->matrixExpressions.find(name) != impl_->matrixExpressions.end();
}

void MaterialPropertyModel::setScalar(const std::string& name, const std::string& expr) {
    impl_->scalarExpressions[name] = expr;
}

void MaterialPropertyModel::setMatrix(const std::string& name, const std::string& expr) {
    impl_->matrixExpressions[name] = expr;
}

struct MaterialDatabase::Impl {
    std::unordered_map<std::string, MaterialPropertyModel> materials;
};

MaterialDatabase::MaterialDatabase()
    : impl_(std::make_unique<Impl>()) {}

MaterialDatabase::~MaterialDatabase() = default;

MaterialDatabase::MaterialDatabase(const MaterialDatabase& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

MaterialDatabase::MaterialDatabase(MaterialDatabase&& other) noexcept = default;

MaterialDatabase& MaterialDatabase::operator=(const MaterialDatabase& other) {
    if (this != &other) {
        *impl_ = *other.impl_;
    }
    return *this;
}

MaterialDatabase& MaterialDatabase::operator=(MaterialDatabase&& other) noexcept = default;

void MaterialDatabase::addMaterial(const MaterialPropertyModel& material) {
    impl_->materials[material.tag()] = material;
}

const MaterialPropertyModel* MaterialDatabase::getMaterial(const std::string& tag) const {
    auto it = impl_->materials.find(tag);
    return it != impl_->materials.end() ? &it->second : nullptr;
}

bool MaterialDatabase::hasMaterial(const std::string& tag) const {
    return impl_->materials.find(tag) != impl_->materials.end();
}

std::vector<std::string> MaterialDatabase::getMaterialTags() const {
    std::vector<std::string> tags;
    tags.reserve(impl_->materials.size());
    for (const auto& [tag, _] : impl_->materials) {
        tags.push_back(tag);
    }
    return tags;
}

size_t MaterialDatabase::size() const {
    return impl_->materials.size();
}

void MaterialDatabase::clear() {
    impl_->materials.clear();
}

} // namespace mpfem
