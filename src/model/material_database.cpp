#include "model/material_database.hpp"

#include "core/exception.hpp"
#include "expr/expression_parser.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace mpfem {

    namespace {

        const MaterialPropertyModel& requireMaterialByDomain(
            const std::unordered_map<int, std::string>& materialTagByDomain,
            const std::unordered_map<std::string, MaterialPropertyModel>& materials,
            int domainId)
        {
            auto it = materialTagByDomain.find(domainId);
            if (it == materialTagByDomain.end()) {
                MPFEM_THROW(ArgumentException, "Material domain not found: " + std::to_string(domainId));
            }

            auto materialIt = materials.find(it->second);
            if (materialIt == materials.end()) {
                MPFEM_THROW(ArgumentException,
                    "Material tag not found for domain " + std::to_string(domainId) + ": " + it->second);
            }
            return materialIt->second;
        }

        std::vector<Real> buildInputValues(const std::vector<std::string>& dependencies,
            const std::map<std::string, Real>& variables)
        {
            std::vector<Real> inputs;
            inputs.reserve(dependencies.size());

            for (const std::string& symbol : dependencies) {
                const auto it = variables.find(symbol);
                if (it == variables.end()) {
                    MPFEM_THROW(ArgumentException, "Unbound variable in material expression: " + symbol);
                }
                inputs.push_back(it->second);
            }

            return inputs;
        }

    } // namespace

    struct MaterialPropertyModel::Impl {
        std::string tag;
        std::string label;
        std::unordered_map<std::string, std::string> scalarExpressions;
        std::unordered_map<std::string, std::string> matrixExpressions;
    };

    MaterialPropertyModel::MaterialPropertyModel()
        : impl_(std::make_unique<Impl>()) { }

    MaterialPropertyModel::~MaterialPropertyModel() = default;

    MaterialPropertyModel::MaterialPropertyModel(const MaterialPropertyModel& other)
        : impl_(std::make_unique<Impl>(*other.impl_)) { }

    MaterialPropertyModel::MaterialPropertyModel(MaterialPropertyModel&& other) noexcept = default;

    MaterialPropertyModel& MaterialPropertyModel::operator=(const MaterialPropertyModel& other)
    {
        if (this != &other) {
            *impl_ = *other.impl_;
        }
        return *this;
    }

    MaterialPropertyModel& MaterialPropertyModel::operator=(MaterialPropertyModel&& other) noexcept = default;

    const std::string& MaterialPropertyModel::tag() const
    {
        return impl_->tag;
    }

    const std::string& MaterialPropertyModel::label() const
    {
        return impl_->label;
    }

    void MaterialPropertyModel::setTag(std::string tag)
    {
        impl_->tag = std::move(tag);
    }

    void MaterialPropertyModel::setLabel(std::string label)
    {
        impl_->label = std::move(label);
    }

    Real MaterialPropertyModel::getScalar(
        const std::string& name,
        const std::map<std::string, Real>& variables) const
    {
        ExpressionParser parser;
        ExpressionParser::ExpressionProgram program = parser.compile(scalarExpression(name));
        const std::vector<Real> inputs = buildInputValues(program.dependencies(), variables);
        return program.evaluate(std::span<const Real>(inputs.data(), inputs.size())).scalar();
    }

    Matrix3 MaterialPropertyModel::getMatrix(
        const std::string& name,
        const std::map<std::string, Real>& variables) const
    {
        ExpressionParser parser;
        ExpressionParser::ExpressionProgram program = parser.compile(matrixExpression(name));
        const std::vector<Real> inputs = buildInputValues(program.dependencies(), variables);
        return program.evaluate(std::span<const Real>(inputs.data(), inputs.size())).toMatrix3();
    }

    const std::string& MaterialPropertyModel::scalarExpression(const std::string& name) const
    {
        auto it = impl_->scalarExpressions.find(name);
        if (it != impl_->scalarExpressions.end()) {
            return it->second;
        }
        MPFEM_THROW(ArgumentException, "Scalar property '" + name + "' not found");
    }

    const std::string& MaterialPropertyModel::matrixExpression(const std::string& name) const
    {
        auto it = impl_->matrixExpressions.find(name);
        if (it != impl_->matrixExpressions.end()) {
            return it->second;
        }
        MPFEM_THROW(ArgumentException, "Matrix property '" + name + "' not found");
    }

    bool MaterialPropertyModel::hasScalar(const std::string& name) const
    {
        return impl_->scalarExpressions.find(name) != impl_->scalarExpressions.end();
    }

    bool MaterialPropertyModel::hasMatrix(const std::string& name) const
    {
        return impl_->matrixExpressions.find(name) != impl_->matrixExpressions.end();
    }

    void MaterialPropertyModel::setScalar(const std::string& name, const std::string& expr)
    {
        impl_->scalarExpressions[name] = expr;
    }

    void MaterialPropertyModel::setMatrix(const std::string& name, const std::string& expr)
    {
        impl_->matrixExpressions[name] = expr;
    }

    struct MaterialDatabase::Impl {
        std::unordered_map<std::string, MaterialPropertyModel> materials;
        std::unordered_map<int, std::string> materialTagByDomain;
        std::vector<int> orderedDomainIds;
    };

    MaterialDatabase::MaterialDatabase()
        : impl_(std::make_unique<Impl>()) { }

    MaterialDatabase::~MaterialDatabase() = default;

    MaterialDatabase::MaterialDatabase(const MaterialDatabase& other)
        : impl_(std::make_unique<Impl>(*other.impl_)) { }

    MaterialDatabase::MaterialDatabase(MaterialDatabase&& other) noexcept = default;

    MaterialDatabase& MaterialDatabase::operator=(const MaterialDatabase& other)
    {
        if (this != &other) {
            *impl_ = *other.impl_;
        }
        return *this;
    }

    MaterialDatabase& MaterialDatabase::operator=(MaterialDatabase&& other) noexcept = default;

    void MaterialDatabase::addMaterial(const MaterialPropertyModel& material)
    {
        impl_->materials[material.tag()] = material;
    }

    void MaterialDatabase::buildDomainIndex(const std::vector<MaterialAssignment>& assignments)
    {
        impl_->materialTagByDomain.clear();
        impl_->orderedDomainIds.clear();
        for (const auto& assignment : assignments) {
            for (int domainId : assignment.domainIds) {
                impl_->materialTagByDomain[domainId] = assignment.materialTag;
            }
        }

        impl_->orderedDomainIds.reserve(impl_->materialTagByDomain.size());
        for (const auto& [domainId, _] : impl_->materialTagByDomain) {
            impl_->orderedDomainIds.push_back(domainId);
        }
        std::sort(impl_->orderedDomainIds.begin(), impl_->orderedDomainIds.end());
    }

    const std::vector<int>& MaterialDatabase::domainIds() const
    {
        return impl_->orderedDomainIds;
    }

    const std::string& MaterialDatabase::scalarExpressionByDomain(
        int domainId,
        std::string_view property) const
    {
        return requireMaterialByDomain(
            impl_->materialTagByDomain,
            impl_->materials,
            domainId)
            .scalarExpression(std::string(property));
    }

    const std::string& MaterialDatabase::matrixExpressionByDomain(
        int domainId,
        std::string_view property) const
    {
        return requireMaterialByDomain(
            impl_->materialTagByDomain,
            impl_->materials,
            domainId)
            .matrixExpression(std::string(property));
    }

    size_t MaterialDatabase::size() const
    {
        return impl_->materials.size();
    }

    void MaterialDatabase::clear()
    {
        impl_->materials.clear();
        impl_->materialTagByDomain.clear();
        impl_->orderedDomainIds.clear();
    }

} // namespace mpfem
