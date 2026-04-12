#ifndef MPFEM_MATERIAL_DATABASE_HPP
#define MPFEM_MATERIAL_DATABASE_HPP

#include "core/exception.hpp"
#include "core/types.hpp"
#include "model/case_definition.hpp"
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mpfem {

    /**
     * @brief Material property model with unified storage for all properties as expressions.
     *
     * All properties stored as expression strings. Constants like "0.35" are just
     * expressions with no variables. Unit handling is automatic via ExpressionParser.
     */
    class MaterialPropertyModel {
    public:
        MaterialPropertyModel();
        ~MaterialPropertyModel();
        MaterialPropertyModel(const MaterialPropertyModel& other);
        MaterialPropertyModel(MaterialPropertyModel&& other) noexcept;
        MaterialPropertyModel& operator=(const MaterialPropertyModel& other);
        MaterialPropertyModel& operator=(MaterialPropertyModel&& other) noexcept;

        const std::string& tag() const;
        const std::string& label() const;
        void setTag(std::string tag);
        void setLabel(std::string label);

        const std::string& scalarExpression(const std::string& name) const;
        const std::string& matrixExpression(const std::string& name) const;

        bool hasScalar(const std::string& name) const;
        bool hasMatrix3(const std::string& name) const;

        void setScalar(const std::string& name, const std::string& expr);
        void setMatrix(const std::string& name, const std::string& expr);

    private:
        struct Impl {
            std::string tag;
            std::string label;
            std::map<std::string, std::string> scalarExpressions;
            std::map<std::string, std::string> matrixExpressions;
        };
        std::unique_ptr<Impl> impl_;
    };

    /**
     * @brief Database of material properties.
     */
    class MaterialDatabase {
    public:
        MaterialDatabase();
        ~MaterialDatabase();
        MaterialDatabase(const MaterialDatabase& other);
        MaterialDatabase(MaterialDatabase&& other) noexcept;
        MaterialDatabase& operator=(const MaterialDatabase& other);
        MaterialDatabase& operator=(MaterialDatabase&& other) noexcept;

        void addMaterial(const MaterialPropertyModel& material);

        void buildDomainIndex(const std::vector<MaterialAssignment>& assignments);
        const std::vector<int>& domainIds() const;
        const std::string& scalarExpressionByDomain(int domainId, std::string_view property) const;
        const std::string& matrixExpressionByDomain(int domainId, std::string_view property) const;

        size_t size() const;
        void clear();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace mpfem

#endif
