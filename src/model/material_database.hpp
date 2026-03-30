#ifndef MPFEM_MATERIAL_DATABASE_HPP
#define MPFEM_MATERIAL_DATABASE_HPP

#include "core/types.hpp"
#include "core/exception.hpp"
#include <map>
#include <memory>
#include <string>
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

    double getScalar(const std::string& name,
                    const std::map<std::string, double>& variables = {}) const;
    Matrix3 getMatrix(const std::string& name,
                    const std::map<std::string, double>& variables = {}) const;

    const std::string& scalarExpression(const std::string& name) const;
    const std::string& matrixExpression(const std::string& name) const;

    bool hasScalar(const std::string& name) const;
    bool hasMatrix(const std::string& name) const;

    void setScalar(const std::string& name, const std::string& expr);
    void setMatrix(const std::string& name, const std::string& expr);

private:
    struct Impl;
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
    const MaterialPropertyModel* getMaterial(const std::string& tag) const;
    bool hasMaterial(const std::string& tag) const;
    std::vector<std::string> getMaterialTags() const;
    size_t size() const;
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mpfem

#endif
