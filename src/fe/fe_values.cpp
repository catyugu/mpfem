#include "fe/fe_values.hpp"
#include "fe/element_transform.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"

namespace mpfem {

void FEValues::registerField(FieldKind kind, GridFunction* gf) {
    if (!gf) {
        LOG_WARN("Attempting to register null GridFunction for field kind " 
                 << static_cast<int>(kind));
        return;
    }
    fields_[kind] = gf;
}

void FEValues::registerField(const std::string& name, GridFunction* gf) {
    if (!gf) {
        LOG_WARN("Attempting to register null GridFunction for field " << name);
        return;
    }
    fieldsByName_[name] = gf;
}

bool FEValues::hasField(FieldKind kind) const {
    return fields_.find(kind) != fields_.end();
}

bool FEValues::hasField(const std::string& name) const {
    return fieldsByName_.find(name) != fieldsByName_.end();
}

GridFunction* FEValues::field(FieldKind kind) {
    auto it = fields_.find(kind);
    return it != fields_.end() ? it->second : nullptr;
}

const GridFunction* FEValues::field(FieldKind kind) const {
    auto it = fields_.find(kind);
    return it != fields_.end() ? it->second : nullptr;
}

GridFunction* FEValues::field(const std::string& name) {
    auto it = fieldsByName_.find(name);
    return it != fieldsByName_.end() ? it->second : nullptr;
}

const GridFunction* FEValues::field(const std::string& name) const {
    auto it = fieldsByName_.find(name);
    return it != fieldsByName_.end() ? it->second : nullptr;
}

Real FEValues::getValue(FieldKind kind, Index elemIdx, const IntegrationPoint& ip) const {
    const GridFunction* gf = field(kind);
    if (!gf) {
        throw FeException("Field not registered: " + std::to_string(static_cast<int>(kind)));
    }
    return gf->eval(elemIdx, ip);
}

Real FEValues::getValue(FieldKind kind, Index elemIdx, const Real* xi) const {
    const GridFunction* gf = field(kind);
    if (!gf) {
        throw FeException("Field not registered: " + std::to_string(static_cast<int>(kind)));
    }
    return gf->eval(elemIdx, xi);
}

Vector3 FEValues::getVectorValue(FieldKind kind, Index elemIdx, const IntegrationPoint& ip) const {
    const GridFunction* gf = field(kind);
    if (!gf) {
        throw FeException("Field not registered: " + std::to_string(static_cast<int>(kind)));
    }
    return gf->evalVector(elemIdx, &ip.xi);
}

Vector3 FEValues::getGradient(FieldKind kind, Index elemIdx, 
                              const IntegrationPoint& ip, 
                              const ElementTransform& trans) const {
    const GridFunction* gf = field(kind);
    if (!gf) {
        throw FeException("Field not registered: " + std::to_string(static_cast<int>(kind)));
    }
    return gf->gradient(elemIdx, ip, trans);
}

Vector3 FEValues::getGradient(FieldKind kind, Index elemIdx, 
                              const Real* xi, 
                              const ElementTransform& trans) const {
    const GridFunction* gf = field(kind);
    if (!gf) {
        throw FeException("Field not registered: " + std::to_string(static_cast<int>(kind)));
    }
    return gf->gradient(elemIdx, xi, trans);
}

}  // namespace mpfem
