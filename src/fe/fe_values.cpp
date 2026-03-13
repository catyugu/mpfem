#include "fe/fe_values.hpp"
#include "fe/element_transform.hpp"
#include "core/exception.hpp"

namespace mpfem {

GridFunction* FEValues::field(FieldKind kind) {
    auto it = fields_.find(kind);
    return it != fields_.end() ? it->second : nullptr;
}

const GridFunction* FEValues::field(FieldKind kind) const {
    auto it = fields_.find(kind);
    return it != fields_.end() ? it->second : nullptr;
}

Real FEValues::getValue(FieldKind kind, Index elem, const Real* xi) const {
    const GridFunction* gf = field(kind);
    if (!gf) throw FeException("Field not registered");
    return gf->eval(elem, xi);
}

Vector3 FEValues::getGradient(FieldKind kind, Index elem, 
                               const Real* xi, 
                               ElementTransform& trans) const {
    const GridFunction* gf = field(kind);
    if (!gf) throw FeException("Field not registered");
    return gf->gradient(elem, xi, trans);
}

}  // namespace mpfem
