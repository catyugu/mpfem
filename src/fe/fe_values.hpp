#ifndef MPFEM_FE_VALUES_HPP
#define MPFEM_FE_VALUES_HPP

#include "fe/grid_function.hpp"
#include "model/field_kind.hpp"
#include "core/types.hpp"
#include <map>

namespace mpfem {

class ElementTransform;

/**
 * @brief Minimal multi-field state manager.
 * Non-owning references to GridFunctions.
 */
class FEValues {
public:
    void registerField(FieldKind kind, GridFunction* gf) {
        fields_[kind] = gf;
    }

    GridFunction* field(FieldKind kind);
    const GridFunction* field(FieldKind kind) const;

    Real getValue(FieldKind kind, Index elem, const Real* xi) const;
    Vector3 getGradient(FieldKind kind, Index elem, 
                        const Real* xi, ElementTransform& trans) const;

    void clear() { fields_.clear(); }
    size_t numFields() const { return fields_.size(); }

private:
    std::map<FieldKind, GridFunction*> fields_;
};

}  // namespace mpfem

#endif  // MPFEM_FE_VALUES_HPP