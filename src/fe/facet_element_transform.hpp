#ifndef MPFEM_FACET_ELEMENT_TRANSFORM_HPP
#define MPFEM_FACET_ELEMENT_TRANSFORM_HPP

#include "fe/element_transform.hpp"

namespace mpfem {

    /**
     * @brief Transformation for boundary facet elements.
     * 
     * Extends ElementTransform with boundary-specific logic (normals, face mapping).
     */
    class FacetElementTransform : public ElementTransform {
    public:
        FacetElementTransform() = default;

        /**
         * @brief Get the unit normal vector at the current integration point.
         * Triggers lazy evaluation of Jacobian.
         */
        Vector3 normal();

        // Boundary-specific topology info
        void setFaceInfo(Index adjElemIdx, int localFaceIdx)
        {
            adjElemIdx_ = adjElemIdx;
            localFaceIdx_ = localFaceIdx;
        }

        Index adjacentElementIndex() const { return adjElemIdx_; }
        int localFaceIndex() const { return localFaceIdx_; }

    private:
        Index adjElemIdx_ = InvalidIndex;
        int localFaceIdx_ = -1;
    };

} // namespace mpfem

#endif // MPFEM_FACET_ELEMENT_TRANSFORM_HPP
