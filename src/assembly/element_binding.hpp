#ifndef MPFEM_ELEMENT_BINDING_HPP
#define MPFEM_ELEMENT_BINDING_HPP

#include "fe/element_transform.hpp"
#include "mesh/mesh.hpp"

namespace mpfem {

    /**
     * @brief Bind an element from a mesh to an ElementTransform.
     * @param trans ElementTransform to bind
     * @param mesh Source mesh
     * @param elemIdx Element index in mesh
     * @param isBoundary If true, treat as boundary element
     *
     * This function lives in the assembly layer because it requires Mesh access.
     * ElementTransform itself is decoupled from Mesh - this helper bridges them.
     */
    inline void bindElementToTransform(ElementTransform& trans, const Mesh& mesh, Index elemIdx, bool isBoundary = false)
    {
        const Element elem = isBoundary ? mesh.bdrElement(elemIdx) : mesh.element(elemIdx);
        std::array<Vector3, MaxNodesPerElement> nodeCoords;
        for (size_t i = 0; i < elem.nodes.size(); ++i) {
            Index nodeIdx = elem.nodes[i];
            nodeCoords[i] = Vector3 {mesh.nodeX(nodeIdx), mesh.nodeY(nodeIdx), mesh.nodeZ(nodeIdx)};
        }
        trans.bindElement(elem.geometry, elem.order, elem.attribute, elemIdx,
            std::span<const Vector3>(nodeCoords.data(), elem.nodes.size()));
    }

} // namespace mpfem

#endif // MPFEM_ELEMENT_BINDING_HPP
