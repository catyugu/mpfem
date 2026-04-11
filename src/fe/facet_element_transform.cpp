#include "fe/facet_element_transform.hpp"
#include "mesh/mesh.hpp"
#include <cmath>

namespace mpfem {

    void FacetElementTransform::setMesh(const Mesh* mesh)
    {
        ElementTransform::setMesh(mesh);
        computeAdjacentElementInfo();
    }

    void FacetElementTransform::setElement(Index bdrElemIdx)
    {
        elemIdx_ = bdrElemIdx;
        elemType_ = BOUNDARY;
        computeGeometryInfo();
        computeAdjacentElementInfo();
    }

    bool FacetElementTransform::hasTopology() const
    {
        return mesh_ && mesh_->hasTopology();
    }

    void FacetElementTransform::computeAdjacentElementInfo()
    {
        adjElemIdx_ = InvalidIndex;
        localFaceIdx_ = -1;

        if (!mesh_ || !mesh_->hasTopology())
            return;

        Index faceIdx = mesh_->getBoundaryFaceIndex(elemIdx_);
        if (faceIdx == InvalidIndex)
            return;

        const auto& faceInfo = mesh_->getFaceInfo(faceIdx);
        adjElemIdx_ = faceInfo.elem1;
        localFaceIdx_ = faceInfo.localFace1;
    }

    bool FacetElementTransform::getAdjacentElementTransform(ElementTransform& trans) const
    {
        if (adjElemIdx_ == InvalidIndex)
            return false;
        trans.setMesh(mesh_);
        trans.setElement(adjElemIdx_);
        return true;
    }

    Vector3 FacetElementTransform::normal() const
    {
        Vector3 n(0.0, 0.0, 0.0);

        if (dim_ == 2 && spaceDim_ == 3) {
            Vector3 t1(jacobian_(0, 0), jacobian_(1, 0), jacobian_(2, 0));
            Vector3 t2(jacobian_(0, 1), jacobian_(1, 1), jacobian_(2, 1));
            n = t1.cross(t2);
            n.normalize();
        }
        else if (dim_ == 1 && spaceDim_ == 2) {
            Vector3 t(jacobian_(0, 0), jacobian_(1, 0), 0.0);
            n = Vector3(-t.y(), t.x(), 0.0);
            n.normalize();
        }
        else if (dim_ == 1 && spaceDim_ == 3) {
            Vector3 t(jacobian_(0, 0), jacobian_(1, 0), jacobian_(2, 0));
            Vector3 ref(0, 0, 1);
            if (std::abs(t.dot(ref)) > 0.9) {
                ref = Vector3(1, 0, 0);
            }
            n = t.cross(ref);
            n.normalize();
        }

        return n;
    }

    bool FacetElementTransform::mapToVolumeElement(const Vector3& bdrXi, Vector3& volXi) const
    {
        if (adjElemIdx_ == InvalidIndex || localFaceIdx_ < 0) {
            return false;
        }

        const Element& volElem = mesh_->element(adjElemIdx_);
        Geometry volGeom = volElem.geometry();

        if (volGeom == Geometry::Tetrahedron) {
            const Real xi = bdrXi.x();
            const Real eta = bdrXi.y();
            Real new_xi, new_eta, new_zeta;

            switch (localFaceIdx_) {
            case 0:
                new_xi = xi;
                new_eta = eta;
                new_zeta = 1.0 - xi - eta;
                break;
            case 1:
                new_xi = 0.0;
                new_eta = eta;
                new_zeta = 1.0 - xi - eta;
                break;
            case 2:
                new_xi = xi;
                new_eta = 0.0;
                new_zeta = 1.0 - xi - eta;
                break;
            case 3:
                new_xi = xi;
                new_eta = eta;
                new_zeta = 0.0;
                break;
            default:
                return false;
            }
            volXi = Vector3(new_xi, new_eta, new_zeta);
            return true;
        }

        if (volGeom == Geometry::Cube) {
            const Real xi = bdrXi.x();
            const Real eta = bdrXi.y();
            Real new_xi, new_eta, new_zeta;

            switch (localFaceIdx_) {
            case 0:
                new_xi = xi;
                new_eta = eta;
                new_zeta = -1.0;
                break;
            case 1:
                new_xi = xi;
                new_eta = eta;
                new_zeta = 1.0;
                break;
            case 2:
                new_xi = xi;
                new_eta = -1.0;
                new_zeta = eta;
                break;
            case 3:
                new_xi = xi;
                new_eta = 1.0;
                new_zeta = eta;
                break;
            case 4:
                new_xi = -1.0;
                new_eta = xi;
                new_zeta = eta;
                break;
            case 5:
                new_xi = 1.0;
                new_eta = xi;
                new_zeta = eta;
                break;
            default:
                return false;
            }
            volXi = Vector3(new_xi, new_eta, new_zeta);
            return true;
        }

        if (volGeom == Geometry::Triangle) {
            const Real xi = bdrXi.x();
            Real new_xi, new_eta;

            switch (localFaceIdx_) {
            case 0:
                new_xi = xi;
                new_eta = 1.0 - xi;
                break;
            case 1:
                new_xi = 0.0;
                new_eta = xi;
                break;
            case 2:
                new_xi = xi;
                new_eta = 0.0;
                break;
            default:
                return false;
            }
            volXi = Vector3(new_xi, new_eta, 0.0);
            return true;
        }

        if (volGeom == Geometry::Square) {
            const Real xi = bdrXi.x();
            Real new_xi, new_eta;

            switch (localFaceIdx_) {
            case 0:
                new_xi = xi;
                new_eta = -1.0;
                break;
            case 1:
                new_xi = 1.0;
                new_eta = xi;
                break;
            case 2:
                new_xi = xi;
                new_eta = 1.0;
                break;
            case 3:
                new_xi = -1.0;
                new_eta = xi;
                break;
            default:
                return false;
            }
            volXi = Vector3(new_xi, new_eta, 0.0);
            return true;
        }

        return false;
    }

} // namespace mpfem