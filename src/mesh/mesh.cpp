#include "mesh/mesh.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace mpfem {

Mesh::Mesh(int dim, Index numVertices, Index numElements, Index numBdrElements)
    : dim_(dim)
    , vertices_(numVertices)
    , elements_(numElements)
    , bdrElements_(numBdrElements) {
}

void Mesh::setDim(int dim) {
    dim_ = dim;
    LOG_DEBUG("Mesh dimension set to " << dim);
}

void Mesh::addVertex(const Vertex& v) {
    vertices_.push_back(v);
}

void Mesh::addVertex(Vertex&& v) {
    vertices_.push_back(std::move(v));
}

Index Mesh::addVertex(Real x, Real y, Real z) {
    vertices_.emplace_back(x, y, z, dim_);
    return static_cast<Index>(vertices_.size() - 1);
}

void Mesh::reserveVertices(Index n) {
    vertices_.reserve(n);
}

void Mesh::addElement(const Element& e) {
    elements_.push_back(e);
}

void Mesh::addElement(Element&& e) {
    elements_.push_back(std::move(e));
}

Index Mesh::addElement(Geometry geom, std::span<const Index> vertices, Index attr) {
    elements_.emplace_back(geom, vertices, attr);
    return static_cast<Index>(elements_.size() - 1);
}

Index Mesh::addElement(Geometry geom, const std::vector<Index>& vertices, Index attr) {
    elements_.emplace_back(geom, vertices, attr);
    return static_cast<Index>(elements_.size() - 1);
}

void Mesh::reserveElements(Index n) {
    elements_.reserve(n);
}

void Mesh::addBdrElement(const Element& e) {
    bdrElements_.push_back(e);
}

void Mesh::addBdrElement(Element&& e) {
    bdrElements_.push_back(std::move(e));
}

Index Mesh::addBdrElement(Geometry geom, std::span<const Index> vertices, Index attr) {
    bdrElements_.emplace_back(geom, vertices, attr);
    return static_cast<Index>(bdrElements_.size() - 1);
}

Index Mesh::addBdrElement(Geometry geom, const std::vector<Index>& vertices, Index attr) {
    bdrElements_.emplace_back(geom, vertices, attr);
    return static_cast<Index>(bdrElements_.size() - 1);
}

void Mesh::reserveBdrElements(Index n) {
    bdrElements_.reserve(n);
}

std::set<Index> Mesh::domainIds() const {
    std::set<Index> ids;
    for (const auto& e : elements_) {
        ids.insert(e.attribute());
    }
    return ids;
}

std::set<Index> Mesh::boundaryIds() const {
    std::set<Index> ids;
    for (const auto& e : bdrElements_) {
        ids.insert(e.attribute());
    }
    return ids;
}

std::vector<Index> Mesh::elementsForDomain(Index domainId) const {
    std::vector<Index> result;
    for (Index i = 0; i < static_cast<Index>(elements_.size()); ++i) {
        if (elements_[i].attribute() == domainId) {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<Index> Mesh::bdrElementsForBoundary(Index boundaryId) const {
    std::vector<Index> result;
    for (Index i = 0; i < static_cast<Index>(bdrElements_.size()); ++i) {
        if (bdrElements_[i].attribute() == boundaryId) {
            result.push_back(i);
        }
    }
    return result;
}

void Mesh::clear() {
    vertices_.clear();
    elements_.clear();
    bdrElements_.clear();
    dim_ = 3;
}

std::pair<Vector3, Vector3> Mesh::getBoundingBox() const {
    if (vertices_.empty()) {
        return {Vector3::Zero(), Vector3::Zero()};
    }
    
    Vector3 minCoord = vertices_[0].toVector();
    Vector3 maxCoord = minCoord;
    
    for (const auto& v : vertices_) {
        Vector3 p = v.toVector();
        minCoord = minCoord.cwiseMin(p);
        maxCoord = maxCoord.cwiseMax(p);
    }
    
    return {minCoord, maxCoord};
}

}  // namespace mpfem