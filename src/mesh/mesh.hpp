/**
 * @file mesh.hpp
 * @brief Main mesh class for mpfem
 */

#ifndef MPFEM_MESH_MESH_HPP
#define MPFEM_MESH_MESH_HPP

#include "element.hpp"
#include "geometry.hpp"
#include "core/types.hpp"
#include "core/logger.hpp"

#include <span>
#include <vector>

namespace mpfem {

/**
 * @brief Main mesh class
 * 
 * Stores mesh vertices, elements organized by type (ElementBlock),
 * and provides geometry entity (domain/boundary) management.
 */
class Mesh {
public:
    Mesh() : dim_(0), num_vertices_(0) {}

    explicit Mesh(int dim) : dim_(dim), num_vertices_(0) {}

    // ============================================================
    // Vertex Access
    // ============================================================

    int dimension() const { return dim_; }
    SizeType num_vertices() const { return num_vertices_; }

    Point<3> vertex(Index i) const {
        if (i < 0 || static_cast<SizeType>(i) >= num_vertices_) {
            return Point<3>::Zero();
        }
        const SizeType idx = static_cast<SizeType>(i) * 3;
        return Point<3>(vertices_[idx], vertices_[idx + 1], vertices_[idx + 2]);
    }

    void set_vertex(Index i, const Point<3>& p) {
        const SizeType idx = static_cast<SizeType>(i) * 3;
        vertices_[idx] = p.x();
        vertices_[idx + 1] = p.y();
        vertices_[idx + 2] = p.z();
    }

    const ScalarArray& vertices() const { return vertices_; }

    void initialize_vertices(SizeType num_vertices) {
        num_vertices_ = num_vertices;
        vertices_.resize(num_vertices * 3, 0.0);
    }

    void set_dimension(int dim) { dim_ = dim; }

    // ============================================================
    // Element Blocks (organized by element type)
    // ============================================================

    /// Get number of cell blocks (blocks, not individual cells)
    SizeType num_cell_blocks() const { return cell_blocks_.size(); }
    
    /// Get number of face blocks (blocks, not individual faces)
    SizeType num_face_blocks() const { return face_blocks_.size(); }
    
    /// Get number of edge blocks
    SizeType num_edge_blocks() const { return edge_blocks_.size(); }

    /// Get total number of cells (volume elements)
    SizeType num_cells() const {
        SizeType count = 0;
        for (const auto& block : cell_blocks_) {
            count += block.size();
        }
        return count;
    }

    /// Get total number of faces (surface elements)
    SizeType num_faces() const {
        SizeType count = 0;
        for (const auto& block : face_blocks_) {
            count += block.size();
        }
        return count;
    }

    /// Get total number of edges
    SizeType num_edges() const {
        SizeType count = 0;
        for (const auto& block : edge_blocks_) {
            count += block.size();
        }
        return count;
    }

    const std::vector<ElementBlock>& cell_blocks() const { return cell_blocks_; }
    const std::vector<ElementBlock>& face_blocks() const { return face_blocks_; }
    const std::vector<ElementBlock>& edge_blocks() const { return edge_blocks_; }

    ElementBlock* add_cell_block(ElementType type) {
        cell_blocks_.emplace_back(type);
        return &cell_blocks_.back();
    }

    ElementBlock* add_face_block(ElementType type) {
        face_blocks_.emplace_back(type);
        return &face_blocks_.back();
    }

    ElementBlock* add_edge_block(ElementType type) {
        edge_blocks_.emplace_back(type);
        return &edge_blocks_.back();
    }

    /// Find cell block by element type (returns nullptr if not found)
    const ElementBlock* get_cell_block(ElementType type) const {
        for (const auto& block : cell_blocks_) {
            if (block.type() == type) return &block;
        }
        return nullptr;
    }

    /// Find face block by element type (returns nullptr if not found)
    const ElementBlock* get_face_block(ElementType type) const {
        for (const auto& block : face_blocks_) {
            if (block.type() == type) return &block;
        }
        return nullptr;
    }

    // ============================================================
    // Geometry Entity Access
    // ============================================================

    const GeometryManager& geometry() const { return geometry_; }
    GeometryManager& geometry() { return geometry_; }

    SizeType num_domains() const { return geometry_.num_domains(); }
    SizeType num_boundaries() const { return geometry_.num_boundaries(); }

    const IndexArray& domain_ids() const { return geometry_.domain_ids(); }
    const IndexArray& boundary_ids() const { return geometry_.boundary_ids(); }

    // ============================================================
    // Bounding Box
    // ============================================================

    Point<3> bbox_min() const {
        if (num_vertices_ == 0) return Point<3>::Zero();

        Point<3> min_pt(
            std::numeric_limits<Scalar>::max(),
            std::numeric_limits<Scalar>::max(),
            std::numeric_limits<Scalar>::max()
        );

        for (SizeType i = 0; i < num_vertices_; ++i) {
            auto v = vertex(static_cast<Index>(i));
            min_pt.x() = std::min(min_pt.x(), v.x());
            min_pt.y() = std::min(min_pt.y(), v.y());
            min_pt.z() = std::min(min_pt.z(), v.z());
        }
        return min_pt;
    }

    Point<3> bbox_max() const {
        if (num_vertices_ == 0) return Point<3>::Zero();

        Point<3> max_pt(
            std::numeric_limits<Scalar>::lowest(),
            std::numeric_limits<Scalar>::lowest(),
            std::numeric_limits<Scalar>::lowest()
        );

        for (SizeType i = 0; i < num_vertices_; ++i) {
            auto v = vertex(static_cast<Index>(i));
            max_pt.x() = std::max(max_pt.x(), v.x());
            max_pt.y() = std::max(max_pt.y(), v.y());
            max_pt.z() = std::max(max_pt.z(), v.z());
        }
        return max_pt;
    }

    // ============================================================
    // Mesh Statistics
    // ============================================================

    void print_info() const {
        MPFEM_INFO("=== Mesh Statistics ===");
        MPFEM_INFO("  Dimension: " << dim_);
        MPFEM_INFO("  Vertices: " << num_vertices_);
        MPFEM_INFO("  Domains: " << num_domains());
        MPFEM_INFO("  Boundaries: " << num_boundaries());
        MPFEM_INFO("  Cells: " << num_cells());
        MPFEM_INFO("  Faces: " << num_faces());
        MPFEM_INFO("  Edges: " << num_edges());

        for (const auto& block : cell_blocks_) {
            MPFEM_INFO("    " << element_type_name(block.type())
                              << ": " << block.size());
        }

        for (const auto& block : face_blocks_) {
            MPFEM_INFO("    " << element_type_name(block.type())
                              << ": " << block.size());
        }
    }

    void clear() {
        vertices_.clear();
        cell_blocks_.clear();
        face_blocks_.clear();
        edge_blocks_.clear();
        geometry_.clear();
        num_vertices_ = 0;
    }

private:
    int dim_;
    SizeType num_vertices_;
    ScalarArray vertices_;

    std::vector<ElementBlock> cell_blocks_;
    std::vector<ElementBlock> face_blocks_;
    std::vector<ElementBlock> edge_blocks_;

    GeometryManager geometry_;
};

}  // namespace mpfem

#endif  // MPFEM_MESH_MESH_HPP