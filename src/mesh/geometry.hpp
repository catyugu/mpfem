/**
 * @file geometry.hpp
 * @brief Geometric entity management (domains and boundaries)
 */

#ifndef MPFEM_MESH_GEOMETRY_HPP
#define MPFEM_MESH_GEOMETRY_HPP

#include "core/types.hpp"

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace mpfem {

/**
 * @brief Represents a geometric domain (volume region in 3D)
 */
struct Domain {
    Index id;                      ///< Domain ID
    std::string name;              ///< Optional name
    std::string material;          ///< Material identifier
    IndexArray cells;              ///< Cell indices belonging to this domain

    Domain() : id(InvalidIndex) {}
    explicit Domain(Index id) : id(id) {}
};

/**
 * @brief Represents a geometric boundary (surface region in 3D)
 */
struct Boundary {
    Index id;                      ///< Boundary ID
    std::string name;              ///< Optional name
    std::string type;              ///< Boundary type (internal/external)
    IndexArray faces;              ///< Face indices belonging to this boundary

    Boundary() : id(InvalidIndex) {}
    explicit Boundary(Index id) : id(id) {}
};

/**
 * @brief Manages geometric entities (domains and boundaries)
 */
class GeometryManager {
public:
    /// Add a domain
    void add_domain(Index id) {
        if (domains_.find(id) == domains_.end()) {
            domains_[id] = Domain(id);
            domain_ids_.push_back(id);
        }
    }

    /// Add a boundary
    void add_boundary(Index id) {
        if (boundaries_.find(id) == boundaries_.end()) {
            boundaries_[id] = Boundary(id);
            boundary_ids_.push_back(id);
        }
    }

    /// Get domain by ID
    const Domain* get_domain(Index id) const {
        auto it = domains_.find(id);
        return it != domains_.end() ? &it->second : nullptr;
    }

    /// Get boundary by ID
    const Boundary* get_boundary(Index id) const {
        auto it = boundaries_.find(id);
        return it != boundaries_.end() ? &it->second : nullptr;
    }

    /// Get mutable domain by ID
    Domain* get_domain_mut(Index id) {
        auto it = domains_.find(id);
        return it != domains_.end() ? &it->second : nullptr;
    }

    /// Get mutable boundary by ID
    Boundary* get_boundary_mut(Index id) {
        auto it = boundaries_.find(id);
        return it != boundaries_.end() ? &it->second : nullptr;
    }

    /// Add cell to domain
    void add_cell_to_domain(Index domain_id, Index cell_id) {
        add_domain(domain_id);
        domains_[domain_id].cells.push_back(cell_id);
    }

    /// Add face to boundary
    void add_face_to_boundary(Index boundary_id, Index face_id) {
        add_boundary(boundary_id);
        boundaries_[boundary_id].faces.push_back(face_id);
    }

    /// Check if domain exists
    bool has_domain(Index id) const {
        return domains_.find(id) != domains_.end();
    }

    /// Check if boundary exists
    bool has_boundary(Index id) const {
        return boundaries_.find(id) != boundaries_.end();
    }

    /// Get all domain IDs
    const IndexArray& domain_ids() const { return domain_ids_; }

    /// Get all boundary IDs
    const IndexArray& boundary_ids() const { return boundary_ids_; }

    /// Get number of domains
    SizeType num_domains() const { return domains_.size(); }

    /// Get number of boundaries
    SizeType num_boundaries() const { return boundaries_.size(); }

    /// Clear all data
    void clear() {
        domains_.clear();
        boundaries_.clear();
        domain_ids_.clear();
        boundary_ids_.clear();
    }

    /// Set domain name
    void set_domain_name(Index id, const std::string& name) {
        add_domain(id);
        domains_[id].name = name;
    }

    /// Set domain material
    void set_domain_material(Index id, const std::string& material) {
        add_domain(id);
        domains_[id].material = material;
    }

    /// Set boundary name
    void set_boundary_name(Index id, const std::string& name) {
        add_boundary(id);
        boundaries_[id].name = name;
    }

private:
    std::unordered_map<Index, Domain> domains_;
    std::unordered_map<Index, Boundary> boundaries_;
    IndexArray domain_ids_;
    IndexArray boundary_ids_;
};

}  // namespace mpfem

#endif  // MPFEM_MESH_GEOMETRY_HPP
