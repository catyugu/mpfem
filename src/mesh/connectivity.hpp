/**
 * @file connectivity.hpp
 * @brief Entity connectivity management in CSR format
 */

#ifndef MPFEM_MESH_CONNECTIVITY_HPP
#define MPFEM_MESH_CONNECTIVITY_HPP

#include "core/types.hpp"

#include <algorithm>
#include <span>
#include <vector>

namespace mpfem {

/**
 * @brief CSR (Compressed Sparse Row) connectivity table
 * 
 * Stores entity-to-entity connectivity efficiently.
 * For example, face-to-cell connectivity, cell-to-vertex connectivity.
 */
class ConnectivityTable {
public:
    ConnectivityTable() = default;

    /// Initialize with given number of entities
    void initialize(SizeType num_entities) {
        offsets_.clear();
        offsets_.resize(num_entities + 1, 0);
        data_.clear();
    }

    /// Build connectivity from a list of connections per entity
    /// @param connections connections[i] contains all entities connected to entity i
    void build(const std::vector<std::vector<Index>>& connections) {
        const SizeType n = connections.size();
        offsets_.resize(n + 1);

        // Count total connections
        SizeType total = 0;
        for (SizeType i = 0; i < n; ++i) {
            total += connections[i].size();
        }

        data_.reserve(total);

        // Fill offsets and data
        offsets_[0] = 0;
        for (SizeType i = 0; i < n; ++i) {
            for (Index conn : connections[i]) {
                data_.push_back(conn);
            }
            offsets_[i + 1] = static_cast<Index>(data_.size());
        }
    }

    /// Get number of entities
    SizeType num_entities() const {
        return offsets_.empty() ? 0 : offsets_.size() - 1;
    }

    /// Get number of connections for entity i
    SizeType num_connections(SizeType i) const {
        if (i >= num_entities()) return 0;
        return offsets_[i + 1] - offsets_[i];
    }

    /// Get connections for entity i
    std::span<const Index> operator[](SizeType i) const {
        if (i >= num_entities()) {
            return {};
        }
        const Index start = offsets_[i];
        const Index end = offsets_[i + 1];
        return std::span<const Index>(data_.data() + start, end - start);
    }

    /// Check if entity i is connected to entity j
    bool is_connected(SizeType i, Index j) const {
        auto conns = (*this)[i];
        return std::find(conns.begin(), conns.end(), j) != conns.end();
    }

    /// Get total number of connections
    SizeType total_connections() const { return data_.size(); }

    /// Get raw offset array
    const IndexArray& offsets() const { return offsets_; }

    /// Get raw data array
    const IndexArray& data() const { return data_; }

    /// Clear all data
    void clear() {
        offsets_.clear();
        data_.clear();
    }

private:
    IndexArray offsets_;  ///< Row pointers (size: num_entities + 1)
    IndexArray data_;     ///< Column indices (connectivity data)
};

/**
 * @brief Bidirectional connectivity between two entity types
 * 
 * Maintains both forward (d1 -> d2) and reverse (d2 -> d1) connectivity.
 */
class BidirectionalConnectivity {
public:
    BidirectionalConnectivity(int d1 = -1, int d2 = -1)
        : dim1_(d1), dim2_(d2) {}

    /// Build connectivity from element data
    /// @param num_entities_d1 Number of entities of dimension d1
    /// @param element_connectivity For each entity of dim d2, list connected entities of dim d1
    void build(SizeType num_entities_d1,
               const std::vector<std::vector<Index>>& element_connectivity) {
        // Build forward (d2 -> d1) connectivity
        forward_.build(element_connectivity);

        // Build reverse (d1 -> d2) connectivity
        const SizeType num_entities_d2 = element_connectivity.size();
        std::vector<std::vector<Index>> reverse_conn(num_entities_d1);

        for (SizeType e2 = 0; e2 < num_entities_d2; ++e2) {
            for (Index e1 : element_connectivity[e2]) {
                if (e1 >= 0 && static_cast<SizeType>(e1) < num_entities_d1) {
                    reverse_conn[e1].push_back(static_cast<Index>(e2));
                }
            }
        }

        reverse_.build(reverse_conn);
    }

    /// Get forward connectivity (d2 -> d1)
    const ConnectivityTable& forward() const { return forward_; }

    /// Get reverse connectivity (d1 -> d2)
    const ConnectivityTable& reverse() const { return reverse_; }

    /// Get dimension of first entity type
    int dim1() const { return dim1_; }

    /// Get dimension of second entity type
    int dim2() const { return dim2_; }

    /// Clear all data
    void clear() {
        forward_.clear();
        reverse_.clear();
    }

private:
    int dim1_, dim2_;
    ConnectivityTable forward_;
    ConnectivityTable reverse_;
};

}  // namespace mpfem

#endif  // MPFEM_MESH_CONNECTIVITY_HPP
