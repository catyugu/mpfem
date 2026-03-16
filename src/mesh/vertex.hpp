#ifndef MPFEM_VERTEX_HPP
#define MPFEM_VERTEX_HPP

#include "core/types.hpp"
#include <Eigen/Dense>
#include <array>

namespace mpfem {

/**
 * @brief Vertex class representing a point in space.
 * 
 * Stores coordinates and provides basic geometric operations.
 * The dimension is determined at runtime based on the mesh.
 */
class Vertex {
public:
    /// Default constructor creates a vertex at origin
    Vertex() : coords_{0.0, 0.0, 0.0}, dim_(3) {}

    /// Construct from array
    explicit Vertex(const std::array<Real, 3>& coords, int dim = 3)
        : coords_(coords), dim_(dim) {}

    /// Construct from individual coordinates
    Vertex(Real x, Real y = 0.0, Real z = 0.0, int dim = 3)
        : coords_{x, y, z}, dim_(dim) {}

    /// Construct from Eigen vector
    explicit Vertex(const Vector3& v, int dim = 3)
        : coords_{v(0), v(1), v(2)}, dim_(dim) {}

    // -------------------------------------------------------------------------
    // Coordinate access
    // -------------------------------------------------------------------------

    /// Get x coordinate
    Real x() const { return coords_[0]; }
    
    /// Get y coordinate
    Real y() const { return coords_[1]; }
    
    /// Get z coordinate
    Real z() const { return coords_[2]; }

    /// Access coordinate by index (0, 1, 2)
    Real operator[](int i) const { return coords_[i]; }
    Real& operator[](int i) { return coords_[i]; }

    /// Get coordinate array
    const std::array<Real, 3>& coords() const { return coords_; }
    std::array<Real, 3>& coords() { return coords_; }

    /// Get dimension
    int dim() const { return dim_; }

    /// Set dimension (truncates or extends coordinates)
    void setDim(int dim) { dim_ = dim; }

    // -------------------------------------------------------------------------
    // Vector operations
    // -------------------------------------------------------------------------

    /// Convert to Eigen vector
    Vector3 toVector() const { return Vector3(coords_[0], coords_[1], coords_[2]); }

    /// Get 2D vector (first two components)
    Vector2 toVector2() const { return Vector2(coords_[0], coords_[1]); }

    /// Euclidean norm
    Real norm() const {
        Real sum = 0.0;
        for (int i = 0; i < dim_; ++i) {
            sum += coords_[i] * coords_[i];
        }
        return std::sqrt(sum);
    }

    /// Squared norm
    Real squaredNorm() const {
        Real sum = 0.0;
        for (int i = 0; i < dim_; ++i) {
            sum += coords_[i] * coords_[i];
        }
        return sum;
    }

    /// Distance to another vertex
    Real distanceTo(const Vertex& other) const {
        Real sum = 0.0;
        for (int i = 0; i < dim_; ++i) {
            Real d = coords_[i] - other.coords_[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    }

    /// Squared distance to another vertex
    Real squaredDistanceTo(const Vertex& other) const {
        Real sum = 0.0;
        for (int i = 0; i < dim_; ++i) {
            Real d = coords_[i] - other.coords_[i];
            sum += d * d;
        }
        return sum;
    }

    // -------------------------------------------------------------------------
    // Operators
    // -------------------------------------------------------------------------

    Vertex operator+(const Vertex& other) const {
        return Vertex(coords_[0] + other.coords_[0],
                      coords_[1] + other.coords_[1],
                      coords_[2] + other.coords_[2], dim_);
    }

    Vertex operator-(const Vertex& other) const {
        return Vertex(coords_[0] - other.coords_[0],
                      coords_[1] - other.coords_[1],
                      coords_[2] - other.coords_[2], dim_);
    }

    Vertex operator*(Real s) const {
        return Vertex(coords_[0] * s, coords_[1] * s, coords_[2] * s, dim_);
    }

    Vertex& operator+=(const Vertex& other) {
        for (int i = 0; i < 3; ++i) coords_[i] += other.coords_[i];
        return *this;
    }

    Vertex& operator-=(const Vertex& other) {
        for (int i = 0; i < 3; ++i) coords_[i] -= other.coords_[i];
        return *this;
    }

    Vertex& operator*=(Real s) {
        for (int i = 0; i < 3; ++i) coords_[i] *= s;
        return *this;
    }

    bool operator==(const Vertex& other) const {
        for (int i = 0; i < dim_; ++i) {
            if (coords_[i] != other.coords_[i]) return false;
        }
        return true;
    }

    bool operator!=(const Vertex& other) const {
        return !(*this == other);
    }

private:
    std::array<Real, 3> coords_;
    int dim_;
};

/// Scalar multiplication (scalar * vertex)
inline Vertex operator*(Real s, const Vertex& v) {
    return v * s;
}

}  // namespace mpfem

#endif  // MPFEM_VERTEX_HPP
