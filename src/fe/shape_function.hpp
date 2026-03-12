#ifndef MPFEM_SHAPE_FUNCTION_HPP
#define MPFEM_SHAPE_FUNCTION_HPP

#include "mesh/geometry.hpp"
#include "core/types.hpp"
#include <vector>
#include <array>
#include <cmath>

namespace mpfem {

/**
 * @brief Shape function values and derivatives at an integration point.
 * 
 * For an element with n shape functions:
 * - values[i] = φ_i(xi)
 * - gradients[i] = ∇φ_i(xi)  (in reference coordinates)
 */
struct ShapeValues {
    std::vector<Real> values;           ///< Shape function values
    std::vector<Vector3> gradients;     ///< Shape function gradients (reference coordinates)
    
    /// Get number of shape functions
    int size() const { return static_cast<int>(values.size()); }
    
    /// Check if empty
    bool empty() const { return values.empty(); }
};

/**
 * @brief Abstract base class for finite element shape functions.
 * 
 * Provides shape function evaluation on reference elements.
 */
class ShapeFunction {
public:
    virtual ~ShapeFunction() = default;
    
    /// Get geometry type
    virtual Geometry geometry() const = 0;
    
    /// Get polynomial order
    virtual int order() const = 0;
    
    /// Get number of shape functions (dofs per element)
    virtual int numDofs() const = 0;
    
    /// Get spatial dimension
    virtual int dim() const = 0;
    
    /**
     * @brief Evaluate shape functions at reference coordinates.
     * @param xi Reference coordinates (size = dim())
     * @return Shape function values and gradients
     */
    virtual ShapeValues eval(const Real* xi) const = 0;
    
    /**
     * @brief Evaluate shape functions at integration point.
     */
    ShapeValues eval(const IntegrationPoint& ip) const {
        return eval(&ip.xi);
    }
    
    /**
     * @brief Get shape function values only (no gradients).
     */
    virtual std::vector<Real> evalValues(const Real* xi) const = 0;
    
    /**
     * @brief Get the reference coordinates of the dof points.
     * For Lagrange elements, these are the node positions.
     */
    virtual std::vector<std::vector<Real>> dofCoords() const = 0;
};

// =============================================================================
// H1 Lagrange Shape Functions
// =============================================================================

/**
 * @brief H1 Lagrange shape functions on segment.
 */
class H1SegmentShape : public ShapeFunction {
public:
    explicit H1SegmentShape(int order);
    
    Geometry geometry() const override { return Geometry::Segment; }
    int order() const override { return order_; }
    int numDofs() const override { return order_ + 1; }
    int dim() const override { return 1; }
    
    ShapeValues eval(const Real* xi) const override;
    std::vector<Real> evalValues(const Real* xi) const override;
    std::vector<std::vector<Real>> dofCoords() const override;
    
private:
    int order_;
};

/**
 * @brief H1 Lagrange shape functions on triangle.
 */
class H1TriangleShape : public ShapeFunction {
public:
    explicit H1TriangleShape(int order);
    
    Geometry geometry() const override { return Geometry::Triangle; }
    int order() const override { return order_; }
    int numDofs() const override;
    int dim() const override { return 2; }
    
    ShapeValues eval(const Real* xi) const override;
    std::vector<Real> evalValues(const Real* xi) const override;
    std::vector<std::vector<Real>> dofCoords() const override;
    
private:
    int order_;
};

/**
 * @brief H1 Lagrange shape functions on square (quadrilateral).
 */
class H1SquareShape : public ShapeFunction {
public:
    explicit H1SquareShape(int order);
    
    Geometry geometry() const override { return Geometry::Square; }
    int order() const override { return order_; }
    int numDofs() const override { return (order_ + 1) * (order_ + 1); }
    int dim() const override { return 2; }
    
    ShapeValues eval(const Real* xi) const override;
    std::vector<Real> evalValues(const Real* xi) const override;
    std::vector<std::vector<Real>> dofCoords() const override;
    
private:
    int order_;
    H1SegmentShape segment1d_;
};

/**
 * @brief H1 Lagrange shape functions on tetrahedron.
 */
class H1TetrahedronShape : public ShapeFunction {
public:
    explicit H1TetrahedronShape(int order);
    
    Geometry geometry() const override { return Geometry::Tetrahedron; }
    int order() const override { return order_; }
    int numDofs() const override;
    int dim() const override { return 3; }
    
    ShapeValues eval(const Real* xi) const override;
    std::vector<Real> evalValues(const Real* xi) const override;
    std::vector<std::vector<Real>> dofCoords() const override;
    
private:
    int order_;
};

/**
 * @brief H1 Lagrange shape functions on cube (hexahedron).
 */
class H1CubeShape : public ShapeFunction {
public:
    explicit H1CubeShape(int order);
    
    Geometry geometry() const override { return Geometry::Cube; }
    int order() const override { return order_; }
    int numDofs() const override { return (order_ + 1) * (order_ + 1) * (order_ + 1); }
    int dim() const override { return 3; }
    
    ShapeValues eval(const Real* xi) const override;
    std::vector<Real> evalValues(const Real* xi) const override;
    std::vector<std::vector<Real>> dofCoords() const override;
    
private:
    int order_;
    H1SegmentShape segment1d_;
};

// =============================================================================
// Inline implementations
// =============================================================================

// H1SegmentShape implementation
inline H1SegmentShape::H1SegmentShape(int order) : order_(order) {
    // Order must be >= 1
    if (order_ < 1) order_ = 1;
}

inline ShapeValues H1SegmentShape::eval(const Real* xi) const {
    ShapeValues sv;
    sv.values.resize(numDofs());
    sv.gradients.resize(numDofs());
    
    const Real x = xi[0];
    
    if (order_ == 1) {
        // Linear: φ0 = (1-x)/2, φ1 = (1+x)/2
        sv.values[0] = 0.5 * (1.0 - x);
        sv.values[1] = 0.5 * (1.0 + x);
        sv.gradients[0] = Vector3(-0.5, 0.0, 0.0);
        sv.gradients[1] = Vector3(0.5, 0.0, 0.0);
    } else if (order_ == 2) {
        // Quadratic: φ0 = x(x-1)/2, φ1 = 1-x^2, φ2 = x(x+1)/2
        sv.values[0] = 0.5 * x * (x - 1.0);
        sv.values[1] = 1.0 - x * x;
        sv.values[2] = 0.5 * x * (x + 1.0);
        sv.gradients[0] = Vector3(x - 0.5, 0.0, 0.0);
        sv.gradients[1] = Vector3(-2.0 * x, 0.0, 0.0);
        sv.gradients[2] = Vector3(x + 0.5, 0.0, 0.0);
    } else {
        // Higher order - evaluate using Lagrange formula
        // Generate node positions on [-1, 1]
        std::vector<Real> nodes(numDofs());
        for (int i = 0; i < numDofs(); ++i) {
            nodes[i] = -1.0 + 2.0 * i / order_;
        }
        
        // Evaluate each shape function
        for (int i = 0; i < numDofs(); ++i) {
            Real val = 1.0;
            Real deriv = 0.0;
            
            for (int j = 0; j < numDofs(); ++j) {
                if (j != i) {
                    val *= (x - nodes[j]) / (nodes[i] - nodes[j]);
                }
            }
            
            // Derivative using product rule
            for (int k = 0; k < numDofs(); ++k) {
                if (k == i) continue;
                Real term = 1.0 / (nodes[i] - nodes[k]);
                for (int j = 0; j < numDofs(); ++j) {
                    if (j != i && j != k) {
                        term *= (x - nodes[j]) / (nodes[i] - nodes[j]);
                    }
                }
                deriv += term;
            }
            
            sv.values[i] = val;
            sv.gradients[i] = Vector3(deriv, 0.0, 0.0);
        }
    }
    
    return sv;
}

inline std::vector<Real> H1SegmentShape::evalValues(const Real* xi) const {
    auto sv = eval(xi);
    return sv.values;
}

inline std::vector<std::vector<Real>> H1SegmentShape::dofCoords() const {
    std::vector<std::vector<Real>> coords(numDofs());
    for (int i = 0; i < numDofs(); ++i) {
        coords[i] = {-1.0 + 2.0 * i / order_};
    }
    return coords;
}

// H1TriangleShape implementation
inline H1TriangleShape::H1TriangleShape(int order) : order_(order) {
    if (order_ < 1) order_ = 1;
    if (order_ > 2) {
        // TODO: Support higher order
        order_ = 2;
    }
}

inline int H1TriangleShape::numDofs() const {
    return (order_ + 1) * (order_ + 2) / 2;
}

inline ShapeValues H1TriangleShape::eval(const Real* xi) const {
    ShapeValues sv;
    sv.values.resize(numDofs());
    sv.gradients.resize(numDofs());
    
    // Barycentric coordinates
    const Real xi1 = xi[0];
    const Real xi2 = xi[1];
    const Real xi3 = 1.0 - xi1 - xi2;
    
    if (order_ == 1) {
        // Linear: φi = λi (barycentric coordinate)
        sv.values[0] = xi3;  // Vertex 0
        sv.values[1] = xi1;  // Vertex 1
        sv.values[2] = xi2;  // Vertex 2
        
        // Gradients: dφi/dξj
        // φ0 = 1 - ξ1 - ξ2: grad = (-1, -1)
        // φ1 = ξ1: grad = (1, 0)
        // φ2 = ξ2: grad = (0, 1)
        sv.gradients[0] = Vector3(-1.0, -1.0, 0.0);
        sv.gradients[1] = Vector3(1.0, 0.0, 0.0);
        sv.gradients[2] = Vector3(0.0, 1.0, 0.0);
    } else if (order_ == 2) {
        // Quadratic (6 nodes)
        // Corner nodes: φi = λi(2λi - 1)
        // Edge nodes: φij = 4λiλj
        
        sv.values[0] = xi3 * (2.0 * xi3 - 1.0);  // Vertex 0
        sv.values[1] = xi1 * (2.0 * xi1 - 1.0);  // Vertex 1
        sv.values[2] = xi2 * (2.0 * xi2 - 1.0);  // Vertex 2
        sv.values[3] = 4.0 * xi1 * xi3;          // Edge 0-1
        sv.values[4] = 4.0 * xi1 * xi2;          // Edge 1-2
        sv.values[5] = 4.0 * xi2 * xi3;          // Edge 2-0
        
        // Gradients
        sv.gradients[0] = Vector3(-4.0*xi3 + 1.0, -4.0*xi3 + 1.0, 0.0);
        sv.gradients[1] = Vector3(4.0*xi1 - 1.0, 0.0, 0.0);
        sv.gradients[2] = Vector3(0.0, 4.0*xi2 - 1.0, 0.0);
        sv.gradients[3] = Vector3(4.0*xi3 - 4.0*xi1, -4.0*xi1, 0.0);
        sv.gradients[4] = Vector3(4.0*xi2, 4.0*xi1, 0.0);
        sv.gradients[5] = Vector3(-4.0*xi2, 4.0*xi3 - 4.0*xi2, 0.0);
    }
    
    return sv;
}

inline std::vector<Real> H1TriangleShape::evalValues(const Real* xi) const {
    auto sv = eval(xi);
    return sv.values;
}

inline std::vector<std::vector<Real>> H1TriangleShape::dofCoords() const {
    std::vector<std::vector<Real>> coords(numDofs());
    
    if (order_ == 1) {
        coords[0] = {0.0, 0.0};  // Vertex 0
        coords[1] = {1.0, 0.0};  // Vertex 1
        coords[2] = {0.0, 1.0};  // Vertex 2
    } else if (order_ == 2) {
        coords[0] = {0.0, 0.0};    // Vertex 0
        coords[1] = {1.0, 0.0};    // Vertex 1
        coords[2] = {0.0, 1.0};    // Vertex 2
        coords[3] = {0.5, 0.0};    // Edge 0-1
        coords[4] = {0.5, 0.5};    // Edge 1-2
        coords[5] = {0.0, 0.5};    // Edge 2-0
    }
    
    return coords;
}

// H1SquareShape implementation
inline H1SquareShape::H1SquareShape(int order) 
    : order_(order), segment1d_(order) {
    if (order_ < 1) order_ = 1;
}

inline ShapeValues H1SquareShape::eval(const Real* xi) const {
    ShapeValues sv;
    sv.values.resize(numDofs());
    sv.gradients.resize(numDofs());
    
    const int n = order_ + 1;
    
    if (order_ == 1) {
        // Linear: tensor product
        auto shape1d_x = segment1d_.evalValues(&xi[0]);
        auto shape1d_y = segment1d_.evalValues(&xi[1]);
        auto sv_x = segment1d_.eval(&xi[0]);
        auto sv_y = segment1d_.eval(&xi[1]);
        
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                int idx = j * n + i;
                sv.values[idx] = shape1d_x[i] * shape1d_y[j];
                sv.gradients[idx] = Vector3(
                    sv_x.gradients[i].x() * shape1d_y[j],
                    shape1d_x[i] * sv_y.gradients[j].x(),
                    0.0
                );
            }
        }
    } else if (order_ == 2) {
        // Quadratic: use geometric node ordering (corners -> edges -> center)
        // Node ordering: 4 corners + 4 edges + 1 center = 9 nodes
        // This matches COMSOL convention
        
        const Real x = xi[0];
        const Real y = xi[1];
        
        // 1D quadratic shape functions on [-1, 1]:
        // L0(x) = x(x-1)/2, L1(x) = 1-x^2, L2(x) = x(x+1)/2
        
        auto eval1D = [](Real t) -> std::array<Real, 3> {
            return {t * (t - 1) * 0.5, 1 - t * t, t * (t + 1) * 0.5};
        };
        auto eval1DGrad = [](Real t) -> std::array<Real, 3> {
            return {t - 0.5, -2 * t, t + 0.5};
        };
        
        auto Lx = eval1D(x);
        auto Ly = eval1D(y);
        auto dLx = eval1DGrad(x);
        auto dLy = eval1DGrad(y);
        
        // Corner nodes (0-3): vertices at corners
        // Corner 0: (-1,-1), Corner 1: (1,-1), Corner 2: (1,1), Corner 3: (-1,1)
        // Maps to tensor product indices: (0,0), (2,0), (2,2), (0,2)
        sv.values[0] = Lx[0] * Ly[0];  // (-1,-1)
        sv.values[1] = Lx[2] * Ly[0];  // (1,-1)
        sv.values[2] = Lx[2] * Ly[2];  // (1,1)
        sv.values[3] = Lx[0] * Ly[2];  // (-1,1)
        
        sv.gradients[0] = Vector3(dLx[0] * Ly[0], Lx[0] * dLy[0], 0.0);
        sv.gradients[1] = Vector3(dLx[2] * Ly[0], Lx[2] * dLy[0], 0.0);
        sv.gradients[2] = Vector3(dLx[2] * Ly[2], Lx[2] * dLy[2], 0.0);
        sv.gradients[3] = Vector3(dLx[0] * Ly[2], Lx[0] * dLy[2], 0.0);
        
        // Edge nodes (4-7): midpoints of edges
        // Edge 0: bottom (y=-1), nodes (0,1) -> midpoint at (0,-1)
        // Edge 1: right (x=1), nodes (1,2) -> midpoint at (1,0)
        // Edge 2: top (y=1), nodes (2,3) -> midpoint at (0,1)
        // Edge 3: left (x=-1), nodes (3,0) -> midpoint at (-1,0)
        // Maps to tensor product: (1,0), (2,1), (1,2), (0,1)
        sv.values[4] = Lx[1] * Ly[0];  // (0,-1)
        sv.values[5] = Lx[2] * Ly[1];  // (1,0)
        sv.values[6] = Lx[1] * Ly[2];  // (0,1)
        sv.values[7] = Lx[0] * Ly[1];  // (-1,0)
        
        sv.gradients[4] = Vector3(dLx[1] * Ly[0], Lx[1] * dLy[0], 0.0);
        sv.gradients[5] = Vector3(dLx[2] * Ly[1], Lx[2] * dLy[1], 0.0);
        sv.gradients[6] = Vector3(dLx[1] * Ly[2], Lx[1] * dLy[2], 0.0);
        sv.gradients[7] = Vector3(dLx[0] * Ly[1], Lx[0] * dLy[1], 0.0);
        
        // Center node (8): center at (0,0)
        // Maps to tensor product: (1,1)
        sv.values[8] = Lx[1] * Ly[1];
        sv.gradients[8] = Vector3(dLx[1] * Ly[1], Lx[1] * dLy[1], 0.0);
    } else {
        // Higher order: tensor product with uniform node spacing
        auto shape1d_x = segment1d_.evalValues(&xi[0]);
        auto shape1d_y = segment1d_.evalValues(&xi[1]);
        auto sv_x = segment1d_.eval(&xi[0]);
        auto sv_y = segment1d_.eval(&xi[1]);
        
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                int idx = j * n + i;
                sv.values[idx] = shape1d_x[i] * shape1d_y[j];
                sv.gradients[idx] = Vector3(
                    sv_x.gradients[i].x() * shape1d_y[j],
                    shape1d_x[i] * sv_y.gradients[j].x(),
                    0.0
                );
            }
        }
    }
    
    return sv;
}

inline std::vector<Real> H1SquareShape::evalValues(const Real* xi) const {
    auto sv = eval(xi);
    return sv.values;
}

inline std::vector<std::vector<Real>> H1SquareShape::dofCoords() const {
    std::vector<std::vector<Real>> coords(numDofs());
    const int n = order_ + 1;
    
    // Node ordering for H1 Square:
    // Order 1 (4 nodes): corners only
    // Order 2 (9 nodes): 4 corners + 4 edge midpoints + 1 center
    // Higher order: tensor product nodes
    
    if (order_ == 1) {
        // Corners only - use tensor product ordering to match eval()
        // Order: (i,j) where i,j ∈ {0,1}, idx = j*2 + i
        coords[0] = {-1.0, -1.0};  // i=0, j=0
        coords[1] = { 1.0, -1.0};  // i=1, j=0
        coords[2] = {-1.0,  1.0};  // i=0, j=1
        coords[3] = { 1.0,  1.0};  // i=1, j=1
    } else if (order_ == 2) {
        // 4 corners
        coords[0] = {-1.0, -1.0};
        coords[1] = { 1.0, -1.0};
        coords[2] = { 1.0,  1.0};
        coords[3] = {-1.0,  1.0};
        // 4 edge midpoints
        coords[4] = { 0.0, -1.0};  // bottom edge
        coords[5] = { 1.0,  0.0};  // right edge
        coords[6] = { 0.0,  1.0};  // top edge
        coords[7] = {-1.0,  0.0};  // left edge
        // 1 center
        coords[8] = { 0.0,  0.0};
    } else {
        // Higher order: tensor product nodes
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                int idx = j * n + i;
                coords[idx] = {
                    -1.0 + 2.0 * i / order_,
                    -1.0 + 2.0 * j / order_
                };
            }
        }
    }
    
    return coords;
}

// H1TetrahedronShape implementation
inline H1TetrahedronShape::H1TetrahedronShape(int order) : order_(order) {
    if (order_ < 1) order_ = 1;
    if (order_ > 2) {
        // TODO: Support higher order
        order_ = 2;
    }
}

inline int H1TetrahedronShape::numDofs() const {
    return (order_ + 1) * (order_ + 2) * (order_ + 3) / 6;
}

inline ShapeValues H1TetrahedronShape::eval(const Real* xi) const {
    ShapeValues sv;
    sv.values.resize(numDofs());
    sv.gradients.resize(numDofs());
    
    // Barycentric coordinates
    const Real xi1 = xi[0];
    const Real xi2 = xi[1];
    const Real xi3 = xi[2];
    const Real xi4 = 1.0 - xi1 - xi2 - xi3;
    
    if (order_ == 1) {
        // Linear: φi = λi
        sv.values[0] = xi4;  // Vertex 0
        sv.values[1] = xi1;  // Vertex 1
        sv.values[2] = xi2;  // Vertex 2
        sv.values[3] = xi3;  // Vertex 3
        
        // Gradients
        sv.gradients[0] = Vector3(-1.0, -1.0, -1.0);
        sv.gradients[1] = Vector3(1.0, 0.0, 0.0);
        sv.gradients[2] = Vector3(0.0, 1.0, 0.0);
        sv.gradients[3] = Vector3(0.0, 0.0, 1.0);
    } else if (order_ == 2) {
        // Quadratic (10 nodes)
        // Corner nodes: φi = λi(2λi - 1)
        // Edge nodes: φij = 4λiλj
        
        sv.values[0] = xi4 * (2.0 * xi4 - 1.0);   // Vertex 0
        sv.values[1] = xi1 * (2.0 * xi1 - 1.0);   // Vertex 1
        sv.values[2] = xi2 * (2.0 * xi2 - 1.0);   // Vertex 2
        sv.values[3] = xi3 * (2.0 * xi3 - 1.0);   // Vertex 3
        sv.values[4] = 4.0 * xi1 * xi4;           // Edge 0-1
        sv.values[5] = 4.0 * xi1 * xi2;           // Edge 1-2
        sv.values[6] = 4.0 * xi2 * xi4;           // Edge 2-0
        sv.values[7] = 4.0 * xi3 * xi4;           // Edge 0-3
        sv.values[8] = 4.0 * xi1 * xi3;           // Edge 1-3
        sv.values[9] = 4.0 * xi2 * xi3;           // Edge 2-3
        
        // Gradients
        sv.gradients[0] = Vector3(-4.0*xi4 + 1.0, -4.0*xi4 + 1.0, -4.0*xi4 + 1.0);
        sv.gradients[1] = Vector3(4.0*xi1 - 1.0, 0.0, 0.0);
        sv.gradients[2] = Vector3(0.0, 4.0*xi2 - 1.0, 0.0);
        sv.gradients[3] = Vector3(0.0, 0.0, 4.0*xi3 - 1.0);
        sv.gradients[4] = Vector3(4.0*xi4 - 4.0*xi1, -4.0*xi1, -4.0*xi1);
        sv.gradients[5] = Vector3(4.0*xi2, 4.0*xi1, 0.0);
        sv.gradients[6] = Vector3(-4.0*xi2, 4.0*xi4 - 4.0*xi2, -4.0*xi2);
        sv.gradients[7] = Vector3(-4.0*xi3, -4.0*xi3, 4.0*xi4 - 4.0*xi3);
        sv.gradients[8] = Vector3(4.0*xi3, 0.0, 4.0*xi1);
        sv.gradients[9] = Vector3(0.0, 4.0*xi3, 4.0*xi2);
    }
    
    return sv;
}

inline std::vector<Real> H1TetrahedronShape::evalValues(const Real* xi) const {
    auto sv = eval(xi);
    return sv.values;
}

inline std::vector<std::vector<Real>> H1TetrahedronShape::dofCoords() const {
    std::vector<std::vector<Real>> coords(numDofs());
    
    if (order_ == 1) {
        coords[0] = {0.0, 0.0, 0.0};  // Vertex 0
        coords[1] = {1.0, 0.0, 0.0};  // Vertex 1
        coords[2] = {0.0, 1.0, 0.0};  // Vertex 2
        coords[3] = {0.0, 0.0, 1.0};  // Vertex 3
    } else if (order_ == 2) {
        coords[0] = {0.0, 0.0, 0.0};    // Vertex 0
        coords[1] = {1.0, 0.0, 0.0};    // Vertex 1
        coords[2] = {0.0, 1.0, 0.0};    // Vertex 2
        coords[3] = {0.0, 0.0, 1.0};    // Vertex 3
        coords[4] = {0.5, 0.0, 0.0};    // Edge 0-1
        coords[5] = {0.5, 0.5, 0.0};    // Edge 1-2
        coords[6] = {0.0, 0.5, 0.0};    // Edge 2-0
        coords[7] = {0.0, 0.0, 0.5};    // Edge 0-3
        coords[8] = {0.5, 0.0, 0.5};    // Edge 1-3
        coords[9] = {0.0, 0.5, 0.5};    // Edge 2-3
    }
    
    return coords;
}

// H1CubeShape implementation
inline H1CubeShape::H1CubeShape(int order) 
    : order_(order), segment1d_(order) {
    if (order_ < 1) order_ = 1;
}

inline ShapeValues H1CubeShape::eval(const Real* xi) const {
    ShapeValues sv;
    sv.values.resize(numDofs());
    sv.gradients.resize(numDofs());
    
    const int n = order_ + 1;
    
    if (order_ == 1) {
        // Linear: tensor product
        auto shape1d_x = segment1d_.evalValues(&xi[0]);
        auto shape1d_y = segment1d_.evalValues(&xi[1]);
        auto shape1d_z = segment1d_.evalValues(&xi[2]);
        auto sv_x = segment1d_.eval(&xi[0]);
        auto sv_y = segment1d_.eval(&xi[1]);
        auto sv_z = segment1d_.eval(&xi[2]);
        
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    int idx = k * n * n + j * n + i;
                    sv.values[idx] = shape1d_x[i] * shape1d_y[j] * shape1d_z[k];
                    sv.gradients[idx] = Vector3(
                        sv_x.gradients[i].x() * shape1d_y[j] * shape1d_z[k],
                        shape1d_x[i] * sv_y.gradients[j].x() * shape1d_z[k],
                        shape1d_x[i] * shape1d_y[j] * sv_z.gradients[k].x()
                    );
                }
            }
        }
    } else if (order_ == 2) {
        // Quadratic: use geometric node ordering
        // Node ordering: 8 corners + 12 edges + 6 faces + 1 center = 27 nodes
        // This matches COMSOL convention
        
        const Real x = xi[0];
        const Real y = xi[1];
        const Real z = xi[2];
        
        // 1D quadratic shape functions
        auto eval1D = [](Real t) -> std::array<Real, 3> {
            return {t * (t - 1) * 0.5, 1 - t * t, t * (t + 1) * 0.5};
        };
        auto eval1DGrad = [](Real t) -> std::array<Real, 3> {
            return {t - 0.5, -2 * t, t + 0.5};
        };
        
        auto Lx = eval1D(x);
        auto Ly = eval1D(y);
        auto Lz = eval1D(z);
        auto dLx = eval1DGrad(x);
        auto dLy = eval1DGrad(y);
        auto dLz = eval1DGrad(z);
        
        int idx = 0;
        
        // 8 corners (0-7)
        // Corner vertices: (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
        //                  (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)
        // Maps to tensor product (i,j,k): (0,0,0), (2,0,0), (2,2,0), (0,2,0),
        //                                 (0,0,2), (2,0,2), (2,2,2), (0,2,2)
        sv.values[idx] = Lx[0] * Ly[0] * Lz[0]; sv.gradients[idx++] = Vector3(dLx[0]*Ly[0]*Lz[0], Lx[0]*dLy[0]*Lz[0], Lx[0]*Ly[0]*dLz[0]); // (-1,-1,-1)
        sv.values[idx] = Lx[2] * Ly[0] * Lz[0]; sv.gradients[idx++] = Vector3(dLx[2]*Ly[0]*Lz[0], Lx[2]*dLy[0]*Lz[0], Lx[2]*Ly[0]*dLz[0]); // ( 1,-1,-1)
        sv.values[idx] = Lx[2] * Ly[2] * Lz[0]; sv.gradients[idx++] = Vector3(dLx[2]*Ly[2]*Lz[0], Lx[2]*dLy[2]*Lz[0], Lx[2]*Ly[2]*dLz[0]); // ( 1, 1,-1)
        sv.values[idx] = Lx[0] * Ly[2] * Lz[0]; sv.gradients[idx++] = Vector3(dLx[0]*Ly[2]*Lz[0], Lx[0]*dLy[2]*Lz[0], Lx[0]*Ly[2]*dLz[0]); // (-1, 1,-1)
        sv.values[idx] = Lx[0] * Ly[0] * Lz[2]; sv.gradients[idx++] = Vector3(dLx[0]*Ly[0]*Lz[2], Lx[0]*dLy[0]*Lz[2], Lx[0]*Ly[0]*dLz[2]); // (-1,-1, 1)
        sv.values[idx] = Lx[2] * Ly[0] * Lz[2]; sv.gradients[idx++] = Vector3(dLx[2]*Ly[0]*Lz[2], Lx[2]*dLy[0]*Lz[2], Lx[2]*Ly[0]*dLz[2]); // ( 1,-1, 1)
        sv.values[idx] = Lx[2] * Ly[2] * Lz[2]; sv.gradients[idx++] = Vector3(dLx[2]*Ly[2]*Lz[2], Lx[2]*dLy[2]*Lz[2], Lx[2]*Ly[2]*dLz[2]); // ( 1, 1, 1)
        sv.values[idx] = Lx[0] * Ly[2] * Lz[2]; sv.gradients[idx++] = Vector3(dLx[0]*Ly[2]*Lz[2], Lx[0]*dLy[2]*Lz[2], Lx[0]*Ly[2]*dLz[2]); // (-1, 1, 1)
        
        // 12 edge midpoints (8-19)
        // Edge ordering matches geometry.hpp edge_table::Cube:
        // Bottom: 0:(0,1), 1:(1,2), 2:(2,3), 3:(3,0)
        // Top: 4:(4,5), 5:(5,6), 6:(6,7), 7:(7,4)
        // Vertical: 8:(0,4), 9:(1,5), 10:(2,6), 11:(3,7)
        // Maps to tensor product midpoints
        sv.values[idx] = Lx[1] * Ly[0] * Lz[0]; sv.gradients[idx++] = Vector3(dLx[1]*Ly[0]*Lz[0], Lx[1]*dLy[0]*Lz[0], Lx[1]*Ly[0]*dLz[0]); // (0,-1,-1)
        sv.values[idx] = Lx[2] * Ly[1] * Lz[0]; sv.gradients[idx++] = Vector3(dLx[2]*Ly[1]*Lz[0], Lx[2]*dLy[1]*Lz[0], Lx[2]*Ly[1]*dLz[0]); // (1, 0,-1)
        sv.values[idx] = Lx[1] * Ly[2] * Lz[0]; sv.gradients[idx++] = Vector3(dLx[1]*Ly[2]*Lz[0], Lx[1]*dLy[2]*Lz[0], Lx[1]*Ly[2]*dLz[0]); // (0, 1,-1)
        sv.values[idx] = Lx[0] * Ly[1] * Lz[0]; sv.gradients[idx++] = Vector3(dLx[0]*Ly[1]*Lz[0], Lx[0]*dLy[1]*Lz[0], Lx[0]*Ly[1]*dLz[0]); // (-1,0,-1)
        sv.values[idx] = Lx[1] * Ly[0] * Lz[2]; sv.gradients[idx++] = Vector3(dLx[1]*Ly[0]*Lz[2], Lx[1]*dLy[0]*Lz[2], Lx[1]*Ly[0]*dLz[2]); // (0,-1, 1)
        sv.values[idx] = Lx[2] * Ly[1] * Lz[2]; sv.gradients[idx++] = Vector3(dLx[2]*Ly[1]*Lz[2], Lx[2]*dLy[1]*Lz[2], Lx[2]*Ly[1]*dLz[2]); // (1, 0, 1)
        sv.values[idx] = Lx[1] * Ly[2] * Lz[2]; sv.gradients[idx++] = Vector3(dLx[1]*Ly[2]*Lz[2], Lx[1]*dLy[2]*Lz[2], Lx[1]*Ly[2]*dLz[2]); // (0, 1, 1)
        sv.values[idx] = Lx[0] * Ly[1] * Lz[2]; sv.gradients[idx++] = Vector3(dLx[0]*Ly[1]*Lz[2], Lx[0]*dLy[1]*Lz[2], Lx[0]*Ly[1]*dLz[2]); // (-1,0, 1)
        sv.values[idx] = Lx[0] * Ly[0] * Lz[1]; sv.gradients[idx++] = Vector3(dLx[0]*Ly[0]*Lz[1], Lx[0]*dLy[0]*Lz[1], Lx[0]*Ly[0]*dLz[1]); // (-1,-1,0)
        sv.values[idx] = Lx[2] * Ly[0] * Lz[1]; sv.gradients[idx++] = Vector3(dLx[2]*Ly[0]*Lz[1], Lx[2]*dLy[0]*Lz[1], Lx[2]*Ly[0]*dLz[1]); // ( 1,-1,0)
        sv.values[idx] = Lx[2] * Ly[2] * Lz[1]; sv.gradients[idx++] = Vector3(dLx[2]*Ly[2]*Lz[1], Lx[2]*dLy[2]*Lz[1], Lx[2]*Ly[2]*dLz[1]); // ( 1, 1,0)
        sv.values[idx] = Lx[0] * Ly[2] * Lz[1]; sv.gradients[idx++] = Vector3(dLx[0]*Ly[2]*Lz[1], Lx[0]*dLy[2]*Lz[1], Lx[0]*Ly[2]*dLz[1]); // (-1, 1,0)
        
        // 6 face centers (20-25)
        // Face ordering matches geometry.hpp face_table::Cube:
        // 0: bottom (-z), 1: top (+z), 2: front (-y), 3: back (+y), 4: left (-x), 5: right (+x)
        sv.values[idx] = Lx[1] * Ly[1] * Lz[0]; sv.gradients[idx++] = Vector3(dLx[1]*Ly[1]*Lz[0], Lx[1]*dLy[1]*Lz[0], Lx[1]*Ly[1]*dLz[0]); // (0,0,-1)
        sv.values[idx] = Lx[1] * Ly[1] * Lz[2]; sv.gradients[idx++] = Vector3(dLx[1]*Ly[1]*Lz[2], Lx[1]*dLy[1]*Lz[2], Lx[1]*Ly[1]*dLz[2]); // (0,0, 1)
        sv.values[idx] = Lx[1] * Ly[0] * Lz[1]; sv.gradients[idx++] = Vector3(dLx[1]*Ly[0]*Lz[1], Lx[1]*dLy[0]*Lz[1], Lx[1]*Ly[0]*dLz[1]); // (0,-1,0)
        sv.values[idx] = Lx[1] * Ly[2] * Lz[1]; sv.gradients[idx++] = Vector3(dLx[1]*Ly[2]*Lz[1], Lx[1]*dLy[2]*Lz[1], Lx[1]*Ly[2]*dLz[1]); // (0, 1,0)
        sv.values[idx] = Lx[0] * Ly[1] * Lz[1]; sv.gradients[idx++] = Vector3(dLx[0]*Ly[1]*Lz[1], Lx[0]*dLy[1]*Lz[1], Lx[0]*Ly[1]*dLz[1]); // (-1,0,0)
        sv.values[idx] = Lx[2] * Ly[1] * Lz[1]; sv.gradients[idx++] = Vector3(dLx[2]*Ly[1]*Lz[1], Lx[2]*dLy[1]*Lz[1], Lx[2]*Ly[1]*dLz[1]); // ( 1,0,0)
        
        // 1 volume center (26)
        sv.values[idx] = Lx[1] * Ly[1] * Lz[1];
        sv.gradients[idx] = Vector3(dLx[1]*Ly[1]*Lz[1], Lx[1]*dLy[1]*Lz[1], Lx[1]*Ly[1]*dLz[1]); // (0,0,0)
    } else {
        // Higher order: tensor product
        auto shape1d_x = segment1d_.evalValues(&xi[0]);
        auto shape1d_y = segment1d_.evalValues(&xi[1]);
        auto shape1d_z = segment1d_.evalValues(&xi[2]);
        auto sv_x = segment1d_.eval(&xi[0]);
        auto sv_y = segment1d_.eval(&xi[1]);
        auto sv_z = segment1d_.eval(&xi[2]);
        
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    int idx = k * n * n + j * n + i;
                    sv.values[idx] = shape1d_x[i] * shape1d_y[j] * shape1d_z[k];
                    sv.gradients[idx] = Vector3(
                        sv_x.gradients[i].x() * shape1d_y[j] * shape1d_z[k],
                        shape1d_x[i] * sv_y.gradients[j].x() * shape1d_z[k],
                        shape1d_x[i] * shape1d_y[j] * sv_z.gradients[k].x()
                    );
                }
            }
        }
    }
    
    return sv;
}

inline std::vector<Real> H1CubeShape::evalValues(const Real* xi) const {
    auto sv = eval(xi);
    return sv.values;
}

inline std::vector<std::vector<Real>> H1CubeShape::dofCoords() const {
    std::vector<std::vector<Real>> coords(numDofs());
    const int n = order_ + 1;
    
    // Node ordering for H1 Cube:
    // Order 1 (8 nodes): corners only
    // Order 2 (27 nodes): 8 corners + 12 edge midpoints + 6 face centers + 1 volume center
    // Higher order: tensor product nodes
    
    if (order_ == 1) {
        // 8 corners only - use tensor product ordering to match eval()
        // Order: (i,j,k) where i,j,k ∈ {0,1}, idx = k*4 + j*2 + i
        // Corresponding to xi ∈ {-1,1}, eta ∈ {-1,1}, zeta ∈ {-1,1}
        coords[0] = {-1.0, -1.0, -1.0};  // i=0, j=0, k=0
        coords[1] = { 1.0, -1.0, -1.0};  // i=1, j=0, k=0
        coords[2] = {-1.0,  1.0, -1.0};  // i=0, j=1, k=0
        coords[3] = { 1.0,  1.0, -1.0};  // i=1, j=1, k=0
        coords[4] = {-1.0, -1.0,  1.0};  // i=0, j=0, k=1
        coords[5] = { 1.0, -1.0,  1.0};  // i=1, j=0, k=1
        coords[6] = {-1.0,  1.0,  1.0};  // i=0, j=1, k=1
        coords[7] = { 1.0,  1.0,  1.0};  // i=1, j=1, k=1
    } else if (order_ == 2) {
        // 8 corners (0-7)
        coords[0] = {-1.0, -1.0, -1.0};
        coords[1] = { 1.0, -1.0, -1.0};
        coords[2] = { 1.0,  1.0, -1.0};
        coords[3] = {-1.0,  1.0, -1.0};
        coords[4] = {-1.0, -1.0,  1.0};
        coords[5] = { 1.0, -1.0,  1.0};
        coords[6] = { 1.0,  1.0,  1.0};
        coords[7] = {-1.0,  1.0,  1.0};
        
        // 12 edge midpoints (8-19)
        coords[8]  = { 0.0, -1.0, -1.0};  // edge 0: bottom front
        coords[9]  = { 1.0,  0.0, -1.0};  // edge 1: bottom right
        coords[10] = { 0.0,  1.0, -1.0};  // edge 2: bottom back
        coords[11] = {-1.0,  0.0, -1.0};  // edge 3: bottom left
        coords[12] = { 0.0, -1.0,  1.0};  // edge 4: top front
        coords[13] = { 1.0,  0.0,  1.0};  // edge 5: top right
        coords[14] = { 0.0,  1.0,  1.0};  // edge 6: top back
        coords[15] = {-1.0,  0.0,  1.0};  // edge 7: top left
        coords[16] = {-1.0, -1.0,  0.0};  // edge 8: front left vertical
        coords[17] = { 1.0, -1.0,  0.0};  // edge 9: front right vertical
        coords[18] = { 1.0,  1.0,  0.0};  // edge 10: back right vertical
        coords[19] = {-1.0,  1.0,  0.0};  // edge 11: back left vertical
        
        // 6 face centers (20-25)
        coords[20] = { 0.0,  0.0, -1.0};  // face 0: bottom (-z)
        coords[21] = { 0.0,  0.0,  1.0};  // face 1: top (+z)
        coords[22] = { 0.0, -1.0,  0.0};  // face 2: front (-y)
        coords[23] = { 0.0,  1.0,  0.0};  // face 3: back (+y)
        coords[24] = {-1.0,  0.0,  0.0};  // face 4: left (-x)
        coords[25] = { 1.0,  0.0,  0.0};  // face 5: right (+x)
        
        // 1 volume center (26)
        coords[26] = { 0.0,  0.0,  0.0};
    } else {
        // Higher order: tensor product nodes
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    int idx = k * n * n + j * n + i;
                    coords[idx] = {
                        -1.0 + 2.0 * i / order_,
                        -1.0 + 2.0 * j / order_,
                        -1.0 + 2.0 * k / order_
                    };
                }
            }
        }
    }
    
    return coords;
}

}  // namespace mpfem

#endif  // MPFEM_SHAPE_FUNCTION_HPP
