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
    
    // Tensor product of 1D shape functions
    const int n = order_ + 1;
    auto shape1d_x = segment1d_.evalValues(&xi[0]);
    auto shape1d_y = segment1d_.evalValues(&xi[1]);
    
    // Get derivatives
    auto sv_x = segment1d_.eval(&xi[0]);
    auto sv_y = segment1d_.eval(&xi[1]);
    
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int idx = j * n + i;
            sv.values[idx] = shape1d_x[i] * shape1d_y[j];
            
            // Gradient: dφ/dξ = dφ_x/dξ * φ_y, dφ/dη = φ_x * dφ_y/dη
            sv.gradients[idx] = Vector3(
                sv_x.gradients[i].x() * shape1d_y[j],
                shape1d_x[i] * sv_y.gradients[j].x(),
                0.0
            );
        }
    }
    
    return sv;
}

inline std::vector<Real> H1SquareShape::evalValues(const Real* xi) const {
    std::vector<Real> values(numDofs());
    
    const int n = order_ + 1;
    auto shape1d_x = segment1d_.evalValues(&xi[0]);
    auto shape1d_y = segment1d_.evalValues(&xi[1]);
    
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            values[j * n + i] = shape1d_x[i] * shape1d_y[j];
        }
    }
    
    return values;
}

inline std::vector<std::vector<Real>> H1SquareShape::dofCoords() const {
    std::vector<std::vector<Real>> coords(numDofs());
    const int n = order_ + 1;
    
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int idx = j * n + i;
            coords[idx] = {
                -1.0 + 2.0 * i / order_,
                -1.0 + 2.0 * j / order_
            };
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
    
    // Tensor product of 1D shape functions
    const int n = order_ + 1;
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
                
                // Gradient
                sv.gradients[idx] = Vector3(
                    sv_x.gradients[i].x() * shape1d_y[j] * shape1d_z[k],
                    shape1d_x[i] * sv_y.gradients[j].x() * shape1d_z[k],
                    shape1d_x[i] * shape1d_y[j] * sv_z.gradients[k].x()
                );
            }
        }
    }
    
    return sv;
}

inline std::vector<Real> H1CubeShape::evalValues(const Real* xi) const {
    std::vector<Real> values(numDofs());
    
    const int n = order_ + 1;
    auto shape1d_x = segment1d_.evalValues(&xi[0]);
    auto shape1d_y = segment1d_.evalValues(&xi[1]);
    auto shape1d_z = segment1d_.evalValues(&xi[2]);
    
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                int idx = k * n * n + j * n + i;
                values[idx] = shape1d_x[i] * shape1d_y[j] * shape1d_z[k];
            }
        }
    }
    
    return values;
}

inline std::vector<std::vector<Real>> H1CubeShape::dofCoords() const {
    std::vector<std::vector<Real>> coords(numDofs());
    const int n = order_ + 1;
    
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
    
    return coords;
}

}  // namespace mpfem

#endif  // MPFEM_SHAPE_FUNCTION_HPP
