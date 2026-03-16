#include "shape_function.hpp"
#include <cassert>

namespace mpfem
{

// =============================================================================
// Factory
// =============================================================================

std::unique_ptr<ShapeFunction> ShapeFunction::create(Geometry geom, int order)
{
    switch (geom)
    {
    case Geometry::Segment:
        return std::make_unique<H1SegmentShape>(order);
    case Geometry::Triangle:
        return std::make_unique<H1TriangleShape>(order);
    case Geometry::Square:
        return std::make_unique<H1SquareShape>(order);
    case Geometry::Tetrahedron:
        return std::make_unique<H1TetrahedronShape>(order);
    case Geometry::Cube:
        return std::make_unique<H1CubeShape>(order);
    default:
        MPFEM_THROW(Exception, "Unsupported geometry for shape function");
    }
}

// =============================================================================
// H1SegmentShape
// =============================================================================

H1SegmentShape::H1SegmentShape(int order) : order_(order)
{
    if (order < 1 || order > 2)
    {
        MPFEM_THROW(Exception, "H1SegmentShape: only order 1 and 2 supported");
    }
}

void H1SegmentShape::evalValues(const Real* xi, Real* values) const
{
    const Real x = xi[0];
    if (order_ == 1)
    {
        // Linear: φ0 = 0.5*(1-x), φ1 = 0.5*(1+x)
        values[0] = 0.5 * (1.0 - x);
        values[1] = 0.5 * (1.0 + x);
    }
    else // order_ == 2
    {
        // Quadratic with nodes at -1, 0, 1
        // φ0 = -0.5*x*(1-x), φ1 = 1-x^2, φ2 = 0.5*x*(1+x)
        values[0] = -0.5 * x * (1.0 - x);
        values[1] = 1.0 - x * x;
        values[2] = 0.5 * x * (1.0 + x);
    }
}

void H1SegmentShape::evalGrads(const Real* xi, Vector3* grads) const
{
    const Real x = xi[0];
    if (order_ == 1)
    {
        grads[0] = Vector3(-0.5, 0.0, 0.0);
        grads[1] = Vector3(0.5, 0.0, 0.0);
    }
    else // order_ == 2
    {
        grads[0] = Vector3(x - 0.5, 0.0, 0.0);
        grads[1] = Vector3(-2.0 * x, 0.0, 0.0);
        grads[2] = Vector3(x + 0.5, 0.0, 0.0);
    }
}

std::vector<std::vector<Real>> H1SegmentShape::dofCoords() const
{
    if (order_ == 1)
    {
        return {{-1.0}, {1.0}};
    }
    else
    {
        return {{-1.0}, {0.0}, {1.0}};
    }
}

// =============================================================================
// H1TriangleShape
// =============================================================================

H1TriangleShape::H1TriangleShape(int order) : order_(order)
{
    if (order < 1 || order > 2)
    {
        MPFEM_THROW(Exception, "H1TriangleShape: only order 1 and 2 supported");
    }
}

int H1TriangleShape::numDofs() const
{
    return (order_ + 1) * (order_ + 2) / 2;
}

void H1TriangleShape::evalValues(const Real* xi, Real* values) const
{
    const Real xi1 = xi[0];
    const Real xi2 = xi[1];

    if (order_ == 1)
    {
        // Linear: φ0 = 1-ξ-η, φ1 = ξ, φ2 = η
        values[0] = 1.0 - xi1 - xi2;
        values[1] = xi1;
        values[2] = xi2;
    }
    else // order_ == 2
    {
        // Quadratic (6 dofs)
        // Vertex nodes
        values[0] = (1.0 - xi1 - xi2) * (1.0 - 2.0 * xi1 - 2.0 * xi2);
        values[1] = xi1 * (2.0 * xi1 - 1.0);
        values[2] = xi2 * (2.0 * xi2 - 1.0);
        // Edge nodes
        values[3] = 4.0 * xi1 * (1.0 - xi1 - xi2);
        values[4] = 4.0 * xi1 * xi2;
        values[5] = 4.0 * xi2 * (1.0 - xi1 - xi2);
    }
}

void H1TriangleShape::evalGrads(const Real* xi, Vector3* grads) const
{
    const Real xi1 = xi[0];
    const Real xi2 = xi[1];

    if (order_ == 1)
    {
        grads[0] = Vector3(-1.0, -1.0, 0.0);
        grads[1] = Vector3(1.0, 0.0, 0.0);
        grads[2] = Vector3(0.0, 1.0, 0.0);
    }
    else // order_ == 2
    {
        // ∇φ0 = ∇[(1-ξ-η)(1-2ξ-2η)]
        grads[0] = Vector3(4.0 * xi1 + 4.0 * xi2 - 3.0,
                           4.0 * xi1 + 4.0 * xi2 - 3.0, 0.0);
        // ∇φ1 = ∇[ξ(2ξ-1)]
        grads[1] = Vector3(4.0 * xi1 - 1.0, 0.0, 0.0);
        // ∇φ2 = ∇[η(2η-1)]
        grads[2] = Vector3(0.0, 4.0 * xi2 - 1.0, 0.0);
        // ∇φ3 = ∇[4ξ(1-ξ-η)]
        grads[3] = Vector3(4.0 - 8.0 * xi1 - 4.0 * xi2,
                           -4.0 * xi1, 0.0);
        // ∇φ4 = ∇[4ξη]
        grads[4] = Vector3(4.0 * xi2, 4.0 * xi1, 0.0);
        // ∇φ5 = ∇[4η(1-ξ-η)]
        grads[5] = Vector3(-4.0 * xi2,
                           4.0 - 4.0 * xi1 - 8.0 * xi2, 0.0);
    }
}

std::vector<std::vector<Real>> H1TriangleShape::dofCoords() const
{
    if (order_ == 1)
    {
        return {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    }
    else
    {
        return {
            {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, // vertices
            {0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5}  // edge midpoints
        };
    }
}

// =============================================================================
// H1SquareShape
// =============================================================================

H1SquareShape::H1SquareShape(int order) : order_(order), segment1d_(order)
{
    if (order < 1 || order > 2)
    {
        MPFEM_THROW(Exception, "H1SquareShape: only order 1 and 2 supported");
    }
}

void H1SquareShape::evalValues(const Real* xi, Real* values) const
{
    const Real x = xi[0];
    const Real y = xi[1];

    if (order_ == 1)
    {
        // Bilinear: φ = φ_x * φ_y (tensor product)
        const Real phi0x = 0.5 * (1.0 - x);
        const Real phi1x = 0.5 * (1.0 + x);
        const Real phi0y = 0.5 * (1.0 - y);
        const Real phi1y = 0.5 * (1.0 + y);

        values[0] = phi0x * phi0y; // (-1,-1)
        values[1] = phi1x * phi0y; // ( 1,-1)
        values[2] = phi1x * phi1y; // ( 1, 1)
        values[3] = phi0x * phi1y; // (-1, 1)
    }
    else // order_ == 2 (9 dofs with center node)
    {
        // Serendipity-like ordering: corners, edges, center
        // Using Lagrange basis with nodes at -1, 0, 1

        // 1D basis: φ_-1 = -0.5*x*(1-x), φ_0 = 1-x^2, φ_1 = 0.5*x*(1+x)
        const Real px0 = -0.5 * x * (1.0 - x);
        const Real px1 = 1.0 - x * x;
        const Real px2 = 0.5 * x * (1.0 + x);
        const Real py0 = -0.5 * y * (1.0 - y);
        const Real py1 = 1.0 - y * y;
        const Real py2 = 0.5 * y * (1.0 + y);

        // Corner nodes (tensor product of -1 and +1 nodes)
        values[0] = px0 * py0; // (-1,-1)
        values[1] = px2 * py0; // ( 1,-1)
        values[2] = px2 * py2; // ( 1, 1)
        values[3] = px0 * py2; // (-1, 1)

        // Edge nodes
        values[4] = px1 * py0; // ( 0,-1) bottom edge
        values[5] = px2 * py1; // ( 1, 0) right edge
        values[6] = px1 * py2; // ( 0, 1) top edge
        values[7] = px0 * py1; // (-1, 0) left edge

        // Center node
        values[8] = px1 * py1; // ( 0, 0)
    }
}

void H1SquareShape::evalGrads(const Real* xi, Vector3* grads) const
{
    const Real x = xi[0];
    const Real y = xi[1];

    if (order_ == 1)
    {
        // ∇φ = (∂φ/∂x, ∂φ/∂y) where φ = φ_x(x) * φ_y(y)
        const Real phi0x = 0.5 * (1.0 - x);
        const Real phi1x = 0.5 * (1.0 + x);
        const Real phi0y = 0.5 * (1.0 - y);
        const Real phi1y = 0.5 * (1.0 + y);

        const Real dphi0x = -0.5;
        const Real dphi1x = 0.5;
        const Real dphi0y = -0.5;
        const Real dphi1y = 0.5;

        grads[0] = Vector3(dphi0x * phi0y, phi0x * dphi0y, 0.0);
        grads[1] = Vector3(dphi1x * phi0y, phi1x * dphi0y, 0.0);
        grads[2] = Vector3(dphi1x * phi1y, phi1x * dphi1y, 0.0);
        grads[3] = Vector3(dphi0x * phi1y, phi0x * dphi1y, 0.0);
    }
    else // order_ == 2
    {
        // 1D basis and derivatives
        const Real px0 = -0.5 * x * (1.0 - x);
        const Real px1 = 1.0 - x * x;
        const Real px2 = 0.5 * x * (1.0 + x);
        const Real py0 = -0.5 * y * (1.0 - y);
        const Real py1 = 1.0 - y * y;
        const Real py2 = 0.5 * y * (1.0 + y);

        const Real dpx0 = x - 0.5;
        const Real dpx1 = -2.0 * x;
        const Real dpx2 = x + 0.5;
        const Real dpy0 = y - 0.5;
        const Real dpy1 = -2.0 * y;
        const Real dpy2 = y + 0.5;

        // Corner nodes
        grads[0] = Vector3(dpx0 * py0, px0 * dpy0, 0.0);
        grads[1] = Vector3(dpx2 * py0, px2 * dpy0, 0.0);
        grads[2] = Vector3(dpx2 * py2, px2 * dpy2, 0.0);
        grads[3] = Vector3(dpx0 * py2, px0 * dpy2, 0.0);

        // Edge nodes
        grads[4] = Vector3(dpx1 * py0, px1 * dpy0, 0.0);
        grads[5] = Vector3(dpx2 * py1, px2 * dpy1, 0.0);
        grads[6] = Vector3(dpx1 * py2, px1 * dpy2, 0.0);
        grads[7] = Vector3(dpx0 * py1, px0 * dpy1, 0.0);

        // Center node
        grads[8] = Vector3(dpx1 * py1, px1 * dpy1, 0.0);
    }
}

std::vector<std::vector<Real>> H1SquareShape::dofCoords() const
{
    if (order_ == 1)
    {
        return {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};
    }
    else
    {
        return {
            {-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}, // corners
            {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0},  // edges
            {0.0, 0.0}                                          // center
        };
    }
}

// =============================================================================
// H1TetrahedronShape
// =============================================================================

H1TetrahedronShape::H1TetrahedronShape(int order) : order_(order)
{
    if (order < 1 || order > 2)
    {
        MPFEM_THROW(Exception, "H1TetrahedronShape: only order 1 and 2 supported");
    }
}

int H1TetrahedronShape::numDofs() const
{
    return (order_ + 1) * (order_ + 2) * (order_ + 3) / 6;
}

void H1TetrahedronShape::evalValues(const Real* xi, Real* values) const
{
    const Real xi1 = xi[0];
    const Real xi2 = xi[1];
    const Real xi3 = xi[2];

    if (order_ == 1)
    {
        // Linear: φ0 = 1-ξ-η-ζ, φ1 = ξ, φ2 = η, φ3 = ζ
        values[0] = 1.0 - xi1 - xi2 - xi3;
        values[1] = xi1;
        values[2] = xi2;
        values[3] = xi3;
    }
    else // order_ == 2 (10 dofs)
    {
        // Vertex nodes
        values[0] = (1.0 - xi1 - xi2 - xi3) * (1.0 - 2.0 * xi1 - 2.0 * xi2 - 2.0 * xi3);
        values[1] = xi1 * (2.0 * xi1 - 1.0);
        values[2] = xi2 * (2.0 * xi2 - 1.0);
        values[3] = xi3 * (2.0 * xi3 - 1.0);

        // Edge nodes (midpoints)
        values[4] = 4.0 * xi1 * (1.0 - xi1 - xi2 - xi3); // edge 0-1
        values[5] = 4.0 * xi1 * xi2;                     // edge 1-2
        values[6] = 4.0 * xi2 * (1.0 - xi1 - xi2 - xi3); // edge 0-2
        values[7] = 4.0 * xi3 * (1.0 - xi1 - xi2 - xi3); // edge 0-3
        values[8] = 4.0 * xi1 * xi3;                     // edge 1-3
        values[9] = 4.0 * xi2 * xi3;                     // edge 2-3
    }
}

void H1TetrahedronShape::evalGrads(const Real* xi, Vector3* grads) const
{
    const Real xi1 = xi[0];
    const Real xi2 = xi[1];
    const Real xi3 = xi[2];

    if (order_ == 1)
    {
        grads[0] = Vector3(-1.0, -1.0, -1.0);
        grads[1] = Vector3(1.0, 0.0, 0.0);
        grads[2] = Vector3(0.0, 1.0, 0.0);
        grads[3] = Vector3(0.0, 0.0, 1.0);
    }
    else // order_ == 2
    {
        // ∇φ0 = ∇[(1-ξ-η-ζ)(1-2ξ-2η-2ζ)]
        grads[0] = Vector3(4.0 * xi1 + 4.0 * xi2 + 4.0 * xi3 - 3.0,
                           4.0 * xi1 + 4.0 * xi2 + 4.0 * xi3 - 3.0,
                           4.0 * xi1 + 4.0 * xi2 + 4.0 * xi3 - 3.0);
        // ∇φ1 = ∇[ξ(2ξ-1)]
        grads[1] = Vector3(4.0 * xi1 - 1.0, 0.0, 0.0);
        // ∇φ2 = ∇[η(2η-1)]
        grads[2] = Vector3(0.0, 4.0 * xi2 - 1.0, 0.0);
        // ∇φ3 = ∇[ζ(2ζ-1)]
        grads[3] = Vector3(0.0, 0.0, 4.0 * xi3 - 1.0);
        // ∇φ4 = ∇[4ξ(1-ξ-η-ζ)]
        grads[4] = Vector3(4.0 - 8.0 * xi1 - 4.0 * xi2 - 4.0 * xi3,
                           -4.0 * xi1, -4.0 * xi1);
        // ∇φ5 = ∇[4ξη]
        grads[5] = Vector3(4.0 * xi2, 4.0 * xi1, 0.0);
        // ∇φ6 = ∇[4η(1-ξ-η-ζ)]
        grads[6] = Vector3(-4.0 * xi2,
                           4.0 - 4.0 * xi1 - 8.0 * xi2 - 4.0 * xi3,
                           -4.0 * xi2);
        // ∇φ7 = ∇[4ζ(1-ξ-η-ζ)]
        grads[7] = Vector3(-4.0 * xi3, -4.0 * xi3,
                           4.0 - 4.0 * xi1 - 4.0 * xi2 - 8.0 * xi3);
        // ∇φ8 = ∇[4ξζ]
        grads[8] = Vector3(4.0 * xi3, 0.0, 4.0 * xi1);
        // ∇φ9 = ∇[4ηζ]
        grads[9] = Vector3(0.0, 4.0 * xi3, 4.0 * xi2);
    }
}

std::vector<std::vector<Real>> H1TetrahedronShape::dofCoords() const
{
    if (order_ == 1)
    {
        return {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    }
    else
    {
        return {
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, // vertices
            {0.5, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.0, 0.5, 0.0},                  // edges on z=0
            {0.0, 0.0, 0.5}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}                   // edges to z
        };
    }
}

// =============================================================================
// H1CubeShape
// =============================================================================

H1CubeShape::H1CubeShape(int order) : order_(order), segment1d_(order)
{
    if (order < 1 || order > 2)
    {
        MPFEM_THROW(Exception, "H1CubeShape: only order 1 and 2 supported");
    }
}

void H1CubeShape::evalValues(const Real* xi, Real* values) const
{
    const Real x = xi[0];
    const Real y = xi[1];
    const Real z = xi[2];

    if (order_ == 1)
    {
        // Trilinear: φ = φ_x * φ_y * φ_z
        const Real phi0x = 0.5 * (1.0 - x);
        const Real phi1x = 0.5 * (1.0 + x);
        const Real phi0y = 0.5 * (1.0 - y);
        const Real phi1y = 0.5 * (1.0 + y);
        const Real phi0z = 0.5 * (1.0 - z);
        const Real phi1z = 0.5 * (1.0 + z);

        // Order: z varies fastest, then y, then x (tensor product ordering)
        values[0] = phi0x * phi0y * phi0z; // (-1,-1,-1)
        values[1] = phi1x * phi0y * phi0z; // ( 1,-1,-1)
        values[2] = phi1x * phi1y * phi0z; // ( 1, 1,-1)
        values[3] = phi0x * phi1y * phi0z; // (-1, 1,-1)
        values[4] = phi0x * phi0y * phi1z; // (-1,-1, 1)
        values[5] = phi1x * phi0y * phi1z; // ( 1,-1, 1)
        values[6] = phi1x * phi1y * phi1z; // ( 1, 1, 1)
        values[7] = phi0x * phi1y * phi1z; // (-1, 1, 1)
    }
    else // order_ == 2 (27 dofs)
    {
        // 1D Lagrange basis with nodes at -1, 0, 1
        const Real px0 = -0.5 * x * (1.0 - x);
        const Real px1 = 1.0 - x * x;
        const Real px2 = 0.5 * x * (1.0 + x);
        const Real py0 = -0.5 * y * (1.0 - y);
        const Real py1 = 1.0 - y * y;
        const Real py2 = 0.5 * y * (1.0 + y);
        const Real pz0 = -0.5 * z * (1.0 - z);
        const Real pz1 = 1.0 - z * z;
        const Real pz2 = 0.5 * z * (1.0 + z);

        // Ordering: corners (8), edges (12), faces (6), center (1)
        // Corners
        values[0] = px0 * py0 * pz0;
        values[1] = px2 * py0 * pz0;
        values[2] = px2 * py2 * pz0;
        values[3] = px0 * py2 * pz0;
        values[4] = px0 * py0 * pz2;
        values[5] = px2 * py0 * pz2;
        values[6] = px2 * py2 * pz2;
        values[7] = px0 * py2 * pz2;

        // Edges (along x, y, z directions)
        values[8] = px1 * py0 * pz0;  // x-edge, y=-1, z=-1
        values[9] = px2 * py1 * pz0;  // y-edge, x=1, z=-1
        values[10] = px1 * py2 * pz0; // x-edge, y=1, z=-1
        values[11] = px0 * py1 * pz0; // y-edge, x=-1, z=-1
        values[12] = px0 * py0 * pz1; // z-edge, x=-1, y=-1
        values[13] = px2 * py0 * pz1; // z-edge, x=1, y=-1
        values[14] = px2 * py2 * pz1; // z-edge, x=1, y=1
        values[15] = px0 * py2 * pz1; // z-edge, x=-1, y=1
        values[16] = px1 * py0 * pz2; // x-edge, y=-1, z=1
        values[17] = px2 * py1 * pz2; // y-edge, x=1, z=1
        values[18] = px1 * py2 * pz2; // x-edge, y=1, z=1
        values[19] = px0 * py1 * pz2; // y-edge, x=-1, z=1

        // Faces
        values[20] = px1 * py1 * pz0; // z=-1 face
        values[21] = px1 * py0 * pz1; // y=-1 face
        values[22] = px2 * py1 * pz1; // x=1 face
        values[23] = px1 * py2 * pz1; // y=1 face
        values[24] = px0 * py1 * pz1; // x=-1 face
        values[25] = px1 * py1 * pz2; // z=1 face

        // Center
        values[26] = px1 * py1 * pz1;
    }
}

void H1CubeShape::evalGrads(const Real* xi, Vector3* grads) const
{
    const Real x = xi[0];
    const Real y = xi[1];
    const Real z = xi[2];

    if (order_ == 1)
    {
        const Real phi0x = 0.5 * (1.0 - x);
        const Real phi1x = 0.5 * (1.0 + x);
        const Real phi0y = 0.5 * (1.0 - y);
        const Real phi1y = 0.5 * (1.0 + y);
        const Real phi0z = 0.5 * (1.0 - z);
        const Real phi1z = 0.5 * (1.0 + z);

        const Real dphi0x = -0.5;
        const Real dphi1x = 0.5;
        const Real dphi0y = -0.5;
        const Real dphi1y = 0.5;
        const Real dphi0z = -0.5;
        const Real dphi1z = 0.5;

        grads[0] = Vector3(dphi0x * phi0y * phi0z, phi0x * dphi0y * phi0z, phi0x * phi0y * dphi0z);
        grads[1] = Vector3(dphi1x * phi0y * phi0z, phi1x * dphi0y * phi0z, phi1x * phi0y * dphi0z);
        grads[2] = Vector3(dphi1x * phi1y * phi0z, phi1x * dphi1y * phi0z, phi1x * phi1y * dphi0z);
        grads[3] = Vector3(dphi0x * phi1y * phi0z, phi0x * dphi1y * phi0z, phi0x * phi1y * dphi0z);
        grads[4] = Vector3(dphi0x * phi0y * phi1z, phi0x * dphi0y * phi1z, phi0x * phi0y * dphi1z);
        grads[5] = Vector3(dphi1x * phi0y * phi1z, phi1x * dphi0y * phi1z, phi1x * phi0y * dphi1z);
        grads[6] = Vector3(dphi1x * phi1y * phi1z, phi1x * dphi1y * phi1z, phi1x * phi1y * dphi1z);
        grads[7] = Vector3(dphi0x * phi1y * phi1z, phi0x * dphi1y * phi1z, phi0x * phi1y * dphi1z);
    }
    else // order_ == 2
    {
        // 1D basis and derivatives
        const Real px0 = -0.5 * x * (1.0 - x);
        const Real px1 = 1.0 - x * x;
        const Real px2 = 0.5 * x * (1.0 + x);
        const Real py0 = -0.5 * y * (1.0 - y);
        const Real py1 = 1.0 - y * y;
        const Real py2 = 0.5 * y * (1.0 + y);
        const Real pz0 = -0.5 * z * (1.0 - z);
        const Real pz1 = 1.0 - z * z;
        const Real pz2 = 0.5 * z * (1.0 + z);

        const Real dpx0 = x - 0.5;
        const Real dpx1 = -2.0 * x;
        const Real dpx2 = x + 0.5;
        const Real dpy0 = y - 0.5;
        const Real dpy1 = -2.0 * y;
        const Real dpy2 = y + 0.5;
        const Real dpz0 = z - 0.5;
        const Real dpz1 = -2.0 * z;
        const Real dpz2 = z + 0.5;

        // Corners
        grads[0] = Vector3(dpx0 * py0 * pz0, px0 * dpy0 * pz0, px0 * py0 * dpz0);
        grads[1] = Vector3(dpx2 * py0 * pz0, px2 * dpy0 * pz0, px2 * py0 * dpz0);
        grads[2] = Vector3(dpx2 * py2 * pz0, px2 * dpy2 * pz0, px2 * py2 * dpz0);
        grads[3] = Vector3(dpx0 * py2 * pz0, px0 * dpy2 * pz0, px0 * py2 * dpz0);
        grads[4] = Vector3(dpx0 * py0 * pz2, px0 * dpy0 * pz2, px0 * py0 * dpz2);
        grads[5] = Vector3(dpx2 * py0 * pz2, px2 * dpy0 * pz2, px2 * py0 * dpz2);
        grads[6] = Vector3(dpx2 * py2 * pz2, px2 * dpy2 * pz2, px2 * py2 * dpz2);
        grads[7] = Vector3(dpx0 * py2 * pz2, px0 * dpy2 * pz2, px0 * py2 * dpz2);

        // Edges
        grads[8] = Vector3(dpx1 * py0 * pz0, px1 * dpy0 * pz0, px1 * py0 * dpz0);
        grads[9] = Vector3(dpx2 * py1 * pz0, px2 * dpy1 * pz0, px2 * py1 * dpz0);
        grads[10] = Vector3(dpx1 * py2 * pz0, px1 * dpy2 * pz0, px1 * py2 * dpz0);
        grads[11] = Vector3(dpx0 * py1 * pz0, px0 * dpy1 * pz0, px0 * py1 * dpz0);
        grads[12] = Vector3(dpx0 * py0 * pz1, px0 * dpy0 * pz1, px0 * py0 * dpz1);
        grads[13] = Vector3(dpx2 * py0 * pz1, px2 * dpy0 * pz1, px2 * py0 * dpz1);
        grads[14] = Vector3(dpx2 * py2 * pz1, px2 * dpy2 * pz1, px2 * py2 * dpz1);
        grads[15] = Vector3(dpx0 * py2 * pz1, px0 * dpy2 * pz1, px0 * py2 * dpz1);
        grads[16] = Vector3(dpx1 * py0 * pz2, px1 * dpy0 * pz2, px1 * py0 * dpz2);
        grads[17] = Vector3(dpx2 * py1 * pz2, px2 * dpy1 * pz2, px2 * py1 * dpz2);
        grads[18] = Vector3(dpx1 * py2 * pz2, px1 * dpy2 * pz2, px1 * py2 * dpz2);
        grads[19] = Vector3(dpx0 * py1 * pz2, px0 * dpy1 * pz2, px0 * py1 * dpz2);

        // Faces
        grads[20] = Vector3(dpx1 * py1 * pz0, px1 * dpy1 * pz0, px1 * py1 * dpz0);
        grads[21] = Vector3(dpx1 * py0 * pz1, px1 * dpy0 * pz1, px1 * py0 * dpz1);
        grads[22] = Vector3(dpx2 * py1 * pz1, px2 * dpy1 * pz1, px2 * py1 * dpz1);
        grads[23] = Vector3(dpx1 * py2 * pz1, px1 * dpy2 * pz1, px1 * py2 * dpz1);
        grads[24] = Vector3(dpx0 * py1 * pz1, px0 * dpy1 * pz1, px0 * py1 * dpz1);
        grads[25] = Vector3(dpx1 * py1 * pz2, px1 * dpy1 * pz2, px1 * py1 * dpz2);

        // Center
        grads[26] = Vector3(dpx1 * py1 * pz1, px1 * dpy1 * pz1, px1 * py1 * dpz1);
    }
}

std::vector<std::vector<Real>> H1CubeShape::dofCoords() const
{
    if (order_ == 1)
    {
        return {
            {-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0},
            {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0}, {1.0, 1.0, 1.0}, {-1.0, 1.0, 1.0}};
    }
    else
    {
        return {
            // Corners
            {-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0},
            {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0}, {1.0, 1.0, 1.0}, {-1.0, 1.0, 1.0},
            // Edges
            {0.0, -1.0, -1.0}, {1.0, 0.0, -1.0}, {0.0, 1.0, -1.0}, {-1.0, 0.0, -1.0},
            {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {1.0, 1.0, 0.0}, {-1.0, 1.0, 0.0},
            {0.0, -1.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {-1.0, 0.0, 1.0},
            // Faces
            {0.0, 0.0, -1.0}, {0.0, -1.0, 0.0}, {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
            // Center
            {0.0, 0.0, 0.0}};
    }
}

} // namespace mpfem
