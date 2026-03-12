#ifndef MPFEM_SHAPE_FUNCTION_HPP
#define MPFEM_SHAPE_FUNCTION_HPP

#include "mesh/geometry.hpp"
#include "core/types.hpp"
#include "core/exception.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <memory>

namespace mpfem
{

    /**
     * @brief Shape function values and derivatives at an integration point.
     *
     * For an element with n shape functions:
     * - values[i] = φ_i(xi)
     * - gradients[i] = ∇φ_i(xi)  (in reference coordinates)
     */
    struct ShapeValues
    {
        std::vector<Real> values;       ///< Shape function values
        std::vector<Vector3> gradients; ///< Shape function gradients (reference coordinates)

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
    class ShapeFunction
    {
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
        virtual ShapeValues eval(const Real *xi) const = 0;

        /**
         * @brief Evaluate shape functions at integration point.
         */
        ShapeValues eval(const IntegrationPoint &ip) const
        {
            return eval(&ip.xi);
        }

        /**
         * @brief Get shape function values only (no gradients).
         */
        virtual std::vector<Real> evalValues(const Real *xi) const = 0;

        /**
         * @brief Get the reference coordinates of the dof points.
         * For Lagrange elements, these are the node positions.
         */
        virtual std::vector<std::vector<Real>> dofCoords() const = 0;

        /**
         * @brief Factory method to create shape function for given geometry and order.
         */
        static std::unique_ptr<ShapeFunction> create(Geometry geom, int order);
    };

    // =============================================================================
    // H1 Lagrange Shape Functions
    // =============================================================================

    /**
     * @brief H1 Lagrange shape functions on segment.
     */
    class H1SegmentShape : public ShapeFunction
    {
    public:
        explicit H1SegmentShape(int order);

        Geometry geometry() const override { return Geometry::Segment; }
        int order() const override { return order_; }
        int numDofs() const override { return order_ + 1; }
        int dim() const override { return 1; }

        ShapeValues eval(const Real *xi) const override;
        std::vector<Real> evalValues(const Real *xi) const override;
        std::vector<std::vector<Real>> dofCoords() const override;

    private:
        int order_;
    };

    /**
     * @brief H1 Lagrange shape functions on triangle.
     */
    class H1TriangleShape : public ShapeFunction
    {
    public:
        explicit H1TriangleShape(int order);

        Geometry geometry() const override { return Geometry::Triangle; }
        int order() const override { return order_; }
        int numDofs() const override;
        int dim() const override { return 2; }

        ShapeValues eval(const Real *xi) const override;
        std::vector<Real> evalValues(const Real *xi) const override;
        std::vector<std::vector<Real>> dofCoords() const override;

    private:
        int order_;
    };

    /**
     * @brief H1 Lagrange shape functions on square (quadrilateral).
     */
    class H1SquareShape : public ShapeFunction
    {
    public:
        explicit H1SquareShape(int order);

        Geometry geometry() const override { return Geometry::Square; }
        int order() const override { return order_; }
        int numDofs() const override { return (order_ + 1) * (order_ + 1); }
        int dim() const override { return 2; }

        ShapeValues eval(const Real *xi) const override;
        std::vector<Real> evalValues(const Real *xi) const override;
        std::vector<std::vector<Real>> dofCoords() const override;

    private:
        int order_;
        H1SegmentShape segment1d_;
    };

    /**
     * @brief H1 Lagrange shape functions on tetrahedron.
     */
    class H1TetrahedronShape : public ShapeFunction
    {
    public:
        explicit H1TetrahedronShape(int order);

        Geometry geometry() const override { return Geometry::Tetrahedron; }
        int order() const override { return order_; }
        int numDofs() const override;
        int dim() const override { return 3; }

        ShapeValues eval(const Real *xi) const override;
        std::vector<Real> evalValues(const Real *xi) const override;
        std::vector<std::vector<Real>> dofCoords() const override;

    private:
        int order_;
    };

    /**
     * @brief H1 Lagrange shape functions on cube (hexahedron).
     */
    class H1CubeShape : public ShapeFunction
    {
    public:
        explicit H1CubeShape(int order);

        Geometry geometry() const override { return Geometry::Cube; }
        int order() const override { return order_; }
        int numDofs() const override { return (order_ + 1) * (order_ + 1) * (order_ + 1); }
        int dim() const override { return 3; }

        ShapeValues eval(const Real *xi) const override;
        std::vector<Real> evalValues(const Real *xi) const override;
        std::vector<std::vector<Real>> dofCoords() const override;

    private:
        int order_;
        H1SegmentShape segment1d_;
    };

} // namespace mpfem
#endif // MPFEM_SHAPE_FUNCTION_HPP