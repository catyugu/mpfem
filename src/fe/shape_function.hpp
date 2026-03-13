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
        
        /// Resize vectors (avoids reallocation if size matches)
        void resize(int n) {
            values.resize(n);
            gradients.resize(n);
        }
        
        /// Check if has capacity for n values
        bool hasCapacity(int n) const {
            return static_cast<int>(values.size()) == n && 
                   static_cast<int>(gradients.size()) == n;
        }
    };

    /**
     * @brief Abstract base class for finite element shape functions.
     *
     * Provides shape function evaluation on reference elements.
     * 
     * Performance-oriented interface:
     * - evalValues(xi, values): Compute only shape function values
     * - evalGrads(xi, grads): Compute only gradients
     * - eval(xi, sv): Compute both (convenience method)
     * 
     * All methods accept pre-allocated arrays to avoid runtime memory allocation.
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
         * @brief Evaluate shape function values only (no gradients).
         * @param xi Reference coordinates (size = dim())
         * @param values Pre-allocated array of size numDofs()
         */
        virtual void evalValues(const Real* xi, Real* values) const = 0;

        /**
         * @brief Evaluate shape function gradients only.
         * @param xi Reference coordinates (size = dim())
         * @param grads Pre-allocated array of size numDofs()
         */
        virtual void evalGrads(const Real* xi, Vector3* grads) const = 0;

        /**
         * @brief Evaluate both values and gradients into pre-allocated storage.
         * @param xi Reference coordinates (size = dim())
         * @param sv Pre-allocated ShapeValues
         */
        virtual void eval(const Real* xi, ShapeValues& sv) const {
            evalValues(xi, sv.values.data());
            evalGrads(xi, sv.gradients.data());
        }

        /**
         * @brief Evaluate at integration point.
         */
        void eval(const IntegrationPoint& ip, ShapeValues& sv) const {
            eval(&ip.xi, sv);
        }

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

        void evalValues(const Real* xi, Real* values) const override;
        void evalGrads(const Real* xi, Vector3* grads) const override;
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

        void evalValues(const Real* xi, Real* values) const override;
        void evalGrads(const Real* xi, Vector3* grads) const override;
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

        void evalValues(const Real* xi, Real* values) const override;
        void evalGrads(const Real* xi, Vector3* grads) const override;
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

        void evalValues(const Real* xi, Real* values) const override;
        void evalGrads(const Real* xi, Vector3* grads) const override;
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

        void evalValues(const Real* xi, Real* values) const override;
        void evalGrads(const Real* xi, Vector3* grads) const override;
        std::vector<std::vector<Real>> dofCoords() const override;

    private:
        int order_;
        H1SegmentShape segment1d_;
    };

} // namespace mpfem
#endif // MPFEM_SHAPE_FUNCTION_HPP
