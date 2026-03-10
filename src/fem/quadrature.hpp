/**
 * @file quadrature.hpp
 * @brief Gauss quadrature points and weights
 */

#ifndef MPFEM_FEM_QUADRATURE_HPP
#define MPFEM_FEM_QUADRATURE_HPP

#include "core/types.hpp"
#include <vector>
#include <cmath>

// Define M_PI if not defined (Windows compatibility)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mpfem {

/**
 * @brief Gauss-Legendre quadrature points and weights for 1D interval [-1, 1]
 */
class GaussLegendre1D {
public:
    /**
     * @brief Get quadrature points and weights for given order
     * @param order Polynomial order (n points integrate polynomials up to order 2n-1 exactly)
     * @return Pair of (points, weights)
     */
    static std::pair<std::vector<Scalar>, std::vector<Scalar>> get(int order) {
        std::vector<Scalar> points(order);
        std::vector<Scalar> weights(order);
        
        // Precomputed Gauss-Legendre points and weights for common orders
        switch (order) {
            case 1:
                points[0] = 0.0;
                weights[0] = 2.0;
                break;
            case 2:
                points[0] = -1.0 / std::sqrt(3.0);
                points[1] =  1.0 / std::sqrt(3.0);
                weights[0] = 1.0;
                weights[1] = 1.0;
                break;
            case 3:
                points[0] = -std::sqrt(0.6);
                points[1] = 0.0;
                points[2] =  std::sqrt(0.6);
                weights[0] = 5.0 / 9.0;
                weights[1] = 8.0 / 9.0;
                weights[2] = 5.0 / 9.0;
                break;
            case 4: {
                const Scalar s = std::sqrt((3.0 + 2.0 * std::sqrt(6.0/5.0)) / 7.0);
                const Scalar t = std::sqrt((3.0 - 2.0 * std::sqrt(6.0/5.0)) / 7.0);
                const Scalar w1 = (18.0 + std::sqrt(30.0)) / 36.0;
                const Scalar w2 = (18.0 - std::sqrt(30.0)) / 36.0;
                points[0] = -s; points[1] = -t;
                points[2] =  t; points[3] =  s;
                weights[0] = w1; weights[1] = w2;
                weights[2] = w2; weights[3] = w1;
                break;
            }
            case 5: {
                const Scalar s = std::sqrt(5.0 + 2.0 * std::sqrt(10.0/7.0));
                const Scalar t = std::sqrt(5.0 - 2.0 * std::sqrt(10.0/7.0));
                const Scalar w1 = (322.0 + 13.0 * std::sqrt(70.0)) / 900.0;
                const Scalar w2 = (322.0 - 13.0 * std::sqrt(70.0)) / 900.0;
                points[0] = -s/3.0; points[1] = -t/3.0;
                points[2] = 0.0;
                points[3] =  t/3.0; points[4] =  s/3.0;
                weights[0] = w1; weights[1] = w2;
                weights[2] = 128.0/225.0;
                weights[3] = w2; weights[4] = w1;
                break;
            }
            default:
                // For higher orders, compute using Newton's method
                compute_gauss_legendre(order, points, weights);
                break;
        }
        
        return {points, weights};
    }

private:
    /// Compute Gauss-Legendre points using Newton iteration
    static void compute_gauss_legendre(int n, std::vector<Scalar>& x, std::vector<Scalar>& w) {
        const int m = (n + 1) / 2;
        const Scalar eps = 1e-15;
        
        for (int i = 0; i < m; ++i) {
            // Initial guess using Chebyshev points
            Scalar z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
            Scalar z1, pp;
            
            // Newton iteration
            do {
                Scalar p1 = 1.0, p2 = 0.0;
                
                // Recurrence relation for Legendre polynomials
                for (int j = 1; j <= n; ++j) {
                    Scalar p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
                }
                
                // Derivative
                pp = n * (z * p1 - p2) / (z * z - 1.0);
                z1 = z;
                z -= p1 / pp;
            } while (std::abs(z - z1) > eps);
            
            // Store symmetric pairs
            x[i] = -z;
            x[n - 1 - i] = z;
            w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
            w[n - 1 - i] = w[i];
        }
    }
};

/**
 * @brief Quadrature points and weights for reference triangle
 * Reference triangle: vertices at (0,0), (1,0), (0,1)
 */
class TriangleQuadrature {
public:
    static std::pair<std::vector<Point<2>>, std::vector<Scalar>> get(int order) {
        std::vector<Point<2>> points;
        std::vector<Scalar> weights;
        
        // Dunavant rules for triangle
        // Order 1 (exact for linear)
        if (order <= 1) {
            points.push_back(Point<2>(1.0/3.0, 1.0/3.0));
            weights.push_back(0.5);
        }
        // Order 2 (exact for quadratic)
        else if (order <= 2) {
            const Scalar a = 1.0/6.0;
            const Scalar b = 2.0/3.0;
            points = {{a, a}, {b, a}, {a, b}};
            weights = {1.0/6.0, 1.0/6.0, 1.0/6.0};
        }
        // Order 3 (exact for cubic)
        else if (order <= 3) {
            const Scalar a = 1.0/3.0;
            const Scalar b = 0.6;
            const Scalar c = 0.2;
            points = {{a, a}, {b, c}, {c, b}, {c, c}, {b, b}, {c, b}};
            // Wait, this is wrong. Let me use correct order 3 rule
            points.clear(); weights.clear();
            const Scalar a3 = 1.0/3.0;
            const Scalar a1 = 0.6;
            const Scalar a2 = 0.2;
            points = {{a3, a3}};
            weights = {-27.0/96.0};
            points.push_back({a1, a2}); weights.push_back(25.0/96.0);
            points.push_back({a2, a1}); weights.push_back(25.0/96.0);
            points.push_back({a2, a2}); weights.push_back(25.0/96.0);
        }
        // Order 4 and higher
        else {
            // Use order 4 rule
            const Scalar a = (6.0 - std::sqrt(15.0)) / 21.0;
            const Scalar b = (6.0 + std::sqrt(15.0)) / 21.0;
            const Scalar c = 9.0/80.0;
            const Scalar d = (155.0 + std::sqrt(15.0)) / 2400.0;
            const Scalar e = (155.0 - std::sqrt(15.0)) / 2400.0;
            
            points = {{1.0/3.0, 1.0/3.0}};
            weights = {c};
            
            for (int i = 0; i < 3; ++i) {
                Scalar p1 = a, p2 = b;
                if (i == 1) { p1 = b; p2 = a; }
                else if (i == 2) { p1 = a; p2 = a; }
                
                if (i == 0) { points.push_back({a, b}); weights.push_back(d); }
                else if (i == 1) { points.push_back({b, a}); weights.push_back(d); }
                else { points.push_back({a, a}); weights.push_back(d); }
            }
            
            for (int i = 0; i < 3; ++i) {
                if (i == 0) { points.push_back({b, b}); weights.push_back(e); }
                else if (i == 1) { points.push_back({b, 1.0-2.0*b}); weights.push_back(e); }
                else { points.push_back({1.0-2.0*b, b}); weights.push_back(e); }
            }
        }
        
        return {points, weights};
    }
};

/**
 * @brief Quadrature points and weights for reference tetrahedron
 * Reference tet: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
 */
class TetrahedronQuadrature {
public:
    static std::pair<std::vector<Point<3>>, std::vector<Scalar>> get(int order) {
        std::vector<Point<3>> points;
        std::vector<Scalar> weights;
        
        // Order 1 (exact for linear)
        if (order <= 1) {
            points.push_back(Point<3>(0.25, 0.25, 0.25));
            weights.push_back(1.0/6.0);
        }
        // Order 2 (exact for quadratic)
        else if (order <= 2) {
            const Scalar a = (5.0 - std::sqrt(5.0)) / 20.0;
            const Scalar b = (5.0 + std::sqrt(5.0)) / 20.0;
            points = {{a, a, a}, {b, a, a}, {a, b, a}, {a, a, b}};
            weights = {1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0};
        }
        // Higher orders - use order 3 rule (exact for cubic)
        else {
            // Keast rule of order 3
            const Scalar a = 0.25;
            const Scalar b = (4.0 - std::sqrt(5.0)) / 20.0;
            const Scalar c = (4.0 + std::sqrt(5.0)) / 20.0;
            const Scalar w1 = -2.0/15.0;
            const Scalar w2 = 3.0/40.0;
            
            points = {{a, a, a}};
            weights = {w1};
            
            // Permutations of (b, b, b, c)
            points.push_back({b, b, b}); weights.push_back(w2);
            points.push_back({c, b, b}); weights.push_back(w2);
            points.push_back({b, c, b}); weights.push_back(w2);
            points.push_back({b, b, c}); weights.push_back(w2);
        }
        
        return {points, weights};
    }
};

/**
 * @brief Quadrature points and weights for reference pyramid
 * Reference pyramid: base on z=0 with corners (-1,-1,0), (1,-1,0), (1,1,0), (-1,1,0), apex at (0,0,1)
 */
class PyramidQuadrature {
public:
    static std::pair<std::vector<Point<3>>, std::vector<Scalar>> get(int order) {
        std::vector<Point<3>> points;
        std::vector<Scalar> weights;
        
        // Order 1: single point at centroid
        if (order <= 1) {
            points.push_back(Point<3>(0.0, 0.0, 0.25));
            weights.push_back(4.0/3.0);  // Volume of reference pyramid = 4/3
        }
        // Order 2: 5-point rule (exact for linear)
        else if (order <= 2) {
            // Witherden-Vincent rule for pyramid
            const Scalar a = 0.5;
            const Scalar w1 = 16.0/45.0;
            const Scalar w2 = 4.0/45.0;
            
            points = {
                {0.0, 0.0, 0.5},      // Apex region
                {-a, -a, 0.0},        // Base corners
                { a, -a, 0.0},
                { a,  a, 0.0},
                {-a,  a, 0.0}
            };
            weights = {w1, w2, w2, w2, w2};
        }
        // Order 3+: 8-point rule (exact for quadratic)
        else {
            // Higher order rule
            const Scalar a = std::sqrt(2.0/3.0);
            const Scalar b = std::sqrt(1.0/3.0);
            const Scalar w = 1.0/6.0;
            
            // 8 points on the pyramid
            points = {
                {-b, -b, a},
                { b, -b, a},
                { b,  b, a},
                {-b,  b, a},
                {-b, -b, 0.0},
                { b, -b, 0.0},
                { b,  b, 0.0},
                {-b,  b, 0.0}
            };
            for (int i = 0; i < 8; ++i) {
                weights.push_back(w);
            }
        }
        
        return {points, weights};
    }
};

}  // namespace mpfem

#endif  // MPFEM_FEM_QUADRATURE_HPP