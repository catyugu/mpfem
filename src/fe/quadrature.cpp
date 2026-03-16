#include "fe/quadrature.hpp"
#include <cmath>

// Define M_PI if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mpfem {

namespace gauss {

std::pair<std::vector<Real>, std::vector<Real>> get1D(int order) {
    std::vector<Real> xi(order);
    std::vector<Real> w(order);
    
    // Gauss-Legendre points and weights on [-1, 1]
    // Using Newton's method to compute
    const int m = (order + 1) / 2;
    const Real eps = 1e-15;
    
    for (int i = 0; i < m; ++i) {
        // Initial guess
        Real z = std::cos(M_PI * (i + 0.75) / (order + 0.5));
        Real z1, pp;
        
        // Newton iteration
        do {
            Real p1 = 1.0, p2 = 0.0;
            
            for (int j = 1; j <= order; ++j) {
                Real p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            
            pp = order * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z = z - p1 / pp;
        } while (std::abs(z - z1) > eps);
        
        xi[i] = -z;
        xi[order - 1 - i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
        w[order - 1 - i] = w[i];
    }
    
    return {xi, w};
}

}  // namespace gauss

namespace dunavant {

// Dunavant triangle quadrature rules
// Reference: Dunavant, D. A. (1985). High degree efficient symmetrical Gaussian
// quadrature rules for the triangle. International Journal for Numerical Methods
// in Engineering, 21(6), 1129-1148.

QuadratureRule getTriangle(int order) {
    QuadratureRule rule;
    
    if (order == 1) {
        // Order 1, 1 point, exact for degree 1
        rule.points().push_back(IntegrationPoint(1.0/3.0, 1.0/3.0, 0.0, 0.5));
    } else if (order == 2) {
        // Order 2, 3 points, exact for degree 2
        const Real a = 1.0/6.0;
        const Real b = 2.0/3.0;
        const Real w = 1.0/6.0;
        rule.points().push_back(IntegrationPoint(a, a, 0.0, w));
        rule.points().push_back(IntegrationPoint(b, a, 0.0, w));
        rule.points().push_back(IntegrationPoint(a, b, 0.0, w));
    } else if (order == 3) {
        // Order 3, 4 points, exact for degree 3
        const Real w1 = -27.0/96.0;
        const Real w2 = 25.0/96.0;
        const Real a = 0.2;
        const Real b = 0.6;
        rule.points().push_back(IntegrationPoint(1.0/3.0, 1.0/3.0, 0.0, w1));
        rule.points().push_back(IntegrationPoint(a, a, 0.0, w2));
        rule.points().push_back(IntegrationPoint(b, a, 0.0, w2));
        rule.points().push_back(IntegrationPoint(a, b, 0.0, w2));
    } else if (order == 4) {
        // Order 4, 6 points, exact for degree 4
        // Reference: Dunavant (1985), same as MFEM
        // AddTriPoints3(off, a, weight) adds points: (a,a), (a,1-2a), (1-2a,a)
        
        // Group 1: a = 0.091576213509771, weight per point = 0.054975871827661
        // Points: (0.091576, 0.091576), (0.091576, 0.816848), (0.816848, 0.091576)
        const Real a1 = 0.091576213509770743460;
        const Real w1 = 0.054975871827660933819;
        const Real b1 = 1.0 - 2.0 * a1;  // = 0.81684757298045851208
        rule.points().push_back(IntegrationPoint(a1, a1, 0.0, w1));
        rule.points().push_back(IntegrationPoint(a1, b1, 0.0, w1));
        rule.points().push_back(IntegrationPoint(b1, a1, 0.0, w1));
        
        // Group 2: a = 0.445948490915965, weight per point = 0.111690794839006
        // Points: (0.445948, 0.445948), (0.445948, 0.108103), (0.108103, 0.445948)
        const Real a2 = 0.44594849091596488632;
        const Real w2 = 0.11169079483900573285;
        const Real b2 = 1.0 - 2.0 * a2;  // = 0.10810301816807022736
        rule.points().push_back(IntegrationPoint(a2, a2, 0.0, w2));
        rule.points().push_back(IntegrationPoint(a2, b2, 0.0, w2));
        rule.points().push_back(IntegrationPoint(b2, a2, 0.0, w2));
    } else if (order == 5) {
        // Order 5, 7 points, exact for degree 5
        const Real w1 = 0.225000000000000;
        const Real w2 = 0.132394152788506;
        const Real w3 = 0.125939180544827;
        const Real a1 = 0.333333333333333;
        const Real a2 = 0.470142064105115;
        const Real b2 = 0.059715871789770;
        const Real a3 = 0.101286507323456;
        const Real b3 = 0.797426985353087;
        
        rule.points().push_back(IntegrationPoint(a1, a1, 0.0, w1));
        rule.points().push_back(IntegrationPoint(a2, b2, 0.0, w2));
        rule.points().push_back(IntegrationPoint(b2, a2, 0.0, w2));
        rule.points().push_back(IntegrationPoint(b2, b2, 0.0, w2));
        rule.points().push_back(IntegrationPoint(a3, a3, 0.0, w3));
        rule.points().push_back(IntegrationPoint(b3, a3, 0.0, w3));
        rule.points().push_back(IntegrationPoint(a3, b3, 0.0, w3));
    } else {
        throw std::runtime_error("Quadrature order >5 not implemented for triangles");
    }
    
    return rule;
}

}  // namespace dunavant

namespace tet_quadrature {

// Tetrahedron quadrature rules
QuadratureRule getTetrahedron(int order) {
    QuadratureRule rule;
    
    // Barycentric coordinates: (xi1, xi2, xi3, xi4) with xi1 + xi2 + xi3 + xi4 = 1
    // Reference coordinates: (xi, eta, zeta) -> barycentric (1-xi-eta-zeta, xi, eta, zeta)
    
    if (order == 1) {
        // Order 1, 1 point
        const Real w = 1.0/6.0;
        rule.points().push_back(IntegrationPoint(0.25, 0.25, 0.25, w));
    } else if (order == 2) {
        // Order 2, 4 points
        const Real a = (5.0 - std::sqrt(5.0)) / 20.0;
        const Real b = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
        const Real w = 1.0/24.0;
        
        // Points are permutations of (a, a, a, b) in barycentric
        // (xi, eta, zeta) = (a, a, a), (a, a, b), (a, b, a), (b, a, a)
        rule.points().push_back(IntegrationPoint(a, a, a, w));
        rule.points().push_back(IntegrationPoint(a, a, b, w));
        rule.points().push_back(IntegrationPoint(a, b, a, w));
        rule.points().push_back(IntegrationPoint(b, a, a, w));
    } else if (order == 3) {
        // Order 3, 5 points
        const Real w0 = -2.0/15.0;
        const Real w1 = 3.0/40.0;
        const Real a = 0.25;
        const Real b = 1.0/6.0;
        const Real c = 0.5;
        
        rule.points().push_back(IntegrationPoint(a, a, a, w0));
        rule.points().push_back(IntegrationPoint(b, b, b, w1));
        rule.points().push_back(IntegrationPoint(c, b, b, w1));
        rule.points().push_back(IntegrationPoint(b, c, b, w1));
        rule.points().push_back(IntegrationPoint(b, b, c, w1));
    } else if (order == 4) {
        // Order 4, 11 points - degree 4 (with negative weight)
        // Reference: MFEM intrules.cpp, same as Keast (1986)
        
        // Group 1: AddTetPoints4(0, 1/14, 343/45000)
        // Permutations of (a, a, a, b) where a = 1/14, b = 11/14
        const Real a1 = 1.0 / 14.0;                    // ≈ 0.0714285714285714
        const Real b1 = 11.0 / 14.0;                   // ≈ 0.7857142857142857
        const Real w1 = 343.0 / 45000.0;               // ≈ 0.0076222222222222
        rule.points().push_back(IntegrationPoint(a1, a1, a1, w1));  // (a,a,a)
        rule.points().push_back(IntegrationPoint(a1, a1, b1, w1));  // (a,a,b)
        rule.points().push_back(IntegrationPoint(a1, b1, a1, w1));  // (a,b,a)
        rule.points().push_back(IntegrationPoint(b1, a1, a1, w1));  // (b,a,a)
        
        // Group 2: AddTetMidPoint(4, -74/5625) - center point with negative weight
        const Real w2 = -74.0 / 5625.0;                // ≈ -0.0131555555555556
        rule.points().push_back(IntegrationPoint(0.25, 0.25, 0.25, w2));
        
        // Group 3: AddTetPoints6(5, a, 28/1125)
        // Permutations of (a, a, b, b) where a = 0.100596423833200795, b = 0.399403576166799205
        const Real a3 = 0.100596423833200795;
        const Real b3 = 0.5 - a3;                      // ≈ 0.399403576166799205
        const Real w3 = 28.0 / 1125.0;                 // ≈ 0.0248888888888889
        // Permutations of (a, a, b): (a,a,b), (a,b,a), (b,a,a)
        rule.points().push_back(IntegrationPoint(a3, a3, b3, w3));
        rule.points().push_back(IntegrationPoint(a3, b3, a3, w3));
        rule.points().push_back(IntegrationPoint(b3, a3, a3, w3));
        // Permutations of (b, b, a): (b,b,a), (b,a,b), (a,b,b)
        rule.points().push_back(IntegrationPoint(b3, b3, a3, w3));
        rule.points().push_back(IntegrationPoint(b3, a3, b3, w3));
        rule.points().push_back(IntegrationPoint(a3, b3, b3, w3));
    } else {
        throw std::runtime_error("Quadrature order >6 not implemented for tetrahedra");
    }
    
    return rule;
}

}  // namespace tet_quadrature

namespace quadrature {

QuadratureRule getTriangle(int order) {
    return dunavant::getTriangle(order);
}

QuadratureRule getTetrahedron(int order) {
    return tet_quadrature::getTetrahedron(order);
}

}  // namespace quadrature

}  // namespace mpfem
