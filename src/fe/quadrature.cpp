#include "fe/quadrature.hpp"
#include "core/exception.hpp"
#include <basix/cell.h>
#include <basix/polyset.h>
#include <basix/quadrature.h>

namespace mpfem {

    namespace {

        basix::cell::type toBasixCell(Geometry geom)
        {
            switch (geom) {
            case Geometry::Segment:
                return basix::cell::type::interval;
            case Geometry::Triangle:
                return basix::cell::type::triangle;
            case Geometry::Square:
                return basix::cell::type::quadrilateral;
            case Geometry::Tetrahedron:
                return basix::cell::type::tetrahedron;
            case Geometry::Cube:
                return basix::cell::type::hexahedron;
            default:
                MPFEM_THROW(Exception, "Unsupported geometry for quadrature");
            }
        }

        QuadratureRule makeRule(const std::vector<double>& pts, const std::vector<double>& wts, int gdim)
        {
            std::vector<IntegrationPoint> points;
            const int num_pts = static_cast<int>(wts.size());
            points.reserve(num_pts);

            // 修复严重Bug：Basix的点数组是 (num_pts, gdim) 行优先排布
            for (int i = 0; i < num_pts; ++i) {
                Real x = pts[i * gdim + 0];
                Real y = (gdim > 1) ? pts[i * gdim + 1] : Real(0);
                Real z = (gdim > 2) ? pts[i * gdim + 2] : Real(0);
                points.emplace_back(x, y, z, wts[i]);
            }
            return QuadratureRule(std::move(points));
        }

    } // anonymous namespace

    namespace quadrature {

        QuadratureRule get(Geometry geom, int order)
        {
            if (geom == Geometry::Point || geom == Geometry::Invalid) {
                return QuadratureRule();
            }
            auto cell = toBasixCell(geom);
            auto result = basix::quadrature::make_quadrature<double>(
                basix::quadrature::type::Default, cell, basix::polyset::type::standard, order);

            return makeRule(result[0], result[1], geom::dim(geom));
        }

        QuadratureRule getSegment(int order) { return get(Geometry::Segment, order); }
        QuadratureRule getTriangle(int order) { return get(Geometry::Triangle, order); }
        QuadratureRule getSquare(int order) { return get(Geometry::Square, order); }
        QuadratureRule getTetrahedron(int order) { return get(Geometry::Tetrahedron, order); }
        QuadratureRule getCube(int order) { return get(Geometry::Cube, order); }

    } // namespace quadrature

} // namespace mpfem