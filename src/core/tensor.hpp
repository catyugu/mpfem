#ifndef MPFEM_TENSOR_HPP
#define MPFEM_TENSOR_HPP

#include "core/exception.hpp"
#include "core/tensor_shape.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <array>
#include <variant>

namespace mpfem {

    /**
     * @brief Unified tensor value using std::variant for zero heap allocation
     *
     * Design principles:
     * - NO std::vector (eliminates ALL heap allocations)
     * - std::variant<Real, Vector3, Matrix3> for type-safe storage
     * - All operations use std::visit for open-closed polymorphic dispatch
     * - Type mismatches throw exceptions (fail-fast, no silent Zero() returns)
     */
    using TensorData = std::variant<Real, Vector3, Matrix3>;

    /// C++17 overloaded visitor pattern - eliminates if-else type chains
    template <class... Ts>
    struct overloaded : Ts... {
        using Ts::operator()...;
    };
    template <class... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;

    class Tensor {
    public:
        Tensor() : data_(Real(0)) { }
        explicit Tensor(Real v) : data_(v) { }
        explicit Tensor(const Vector3& v) : data_(v) { }
        explicit Tensor(const Matrix3& m) : data_(m) { }

        // Shape query
        TensorShape shape() const
        {
            if (isScalar())
                return TensorShape();
            if (isVector())
                return TensorShape::vector(3);
            return TensorShape::matrix(3, 3);
        }
        bool isScalar() const { return std::holds_alternative<Real>(data_); }
        bool isVector() const { return std::holds_alternative<Vector3>(data_); }
        bool isMatrix() const { return std::holds_alternative<Matrix3>(data_); }

        // Type-safe extractors
        Real asScalar() const { return std::get<Real>(data_); }
        const Vector3& asVector() const { return std::get<Vector3>(data_); }
        const Matrix3& asMatrix() const { return std::get<Matrix3>(data_); }

        // Raw buffer access
        Real* data()
        {
            return std::visit(overloaded {
                                  [](Real& x) -> Real* { return &x; },
                                  [](Vector3& x) -> Real* { return x.data(); },
                                  [](Matrix3& x) -> Real* { return x.data(); }},
                data_);
        }
        const Real* data() const
        {
            return std::visit(overloaded {
                                  [](const Real& x) -> const Real* { return &x; },
                                  [](const Vector3& x) -> const Real* { return x.data(); },
                                  [](const Matrix3& x) -> const Real* { return x.data(); }},
                data_);
        }
        size_t size() const
        {
            return std::visit(overloaded {
                                  [](const Real&) -> size_t { return 1; },
                                  [](const Vector3&) -> size_t { return 3; },
                                  [](const Matrix3&) -> size_t { return 9; }},
                data_);
        }

        // Element access
        Real scalar() const { return asScalar(); }
        Real operator[](int i) const
        {
            return std::visit(overloaded {
                                  [](const Real& x) -> Real { return x; },
                                  [i](const Vector3& v) -> Real { return v(i); },
                                  [i](const Matrix3& m) -> Real { return m.data()[i]; }},
                data_);
        }
        Real at(int row, int col) const { return asMatrix()(row, col); }

        // Factory methods
        static Tensor scalar(Real v) { return Tensor(v); }
        static Tensor vector(Real x, Real y, Real z) { return Tensor(Vector3(x, y, z)); }
        static Tensor vector3(const Vector3& v) { return Tensor(v); }
        static Tensor matrix3(const Matrix3& m) { return Tensor(m); }
        static Tensor matrix3(Real m00, Real m01, Real m02,
            Real m10, Real m11, Real m12,
            Real m20, Real m21, Real m22)
        {
            Matrix3 m;
            m << m00, m01, m02, m10, m11, m12, m20, m21, m22;
            return Tensor(m);
        }
        static Tensor identity3()
        {
            Matrix3 m;
            m.setIdentity();
            return Tensor(m);
        }
        static Tensor zero(const TensorShape& shape)
        {
            if (shape.isScalar())
                return Tensor(Real(0));
            if (shape.isVector())
                return Tensor(Vector3::Zero().eval());
            return Tensor(Matrix3::Zero().eval());
        }

        // Extractors
        Real toScalar() const { return asScalar(); }
        Vector3 toVector3() const { return asVector(); }
        Matrix3 toMatrix3() const { return asMatrix(); }
        void copyTo(Real* dest) const
        {
            const Real* src = data();
            for (size_t i = 0; i < size(); ++i)
                dest[i] = src[i];
        }

        // Variant accessor
        const TensorData& variant() const { return data_; }

        // =============================================================================
        // Operator overloads - std::visit with overloaded pattern
        // Throws on type mismatch (fail-fast, not silent Zero() return)
        // =============================================================================

        Tensor operator+(const Tensor& rhs) const
        {
            return applySameShapeBinary(data_, rhs.data_, [](const auto& a, const auto& b) { return a + b; }, "Tensor shape mismatch in add");
        }

        Tensor operator-(const Tensor& rhs) const
        {
            return applySameShapeBinary(data_, rhs.data_, [](const auto& a, const auto& b) { return a - b; }, "Tensor shape mismatch in subtract");
        }

        Tensor operator-() const
        {
            return std::visit(overloaded {
                                  [](Real x) -> Tensor { return Tensor(-x); },
                                  [](const Vector3& v) -> Tensor { return Tensor((-v).eval()); },
                                  [](const Matrix3& m) -> Tensor { return Tensor((-m).eval()); }},
                data_);
        }

        Tensor operator*(const Tensor& rhs) const
        {
            return std::visit(overloaded {
                                  // Scalar * Scalar
                                  [](Real a, Real b) -> Tensor { return Tensor(a * b); },
                                  // Scalar * Vector
                                  [](Real a, const Vector3& v) -> Tensor { return Tensor((a * v).eval()); },
                                  // Scalar * Matrix
                                  [](Real a, const Matrix3& m) -> Tensor { return Tensor((a * m).eval()); },
                                  // Vector * Scalar
                                  [](const Vector3& v, Real a) -> Tensor { return Tensor((v * a).eval()); },
                                  // Matrix * Scalar
                                  [](const Matrix3& m, Real a) -> Tensor { return Tensor((m * a).eval()); },
                                  // Matrix * Vector
                                  [](const Matrix3& A, const Vector3& v) -> Tensor { return Tensor((A * v).eval()); },
                                  // Matrix * Matrix
                                  [](const Matrix3& A, const Matrix3& B) -> Tensor { return Tensor((A * B).eval()); },
                                  // Invalid combinations
                                  [](auto&&, auto&&) -> Tensor {
                                      MPFEM_THROW(ArgumentException, "Invalid tensor multiplication shapes");
                                  }},
                data_, rhs.data_);
        }

        Tensor operator/(const Tensor& rhs) const
        {
            return std::visit(overloaded {
                                  [](Real a, Real b) -> Tensor { return Tensor(a / b); },
                                  [](const Vector3& v, Real s) -> Tensor { return Tensor((v / s).eval()); },
                                  [](const Matrix3& m, Real s) -> Tensor { return Tensor((m / s).eval()); },
                                  [](auto&&, auto&&) -> Tensor {
                                      MPFEM_THROW(ArgumentException, "Invalid tensor division: only tensor/scalar supported");
                                  }},
                data_, rhs.data_);
        }

    private:
        template <typename Op>
        static Tensor applySameShapeBinary(const TensorData& lhs, const TensorData& rhs, Op op, const char* error)
        {
            return std::visit(overloaded {
                                  [&op](Real a, Real b) -> Tensor { return Tensor(op(a, b)); },
                                  [&op](const Vector3& a, const Vector3& b) -> Tensor {
                                      return Tensor(op(a, b).eval());
                                  },
                                  [&op](const Matrix3& a, const Matrix3& b) -> Tensor {
                                      return Tensor(op(a, b).eval());
                                  },
                                  [error](auto&&, auto&&) -> Tensor {
                                      MPFEM_THROW(ArgumentException, error);
                                  }},
                lhs, rhs);
        }

        TensorData data_;
    };

    // =============================================================================
    // Free functions - also using std::visit for consistency
    // =============================================================================

    inline Tensor scale(const Tensor& t, Real s)
    {
        return std::visit(overloaded {
                              [s](Real x) -> Tensor { return Tensor(x * s); },
                              [s](const Vector3& v) -> Tensor { return Tensor((v * s).eval()); },
                              [s](const Matrix3& m) -> Tensor { return Tensor((m * s).eval()); }},
            t.variant());
    }

    inline Tensor add(const Tensor& a, const Tensor& b)
    {
        return a + b; // Delegates to operator+
    }

    inline Tensor subtract(const Tensor& a, const Tensor& b)
    {
        return a - b; // Delegates to operator-
    }

    inline Tensor negate(const Tensor& t)
    {
        return -t; // Delegates to operator-
    }

    inline Tensor matvec(const Tensor& A, const Tensor& b)
    {
        return std::visit(overloaded {
                              [](const Matrix3& A, const Vector3& v) -> Tensor {
                                  return Tensor((A * v).eval());
                              },
                              [](auto&&, auto&&) -> Tensor {
                                  MPFEM_THROW(ArgumentException, "matvec requires Matrix3 * Vector3");
                              }},
            A.variant(), b.variant());
    }

    inline Tensor matmat(const Tensor& A, const Tensor& B)
    {
        return std::visit(overloaded {
                              [](const Matrix3& A, const Matrix3& B) -> Tensor {
                                  return Tensor((A * B).eval());
                              },
                              [](auto&&, auto&&) -> Tensor {
                                  MPFEM_THROW(ArgumentException, "matmat requires Matrix3 * Matrix3");
                              }},
            A.variant(), B.variant());
    }

    inline Real dot(const Tensor& a, const Tensor& b)
    {
        return std::visit(overloaded {
                              [](const Vector3& u, const Vector3& v) -> Real { return u.dot(v); },
                              [](Real a, Real b) -> Real { return a * b; },
                              [](auto&&, auto&&) -> Real {
                                  MPFEM_THROW(ArgumentException, "dot requires Vector3.Vector3 or Scalar*Scalar");
                              }},
            a.variant(), b.variant());
    }

    inline Tensor cross(const Tensor& a, const Tensor& b)
    {
        return std::visit(overloaded {
                              [](const Vector3& u, const Vector3& v) -> Tensor {
                                  return Tensor(u.cross(v));
                              },
                              [](auto&&, auto&&) -> Tensor {
                                  MPFEM_THROW(ArgumentException, "cross requires Vector3 x Vector3");
                              }},
            a.variant(), b.variant());
    }

    inline Real trace(const Tensor& m)
    {
        return std::visit(overloaded {
                              [](const Matrix3& mat) -> Real { return mat.trace(); },
                              [](Real r) -> Real { return r; },
                              [](auto&&) -> Real {
                                  MPFEM_THROW(ArgumentException, "trace requires Matrix3 or Scalar");
                              }},
            m.variant());
    }

    inline Tensor transpose(const Tensor& m)
    {
        return std::visit(overloaded {
                              [](const Matrix3& mat) -> Tensor {
                                  return Tensor(Matrix3(mat.transpose().eval()));
                              },
                              [](auto&&) -> Tensor {
                                  MPFEM_THROW(ArgumentException, "transpose requires Matrix3");
                              }},
            m.variant());
    }

    inline Tensor sym(const Tensor& m)
    {
        return std::visit(overloaded {
                              [](const Matrix3& mat) -> Tensor {
                                  return Tensor(Matrix3((Real(0.5) * (mat + mat.transpose())).eval()));
                              },
                              [](auto&&) -> Tensor {
                                  MPFEM_THROW(ArgumentException, "sym requires Matrix3");
                              }},
            m.variant());
    }

    inline Real norm(const Tensor& t)
    {
        return std::visit(overloaded {
                              [](const Real& x) -> Real { return std::abs(x); },
                              [](const Vector3& v) -> Real { return v.norm(); },
                              [](const Matrix3& m) -> Real { return m.norm(); }},
            t.variant());
    }

} // namespace mpfem

#endif // MPFEM_TENSOR_HPP
