#ifndef MPFEM_TENSOR_VALUE_HPP
#define MPFEM_TENSOR_VALUE_HPP

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

    class TensorValue {
    public:
        TensorValue() : data_(Real(0)) { }
        explicit TensorValue(Real v) : data_(v) { }
        explicit TensorValue(const Vector3& v) : data_(v) { }
        explicit TensorValue(const Matrix3& m) : data_(m) { }

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
        static TensorValue scalar(Real v) { return TensorValue(v); }
        static TensorValue vector(Real x, Real y, Real z) { return TensorValue(Vector3(x, y, z)); }
        static TensorValue vector3(const Vector3& v) { return TensorValue(v); }
        static TensorValue matrix3(const Matrix3& m) { return TensorValue(m); }
        static TensorValue matrix3(Real m00, Real m01, Real m02,
            Real m10, Real m11, Real m12,
            Real m20, Real m21, Real m22)
        {
            Matrix3 m;
            m << m00, m01, m02, m10, m11, m12, m20, m21, m22;
            return TensorValue(m);
        }
        static TensorValue identity3()
        {
            Matrix3 m;
            m.setIdentity();
            return TensorValue(m);
        }
        static TensorValue zero(const TensorShape& shape)
        {
            if (shape.isScalar())
                return TensorValue(Real(0));
            if (shape.isVector())
                return TensorValue(Vector3::Zero().eval());
            return TensorValue(Matrix3::Zero().eval());
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

        TensorValue operator+(const TensorValue& rhs) const
        {
            return std::visit(overloaded {
                                  [](Real a, Real b) -> TensorValue { return TensorValue(a + b); },
                                  [](const Vector3& a, const Vector3& b) -> TensorValue {
                                      return TensorValue((a + b).eval());
                                  },
                                  [](const Matrix3& a, const Matrix3& b) -> TensorValue {
                                      return TensorValue((a + b).eval());
                                  },
                                  [](auto&&, auto&&) -> TensorValue {
                                      MPFEM_THROW(ArgumentException, "Tensor shape mismatch in add");
                                  }},
                data_, rhs.data_);
        }

        TensorValue operator-(const TensorValue& rhs) const
        {
            return std::visit(overloaded {
                                  [](Real a, Real b) -> TensorValue { return TensorValue(a - b); },
                                  [](const Vector3& a, const Vector3& b) -> TensorValue {
                                      return TensorValue((a - b).eval());
                                  },
                                  [](const Matrix3& a, const Matrix3& b) -> TensorValue {
                                      return TensorValue((a - b).eval());
                                  },
                                  [](auto&&, auto&&) -> TensorValue {
                                      MPFEM_THROW(ArgumentException, "Tensor shape mismatch in subtract");
                                  }},
                data_, rhs.data_);
        }

        TensorValue operator-() const
        {
            return std::visit(overloaded {
                                  [](Real x) -> TensorValue { return TensorValue(-x); },
                                  [](const Vector3& v) -> TensorValue { return TensorValue((-v).eval()); },
                                  [](const Matrix3& m) -> TensorValue { return TensorValue((-m).eval()); }},
                data_);
        }

        TensorValue operator*(const TensorValue& rhs) const
        {
            return std::visit(overloaded {
                                  // Scalar * Scalar
                                  [](Real a, Real b) -> TensorValue { return TensorValue(a * b); },
                                  // Scalar * Vector
                                  [](Real a, const Vector3& v) -> TensorValue { return TensorValue((a * v).eval()); },
                                  // Scalar * Matrix
                                  [](Real a, const Matrix3& m) -> TensorValue { return TensorValue((a * m).eval()); },
                                  // Vector * Scalar
                                  [](const Vector3& v, Real a) -> TensorValue { return TensorValue((v * a).eval()); },
                                  // Matrix * Scalar
                                  [](const Matrix3& m, Real a) -> TensorValue { return TensorValue((m * a).eval()); },
                                  // Matrix * Vector
                                  [](const Matrix3& A, const Vector3& v) -> TensorValue { return TensorValue((A * v).eval()); },
                                  // Matrix * Matrix
                                  [](const Matrix3& A, const Matrix3& B) -> TensorValue { return TensorValue((A * B).eval()); },
                                  // Invalid combinations
                                  [](auto&&, auto&&) -> TensorValue {
                                      MPFEM_THROW(ArgumentException, "Invalid tensor multiplication shapes");
                                  }},
                data_, rhs.data_);
        }

        TensorValue operator/(const TensorValue& rhs) const
        {
            return std::visit(overloaded {
                                  [](Real a, Real b) -> TensorValue { return TensorValue(a / b); },
                                  [](const Vector3& v, Real s) -> TensorValue { return TensorValue((v / s).eval()); },
                                  [](const Matrix3& m, Real s) -> TensorValue { return TensorValue((m / s).eval()); },
                                  [](auto&&, auto&&) -> TensorValue {
                                      MPFEM_THROW(ArgumentException, "Invalid tensor division: only tensor/scalar supported");
                                  }},
                data_, rhs.data_);
        }

    private:
        TensorData data_;
    };

    // =============================================================================
    // Free functions - also using std::visit for consistency
    // =============================================================================

    inline TensorValue scale(const TensorValue& t, Real s)
    {
        return std::visit(overloaded {
                              [s](Real x) -> TensorValue { return TensorValue(x * s); },
                              [s](const Vector3& v) -> TensorValue { return TensorValue((v * s).eval()); },
                              [s](const Matrix3& m) -> TensorValue { return TensorValue((m * s).eval()); }},
            t.variant());
    }

    inline TensorValue add(const TensorValue& a, const TensorValue& b)
    {
        return a + b; // Delegates to operator+
    }

    inline TensorValue subtract(const TensorValue& a, const TensorValue& b)
    {
        return a - b; // Delegates to operator-
    }

    inline TensorValue negate(const TensorValue& t)
    {
        return -t; // Delegates to operator-
    }

    inline TensorValue matvec(const TensorValue& A, const TensorValue& b)
    {
        return std::visit(overloaded {
                              [](const Matrix3& A, const Vector3& v) -> TensorValue {
                                  return TensorValue((A * v).eval());
                              },
                              [](auto&&, auto&&) -> TensorValue {
                                  MPFEM_THROW(ArgumentException, "matvec requires Matrix3 * Vector3");
                              }},
            A.variant(), b.variant());
    }

    inline TensorValue matmat(const TensorValue& A, const TensorValue& B)
    {
        return std::visit(overloaded {
                              [](const Matrix3& A, const Matrix3& B) -> TensorValue {
                                  return TensorValue((A * B).eval());
                              },
                              [](auto&&, auto&&) -> TensorValue {
                                  MPFEM_THROW(ArgumentException, "matmat requires Matrix3 * Matrix3");
                              }},
            A.variant(), B.variant());
    }

    inline Real dot(const TensorValue& a, const TensorValue& b)
    {
        return std::visit(overloaded {
                              [](const Vector3& u, const Vector3& v) -> Real { return u.dot(v); },
                              [](Real a, Real b) -> Real { return a * b; },
                              [](auto&&, auto&&) -> Real {
                                  MPFEM_THROW(ArgumentException, "dot requires Vector3.Vector3 or Scalar*Scalar");
                              }},
            a.variant(), b.variant());
    }

    inline TensorValue cross(const TensorValue& a, const TensorValue& b)
    {
        return std::visit(overloaded {
                              [](const Vector3& u, const Vector3& v) -> TensorValue {
                                  return TensorValue(u.cross(v));
                              },
                              [](auto&&, auto&&) -> TensorValue {
                                  MPFEM_THROW(ArgumentException, "cross requires Vector3 x Vector3");
                              }},
            a.variant(), b.variant());
    }

    inline Real trace(const TensorValue& m)
    {
        return std::visit(overloaded {
                              [](const Matrix3& mat) -> Real { return mat.trace(); },
                              [](Real r) -> Real { return r; },
                              [](auto&&) -> Real {
                                  MPFEM_THROW(ArgumentException, "trace requires Matrix3 or Scalar");
                              }},
            m.variant());
    }

    inline TensorValue transpose(const TensorValue& m)
    {
        return std::visit(overloaded {
                              [](const Matrix3& mat) -> TensorValue {
                                  return TensorValue(Matrix3(mat.transpose().eval()));
                              },
                              [](auto&&) -> TensorValue {
                                  MPFEM_THROW(ArgumentException, "transpose requires Matrix3");
                              }},
            m.variant());
    }

    inline TensorValue sym(const TensorValue& m)
    {
        return std::visit(overloaded {
                              [](const Matrix3& mat) -> TensorValue {
                                  return TensorValue(Matrix3((Real(0.5) * (mat + mat.transpose())).eval()));
                              },
                              [](auto&&) -> TensorValue {
                                  MPFEM_THROW(ArgumentException, "sym requires Matrix3");
                              }},
            m.variant());
    }

    inline Real norm(const TensorValue& t)
    {
        return std::visit(overloaded {
                              [](const Real& x) -> Real { return std::abs(x); },
                              [](const Vector3& v) -> Real { return v.norm(); },
                              [](const Matrix3& m) -> Real { return m.norm(); }},
            t.variant());
    }

} // namespace mpfem

#endif // MPFEM_TENSOR_VALUE_HPP
