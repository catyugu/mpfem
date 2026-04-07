#ifndef MPFEM_TENSOR_VALUE_HPP
#define MPFEM_TENSOR_VALUE_HPP

#include "core/tensor_shape.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <array>

namespace mpfem {

    /**
     * @brief Unified tensor value using fixed-size stack storage
     *
     * Design principles:
     * - NO std::vector (eliminates ALL heap allocations)
     * - Fixed-size std::array<Real, 9> for maximum 3x3 matrices
     * - Scalar/Vector/Matrix distinguished by SHAPE, not type tag
     * - Multi-index helpers for accessing elements
     *
     * Storage layout (all in stack, no heap):
     * - Scalar: buffer[0], shape = {}
     * - Vector: buffer[0..2], shape = {3}
     * - Matrix: buffer[0..8], shape = {3,3}
     */
    class TensorValue {
    public:
        // Default: scalar zero
        TensorValue() : shape_(TensorShape::scalar()) { buffer_.fill(Real(0)); }

        // Scalar constructor
        explicit TensorValue(Real scalar) : shape_(TensorShape::scalar())
        {
            buffer_.fill(Real(0));
            buffer_[0] = scalar;
        }

        // Vector constructor (fixed size 3)
        explicit TensorValue(const Vector3& vec) : shape_(TensorShape::vector(3))
        {
            buffer_[0] = vec[0];
            buffer_[1] = vec[1];
            buffer_[2] = vec[2];
        }

        // Matrix constructor (fixed 3x3)
        explicit TensorValue(const Matrix3& mat) : shape_(TensorShape::matrix(3, 3))
        {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    buffer_[r * 3 + c] = mat(r, c);
                }
            }
        }

        // -------------------------------------------------------------------------
        // Shape query
        // -------------------------------------------------------------------------
        const TensorShape& shape() const { return shape_; }
        bool isScalar() const { return shape_.isScalar(); }
        bool isVector() const { return shape_.isVector(); }
        bool isMatrix() const { return shape_.isMatrix(); }

        // -------------------------------------------------------------------------
        // Direct buffer access (returns pointer to stack buffer)
        // -------------------------------------------------------------------------
        Real* data() { return buffer_.data(); }
        const Real* data() const { return buffer_.data(); }
        size_t size() const { return shape_.size(); }

        // -------------------------------------------------------------------------
        // Element access
        // -------------------------------------------------------------------------
        Real& scalar() { return buffer_[0]; }
        Real scalar() const { return buffer_[0]; }

        Real& operator[](int i) { return buffer_[i]; }
        Real operator[](int i) const { return buffer_[i]; }

        Real& at(int row, int col) { return buffer_[row * shape_.cols() + col]; }
        Real at(int row, int col) const { return buffer_[row * shape_.cols() + col]; }

        // -------------------------------------------------------------------------
        // Factory methods
        // -------------------------------------------------------------------------
        static TensorValue scalar(Real v) { return TensorValue(v); }

        static TensorValue vector(Real x, Real y, Real z)
        {
            TensorValue tv;
            tv.buffer_[0] = x;
            tv.buffer_[1] = y;
            tv.buffer_[2] = z;
            tv.shape_ = TensorShape::vector(3);
            return tv;
        }

        static TensorValue vector3(const Vector3& v) { return TensorValue(v); }

        static TensorValue matrix3(Real m00, Real m01, Real m02,
            Real m10, Real m11, Real m12,
            Real m20, Real m21, Real m22)
        {
            TensorValue tv;
            tv.buffer_[0] = m00;
            tv.buffer_[1] = m01;
            tv.buffer_[2] = m02;
            tv.buffer_[3] = m10;
            tv.buffer_[4] = m11;
            tv.buffer_[5] = m12;
            tv.buffer_[6] = m20;
            tv.buffer_[7] = m21;
            tv.buffer_[8] = m22;
            tv.shape_ = TensorShape::matrix(3, 3);
            return tv;
        }

        static TensorValue matrix3(const Matrix3& m) { return TensorValue(m); }

        static TensorValue identity3()
        {
            return matrix3(1, 0, 0, 0, 1, 0, 0, 0, 1);
        }

        static TensorValue zero(const TensorShape& shape)
        {
            TensorValue tv;
            tv.shape_ = shape;
            tv.buffer_.fill(Real(0));
            return tv;
        }

        // -------------------------------------------------------------------------
        // Extractors - convert back to Eigen types
        // -------------------------------------------------------------------------
        Real toScalar() const { return buffer_[0]; }

        Vector3 toVector3() const
        {
            return Vector3(buffer_[0], buffer_[1], buffer_[2]);
        }

        Matrix3 toMatrix3() const
        {
            Matrix3 m;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    m(r, c) = buffer_[r * 3 + c];
                }
            }
            return m;
        }

        void copyTo(Real* dest) const
        {
            for (size_t i = 0; i < size(); ++i)
                dest[i] = buffer_[i];
        }

    private:
        std::array<Real, 9> buffer_ {};
        TensorShape shape_;
    };

    // =============================================================================
    // Tensor operations - free functions for expression evaluation
    // All operations use only public interface (data(), size(), shape(), etc.)
    // =============================================================================

    // Scale tensor by scalar
    inline TensorValue scale(const TensorValue& t, Real s)
    {
        TensorValue result = t;
        for (size_t i = 0; i < t.size(); ++i)
            result[i] *= s;
        return result;
    }

    // Add tensors (same shape)
    inline TensorValue add(const TensorValue& a, const TensorValue& b)
    {
        TensorValue result = a;
        for (size_t i = 0; i < a.size(); ++i)
            result[i] += b[i];
        return result;
    }

    // Subtract tensors
    inline TensorValue subtract(const TensorValue& a, const TensorValue& b)
    {
        TensorValue result = a;
        for (size_t i = 0; i < a.size(); ++i)
            result[i] -= b[i];
        return result;
    }

    // Negate tensor
    inline TensorValue negate(const TensorValue& t)
    {
        TensorValue result = t;
        for (size_t i = 0; i < t.size(); ++i)
            result[i] = -result[i];
        return result;
    }

    // Matrix-vector product
    inline TensorValue matvec(const TensorValue& A, const TensorValue& b)
    {
        TensorValue result = TensorValue::zero(TensorShape::vector(3));
        for (int r_ = 0; r_ < 3; ++r_) {
            for (int c = 0; c < 3; ++c) {
                result[r_] += A.at(r_, c) * b[c];
            }
        }
        return result;
    }

    // Matrix-matrix product
    inline TensorValue matmat(const TensorValue& A, const TensorValue& B)
    {
        TensorValue result = TensorValue::zero(TensorShape::matrix(3, 3));
        for (int r_ = 0; r_ < 3; ++r_) {
            for (int c = 0; c < 3; ++c) {
                Real sum = 0;
                for (int k = 0; k < 3; ++k)
                    sum += A.at(r_, k) * B.at(k, c);
                result.at(r_, c) = sum;
            }
        }
        return result;
    }

    // Dot product of vectors
    inline Real dot(const TensorValue& a, const TensorValue& b)
    {
        Real sum = 0;
        for (size_t i = 0; i < a.size(); ++i)
            sum += a[i] * b[i];
        return sum;
    }

    // Vector cross product
    inline TensorValue cross(const TensorValue& a, const TensorValue& b)
    {
        return TensorValue::vector(
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]);
    }

    // Trace of matrix
    inline Real trace(const TensorValue& m)
    {
        return m.at(0, 0) + m.at(1, 1) + m.at(2, 2);
    }

    // Transpose of 3x3 matrix
    inline TensorValue transpose(const TensorValue& m)
    {
        TensorValue result = TensorValue::zero(TensorShape::matrix(3, 3));
        for (int r_ = 0; r_ < 3; ++r_) {
            for (int c = 0; c < 3; ++c) {
                result.at(r_, c) = m.at(c, r_);
            }
        }
        return result;
    }

    // Symmetric part: 0.5 * (m + m^T)
    inline TensorValue sym(const TensorValue& m)
    {
        TensorValue result = TensorValue::zero(TensorShape::matrix(3, 3));
        for (int r_ = 0; r_ < 3; ++r_) {
            for (int c = 0; c < 3; ++c) {
                result.at(r_, c) = Real(0.5) * (m.at(r_, c) + m.at(c, r_));
            }
        }
        return result;
    }

    // Frobenius norm
    inline Real norm(const TensorValue& t)
    {
        Real sum = 0;
        for (size_t i = 0; i < t.size(); ++i)
            sum += t[i] * t[i];
        return std::sqrt(sum);
    }

} // namespace mpfem

#endif // MPFEM_TENSOR_VALUE_HPP
