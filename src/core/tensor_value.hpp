#ifndef MPFEM_TENSOR_VALUE_HPP
#define MPFEM_TENSOR_VALUE_HPP

#include "core/tensor_shape.hpp"
#include "core/types.hpp"

namespace mpfem {

    /**
     * @brief Unified tensor value using flat storage with shape-based indexing
     *
     * Design principles:
     * - NO std::variant (eliminates runtime type dispatch overhead)
     * - Flat Real buffer with shape metadata
     * - Scalar/Vector/Matrix distinguished by SHAPE, not type tag
     * - Multi-index helpers for accessing elements
     *
     * Storage layout:
     * - Scalar: buffer[0], shape = {}
     * - Vector: buffer[0..n-1], shape = {n}
     * - Matrix: buffer[0..rows*cols-1] in row-major order, shape = {rows, cols}
     */
    class TensorValue {
    public:
        // Default: scalar zero
        TensorValue() : buffer_(1, Real(0)), shape_(TensorShape::scalar()) { }

        // -------------------------------------------------------------------------
        // Constructors from raw data
        // -------------------------------------------------------------------------

        // Scalar constructor
        explicit TensorValue(Real scalar)
            : buffer_(1, scalar), shape_(TensorShape::scalar()) { }

        // Vector constructor (fixed size 3 for physics)
        explicit TensorValue(const Vector3& vec) : TensorValue()
        {
            buffer_.resize(3);
            buffer_[0] = vec[0];
            buffer_[1] = vec[1];
            buffer_[2] = vec[2];
            shape_ = TensorShape::vector(3);
        }

        // Matrix constructor (fixed 3x3 for physics)
        explicit TensorValue(const Matrix3& mat) : TensorValue()
        {
            buffer_.resize(9);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    buffer_[r * 3 + c] = mat(r, c);
                }
            }
            shape_ = TensorShape::matrix(3, 3);
        }

        // Construct from flat buffer with given shape
        TensorValue(const Real* data, size_t size, const TensorShape& shape)
            : buffer_(data, data + size), shape_(shape) { }

        // Construct from span with given shape
        TensorValue(std::span<const Real> data, const TensorShape& shape)
            : buffer_(data.begin(), data.end()), shape_(shape) { }

        // -------------------------------------------------------------------------
        // Extractors - convert back to Eigen types
        // -------------------------------------------------------------------------

        Real toScalar() const
        {
            return buffer_[0];
        }

        Vector3 toVector3() const
        {
            Vector3 v;
            v[0] = buffer_[0];
            v[1] = buffer_[1];
            v[2] = buffer_[2];
            return v;
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

        // Copy to existing buffer
        void copyTo(Real* dest) const
        {
            std::copy(buffer_.begin(), buffer_.end(), dest);
        }

        // -------------------------------------------------------------------------
        // Shape query
        // -------------------------------------------------------------------------
        const TensorShape& shape() const { return shape_; }

        // Convenience shape checks (inline, no branch)
        bool isScalar() const { return shape_.isScalar(); }
        bool isVector() const { return shape_.isVector(); }
        bool isMatrix() const { return shape_.isMatrix(); }

        // -------------------------------------------------------------------------
        // Direct buffer access
        // -------------------------------------------------------------------------
        Real* data() { return buffer_.data(); }
        const Real* data() const { return buffer_.data(); }
        size_t size() const { return buffer_.size(); }

        // Scalar access (only valid if isScalar())
        Real& scalar() { return buffer_[0]; }
        Real scalar() const { return buffer_[0]; }

        // Flat index access
        Real& flat(size_t i) { return buffer_[i]; }
        Real flat(size_t i) const { return buffer_[i]; }

        // Matrix multi-index access (row-major, only valid if isMatrix())
        Real& at(int row, int col) { return buffer_[row * shape_.cols() + col]; }
        Real at(int row, int col) const { return buffer_[row * shape_.cols() + col]; }

        // Vector element access (only valid if isVector())
        Real& operator[](int i) { return buffer_[i]; }
        Real operator[](int i) const { return buffer_[i]; }

        // -------------------------------------------------------------------------
        // Factory methods
        // -------------------------------------------------------------------------
        static TensorValue scalar(Real v) { return TensorValue(v); }

        static TensorValue vector(Real x, Real y, Real z)
        {
            TensorValue tv;
            tv.buffer_ = {x, y, z};
            tv.shape_ = TensorShape::vector(3);
            return tv;
        }

        static TensorValue vector3(const Vector3& v) { return TensorValue(v); }

        static TensorValue matrix3(Real m00, Real m01, Real m02,
            Real m10, Real m11, Real m12,
            Real m20, Real m21, Real m22)
        {
            TensorValue tv;
            tv.buffer_ = {m00, m01, m02, m10, m11, m12, m20, m21, m22};
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
            tv.buffer_.assign(shape.size(), Real(0));
            tv.shape_ = shape;
            return tv;
        }

    private:
        std::vector<Real> buffer_;
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
        Real* d = result.data();
        for (size_t i = 0; i < t.size(); ++i)
            d[i] *= s;
        return result;
    }

    // Add tensors (same shape)
    inline TensorValue add(const TensorValue& a, const TensorValue& b)
    {
        TensorValue result = a;
        Real* d = result.data();
        const Real* db = b.data();
        for (size_t i = 0; i < a.size(); ++i)
            d[i] += db[i];
        return result;
    }

    // Subtract tensors
    inline TensorValue subtract(const TensorValue& a, const TensorValue& b)
    {
        TensorValue result = a;
        Real* d = result.data();
        const Real* db = b.data();
        for (size_t i = 0; i < a.size(); ++i)
            d[i] -= db[i];
        return result;
    }

    // Negate tensor
    inline TensorValue negate(const TensorValue& t)
    {
        TensorValue result = t;
        Real* d = result.data();
        for (size_t i = 0; i < t.size(); ++i)
            d[i] = -d[i];
        return result;
    }

    // Matrix-vector product
    inline TensorValue matvec(const TensorValue& A, const TensorValue& b)
    {
        TensorValue result = TensorValue::zero(TensorShape::vector(3));
        Real* r = result.data();
        for (int r_ = 0; r_ < 3; ++r_) {
            for (int c = 0; c < 3; ++c) {
                r[r_] += A.at(r_, c) * b[c];
            }
        }
        return result;
    }

    // Matrix-matrix product
    inline TensorValue matmat(const TensorValue& A, const TensorValue& B)
    {
        TensorValue result = TensorValue::zero(TensorShape::matrix(3, 3));
        Real* r = result.data();
        for (int r_ = 0; r_ < 3; ++r_) {
            for (int c = 0; c < 3; ++c) {
                Real sum = 0;
                for (int k = 0; k < 3; ++k)
                    sum += A.at(r_, k) * B.at(k, c);
                r[r_ * 3 + c] = sum;
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
        const Real* d = t.data();
        for (size_t i = 0; i < t.size(); ++i)
            sum += d[i] * d[i];
        return std::sqrt(sum);
    }

} // namespace mpfem

#endif // MPFEM_TENSOR_VALUE_HPP
