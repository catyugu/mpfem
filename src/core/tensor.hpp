#ifndef MPFEM_TENSOR_HPP
#define MPFEM_TENSOR_HPP

#include "core/exception.hpp"
#include "core/tensor_shape.hpp"
#include "core/types.hpp"
#include <Eigen/Dense>

namespace mpfem {

    /**
     * @brief 最高支持 36 个元素的栈上零堆分配张量（例如 6x6 矩阵或 36维向量），超出则自动降级为堆分配
     * 
     * 内部存储顺序：Column-Major (与 Eigen 默认一致)
     */
    using TensorData = Eigen::Matrix<Real, Eigen::Dynamic, 1, Eigen::ColMajor, 36, 1>;

    class Tensor {
    public:
        TensorShape shape_;
        TensorData data_;

        Tensor() : shape_(TensorShape::scalar())
        {
            data_.setZero(1);
        }

        explicit Tensor(Real v) : shape_(TensorShape::scalar())
        {
            data_.setConstant(1, v);
        }

        // 支持任意形状初始化
        Tensor(TensorShape shape, TensorData data) : shape_(shape), data_(std::move(data)) { }

        // 生成工厂
        static Tensor scalar(Real v) { return Tensor(v); }

        static Tensor vector(const TensorData& v)
        {
            return Tensor(TensorShape::vector(static_cast<int>(v.size())), v);
        }

        static Tensor vector(std::initializer_list<Real> vals)
        {
            TensorData v(static_cast<Index>(vals.size()));
            int i = 0;
            for (auto val : vals) v[i++] = val;
            return vector(v);
        }

        static Tensor matrix(int r, int c, const TensorData& m)
        {
            MPFEM_ASSERT(m.size() == static_cast<Index>(r) * c, "Matrix data size mismatch");
            return Tensor(TensorShape::matrix(r, c), m);
        }

        template <typename Derived>
        static Tensor matrix(int r, int c, const Eigen::MatrixBase<Derived>& m)
        {
            static_assert(std::is_same_v<typename Derived::Scalar, Real>, "Matrix scalar type mismatch");
            MPFEM_ASSERT(m.rows() == r && m.cols() == c, "Matrix dimension mismatch");
            return Tensor(TensorShape::matrix(r, c), Eigen::Map<const TensorData>(m.derived().data(), m.size()));
        }

        /**
         * @brief 从行优先初始化列表创建矩阵 (转为内部列优先存储)
         */
        static Tensor matrix(int r, int c, std::initializer_list<Real> vals)
        {
            MPFEM_ASSERT(vals.size() == static_cast<size_t>(r * c), "Matrix data size mismatch");
            TensorData m(static_cast<Index>(vals.size()));
            auto it = vals.begin();
            for (int i = 0; i < r; ++i) {
                for (int j = 0; j < c; ++j) {
                    m[j * r + i] = *it++; // Row-major to Column-major
                }
            }
            return matrix(r, c, m);
        }

        static Tensor zero(const TensorShape& shape)
        {
            Tensor t;
            t.shape_ = shape;
            t.data_.setZero(static_cast<Index>(shape.size()));
            return t;
        }

        // Generic accessors returning Eigen::Map
        Eigen::Map<const Vector> vector() const
        {
            if (!shape_.isVector()) {
                MPFEM_THROW(ArgumentException, "Not a vector");
            }
            return Eigen::Map<const Vector>(data_.data(), data_.size());
        }

        Eigen::Map<const Matrix> matrix() const
        {
            if (!shape_.isMatrix()) {
                MPFEM_THROW(ArgumentException, "Not a matrix");
            }
            return Eigen::Map<const Matrix>(data_.data(), shape_.rows(), shape_.cols());
        }

        // Shape query
        const TensorShape& shape() const { return shape_; }
        bool isScalar() const { return shape_.isScalar(); }
        bool isVector() const { return shape_.isVector(); }
        bool isMatrix() const { return shape_.isMatrix(); }

        // Element access
        Real scalar() const { return data_[0]; }
        Real operator[](int i) const { return data_[i]; }
        Real& operator[](int i) { return data_[i]; }

        Real operator()(int r, int c) const
        {
            MPFEM_ASSERT(isMatrix(), "Not a matrix");
            return data_[c * shape_.rows() + r]; // Column-major indexing
        }

        // --- 统一的数学运算 ---

        Tensor operator+(const Tensor& rhs) const
        {
            MPFEM_ASSERT(shape_ == rhs.shape_, "Tensor shape mismatch in add");
            return Tensor(shape_, data_ + rhs.data_);
        }

        Tensor operator-(const Tensor& rhs) const
        {
            MPFEM_ASSERT(shape_ == rhs.shape_, "Tensor shape mismatch in sub");
            return Tensor(shape_, data_ - rhs.data_);
        }

        Tensor operator-() const { return Tensor(shape_, -data_); }

        Tensor operator*(const Tensor& rhs) const
        {
            if (shape_.isScalar()) return Tensor(rhs.shape_, data_[0] * rhs.data_);
            if (rhs.shape_.isScalar()) return Tensor(shape_, data_ * rhs.data_[0]);

            // 矩阵 * 向量
            if (shape_.isMatrix() && rhs.shape_.isVector()) {
                MPFEM_ASSERT(shape_.cols() == rhs.shape_.rows(), "Mat*Vec dim mismatch");
                Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat(data_.data(), shape_.rows(), shape_.cols());
                return Tensor(TensorShape::vector(shape_.rows()), mat * rhs.data_);
            }
            // 矩阵 * 矩阵
            if (shape_.isMatrix() && rhs.shape_.isMatrix()) {
                MPFEM_ASSERT(shape_.cols() == rhs.shape_.rows(), "Mat*Mat dim mismatch");
                Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> A(data_.data(), shape_.rows(), shape_.cols());
                Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> B(rhs.data_.data(), rhs.shape_.rows(), rhs.shape_.cols());
                Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> C = A * B;
                return Tensor(TensorShape::matrix(shape_.rows(), rhs.shape_.cols()),
                    Eigen::Map<TensorData>(C.data(), C.size()));
            }

            MPFEM_THROW(ArgumentException, "Invalid tensor multiplication shapes");
        }

        Tensor operator/(const Tensor& rhs) const
        {
            MPFEM_ASSERT(rhs.shape_.isScalar(), "Only divide by scalar supported");
            return Tensor(shape_, data_ / rhs.data_[0]);
        }
    };

    // Free functions
    inline Real dot(const Tensor& a, const Tensor& b)
    {
        MPFEM_ASSERT(a.shape_ == b.shape_, "Dot product requires matching shapes");
        return a.data_.dot(b.data_);
    }

    inline Tensor transpose(const Tensor& a)
    {
        if (!a.shape_.isMatrix()) return a;
        Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat(a.data_.data(), a.shape_.rows(), a.shape_.cols());
        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> trans = mat.transpose();
        return Tensor(TensorShape::matrix(a.shape_.cols(), a.shape_.rows()),
            Eigen::Map<TensorData>(trans.data(), trans.size()));
    }

    inline Real trace(const Tensor& m)
    {
        if (m.shape_.isScalar()) return m.scalar();
        MPFEM_ASSERT(m.shape_.isMatrix() && m.shape_.rows() == m.shape_.cols(), "Trace requires square matrix");
        Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat(m.data_.data(), m.shape_.rows(), m.shape_.cols());
        return mat.trace();
    }

    inline Tensor sym(const Tensor& m)
    {
        if (m.shape_.isScalar()) return m;
        MPFEM_ASSERT(m.shape_.isMatrix() && m.shape_.rows() == m.shape_.cols(), "Sym requires square matrix");
        Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat(m.data_.data(), m.shape_.rows(), m.shape_.cols());
        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> res = 0.5 * (mat + mat.transpose());
        return Tensor(m.shape_, Eigen::Map<TensorData>(res.data(), res.size()));
    }

    inline Real norm(const Tensor& t)
    {
        return t.data_.norm();
    }

} // namespace mpfem

#endif // MPFEM_TENSOR_HPP
