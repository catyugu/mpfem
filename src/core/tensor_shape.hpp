#ifndef MPFEM_TENSOR_SHAPE_HPP
#define MPFEM_TENSOR_SHAPE_HPP

#include "core/types.hpp"
#include <algorithm>
#include <cstddef>
#include <vector>

namespace mpfem {

    /**
     * @brief通用张量形状描述
     *
     * 支持任意维度的张量：
     * - 标量: dims = {} 或 dims = {1}
     * - 向量: dims = {n}
     * - 矩阵: dims = {3, 3}
     * - 高阶张量: dims = {a, b, c, ...}
     */
    struct TensorShape {
        std::vector<int> dims;

        TensorShape() = default;

        explicit TensorShape(std::vector<int> dims_) : dims(std::move(dims_)) { }

        // 工厂方法
        static TensorShape scalar() { return TensorShape(); }
        static TensorShape vector(int n) { return TensorShape({n}); }
        static TensorShape matrix(int rows, int cols) { return TensorShape({rows, cols}); }
        static TensorShape fromEnum(VariableShape shape)
        {
            switch (shape) {
            case VariableShape::Scalar:
                return scalar();
            case VariableShape::Vector:
                return vector(3);
            case VariableShape::Matrix:
                return matrix(3, 3);
            }
            return scalar();
        }

        /// 计算总元素数量
        size_t size() const
        {
            if (dims.empty())
                return 1;
            size_t s = 1;
            for (int d : dims)
                s *= static_cast<size_t>(d);
            return s;
        }

        /// 是否是标量
        bool isScalar() const { return dims.empty() || (dims.size() == 1 && dims[0] == 1); }

        /// 是否是向量
        bool isVector() const { return dims.size() == 1 && dims[0] > 1; }

        /// 是否是矩阵
        bool isMatrix() const { return dims.size() == 2; }

        /// 获取维度数
        int numDimensions() const { return static_cast<int>(dims.size()); }

        /// 获取第 i 维的大小
        int dimension(int i) const { return dims.at(i); }

        /// 获取行数（用于矩阵）
        int rows() const { return dims.size() >= 1 ? dims[0] : 1; }

        /// 获取列数（用于矩阵）
        int cols() const { return dims.size() >= 2 ? dims[1] : 1; }

        // 比较运算符
        bool operator==(const TensorShape& other) const { return dims == other.dims; }
        bool operator!=(const TensorShape& other) const { return dims != other.dims; }
    };

} // namespace mpfem

#endif // MPFEM_TENSOR_SHAPE_HPP
