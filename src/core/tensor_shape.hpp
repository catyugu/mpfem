#ifndef MPFEM_TENSOR_SHAPE_HPP
#define MPFEM_TENSOR_SHAPE_HPP

#include "core/types.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace mpfem {

    /**
     * @brief 通用张量形状描述
     *
     * 支持任意维度的张量：
     * - 标量: dims = {} 或 dims = {1}
     * - 向量: dims = {n}
     * - 矩阵: dims = {3, 3}
     * - 高阶张量: dims = {a, b, c, ...}
     */
    struct TensorShape {
        std::array<int, 4> dims{0, 0, 0, 0};
        int num_dims = 0;

        TensorShape() = default;

        explicit TensorShape(std::initializer_list<int> d)
        {
            num_dims = static_cast<int>(std::min<size_t>(d.size(), 4));
            int i = 0;
            for (auto val : d) {
                if (i < 4) dims[i++] = val;
            }
        }

        explicit TensorShape(const std::vector<int>& d)
        {
            num_dims = static_cast<int>(std::min<size_t>(d.size(), 4));
            for (int i = 0; i < num_dims; ++i) {
                dims[i] = d[i];
            }
        }

        // 工厂方法
        static TensorShape scalar() { return TensorShape(); }
        static TensorShape vector(int n) { return TensorShape({n}); }
        static TensorShape matrix(int rows, int cols) { return TensorShape({rows, cols}); }

        /// 计算总元素数量
        size_t size() const
        {
            if (num_dims == 0)
                return 1;
            size_t s = 1;
            for (int i = 0; i < num_dims; ++i)
                s *= static_cast<size_t>(dims[i]);
            return s;
        }

        /// 是否是标量
        bool isScalar() const { return num_dims == 0 || (num_dims == 1 && dims[0] == 1); }

        /// 是否是向量
        bool isVector() const { return num_dims == 1 && dims[0] > 1; }

        /// 是否是矩阵
        bool isMatrix() const { return num_dims == 2; }

        /// 获取维度数
        int numDimensions() const { return num_dims; }

        /// 获取第 i 维的大小
        int dimension(int i) const { return dims[i]; }

        /// 获取行数（用于矩阵）
        int rows() const { return num_dims >= 1 ? dims[0] : 1; }

        /// 获取列数（用于矩阵）
        int cols() const { return num_dims >= 2 ? dims[1] : 1; }

        // 比较运算符
        bool operator==(const TensorShape& other) const
        {
            if (num_dims != other.num_dims) return false;
            for (int i = 0; i < num_dims; ++i) {
                if (dims[i] != other.dims[i]) return false;
            }
            return true;
        }
        bool operator!=(const TensorShape& other) const { return !(*this == other); }
    };

} // namespace mpfem

#endif // MPFEM_TENSOR_SHAPE_HPP
