#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace mpfem {

// 动态大小向量
using Vector = Eigen::VectorXd;
using VectorXd = Eigen::VectorXd;  // 兼容别名

// 动态大小稠密矩阵
using DenseMatrix = Eigen::MatrixXd;
using MatXd = Eigen::MatrixXd;  // 兼容别名
using MatrixXd = Eigen::MatrixXd;  // 兼容别名

// 稀疏矩阵 (CSR 格式)
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

// 三重点用于稀疏矩阵构建
using Triplet = Eigen::Triplet<double>;

// 固定大小向量
template <int N>
using Vec = Eigen::Matrix<double, N, 1>;

using Vec2 = Vec<2>;
using Vec3 = Vec<3>;
using Vec4 = Vec<4>;
using Vec6 = Vec<6>;

// 固定大小矩阵
template <int M, int N>
using Mat = Eigen::Matrix<double, M, N>;

using Mat2 = Mat<2, 2>;
using Mat3 = Mat<3, 3>;
using Mat4 = Mat<4, 4>;
using Mat6 = Mat<6, 6>;
using Mat3x4 = Mat<3, 4>;
using Mat3x6 = Mat<3, 6>;
using Mat6x3 = Mat<6, 3>;

}  // namespace mpfem
