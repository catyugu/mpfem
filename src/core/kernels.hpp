#ifndef MPFEM_KERNELS_HPP
#define MPFEM_KERNELS_HPP

#include "core/types.hpp"

// 强制内联宏
#if defined(__GNUC__) || defined(__clang__)
#define MPFEM_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define MPFEM_ALWAYS_INLINE __forceinline
#else
#define MPFEM_ALWAYS_INLINE inline
#endif

namespace mpfem {
namespace kernels {

// =============================================================================
// 行列式计算 - 维度特化，完全展开
// =============================================================================

/// 1x1 行列式
MPFEM_ALWAYS_INLINE
Real det1(const Real* d) {
    return d[0];
}

/// 2x2 行列式: d[0]*d[3] - d[1]*d[2]
MPFEM_ALWAYS_INLINE
Real det2(const Real* d) {
    return d[0] * d[3] - d[1] * d[2];
}

/// 3x3 行列式 - 完全展开
MPFEM_ALWAYS_INLINE
Real det3(const Real* d) {
    return d[0] * (d[4] * d[8] - d[5] * d[7]) -
           d[3] * (d[1] * d[8] - d[2] * d[7]) +
           d[6] * (d[1] * d[5] - d[2] * d[4]);
}

/// 模板版本 - 编译期维度选择
template <int dim>
MPFEM_ALWAYS_INLINE
Real det(const Real* d);

template <>
MPFEM_ALWAYS_INLINE
Real det<1>(const Real* d) { return det1(d); }

template <>
MPFEM_ALWAYS_INLINE
Real det<2>(const Real* d) { return det2(d); }

template <>
MPFEM_ALWAYS_INLINE
Real det<3>(const Real* d) { return det3(d); }

// =============================================================================
// 伴随矩阵计算
// =============================================================================

/// 2x2 伴随矩阵
MPFEM_ALWAYS_INLINE
void adjugate2(const Real* d, Real* adj) {
    adj[0] =  d[3];
    adj[1] = -d[1];
    adj[2] = -d[2];
    adj[3] =  d[0];
}

/// 3x3 伴随矩阵 - 完全展开
MPFEM_ALWAYS_INLINE
void adjugate3(const Real* d, Real* adj) {
    adj[0] = d[4] * d[8] - d[5] * d[7];
    adj[1] = d[2] * d[7] - d[1] * d[8];
    adj[2] = d[1] * d[5] - d[2] * d[4];
    adj[3] = d[5] * d[6] - d[3] * d[8];
    adj[4] = d[0] * d[8] - d[2] * d[6];
    adj[5] = d[2] * d[3] - d[0] * d[5];
    adj[6] = d[3] * d[7] - d[4] * d[6];
    adj[7] = d[1] * d[6] - d[0] * d[7];
    adj[8] = d[0] * d[4] - d[1] * d[3];
}

// =============================================================================
// 逆矩阵计算
// =============================================================================

/// 2x2 逆矩阵
MPFEM_ALWAYS_INLINE
void inverse2(const Real* d, Real* inv) {
    const Real det = det2(d);
    const Real invDet = 1.0 / det;
    inv[0] =  d[3] * invDet;
    inv[1] = -d[1] * invDet;
    inv[2] = -d[2] * invDet;
    inv[3] =  d[0] * invDet;
}

/// 3x3 逆矩阵
MPFEM_ALWAYS_INLINE
void inverse3(const Real* d, Real* inv) {
    Real adj[9];
    adjugate3(d, adj);
    const Real det = det3(d);
    const Real invDet = 1.0 / det;
    for (int i = 0; i < 9; ++i) {
        inv[i] = adj[i] * invDet;
    }
}

// =============================================================================
// 矩阵向量乘法 - 小维度特化
// =============================================================================

/// 2x2 矩阵乘以 2x1 向量: y = A * x
MPFEM_ALWAYS_INLINE
void matvec2(const Real* A, const Real* x, Real* y) {
    y[0] = A[0] * x[0] + A[2] * x[1];
    y[1] = A[1] * x[0] + A[3] * x[1];
}

/// 3x3 矩阵乘以 3x1 向量: y = A * x
MPFEM_ALWAYS_INLINE
void matvec3(const Real* A, const Real* x, Real* y) {
    y[0] = A[0] * x[0] + A[3] * x[1] + A[6] * x[2];
    y[1] = A[1] * x[0] + A[4] * x[1] + A[7] * x[2];
    y[2] = A[2] * x[0] + A[5] * x[1] + A[8] * x[2];
}

// =============================================================================
// 转置矩阵向量乘法: y = A^T * x
// =============================================================================

/// 2x2 转置矩阵乘以 2x1 向量
MPFEM_ALWAYS_INLINE
void matvecT2(const Real* A, const Real* x, Real* y) {
    y[0] = A[0] * x[0] + A[1] * x[1];
    y[1] = A[2] * x[0] + A[3] * x[1];
}

/// 3x3 转置矩阵乘以 3x1 向量
MPFEM_ALWAYS_INLINE
void matvecT3(const Real* A, const Real* x, Real* y) {
    y[0] = A[0] * x[0] + A[1] * x[1] + A[2] * x[2];
    y[1] = A[3] * x[0] + A[4] * x[1] + A[5] * x[2];
    y[2] = A[6] * x[0] + A[7] * x[1] + A[8] * x[2];
}

// =============================================================================
// 点积
// =============================================================================

MPFEM_ALWAYS_INLINE
Real dot2(const Real* a, const Real* b) {
    return a[0] * b[0] + a[1] * b[1];
}

MPFEM_ALWAYS_INLINE
Real dot3(const Real* a, const Real* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// =============================================================================
// 向量范数
// =============================================================================

MPFEM_ALWAYS_INLINE
Real norm2sq2(const Real* a) {
    return a[0] * a[0] + a[1] * a[1];
}

MPFEM_ALWAYS_INLINE
Real norm2sq3(const Real* a) {
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}

}  // namespace kernels
}  // namespace mpfem

#endif  // MPFEM_KERNELS_HPP
