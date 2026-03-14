#ifndef MPFEM_ASSEMBLER_HPP
#define MPFEM_ASSEMBLER_HPP

#include "assembly/integrator.hpp"
#include "fe/fe_space.hpp"
#include "solver/sparse_matrix.hpp"
#include "core/logger.hpp"
#include <vector>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mpfem {

// =============================================================================
// 编译期常量
// =============================================================================

constexpr int MAX_DOFS = 27;      // 二阶六面体
constexpr int MAX_DIM = 3;

// =============================================================================
// 线程本地缓冲区（零分配）
// =============================================================================

struct alignas(64) ThreadBuffer {
    Eigen::Matrix<Real, MAX_DOFS, MAX_DOFS, Eigen::RowMajor> elmatScalar;
    Eigen::Matrix<Real, MAX_DOFS * 3, MAX_DOFS * 3, Eigen::RowMajor> elmatVector;
    Eigen::Matrix<Real, MAX_DOFS, 1> elvecScalar;
    Eigen::Matrix<Real, MAX_DOFS * 3, 1> elvecVector;
    std::array<Index, MAX_DOFS * 3> dofs;
    int numDofs = 0;
};

// =============================================================================
// 双线性型组装器
// =============================================================================

class BilinearFormAssembler {
public:
    explicit BilinearFormAssembler(const FESpace* fes);
    
    void addDomainIntegrator(std::unique_ptr<DomainBilinearIntegrator> integ) {
        domainIntegs_.push_back(std::move(integ));
    }
    void addBoundaryIntegrator(std::unique_ptr<FaceBilinearIntegrator> integ, int bid = -1) {
        bdrIntegs_.push_back(std::move(integ));
        bdrIds_.push_back(bid);
    }
    
    void clearIntegrators() { domainIntegs_.clear(); bdrIntegs_.clear(); bdrIds_.clear(); }
    void clear() { mat_.setZero(); }
    
    void computeSparsityPattern();
    void assemble();
    void finalize() { mat_.makeCompressed(); }
    
    SparseMatrix& matrix() { return mat_; }
    Index rows() const { return mat_.rows(); }
    
private:
    /// 扩展标量矩阵到向量场对角块
    void expandScalarToVector(const Matrix& scalarMat, Matrix& vectorMat, int nd, int vdim);
    
    const FESpace* fes_;
    std::vector<std::unique_ptr<DomainBilinearIntegrator>> domainIntegs_;
    std::vector<std::unique_ptr<FaceBilinearIntegrator>> bdrIntegs_;
    std::vector<int> bdrIds_;
    SparseMatrix mat_;
    std::vector<ThreadBuffer> buffers_;
    std::vector<SparseMatrix::Triplet> triplets_;
};

// =============================================================================
// 线性型组装器
// =============================================================================

class LinearFormAssembler {
public:
    explicit LinearFormAssembler(const FESpace* fes);
    
    void addDomainIntegrator(std::unique_ptr<DomainLinearIntegrator> integ) {
        domainIntegs_.push_back(std::move(integ));
    }
    void addBoundaryIntegrator(std::unique_ptr<FaceLinearIntegrator> integ, int bid = -1) {
        bdrIntegs_.push_back(std::move(integ));
        bdrIds_.push_back(bid);
    }
    
    void clearIntegrators() { domainIntegs_.clear(); bdrIntegs_.clear(); bdrIds_.clear(); }
    void clear() { vec_.setZero(); }
    
    void assemble();
    
    Vector& vector() { return vec_; }
    
private:
    /// 扩展标量向量到向量场
    void expandScalarToVector(const Vector& scalarVec, Vector& vectorVec, int nd, int vdim);
    
    const FESpace* fes_;
    std::vector<std::unique_ptr<DomainLinearIntegrator>> domainIntegs_;
    std::vector<std::unique_ptr<FaceLinearIntegrator>> bdrIntegs_;
    std::vector<int> bdrIds_;
    Vector vec_;
    std::vector<ThreadBuffer> buffers_;
    std::vector<Vector> threadVectors_;
};

}  // namespace mpfem

#endif  // MPFEM_ASSEMBLER_HPP