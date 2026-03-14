#include "integrators.hpp"

namespace mpfem {

void DiffusionIntegrator::assembleElementMatrix(const ReferenceElement& ref,
                                                 ElementTransform& trans,
                                                 Matrix& elmat,
                                                 int vdim) const {
    const int nd = ref.numDofs();
    const int dim = ref.dim();
    const int nq = ref.numQuadraturePoints();
    
    elmat.setZero(nd * vdim, nd * vdim);
    Eigen::MatrixXd gradMat(nd, dim);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Real coef = evalCoef(trans);
        
        const Vector3* refGrads = ref.shapeGradientsAtQuad(q);
        
        for (int i = 0; i < nd; ++i) {
            Vector3 physGrad;
            trans.transformGradient(refGrads[i].data(), physGrad.data());
            for (int d = 0; d < dim; ++d)
                gradMat(i, d) = physGrad[d];
        }
        
        // 标量场：直接组装
        // 向量场：对每个分量组装对角块
        if (vdim == 1) {
            elmat.noalias() += w * coef * (gradMat * gradMat.transpose());
        } else {
            for (int c = 0; c < vdim; ++c) {
                elmat.block(c * nd, c * nd, nd, nd).noalias() += 
                    w * coef * (gradMat * gradMat.transpose());
            }
        }
    }
}

void MassIntegrator::assembleElementMatrix(const ReferenceElement& ref,
                                            ElementTransform& trans,
                                            Matrix& elmat,
                                            int vdim) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    
    elmat.setZero(nd * vdim, nd * vdim);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Real coef = evalCoef(trans);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        if (vdim == 1) {
            elmat.noalias() += w * coef * (phiMap * phiMap.transpose());
        } else {
            for (int c = 0; c < vdim; ++c) {
                elmat.block(c * nd, c * nd, nd, nd).noalias() += 
                    w * coef * (phiMap * phiMap.transpose());
            }
        }
    }
}

void DomainLFIntegrator::assembleElementVector(const ReferenceElement& ref,
                                                ElementTransform& trans,
                                                Vector& elvec,
                                                int vdim) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    
    elvec.setZero(nd * vdim);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Real f = evalCoef(trans);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        if (vdim == 1) {
            elvec.noalias() += w * f * phiMap;
        } else {
            for (int c = 0; c < vdim; ++c) {
                elvec.segment(c * nd, nd).noalias() += w * f * phiMap;
            }
        }
    }
}

void BoundaryLFIntegrator::assembleFaceVector(const ReferenceElement& ref,
                                               FacetElementTransform& trans,
                                               Vector& elvec,
                                               int vdim) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    
    elvec.setZero(nd * vdim);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Real g = evalCoef(trans);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        if (vdim == 1) {
            elvec.noalias() += w * g * phiMap;
        } else {
            for (int c = 0; c < vdim; ++c) {
                elvec.segment(c * nd, nd).noalias() += w * g * phiMap;
            }
        }
    }
}

void ConvectionMassIntegrator::assembleFaceMatrix(const ReferenceElement& ref,
                                                   FacetElementTransform& trans,
                                                   Matrix& elmat,
                                                   int vdim) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    
    elmat.setZero(nd * vdim, nd * vdim);
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Real h = evalCoef(trans);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        if (vdim == 1) {
            elmat.noalias() += w * h * (phiMap * phiMap.transpose());
        } else {
            for (int c = 0; c < vdim; ++c) {
                elmat.block(c * nd, c * nd, nd, nd).noalias() += 
                    w * h * (phiMap * phiMap.transpose());
            }
        }
    }
}

void ConvectionLFIntegrator::assembleFaceVector(const ReferenceElement& ref,
                                                 FacetElementTransform& trans,
                                                 Vector& elvec,
                                                 int vdim) const {
    const int nd = ref.numDofs();
    const int nq = ref.numQuadraturePoints();
    
    elvec.setZero(nd * vdim);
    
    if (!Tinf_) return;
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Real h = evalCoef(trans);
        const Real Tinf = Tinf_->eval(trans);
        
        const Real* phi = ref.shapeValuesAtQuad(q);
        Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);
        
        if (vdim == 1) {
            elvec.noalias() += w * h * Tinf * phiMap;
        } else {
            for (int c = 0; c < vdim; ++c) {
                elvec.segment(c * nd, nd).noalias() += w * h * Tinf * phiMap;
            }
        }
    }
}

void ElasticityIntegrator::assembleElementMatrix(const ReferenceElement& ref,
                                                  ElementTransform& trans,
                                                  Matrix& elmat,
                                                  int vdim) const {
    const int nd = ref.numDofs();
    const int dim = ref.dim();
    const int nq = ref.numQuadraturePoints();
    
    elmat.setZero(nd * vdim, nd * vdim);
    
    // 获取材料属性
    Real E = E_ ? E_->eval(trans) : 1.0;
    Real nu = nu_ ? nu_->eval(trans) : 0.3;
    
    // Lame参数
    Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Real mu = E / (2.0 * (1.0 + nu));
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        const Vector3* refGrads = ref.shapeGradientsAtQuad(q);
        
        // 计算所有形函数的物理梯度
        // B矩阵: 对于每个节点a, B_a是6x3的应变-位移矩阵
        // epsilon = [du/dx, dv/dy, dw/dz, dv/dz+dw/dy, du/dz+dw/dx, du/dy+dv/dx]
        Eigen::MatrixXd B(6, nd * vdim);
        B.setZero();
        
        for (int a = 0; a < nd; ++a) {
            Vector3 physGrad;
            trans.transformGradient(refGrads[a].data(), physGrad.data());
            
            // B矩阵的第a个节点的贡献
            // 对于位移分量 i (0=x, 1=y, 2=z):
            int col = a * vdim;
            
            // du/dx 贡献到 epsilon_11
            B(0, col + 0) = physGrad[0];
            // dv/dy 贡献到 epsilon_22
            B(1, col + 1) = physGrad[1];
            // dw/dz 贡献到 epsilon_33
            B(2, col + 2) = physGrad[2];
            // dv/dz + dw/dy 贡献到 epsilon_23
            B(3, col + 1) = physGrad[2];
            B(3, col + 2) = physGrad[1];
            // du/dz + dw/dx 贡献到 epsilon_13
            B(4, col + 0) = physGrad[2];
            B(4, col + 2) = physGrad[0];
            // du/dy + dv/dx 贡献到 epsilon_12
            B(5, col + 0) = physGrad[1];
            B(5, col + 1) = physGrad[0];
        }
        
        // 本构矩阵 C (各向同性)
        // sigma = C * epsilon
        // 对于平面应变和3D问题:
        // C_11 = C_22 = C_33 = lambda + 2*mu
        // C_12 = C_13 = C_23 = lambda
        // C_44 = C_55 = C_66 = mu
        Eigen::Matrix<Real, 6, 6> C;
        C.setZero();
        C(0, 0) = C(1, 1) = C(2, 2) = lambda + 2.0 * mu;
        C(0, 1) = C(0, 2) = C(1, 0) = C(1, 2) = C(2, 0) = C(2, 1) = lambda;
        C(3, 3) = C(4, 4) = C(5, 5) = mu;
        
        // 刚度矩阵: K = B^T * C * B
        elmat.noalias() += w * (B.transpose() * C * B);
    }
}

void ThermalLoadIntegrator::assembleElementVector(const ReferenceElement& ref,
                                                   ElementTransform& trans,
                                                   Vector& elvec,
                                                   int vdim) const {
    const int nd = ref.numDofs();
    const int dim = ref.dim();
    const int nq = ref.numQuadraturePoints();
    
    elvec.setZero(nd * vdim);
    
    if (!T_ || !alphaT_) return;
    
    for (int q = 0; q < nq; ++q) {
        const IntegrationPoint& ip = ref.integrationPoint(q);
        Real xi[3] = {ip.xi, ip.eta, ip.zeta};
        trans.setIntegrationPoint(xi);
        
        const Real w = ip.weight * trans.weight();
        
        // 获取材料属性
        Real E = E_ ? E_->eval(trans) : 1.0;
        Real nu = nu_ ? nu_->eval(trans) : 0.3;
        Real alpha = alphaT_->eval(trans);
        
        // 获取温度
        Real T_val = T_->eval(trans.elementIndex(), xi);
        Real dT = T_val - Tref_;
        
        // Lame参数
        Real lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Real mu = E / (2.0 * (1.0 + nu));
        
        // 热应变: epsilon_th = alpha * dT * I
        // 热应力对应的应变向量: [alpha*dT, alpha*dT, alpha*dT, 0, 0, 0]
        Eigen::Matrix<Real, 6, 1> epsilon_th;
        epsilon_th << alpha * dT, alpha * dT, alpha * dT, 0, 0, 0;
        
        // 热应力: sigma_th = C * epsilon_th
        // 对于各向同性材料: sigma_th = (3*lambda + 2*mu) * alpha * dT * I
        // 或者用体积模量表示: sigma_th = 3*K*alpha*dT*I, K = lambda + 2/3*mu
        Eigen::Matrix<Real, 6, 1> sigma_th;
        Real diag = (3.0 * lambda + 2.0 * mu) * alpha * dT;
        sigma_th << diag, diag, diag, 0, 0, 0;
        
        const Vector3* refGrads = ref.shapeGradientsAtQuad(q);
        
        // 计算B矩阵
        Eigen::MatrixXd B(6, nd * vdim);
        B.setZero();
        
        for (int a = 0; a < nd; ++a) {
            Vector3 physGrad;
            trans.transformGradient(refGrads[a].data(), physGrad.data());
            
            int col = a * vdim;
            B(0, col + 0) = physGrad[0];
            B(1, col + 1) = physGrad[1];
            B(2, col + 2) = physGrad[2];
            B(3, col + 1) = physGrad[2];
            B(3, col + 2) = physGrad[1];
            B(4, col + 0) = physGrad[2];
            B(4, col + 2) = physGrad[0];
            B(5, col + 0) = physGrad[1];
            B(5, col + 1) = physGrad[0];
        }
        
        // 热载荷向量: f = B^T * sigma_th
        // 注意符号: 热膨胀产生初始应变，在RHS中为负
        elvec.noalias() -= w * (B.transpose() * sigma_th);
    }
}

}  // namespace mpfem