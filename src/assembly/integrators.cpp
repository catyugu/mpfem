#include "integrators.hpp"

#include <array>

namespace mpfem {

    namespace {

        EvaluationContext makeSinglePointContext(ElementTransform& trans,
            std::array<Vector3, 1>& refPts,
            std::array<Vector3, 1>& physPts,
            std::array<ElementTransform*, 1>& transforms)
        {
            const IntegrationPoint& ip = trans.integrationPoint();
            refPts[0] = Vector3(ip.xi, ip.eta, ip.zeta);
            trans.transform(ip, physPts[0]);
            transforms[0] = &trans;

            EvaluationContext ctx;
            ctx.domainId = static_cast<int>(trans.attribute());
            ctx.elementId = trans.elementIndex();
            ctx.referencePoints = std::span<const Vector3>(refPts.data(), refPts.size());
            ctx.physicalPoints = std::span<const Vector3>(physPts.data(), physPts.size());
            ctx.transforms = std::span<ElementTransform* const>(transforms.data(), transforms.size());
            return ctx;
        }

        Real evalScalarNode(const VariableNode* node, ElementTransform& trans)
        {
            if (!node) {
                MPFEM_THROW(ArgumentException, "Expected scalar variable node.");
            }
            std::array<Vector3, 1> refPts;
            std::array<Vector3, 1> physPts;
            std::array<ElementTransform*, 1> transforms {};
            std::array<Tensor, 1> value {};
            const EvaluationContext ctx = makeSinglePointContext(trans, refPts, physPts, transforms);
            node->evaluateBatch(ctx, std::span<Tensor>(value.data(), value.size()));
            return value[0].scalar();
        }

        Matrix3 evalMatrixNode(const VariableNode* node, ElementTransform& trans)
        {
            if (!node) {
                MPFEM_THROW(ArgumentException, "Expected matrix variable node.");
            }

            std::array<Vector3, 1> refPts;
            std::array<Vector3, 1> physPts;
            std::array<ElementTransform*, 1> transforms {};
            std::array<Tensor, 1> value {};
            const EvaluationContext ctx = makeSinglePointContext(trans, refPts, physPts, transforms);
            node->evaluateBatch(ctx, std::span<Tensor>(value.data(), value.size()));

            return value[0].toMatrix3();
        }

    } // namespace

    // =============================================================================
    // Diffusion integrator (matrix coefficient)
    // =============================================================================

    void DiffusionIntegrator::assembleElementMatrix(const ReferenceElement& ref,
        ElementTransform& trans,
        Matrix& elmat) const
    {
        const int nd = ref.numDofs();
        const int nq = ref.numQuadraturePoints();

        elmat.setZero(nd, nd);
        Eigen::Matrix<Real, MaxDofsPerElement, 3> gradMatFull;
        Eigen::Matrix<Real, MaxDofsPerElement, 3> dGradMatFull;
        auto gradMat = gradMatFull.topRows(nd);
        auto dGradMat = dGradMatFull.topRows(nd);

        for (int q = 0; q < nq; ++q) {
            const IntegrationPoint& ip = ref.integrationPoint(q);
            Real xi[3] = {ip.xi, ip.eta, ip.zeta};
            trans.setIntegrationPoint(xi);

            const Real w = ip.weight * trans.weight();
            const Matrix3 D = evalMatrixNode(coef_, trans);

            const Vector3* refGrads = ref.shapeGradientsAtQuad(q);

            for (int i = 0; i < nd; ++i) {
                Vector3 physGrad;
                trans.transformGradient(refGrads[i].data(), physGrad.data());
                gradMat.row(i) = physGrad;
            }

            dGradMat.noalias() = gradMat * D;
            elmat.noalias() += w * (gradMat * dGradMat.transpose());
        }
    }

    // =============================================================================
    // Mass integrator
    // =============================================================================

    void MassIntegrator::assembleElementMatrix(const ReferenceElement& ref,
        ElementTransform& trans,
        Matrix& elmat) const
    {
        const int nd = ref.numDofs();
        const int nq = ref.numQuadraturePoints();

        elmat.setZero(nd, nd);

        for (int q = 0; q < nq; ++q) {
            const IntegrationPoint& ip = ref.integrationPoint(q);
            Real xi[3] = {ip.xi, ip.eta, ip.zeta};
            trans.setIntegrationPoint(xi);

            const Real w = ip.weight * trans.weight();
            const Real coef = evalScalarNode(coef_, trans);

            const Real* phi = ref.shapeValuesAtQuad(q);
            Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);

            elmat.noalias() += w * coef * (phiMap * phiMap.transpose());
        }
    }

    // =============================================================================
    // Domain load integrator
    // =============================================================================

    void DomainLFIntegrator::assembleElementVector(const ReferenceElement& ref,
        ElementTransform& trans,
        Vector& elvec) const
    {
        const int nd = ref.numDofs();
        const int nq = ref.numQuadraturePoints();

        elvec.setZero(nd);

        for (int q = 0; q < nq; ++q) {
            const IntegrationPoint& ip = ref.integrationPoint(q);
            Real xi[3] = {ip.xi, ip.eta, ip.zeta};
            trans.setIntegrationPoint(xi);

            const Real w = ip.weight * trans.weight();
            const Real f = evalScalarNode(coef_, trans);

            const Real* phi = ref.shapeValuesAtQuad(q);
            Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);

            elvec.noalias() += w * f * phiMap;
        }
    }

    // =============================================================================
    // Boundary load integrator
    // =============================================================================

    void BoundaryLFIntegrator::assembleFaceVector(const ReferenceElement& ref,
        FacetElementTransform& trans,
        Vector& elvec) const
    {
        const int nd = ref.numDofs();
        const int nq = ref.numQuadraturePoints();

        elvec.setZero(nd);

        for (int q = 0; q < nq; ++q) {
            const IntegrationPoint& ip = ref.integrationPoint(q);
            Real xi[3] = {ip.xi, ip.eta, ip.zeta};
            trans.setIntegrationPoint(xi);

            const Real w = ip.weight * trans.weight();
            const Real g = evalScalarNode(coef_, trans);

            const Real* phi = ref.shapeValuesAtQuad(q);
            Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);

            elvec.noalias() += w * g * phiMap;
        }
    }

    // =============================================================================
    // Convection integrators (Robin BC)
    // =============================================================================

    void ConvectionMassIntegrator::assembleFaceMatrix(const ReferenceElement& ref,
        FacetElementTransform& trans,
        Matrix& elmat) const
    {
        const int nd = ref.numDofs();
        const int nq = ref.numQuadraturePoints();

        elmat.setZero(nd, nd);

        for (int q = 0; q < nq; ++q) {
            const IntegrationPoint& ip = ref.integrationPoint(q);
            Real xi[3] = {ip.xi, ip.eta, ip.zeta};
            trans.setIntegrationPoint(xi);

            const Real w = ip.weight * trans.weight();
            const Real h = evalScalarNode(coef_, trans);

            const Real* phi = ref.shapeValuesAtQuad(q);
            Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);

            elmat.noalias() += w * h * (phiMap * phiMap.transpose());
        }
    }

    void ConvectionLFIntegrator::assembleFaceVector(const ReferenceElement& ref,
        FacetElementTransform& trans,
        Vector& elvec) const
    {
        const int nd = ref.numDofs();
        const int nq = ref.numQuadraturePoints();

        elvec.setZero(nd);

        for (int q = 0; q < nq; ++q) {
            const IntegrationPoint& ip = ref.integrationPoint(q);
            Real xi[3] = {ip.xi, ip.eta, ip.zeta};
            trans.setIntegrationPoint(xi);

            const Real w = ip.weight * trans.weight();
            const Real h = evalScalarNode(coef_, trans);
            const Real Tinf = evalScalarNode(Tinf_, trans);

            const Real* phi = ref.shapeValuesAtQuad(q);
            Eigen::Map<const Eigen::VectorXd> phiMap(phi, nd);

            elvec.noalias() += w * h * Tinf * phiMap;
        }
    }

    // =============================================================================
    // Elasticity integrators
    // =============================================================================

    namespace {
        // Helper function to compute strain-displacement B matrix from shape function gradients
        template <typename BMatrix>
        inline void computeStrainDispMatrix(
            BMatrix& B,
            const ReferenceElement& ref,
            ElementTransform& trans,
            int nd,
            int vdim,
            int q)
        {
            const Vector3* refGrads = ref.shapeGradientsAtQuad(q);
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
        }

        // Helper function to fill elasticity tensor C from Lamé parameters
        inline void fillElasticityTensorC(Eigen::Matrix<Real, 6, 6>& C, Real lambda, Real mu)
        {
            C.setZero();
            C(0, 0) = C(1, 1) = C(2, 2) = lambda + 2.0 * mu;
            C(0, 1) = C(0, 2) = C(1, 0) = C(1, 2) = C(2, 0) = C(2, 1) = lambda;
            C(3, 3) = C(4, 4) = C(5, 5) = mu;
        }

    } // anonymous namespace

    void ElasticityIntegrator::assembleElementMatrix(const ReferenceElement& ref,
        ElementTransform& trans,
        Matrix& elmat) const
    {
        const int nd = ref.numDofs();
        const int nq = ref.numQuadraturePoints();
        const int totalDofs = nd * vdim_;

        elmat.setZero(totalDofs, totalDofs);

        const Real E_val = evalScalarNode(E_, trans);
        const Real nu_val = evalScalarNode(nu_, trans);

        Real lambda = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val));
        Real mu = E_val / (2.0 * (1.0 + nu_val));

        Eigen::Matrix<Real, 6, 6> C;
        fillElasticityTensorC(C, lambda, mu);

        Eigen::Matrix<Real, MaxStrainComponents, MaxVectorDofsPerElement> B_full;
        Eigen::Matrix<Real, MaxStrainComponents, MaxVectorDofsPerElement> CB_full;

        auto B = B_full.leftCols(totalDofs);
        auto CB = CB_full.leftCols(totalDofs);

        for (int q = 0; q < nq; ++q) {
            const IntegrationPoint& ip = ref.integrationPoint(q);
            Real xi[3] = {ip.xi, ip.eta, ip.zeta};
            trans.setIntegrationPoint(xi);

            const Real w = ip.weight * trans.weight();

            B.setZero();

            computeStrainDispMatrix(B, ref, trans, nd, vdim_, q);

            CB.noalias() = C * B;
            elmat.noalias() += w * (B.transpose() * CB);
        }
    }

    void StrainLoadIntegrator::assembleElementVector(const ReferenceElement& ref,
        ElementTransform& trans,
        Vector& elvec) const
    {
        const int nd = ref.numDofs();
        const int nq = ref.numQuadraturePoints();
        const int totalDofs = nd * vdim_;

        elvec.setZero(totalDofs);

        Eigen::Matrix<Real, MaxStrainComponents, MaxVectorDofsPerElement> B_full;
        Eigen::Matrix<Real, 6, 1> sigmaVoigt;
        Matrix3 stress;
        auto B = B_full.leftCols(totalDofs);

        for (int q = 0; q < nq; ++q) {
            const IntegrationPoint& ip = ref.integrationPoint(q);
            Real xi[3] = {ip.xi, ip.eta, ip.zeta};
            trans.setIntegrationPoint(xi);

            const Real w = ip.weight * trans.weight();

            stress = evalMatrixNode(stress_, trans);

            if (stress.norm() < 1e-20)
                continue;

            sigmaVoigt(0) = stress(0, 0);
            sigmaVoigt(1) = stress(1, 1);
            sigmaVoigt(2) = stress(2, 2);
            sigmaVoigt(3) = stress(1, 2);
            sigmaVoigt(4) = stress(0, 2);
            sigmaVoigt(5) = stress(0, 1);

            B.setZero();

            computeStrainDispMatrix(B, ref, trans, nd, vdim_, q);

            elvec.noalias() += w * (B.transpose() * sigmaVoigt);
        }
    }

} // namespace mpfem
