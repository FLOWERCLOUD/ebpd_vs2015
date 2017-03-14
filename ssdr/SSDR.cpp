#include "SSDR.h"
#include <limits>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include "QuadProg.h"
#include "HorseObject.h"
#ifdef ENABLE_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif //ENABLE_TBB

#include <fstream>
#include <sstream>
using namespace std;
using namespace DirectX;
using namespace Eigen;



namespace SSDR {

double ComputeApproximationErrorSq(const Output& output, const Input& input, const Parameter& param)
{
    std::vector<double> errsq(input.numExamples);
    double rsqsum = 0;
    for (int s = 0; s < input.numExamples; ++s)
    {
        const int numVertices = input.numVertices;
        const int numIndices = param.numIndices;
        const int numBones = output.numBones;
        const int numExamples = input.numExamples;

        for (int s = 0; s < numExamples; ++s)
        {
            for (int v = 0; v < numVertices; ++v)
            {
                XMVECTOR residual = XMLoadFloat3A(&input.sample[s * numVertices + v]);
                const XMVECTOR& p = XMLoadFloat3A(&input.bindModel[v]);
                for (int i = 0; i < numIndices; ++i)
                {
                    const int b = output.index[v * numIndices + i];
                    const float w = output.weight[v * numIndices + i];
                    const RigidTransform& rt = output.boneTrans[s * numBones + b];
                    residual -= w * rt.TransformCoord(p);
                }
                rsqsum += XMVectorGetX(XMVector3LengthSq(residual));
            }
        }
    }
    return rsqsum;
}
#ifdef ENABLE_TBB
class WeightMapUpdator
{
private:
    Output* output;
    const Input* input;
    const Parameter* param;
    const MatrixXd* cem;
    const MatrixXd* cim;
    const VectorXd* cev;
    const VectorXd* civ;
    const MatrixXd* scem;
    const MatrixXd* scim;
    const VectorXd* sciv;
public:
    WeightMapUpdator(Output* output_, const Input* input_, const Parameter* param_,
        const MatrixXd* cem_, const MatrixXd* cim_, const VectorXd* cev_, const VectorXd* civ_,
        const MatrixXd* scem_, const MatrixXd* scim_, const VectorXd* sciv_)
        : output(output_), input(input_), param(param_),
        cem(cem_), cim(cim_), cev(cev_), civ(civ_),
        scem(scem_), scim(scim_), sciv(sciv_)
    {
    }
    void operator ()(const tbb::blocked_range<int>& range) const
    {
        const int numVertices = input->numVertices;
        const int numExamples = input->numExamples;
        const int numIndices = param->numIndices;
        const int numBones = output->numBones;

        MatrixXd gm = MatrixXd::Zero(numBones, numBones), sgm = MatrixXd::Zero(numIndices, numIndices);
        VectorXd gv = VectorXd::Zero(numBones), sgv = VectorXd::Zero(numIndices);

        VectorXd weight = VectorXd::Zero(numBones), w0, sweight = VectorXd::Zero(numIndices);
        MatrixXd basis = MatrixXd::Zero(numBones, numExamples * 3), sbasis = MatrixXd::Zero(numIndices, numExamples * 3);
        VectorXd targetVertex = VectorXd::Zero(numExamples * 3);

        for (int v = range.begin(); v != range.end(); ++v)
        {
            const XMVECTOR restVertex = XMLoadFloat3A(&input->bindModel[v]);
            for (int s = 0; s < numExamples; ++s)
            {
                for (int b = 0; b < numBones; ++b)
                {
                    const RigidTransform& rt = output->boneTrans[s * numBones + b];
                    XMVECTOR tv = rt.TransformCoord(restVertex);
                    basis(b, s * 3 + 0) = XMVectorGetX(tv);
                    basis(b, s * 3 + 1) = XMVectorGetY(tv);
                    basis(b, s * 3 + 2) = XMVectorGetZ(tv);
                }
            }
            for (int s = 0; s < numExamples; ++s)
            {
                targetVertex[s * 3 + 0] = input->sample[s * numVertices + v].x;
                targetVertex[s * 3 + 1] = input->sample[s * numVertices + v].y;
                targetVertex[s * 3 + 2] = input->sample[s * numVertices + v].z;
            }
            // G = A * A^T
            gm = basis * basis.transpose();
            // g = A^T * b
            gv = -basis * targetVertex;

            double qperr = SolveQP(gm, gv, *cem, *cev, *cim, *civ, weight);
            assert(qperr != std::numeric_limits<double>::infinity());

            float weightSum = 0;
            for (int i = 0; i < numIndices; ++i)
            {
                double maxw = -std::numeric_limits<double>::max();
                int bestbone = -1;
                for (int b = 0; b < numBones; ++b)
                {
                    if (weight[b] > maxw)
                    {
                        maxw = weight[b];
                        bestbone = b;
                    }
                }
                if (maxw <= 0)
                {
                    break;
                }

                output->index[v * numIndices + i] = bestbone;
                output->weight[v * numIndices + i] = static_cast<float>(maxw);
                weightSum += static_cast<float>(maxw);
                weight[bestbone] = 0;
            }

            if (weightSum < 1.0f)
            {
                for (int j = 0; j < numExamples * 3; ++j)
                {
                    for (int i = 0; i < numIndices; ++i)
                    {
                        sbasis(i, j) = basis(output->index[v * numIndices + i], j);
                    }
                }
                sgm = sbasis * sbasis.transpose();
                sgv = -sbasis * targetVertex;
                qperr = SolveQP(sgm, sgv, *scem, *cev, *scim, *sciv, sweight);
                if (qperr != std::numeric_limits<double>::infinity())
                {
                    for (int i = 0; i < numIndices; ++i)
                    {
                        output->weight[v * numIndices + i] = static_cast<float>(sweight[i]);
                    }
                }
                else
                {
                    for (int i = 0; i < numIndices; ++i)
                    {
                        output->weight[v * numIndices + i] /= weightSum;
                    }
                }
            }
        }
    }
};
void UpdateWeightMap(Output& output, const Input& input, const Parameter& param)
{
    const int numBones = output.numBones;
    const int numIndices = param.numIndices;

    // partition of unity constraint : cem^T xv + cev = 0
    MatrixXd cem = MatrixXd::Zero(1, numBones);
    MatrixXd scem = MatrixXd::Zero(1, numIndices);
    VectorXd cev = VectorXd::Zero(1);
    //nonnegativity constraint : cim^T xv + civ >= 0
    MatrixXd cim = MatrixXd::Zero(numBones, numBones);
    MatrixXd scim = MatrixXd::Zero(numIndices, numIndices);
    VectorXd civ = VectorXd::Zero(numBones);
    VectorXd sciv = VectorXd::Zero(numIndices);
    for (int b = 0; b < numBones; ++b)
    {
        cem(0, b) = -1.0;
        cim(b, b) = 1.0;
        civ(b) = 0;
    }
    for (int i = 0; i < numIndices; ++i)
    {
        scem(0, i) = -1.0;
        scim(i, i) = 1.0;
        sciv(i) = 0;
    }
    cev(0) = 1.0;

    tbb::parallel_for(tbb::blocked_range<int>(0, input.numVertices),
        WeightMapUpdator(&output, &input, &param,
        &cem, &cim, &cev, &civ, &scem,
        &scim, &sciv));
}
#else
void UpdateWeightMap(Output& output, const Input& input, const Parameter& param)
{
    const int numVertices = input.numVertices;
    const int numExamples = input.numExamples;
    const int numIndices = param.numIndices;
    const int numBones = output.numBones;

    // Sum constraint : cem * xv + cev = 0
    MatrixXd cem = MatrixXd::Zero(1, numBones);
    MatrixXd scem = MatrixXd::Zero(1, numIndices);
    VectorXd cev = VectorXd::Zero(1);
    for (int b = 0; b < numBones; ++b)
    {
        cem(0, b) = 1.0;
    }
    cev(0) = -1.0;
    //Nonnegativity constraint : cim * xv + civ >= 0
    MatrixXd cim = MatrixXd::Zero(numBones, numBones);
    MatrixXd scim = MatrixXd::Zero(numIndices, numIndices);
    VectorXd civ = VectorXd::Zero(numBones);
    VectorXd sciv = VectorXd::Zero(numIndices);
    for (int b = 0; b < numBones; ++b)
    {
        cim(b, b) = 1.0;
        civ(b) = 0;
    }
    for (int i = 0; i < numIndices; ++i)
    {
        scem(0, i) = 1.0;
        scim(i, i) = 1.0;
        sciv(i) = 0;
    }
    
    MatrixXd gm = MatrixXd::Zero(numBones, numBones);
    MatrixXd sgm = MatrixXd::Zero(numIndices, numIndices);
    VectorXd gv = VectorXd::Zero(numBones);
    VectorXd sgv = VectorXd::Zero(numIndices);

    VectorXd weight = VectorXd::Zero(numBones);
    VectorXd sweight = VectorXd::Zero(numIndices);
    MatrixXd am = MatrixXd::Zero(numBones, numExamples * 3);
    MatrixXd sam = MatrixXd::Zero(numIndices, numExamples * 3);
    VectorXd bv = VectorXd::Zero(numExamples * 3);

    for (int v = 0; v < numVertices; ++v)
    {
        const XMVECTOR restVertex = XMLoadFloat3A(&input.bindModel[v]);
        for (int s = 0; s < numExamples; ++s)
        {
            for (int b = 0; b < numBones; ++b)
            {
                const RigidTransform& rt = output.boneTrans[s * numBones + b];
                XMVECTOR tv = rt.TransformCoord(restVertex);
                am(b, s * 3 + 0) = XMVectorGetX(tv);
                am(b, s * 3 + 1) = XMVectorGetY(tv);
                am(b, s * 3 + 2) = XMVectorGetZ(tv);
            }
        }
        for (int s = 0; s < numExamples; ++s)
        {
            bv[s * 3 + 0] = input.sample[s * numVertices + v].x;
            bv[s * 3 + 1] = input.sample[s * numVertices + v].y;
            bv[s * 3 + 2] = input.sample[s * numVertices + v].z;
        }
        // G = A * A^T
        gm = am * am.transpose();
        // g = A^T * b
        gv = -am * bv;

        double qperr = SolveQP(gm, gv, cem, cev, cim, civ, weight);
        assert(qperr != std::numeric_limits<double>::infinity());

        float weightSum = 0;
        for (int i = 0; i < numIndices; ++i)
        {
            double maxw = -std::numeric_limits<double>::max();
            int bestbone = -1;
            for (int b = 0; b < numBones; ++b)
            {
                if (weight[b] > maxw)
                {
                    maxw = weight[b];
                    bestbone = b;
                }
            }
            if (maxw <= 0)
            {
                break;
            }

            output.index[v * numIndices + i] = bestbone;
            output.weight[v * numIndices + i] = static_cast<float>(maxw);
            weightSum += static_cast<float>(maxw);
            weight[bestbone] = 0;
        }

        if (weightSum < 1.0f)
        {
            for (int j = 0; j < numExamples * 3; ++j)
            {
                for (int i = 0; i < numIndices; ++i)
                {
                    sam(i, j) = am(output.index[v * numIndices + i], j);
                }
            }
            sgm = sam * sam.transpose();
            sgv = -sam * bv;
            qperr = SolveQP(sgm, sgv, scem, cev, scim, sciv, sweight);
            if (qperr != std::numeric_limits<double>::infinity())
            {
                for (int i = 0; i < numIndices; ++i)
                {
                    output.weight[v * numIndices + i] = static_cast<float>(sweight[i]);
                }
            }
            else
            {
                for (int i = 0; i < numIndices; ++i)
                {
                    output.weight[v * numIndices + i] /= weightSum;
                }
            }
        }
    }
}
#endif

// List xxx.13: Horn point cloud alignment algorithm
RigidTransform CalcPointsAlignment(size_t numPoints, std::vector<XMFLOAT3A>::const_iterator ps, std::vector<XMFLOAT3A>::const_iterator pd)
{
    RigidTransform transform;

    // Calculate the barycentric coordinates of each point group
    XMVECTOR cs = XMVectorZero(), cd = XMVectorZero();
    std::vector<XMFLOAT3A>::const_iterator sit = ps;
    std::vector<XMFLOAT3A>::const_iterator dit = pd;
    for (size_t i = 0; i < numPoints; ++i, ++sit, ++dit)
    {
        cs += XMLoadFloat3A(&(*sit));
        cd += XMLoadFloat3A(&(*dit));
    }
    cs /= numPoints;
    cd /= numPoints;

    // If rotation can not be estimated or if rotation is not estimated, only parallel movement components are returned
    if (numPoints < 3)
    {
        XMStoreFloat3A(&transform.Translation(), cd - cs);
        return transform;
    }

    // Calculate the moment matrix
    Matrix<double, 4, 4> moment;
    double sxx = 0, sxy = 0, sxz = 0, syx = 0, syy = 0, syz = 0, szx = 0, szy = 0, szz = 0;
    sit = ps;
    dit = pd;
    for (size_t i = 0; i < numPoints; ++i, ++sit, ++dit)
    {
        sxx += (sit->x - XMVectorGetX(cs)) * (dit->x - XMVectorGetX(cd));
        sxy += (sit->x - XMVectorGetX(cs)) * (dit->y - XMVectorGetY(cd));
        sxz += (sit->x - XMVectorGetX(cs)) * (dit->z - XMVectorGetZ(cd));
        syx += (sit->y - XMVectorGetY(cs)) * (dit->x - XMVectorGetX(cd));
        syy += (sit->y - XMVectorGetY(cs)) * (dit->y - XMVectorGetY(cd));
        syz += (sit->y - XMVectorGetY(cs)) * (dit->z - XMVectorGetZ(cd));
        szx += (sit->z - XMVectorGetZ(cs)) * (dit->x - XMVectorGetX(cd));
        szy += (sit->z - XMVectorGetZ(cs)) * (dit->y - XMVectorGetY(cd));
        szz += (sit->z - XMVectorGetZ(cs)) * (dit->z - XMVectorGetZ(cd));
    }
    moment(0, 0) = sxx + syy + szz;
    moment(0, 1) = syz - szy;        moment(1, 0) = moment(0, 1);
    moment(0, 2) = szx - sxz;        moment(2, 0) = moment(0, 2);
    moment(0, 3) = sxy - syx;        moment(3, 0) = moment(0, 3);
    moment(1, 1) = sxx - syy - szz;
    moment(1, 2) = sxy + syx;        moment(2, 1) = moment(1, 2);
    moment(1, 3) = szx + sxz;        moment(3, 1) = moment(1, 3);
    moment(2, 2) = -sxx + syy - szz;
    moment(2, 3) = syz + szy;        moment(3, 2) = moment(2, 3);
    moment(3, 3) = -sxx - syy + szz;

    if (moment.norm() > 0)
    {
        // Get the eigenvector corresponding to signed maximum eigenvalue
        EigenSolver<Matrix<double, 4, 4>> es(moment);
        int maxi = 0;
        for (int i = 1; i < 4; ++i)
        {
            if (es.eigenvalues()(maxi).real() < es.eigenvalues()(i).real())
            {
                maxi = i;
            }
        }
        transform.Rotation() = XMFLOAT4A(
            static_cast<float>(es.eigenvectors()(1, maxi).real()),
            static_cast<float>(es.eigenvectors()(2, maxi).real()),
            static_cast<float>(es.eigenvectors()(3, maxi).real()),
            static_cast<float>(es.eigenvectors()(0, maxi).real()));
    }

    // translation component
    //
    XMVECTOR cs0 = transform.TransformCoord(cs);
    XMStoreFloat3A(&transform.Translation(), cd - cs0);
    return transform;
}

// Expression xxx.9: \ tilde {q} _ {j, n}
void ComputeExamplePoints(std::vector<XMFLOAT3A>& example, int sid, int bone, const Output& output, const Input& input, const Parameter& param)
{
    const int numVertices = input.numVertices;
    const int numIndices = param.numIndices;
    const int numBones = output.numBones;
    for (int v = 0; v < numVertices; ++v)
    {
        example[v] = input.sample[sid * numVertices + v];
        const XMVECTOR s = XMLoadFloat3A(&input.bindModel[v]);
        for (int i = 0; i < numIndices; ++i)
        {
            const int b = output.index[v * numIndices + i];
            if (b != bone)
            {
                const float w = output.weight[v * numIndices + i];
                const RigidTransform& at = output.boneTrans[sid * numBones + b];
                XMVECTOR r = XMLoadFloat3A(&example[v]);
                XMStoreFloat3A(&example[v], r - w * at.TransformCoord(s));
            }
        }
    }
}

void SubtractCentroid(std::vector<XMFLOAT3A>& model, std::vector<XMFLOAT3A>& example, XMFLOAT3A& corModel, XMFLOAT3A& corExample, const VectorXd& weight, const Output& output, const Input& input)
{
    const int numVertices = input.numVertices;

// Expression xxx.10: \ bar {p} _n, \ bar {q} _ {j, n}
    double wsqsum = 0;
    XMVECTOR dmodel = XMVectorZero();
    XMVECTOR dexample = XMVectorZero();
    for (int v = 0; v < numVertices; ++v)
    {
        const double w = weight[v];
        dmodel += static_cast<float>(w * w) * XMLoadFloat3A(&input.bindModel[v]);
        dexample += static_cast<float>(w)* XMLoadFloat3A(&example[v]);
        wsqsum += w * w;
    }
    dmodel /= static_cast<float>(wsqsum);
    dexample /= static_cast<float>(wsqsum);
    XMStoreFloat3A(&corModel, dmodel);
    XMStoreFloat3A(&corExample, dexample);

    for (int v = 0; v < numVertices; ++v)
    {
        // Expression xxx.11: w_ {j, c} p_j
        XMVECTOR d = XMLoadFloat3A(&input.bindModel[v]) - XMLoadFloat3A(&corModel);
        XMStoreFloat3A(&model[v], static_cast<float>(weight[v]) * d);
        // Expression xxx.11: q_ {j, n}
        d = XMLoadFloat3A(&example[v]) - static_cast<float>(weight[v]) * XMLoadFloat3A(&corExample);
        XMStoreFloat3A(&example[v], d);
    }
}

#ifdef ENABLE_TBB
class BoneTransformUpdator
{
private:
    Output* output;
    const Input* input;
    const Parameter* param;
    const VectorXd* weight;
    int bone;
public:
    BoneTransformUpdator(Output* output_, const Input* input_, const Parameter* param_, const VectorXd* weight_)
        : bone(0), output(output_), input(input_), param(param_), weight(weight_)
    {
    }
    void ChangeBone(int b)
    {
        bone = b;
    }
    void operator () (const tbb::blocked_range<int>& range) const
    {
        std::vector<XMFLOAT3A> model(input->numVertices), example(input->numVertices);
        for (int s = range.begin(); s != range.end(); ++s)
        {
            // Expression xxx.9: \ tilde {q} _ {j, n}
            ComputeExamplePoints(example, s, bone, *output, *input, *param);

            // Expressions xxx.10, xxx.11
            XMFLOAT3A corModel(0, 0, 0), corExample(0, 0, 0);
            SubtractCentroid(model, example, corModel, corExample, *weight, *output, *input);

            // the solution of the expression xxx.12
            RigidTransform transform = CalcPointsAlignment(model.size(), model.begin(), example.begin());
            // Expression xxx.13
            XMVECTOR d = XMLoadFloat3A(&corExample) - transform.TransformCoord(XMLoadFloat3A(&corModel));
            XMStoreFloat3A(&transform.Translation(), d + XMLoadFloat3A(&transform.Translation()));
            output->boneTrans[s * output->numBones + bone] = transform;
        }
    }
};
void UpdateBoneTransform(Output& output, const Input& input, const Parameter& param)
{
    const int numVertices = input.numVertices;
    const int numExamples = input.numExamples;
    const int numIndices = param.numIndices;
    const int numBones = output.numBones;

    VectorXd boneWeight = VectorXd::Zero(numVertices);
    BoneTransformUpdator transformUpdator(&output, &input, &param, &boneWeight);
    tbb::blocked_range<int> blockedRange(0, numExamples);
    for (int bone = 0; bone < numBones; ++bone)
    {
        for (int v = 0; v < numVertices; ++v)
        {
            boneWeight[v] = 0;
            for (int i = 0; i < numIndices; ++i)
            {
                if (output.index[v * numIndices + i] == bone)
                {
                    boneWeight[v] = output.weight[v * numIndices + i];
                    break;
                }
            }
        }
        transformUpdator.ChangeBone(bone);
        tbb::parallel_for(blockedRange, transformUpdator);
    }
}
#else
void UpdateBoneTransform(Output& output, const Input& input, const Parameter& param)
{
    const int numVertices = input.numVertices;
    const int numExamples = input.numExamples;
    const int numIndices = param.numIndices;
    const int numBones = output.numBones;

    VectorXd weight = VectorXd::Zero(numVertices);
    std::vector<XMFLOAT3A> model(input.numVertices), example(input.numVertices);
    for (int bone = 0; bone < numBones; ++bone)
    {
        for (int v = 0; v < numVertices; ++v)
        {
            weight[v] = 0;
            for (int i = 0; i < numIndices; ++i)
            {
                if (output.index[v * numIndices + i] == bone)
                {
                    weight[v] = output.weight[v * numIndices + i];
                    break;
                }
            }
        }

        for (int s = 0; s < numExamples; ++s)
        {
            // Expression xxx.9: \ tilde {q} _ {j, n}
            ComputeExamplePoints(example, s, bone, output, input, param);
             // Expressions xxx.10, xxx.11
            XMFLOAT3A corModel(0, 0, 0), corExample(0, 0, 0);
            SubtractCentroid(model, example, corModel, corExample, weight, output, input);
             // the solution of the expression xxx.12
            RigidTransform transform = CalcPointsAlignment(model.size(), model.begin(), example.begin());
             // Expression xxx.13
            XMVECTOR d = XMLoadFloat3A(&corExample) - transform.TransformCoord(XMLoadFloat3A(&corModel));
            XMStoreFloat3A(&transform.Translation(), d + XMLoadFloat3A(&transform.Translation()));
            output.boneTrans[s * output.numBones + bone] = transform;
        }
    }
}
#endif
void UpdateBoneTransform(std::vector<RigidTransform>& boneTrans, int numBones, const Output& output, const Input& input, const Parameter& param)
{
    const int numVertices = input.numVertices;
    const int numExamples = input.numExamples;
    const int numIndices = param.numIndices;

    std::vector<int> numBoneVertices(numBones, 0);
    for (int v = 0; v < numVertices; ++v)
    {
        ++numBoneVertices[output.index[v * numIndices + 0]];
    }
    std::vector<int> boneVertexId(numBones, 0);
    for (int i = 1; i < numBones; ++i)
    {
        boneVertexId[i] = boneVertexId[i - 1] + numBoneVertices[i - 1];
    }
    std::vector<XMFLOAT3A> skin(numVertices, XMFLOAT3A(0, 0, 0));
    std::vector<XMFLOAT3A> anim(numVertices * numExamples, XMFLOAT3A(0, 0, 0));
    for (int v = 0; v < numVertices; ++v)
    {
        const int bs = output.index[v * numIndices + 0];
        const int bd = boneVertexId[bs];
        skin[bd] = input.bindModel[v];
        for (int s = 0; s < numExamples; ++s)
        {
            anim[s * numVertices + bd] = input.sample[s * numVertices + v];
        }
        ++boneVertexId[bs];
    }
    boneVertexId[0] = 0;
    for (int i = 1; i < numBones; ++i)
    {
        boneVertexId[i] = boneVertexId[i - 1] + numBoneVertices[i - 1];
    }
    for (int b = 0; b < numBones; ++b)
    {
        for (int s = 0; s < numExamples; ++s)
        {
			if(numBoneVertices[b])
			{
				boneTrans[s * numBones + b] = CalcPointsAlignment(numBoneVertices[b], skin.begin() + boneVertexId[b], anim.begin() + s * numVertices + boneVertexId[b]);
			}else
			{
//				std::cout<<"b "<< b <<"s "<< s <<"numBoneVertices zero "<<std::endl;
			}
        }
    }
}

int BindVertexToBone(Output& output, std::vector<RigidTransform>& boneTrans, const Input& input, const Parameter& param)
{
    const int numVertices = input.numVertices;
    const int numExamples = input.numExamples;
    const int numIndices = param.numIndices;
    int numBones = static_cast<int>(boneTrans.size() / numExamples);

    std::vector<int> numBoneVertices(numBones, 0);
    std::vector<float> vertexError(numVertices, 0);

    for (int v = 0; v < numVertices; ++v)
    {
        int bestBone = 0;
        float minErr = std::numeric_limits<float>::max();
        const XMVECTOR bindModelPos = XMLoadFloat3A(&input.bindModel[v]);
        for (int b = 0; b < numBones; ++b)
        {
            float errsq = 0;
            for (int s = 0; s < numExamples; ++s)
            {
                const RigidTransform& at = boneTrans[s * numBones + b];
                XMVECTOR diff = XMLoadFloat3A(&input.sample[s * numVertices + v])
                              - at.TransformCoord(XMLoadFloat3A(&input.bindModel[v]));
                errsq += XMVectorGetX(XMVector3LengthSq(diff));
            }
            if (errsq < minErr)
            {
                bestBone = b;
                minErr = errsq;
            }
        }
        ++numBoneVertices[bestBone];
        output.index[v * numIndices + 0] = bestBone;
        vertexError[v] = minErr;
    }

    // Removal of empty cluster
    std::vector<int>::iterator smallestBoneSize = std::min_element(numBoneVertices.begin(), numBoneVertices.end());
    while (*smallestBoneSize <= 0)
    {
        const int smallestBone = static_cast<int>(smallestBoneSize - numBoneVertices.begin());
        numBoneVertices.erase(numBoneVertices.begin() + smallestBone);
        for (int s = input.numExamples - 1; s >= 0; --s)
        {
            boneTrans.erase(boneTrans.begin() + s * numBones + smallestBone);
        }
        for (int v = 0; v < numVertices; ++v)
        {
            int b = output.index[v * numIndices + 0];
            if (b >= smallestBone)
            {
                --output.index[v * numIndices + 0];
            }
        }
        --numBones;
        smallestBoneSize = std::min_element(numBoneVertices.begin(), numBoneVertices.end());
    }
    return static_cast<int>(numBoneVertices.size());
}

int ClusterInitialBones(Output& output, const Input& input, const Parameter& param)
{
    const int numVertices = input.numVertices;
    const int numExamples = input.numExamples;
    const int numIndices = param.numIndices;

    std::fill(output.index.begin(), output.index.end(), 0);
    std::fill(output.weight.begin(), output.weight.end(), 0.0f);
    for (int v = 0; v < numVertices; ++v)
    {
        output.weight[v * numIndices + 0] = 1.0f;
    }

    int numClusters = 1;
    std::vector<RigidTransform> boneTrans(numExamples);
    UpdateBoneTransform(boneTrans, numClusters, output, input, param);

    while (numClusters < param.numMinBones)
    {
        std::vector<XMFLOAT3A> clusterCenter(numClusters, XMFLOAT3A(0, 0, 0));
        std::vector<int> numBoneVertices(numClusters, 0);
        for (int v = 0; v < numVertices; ++v)
        {
            const int c = output.index[v * numIndices + 0];
            clusterCenter[c].x += input.bindModel[v].x;
            clusterCenter[c].y += input.bindModel[v].y;
            clusterCenter[c].z += input.bindModel[v].z;
            ++numBoneVertices[c];
        }
        for (int c = 0; c < numClusters; ++c)
        {
            clusterCenter[c].x /= static_cast<float>(numBoneVertices[c]);
            clusterCenter[c].y /= static_cast<float>(numBoneVertices[c]);
            clusterCenter[c].z /= static_cast<float>(numBoneVertices[c]);
        }

        std::vector<float> maxClusterError(numClusters, -std::numeric_limits<float>::max());
        std::vector<int> mostDistantVertex(numClusters, -1);
        for (int v = 0; v < numVertices; ++v)
        {
            const int c = output.index[v * numIndices + 0];
            float sumApproxErrorSq = 0;
            for (int s = 0; s < numExamples; ++s)
            {
                XMVECTOR diff = XMLoadFloat3A(&input.sample[s * numVertices + v])
                    - boneTrans[s * numClusters + c].TransformCoord(XMLoadFloat3A(&input.bindModel[v]));
                sumApproxErrorSq += XMVectorGetX(XMVector3LengthSq(diff));
            }
            XMVECTOR d = XMLoadFloat3A(&input.bindModel[v]) - XMLoadFloat3A(&clusterCenter[c]);
            float errSq = sumApproxErrorSq * XMVectorGetX(XMVector3LengthSq(d));
            if (errSq > maxClusterError[c])
            {
                maxClusterError[c] = errSq;
                mostDistantVertex[c] = v;
            }
        }
        int numPrevClusters = numClusters;
        for (int c = 0; c < numPrevClusters; ++c)
        {
            output.index[mostDistantVertex[c] * numIndices + 0] = numClusters++;
            --numBoneVertices[c];
            numBoneVertices.push_back(1);
        }
        boneTrans.resize(numExamples * numClusters);

        UpdateBoneTransform(boneTrans, numClusters, output, input, param);
        numClusters = BindVertexToBone(output, boneTrans, input, param);
    }
    return numClusters;
}

#pragma region Decompose
double Decompose(Output& output, const Input& input, const Parameter& param)
{
    const int numVertices = input.numVertices;
    const int numExamples = input.numExamples;
    const int numIndices = param.numIndices;

    output.index.assign(numVertices * numIndices, 0);
    output.weight.assign(numVertices * numIndices, 0.0f);

    //Initial binding using cluster partition expectation maximization method
    output.numBones = ClusterInitialBones(output, input, param);
    // Initial bone transform
    output.boneTrans.assign(numExamples * output.numBones, RigidTransform::Identity());
    UpdateBoneTransform(output.boneTrans, output.numBones, output, input, param);

     //Alternate optimization of skinning weight and bone attitude by BCD algorithm
    for (int loop = 0; loop < param.numMaxIterations; ++loop)
    {
        UpdateWeightMap(output, input, param);
        UpdateBoneTransform(output, input, param);
    }
    return ComputeApproximationErrorSq(output, input, param);
}

static void getWholeVerticesArray(std::vector<float>& _input ,int numVertices , const HorseObject* const obj )
{
		const HorseObject::CustomVertex* const vertexBufferCPU = obj->getvertexBufferCPU();
		_input.resize(3*numVertices );
		for (int v = 0; v < numVertices; ++v)
		{
			XMVECTOR pos = XMVectorZero();
			//for (long i = 0; i < NumInfluences; ++i)
			//{
			//	const RigidTransform& rt = boneAnim[numBones * frame + vertexBufferCPU[v].indices[i]];
			//	XMVECTOR xv = XMLoadFloat3A(&vertexBufferCPU[v].position);
			//	pos += vertexBufferCPU[v].weight[i] * rt.TransformCoord(xv);
			//}
			pos = XMLoadFloat3A(&vertexBufferCPU[v].position);
			_input[3*v] = XMVectorGetX(pos);
			_input[3*v+1] = XMVectorGetY(pos);
			_input[3*v+2] = XMVectorGetZ(pos);
		}

}

static void caculateCenterOfVertices(const std::vector<float>& _input , float center[3])
{

	float sum[3];
	sum[0] = sum[1] =sum[2] = 0.0f;
	int numvertice = _input.size()/3;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < numvertice; j++)
		{
			sum[i] += _input[3*j+i];
		}
		sum[i] /=  numvertice;
	}
	center[0] = sum[0];
	center[1] = sum[1];
	center[2] = sum[2];




}



void WriteRigToFile(const Output& output,const Input& ssdrIn ,const Parameter& ssdrParam, std::string file_paths)
{
	ofstream ofs(file_paths);
	ofs.setf( ios::fixed ,ios::floatfield);
	ofs.precision(6);
	ofs<<"# num bone "<< output.numBones<<std::endl;

	int numVertices = ssdrIn.numVertices;
	int numIndices = ssdrParam.numIndices;
	int numBones = output.numBones ;
	int numExamples = ssdrIn.numExamples;

	ofs<<"#  BONES: Num_of_Frames Num_of_Bones "<<std::endl;
	ofs<<numExamples<<" "<<numBones<<std::endl;
	for (int s = 0; s < numExamples; ++s)
	{
		ofs<<s<<std::endl;
		for (int b = 0; b < numBones; ++b)
		{
			ofs<<b<<" "<<" ";
			const RigidTransform& at =output.boneTrans[s * numBones + b];
			const DirectX::XMFLOAT4A& r = at.Rotation(); 
			const DirectX::XMFLOAT3A& t = at.Translation();
			ofs<<r.x<<" "<<r.y<<" "<<r.z<<" "<<r.w<<" ";
			ofs<<t.x<<" "<<t.y<<" "<<t.z<<std::endl;

		}
	}
	ofs<<"# WEIGHTS: Num_of_Vertices Num_of_Bones Max_Bone "<<std::endl;
	ofs<<numVertices<<" "<<numBones<<" "<<numIndices<<std::endl;
	for (int v = 0; v < numVertices; ++v)
	{
		ofs<<v<<" "<<" ";
		for (int b = 0; b < numIndices; ++b)
		{
			ofs<<output.index[v * numIndices + b]<<" ";
			ofs<<output.weight[v * numIndices + b]<<" ";
			ofs<<" ";
		}
		ofs<<std::endl;
	}

}

void WriteRigToFileFormat2(const Output& output,const Input& ssdrIn,const Parameter& ssdrParam ,
						   	const HorseObject* const obj,
						   std::string _file_paths_dir,std::string _fine_prifixname)
{
	using namespace std;
	string skeleton_file_path = _file_paths_dir+_fine_prifixname+".skel";
	string weight_file_path = _file_paths_dir+_fine_prifixname+".weights";

	ofstream ofs(skeleton_file_path);
	ofs.setf( ios::fixed ,ios::floatfield);
	ofs.precision(6);
	int numVertices = ssdrIn.numVertices;
	int numIndices = ssdrParam.numIndices;
	int numBones = output.numBones ;
	int numExamples = ssdrIn.numExamples;
	//number of joints and joint edge
	ofs<< 2*numBones+1<<" "<<2*numBones<<std::endl;
	//first joint is root joint ,we caculate the center of the mesh as the bine position of root joint
	ofs<<0.f<<" "<< 0.f<<" "<<0.f<<std::endl;
	vector<float> vertices;
	getWholeVerticesArray( vertices,numVertices,obj);
	float whole_center[3];
	caculateCenterOfVertices( vertices , whole_center);
	ofs<<  whole_center[0]<<" "<<whole_center[1]<<" "<<whole_center[2]<<std::endl;

	float (*centerOfBone)[3] = new float[numBones][3]; //to caculate the center of bone
	for (int i = 0; i < numBones; i++)
	{
		centerOfBone[i][0] = 0.f;
		centerOfBone[i][1] = 0.f;
		centerOfBone[i][2] = 0.f;
	}
	std::vector<int> numOfVertexInBone( numBones ,0);
	std::vector<float> maxWeight( numBones ,0);
	std::vector<int> maxWeightOfVetex( numBones ,0);
	for (int v = 0; v < numVertices; ++v)
	{
		//int maxweigntbone = -1;
		//int max_weight = 0.1;
		for (int b = 0; b < numIndices; ++b)
		{

			if( output.weight[v * numIndices + b] > maxWeight[output.index[v * numIndices + b]])
			{
				maxWeightOfVetex[output.index[v * numIndices + b]] =  v;
				maxWeight[output.index[v * numIndices + b]] = output.weight[v * numIndices + b];
			}
			//if(output.weight[v * numIndices + b] > max_weight)
			//{
			//	//max_weight = output.weight[v * numIndices + b];
			//	maxweigntbone = output.index[v * numIndices + b];
			//	centerOfBone[ maxweigntbone][0] += vertices[3*v];
			//	centerOfBone[ maxweigntbone][1] += vertices[3*v +1];
			//	centerOfBone[ maxweigntbone][2] += vertices[3*v +2];
			//	++numOfVertexInBone[ maxweigntbone];
			//}
		}

	}

	for (int i = 0; i < numBones; i++)
	{
		//centerOfBone[i][0] /= numOfVertexInBone[i];
		//centerOfBone[i][1] /= numOfVertexInBone[i];
		//centerOfBone[i][2] /= numOfVertexInBone[i];
		centerOfBone[i][0] = vertices[3*maxWeightOfVetex[i]+0];
		centerOfBone[i][1] = vertices[3*maxWeightOfVetex[i]+1];
		centerOfBone[i][2] = vertices[3*maxWeightOfVetex[i]+2];
		ofs<<  centerOfBone[i][0]<<" "<<centerOfBone[i][1]<<" "<<centerOfBone[i][2]<<std::endl;
		if( i!=0)
			ofs<<  centerOfBone[i][0]+0.01<<" "<<centerOfBone[i][1]+0.01<<" "<<centerOfBone[i][2]+0.01<<std::endl;
	}

	delete[] centerOfBone;
	for (int i = 0; i < numBones; i++)
	{

		ofs<< 0<<" "<<(i+1)*2 -1<<" "<<std::endl;
		ofs<<(i+1)*2 -1<<" "<<(i+1)*2<<" "<<std::endl;
	}

	ofs.close();

	ofstream ofs_weight( weight_file_path);
	ofs.setf( ios::fixed ,ios::floatfield);
	ofs.precision(6);
	for (int v = 0; v < numVertices; ++v)
	{
		for (int b = 0; b < numIndices; ++b)
		{
			//ofs_weight<< (output.index[v * numIndices + b] +2)<<" "<<output.weight[v * numIndices + b];
			ofs_weight<< 2*(output.index[v * numIndices + b] +1)<<" "<<output.weight[v * numIndices + b];

			if( b+1 < numIndices)
				ofs_weight<<" ";
		}
		ofs_weight<<std::endl;
	}
	ofs_weight.close();

}
void GetRigFromFile(Output& result , std::string file_paths)
{
	ifstream ifs(file_paths);


	int numVertices;
	int numExamples;
	int numIndices;
	int numBones;

	const int LINE_LENGTH = 200;
	char str[LINE_LENGTH];
	//skip #
	while (ifs.getline( str,LINE_LENGTH))
	{
//		cout<<" read from file"<<str<<std::endl;
		if( str[0] !='#')break;
	}
	stringstream s( str);
	s>>numExamples>>numBones;
//	cout<<numExamples<<" "<<numBones<<endl;
	result.boneTrans.assign(numExamples * numBones, RigidTransform::Identity());

	for (int s = 0; s < numExamples; ++s)
	{
		int curexam;
		ifs>>curexam;
//		cout<<curexam<<endl;
		for (int b = 0; b < numBones; ++b)
		{
			int curbone;
			ifs>>curbone;
//			cout<<curbone<<" ";
			RigidTransform& at = result.boneTrans[s * numBones + b];
			DirectX::XMFLOAT4A r; 
			DirectX::XMFLOAT3A t;
			ifs>>r.x>>r.y>>r.z>>r.w;
			ifs>>t.x>>t.y>>t.z;
			at.Set( r , t );
//			cout<<r.x<<" "<<r.y<<" "<<r.z<<" "<<r.w<<" ";
//			cout<<t.x<<" "<<t.y<<" "<<t.z<<endl;
		}
//		cout<<endl;
	}

	//skip #
	ifs.getline( str,LINE_LENGTH);
	while (ifs.getline( str,LINE_LENGTH))
	{
//		cout<<" read from file"<<str<<std::endl;
		if( str[0] !='#')break;
	}
	s = stringstream("");
	s<<str;
	s>>numVertices>>numBones>>numIndices;
//	cout<<numVertices<<" "<<numBones<<std::endl;
	result.index.assign(numVertices * numIndices, 0);
	result.weight.assign(numVertices * numIndices, 0.0);
	for (int v = 0; v < numVertices; ++v)
	{
		int curvertice;
		ifs>>curvertice;
//		cout<<curvertice<<" ";
		for (int b = 0; b < numIndices; ++b)
		{
			int  curbone;
			ifs>>curbone;
//			cout<<curbone<<" ";
			ifs>>result.index[v * numIndices + b];
			ifs>>result.weight[v * numIndices + b];
//			cout<<result.weight[v * numIndices + b]<<" ";

		}
//		cout<<endl;

	}


	ifs.close();
}

void WriteAnimationToFile(std::string file_paths,
						  const std::vector<RigidTransform>& boneAnim, 
						  const HorseObject* const obj,
						  //const HorseObject::CustomVertex* const vertexBufferCPU,
						  const std::vector<DWORD>& index,
						  int Numfaces,
						  const Output& output,const Input& ssdrIn ,const Parameter& ssdrParam )
{
	const HorseObject::CustomVertex* const vertexBufferCPU = obj->getvertexBufferCPU();
	int numFaces = Numfaces;
	int NumInfluences = ssdrParam.numIndices;
	int numVertices = ssdrIn.numVertices;
	int numBones = output.numBones ;
	int numExamples = ssdrIn.numExamples;
	for (int frame = 0; frame < numExamples; frame++)
	{
		stringstream s;
		s<<file_paths<<frame<<".obj";
		std::string fullpath;
		s>>fullpath;
		ofstream ofs(fullpath);
		ofs.setf( ios::fixed ,ios::floatfield);
		ofs.precision(6);
		for (int v = 0; v < numVertices; ++v)
		{
			XMVECTOR pos = XMVectorZero();
			for (long i = 0; i < NumInfluences; ++i)
			{
				const RigidTransform& rt = boneAnim[numBones * frame + vertexBufferCPU[v].indices[i]];
				XMVECTOR xv = XMLoadFloat3A(&vertexBufferCPU[v].position);
				pos += vertexBufferCPU[v].weight[i] * rt.TransformCoord(xv);
			}
			ofs<<"v "<<XMVectorGetX(pos)<<" "<<XMVectorGetY(pos)<<" "<<XMVectorGetZ(pos)<<std::endl;
		}
		for( int i = 0 ;i< numFaces ;++i)
		{
			ofs<<"f "<< index[ 3 * i + 0]+1<<" "<<index[ 3 * i + 1]+1<<" "<<index[ 3 * i + 2]+1<<std::endl;
		}


		ofs.close();
	}

}

void rtRigidToCom(const RigidTransform& rt , RTransform& ct)
{
	ct.rotation[0] = rt.Rotation().x;
	ct.rotation[1] = rt.Rotation().y;
	ct.rotation[2] = rt.Rotation().z;
	ct.rotation[3] = rt.Rotation().w;
	ct.translation[0] = rt.Translation().x;
	ct.translation[1] = rt.Translation().y;
	ct.translation[2] = rt.Translation().z;

}

void ComTOrtRigid(const RTransform& ct ,RigidTransform& rt )
{
	 rt.Rotation().x =ct.rotation[0];
	 rt.Rotation().y =ct.rotation[1];
	 rt.Rotation().z =ct.rotation[2];
	 rt.Rotation().w =ct.rotation[3];
	 rt.Translation().x =ct.translation[0];
	 rt.Translation().y =ct.translation[1];
	 rt.Translation().z =ct.translation[2];
}

#pragma endregion

} //namespace SSDR