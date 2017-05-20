#include "Cluster.h"
#include <iostream>
using namespace std;
using namespace cv;

Cluster::Cluster(void)
{
	q = Mat(1, 3, CV_32FC1);
	covMatrix = Mat(3, 3, CV_32FC1);
	eigenvectors = Mat(3, 3, CV_32FC1);
	eigenvalues = Mat(3, 1, CV_32FC1);
	e = Mat(3, 1, CV_32FC1);
}

Cluster::Cluster(const vector<pair<cv::Vec3f, float> > &sample_set)
{
	q = Mat(1, 3, CV_32FC1);
	covMatrix = Mat(3, 3, CV_32FC1);
	eigenvectors = Mat(3, 3, CV_32FC1);
	eigenvalues = Mat(3, 1, CV_32FC1);
	e = Mat(3, 1, CV_32FC1);
	this->sample_set = sample_set;
}

//原始计算方法，暂时没用
void Cluster::Calc(void)
{
	//calc_SSE();
	float totalWeight = 0.0f;
	size_t elemNum = sample_set.size();
	Mat X = Mat(elemNum, 3, CV_32FC1);		//sample point vector
	Mat w = Mat(elemNum, 1, CV_32FC1);	//sample point weight vector
	Mat wpixels = Mat(elemNum, 3, CV_32FC1);	//X*w
	Vec3f color;

	for (size_t k = 0; k < elemNum; k++)
	{
		color = sample_set[k].first;

		X.at<float>(k, 0) = color[0];	//B
		X.at<float>(k, 1) = color[1];	//G
		X.at<float>(k, 2) = color[2];	//R

		w.at<float>(k, 0) = sample_set[k].second;

		wpixels.at<float>(k, 0) = color[0] * sample_set[k].second;
		wpixels.at<float>(k, 1) = color[1] * sample_set[k].second;
		wpixels.at<float>(k, 2) = color[2] * sample_set[k].second;

		totalWeight += sample_set[k].second;
	}

	float s1 = 0.0f;
	float s2 = 0.0f;
	float s3 = 0.0f;
	for (size_t k = 0; k < elemNum; k++)
	{
		s1 = s1 + wpixels.at<float>(k, 0);
		s2 = s2 + wpixels.at<float>(k, 1);
		s3 = s3 + wpixels.at<float>(k, 2);
	}

	q.at<float>(0, 0) = s1 / totalWeight;	//weighted mean color
	q.at<float>(0, 1) = s2 / totalWeight;
	q.at<float>(0, 2) = s3 / totalWeight;

	//compute weighted covariance matrix
	Mat qwork = Mat(elemNum, 3, CV_32FC1);
	for (size_t k = 0; k < elemNum; k++)
	{
		qwork.at<float>(k, 0) = q.at<float>(0, 0);
		qwork.at<float>(k, 1) = q.at<float>(0, 1);
		qwork.at<float>(k, 2) = q.at<float>(0, 2);
	}

	Mat twork = X - qwork;	//for ∑ computing

	Mat wwork = Mat(elemNum, 3, CV_32FC1);
	for (size_t k = 0; k < elemNum; k++)
	{
		float w_sqrt = sqrtf(w.at<float>(k, 0));
		wwork.at<float>(k, 0) = w_sqrt;
		wwork.at<float>(k, 1) = w_sqrt;
		wwork.at<float>(k, 2) = w_sqrt;
	}

	Mat t = Mat(elemNum, 3, CV_32FC1);

	for (int r = 0; r < wwork.rows; r++)
	{
		for (int c = 0; c < wwork.cols; c++)
		{
			t.at<float>(r, c) = twork.at<float>(r, c) * wwork.at<float>(r, c);
		}
	}

	//weighted covariance matrix
	covMatrix = (t.t() * t) / totalWeight + Mat::eye(3, 3, CV_32FC1) * 1e-5;	//t():transpose；eye():identity matrix；+1e-5防止协方差矩阵为0矩阵

																				/*cv::Mat eigval = cv::Mat(3, 1, CV_32FC1);
																				cv::Mat eigvec = cv::Mat(3, 3, CV_32FC1);

																				cv::SVD svd(covMatrix);
																				svd.w.copyTo(eigval);
																				svd.u.copyTo(eigvec);

																				e = eigvec.col(0);
																				lambda = eigval.at<float>(0, 0);*/

																				//finds eigenvalues and eigenvectors of a symmetric matrix
	cv::eigen(covMatrix, eigenvalues, eigenvectors);


	e.at<float>(0, 0) = eigenvectors.at<float>(0, 0);
	e.at<float>(1, 0) = eigenvectors.at<float>(0, 1);
	e.at<float>(2, 0) = eigenvectors.at<float>(0, 2);

	lambda = eigenvalues.at<float>(0, 0);	//the greatest eigenvalue 

											/*cout<<"lambda:"<<lambda<<endl;

											cout<<endl;
											for(int i=0; i<eigenvalues.rows; i++) {
											for(int j=0; j<eigenvalues.cols; j++) {
											cout<<eigenvalues.at<float>(i,j)<<endl;
											}
											}

											cout<<endl;
											for(int i=0; i<eigenvectors.rows; i++) {
											for(int j=0; j<eigenvectors.cols; j++) {
											cout<<eigenvectors.at<float>(i,j)<<" ";
											} cout<<endl;
											}
											system("pause");*/
}

void Cluster::calc_SSE(void)	//加速的方法
{
	//Calc();
	//R、G、B同时进行计算
	__m128 mean = _mm_setzero_ps();
	__m128 totalWeight = _mm_setzero_ps();
	size_t elemNum = sample_set.size();
	__m128 colorX;
	cv::Vec3f color;

	//compute mean color
	for (size_t k = 0; k<elemNum; k++)
	{
		color = sample_set.at(k).first;
		colorX = _mm_set_ps(1.0f, color[2], color[1], color[0]);	//Data0~3: BGR1.0
		__m128 weight = _mm_set_ps1(sample_set.at(k).second);	//四个元素都设为同一个浮点数
		mean = _mm_add_ps(mean, _mm_mul_ps(colorX, weight));		//加权和
		totalWeight = _mm_add_ps(totalWeight, weight);
	}
	mean = _mm_div_ps(mean, totalWeight);	//Data3~0: 1.0RGB

	q = cv::Mat(1, 3, CV_32FC1);						//weighted mean color
	q.at<float>(0, 0) = mean.m128_f32[0];		//B
	q.at<float>(0, 1) = mean.m128_f32[1];		//G
	q.at<float>(0, 2) = mean.m128_f32[2];		//R

												//compute weighted covariance matrix
												/************************************************************************/
												/* Cov(R,R)		Cov(R,G)		Cov(R,B)
												Cov(G,R)		Cov(G,G)		Cov(G,B)
												Cov(B,R)		Cov(B,G)		Cov(B,B)*/
												/************************************************************************/
	__m128 cov_XX_BB_GG_RR = _mm_setzero_ps();
	__m128 cov_XX_BR_GB_RG = _mm_setzero_ps();
	for (size_t k = 0; k<elemNum; k++)
	{
		color = sample_set.at(k).first;
		colorX = _mm_set_ps(1.0f, color[2], color[1], color[0]);	//Data0~3: BGR1.0
		__m128 weight = _mm_set_ps1(sample_set.at(k).second);	//四个元素都设为同一个浮点数
		__m128 offsetXBGR = _mm_sub_ps(colorX, mean);	//(X-mean(X))
		cov_XX_BB_GG_RR = _mm_add_ps(cov_XX_BB_GG_RR, _mm_mul_ps(_mm_mul_ps(offsetXBGR, offsetXBGR), weight));

		__m128 offsetXRBG = _mm_shuffle_ps(offsetXBGR, offsetXBGR, _MM_SHUFFLE(3, 0, 2, 1));	//调整data位置，（洗牌）
		cov_XX_BR_GB_RG = _mm_add_ps(cov_XX_BR_GB_RG, _mm_mul_ps(_mm_mul_ps(offsetXBGR, offsetXRBG), weight));
	}

	__m128 covDelta = _mm_set_ps1(COVARIANCE_DELTA);
	cov_XX_BB_GG_RR = _mm_add_ps(_mm_div_ps(cov_XX_BB_GG_RR, totalWeight), covDelta);
	cov_XX_BR_GB_RG = _mm_add_ps(_mm_div_ps(cov_XX_BR_GB_RG, totalWeight), covDelta);

	covMatrix.ptr<float>(0)[0] = cov_XX_BB_GG_RR.m128_f32[0];	//Cov(R,R)
	covMatrix.ptr<float>(1)[1] = cov_XX_BB_GG_RR.m128_f32[1];	//Cov(G,G)
	covMatrix.ptr<float>(2)[2] = cov_XX_BB_GG_RR.m128_f32[2];	//Cov(B,B)

	covMatrix.ptr<float>(0)[1] = cov_XX_BR_GB_RG.m128_f32[0];	//Cov(R,G)
	covMatrix.ptr<float>(0)[2] = cov_XX_BR_GB_RG.m128_f32[2];	//Cov(R,B)
	covMatrix.ptr<float>(1)[2] = cov_XX_BR_GB_RG.m128_f32[1];	//Cov(G,B)

	covMatrix.ptr<float>(1)[0] = cov_XX_BR_GB_RG.m128_f32[0];	//Cov(R,G)
	covMatrix.ptr<float>(2)[0] = cov_XX_BR_GB_RG.m128_f32[2];	//Cov(R,B)
	covMatrix.ptr<float>(2)[1] = cov_XX_BR_GB_RG.m128_f32[1];	//Cov(G,B)

																//finds eigenvalues and eigenvectors of a symmetric matrix
	cv::eigen(covMatrix, eigenvalues, eigenvectors);

	e.at<float>(0, 0) = eigenvectors.at<float>(0, 0);
	e.at<float>(1, 0) = eigenvectors.at<float>(0, 1);
	e.at<float>(2, 0) = eigenvectors.at<float>(0, 2);

	lambda = eigenvalues.at<float>(0, 0);	//the greatest eigenvalue 
}
