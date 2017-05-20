#pragma once
#define COVARIANCE_DELTA 1e-5f			// 一个小值，防止协方差矩阵为0矩阵
#include <vector>
#include <opencv2/core/core.hpp>

class Cluster
{
public:
	Cluster(void);

	Cluster(const std::vector<std::pair<cv::Vec3f, float> > &sample_set);

	void Calc(void);
	void calc_SSE(void);
	std::vector<std::pair<cv::Vec3f, float> > sample_set;	//sample point and corresponding weight

	cv::Mat q;	//weighted mean color
	cv::Mat covMatrix;	//weighted covariance matrix
	cv::Mat eigenvalues, eigenvectors, e;	//e:transpose of eigenvectors
	float lambda;	//the greatest eigenvalue


};