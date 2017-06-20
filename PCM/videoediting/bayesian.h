#pragma once
#include "VideoEdittingParameter.h"
#include "Cluster.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


class BayesianMatting
{
public:
	BayesianMatting(const cv::Mat& cImg, const cv::Mat& tmap);

	~BayesianMatting(void);

	void SetParameters(int nearest, double sigma, double sigma_c);

	virtual void Solve(void);		//Ðéº¯Êý

	void Composite(const cv::Mat &composite, cv::Mat *result);

	void Composite_SSE(const cv::Mat &composite, cv::Mat *result);

	void getResultForeground(cv::Mat &result);

	void getAlphaResult(cv::Mat &alphaResult)
	{
		alphaResult = this->alphamap;
	}

protected:
	void Initialize(void);

	float BayesianMatting::InitAlpha(const int x, const int y);

	void CollectFgSamples(const int x, const int y, std::vector<std::pair<cv::Vec3f, float> > *sample_set);

	void CollectBgSamples(const int x, const int y, std::vector<std::pair<cv::Vec3f, float> > *sample_set);

	float max_lambda(const std::vector<Cluster> &nodes, int *idx);

	void Split(std::vector<Cluster> *nodes);

	void Cluster_OrchardBouman(
		const int x, const int y,
		const std::vector<std::pair<cv::Vec3f, float> > &sample_set,
		std::vector<Cluster> *clusters);

	void AddCamVar(std::vector<Cluster> *clusters);

	void Optimize(
		const int x, const int y,
		const std::vector<Cluster> &fg_clusters, const std::vector<Cluster> &bg_clusters, const float alpha_init,
		cv::Vec3f *F, cv::Vec3f *B, float *a);

	void Optimize_SSE(
		const int x, const int y,
		const std::vector<Cluster> &fg_clusters, const std::vector<Cluster> &bg_clusters, const float alpha_init,
		cv::Vec3f *F, cv::Vec3f *B, float *a);

	float ComputeLikelihood(
		const int x, const int y,
		const cv::Mat &mu_Fi, const cv::Mat &invSigma_Fi,
		const cv::Mat &mu_Bj, const cv::Mat &invSigma_Bj,
		const cv::Vec3f &c_color, const cv::Vec3f &fg_color, const cv::Vec3f &bg_color, const float alpha);

	float ComputeLikelihood_SSE_for_original(
		const int x, const int y,
		const cv::Mat &mu_Fi, const cv::Mat &invSigma_Fi,
		const cv::Mat &mu_Bj, const cv::Mat &invSigma_Bj,
		const cv::Vec3f &c_color, const cv::Vec3f &fg_color, const cv::Vec3f &bg_color, const float alpha);

	float ComputeLikelihood_SSE(
		const int x, const int y,
		const __m128 &mMeanF, const cv::Mat &invSigma_Fi,
		const __m128 &mMeanB, const cv::Mat &invSigma_Bj,
		const __m128 &mC, const __m128 &mF, const __m128 &mB, const __m128 &mAlpha);

	unsigned nUnknown;
	unsigned nearest;
	double sigma;
	double sigma_c;

	cv::Mat fgmask, bgmask, unmask;
	cv::Mat unsolvedmask;

	cv::Mat trimap;
	cv::Mat alphamap;
	cv::Size imgSize;
	cv::Mat colorImg, fgImg, bgImg;
};