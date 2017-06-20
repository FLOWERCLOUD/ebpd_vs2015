#pragma once
#include "bayesian.h"

class MyBayesian :
	public BayesianMatting		//继承自BayesianMatting
{
public:
	MyBayesian::MyBayesian(const cv::Mat& cImg, const cv::Mat& tmap, int lamda, int IterationTimes);
	~MyBayesian(void);

	void Solve(void);
	void setIterationTimes(int it)
	{
		IterationTimes = it;
	}

	void setLamda(int lamda)
	{
		this->lamda = lamda;
	}

private:
	void sumNeighboor(double **p);		//8邻域像素alpha求和
	void ComputeWholeImageWeight(double** pW);
	bool neighboor_count_tag(double** pW, int row, int col, int count_tag);
	bool isBeingConsidered(int row, int col);
	void resetMaskArry();
	void solveSparseMatrix_alpha_weight(double sigma_C, double**pW, int*pNeighboorIndex, double** pSumN);
	void getNeighboorSum(double Neighboor[], double** p, int row, int col);
	void neighboorIndex_to_coordinate(int m, int row, int col, int& Cirow, int& Cicol);


private:
	int IterationTimes;		//迭代次数
	float lamda, lamda_bayes;
	std::vector<cv::Point> CurrentComputeArrary;		//当前计算像素?
};