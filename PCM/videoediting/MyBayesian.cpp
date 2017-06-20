#include "MyBayesian.h"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <unsupported/Eigen/SparseExtra>
using namespace cv;
using namespace std;
MyBayesian::MyBayesian(const cv::Mat& cImg, const cv::Mat& tmap, int lamda, int IterationTimes) :BayesianMatting(cImg, tmap)
{
	this->lamda = lamda;
	this->IterationTimes = IterationTimes;
}

MyBayesian::~MyBayesian(void)
{
}

void MyBayesian::Solve(void)
{
	int *index = new int[imgSize.width*imgSize.height];		//记录未知像素
	double** pSumN = new double*[imgSize.height], **pW_weight = new double*[imgSize.height];
	for (int i = 0; i < imgSize.height; i++)
	{
		pSumN[i] = new double[imgSize.width];
		pW_weight[i] = new double[imgSize.width];
	}

	BayesianMatting::Solve();		//计算初始F,B,alpha

	nUnknown = 0;
	CurrentComputeArrary.clear();
	for (int i = 0, ith = 0; i < imgSize.height; i++)
	{
		for (int j = 0; j < imgSize.width; j++, ith++)
		{
			if (trimap.at<uchar>(i, j) == MASK_COMPUTE)	//i行j列
			{
				Point pos;
				pos.x = i, pos.y = j;	//pos.x-->row, pos.y-->col
				CurrentComputeArrary.push_back(pos);
				index[ith] = nUnknown++;	//第几个未知像素
			}
			else
				index[ith] = -1;
		}
	}

	if (0 == nUnknown)
		return;

	sumNeighboor(pSumN);
	ComputeWholeImageWeight(pW_weight);	//平滑项wi

	while (IterationTimes--)
	{
		lamda_bayes = 1.0 / (1 + lamda);
		solveSparseMatrix_alpha_weight(sigma_c, pW_weight, index, pSumN);		//新算法算alpha
	}

	//delete
	for (int i = 0; i < imgSize.height; i++)
	{
		delete pSumN[i];
		delete pW_weight[i];
	}
	delete[] pSumN; pSumN = NULL;
	delete[] pW_weight; pW_weight = NULL;
	delete[] index; index = NULL;
}

void MyBayesian::sumNeighboor(double **p)
{
	for (int i = 0; i < imgSize.height; i++)
	{
		for (int j = 0; j < imgSize.width; j++)
		{
			p[i][j] = 0;
			if (i > 0 && j > 0)
			{
				if (trimap.at<uchar>(i - 1, j - 1) != MASK_COMPUTE)
					p[i][j] += alphamap.at<float>(i - 1, j - 1);
			}

			if (i > 0)
			{
				if (trimap.at<uchar>(i - 1, j) != MASK_COMPUTE)
					p[i][j] += alphamap.at<float>(i - 1, j);
			}

			if (i > 0 && j < imgSize.width - 1)
			{
				if (trimap.at<uchar>(i - 1, j + 1) != MASK_COMPUTE)
					p[i][j] += alphamap.at<float>(i - 1, j + 1);
			}

			if (j > 0)
			{
				if (trimap.at<uchar>(i, j - 1) != MASK_COMPUTE)
					p[i][j] += alphamap.at<float>(i, j - 1);
			}

			if (j < imgSize.width - 1)
			{
				if (trimap.at<uchar>(i, j + 1) != MASK_COMPUTE)
					p[i][j] += alphamap.at<float>(i, j + 1);
			}

			if (i < imgSize.height - 1 && j>0)
			{
				if (trimap.at<uchar>(i + 1, j - 1) != MASK_COMPUTE)
					p[i][j] += alphamap.at<float>(i + 1, j - 1);
			}

			if (i < imgSize.height - 1)
			{
				if (trimap.at<uchar>(i + 1, j) != MASK_COMPUTE)
					p[i][j] += alphamap.at<float>(i + 1, j);
			}

			if (i < imgSize.height - 1 && j < imgSize.width - 1)
			{
				if (trimap.at<uchar>(i + 1, j + 1) != MASK_COMPUTE)
					p[i][j] += alphamap.at<float>(i + 1, j + 1);
			}
		}
	}
}

void MyBayesian::ComputeWholeImageWeight(double** pW)
{
	for (int i = 0; i < imgSize.height; i++)
	{
		for (int j = 0; j < imgSize.width; j++)
		{
			if (trimap.at<uchar>(i, j) != MASK_COMPUTE && isBeingConsidered(i, j))	//参与计算的已知像素
				pW[i][j] = 1.0;
			else		//未知像素
				pW[i][j] = 0.0;
		}
	}

	int count_tag = 1;
	bool exchange = true;
	while (exchange)
	{
		exchange = false;
		for (int i = 0; i < nUnknown; i++)
		{
			int row = CurrentComputeArrary.at(i).x, col = CurrentComputeArrary.at(i).y;
			if (pW[row][col] == 0 && neighboor_count_tag(pW, row, col, count_tag))
			{
				pW[row][col] = (double)(count_tag + 1);	//到已知区域的最小距离
				exchange = true;
			}
		}
		count_tag++;
	}

	for (int i = 0; i < imgSize.height; i++)
		for (int j = 0; j < imgSize.width; j++)
		{
			pW[i][j] = exp(-(pW[i][j] * pW[i][j]) / (SIGMA_WEIGHT*SIGMA_WEIGHT));
		}
}

//在(row, col)的8邻域内有未知像素，即是否为与未知像素相邻的已知像素
bool MyBayesian::isBeingConsidered(int row, int col)
{
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			if (row + i >= 0 && row + i < imgSize.height && col + j >= 0 && col + j < imgSize.width && trimap.at<uchar>(row + i, col + j) == MASK_COMPUTE)
				return true;
		}
	}
	return false;
}

//邻域内像素有可到达计算区域的像素，count_tag+1为最少经过多少距离到达已知区域
bool MyBayesian::neighboor_count_tag(double** pW, int row, int col, int count_tag)
{
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			if (i == 0 && j == 0) continue;
			if (row + i >= 0 && row + i < imgSize.height && col + j >= 0 && col + j < imgSize.width && pW[row + i][col + j] == count_tag)
				return true;
		}
	}
	return false;
}

void MyBayesian::solveSparseMatrix_alpha_weight(double sigma_C, double**pW, int*pNeighboorIndex, double** pSumN)
{
	using namespace Eigen;
	DynamicSparseMatrix<double> A(nUnknown, nUnknown);	//Ax+b=0;
	MatrixXd b(nUnknown, 1), x(nUnknown, 1);
	A.setZero();
	b.setZero();
	x.setZero();

	for (int i = 0; i < nUnknown; i++)
	{
		int row = CurrentComputeArrary[i].x, col = CurrentComputeArrary[i].y;
		float Cr, Cg, Cb, Fr, Fg, Fb, Br, Bg, Bb;
		double NeighboorSumAlpha[9], constantB = 0;

		Fr = fgImg.at<Vec3f>(row, col)[2];
		Fg = fgImg.at<Vec3f>(row, col)[1];
		Fb = fgImg.at<Vec3f>(row, col)[0];
		Br = bgImg.at<Vec3f>(row, col)[2];
		Bg = bgImg.at<Vec3f>(row, col)[1];
		Bb = bgImg.at<Vec3f>(row, col)[0];
		Cr = colorImg.at<Vec3f>(row, col)[2];
		Cg = colorImg.at<Vec3f>(row, col)[1];
		Cb = colorImg.at<Vec3f>(row, col)[0];

		getNeighboorSum(NeighboorSumAlpha, pSumN, row, col); //获取当前像素(row,col)的8个像素每个像素邻域alpha和。用于累计b.coeff;
		A.coeffRef(i, i) = pW[row][col] * lamda;
		constantB += NeighboorSumAlpha[8] * pW[row][col] * lamda / 8;

		for (int m = 0; m < 8; m++)
		{
			int m_index, m_row, m_col;
			neighboorIndex_to_coordinate(m, row, col, m_row, m_col);//8个邻域转化成坐标
			m_index = pNeighboorIndex[m_row*imgSize.width + m_col];                   //获取邻域的index。

																					  //找到像素周围位置像素，使其系数+pW[m_row][m_col]/64
			for (int k = 0; k < 8; k++)
			{
				int  k_row, k_col;
				neighboorIndex_to_coordinate(k, m_row, m_col, k_row, k_col);
				if (pNeighboorIndex[k_row*imgSize.width + k_col] >= 0)
					A.coeffRef(i, pNeighboorIndex[k_row*imgSize.width + k_col]) += pW[m_row][m_col] * lamda / 64;
			}

			//下面是已知的邻域与未知邻域处理不一样。
			if (m_index < 0)   //这个邻域为已知的
				constantB += (alphamap.at<float>(m_row, m_col) / 8 - NeighboorSumAlpha[m] / 64)*pW[m_row][m_col] * lamda;
			else                      //这个邻域为未知的
			{
				A.coeffRef(i, m_index) -= (pW[row][col] + pW[m_row][m_col])*lamda / 8;
				constantB -= pW[m_row][m_col] * NeighboorSumAlpha[m] * lamda / 64;
			}
		}	//for(m)
		A.coeffRef(i, i) += ((Br - Fr)*(Br - Fr) + (Bg - Fg)*(Bg - Fg) + (Bb - Fb)*(Bb - Fb)) * lamda_bayes / (sigma_C*sigma_C);
		b.coeffRef(i, 0) = constantB + ((Cr - Br)*(Fr - Br) + (Cg - Bg)*(Fg - Bg) + (Cb - Bb)*(Fb - Bb)) * lamda_bayes / (sigma_C*sigma_C);
	}//for(i)


#define UMFPACK_METHOD 0
#define	SPARSEQR_METHOD 1
#define	SPARSELU_METHOD 2
#define SOLVE_METHOD 	SPARSELU_METHOD
#if   SOLVE_METHOD  == UMFPACK_METHOD //i don't have 64 bit umpack lib ,so not use umfpack
	SparseMatrix<double> solverX(A);
	SparseLU<SparseMatrix<double>, UmfPack >  lu_ofX(solverX);
	lu_ofX.solve(b, &x);

#else 
#if   SOLVE_METHOD  == SPARSELU_METHOD
	 //use sparse LU ,its speed is quite quick then QR
	SparseLU< SparseMatrix<double>, AMDOrdering < int > >   solver;
	// fill A and b;
	// Compute the ordering permutation vector from the structural pattern of A
	//		solver.analyzePattern(A);
	// Compute the numerical factorization 
	//	solver.factorize(A);
	//Use the factors to solve the linear system 

	solver.compute(A);
	x = solver.solve(b);
	cout << "LU" << endl;
#else

#if SOLVE_METHOD == SPARSEQR_METHOD
	 //QR METHOD IS A LOT SLOWER THEN LU
	SparseQR < SparseMatrix < double >, AMDOrdering < int > > qr;
	// 计算分解
	qr.compute(A);
	x = qr.solve(b);
	cout << "QR" << endl;
#endif
#endif



#endif // 0


	for (int i = 0; i < nUnknown; i++)
	{
		int row = CurrentComputeArrary.at(i).x, col = CurrentComputeArrary.at(i).y;
		alphamap.at<float>(row, col) = MAX(MIN(x.coeff(i, 0), 1), 0);
	}
}

void MyBayesian::getNeighboorSum(double Neighboor[], double** p, int row, int col)
{
	for (int i = 0; i < 8; i++)
		Neighboor[i] = 0;

	if (row > 0 && col > 0)
		Neighboor[0] = p[row - 1][col - 1];

	if (row > 0)
		Neighboor[1] = p[row - 1][col];

	if (row > 0 && col < imgSize.width - 1)
		Neighboor[2] = p[row - 1][col + 1];

	if (col > 0)
		Neighboor[3] = p[row][col - 1];

	if (col < imgSize.width - 1)
		Neighboor[4] = p[row][col + 1];

	if (row < imgSize.height - 1 && col>0)
		Neighboor[5] = p[row + 1][col - 1];

	if (row < imgSize.height - 1)
		Neighboor[6] = p[row + 1][col];

	if (row < imgSize.height - 1 && col < imgSize.width - 1)
		Neighboor[7] = p[row + 1][col + 1];

	Neighboor[8] = p[row][col];
}

void MyBayesian::neighboorIndex_to_coordinate(int m, int row, int col, int&Cirow, int& Cicol)
{
	switch (m)  //找出这个邻域的坐标
	{
	case 0: Cirow = min(max(row - 1, 0), imgSize.height - 1);  Cicol = min(max(col - 1, 0), imgSize.width - 1); break;
	case 1: Cirow = min(max(row - 1, 0), imgSize.height - 1);  Cicol = min(max(col, 0), imgSize.width - 1); break;
	case 2: Cirow = min(max(row - 1, 0), imgSize.height - 1);  Cicol = min(max(col + 1, 0), imgSize.width - 1); break;
	case 3: Cirow = min(max(row, 0), imgSize.height - 1);     Cicol = min(max(col - 1, 0), imgSize.width - 1); break;
	case 4: Cirow = min(max(row, 0), imgSize.height - 1);     Cicol = min(max(col + 1, 0), imgSize.width - 1); break;
	case 5: Cirow = min(max(row + 1, 0), imgSize.height - 1); Cicol = min(max(col - 1, 0), imgSize.width - 1); break;
	case 6: Cirow = min(max(row + 1, 0), imgSize.height - 1); Cicol = min(max(col, 0), imgSize.width - 1); break;
	case 7: Cirow = min(max(row + 1, 0), imgSize.height - 1); Cicol = min(max(col + 1, 0), imgSize.width - 1); break;
	default: break;
	}
}