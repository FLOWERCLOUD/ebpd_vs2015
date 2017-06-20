#include "bayesian.h"
#include <iostream>
#include <climits>
#include <fstream>
using namespace std;
using namespace cv;

BayesianMatting::BayesianMatting(const cv::Mat& cImg, const cv::Mat& tmap)
{
	CV_Assert(cImg.size() == tmap.size());
	imgSize = cImg.size();
	cImg.convertTo(colorImg, CV_32F, 1.0f / 255.0f);	//Init colorImg

														// Convert the trimap into a single channel image
	if (tmap.channels() == 3)
	{
		cv::cvtColor(tmap, trimap, CV_RGB2GRAY);  //RGB[A] TO Gray : y = 0.299*R + 0.587*G +0.114*B
	}
	else if (tmap.channels() == 1)
	{
		this->trimap = tmap.clone();
	}

	Initialize();

	SetParameters(ADJECENCY_N, SIGMA, SIGMA_C);
}

BayesianMatting::~BayesianMatting(void)
{
}

void BayesianMatting::Initialize(void)
{
	nUnknown = 0;

	this->fgImg = cv::Mat(imgSize, CV_32FC3, cv::Scalar(0, 0, 0));	//foreground
	this->bgImg = cv::Mat(imgSize, CV_32FC3, cv::Scalar(0, 0, 0));	//background
	this->fgmask = cv::Mat(imgSize, CV_8UC1, cv::Scalar(0));	//foreground mask
	this->bgmask = cv::Mat(imgSize, CV_8UC1, cv::Scalar(0));	//background mask
	this->unmask = cv::Mat(imgSize, CV_8UC1, cv::Scalar(0));	//unknown mask
	this->alphamap = cv::Mat(imgSize, CV_32FC1, cv::Scalar(0));

	for (int y = 0; y < imgSize.height; y++)
	{
		for (int x = 0; x < imgSize.width; x++)
		{
			uchar v = trimap.at<uchar>(y, x);
			if (v == MASK_BACKGROUND)	//if background
			{
				bgmask.at<uchar>(y, x) = 255;
			}
			else if (v == MASK_FOREGROUND)	//if foreground
			{
				fgmask.at<uchar>(y, x) = 255;
			}
			else	// if unknown
			{
				unmask.at<uchar>(y, x) = 255;
			}
		}
	}

	this->colorImg.copyTo(fgImg, fgmask);
	this->colorImg.copyTo(bgImg, bgmask);
	//Initialize alphamap
	this->alphamap.setTo(cv::Scalar(0), bgmask);
	this->alphamap.setTo(cv::Scalar(1), fgmask);
	this->alphamap.setTo(cv::Scalar(-1.0f), unmask);
	this->unsolvedmask = unmask.clone();

	for (int r = 0; r < unmask.rows; r++)
	{
		for (int c = 0; c < unmask.cols; c++)
		{
			if (unmask.at<uchar>(r, c) == 255)
			{
				nUnknown = nUnknown + 1;
			}
		}
	}

	/*std::cout << "nUnknown = " << nUnknown << std::endl;
	cv::imshow("fgmask", fgmask);
	cv::imshow("bgmask", bgmask);
	cv::imshow("unmask", unmask);
	cv::imshow("fgImg", fgImg);
	cv::imshow("bgImg", bgImg);
	cv::imshow("alpha", alphamap);
	cv::waitKey(0);*/
}

void BayesianMatting::SetParameters(int nearest, double sigma, double sigma_c)
{
	this->nearest = nearest;
	this->sigma = sigma;
	this->sigma_c = sigma_c;
}

//Init alpha for pixel(x,y) by the mean of N known neighbors
float BayesianMatting::InitAlpha(const int x, const int y)
{
	unsigned alpha_num = 0;
	float alpha_init = 0;
	int dist = 1;

	while (alpha_num < nearest)
	{
		if (y - dist >= 0)
		{
			for (int z = std::max(0, x - dist); z <= std::min(alphamap.cols - 1, x + dist); z++)
			{
				// We know this pixel belongs to the foreground
				if (fgmask.at<uchar>(y - dist, z) != 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(y - dist, z);
					alpha_num = alpha_num + 1;
					if (alpha_num == nearest)
					{
						goto DONE;
					}
				}
				else if (unmask.at<uchar>(y - dist, z) != 0 && unsolvedmask.at<uchar>(y - dist, z) == 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(y - dist, z);
					alpha_num = alpha_num + 1;
					if (alpha_num == nearest)
					{
						goto DONE;
					}
				}
				else if (bgmask.at<uchar>(y - dist, z) != 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(y - dist, z);
					alpha_num = alpha_num + 1;
					if (alpha_num == nearest)
					{
						goto DONE;
					}
				}
			}
		}

		if (y + dist < alphamap.rows)
		{
			for (int z = std::max(0, x - dist); z <= std::min(alphamap.cols - 1, x + dist); z++)
			{
				if (fgmask.at<uchar>(y + dist, z) != 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(y + dist, z);
					alpha_num = alpha_num + 1;
					if (alpha_num == nearest)
					{
						goto DONE;
					}
				}
				else if (unmask.at<uchar>(y + dist, z) != 0 && unsolvedmask.at<uchar>(y + dist, z) == 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(y + dist, z);
					alpha_num = alpha_num + 1;
					if (alpha_num == nearest)
					{
						goto DONE;
					}
				}
				else if (bgmask.at<uchar>(y + dist, z) != 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(y + dist, z);
					alpha_num = alpha_num + 1;
					if (alpha_num == nearest)
					{
						goto DONE;
					}
				}
			}
		}

		if (x - dist >= 0)
		{
			for (int z = std::max(0, y - dist + 1); z <= std::min(alphamap.rows - 1, y + dist - 1); z++)
			{
				if (fgmask.at<uchar>(z, x - dist) != 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(z, x - dist);
					alpha_num = alpha_num + 1;
					if (alpha_num == nearest)
					{
						goto DONE;
					}
				}
				else if (unmask.at<uchar>(z, x - dist) != 0 && unsolvedmask.at<uchar>(z, x - dist) == 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(z, x - dist);
					alpha_num = alpha_num + 1;
					if (alpha_num == nearest)
					{
						goto DONE;
					}
				}
				else if (bgmask.at<uchar>(z, x - dist) != 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(z, x - dist);
					alpha_num = alpha_num + 1;
					if (alpha_num == nearest)
					{
						goto DONE;
					}
				}
			}
		}

		if (x + dist < alphamap.cols)
		{
			for (int z = std::max(0, y - dist + 1); z <= std::min(alphamap.rows - 1, y + dist - 1); z++)
			{
				if (fgmask.at<uchar>(z, x + dist) != 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(z, x + dist);
					alpha_num = alpha_num + 1;
					if (alpha_num == nearest)
					{
						goto DONE;
					}
				}
				else if (unmask.at<uchar>(z, x + dist) != 0 && unsolvedmask.at<uchar>(z, x + dist) == 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(z, x + dist);
					alpha_num = alpha_num + 1;
					if (alpha_num == nearest)
					{
						goto DONE;
					}
				}
				else if (bgmask.at<uchar>(z, x + dist) != 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(z, x + dist);
					alpha_num = alpha_num + 1;
					if (alpha_num == nearest)
					{
						goto DONE;
					}
				}
			}
		}

		dist = dist + 1;
	}

DONE:
	return (alpha_init / alpha_num);
}

//collect N foreground samples for (x,y) pixel,compute weight for each sample,由内向外，dist++
void BayesianMatting::CollectFgSamples(const int x, const int y, std::vector<std::pair<cv::Vec3f, float> > *sample_set)
{
	sample_set->clear();

	std::pair<cv::Vec3f, float> sample;
	float dist_weight;
	float inv_2sigma_square = 1.0f / (2.0f * this->sigma * this->sigma);	// for Gaussian fall-off σ=8
	int dist = 1;

	while (sample_set->size() < nearest)	//nearest=N
	{
		if (y - dist >= 0)	//up
		{
			for (int z = std::max(0, x - dist); z <= std::min(fgImg.cols - 1, x + dist); z++)
			{
				dist_weight = std::expf(-(dist * dist + (z - x) * (z - x)) * inv_2sigma_square);	//dist=(y1-y2),z-x=(x1-x2)

				if (fgmask.at<uchar>(y - dist, z) != 0)	//if foreground
				{
					sample.first = fgImg.at<cv::Vec3f>(y - dist, z);	//value of pixels
					sample.second = dist_weight;	//weight (foreground alpha=1)
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
				else if (unmask.at<uchar>(y - dist, z) != 0 && unsolvedmask.at<uchar>(y - dist, z) == 0)	//if unknown area but have been solved
				{
					sample.first = fgImg.at<cv::Vec3f>(y - dist, z);
					float alpha = alphamap.at<float>(y - dist, z);
					sample.second = dist_weight * alpha * alpha;	//alpha ranges (0,1)
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
			}
		}

		if (y + dist < fgImg.rows)	//down
		{
			for (int z = std::max(0, x - dist); z <= std::min(fgImg.cols - 1, x + dist); z++)
			{
				dist_weight = std::expf(-(dist * dist + (z - x) * (z - x)) * inv_2sigma_square);

				if (fgmask.at<uchar>(y + dist, z) != 0)
				{
					sample.first = fgImg.at<cv::Vec3f>(y + dist, z);
					sample.second = dist_weight;
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
				else if (unmask.at<uchar>(y + dist, z) != 0 && unsolvedmask.at<uchar>(y + dist, z) == 0)
				{
					sample.first = fgImg.at<cv::Vec3f>(y + dist, z);
					float alpha = alphamap.at<float>(y + dist, z);
					sample.second = dist_weight * alpha * alpha;
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
			}
		}

		if (x - dist >= 0)	//left
		{
			for (int z = std::max(0, y - dist + 1); z <= std::min(fgImg.rows - 1, y + dist - 1); z++)
			{
				dist_weight = std::expf(-(dist * dist + (z - y) * (z - y)) * inv_2sigma_square);	//z-y=(y1-y2),dist=(x1-x2)

				if (fgmask.at<uchar>(z, x - dist) != 0)
				{
					sample.first = fgImg.at<cv::Vec3f>(z, x - dist);
					sample.second = dist_weight;
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
				else if (unmask.at<uchar>(z, x - dist) != 0 && unsolvedmask.at<uchar>(z, x - dist) == 0)
				{
					sample.first = fgImg.at<cv::Vec3f>(z, x - dist);
					float alpha = alphamap.at<float>(z, x - dist);
					sample.second = dist_weight * alpha * alpha;
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
			}
		}

		if (x + dist < fgImg.cols)	//right
		{
			for (int z = std::max(0, y - dist + 1); z <= std::min(fgImg.rows - 1, y + dist - 1); z++)
			{
				dist_weight = std::expf(-(dist * dist + (y - z) * (y - z)) * inv_2sigma_square);

				if (fgmask.at<uchar>(z, x + dist) != 0)
				{
					sample.first = fgImg.at<cv::Vec3f>(z, x + dist);
					sample.second = dist_weight;
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
				else if (unmask.at<uchar>(z, x + dist) != 0 && unsolvedmask.at<uchar>(z, x + dist) == 0)
				{
					sample.first = fgImg.at<cv::Vec3f>(z, x + dist);
					float alpha = alphamap.at<float>(z, x + dist);
					sample.second = dist_weight * alpha * alpha;
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
			}
		}

		dist = dist + 1;
	}

DONE:
	CV_Assert(sample_set->size() == nearest);
}

//collect N background samples for (x,y) pixel,compute weight for each sample
void BayesianMatting::CollectBgSamples(const int x, const int y, std::vector<std::pair<cv::Vec3f, float> > *sample_set)
{
	sample_set->clear();

	std::pair<cv::Vec3f, float> sample;
	float dist_weight;
	float inv_2sigma_square = 1.0f / (2.0f * this->sigma * this->sigma);
	int dist = 1;

	while (sample_set->size() < nearest)
	{
		if (y - dist >= 0)
		{
			for (int z = std::max(0, x - dist); z <= std::min(bgImg.cols - 1, x + dist); z++)
			{
				dist_weight = std::expf(-(dist * dist + (z - x) * (z - x)) * inv_2sigma_square);

				if (bgmask.at<uchar>(y - dist, z) != 0)
				{
					sample.first = bgImg.at<cv::Vec3f>(y - dist, z);
					sample.second = dist_weight;	//1-alpha=1
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
				else if (unmask.at<uchar>(y - dist, z) != 0 && unsolvedmask.at<uchar>(y - dist, z) == 0)
				{
					sample.first = bgImg.at<cv::Vec3f>(y - dist, z);
					float alpha = alphamap.at<float>(y - dist, z);
					sample.second = dist_weight * (1 - alpha) * (1 - alpha);
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
			}
		}

		if (y + dist < bgImg.rows)
		{
			for (int z = std::max(0, x - dist); z <= std::min(bgImg.cols - 1, x + dist); z++)
			{
				dist_weight = std::expf(-(dist * dist + (z - x) * (z - x)) * inv_2sigma_square);

				if (bgmask.at<uchar>(y + dist, z) != 0)
				{
					sample.first = bgImg.at<cv::Vec3f>(y + dist, z);
					sample.second = dist_weight;
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
				else if (unmask.at<uchar>(y + dist, z) != 0 && unsolvedmask.at<uchar>(y + dist, z) == 0)
				{
					sample.first = bgImg.at<cv::Vec3f>(y + dist, z);
					float alpha = alphamap.at<float>(y + dist, z);
					sample.second = dist_weight * (1 - alpha) * (1 - alpha);
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
			}
		}

		if (x - dist >= 0)
		{
			for (int z = std::max(0, y - dist + 1); z <= std::min(bgImg.rows - 1, y + dist - 1); z++)
			{
				dist_weight = std::expf(-(dist * dist + (z - y) * (z - y)) * inv_2sigma_square);

				if (bgmask.at<uchar>(z, x - dist) != 0)
				{
					sample.first = bgImg.at<cv::Vec3f>(z, x - dist);
					sample.second = dist_weight;
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
				else if (unmask.at<uchar>(z, x - dist) != 0 && unsolvedmask.at<uchar>(z, x - dist) == 0)
				{
					sample.first = bgImg.at<cv::Vec3f>(z, x - dist);
					float alpha = alphamap.at<float>(z, x - dist);
					sample.second = dist_weight * (1 - alpha) * (1 - alpha);
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
			}
		}

		if (x + dist < bgImg.cols)
		{
			for (int z = std::max(0, y - dist + 1); z <= std::min(bgImg.rows - 1, y + dist - 1); z++)
			{
				dist_weight = std::expf(-(dist * dist + (y - z) * (y - z)) * inv_2sigma_square);

				if (bgmask.at<uchar>(z, x + dist) != 0)
				{
					sample.first = bgImg.at<cv::Vec3f>(z, x + dist);
					sample.second = dist_weight;
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
				else if (unmask.at<uchar>(z, x + dist) != 0 && unsolvedmask.at<uchar>(z, x + dist) == 0)
				{
					sample.first = bgImg.at<cv::Vec3f>(z, x + dist);
					float alpha = alphamap.at<float>(z, x + dist);
					sample.second = dist_weight * (1 - alpha) * (1 - alpha);
					sample_set->push_back(sample);

					if (sample_set->size() == nearest)
					{
						goto DONE;
					}
				}
			}
		}

		dist = dist + 1;
	}

DONE:
	CV_Assert(sample_set->size() == nearest);
}

void BayesianMatting::Solve(void)
{
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat unkreg = unsolvedmask.clone();	//unknown region
	cv::Mat unkreg_not;
	cv::Mat unkpixels;
	std::vector<cv::Point> toSolveList;

	std::vector<std::pair<cv::Vec3f, float> > fg_set;	//foreground sample set
	std::vector<std::pair<cv::Vec3f, float> > bg_set;	//background sample set

	std::vector<Cluster> fg_clusters;
	std::vector<Cluster> bg_clusters;

	float alpha_init;	//mean of pixel(x,y)'s N known neighbors

	cv::Vec3f f;
	cv::Vec3f b;
	float alpha;

	unsigned loop = 1;
	unsigned n = 0;

	while (n < nUnknown)
	{
		/*std::cout << std::endl << "Starting iteration: " << loop << std::endl;*/

		//Find the boundary of unsolved unknown region, i.e. the pixels to be solved
		cv::erode(unkreg, unkreg, element);	//erode
		cv::bitwise_not(unkreg, unkreg_not);	//unkreg_not = ~unkreg		
		cv::bitwise_and(unkreg_not, unsolvedmask, unkpixels);	//unkpixels = unkreg_not & unsolvedmask

		toSolveList.clear();

		for (int r = 0; r < unkpixels.rows; r++)
		{
			for (int c = 0; c < unkpixels.cols; c++)
			{
				if (unkpixels.at<uchar>(r, c) == 255)
				{
					toSolveList.push_back(cv::Point(c, r));
				}
			}
		}

		/*std::cout << "Find " << toSolveList.size() << " pixels to be solved" << std::endl;*/

		/*cv::imshow("unmask", unmask);
		cv::imshow("unkreg", unkreg);
		cv::imshow("not", unkreg_not);
		cv::imshow("unkpixels", unkpixels);
		cv::waitKey(0);*/

		for (size_t k = 0; k < toSolveList.size(); k++)
		{
			int x = toSolveList[k].x;
			int y = toSolveList[k].y;

			alpha_init = InitAlpha(x, y);
			CollectFgSamples(x, y, &fg_set);
			CollectBgSamples(x, y, &bg_set);

			Cluster_OrchardBouman(x, y, fg_set, &fg_clusters);
			Cluster_OrchardBouman(x, y, bg_set, &bg_clusters);

			AddCamVar(&fg_clusters);
			AddCamVar(&bg_clusters);

			//Optimize(x, y, fg_clusters, bg_clusters, alpha_init, &f, &b, &alpha);

			Optimize_SSE(x, y, fg_clusters, bg_clusters, alpha_init, &f, &b, &alpha);

			// Update foreground, background and alpha values
			fgImg.at<cv::Vec3f>(y, x) = f;
			bgImg.at<cv::Vec3f>(y, x) = b;
			alphamap.at<float>(y, x) = alpha;

			// Remove solved pixels from unsolvedmask to indicate it was solved
			unsolvedmask.at<uchar>(y, x) = 0;
		}

		n = n + toSolveList.size();
		loop = loop + 1;
	}

	//std::cout << std::endl;
	//std::cout << "Total " << n << " pixels have been solved" << std::endl;

	/*cv::imshow("alphamap", alphamap);
	cv::waitKey(0);*/
	/*cv::imshow("fgImg", fgImg);
	cv::waitKey(0);*/
}

//Solve two sub-problem ,未加速的方法，现在没用
void BayesianMatting::Optimize(
	const int x, const int y,
	const std::vector<Cluster> &fg_clusters, const std::vector<Cluster> &bg_clusters, const float alpha_init,
	cv::Vec3f *F, cv::Vec3f *B, float *a)
{
	float alpha1, alpha2;
	float alpha, max_alpha;
	float like, lastLike, maxLike;
	cv::Vec3f c_color, fg_color, bg_color;
	cv::Vec3f max_fg_color, max_bg_color;

	c_color = colorImg.at<cv::Vec3f>(y, x);

	cv::Mat A = cv::Mat(6, 6, CV_32FC1, cv::Scalar(0));
	cv::Mat X = cv::Mat(6, 1, CV_32FC1, cv::Scalar(0));
	cv::Mat b = cv::Mat(6, 1, CV_32FC1, cv::Scalar(0));
	cv::Mat I = cv::Mat::eye(3, 3, CV_32FC1);

	cv::Mat work_3x3 = cv::Mat(3, 3, CV_32FC1, cv::Scalar(0));
	cv::Mat work_3x1 = cv::Mat(3, 1, CV_32FC1, cv::Scalar(0));

	cv::Mat work_c = cv::Mat(3, 1, CV_32FC1);
	cv::Mat work_f = cv::Mat(3, 1, CV_32FC1);
	cv::Mat work_b = cv::Mat(3, 1, CV_32FC1);
	cv::Mat work_1x1_n = cv::Mat(1, 1, CV_32FC1);
	cv::Mat work_1x1_d = cv::Mat(1, 1, CV_32FC1);

	max_alpha = alpha_init;
	maxLike = -FLT_MAX;

	cv::Mat mu_Fi;
	cv::Mat invSigma_Fi;

	cv::Mat mu_Bj;
	cv::Mat invSigma_Bj;

	int iter;

	//	float mlike;

	// 	float oriTime=0;
	// 	float sseTime=0;	
	// 	string TimeFile ="Time.txt";
	// 	string likeFile = "like.txt";
	// 	ofstream ftimeout(TimeFile,ios::app);
	// 	ofstream flikeout(likeFile,ios::app);

	//calculate each (F,B) pairs
	for (size_t i = 0; i < fg_clusters.size(); i++)
	{
		mu_Fi = fg_clusters[i].q.t();
		invSigma_Fi = fg_clusters[i].covMatrix.inv();

		for (size_t j = 0; j < bg_clusters.size(); j++)
		{
			mu_Bj = bg_clusters[j].q.t();
			invSigma_Bj = bg_clusters[j].covMatrix.inv();

			alpha = alpha_init;
			lastLike = -FLT_MAX;
			iter = 1;

			while (1)
			{
				//****************************************************************************//
				// Solve for foreground and background, the first sub-problem
				float inv_sigmac_square = 1.0f / (sigma_c * sigma_c);

				work_3x3 = I * (alpha * alpha * inv_sigmac_square);
				work_3x3 = work_3x3 + invSigma_Fi;

				for (int h = 0; h < 3; h++)
				{
					for (int k = 0; k < 3; k++)
					{
						A.at<float>(h, k) = work_3x3.at<float>(h, k);
					}
				}

				work_3x3 = I * (alpha * (1 - alpha) * inv_sigmac_square);
				for (int h = 0; h < 3; h++)
				{
					for (int k = 0; k < 3; k++)
					{
						A.at<float>(h, 3 + k) = work_3x3.at<float>(h, k);
						A.at<float>(3 + h, k) = work_3x3.at<float>(h, k);
					}
				}

				work_3x3 = I * (1 - alpha) * (1 - alpha) * inv_sigmac_square;
				work_3x3 = work_3x3 + invSigma_Bj;
				for (int h = 0; h < 3; h++)
				{
					for (int k = 0; k < 3; k++)
					{
						A.at<float>(3 + h, 3 + k) = work_3x3.at<float>(h, k);
					}
				}

				work_3x1 = invSigma_Fi * mu_Fi;
				for (int h = 0; h < 3; h++)
				{
					b.at<float>(h, 0) = work_3x1.at<float>(h, 0) + c_color[h] * alpha * inv_sigmac_square;
				}

				work_3x1 = invSigma_Bj * mu_Bj;
				for (int h = 0; h < 3; h++)
				{
					b.at<float>(3 + h, 0) = work_3x1.at<float>(h, 0) + c_color[h] * (1 - alpha) * inv_sigmac_square;
				}

				cv::solve(A, b, X);	//AX=b

									// foreground color
				fg_color[0] = static_cast<float>(std::max(0.0, std::min(1.0, (double)X.at<float>(0, 0))));
				fg_color[1] = static_cast<float>(std::max(0.0, std::min(1.0, (double)X.at<float>(1, 0))));
				fg_color[2] = static_cast<float>(std::max(0.0, std::min(1.0, (double)X.at<float>(2, 0))));

				// background color
				bg_color[0] = static_cast<float>(std::max(0.0, std::min(1.0, (double)X.at<float>(3, 0))));
				bg_color[1] = static_cast<float>(std::max(0.0, std::min(1.0, (double)X.at<float>(4, 0))));
				bg_color[2] = static_cast<float>(std::max(0.0, std::min(1.0, (double)X.at<float>(5, 0))));

				//****************************************************************************//
				// Solve for alpha,the second sud-problem
				for (int h = 0; h < 3; h++)
				{
					work_c.at<float>(h, 0) = c_color[h];
					work_f.at<float>(h, 0) = fg_color[h];
					work_b.at<float>(h, 0) = bg_color[h];
				}

				work_3x1 = work_c - work_b;
				work_1x1_n = work_3x1.t() * (work_f - work_b);		//分子

				work_3x1 = work_f - work_b;
				work_1x1_d = work_3x1.t() * work_3x1;	//分母

				alpha1 = work_1x1_n.at<float>(0, 0);
				alpha2 = work_1x1_d.at<float>(0, 0);

				alpha = static_cast<float>(std::max(0.0, std::min(1.0, static_cast<double>(alpha1) / static_cast<double>(alpha2))));

				//				double oriT = (double)cvGetTickCount();

				like = ComputeLikelihood(x, y, mu_Fi, invSigma_Fi, mu_Bj, invSigma_Bj, c_color, fg_color, bg_color, alpha);

				// 				oriT = (double)cvGetTickCount()-oriT;
				// 				float oriCost = oriT/(cvGetTickFrequency()*1000);
				// 				oriTime+=oriCost;
				// 
				// 				double sseT = (double)cvGetTickCount();

				//				like = ComputeLikelihood_SSE_for_original(x, y, mu_Fi, invSigma_Fi, mu_Bj, invSigma_Bj, c_color, fg_color, bg_color, alpha);

				// 				sseT = (double)cvGetTickCount()-sseT;
				// 				float sseCost = sseT/(cvGetTickFrequency()*1000);
				// 				sseTime += sseCost;
				// 
				// 				flikeout<<like<<"	,	"<<mlike<<endl;

				if (iter >= MAX_ITERATION || std::fabs(like - lastLike) <= MIN_LIKE)	//achieve max iteration or convergence
				{
					break;
				}

				lastLike = like;
				iter = iter + 1;
			}

			//Find the pair with the maximum likelihood
			if (like > maxLike)
			{
				maxLike = like;
				max_fg_color = fg_color;
				max_bg_color = bg_color;
				max_alpha = alpha;
			}
		}
	}

	(*F) = max_fg_color;
	(*B) = max_bg_color;
	(*a) = max_alpha;
	//	ftimeout<<oriTime<<"	,	"<<sseTime<<endl;
}

//用SSE指令集加速
void BayesianMatting::Optimize_SSE(
	const int x, const int y,
	const std::vector<Cluster> &fg_clusters, const std::vector<Cluster> &bg_clusters, const float alpha_init,
	cv::Vec3f *F, cv::Vec3f *B, float *a)
{
	float like, lastLike, maxLike = -FLT_MAX;
	int iter;		//迭代次数

	const __m128 constOne = _mm_set_ps1(1.0f);	//1
	const __m128 constZero = _mm_setzero_ps();	//0

	__m128 alpha_Numer;		//求解alpha时分子，即dot( (C-B) , (F-B) )
	__m128 alpha_Deno;		//求解alpha时分母，即dot( (F-B) , (F-B) )
	__m128 mAlpha;		//求解时每次得到的中间alpha
	__m128 max_alpha = _mm_set_ps1(alpha_init);		//最大似然度对应alpha, (alpha_init, alpha_init, alpha_init, alpha_init)
	__m128 mC = _mm_set_ps(1.0f, colorImg.at<cv::Vec3f>(y, x)[2], colorImg.at<cv::Vec3f>(y, x)[1], colorImg.at<cv::Vec3f>(y, x)[0]);
	__m128 mF, mB;		//求解时每次得到的中间F,B
	__m128 mMaxF, mMaxB;		//最大似然度对应的F,B
	__m128 mMeanF;		//F均值
	__m128 mMeanB;		//B均值

	cv::Mat invSigma_Fi;	//F协方差矩阵的逆	
	cv::Mat invSigma_Bj;
	__m128 inv_sigmac_square = _mm_set_ps1(1.0f / (sigma_c * sigma_c));

	/*************************************************************************************************/
	/*														AX=b
	[ coefF		coefS ]	coefF = invSigma_Fi + I * alpha * alpha * inv_sigmac_square
	A=[						  ]	coefS = I * alpha * (1-alpha) * inv_sigmac_square
	[ coefS	coefB ]	coefB = invSigma_Bj + I * (1-alpha) * (1-alpha) * inv_sigmac_square

	[ constItemF ] constItemF = invSigma_Fi*meanF + C* alpha * inv_sigmac_square
	b=[                    ]
	[ constItemB ] constItemB = invSigma_Bj*meanB + C*(1-alpha) * inv_sigmac_square

	[ F ]
	X=[    ]
	[ B ]
	*/
	/*************************************************************************************************/
	cv::Mat A = cv::Mat(6, 6, CV_32FC1, cv::Scalar(0));
	cv::Mat X = cv::Mat(6, 1, CV_32FC1, cv::Scalar(0));
	cv::Mat b = cv::Mat(6, 1, CV_32FC1, cv::Scalar(0));

	__m128 coefF_0 = _mm_setzero_ps();	//coefF的第一列
	__m128 coefF_1 = _mm_setzero_ps();	//coefF的第二列
	__m128 coefF_2 = _mm_setzero_ps();	//coefF的第三列

	__m128 coefB_0 = _mm_setzero_ps();
	__m128 coefB_1 = _mm_setzero_ps();
	__m128 coefB_2 = _mm_setzero_ps();

	__m128 coefS_0 = _mm_setzero_ps();
	__m128 coefS_1 = _mm_setzero_ps();
	__m128 coefS_2 = _mm_setzero_ps();

	__m128 Identity_0 = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);		//3*3的单位矩阵第一列，最后一行补0
	__m128 Identity_1 = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
	__m128 Identity_2 = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);

	__m128 constItemF, constItemB;	//b

									//calculate each (F,B) pairs
	for (size_t i = 0; i < fg_clusters.size(); i++)
	{
		mMeanF = _mm_set_ps(1.0f, fg_clusters.at(i).q.at<float>(0, 2), fg_clusters.at(i).q.at<float>(0, 1), fg_clusters.at(i).q.at<float>(0, 0));	//Data3~0: 1.0RGB
		invSigma_Fi = fg_clusters.at(i).covMatrix.inv();
		for (size_t j = 0; j<bg_clusters.size(); j++)
		{
			mMeanB = _mm_set_ps(1.0f, bg_clusters.at(j).q.at<float>(0, 2), bg_clusters.at(j).q.at<float>(0, 1), bg_clusters.at(j).q.at<float>(0, 0));	//Data3~0: 1.0RGB
			invSigma_Bj = bg_clusters.at(j).covMatrix.inv();

			mAlpha = _mm_set_ps1(alpha_init);	//(alpha_init, alpha_init, alpha_init, alpha_init)

			lastLike = -FLT_MAX;
			iter = 1;
			while (1)
			{
				//****************************************************************************//
				// Solve for foreground and background, the first sub-problem

				//Compute A Begin
				//coefS =  I * alpha * (1-alpha) * inv_sigmac_square 
				coefS_0 = _mm_mul_ps(Identity_0, _mm_mul_ps(_mm_mul_ps(mAlpha, _mm_sub_ps(constOne, mAlpha)), inv_sigmac_square));	//I*alpha*(1-alpha)*inv_sigmaC_square
				coefS_1 = _mm_mul_ps(Identity_1, _mm_mul_ps(_mm_mul_ps(mAlpha, _mm_sub_ps(constOne, mAlpha)), inv_sigmac_square));
				coefS_2 = _mm_mul_ps(Identity_2, _mm_mul_ps(_mm_mul_ps(mAlpha, _mm_sub_ps(constOne, mAlpha)), inv_sigmac_square));

				for (int row = 0; row<3; row++)
				{
					A.at<float>(row, 3) = coefS_0.m128_f32[row];	//a03,a13,a23
					A.at<float>(row, 4) = coefS_1.m128_f32[row];	//a04,a14,a24
					A.at<float>(row, 5) = coefS_2.m128_f32[row];	//a05,a15,a25

					A.at<float>(row + 3, 0) = coefS_0.m128_f32[row];	//a30,a40,a50					
					A.at<float>(row + 3, 1) = coefS_1.m128_f32[row];	//a31,a41,a51					
					A.at<float>(row + 3, 2) = coefS_2.m128_f32[row];	//a32,a42,a52
				}

				//coefF = invSigma_Fi + I * alpha * alpha * inv_sigmac_square
				__m128 invSigma_Fi0 = _mm_set_ps(0.0f, invSigma_Fi.at<float>(2, 0), invSigma_Fi.at<float>(1, 0), invSigma_Fi.at<float>(0, 0));		//得到矩阵的一列
				__m128 invSigma_Fi1 = _mm_set_ps(0.0f, invSigma_Fi.at<float>(2, 1), invSigma_Fi.at<float>(1, 1), invSigma_Fi.at<float>(0, 1));
				__m128 invSigma_Fi2 = _mm_set_ps(0.0f, invSigma_Fi.at<float>(2, 2), invSigma_Fi.at<float>(1, 2), invSigma_Fi.at<float>(0, 2));

				coefF_0 = _mm_add_ps(invSigma_Fi0, _mm_mul_ps(Identity_0, _mm_mul_ps(_mm_mul_ps(mAlpha, mAlpha), inv_sigmac_square)));	//invSigma_Fi + I*alpha*alpha*inv_sigmaC_square
				coefF_1 = _mm_add_ps(invSigma_Fi1, _mm_mul_ps(Identity_1, _mm_mul_ps(_mm_mul_ps(mAlpha, mAlpha), inv_sigmac_square)));
				coefF_2 = _mm_add_ps(invSigma_Fi2, _mm_mul_ps(Identity_2, _mm_mul_ps(_mm_mul_ps(mAlpha, mAlpha), inv_sigmac_square)));

				for (int row = 0; row <3; row++)
				{
					A.at<float>(row, 0) = coefF_0.m128_f32[row];	//a00,a10,a20 第一列
					A.at<float>(row, 1) = coefF_1.m128_f32[row];	//a01,a11,a21
					A.at<float>(row, 2) = coefF_2.m128_f32[row];	//a02,a12,a22
				}

				//coefB = invSigma_Bj + I * (1-alpha) * (1-alpha) * inv_sigmac_square
				__m128 invSigma_Bj0 = _mm_set_ps(0.0f, invSigma_Bj.at<float>(2, 0), invSigma_Bj.at<float>(1, 0), invSigma_Bj.at<float>(0, 0));		//得到矩阵的一列
				__m128 invSigma_Bj1 = _mm_set_ps(0.0f, invSigma_Bj.at<float>(2, 1), invSigma_Bj.at<float>(1, 1), invSigma_Bj.at<float>(0, 1));
				__m128 invSigma_Bj2 = _mm_set_ps(0.0f, invSigma_Bj.at<float>(2, 2), invSigma_Bj.at<float>(1, 2), invSigma_Bj.at<float>(0, 2));

				coefB_0 = _mm_add_ps(invSigma_Bj0, _mm_mul_ps(Identity_0, _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(constOne, mAlpha), _mm_sub_ps(constOne, mAlpha)), inv_sigmac_square)));	// invSigma_Bj + I * (1-alpha) * (1-alpha) * inv_sigmac_square
				coefB_1 = _mm_add_ps(invSigma_Bj1, _mm_mul_ps(Identity_1, _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(constOne, mAlpha), _mm_sub_ps(constOne, mAlpha)), inv_sigmac_square)));
				coefB_2 = _mm_add_ps(invSigma_Bj2, _mm_mul_ps(Identity_2, _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(constOne, mAlpha), _mm_sub_ps(constOne, mAlpha)), inv_sigmac_square)));

				for (int row = 0; row<3; row++)
				{
					A.at<float>(row + 3, 3) = coefB_0.m128_f32[row];	//a33,a43,a53
					A.at<float>(row + 3, 4) = coefB_1.m128_f32[row];	//a34,a44,a54
					A.at<float>(row + 3, 5) = coefB_2.m128_f32[row];	//a35,a45,a55
				}
				//Compute A finish

				//Compute b begin
				//constItemF = invSigma_Fi*meanF + C* alpha * inv_sigmac_square
				invSigma_Fi0 = _mm_set_ps(0.0f, invSigma_Fi.at<float>(0, 2), invSigma_Fi.at<float>(0, 1), invSigma_Fi.at<float>(0, 0));	//得到矩阵的一行
				invSigma_Fi1 = _mm_set_ps(0.0f, invSigma_Fi.at<float>(1, 2), invSigma_Fi.at<float>(1, 1), invSigma_Fi.at<float>(1, 0));
				invSigma_Fi2 = _mm_set_ps(0.0f, invSigma_Fi.at<float>(2, 2), invSigma_Fi.at<float>(2, 1), invSigma_Fi.at<float>(2, 0));

				__m128 dotproduct0 = _mm_mul_ps(invSigma_Fi0, mMeanF);
				dotproduct0 = _mm_add_ps(dotproduct0, _mm_movehl_ps(dotproduct0, dotproduct0));
				dotproduct0 = _mm_add_ss(dotproduct0, _mm_shuffle_ps(dotproduct0, dotproduct0, 1));	//data0即为invSigma_Fi*meanF的第一个元素

				__m128 dotproduct1 = _mm_mul_ps(invSigma_Fi1, mMeanF);
				dotproduct1 = _mm_add_ps(dotproduct1, _mm_movehl_ps(dotproduct1, dotproduct1));
				dotproduct1 = _mm_add_ss(dotproduct1, _mm_shuffle_ps(dotproduct1, dotproduct1, 1));	//data0即为invSigma_Fi*meanF的第二个元素

				__m128 dotproduct2 = _mm_mul_ps(invSigma_Fi2, mMeanF);
				dotproduct2 = _mm_add_ps(dotproduct2, _mm_movehl_ps(dotproduct2, dotproduct2));
				dotproduct2 = _mm_add_ss(dotproduct2, _mm_shuffle_ps(dotproduct2, dotproduct2, 1));	//data0即为invSigma_Fi*meanF的第三个元素

				constItemF = _mm_set_ps(0.0f, dotproduct2.m128_f32[0], dotproduct1.m128_f32[0], dotproduct0.m128_f32[0]);		//invSigma_Fi*meanF
				constItemF = _mm_add_ps(constItemF, _mm_mul_ps(_mm_mul_ps(mC, mAlpha), inv_sigmac_square));		//invSigma_Fi*meanF + C* alpha * inv_sigmac_square

																												//constItemB = invSigma_Bj*meanB + C*(1-alpha) * inv_sigmac_square
				invSigma_Bj0 = _mm_set_ps(0.0f, invSigma_Bj.at<float>(0, 2), invSigma_Bj.at<float>(0, 1), invSigma_Bj.at<float>(0, 0));	//得到矩阵的一行
				invSigma_Bj1 = _mm_set_ps(0.0f, invSigma_Bj.at<float>(1, 2), invSigma_Bj.at<float>(1, 1), invSigma_Bj.at<float>(1, 0));
				invSigma_Bj2 = _mm_set_ps(0.0f, invSigma_Bj.at<float>(2, 2), invSigma_Bj.at<float>(2, 1), invSigma_Bj.at<float>(2, 0));

				dotproduct0 = _mm_mul_ps(invSigma_Bj0, mMeanB);
				dotproduct0 = _mm_add_ps(dotproduct0, _mm_movehl_ps(dotproduct0, dotproduct0));
				dotproduct0 = _mm_add_ss(dotproduct0, _mm_shuffle_ps(dotproduct0, dotproduct0, 1));	//data0即为invSigma_Bj*meanB的第一个元素

				dotproduct1 = _mm_mul_ps(invSigma_Bj1, mMeanB);
				dotproduct1 = _mm_add_ps(dotproduct1, _mm_movehl_ps(dotproduct1, dotproduct1));
				dotproduct1 = _mm_add_ss(dotproduct1, _mm_shuffle_ps(dotproduct1, dotproduct1, 1));	//data0即为invSigma_Bj*meanB的第二个元素

				dotproduct2 = _mm_mul_ps(invSigma_Bj2, mMeanB);
				dotproduct2 = _mm_add_ps(dotproduct2, _mm_movehl_ps(dotproduct2, dotproduct2));
				dotproduct2 = _mm_add_ss(dotproduct2, _mm_shuffle_ps(dotproduct2, dotproduct2, 1));	//data0即为invSigma_Bj*meanB的第三个元素

				constItemB = _mm_set_ps(0.0f, dotproduct2.m128_f32[0], dotproduct1.m128_f32[0], dotproduct0.m128_f32[0]);		//invSigma_Bj*meanB
				constItemB = _mm_add_ps(constItemB, _mm_mul_ps(_mm_mul_ps(mC, _mm_sub_ps(constOne, mAlpha)), inv_sigmac_square));	//invSigma_Bj*meanB + C*(1-alpha) * inv_sigmac_square

				for (int row = 0; row<3; row++)
				{
					b.at<float>(row, 0) = constItemF.m128_f32[row];
					b.at<float>(row + 3, 0) = constItemB.m128_f32[row];
				}
				//Compute b finish

				cv::solve(A, b, X);	//AX=b

				mF = _mm_set_ps(1.0f, X.at<float>(2, 0), X.at<float>(1, 0), X.at<float>(0, 0));		// foreground color
				mF = _mm_max_ps(_mm_min_ps(mF, constOne), constZero);		//逐项取最小值最大值
				mB = _mm_set_ps(1.0f, X.at<float>(5, 0), X.at<float>(4, 0), X.at<float>(3, 0));		// background color
				mB = _mm_max_ps(_mm_min_ps(mB, constOne), constZero);

				//****************************************************************************//
				// Solve for alpha,the second sud-problem
				//mAlpha = dot((C - B) , (F - B)) / dot((F - B) , (F - B))
				__m128 FMinusB = _mm_sub_ps(mF, mB);
				__m128 CMinusB = _mm_sub_ps(mC, mB);
				__m128 tmp;

				tmp = _mm_mul_ps(CMinusB, FMinusB);
				alpha_Numer = _mm_add_ss(tmp, _mm_movehl_ps(tmp, tmp));
				alpha_Numer = _mm_add_ss(alpha_Numer, _mm_shuffle_ps(alpha_Numer, alpha_Numer, 1));

				tmp = _mm_mul_ps(FMinusB, FMinusB);
				alpha_Deno = _mm_add_ss(tmp, _mm_movehl_ps(tmp, tmp));
				alpha_Deno = _mm_add_ss(alpha_Deno, _mm_shuffle_ps(alpha_Deno, alpha_Deno, 1));

				__m128 newAlpha = _mm_div_ss(alpha_Numer, alpha_Deno);
				mAlpha = _mm_shuffle_ps(newAlpha, newAlpha, _MM_SHUFFLE(0, 0, 0, 0));
				mAlpha = _mm_max_ps(_mm_min_ps(mAlpha, constOne), constZero);
				//compute new alpha finish

				like = ComputeLikelihood_SSE(x, y, mMeanF, invSigma_Fi, mMeanB, invSigma_Bj, mC, mF, mB, mAlpha);	//求出似然度

				if (iter >= MAX_ITERATION || std::fabs(like - lastLike) <= MIN_LIKE)	//achieve max iteration or convergence
				{
					break;
				}

				lastLike = like;
				iter = iter + 1;
			}

			//Find the pair with the maximum likelihood
			if (like > maxLike)
			{
				maxLike = like;
				mMaxF = mF;
				mMaxB = mB;
				max_alpha = mAlpha;
			}
		}
	}

	(*F)[0] = mMaxF.m128_f32[0];
	(*F)[1] = mMaxF.m128_f32[1];
	(*F)[2] = mMaxF.m128_f32[2];

	(*B)[0] = mMaxB.m128_f32[0];
	(*B)[1] = mMaxB.m128_f32[1];
	(*B)[2] = mMaxB.m128_f32[2];

	*a = max_alpha.m128_f32[0];
}

//未加速的方法，现在没用
float BayesianMatting::ComputeLikelihood(
	const int x, const int y,
	const cv::Mat &mu_Fi, const cv::Mat &invSigma_Fi,
	const cv::Mat &mu_Bj, const cv::Mat &invSigma_Bj,
	const cv::Vec3f &c_color, const cv::Vec3f &fg_color, const cv::Vec3f &bg_color, const float alpha)
{
	float L_C, L_F, L_B;
	float inv_sigmac_square = 1.0f / (sigma_c * sigma_c);

	cv::Mat work3x1 = cv::Mat(3, 1, CV_32FC1);
	cv::Mat work1x1 = cv::Mat(1, 1, CV_32FC1);

	cv::Mat F = cv::Mat(3, 1, CV_32FC1);
	cv::Mat B = cv::Mat(3, 1, CV_32FC1);
	cv::Mat C = cv::Mat(3, 1, CV_32FC1);

	F.at<float>(0, 0) = fg_color[0];
	F.at<float>(1, 0) = fg_color[1];
	F.at<float>(2, 0) = fg_color[2];

	B.at<float>(0, 0) = bg_color[0];
	B.at<float>(1, 0) = bg_color[1];
	B.at<float>(2, 0) = bg_color[2];

	C.at<float>(0, 0) = c_color[0];
	C.at<float>(1, 0) = c_color[1];
	C.at<float>(2, 0) = c_color[2];

	work3x1 = F - mu_Fi;
	work1x1 = work3x1.t() * invSigma_Fi * work3x1;

	L_F = -1.0f * work1x1.at<float>(0, 0) / 2.0f;

	work3x1 = B - mu_Bj;
	work1x1 = work3x1.t() * invSigma_Bj * work3x1;

	L_B = -1.0f * work1x1.at<float>(0, 0) / 2.0f;

	work3x1 = C - (F * alpha) - (B * (1.0f - alpha));
	work1x1 = work3x1.t() * work3x1;

	L_C = -1.0f * work1x1.at<float>(0, 0) * inv_sigmac_square;

	return L_F + L_B + L_C;
}

//用SSE指令集加速的计算似然度，供原始未加速Optimize函数使用，现在没用
float BayesianMatting::ComputeLikelihood_SSE_for_original(
	const int x, const int y,
	const cv::Mat &mu_Fi, const cv::Mat &invSigma_Fi,
	const cv::Mat &mu_Bj, const cv::Mat &invSigma_Bj,
	const cv::Vec3f &c_color, const cv::Vec3f &fg_color, const cv::Vec3f &bg_color, const float alpha)
{
	__m128 L_C, L_F, L_B;
	__m128 inv_sigmac_square = _mm_set_ps1(1.0f / (sigma_c * sigma_c));

	__m128 mF = _mm_set_ps(1.0f, fg_color[2], fg_color[1], fg_color[0]);		//Data3~0: 1.0RGB
	__m128 mB = _mm_set_ps(1.0f, bg_color[2], bg_color[1], bg_color[0]);
	__m128 mC = _mm_set_ps(1.0f, c_color[2], c_color[1], c_color[0]);
	__m128 mMeanF = _mm_set_ps(1.0f, mu_Fi.at<float>(2, 0), mu_Fi.at<float>(1, 0), mu_Fi.at<float>(0, 0));	//Data3~0: 1.0RGB
	__m128 mMeanB = _mm_set_ps(1.0f, mu_Bj.at<float>(2, 0), mu_Bj.at<float>(1, 0), mu_Bj.at<float>(0, 0));	//Data3~0: 1.0RGB
	__m128 mAlpha = _mm_set_ps1(alpha);

	const __m128 constOne = _mm_set_ps1(1.0f);	//1
	const __m128 constZero = _mm_setzero_ps();	//0
	const __m128 constCoef = _mm_set_ps1(-0.5f);		//-1/2

	__m128 mOneMinusAlpha = _mm_sub_ps(constOne, mAlpha);	//1-alpha

	__m128 deltaF = _mm_sub_ps(mF, mMeanF);		//F-meanF
	__m128 deltaB = _mm_sub_ps(mB, mMeanB);		//B-meanB
	__m128 deltaC = _mm_sub_ps(_mm_sub_ps(mC, _mm_mul_ps(mAlpha, mF)), _mm_mul_ps(mOneMinusAlpha, mB));	//C-alpha*F-(1-alpha)*B
	__m128 tmp, dotproduct;	//求dot()用的中间变量

							//L_C= - dot(deltaC, deltaC) * invSqSigmaC
	tmp = _mm_mul_ps(deltaC, deltaC);	//(a^2 , b^2 , c^2 , d^2)
	dotproduct = _mm_add_ps(tmp, _mm_movehl_ps(tmp, tmp));	//(a^2 , b^2 , c^2 , d^2)+(a^2 , b^2 , a^2 , b^2) = (2*a^2 , 2*b^2 , c^2+a^2 , d^2+b^2)
	dotproduct = _mm_add_ss(dotproduct, _mm_shuffle_ps(dotproduct, dotproduct, 1));		// (2*a^2 , 2*b^2 , c^2+a^2 , d^2+b^2)+( c^2+a^2 ,  c^2+a^2 ,  c^2+a^2 ,  c^2+a^2) = (... , ... , ... ,  c^2+a^2+d^2+b^2)  data0即(a,b,c,d) .*(a,b,c,d)
	L_C = _mm_sub_ps(constZero, _mm_mul_ps(dotproduct, inv_sigmac_square));	//LC = L_C.m128_f32[0]

																			//L_F = -0.5*deltaF*invSigma_Fi*deltaF （矩阵相乘）
	__m128 invSigma_Fi0 = _mm_set_ps(0.0f, invSigma_Fi.at<float>(2, 0), invSigma_Fi.at<float>(1, 0), invSigma_Fi.at<float>(0, 0));		//得到矩阵的一列
	__m128 invSigma_Fi1 = _mm_set_ps(0.0f, invSigma_Fi.at<float>(2, 1), invSigma_Fi.at<float>(1, 1), invSigma_Fi.at<float>(0, 1));
	__m128 invSigma_Fi2 = _mm_set_ps(0.0f, invSigma_Fi.at<float>(2, 2), invSigma_Fi.at<float>(1, 2), invSigma_Fi.at<float>(0, 2));

	__m128 tmp0 = _mm_mul_ps(deltaF, invSigma_Fi0);	//(0*0 , r1*r2 , g1*g2 , b1*b2)
	tmp0 = _mm_add_ps(tmp0, _mm_movehl_ps(tmp0, tmp0));		//(0*0 , r1*r2 , g1*g2 , b1*b2)+(0*0 , r1*r2 , 0*0 , r1*r2) = (0 , 2*r1*r2, g1*g2+0 , b1*b2+r1*r2)
	tmp0 = _mm_add_ss(tmp0, _mm_shuffle_ps(tmp0, tmp0, 1));	//(0 , 2*r1*r2, g1*g2+0 , b1*b2+r1*r2)+(g1*g2+0 , g1*g2+0 , g1*g2+0 , g1*g2+0) = (... , ... , ... , b1*b2+r1*r2+g1*g2+0)		data0即为（deltaF）^T*invsigma_Fi结果的第一项tmp(0,0)

	__m128 tmp1 = _mm_mul_ps(deltaF, invSigma_Fi1);
	tmp1 = _mm_add_ps(tmp1, _mm_movehl_ps(tmp1, tmp1));
	tmp1 = _mm_add_ss(tmp1, _mm_shuffle_ps(tmp1, tmp1, 1));	//data0即为（deltaF）^T*invsigma_Fi结果的第二项tmp(0,1)

	__m128 tmp2 = _mm_mul_ps(deltaF, invSigma_Fi2);
	tmp2 = _mm_add_ps(tmp2, _mm_movehl_ps(tmp2, tmp2));
	tmp2 = _mm_add_ss(tmp2, _mm_shuffle_ps(tmp2, tmp2, 1));	//data0即为（deltaF）^T*invsigma_Fi结果的第三项tmp(0,2)

	__m128 mul_deltaF_invSigma_Fi = _mm_set_ps(0.0f, tmp2.m128_f32[0], tmp1.m128_f32[0], tmp0.m128_f32[0]);	//（deltaF）^T*invsigma_Fi，得到一个1*3的矩阵

	dotproduct = _mm_mul_ps(mul_deltaF_invSigma_Fi, deltaF);
	dotproduct = _mm_add_ps(dotproduct, _mm_movehl_ps(dotproduct, dotproduct));
	dotproduct = _mm_add_ss(dotproduct, _mm_shuffle_ps(dotproduct, dotproduct, 1));		//data0即为deltaF*invSigma_Fi*deltaF 
	L_F = _mm_mul_ps(constCoef, dotproduct);	//LF = L_F.m128_f32[0];

												//L_B = -0.5*deltaB*invSigma_Bj*deltaB （矩阵相乘）
	__m128 invSigma_Bj0 = _mm_set_ps(0.0f, invSigma_Bj.at<float>(2, 0), invSigma_Bj.at<float>(1, 0), invSigma_Bj.at<float>(0, 0));		//得到矩阵的一列
	__m128 invSigma_Bj1 = _mm_set_ps(0.0f, invSigma_Bj.at<float>(2, 1), invSigma_Bj.at<float>(1, 1), invSigma_Bj.at<float>(0, 1));
	__m128 invSigma_Bj2 = _mm_set_ps(0.0f, invSigma_Bj.at<float>(2, 2), invSigma_Bj.at<float>(1, 2), invSigma_Bj.at<float>(0, 2));

	tmp0 = _mm_mul_ps(deltaB, invSigma_Bj0);
	tmp0 = _mm_add_ps(tmp0, _mm_movehl_ps(tmp0, tmp0));
	tmp0 = _mm_add_ss(tmp0, _mm_shuffle_ps(tmp0, tmp0, 1));

	tmp1 = _mm_mul_ps(deltaB, invSigma_Bj1);
	tmp1 = _mm_add_ps(tmp1, _mm_movehl_ps(tmp1, tmp1));
	tmp1 = _mm_add_ss(tmp1, _mm_shuffle_ps(tmp1, tmp1, 1));

	tmp2 = _mm_mul_ps(deltaB, invSigma_Bj2);
	tmp2 = _mm_add_ps(tmp2, _mm_movehl_ps(tmp2, tmp2));
	tmp2 = _mm_add_ss(tmp2, _mm_shuffle_ps(tmp2, tmp2, 1));

	__m128 mul_deltaB_invSigma_Bj = _mm_set_ps(0.0f, tmp2.m128_f32[0], tmp1.m128_f32[0], tmp0.m128_f32[0]);

	dotproduct = _mm_mul_ps(mul_deltaB_invSigma_Bj, deltaB);
	dotproduct = _mm_add_ps(dotproduct, _mm_movehl_ps(dotproduct, dotproduct));
	dotproduct = _mm_add_ss(dotproduct, _mm_shuffle_ps(dotproduct, dotproduct, 1));		//data0即为deltaB*invSigma_Bj*deltaB 
	L_B = _mm_mul_ps(constCoef, dotproduct);	//LB = L_B.m128_f32[0];

	return _mm_add_ss(L_C, _mm_add_ss(L_F, L_B)).m128_f32[0];	//L_C+L_F+L_B , likelihood = result.m128_f32[0]
}

//用SSE指令集加速的计算似然度
float BayesianMatting::ComputeLikelihood_SSE(
	const int x, const int y,
	const __m128 &mMeanF, const cv::Mat &invSigma_Fi,
	const __m128 &mMeanB, const cv::Mat &invSigma_Bj,
	const __m128 &mC, const __m128 &mF, const __m128 &mB, const __m128 &mAlpha)
{
	__m128 L_C, L_F, L_B;
	__m128 inv_sigmac_square = _mm_set_ps1(1.0f / (sigma_c * sigma_c));

	//__m128 mF = _mm_set_ps(1.0f,fg_color[2], fg_color[1],fg_color[0]);		//Data3~0: 1.0RGB
	//__m128 mB = _mm_set_ps(1.0f,bg_color[2],bg_color[1],bg_color[0]);
	//__m128 mC = _mm_set_ps(1.0f,c_color[2],c_color[1],c_color[0]);
	//__m128 mMeanF = _mm_set_ps(1.0f,mu_Fi.at<float>(2,0),mu_Fi.at<float>(1,0),mu_Fi.at<float>(0,0));	//Data3~0: 1.0RGB
	//__m128 mMeanB = _mm_set_ps(1.0f,mu_Bj.at<float>(2,0),mu_Bj.at<float>(1,0),mu_Bj.at<float>(0,0));	//Data3~0: 1.0RGB
	//__m128 mAlpha = _mm_set_ps1(alpha);	

	const __m128 constOne = _mm_set_ps1(1.0f);	//1
	const __m128 constZero = _mm_setzero_ps();	//0
	const __m128 constCoef = _mm_set_ps1(-0.5f);		//-1/2

	__m128 mOneMinusAlpha = _mm_sub_ps(constOne, mAlpha);	//1-alpha

	__m128 deltaF = _mm_sub_ps(mF, mMeanF);		//F-meanF
	__m128 deltaB = _mm_sub_ps(mB, mMeanB);		//B-meanB
	__m128 deltaC = _mm_sub_ps(_mm_sub_ps(mC, _mm_mul_ps(mAlpha, mF)), _mm_mul_ps(mOneMinusAlpha, mB));	//C-alpha*F-(1-alpha)*B
	__m128 tmp, dotproduct;	//求dot()用的中间变量

							//L_C= - dot(deltaC, deltaC) * invSqSigmaC
	tmp = _mm_mul_ps(deltaC, deltaC);	//(a^2 , b^2 , c^2 , d^2)
	dotproduct = _mm_add_ps(tmp, _mm_movehl_ps(tmp, tmp));	//(a^2 , b^2 , c^2 , d^2)+(a^2 , b^2 , a^2 , b^2) = (2*a^2 , 2*b^2 , c^2+a^2 , d^2+b^2)
	dotproduct = _mm_add_ss(dotproduct, _mm_shuffle_ps(dotproduct, dotproduct, 1));		// (2*a^2 , 2*b^2 , c^2+a^2 , d^2+b^2)+( c^2+a^2 ,  c^2+a^2 ,  c^2+a^2 ,  c^2+a^2) = (... , ... , ... ,  c^2+a^2+d^2+b^2)  data0即(a,b,c,d) .*(a,b,c,d)
	L_C = _mm_sub_ps(constZero, _mm_mul_ps(dotproduct, inv_sigmac_square));	//LC = L_C.m128_f32[0]

																			//L_F = -0.5*deltaF*invSigma_Fi*deltaF （矩阵相乘）
	__m128 invSigma_Fi0 = _mm_set_ps(0.0f, invSigma_Fi.at<float>(2, 0), invSigma_Fi.at<float>(1, 0), invSigma_Fi.at<float>(0, 0));		//得到矩阵的一列
	__m128 invSigma_Fi1 = _mm_set_ps(0.0f, invSigma_Fi.at<float>(2, 1), invSigma_Fi.at<float>(1, 1), invSigma_Fi.at<float>(0, 1));
	__m128 invSigma_Fi2 = _mm_set_ps(0.0f, invSigma_Fi.at<float>(2, 2), invSigma_Fi.at<float>(1, 2), invSigma_Fi.at<float>(0, 2));

	__m128 tmp0 = _mm_mul_ps(deltaF, invSigma_Fi0);	//(0*0 , r1*r2 , g1*g2 , b1*b2)
	tmp0 = _mm_add_ps(tmp0, _mm_movehl_ps(tmp0, tmp0));		//(0*0 , r1*r2 , g1*g2 , b1*b2)+(0*0 , r1*r2 , 0*0 , r1*r2) = (0 , 2*r1*r2, g1*g2+0 , b1*b2+r1*r2)
	tmp0 = _mm_add_ss(tmp0, _mm_shuffle_ps(tmp0, tmp0, 1));	//(0 , 2*r1*r2, g1*g2+0 , b1*b2+r1*r2)+(g1*g2+0 , g1*g2+0 , g1*g2+0 , g1*g2+0) = (... , ... , ... , b1*b2+r1*r2+g1*g2+0)		data0即为（deltaF）^T*invsigma_Fi结果的第一项tmp(0,0)

	__m128 tmp1 = _mm_mul_ps(deltaF, invSigma_Fi1);
	tmp1 = _mm_add_ps(tmp1, _mm_movehl_ps(tmp1, tmp1));
	tmp1 = _mm_add_ss(tmp1, _mm_shuffle_ps(tmp1, tmp1, 1));	//data0即为（deltaF）^T*invsigma_Fi结果的第二项tmp(0,1)

	__m128 tmp2 = _mm_mul_ps(deltaF, invSigma_Fi2);
	tmp2 = _mm_add_ps(tmp2, _mm_movehl_ps(tmp2, tmp2));
	tmp2 = _mm_add_ss(tmp2, _mm_shuffle_ps(tmp2, tmp2, 1));	//data0即为（deltaF）^T*invsigma_Fi结果的第三项tmp(0,2)

	__m128 mul_deltaF_invSigma_Fi = _mm_set_ps(0.0f, tmp2.m128_f32[0], tmp1.m128_f32[0], tmp0.m128_f32[0]);	//（deltaF）^T*invsigma_Fi，得到一个1*3的矩阵

	dotproduct = _mm_mul_ps(mul_deltaF_invSigma_Fi, deltaF);
	dotproduct = _mm_add_ps(dotproduct, _mm_movehl_ps(dotproduct, dotproduct));
	dotproduct = _mm_add_ss(dotproduct, _mm_shuffle_ps(dotproduct, dotproduct, 1));		//data0即为deltaF*invSigma_Fi*deltaF 
	L_F = _mm_mul_ps(constCoef, dotproduct);	//LF = L_F.m128_f32[0];

												//L_B = -0.5*deltaB*invSigma_Bj*deltaB （矩阵相乘）
	__m128 invSigma_Bj0 = _mm_set_ps(0.0f, invSigma_Bj.at<float>(2, 0), invSigma_Bj.at<float>(1, 0), invSigma_Bj.at<float>(0, 0));		//得到矩阵的一列
	__m128 invSigma_Bj1 = _mm_set_ps(0.0f, invSigma_Bj.at<float>(2, 1), invSigma_Bj.at<float>(1, 1), invSigma_Bj.at<float>(0, 1));
	__m128 invSigma_Bj2 = _mm_set_ps(0.0f, invSigma_Bj.at<float>(2, 2), invSigma_Bj.at<float>(1, 2), invSigma_Bj.at<float>(0, 2));

	tmp0 = _mm_mul_ps(deltaB, invSigma_Bj0);
	tmp0 = _mm_add_ps(tmp0, _mm_movehl_ps(tmp0, tmp0));
	tmp0 = _mm_add_ss(tmp0, _mm_shuffle_ps(tmp0, tmp0, 1));

	tmp1 = _mm_mul_ps(deltaB, invSigma_Bj1);
	tmp1 = _mm_add_ps(tmp1, _mm_movehl_ps(tmp1, tmp1));
	tmp1 = _mm_add_ss(tmp1, _mm_shuffle_ps(tmp1, tmp1, 1));

	tmp2 = _mm_mul_ps(deltaB, invSigma_Bj2);
	tmp2 = _mm_add_ps(tmp2, _mm_movehl_ps(tmp2, tmp2));
	tmp2 = _mm_add_ss(tmp2, _mm_shuffle_ps(tmp2, tmp2, 1));

	__m128 mul_deltaB_invSigma_Bj = _mm_set_ps(0.0f, tmp2.m128_f32[0], tmp1.m128_f32[0], tmp0.m128_f32[0]);

	dotproduct = _mm_mul_ps(mul_deltaB_invSigma_Bj, deltaB);
	dotproduct = _mm_add_ps(dotproduct, _mm_movehl_ps(dotproduct, dotproduct));
	dotproduct = _mm_add_ss(dotproduct, _mm_shuffle_ps(dotproduct, dotproduct, 1));		//data0即为deltaB*invSigma_Bj*deltaB 
	L_B = _mm_mul_ps(constCoef, dotproduct);	//LB = L_B.m128_f32[0];

	return _mm_add_ss(L_C, _mm_add_ss(L_F, L_B)).m128_f32[0];	//L_C+L_F+L_B , likelihood = result.m128_f32[0]
}

//Find the cluster which covariance matrix has the greatest eigenvalue
float BayesianMatting::max_lambda(const std::vector<Cluster> &nodes, int *idx)
{
	CV_Assert(nodes.size() != 0);

	float max = nodes[0].lambda;
	(*idx) = 0;

	for (size_t k = 1; k < nodes.size(); k++)
	{
		if (nodes[k].lambda > max)
		{
			max = nodes[k].lambda;
			(*idx) = k;
		}
	}

	return max;
}

//Split cluster with large eigenvalue into two clusters with small eigenvalue
void BayesianMatting::Split(std::vector<Cluster> *nodes)
{
	int idx;
	max_lambda((*nodes), &idx);

	Cluster Ci = (*nodes)[idx];
	Cluster Ca;
	Cluster Cb;

	cv::Mat bo = Ci.q * Ci.e;
	double boundary = bo.at<float>(0, 0);

	cv::Mat cur_color = cv::Mat(3, 1, CV_32FC1);

	for (size_t i = 0; i < Ci.sample_set.size(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cur_color.at<float>(j, 0) = Ci.sample_set[i].first[j];
		}

		if (cur_color.dot(Ci.e) <= boundary)
		{
			Ca.sample_set.push_back(Ci.sample_set[i]);
		}
		else
		{
			Cb.sample_set.push_back(Ci.sample_set[i]);
		}
	}

	Ca.calc_SSE();
	Cb.calc_SSE();

	/*Ca.Calc();
	Cb.Calc();*/

	nodes->erase(nodes->begin() + idx);
	nodes->push_back(Ca);
	nodes->push_back(Cb);
}

//Split cluster with large eigenvalue into clusters with small eigenvalue
void BayesianMatting::Cluster_OrchardBouman(
	const int x, const int y,
	const std::vector<std::pair<cv::Vec3f, float> > &sample_set,
	std::vector<Cluster> *Clusters)
{
	Clusters->clear();

	Cluster C1(sample_set);
	C1.calc_SSE();
	//C1.Calc();
	Clusters->push_back(C1);

	int idx;
	while (max_lambda((*Clusters), &idx) > MIN_VAR)
	{
		Split(Clusters);
	}
}

void BayesianMatting::AddCamVar(std::vector<Cluster> *clusters)
{
	float sigma_c_square = sigma_c * sigma_c;
	cv::Mat diag;

	for (size_t k = 0; k < clusters->size(); k++)
	{
		diag = cv::Mat::zeros(3, 3, CV_32FC1);

		///****************************************************************/
		//std::cout<<std::endl;
		//for(int i=0; i<(*clusters)[k].covMatrix.rows; i++) 
		//{
		//	for(int j=0; j<(*clusters)[k].covMatrix.cols; j++) 
		//	{
		//		std::cout<<(*clusters)[k].covMatrix.at<float>(i,j)<<" ";
		//	}std::cout<<std::endl;
		//}
		//system("pause");
		///****************************************************************/

		cv::SVD svd((*clusters)[k].covMatrix);

		diag.at<float>(0, 0) = svd.w.at<float>(0, 0) + sigma_c_square;
		diag.at<float>(1, 1) = svd.w.at<float>(1, 0) + sigma_c_square;
		diag.at<float>(2, 2) = svd.w.at<float>(2, 0) + sigma_c_square;

		(*clusters)[k].covMatrix = svd.u * diag * svd.vt;

		///****************************************************************/
		//std::cout<<std::endl;
		//for(int i=0; i<(*clusters)[k].covMatrix.rows; i++) 
		//{
		//	for(int j=0; j<(*clusters)[k].covMatrix.cols; j++) 
		//	{
		//		std::cout<<(*clusters)[k].covMatrix.at<float>(i,j)<<" ";
		//	}std::cout<<std::endl;
		//}
		//system("pause");
		///****************************************************************/

	}
}

void BayesianMatting::Composite(const cv::Mat &composite, cv::Mat *result)
{
	cv::Mat coImg;
	cv::Vec3f f, b;
	float alpha;

	composite.convertTo(coImg, CV_32F, 1.0f / 255.0f);

	result->create(composite.size(), CV_32FC3);

	for (int y = 0; y < result->rows; y++)
	{
		for (int x = 0; x < result->cols; x++)
		{
			f = fgImg.at<cv::Vec3f>(y, x);
			b = coImg.at<cv::Vec3f>(y, x);
			alpha = alphamap.at<float>(y, x);
			/*result->at<cv::Vec3f>(y, x)[0] = static_cast<float>(std::max(0.0, std::min(1.0, f[0] * alpha + b[0] * (1.0 - alpha))));
			result->at<cv::Vec3f>(y, x)[1] = static_cast<float>(std::max(0.0, std::min(1.0, f[1] * alpha + b[1] * (1.0 - alpha))));
			result->at<cv::Vec3f>(y, x)[2] = static_cast<float>(std::max(0.0, std::min(1.0, f[2] * alpha + b[2] * (1.0 - alpha))));*/

			result->at<cv::Vec3f>(y, x)[0] = static_cast<float>(std::max(0.0, std::min(1.0, b[0] + 1.0*alpha*(f[0] - b[0]))));
			result->at<cv::Vec3f>(y, x)[1] = static_cast<float>(std::max(0.0, std::min(1.0, b[1] + 1.0*alpha*(f[1] - b[1]))));
			result->at<cv::Vec3f>(y, x)[2] = static_cast<float>(std::max(0.0, std::min(1.0, b[2] + 1.0*alpha*(f[2] - b[2]))));

			/*
			cout = αf + (1 - α)b
			cout = αf + b - αb
			cout = b + α(f - b)
			*/
		}
	}
}

/*加速的方法*/
void BayesianMatting::Composite_SSE(const cv::Mat &composite, cv::Mat *result)
{
	__m128 mF, mB;
	__m128 mAlpha;
	__m128 mResult;
	__m128 constZero = _mm_setzero_ps();
	__m128 constOne = _mm_set_ps1(1.0f);

	cv::Mat coImg;
	composite.convertTo(coImg, CV_32F, 1.0f / 255.0f);

	result->create(composite.size(), CV_32FC3);

	for (int y = 0; y < result->rows; y++)
	{
		for (int x = 0; x < result->cols; x++)
		{
			mF = _mm_set_ps(0.0f, fgImg.at<cv::Vec3f>(y, x)[2], fgImg.at<cv::Vec3f>(y, x)[1], fgImg.at<cv::Vec3f>(y, x)[0]);
			mB = _mm_set_ps(0.0f, coImg.at<cv::Vec3f>(y, x)[2], coImg.at<cv::Vec3f>(y, x)[1], coImg.at<cv::Vec3f>(y, x)[0]);
			mAlpha = _mm_set_ps1(alphamap.at<float>(y, x));

			mResult = _mm_add_ps(mB, _mm_mul_ps(mAlpha, _mm_sub_ps(mF, mB)));
			mResult = _mm_max_ps(_mm_min_ps(mResult, constOne), constZero);

			result->at<cv::Vec3f>(y, x)[0] = mResult.m128_f32[0];
			result->at<cv::Vec3f>(y, x)[1] = mResult.m128_f32[1];
			result->at<cv::Vec3f>(y, x)[2] = mResult.m128_f32[2];

			/*
			cout = αf + (1 - α)b
			cout = αf + b - αb
			cout = b + α(f - b)
			*/
		}
	}
}

void BayesianMatting::getResultForeground(cv::Mat &result)
{
	result = this->fgImg;
}