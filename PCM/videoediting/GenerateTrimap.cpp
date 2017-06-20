#include "GenerateTrimap.h"
#include "VideoEdittingParameter.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>

using namespace std;
using std::fstream;
using namespace cv;
GenerateTrimap::GenerateTrimap(void)
{
	thresholdGrad = INIT_THRESHOLD_GRAD;

	computeKernel(INIT_KERNELSIZE);
	thickness = (INIT_KERNELSIZE * 10) / 2;
}

GenerateTrimap::~GenerateTrimap(void)
{
}

Mat GenerateTrimap::getMask()
{
	return binmask;
}

void GenerateTrimap::interactMask(QImage cutMask)	//Grabcut����
{
	if ((cutMask.width() != matInput.cols) || (cutMask.height() != matInput.rows))
		return;
	int w = cutMask.width();
	int h = cutMask.height();
	unsigned* p = (unsigned*)cutMask.bits();
	for (int i = 0; i<w*h; i++)
	{
		if (p[i] == BACKGROUND_CUT_VALUE)
		{
			mask.data[i] = GC_BGD;
		}
		else if (p[i] == FOREGROUND_CUT_VALUE)
		{
			mask.data[i] = GC_FGD;
		}
	}
}

void GenerateTrimap::setRectInMask(Rect rect)
{
	assert(!mask.empty());
	mask.setTo(GC_BGD);	//GC_BGD == 0
	this->rect.x = max(0, rect.x);
	this->rect.y = max(0, rect.y);
	this->rect.width = min(rect.width, matInput.cols - rect.x);
	this->rect.height = min(rect.height, matInput.rows - rect.y);
	(mask(this->rect)).setTo(Scalar(GC_PR_FGD));	//GC_PR_FGD == 3�������ڲ�,Ϊ���ܵ�ǰ����
}

void GenerateTrimap::getBinMask(const Mat& comMask, Mat& binMask)
{
	if (comMask.empty() || comMask.type() != CV_8UC1)
		CV_Error(CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
		binMask.create(comMask.size(), CV_8UC1);
	//�õ�mask�����λ, ʵ������ֻ����ȷ���Ļ����п��ܵ�ǰ���㵱��mask	
	binMask = comMask & 1;
}

void GenerateTrimap::initMask(const Mat& input)
{
	matInput = input;
	mask = Mat(input.rows, input.cols, CV_8UC1);
}
void GenerateTrimap::processGrabcut(GrabcutMode mode)
{
	if (mode == GRABCUT_WITH_RECT)
	{
		grabCut(matInput, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);
		getBinMask(mask, binmask);
	}
	else if (mode == GRABCUT_WITH_MASK)
	{
		grabCut(matInput, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK);
		getBinMask(mask, binmask);
	}
	else
	{
		grabCut(matInput, mask, rect, bgdModel, fgdModel, 1);
		getBinMask(mask, binmask);
	}

}
void GenerateTrimap::generateComputeArea(QImage& output)
{
	int w = matInput.cols;
	int h = matInput.rows;
	int nPixels = w*h;
	QSize size(w, h);
	float* maskImage = new float[nPixels];
	for (int i = 0; i<nPixels; i++)
	{
		int y = i / w;
		int x = i%w;
		if ((int)binmask.at<uchar>(y, x) == 0)	//�����ܵģ�����
			maskImage[i] = 1.0f;
		if ((int)binmask.at<uchar>(y, x) == 1)	//�����ܵģ�ǰ��
			maskImage[i] = 0.0f;
	}
	/*{
	Mat forShow=Mat(h,w,CV_8UC1);
	for(int i=0;i<w*h;i++)
	{
	int y = i/w;
	int x = i%w;
	if((int)maskImage[i]==0)
	forShow.at<uchar>(y,x)=0;
	if((int)maskImage[i]==1)
	forShow.at<uchar>(y,x)=255;
	}
	imshow("maskImage",forShow);
	waitKey(100);
	}*/


	/*{
	Mat forShow=Mat(h,w,CV_8UC1);
	for(int i=0;i<w*h;i++)
	{
	int y = i/w;
	int x = i%w;
	forShow.at<uchar>(y,x)=255-255*maskImage[i];
	}
	imshow("maskImage",forShow);
	waitKey(100);
	imwrite("maskImage.png",forShow);
	}*/

	float* blurImage = new float[nPixels];
	blurMask(maskImage, size, blurImage);	//��˹ģ��

											//{
											//Mat forShow=Mat(h,w,CV_8UC1);
											//for(int i=0;i<w*h;i++)
											//{
											//int y = i/w;
											//int x = i%w;
											//if((int)blurImage[i]==0)	
											//forShow.at<uchar>(y,x)=0;
											//if((int)blurImage[i]==1)
											//forShow.at<uchar>(y,x)=255;
											//}
											//imshow("blurImage",forShow);
											//waitKey(100);
											//}

											/*{
											Mat forShow=Mat(h,w,CV_8UC1);
											for(int i=0;i<w*h;i++)
											{
											int y = i/w;
											int x = i%w;
											forShow.at<uchar>(y,x)=255-255*blurImage[i];
											}
											imshow("blurImage",forShow);
											waitKey(100);
											imwrite("blurImage.png",forShow);
											}*/

	QImage img(w, h, QImage::Format_ARGB32);
	extractComputeArea(blurImage, size, img);	//��ȡ��������

												/*{
												Mat forShow(binmask);
												for(int i=0;i<w*h;i++)
												{
												int y = i/w;
												int x = i%w;
												if((int)binmask.at<uchar>(y,x)==0)
												forShow.at<uchar>(y,x)=0;
												if((int)binmask.at<uchar>(y,x)==1)
												forShow.at<uchar>(y,x)=255;
												}
												imshow("binmask",forShow);
												waitKey(100);
												}*/

	output = img;
}

void GenerateTrimap::setGaussianKernelSize(float size)
{
	computeKernel(size);
	thickness = (size * 10) / 2;
}

void GenerateTrimap::computeKernel(float sigma)	//�����˹�˾���
{
	float coef = 1.0f / sqrt(2 * M_PI * sigma * sigma);
	float expCoef = -1.0f / (2 * sigma * sigma);
	float* data = gausiansKernel/*.data()*/;
	gausianTotalWeight = 0.0f;

	for (int y = -BLUR_KERNEL_HALF_SIZE, ithPixel = 0; y <= BLUR_KERNEL_HALF_SIZE; ++y)
	{
		for (int x = -BLUR_KERNEL_HALF_SIZE; x <= BLUR_KERNEL_HALF_SIZE; ++x, ++ithPixel)
		{
			float gaussWeight = coef * coef * exp((x*x + y*y) * expCoef);
			gausianTotalWeight += gaussWeight;
			//qDebug("%d %f\n", ithPixel, gaussWeight);
		}
	}

	gausianTotalWeight = sqrt(gausianTotalWeight);
	for (int y = -BLUR_KERNEL_HALF_SIZE, ithPixel = 0; y <= BLUR_KERNEL_HALF_SIZE; ++y)
	{
		data[y + BLUR_KERNEL_HALF_SIZE] = coef * exp(y * y * expCoef) / gausianTotalWeight;	//��һ����һά��˹��
	}
}

void GenerateTrimap::blurMask(const float* maskArray, const QSize& size, float* blurredMaskArray)		//����һά��˹ģ��
{
	int width = size.width();
	int height = size.height();
	float* gausiansKernelData = gausiansKernel/*.data()*/;
	float* temp = new float[width * height];

	for (int ithPixel = 0; ithPixel < width * height; ++ithPixel)
	{
		temp[ithPixel] = 0;
		blurredMaskArray[ithPixel] = 0;
	}

	//y����һά��˹ģ��
	for (int offset = -BLUR_KERNEL_HALF_SIZE; offset <= BLUR_KERNEL_HALF_SIZE; ++offset)
	{
		float *pDst = temp;
		float factor = gausiansKernelData[offset + BLUR_KERNEL_HALF_SIZE];
		for (int y = 0; y < -offset; y++)
		{
			for (int x = 0; x < width; ++x, ++pDst)
			{
				*pDst += maskArray[x] * factor;
			}
		}

		int midY0 = qMax(0, -offset);
		int midY1 = qMin(height, height - offset);
		for (int y = midY0; y < midY1; ++y)
		{
			const float *pRow = maskArray + (y + offset) * width;
			for (int x = 0; x < width; ++x, ++pRow, ++pDst)
			{
				*pDst += (*pRow) * factor;
			}
		}

		const float* pLast = maskArray + (height - 1) * width;
		for (int y = midY1; y < height; ++y)
		{
			for (int x = 0; x < width; ++x, ++pDst)
			{
				*pDst += pLast[x] * factor;
			}
		}
	}

	//x����һά��˹ģ��
	for (int offset = -BLUR_KERNEL_HALF_SIZE; offset <= BLUR_KERNEL_HALF_SIZE; ++offset)
	{
		float factor = gausiansKernelData[offset + BLUR_KERNEL_HALF_SIZE];
		for (int y = 0; y < height; ++y)
		{
			float* pRow = blurredMaskArray + y * width;
			float tempV = temp[y * width] * factor;
			for (int x = 0; x < -offset; x++, ++pRow)
			{
				*pRow += tempV;
			}
		}

		int midX0 = qMax(0, -offset);
		int midX1 = qMin(width, width - offset);
		for (int y = 0; y < height; ++y)
		{
			float* pSrcRow = temp + y * width + qMax(0, offset);
			float* pDstRow = blurredMaskArray + y * width + midX0;
			for (int x = midX0; x < midX1; ++x, ++pSrcRow, ++pDstRow)
			{
				*pDstRow += *pSrcRow * factor;
			}
		}

		for (int y = 0; y < height; ++y)
		{
			float tempV = temp[y * width + width - 1] * factor;
			float* pDstRow = blurredMaskArray + y * width + midX1;
			for (int x = midX1; x < width; ++x, ++pDstRow)
			{
				*pDstRow += tempV;
			}
		}
	}
	delete[] temp;
}

void GenerateTrimap::extractComputeArea(float* blurred, const QSize& size, QImage& output)	//��ģ�����ͼ�񣬼�����Ӧ�ݶȣ��ҵ���������
{
	float maxValue = 0;
	float* srcPixel = blurred;

	// �����ݶ�
	for (int y = 0, ithPixel = 0; y < size.height(); ++y)
	{
		int isFinalY = (y != size.height() - 1) * size.width();	//y�������һ�����أ���isFinalYΪwidth��y�����һ�����أ���isFinalYΪ0

		for (int x = 0; x < size.width(); ++x, ++srcPixel)
		{
			int isFinalX = x != size.width() - 1;		//trueΪ1��falseΪ0��x�������һ�����أ���isFinalXΪ1��x�����һ�����أ���isFinalXΪ0
			float deltaX = *(srcPixel + isFinalX) - *srcPixel;	//�ұߵ�����-��ǰ	��x+1��
			float deltaY = *(srcPixel + isFinalY) - *srcPixel;	//�±ߵ�����-��ǰ	��y+1��
			*srcPixel = sqrt(deltaX * deltaX + deltaY * deltaY) * 100.0;
			maxValue = qMax(maxValue, *srcPixel);	//����ݶ�
		}
	}
	if (output.size() != size)
	{
		output = QImage(size, QImage::Format_ARGB32);
	}
	output.fill(BACKGROUND_AREA_VALUE);
	unsigned* data = (unsigned*)output.bits();

	float invMaxValue = maxValue <= 1e-5f ? 0.0 : 1.0f / maxValue;	//����

																	//#define  REMOVE_ISLAND	//�����ڲ��׶���һ��Ҫȥ������������Ҫ��һ��
#ifndef REMOVE_ISLAND

	for (int ithPixel = 0; ithPixel < size.width() * size.height(); ++ithPixel)
	{
		if (blurred[ithPixel] * invMaxValue  > thresholdGrad)
		{
			data[ithPixel] = COMPUTE_AREA_VALUE;
		}
	}

	for (int i = 0; i<size.width(); i++)
	{
		for (int j = size.height() - 1; j>0; j--)
		{
			unsigned* pixel = data + j*size.width() + i;
			if (binmask.at<uchar>(j, i) == 1 && *pixel != COMPUTE_AREA_VALUE)
			{
				*pixel = FOREGROUND_AREA_VALUE;
			}
		}
	}
#else
	Mat unkmask = Mat::zeros(size.height(), size.width(), CV_8UC1);
	for (int j = 0, ithPixel = 0; j < size.height(); ++j)
	{
		for (int i = 0; i < size.width(); ++i, ++ithPixel)
		{
			if (blurred[ithPixel] * invMaxValue  > thresholdGrad)
			{
				//data[ithPixel] = COMPUTE_AREA_VALUE;
				unkmask.ptr<char>(j)[i] = 1;
			}
		}
	}

	{
		int w = size.width(), h = size.height();
		Mat forShow(unkmask);
		for (int i = 0; i<w*h; i++)
		{
			int y = i / w;
			int x = i%w;
			if ((int)unkmask.at<uchar>(y, x) == 0)
				forShow.at<uchar>(y, x) = 0;
			if ((int)unkmask.at<uchar>(y, x) == 1)
				forShow.at<uchar>(y, x) = 255;
		}
		imshow("unkmask1", forShow);
		waitKey(100);
	}

	//Mat element = Mat::ones(8,8,CV_8UC1);	
	//morphologyEx(unkmask,unkmask,cv::MORPH_CLOSE,element);	//������

	//{
	//	int w=size.width(),h=size.height();
	//	Mat forShow(unkmask);
	//	for(int i=0;i<w*h;i++)
	//	{
	//		int y = i/w;
	//		int x = i%w;
	//		if((int)unkmask.at<uchar>(y,x)==0)	
	//			forShow.at<uchar>(y,x)=0;
	//		if((int)unkmask.at<uchar>(y,x)==1)	
	//			forShow.at<uchar>(y,x)=255;
	//	}
	//	imshow("unkmask",forShow);
	//	waitKey(100);
	//}

	// ͨ��findContours������������(������)
	// ����������contours��
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(unkmask, contours, hierarchy,
		//CV_RETR_EXTERNAL,		//ֻ���������
		CV_RETR_CCOMP,		//�������������Ȼ����֯��2��ģʽ���������ⲿ�������ڶ������ڲ�����������������滹��Ƕ�������������³�Ϊ2����ϵ��
		CV_CHAIN_APPROX_SIMPLE,	//ѹ��ˮƽ���򣬴�ֱ���򣬶Խ��߷����Ԫ�أ�ֻ�����÷�����յ����꣬���磬һ����������ֻ��4����
		Point(0, 0));

	// ���������ܳ�
	// ���ܳ�����������Ϊ��������
	int max_length = 0;
	int max_index;
	int second_length = 0;
	int second_index = 0;	//��ʼ��Ϊ0�������Ҳ����ڶ������������Ψһ������Ҳ��Ϊ�ڶ�������Ϊǰ��
	int curr_length = 0;

	for (int i = 0; i<contours.size(); i++)
	{
		curr_length = contours[i].size();
		if (curr_length > max_length)
		{
			max_length = curr_length;
			max_index = i;
		}
	}
#define MAX_BIG_REGIONS 20
	int bigRegionIndices[MAX_BIG_REGIONS];
	int nBigRegions = 0;

	for (int i = 0; i<contours.size(); i++)
	{
		curr_length = contours[i].size();
		if ((float)(curr_length) / (float)max_length > 0.2f && nBigRegions < MAX_BIG_REGIONS)
		{
			bigRegionIndices[nBigRegions++] = i;
		}
	}

	// �Ա���������л��ƣ��������򶼲����ƣ��ﵽ��ȥ����ͨ�����Ŀ��
	Scalar outside_contour_color(255, 0, 0);	// ��������Ļ�����ɫ
	Scalar inside_contour_color(0, 0, 255);
	Mat result = Mat::zeros(unkmask.size(), CV_8UC3);	// ���ƽ��
	for (int ithRegion = 0; ithRegion < nBigRegions; ++ithRegion)
	{
		drawContours(result, contours, bigRegionIndices[ithRegion], outside_contour_color, -1, 8, hierarchy, 0, Point(0, 0));		//�������Ϊoutside_contour_color
		drawContours(result, contours, bigRegionIndices[ithRegion], outside_contour_color, 2, 8, hierarchy, 0, Point(0, 0));
	}

	//for (int ithRegion = 0;ithRegion<nBigRegions;++ithRegion)	//�ҵ�����������һ����������Ҫ�ġ���
	//{
	//	if (hierarchy[bigRegionIndices[ithRegion]][3]==-1		//û�и�����
	//		&& hierarchy[bigRegionIndices[ithRegion]][2]!=-1)	//����������
	//	{
	//		drawContours(result, contours, hierarchy[bigRegionIndices[ithRegion]][2], inside_contour_color, -1, 8, hierarchy, 0, Point(0,0));		//���������Ϊinside_contour_color
	//	}
	//}
	if (nBigRegions >= 2)		//����ҵ������ϴ��������һ����������һ������������һ��������ǰ������
	{
		for (int ithRegion = 0; ithRegion < nBigRegions; ++ithRegion)	//�ҵ��ڶ�����������ٶ�Ϊ���������д���ȶ��������ǰ��ֻ��һ������
		{
			curr_length = contours[bigRegionIndices[ithRegion]].size();
			if (curr_length>second_length && curr_length<max_length)
			{
				second_length = curr_length;
				second_index = bigRegionIndices[ithRegion];
			}
		}
		drawContours(result, contours, second_index, inside_contour_color, -1, 8, hierarchy, 0, Point(0, 0));		//���������Ϊinside_contour_color
		drawContours(result, contours, second_index, outside_contour_color, 2, 8, hierarchy, 0, Point(0, 0));

		// ����result���ǽ���ˡ����Ϊǰ������Ϊ0������������Ϊ0��ͼ��
		for (int j = 0, ithPixel = 0; j < size.height(); ++j)
		{
			for (int i = 0; i < size.width(); ++i, ++ithPixel)
			{
				//b = result.ptr<uchar>(j)[i*3];			//B=(double)matInput.at<Vec3b>(j,i)[0];				
				//g= result.ptr<uchar>(j)[i*3+1];		//G=(double)matInput.at<Vec3b>(j,i)[1];
				//r = result.ptr<uchar>(j)[i*3+2];		//R=(double)matInput.at<Vec3b>(j,i)[2];

				if (result.ptr<uchar>(j)[i * 3] == 255)
				{
					data[ithPixel] = COMPUTE_AREA_VALUE;
				}

				if (result.ptr<uchar>(j)[i * 3 + 2] == 255)		//(0,0,255)
				{
					data[ithPixel] = FOREGROUND_AREA_VALUE;
				}
			}
		}
	}
	else		//nBigRegions=1��ֻ�ҵ�һ����������ǰ��������ͼ���б��ضϣ�unmask������һ����ͨ��Բ���������
	{
		// ����result���ǽ���ˡ����Ϊǰ������Ϊ0������������Ϊ0��ͼ��
		for (int j = 0, ithPixel = 0; j < size.height(); ++j)
		{
			for (int i = 0; i < size.width(); ++i, ++ithPixel)
			{
				if (result.ptr<char>(j)[i * 3] != 0)
				{
					data[ithPixel] = COMPUTE_AREA_VALUE;
				}
			}
		}

		for (int i = 0; i<size.width(); i++)
		{
			for (int j = size.height() - 1; j>0; j--)
			{
				unsigned* pixel = data + j*size.width() + i;
				if (binmask.at<uchar>(j, i) == 1 && *pixel != COMPUTE_AREA_VALUE)
				{
					*pixel = FOREGROUND_AREA_VALUE;
				}
			}
		}

	}

	{
		imshow("result", result);
		waitKey(100);
	}



#endif
}

PreprocessThread::PreprocessThread(GenerateTrimap* processor, QObject *parent /*= 0*/)//:fullSlot(0)
{
	this->processor = processor;
}

void InterpolationTrimap::interpolationTrimap(vector<FrameInfo> &frame, const list<int> &keyFrameNo, VideoCapture &capture, const Size size, string currentfilePath)
{
	Mat preframeMat, frameMat, pregray, gray, forwardFlow, backwardFlow;	//for computing optical flow
	Mat flowframe;		//image predicted by optical flow
	char filename[200];
	int interframeNum;
	vector<FlowError> flowError;		//һ����ֵ�����и�֡�����
	vector<Mat> backwardTrimap;		//һ����ֵ�����и�֡�ɷ�������õ���trimap

	const char* filePathCh = currentfilePath.c_str();
	//ofstream errorOut("error.txt",ios::app);

	list<int>::const_iterator it = keyFrameNo.cbegin();

	////ͳ��ʱ��	
	//string TimeFile ="interpolationTrimapTime.txt";
	//ofstream ftimeout(TimeFile,ios::app);
	//double interpolationTrimapT = (double)cvGetTickCount();

	while (*it != keyFrameNo.back())		//����ֱ��it�������һ��Ԫ��ֹͣѭ��
	{
		long begin = *it, end = *(++it);		//��ȡ�������ڵĹؼ�֡��ţ��õ�һ����ֵ����
		interframeNum = end - begin - 1;
		if (!flowError.empty())
		{
			flowError.clear();
		}
		if (!backwardTrimap.empty())
		{
			backwardTrimap.clear();
		}
		flowError.resize(interframeNum);
		backwardTrimap.resize(interframeNum + 1);
		backwardTrimap.at(end - begin - 1) = frame.at(end).trimap.clone();	//backwardtrimap����԰�����˳��洢�����һ��Ϊ�ؼ�֡��trimap

		for (int j = 0; j<interframeNum; j++)		//��ʼ�����
		{
			flowError.at(j).framePos = begin + 1 + j;	//��beginΪ��׼��֡λ��
			flowError.at(j).forwardErrorMap = Mat::zeros(size, CV_32FC1);
			flowError.at(j).backwardErrorMap = Mat::zeros(size, CV_32FC1);
			flowError.at(j).forwardAccumulatedError = Mat::zeros(size, CV_32FC1);
			flowError.at(j).backwardAccumulatedError = Mat::zeros(size, CV_32FC1);

			//flowError.at(j).validityBit = Mat(size,CV_8UC1, cv::Scalar(VALID));		//��ʼ��Ϊ1����ʾ���ŵļ���������Edited at 2015.05.27��
		}

		//���������������
		for (long i = begin; i<end; ++i)		//begin-->end
		{
			//QMessageBox::information(this, "Information", QString::number(i, 10),QMessageBox::Ok);
			capture.set(CV_CAP_PROP_POS_FRAMES, i);
			capture >> frameMat;
			cvtColor(frameMat, gray, COLOR_BGR2GRAY);

			flowframe = Mat(frameMat.rows, frameMat.cols, frameMat.type(), cv::Scalar(0, 0, 0));	//image predicted by optical flow

			if (i != begin)
			{
				if (pregray.data)
				{
					//for (int y = 0; y < flowframe.rows; y++)
					//{
					//	for (int x = 0;x < flowframe.cols; x++)
					//	{
					//		flowframe.ptr<uchar>(y)[x*3] = 0;			//b
					//		flowframe.ptr<uchar>(y)[x*3+1] = 0;		//g
					//		flowframe.ptr<uchar>(y)[x*3+2] = 0;		//r
					//	}
					//}				

					Ptr<DenseOpticalFlow> tvl1 = createOptFlow_DualTVL1();		//�����ܼ�����
					tvl1->calc(pregray, gray, forwardFlow);

					/*******************************************************************************************************************************/
					/*    ����ֻ��ǰ�������δ֪����Ĺ������д�������ֻ����frame.at(i-1).trimap.ptr<uchar>(y)[x]==MASK_FOREGROUND
					����MASK_COMPUTE��(newx, newy)��������Ĭ��Ϊ�������������Ļ���������ۻ����ȵļ��㶼Ҫ�޸�	  2015.06.16*/
					/*******************************************************************************************************************************/

					for (int y = 0; y < forwardFlow.rows; y++)
					{
						for (int x = 0; x< forwardFlow.cols; x++)
						{
							const Point2f& fxy = forwardFlow.at<Point2f>(y, x);
							if (isFlowCorrect(fxy))
							{
								int newy = cvRound(y + fxy.y) > 0 ? cvRound(y + fxy.y) : 0;
								newy = newy < forwardFlow.rows ? newy : (forwardFlow.rows - 1);
								int newx = cvRound(x + fxy.x) > 0 ? cvRound(x + fxy.x) : 0;
								newx = newx < forwardFlow.cols ? newx : (forwardFlow.cols - 1);
								frame.at(i).trimap.ptr<uchar>(newy)[newx] = frame.at(i - 1).trimap.ptr<uchar>(y)[x];		//��������������ɵ���Ϊ��ʼֵ�������Ϸ���������ɵĽ��и���

																															//������ǰһ֡Ԥ��õ���ǰ֡��Ӧ��ͼ��
								flowframe.ptr<uchar>(newy)[newx * 3] = preframeMat.ptr<uchar>(y)[x * 3];
								flowframe.ptr<uchar>(newy)[newx * 3 + 1] = preframeMat.ptr<uchar>(y)[x * 3 + 1];
								flowframe.ptr<uchar>(newy)[newx * 3 + 2] = preframeMat.ptr<uchar>(y)[x * 3 + 2];

								if (i != begin + 1)
								{
									flowError.at(i - begin - 1).forwardAccumulatedError.ptr<float>(newy)[newx] = flowError.at(i - begin - 1 - 1).forwardAccumulatedError.ptr<float>(y)[x];	//����һ֡(x,y)�����ۻ����ݵ���ǰ֡��Ӧλ��(newx,newy)
																																															//flowError.at(i-begin-1).validityBit.ptr<uchar>(newy)[newx] = flowError.at(i-begin-1-1).validityBit.ptr<uchar>(y)[x];		//����һ֡(x,y)������Чλ���ݵ���ǰ֡��Ӧλ��(newx,newy)	��Edited at 2015.05.27��
								}
							}
						}
					}
					/*imshow("flowframe",flowframe);
					waitKey(20);*/

					double r1, g1, b1;
					double r2, g2, b2;

					/*double error_value;
					int count=0;*/

					for (int y = 0; y<frameMat.rows; y++)
					{
						for (int x = 0; x<frameMat.cols; x++)
						{
							b1 = (double)frameMat.ptr<uchar>(y)[x * 3];
							g1 = (double)frameMat.ptr<uchar>(y)[x * 3 + 1];
							r1 = (double)frameMat.ptr<uchar>(y)[x * 3 + 2];

							b2 = (double)flowframe.ptr<uchar>(y)[x * 3];
							g2 = (double)flowframe.ptr<uchar>(y)[x * 3 + 1];
							r2 = (double)flowframe.ptr<uchar>(y)[x * 3 + 2];

							flowError.at(i - begin - 1).forwardErrorMap.ptr<float>(y)[x] = sqrt((b1 - b2) * (b1 - b2) + (g1 - g2) * (g1 - g2) + (r1 - r2) * (r1 - r2));		//���������Ԥ��ͼ����ʵ��ͼ�����ص�ŷ�Ͼ���
																																											//errorOut<<flowError.at(i-begin-1).forwardErrorMap.ptr<float>(y)[x]<<endl;
																																											/*error_value = flowError.at(i-begin-1).forwardErrorMap.ptr<float>(y)[x];
																																											if (error_value<30)
																																											{
																																											count++;
																																											}*/

							if (i == begin + 1)
							{
								flowError.at(i - begin - 1).forwardAccumulatedError.ptr<float>(y)[x] = flowError.at(i - begin - 1).forwardErrorMap.ptr<float>(y)[x];
								////Edited at 2015.05.27
								//if (flowError.at(i-begin-1).forwardErrorMap.ptr<float>(y)[x]>VALIDITY_THRESHOLD)
								//{
								//	flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] = INVALID;	//���������30������Чλ��Ϊ0
								//}
								////Edited at 2015.05.27
							}
							else
							{
								flowError.at(i - begin - 1).forwardAccumulatedError.ptr<float>(y)[x] += flowError.at(i - begin - 1).forwardErrorMap.ptr<float>(y)[x];
								//flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] = ( ( flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] ) && ( flowError.at(i-begin-1).forwardErrorMap.ptr<float>(y)[x]<VALIDITY_THRESHOLD ) );	//(Edited at 2015.05.27)
							}
						}
					}

					////Edited at 2015.05.28
					//imshow("orilTrimap",frame.at(i).trimap);
					//waitKey(20);

					//for (int y=0; y<frameMat.rows;y++)		
					//{
					//	for (int x=0; x<frameMat.cols;x++)
					//	{
					//		if (flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] == 0)	//��ЧλΪ�㣬����Ϊδ֪����
					//		{
					//			frame.at(i).trimap.ptr<uchar>(y)[x] = MASK_COMPUTE;
					//		}
					//	}
					//}

					//imshow("before improved trimap",frame.at(i).trimap);
					//waitKey(20);

					//imshow("frameMat",frameMat);
					//waitKey(20);

					//Mat alphaResult;
					//MyBayesian matting(frameMat,frame.at(i).trimap,DEFAULT_LAMDA,DEFAULT_ITERATION_TIMES);
					//matting.Solve();
					//matting.getAlphaResult(alphaResult);

					//for (int y=0; y<frameMat.rows;y++)		
					//{
					//	for (int x=0; x<frameMat.cols;x++)
					//	{
					//		if (alphaResult.ptr<float>(y)[x] == 1)	//definitely foreground
					//		{
					//			frame.at(i).trimap.ptr<uchar>(y)[x] =MASK_FOREGROUND;
					//			flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] = 1;
					//		}
					//		else if (alphaResult.ptr<float>(y)[x] ==0)	//definitely background
					//		{
					//			frame.at(i).trimap.ptr<uchar>(y)[x] =MASK_BACKGROUND;
					//			flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] = 1;
					//		}
					//		else if (alphaResult.ptr<float>(y)[x]>0 && alphaResult.ptr<float>(y)[x]<1)
					//		{
					//			frame.at(i).trimap.ptr<uchar>(y)[x] = MASK_COMPUTE;
					//			flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] = 0;
					//		}
					//	}
					//}

					//imshow("improved trimap",frame.at(i).trimap);
					//waitKey(20);
					////Edit at 2015.05.28

					Mat element = Mat::ones(8, 8, CV_8UC1);
					morphologyEx(frame.at(i).trimap, frame.at(i).trimap, cv::MORPH_CLOSE, element);	//������
				}
			}
			sprintf(filename, "%sTrimap/forwardtrimap%.4d.jpg", filePathCh, i);
			imwrite(filename, frame.at(i).trimap);


			std::swap(pregray, gray);	//��һ�μ���ʱ����ǰ֡�����һ֡	
			std::swap(preframeMat, frameMat);
		}

		//���������������
		for (long i = end; i>begin; --i)	//end-->begin
		{
			capture.set(CV_CAP_PROP_POS_FRAMES, i);
			capture >> frameMat;
			cvtColor(frameMat, gray, COLOR_BGR2GRAY);

			flowframe = Mat(frameMat.rows, frameMat.cols, frameMat.type(), cv::Scalar(0, 0, 0));

			if (i != end)
			{
				if (pregray.data)
				{
					backwardTrimap.at(i - begin - 1) = Mat(frameMat.rows, frameMat.cols, CV_8UC1, cv::Scalar(0));
					//for (int y=0;y<frameMat.rows;y++)
					//{
					//	for (int x=0;x<frameMat.cols;x++)
					//	{
					//		flowframe.ptr<uchar>(y)[x*3] = 0;		//b
					//		flowframe.ptr<uchar>(y)[x*3+1] = 0;		//g
					//		flowframe.ptr<uchar>(y)[x*3+2] = 0;		//r
					//	}
					//}

					Ptr<DenseOpticalFlow> tvl1 = createOptFlow_DualTVL1();
					tvl1->calc(pregray, gray, backwardFlow);

					for (int y = 0; y<backwardFlow.rows; y++)
					{
						for (int x = 0; x<backwardFlow.cols; x++)
						{
							const Point2f& fxy = backwardFlow.at<Point2f>(y, x);
							if (isFlowCorrect(fxy))
							{
								int newy = cvRound(y + fxy.y) > 0 ? cvRound(y + fxy.y) : 0;
								newy = newy <  backwardFlow.rows ? newy : (backwardFlow.rows - 1);
								int newx = cvRound(x + fxy.x) > 0 ? cvRound(x + fxy.x) : 0;
								newx = newx <  backwardFlow.cols ? newx : (backwardFlow.cols - 1);

								backwardTrimap.at(i - begin - 1).ptr<uchar>(newy)[newx] = backwardTrimap.at(i - begin - 1 + 1).ptr<uchar>(y)[x];		//���򴫲���j+1-->j

																																						//����Ԥ��õ���ͼ��
								flowframe.ptr<uchar>(newy)[newx * 3] = preframeMat.ptr<uchar>(y)[x * 3];
								flowframe.ptr<uchar>(newy)[newx * 3 + 1] = preframeMat.ptr<uchar>(y)[x * 3 + 1];
								flowframe.ptr<uchar>(newy)[newx * 3 + 2] = preframeMat.ptr<uchar>(y)[x * 3 + 2];

								if (i != end - 1)
								{
									flowError.at(i - begin - 1).backwardAccumulatedError.ptr<float>(newy)[newx] = flowError.at(i - begin - 1 + 1).backwardAccumulatedError.ptr<float>(y)[x];		//���չ����켣���ݵ�ǰ�ۻ����
																																																	//flowError.at(i-begin-1).validityBit.ptr<uchar>(newy)[newx] = flowError.at(i-begin-1+1).validityBit.ptr<uchar>(y)[x];		//����һ֡(x,y)������Чλ���ݵ���ǰ֡��Ӧλ��(newx,newy)	��Edited at 2015.05.28��
								}
							}
						}
					}

					/*imshow("flowframe",flowframe);
					waitKey(20);*/

					double r1, g1, b1;
					double r2, g2, b2;

					/*double error_value;
					int count=0;*/

					for (int y = 0; y<frameMat.rows; y++)
					{
						for (int x = 0; x<frameMat.cols; x++)
						{
							b1 = (double)frameMat.ptr<uchar>(y)[x * 3];
							g1 = (double)frameMat.ptr<uchar>(y)[x * 3 + 1];
							r1 = (double)frameMat.ptr<uchar>(y)[x * 3 + 2];

							b2 = (double)flowframe.ptr<uchar>(y)[x * 3];
							g2 = (double)flowframe.ptr<uchar>(y)[x * 3 + 1];
							r2 = (double)flowframe.ptr<uchar>(y)[x * 3 + 2];

							flowError.at(i - begin - 1).backwardErrorMap.ptr<float>(y)[x] = sqrt((b1 - b2) * (b1 - b2) + (g1 - g2) * (g1 - g2) + (r1 - r2) * (r1 - r2));		//���������Ԥ��ͼ����ʵ��ͼ�����ص�ŷ�Ͼ���

																																												/*error_value = flowError.at(i-begin-1).backwardErrorMap.ptr<float>(y)[x];
																																												if (error_value<30)
																																												{
																																												count++;
																																												}*/

							if (i == end - 1)	//����ǹؼ�֡����һ֡���ۻ������ǵ�ǰ���
							{
								flowError.at(i - begin - 1).backwardAccumulatedError.ptr<float>(y)[x] = flowError.at(i - begin - 1).backwardErrorMap.ptr<float>(y)[x];

								////Edited at 2015.05.28
								//if (flowError.at(i-begin-1).backwardErrorMap.ptr<float>(y)[x]>VALIDITY_THRESHOLD)
								//{
								//	flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] = INVALID;	//���������30������Чλ��Ϊ0
								//}
								////Edited at 2015.05.28

							}
							else
							{
								flowError.at(i - begin - 1).backwardAccumulatedError.ptr<float>(y)[x] += flowError.at(i - begin - 1).backwardErrorMap.ptr<float>(y)[x];
								//flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] = ( ( flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] ) && ( flowError.at(i-begin-1).backwardErrorMap.ptr<float>(y)[x]<VALIDITY_THRESHOLD ) );	//(Edited at 2015.05.28)
							}
						}
					}

					////Edited at 2015.05.28
					//imshow("orilbackTrimap",backwardTrimap.at(i-begin-1));
					//waitKey(20);

					//for (int y=0; y<frameMat.rows;y++)		
					//{
					//	for (int x=0; x<frameMat.cols;x++)
					//	{
					//		if (flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] == 0)	//��ЧλΪ�㣬����Ϊδ֪����
					//		{
					//			backwardTrimap.at(i-begin-1).ptr<uchar>(y)[x] = MASK_COMPUTE;
					//		}
					//	}
					//}

					//imshow("before improved back trimap",backwardTrimap.at(i-begin-1));
					//waitKey(20);

					//imshow("frameMat",frameMat);
					//waitKey(20);

					//Mat alphaResult;
					//MyBayesian matting(frameMat,backwardTrimap.at(i-begin-1),DEFAULT_LAMDA,DEFAULT_ITERATION_TIMES);
					//matting.Solve();
					//matting.getAlphaResult(alphaResult);

					//for (int y=0; y<frameMat.rows;y++)		
					//{
					//	for (int x=0; x<frameMat.cols;x++)
					//	{
					//		if (alphaResult.ptr<float>(y)[x] == 1)	//definitely foreground
					//		{
					//			backwardTrimap.at(i-begin-1).ptr<uchar>(y)[x] =MASK_FOREGROUND;
					//			flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] = 1;
					//		}
					//		else if (alphaResult.ptr<float>(y)[x] ==0)	//definitely background
					//		{
					//			backwardTrimap.at(i-begin-1).ptr<uchar>(y)[x] =MASK_BACKGROUND;
					//			flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] = 1;
					//		}
					//		else if (alphaResult.ptr<float>(y)[x]>0 && alphaResult.ptr<float>(y)[x]<1)
					//		{
					//			backwardTrimap.at(i-begin-1).ptr<uchar>(y)[x] = MASK_COMPUTE;
					//			flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] = 0;
					//		}
					//	}
					//}

					//imshow("improved back trimap",backwardTrimap.at(i-begin-1));
					//waitKey(20);
					////Edit at 2015.05.28

					for (int y = 0; y<frame.at(i).trimap.rows; y++)
					{
						for (int x = 0; x<frame.at(i).trimap.cols; x++)
						{
							//Edited at 2015.05.28,Ϊδ֪������ӳͷ���
							if (frame.at(i).trimap.ptr<uchar>(y)[x] == MASK_COMPUTE)
							{
								flowError.at(i - begin - 1).forwardAccumulatedError.ptr<float>(y)[x] += PENALTY_TERM;
							}
							if (backwardTrimap.at(i - begin - 1).ptr<uchar>(y)[x] == MASK_COMPUTE)
							{
								flowError.at(i - begin - 1).backwardAccumulatedError.ptr<float>(y)[x] += PENALTY_TERM;
							}
							//Edited at 2015.05.28

							if (flowError.at(i - begin - 1).backwardAccumulatedError.ptr<float>(y)[x] < flowError.at(i - begin - 1).forwardAccumulatedError.ptr<float>(y)[x])	//Ĭ��ֵΪ����Ԥ����������������������С�������Ϊ����Ԥ����
							{
								frame.at(i).trimap.ptr<uchar>(y)[x] = backwardTrimap.at(i - begin - 1).ptr<uchar>(y)[x];
							}
							else if (abs(flowError.at(i - begin - 1).backwardAccumulatedError.ptr<float>(y)[x] - flowError.at(i - begin - 1).forwardAccumulatedError.ptr<float>(y)[x])<FLT_EPSILON)	//�������ȡ��ֵ???
							{
								frame.at(i).trimap.ptr<uchar>(y)[x] = (frame.at(i).trimap.ptr<uchar>(y)[x] + backwardTrimap.at(i - begin - 1).ptr<uchar>(y)[x]) / 2;
							}
						}
					}

					/*	imshow("final trimap",frame.at(i).trimap);
					waitKey();*/

					Mat element = Mat::ones(8, 8, CV_8UC1);
					morphologyEx(backwardTrimap.at(i - begin - 1), backwardTrimap.at(i - begin - 1), cv::MORPH_CLOSE, element);	//������
					morphologyEx(frame.at(i).trimap, frame.at(i).trimap, cv::MORPH_CLOSE, element);	//������
				}
			}
			sprintf(filename, "%sTrimap/backwardtrimap%.4d.jpg", filePathCh, i);
			imwrite(filename, backwardTrimap.at(i - begin - 1));

			sprintf(filename, "%sTrimap/finaltrimap%.4d.jpg", filePathCh, i);
			imwrite(filename, frame.at(i).trimap);

			std::swap(pregray, gray);//��һ�μ���ʱ����ǰ֡�����һ֡	
			std::swap(preframeMat, frameMat);
		}
	}

	////ͳ��ʱ��
	//interpolationTrimapT=(double)cvGetTickCount()-interpolationTrimapT;
	//float interpolationTrimapTime=interpolationTrimapT/(cvGetTickFrequency()*1000);;
	//ftimeout<<interpolationTrimapTime<<"ms,	Frame number: "<<keyFrameNo.back()<<endl;
}