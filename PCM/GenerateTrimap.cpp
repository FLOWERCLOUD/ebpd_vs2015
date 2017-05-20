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

void GenerateTrimap::interactMask(QImage cutMask)	//Grabcut交互
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
	(mask(this->rect)).setTo(Scalar(GC_PR_FGD));	//GC_PR_FGD == 3，矩形内部,为可能的前景点
}

void GenerateTrimap::getBinMask(const Mat& comMask, Mat& binMask)
{
	if (comMask.empty() || comMask.type() != CV_8UC1)
		CV_Error(CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
		binMask.create(comMask.size(), CV_8UC1);
	//得到mask的最低位, 实际上是只保留确定的或者有可能的前景点当做mask	
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
		if ((int)binmask.at<uchar>(y, x) == 0)	//（可能的）背景
			maskImage[i] = 1.0f;
		if ((int)binmask.at<uchar>(y, x) == 1)	//（可能的）前景
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
	blurMask(maskImage, size, blurImage);	//高斯模糊

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
	extractComputeArea(blurImage, size, img);	//提取计算区域

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

void GenerateTrimap::computeKernel(float sigma)	//计算高斯核矩阵
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
		data[y + BLUR_KERNEL_HALF_SIZE] = coef * exp(y * y * expCoef) / gausianTotalWeight;	//归一化的一维高斯核
	}
}

void GenerateTrimap::blurMask(const float* maskArray, const QSize& size, float* blurredMaskArray)		//两次一维高斯模糊
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

	//y方向一维高斯模糊
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

	//x方向一维高斯模糊
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

void GenerateTrimap::extractComputeArea(float* blurred, const QSize& size, QImage& output)	//对模糊后的图像，计算相应梯度，找到计算区域
{
	float maxValue = 0;
	float* srcPixel = blurred;

	// 计算梯度
	for (int y = 0, ithPixel = 0; y < size.height(); ++y)
	{
		int isFinalY = (y != size.height() - 1) * size.width();	//y不是最后一个像素，则isFinalY为width，y是最后一个像素，则isFinalY为0

		for (int x = 0; x < size.width(); ++x, ++srcPixel)
		{
			int isFinalX = x != size.width() - 1;		//true为1，false为0，x不是最后一个像素，则isFinalX为1，x是最后一个像素，则isFinalX为0
			float deltaX = *(srcPixel + isFinalX) - *srcPixel;	//右边的像素-当前	（x+1）
			float deltaY = *(srcPixel + isFinalY) - *srcPixel;	//下边的像素-当前	（y+1）
			*srcPixel = sqrt(deltaX * deltaX + deltaY * deltaY) * 100.0;
			maxValue = qMax(maxValue, *srcPixel);	//最大梯度
		}
	}
	if (output.size() != size)
	{
		output = QImage(size, QImage::Format_ARGB32);
	}
	output.fill(BACKGROUND_AREA_VALUE);
	unsigned* data = (unsigned*)output.bits();

	float invMaxValue = maxValue <= 1e-5f ? 0.0 : 1.0f / maxValue;	//倒数

																	//#define  REMOVE_ISLAND	//物体内部孔洞不一定要去除，后面这里要改一下
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
	//morphologyEx(unkmask,unkmask,cv::MORPH_CLOSE,element);	//闭运算

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

	// 通过findContours函数查找轮廓(点序列)
	// 轮廓保留在contours中
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(unkmask, contours, hierarchy,
		//CV_RETR_EXTERNAL,		//只检测外轮廓
		CV_RETR_CCOMP,		//检测所有轮廓，然后组织成2级模式，顶层是外部轮廓，第二级是内部洞的轮廓。如果里面还有嵌套轮廓，会重新成为2级关系。
		CV_CHAIN_APPROX_SIMPLE,	//压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如，一个矩形轮廓只需4个点
		Point(0, 0));

	// 查找最大的周长
	// 以周长最大的轮廓作为保留区域
	int max_length = 0;
	int max_index;
	int second_length = 0;
	int second_index = 0;	//初始化为0，以免找不到第二大的轮廓，则将唯一的轮廓也作为第二大的填充为前景
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

	// 对保留区域进行绘制，其余区域都不绘制，达到抛去不连通区域的目的
	Scalar outside_contour_color(255, 0, 0);	// 保留区域的绘制颜色
	Scalar inside_contour_color(0, 0, 255);
	Mat result = Mat::zeros(unkmask.size(), CV_8UC3);	// 绘制结果
	for (int ithRegion = 0; ithRegion < nBigRegions; ++ithRegion)
	{
		drawContours(result, contours, bigRegionIndices[ithRegion], outside_contour_color, -1, 8, hierarchy, 0, Point(0, 0));		//填充轮廓为outside_contour_color
		drawContours(result, contours, bigRegionIndices[ithRegion], outside_contour_color, 2, 8, hierarchy, 0, Point(0, 0));
	}

	//for (int ithRegion = 0;ithRegion<nBigRegions;++ithRegion)	//找到的子轮廓不一定是我们想要的……
	//{
	//	if (hierarchy[bigRegionIndices[ithRegion]][3]==-1		//没有父轮廓
	//		&& hierarchy[bigRegionIndices[ithRegion]][2]!=-1)	//存在子轮廓
	//	{
	//		drawContours(result, contours, hierarchy[bigRegionIndices[ithRegion]][2], inside_contour_color, -1, 8, hierarchy, 0, Point(0,0));		//填充子轮廓为inside_contour_color
	//	}
	//}
	if (nBigRegions >= 2)		//如果找到两个较大的轮廓，一个内轮廓，一个外轮廓，有一个完整的前景部分
	{
		for (int ithRegion = 0; ithRegion < nBigRegions; ++ithRegion)	//找到第二大的轮廓，假定为内轮廓，有待商榷……假设前景只有一个物体
		{
			curr_length = contours[bigRegionIndices[ithRegion]].size();
			if (curr_length>second_length && curr_length<max_length)
			{
				second_length = curr_length;
				second_index = bigRegionIndices[ithRegion];
			}
		}
		drawContours(result, contours, second_index, inside_contour_color, -1, 8, hierarchy, 0, Point(0, 0));		//填充内轮廓为inside_contour_color
		drawContours(result, contours, second_index, outside_contour_color, 2, 8, hierarchy, 0, Point(0, 0));

		// 变量result就是结果了。结果为前景背景为0、最大过渡区域为0的图像。
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
	else		//nBigRegions=1，只找到一个大轮廓，前景物体在图像中被截断，unmask并不是一个连通的圆，如半身像
	{
		// 变量result就是结果了。结果为前景背景为0、最大过渡区域为0的图像。
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
	vector<FlowError> flowError;		//一个插值区间中各帧的误差
	vector<Mat> backwardTrimap;		//一个插值区间中各帧由反向光流得到的trimap

	const char* filePathCh = currentfilePath.c_str();
	//ofstream errorOut("error.txt",ios::app);

	list<int>::const_iterator it = keyFrameNo.cbegin();

	////统计时间	
	//string TimeFile ="interpolationTrimapTime.txt";
	//ofstream ftimeout(TimeFile,ios::app);
	//double interpolationTrimapT = (double)cvGetTickCount();

	while (*it != keyFrameNo.back())		//正向，直到it到达最后一个元素停止循环
	{
		long begin = *it, end = *(++it);		//获取两个相邻的关键帧序号，得到一个插值区间
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
		backwardTrimap.at(end - begin - 1) = frame.at(end).trimap.clone();	//backwardtrimap序号仍按正向顺序存储，最后一个为关键帧的trimap

		for (int j = 0; j<interframeNum; j++)		//初始化误差
		{
			flowError.at(j).framePos = begin + 1 + j;	//以begin为基准的帧位置
			flowError.at(j).forwardErrorMap = Mat::zeros(size, CV_32FC1);
			flowError.at(j).backwardErrorMap = Mat::zeros(size, CV_32FC1);
			flowError.at(j).forwardAccumulatedError = Mat::zeros(size, CV_32FC1);
			flowError.at(j).backwardAccumulatedError = Mat::zeros(size, CV_32FC1);

			//flowError.at(j).validityBit = Mat(size,CV_8UC1, cv::Scalar(VALID));		//初始化为1（表示可信的计算结果）（Edited at 2015.05.27）
		}

		//正向光流及误差计算
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

					Ptr<DenseOpticalFlow> tvl1 = createOptFlow_DualTVL1();		//计算密集光流
					tvl1->calc(pregray, gray, forwardFlow);

					/*******************************************************************************************************************************/
					/*    考虑只对前景区域和未知区域的光流进行传播，即只计算frame.at(i-1).trimap.ptr<uchar>(y)[x]==MASK_FOREGROUND
					或者MASK_COMPUTE的(newx, newy)，其他的默认为背景区域，这样的话，后面的累积误差等的计算都要修改	  2015.06.16*/
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
								frame.at(i).trimap.ptr<uchar>(newy)[newx] = frame.at(i - 1).trimap.ptr<uchar>(y)[x];		//先以正向光流生成的作为初始值，后面结合反向光流生成的进行改善

																															//光流从前一帧预测得到当前帧对应的图像
								flowframe.ptr<uchar>(newy)[newx * 3] = preframeMat.ptr<uchar>(y)[x * 3];
								flowframe.ptr<uchar>(newy)[newx * 3 + 1] = preframeMat.ptr<uchar>(y)[x * 3 + 1];
								flowframe.ptr<uchar>(newy)[newx * 3 + 2] = preframeMat.ptr<uchar>(y)[x * 3 + 2];

								if (i != begin + 1)
								{
									flowError.at(i - begin - 1).forwardAccumulatedError.ptr<float>(newy)[newx] = flowError.at(i - begin - 1 - 1).forwardAccumulatedError.ptr<float>(y)[x];	//将上一帧(x,y)处的累积误差传递到当前帧对应位置(newx,newy)
																																															//flowError.at(i-begin-1).validityBit.ptr<uchar>(newy)[newx] = flowError.at(i-begin-1-1).validityBit.ptr<uchar>(y)[x];		//将上一帧(x,y)处的有效位传递到当前帧对应位置(newx,newy)	（Edited at 2015.05.27）
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

							flowError.at(i - begin - 1).forwardErrorMap.ptr<float>(y)[x] = sqrt((b1 - b2) * (b1 - b2) + (g1 - g2) * (g1 - g2) + (r1 - r2) * (r1 - r2));		//正向光流误差：预测图像与实际图像像素的欧氏距离
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
								//	flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] = INVALID;	//如果误差大于30，则有效位设为0
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
					//		if (flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] == 0)	//有效位为零，则设为未知像素
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
					morphologyEx(frame.at(i).trimap, frame.at(i).trimap, cv::MORPH_CLOSE, element);	//闭运算
				}
			}
			sprintf(filename, "%sTrimap/forwardtrimap%.4d.jpg", filePathCh, i);
			imwrite(filename, frame.at(i).trimap);


			std::swap(pregray, gray);	//下一次计算时，当前帧变成下一帧	
			std::swap(preframeMat, frameMat);
		}

		//反向光流及误差计算
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

								backwardTrimap.at(i - begin - 1).ptr<uchar>(newy)[newx] = backwardTrimap.at(i - begin - 1 + 1).ptr<uchar>(y)[x];		//反向传播，j+1-->j

																																						//光流预测得到的图像
								flowframe.ptr<uchar>(newy)[newx * 3] = preframeMat.ptr<uchar>(y)[x * 3];
								flowframe.ptr<uchar>(newy)[newx * 3 + 1] = preframeMat.ptr<uchar>(y)[x * 3 + 1];
								flowframe.ptr<uchar>(newy)[newx * 3 + 2] = preframeMat.ptr<uchar>(y)[x * 3 + 2];

								if (i != end - 1)
								{
									flowError.at(i - begin - 1).backwardAccumulatedError.ptr<float>(newy)[newx] = flowError.at(i - begin - 1 + 1).backwardAccumulatedError.ptr<float>(y)[x];		//按照光流轨迹传递当前累积误差
																																																	//flowError.at(i-begin-1).validityBit.ptr<uchar>(newy)[newx] = flowError.at(i-begin-1+1).validityBit.ptr<uchar>(y)[x];		//将上一帧(x,y)处的有效位传递到当前帧对应位置(newx,newy)	（Edited at 2015.05.28）
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

							flowError.at(i - begin - 1).backwardErrorMap.ptr<float>(y)[x] = sqrt((b1 - b2) * (b1 - b2) + (g1 - g2) * (g1 - g2) + (r1 - r2) * (r1 - r2));		//反向光流误差：预测图像与实际图像像素的欧氏距离

																																												/*error_value = flowError.at(i-begin-1).backwardErrorMap.ptr<float>(y)[x];
																																												if (error_value<30)
																																												{
																																												count++;
																																												}*/

							if (i == end - 1)	//如果是关键帧的下一帧，累积误差就是当前误差
							{
								flowError.at(i - begin - 1).backwardAccumulatedError.ptr<float>(y)[x] = flowError.at(i - begin - 1).backwardErrorMap.ptr<float>(y)[x];

								////Edited at 2015.05.28
								//if (flowError.at(i-begin-1).backwardErrorMap.ptr<float>(y)[x]>VALIDITY_THRESHOLD)
								//{
								//	flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] = INVALID;	//如果误差大于30，则有效位设为0
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
					//		if (flowError.at(i-begin-1).validityBit.ptr<uchar>(y)[x] == 0)	//有效位为零，则设为未知像素
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
							//Edited at 2015.05.28,为未知像素添加惩罚项
							if (frame.at(i).trimap.ptr<uchar>(y)[x] == MASK_COMPUTE)
							{
								flowError.at(i - begin - 1).forwardAccumulatedError.ptr<float>(y)[x] += PENALTY_TERM;
							}
							if (backwardTrimap.at(i - begin - 1).ptr<uchar>(y)[x] == MASK_COMPUTE)
							{
								flowError.at(i - begin - 1).backwardAccumulatedError.ptr<float>(y)[x] += PENALTY_TERM;
							}
							//Edited at 2015.05.28

							if (flowError.at(i - begin - 1).backwardAccumulatedError.ptr<float>(y)[x] < flowError.at(i - begin - 1).forwardAccumulatedError.ptr<float>(y)[x])	//默认值为正向预测结果，如果反向光流的误差更小，则更改为反向预测结果
							{
								frame.at(i).trimap.ptr<uchar>(y)[x] = backwardTrimap.at(i - begin - 1).ptr<uchar>(y)[x];
							}
							else if (abs(flowError.at(i - begin - 1).backwardAccumulatedError.ptr<float>(y)[x] - flowError.at(i - begin - 1).forwardAccumulatedError.ptr<float>(y)[x])<FLT_EPSILON)	//若相等则取均值???
							{
								frame.at(i).trimap.ptr<uchar>(y)[x] = (frame.at(i).trimap.ptr<uchar>(y)[x] + backwardTrimap.at(i - begin - 1).ptr<uchar>(y)[x]) / 2;
							}
						}
					}

					/*	imshow("final trimap",frame.at(i).trimap);
					waitKey();*/

					Mat element = Mat::ones(8, 8, CV_8UC1);
					morphologyEx(backwardTrimap.at(i - begin - 1), backwardTrimap.at(i - begin - 1), cv::MORPH_CLOSE, element);	//闭运算
					morphologyEx(frame.at(i).trimap, frame.at(i).trimap, cv::MORPH_CLOSE, element);	//闭运算
				}
			}
			sprintf(filename, "%sTrimap/backwardtrimap%.4d.jpg", filePathCh, i);
			imwrite(filename, backwardTrimap.at(i - begin - 1));

			sprintf(filename, "%sTrimap/finaltrimap%.4d.jpg", filePathCh, i);
			imwrite(filename, frame.at(i).trimap);

			std::swap(pregray, gray);//下一次计算时，当前帧变成下一帧	
			std::swap(preframeMat, frameMat);
		}
	}

	////统计时间
	//interpolationTrimapT=(double)cvGetTickCount()-interpolationTrimapT;
	//float interpolationTrimapTime=interpolationTrimapT/(cvGetTickFrequency()*1000);;
	//ftimeout<<interpolationTrimapTime<<"ms,	Frame number: "<<keyFrameNo.back()<<endl;
}