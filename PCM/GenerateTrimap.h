#pragma once
#include "VideoEdittingParameter.h"

#include <QMainWindow>
#include <QThread>
#include <QMutex>
#include <QSemaphore>
#include <QSharedPointer>
#include <QImage>
#include <QVector2D>
#include <fstream>
#include <iostream>
#include "qmath.h"
#include <vector>
#include <list>
#include <iterator>
#include <algorithm>
#include <QMessageBox>

//#include <iostream>

using namespace std;
/*using std::fstream;*/


class GenerateTrimap
{
public:
	GenerateTrimap(void);
	~GenerateTrimap(void);

	void setGaussianKernelSize(float size);
	//grabCut
	void initMask(const cv::Mat& input);
	void processGrabcut(GrabcutMode mode = GRABCUT_EVAL);
	void generateComputeArea(QImage& output);
	void getBinMask(const cv::Mat& comMask, cv::Mat& binMask);
	void interactMask(QImage cutMask);
	void setRectInMask(cv::Rect rect);
	cv::Mat getMask();


	//		
	//void GrabCut_trimap( const ARGB* input, ARGB* output, int w, int h )
	//{
	//	QImage srcImage=QImage((const uchar*)input,w,h,QImage::Format_ARGB32);
	//	processGrabcut1(&srcImage,w,h);
	//	processGrabcut();
	//	generateComputeArea(output);
	//}

private:
	// �����˹�˲���
	void computeKernel(float sigma);
	// ģ��
	void blurMask(const float* maskArray, const QSize& size, float* blurredMaskArray);
	// ��ȡ��������
	void extractComputeArea(float* blurred, const QSize& size, QImage& output);

	//struct PixelPos{short x,y;};
	float thresholdGrad;

	cv::Mat mask;
	cv::Mat binmask;
	cv::Rect rect;
	cv::Mat bgdModel, fgdModel;
	cv::Mat matInput;
	//Mat edge;
	cv::Mat binTrimap;
	/*vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;	*/
	int thickness;
	float gausiansKernel[BLUR_KERNEL_SIZE];
	float  gausianTotalWeight;
};


class InterpolationTrimap
{
public:
	inline bool isFlowCorrect(cv::Point2f u)
	{
		return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
	}
	void interpolationTrimap(std::vector<FrameInfo> &frame, const std::list<int> &keyFrameNo, cv::VideoCapture &capture, const cv::Size size, std::string currentfilePath);
};


class PreprocessThread : public QThread
{
	Q_OBJECT
public:
	PreprocessThread(GenerateTrimap* processor, QObject *parent = 0);
	~PreprocessThread() {}

	void mat2QImage(const cv::Mat& srcMat, QImage& desQImage)
	{
		/*int nChannel=srcMat.channels();
		if (nChannel==3)
		{
		Mat srcTemp= Mat(srcMat.rows,srcMat.cols,srcMat.type());
		cvtColor(srcMat,srcTemp,CV_BGR2RGB);
		desQImage = QImage((const unsigned char*)(srcTemp.data),srcTemp.cols,srcTemp.rows,QImage::Format_RGB888);
		}
		else if (nChannel==4||nChannel==1)
		{
		desQImage = QImage((const unsigned char*)srcMat.data,srcMat.cols,srcMat.rows,QImage::Format_ARGB32);
		}*/

		int nChannel = srcMat.channels();
		if (nChannel == 3)
		{
			cv::Mat srcTemp = cv::Mat(srcMat.rows, srcMat.cols, srcMat.type());
			cvtColor(srcMat, srcTemp, CV_BGR2RGB);
			//unsigned char * matBits=(unsigned char *)srcMat.data;
			//unsigned char * qimageBits;
			desQImage = QImage(srcTemp.cols, srcTemp.rows, QImage::Format_RGB888);
			//int w=desQImage.width(),h=desQImage.height(),scanLine=desQImage.bytesPerLine();
			memcpy(desQImage.bits(), srcTemp.data, srcTemp.cols*srcTemp.rows * 3 * sizeof(unsigned char));	//ֱ�Ӹ������ݣ����ڿ������cols����4�ı������������ڴ��������⣬�Ǿ�һ��һ�еĽ���
																											//desQImage = QImage((const unsigned char*)(srcTemp.data),srcTemp.cols,srcTemp.rows,srcTemp.cols*srcTemp.channels(),QImage::Format_RGB888);		//desQImage��srcTemp�����ڴ棬srcTemp����ʱ��desQImage������Ҳû��
																											//qimageBits=(unsigned char *)desQImage.bits();
		}
		else if (nChannel == 4 || nChannel == 1)
		{
			desQImage = QImage((const unsigned char*)srcMat.data, srcMat.cols, srcMat.rows, srcMat.step, QImage::Format_ARGB32);
		}

	}

	void QImage2Mat(const QImage& srcQImage, cv::Mat& desMat)
	{
		cv::Mat matQ = cv::Mat(srcQImage.height(), srcQImage.width(), CV_8UC4, (uchar*)srcQImage.bits(), srcQImage.bytesPerLine());
		desMat = cv::Mat(matQ.rows, matQ.cols, CV_8UC3);
		int from_to[] = { 0,0, 1,1, 2,2 };
		cv::mixChannels(&matQ, 1, &desMat, 1, from_to, 3);
	}

	void setImage(const cv::Mat& srcImage)
	{
		this->srcImage = srcImage.clone();
	}
	void setTrimap(const QImage trimap)
	{
		dstImage = trimap;
	}
	const QImage& getTrimap() { return dstImage; }

	const cv::Mat& getBinmask() { return binmask; }

	void setDirty()
	{
		GlobalData::getTrimapSema().release(1);	//�ͷ�һ����Դ����һ���µ���Դ
		std::cout << "set dirty" << endl;
	}
	void GrabCutItera()
	{
		QImage temp;
		mat2QImage(srcImage, temp);
		processor->processGrabcut(GRABCUT_EVAL);
		processor->generateComputeArea(temp);
		dstImage = temp;
		emit changeTrimap();
	}
	void selectCut(cv::Rect rect)	//�Ծ��ο��ڵĲ��ֽ��г�ʼ��Grabcut
	{
		processor->initMask(srcImage);
		processor->setRectInMask(rect);	//���þ��ο���Ϊ���ܵ�ǰ�������ο���Ϊȷ������
		QImage temp;
		mat2QImage(srcImage, temp);

		////ͳ��ʱ��	
		//string TimeFile ="selectCutTime.txt";
		//ofstream ftimeout(TimeFile,ios::app);
		//double selectCutT = (double)cvGetTickCount();

		processor->processGrabcut(GRABCUT_WITH_RECT);
		processor->generateComputeArea(temp);

		////ͳ��ʱ��
		//selectCutT=(double)cvGetTickCount()-selectCutT;
		//float selectCutTime=selectCutT/(cvGetTickFrequency()*1000);;
		//ftimeout<<selectCutTime<<"ms"<<endl;

		dstImage = temp;
		emit changeTrimap();
	}
	void interactCut(QImage cutMask)	//��ӽ�����Grabcut
	{
		processor->interactMask(cutMask);	//���ȷ��ǰ��ȷ��������Ϣ
		QImage temp;
		mat2QImage(srcImage, temp);

		////ͳ��ʱ��	
		//string TimeFile ="interactCutTime.txt";
		//ofstream ftimeout(TimeFile,ios::app);
		//double interactCutT = (double)cvGetTickCount();

		processor->processGrabcut(GRABCUT_WITH_MASK);
		processor->generateComputeArea(temp);

		////ͳ��ʱ��
		//interactCutT=(double)cvGetTickCount()-interactCutT;
		//float interactCutTime=interactCutT/(cvGetTickFrequency()*1000);;
		//ftimeout<<interactCutTime<<"ms"<<endl;

		dstImage = temp;
		emit changeTrimap();
	}



signals:
	void changeTrimap();
protected:
	void run()
	{
		while (1)
		{

			int nRequests = GlobalData::getTrimapSema().available();
			std::cout << "nRequests available " << nRequests << endl;
			GlobalData::getTrimapSema().acquire(nRequests + 1);	//����nRequests+1����Դ�����û����Щ��Դ���߳̽���������ֱ�����㹻����Դ
			int nRequests2 = GlobalData::getTrimapSema().available();
			if (!srcImage.empty())	//Ŀǰֻ��������GaussianKernelʱ�Ż�releaseһ����Դ�����´���Ż�����
			{
				QImage temp;
				mat2QImage(srcImage, temp);

				GlobalData::getMutex().lock();	//ֱ���ͷ��������е���Դ���ܱ������̷߳���

				processor->processGrabcut();
				processor->generateComputeArea(temp);
				dstImage = temp;
				binmask = processor->getMask();
				emit changeTrimap();
				GlobalData::getMutex().unlock();
				std::cout << "changeTrimap " << endl;
			}

		}

	}
private:
	cv::Mat srcImage;
	QImage dstImage;
	cv::Mat binmask;
	GenerateTrimap* processor;
};