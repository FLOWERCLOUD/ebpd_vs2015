#pragma once
#include <QSemaphore>
#include <QMutex>
#include <QDataStream>
#include <QImage>
#include <QDebug>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include "math.h"

#define fps	29.0	//֡��
#define perKF 15	//�ؼ�֡ƽ�����
//�죺0xffff0000       �̣�0xff00ff00       ����0xff0000ff  for QImage
#define COMPUTE_AREA_VALUE			0xff00ff00		//(��һ��0xff��ʾ͸����100%������ΪA��0xff����R(00)��G(ff)��B(00))
#define BACKGROUND_AREA_VALUE	0xff7f7f7f
#define FOREGROUND_AREA_VALUE	0xffffffff

#define FOREGROUND_CUT_VALUE		0xffff0000
#define BACKGROUND_CUT_VALUE		0xff0000ff

#define INIT_KERNELSIZE       2
#define BLUR_KERNEL_HALF_SIZE 20
#define BLUR_KERNEL_SIZE (BLUR_KERNEL_HALF_SIZE * 2 + 1)
#define INIT_THRESHOLD_GRAD   0.1

#define MASK_FOREGROUND 255	//for Mat trimap
#define MASK_BACKGROUND 0
#define MASK_COMPUTE    128

#define  SIGMA_WEIGHT 2   //myBayesian 
#define DEFAULT_LAMDA 2000
#define DEFAULT_ITERATION_TIMES 1
//Bayesian
#define ADJECENCY_N              255		//�����С	
static const double SIGMA = 8;		//��
static const double SIGMA_C = 0.01;

#define MIN_VAR        0.05f
#define MAX_ITERATION  50
#define MIN_LIKE       1e-6	
enum GrabcutMode { GRABCUT_WITH_RECT = 0, GRABCUT_WITH_MASK, GRABCUT_EVAL };

#define PENALTY_TERM 50	//�ۻ����ͷ���
//#define VALIDITY_THRESHOLD 200		//�����ֵ��Edited at 2015.05.28��
//#define VALID	1	//��Ч
//#define INVALID 0	//��Ч
struct FrameInfo	//ÿһ֡��Ӧ������
{
	int framePos;								//֡λ�ã�>=0��	
												/************************************************************************/
												/* �����Ᵽ��srcFrame,ͨ��֡λ������ʾ��Ӧ��ԭʼ֡�������ڴ����*/
												/************************************************************************/
	bool ifKeyFrame;							//�Ƿ�ؼ�֡
	cv::Mat trimap;									//��Ӧ��Trimap
	FrameInfo()
	{
		ifKeyFrame = false;		//Ĭ�ϲ��ǹؼ�֡
	}
	friend QDataStream& operator<<(QDataStream& out, const FrameInfo&info);
	friend QDataStream& operator >> (QDataStream& in, FrameInfo& infor);
};

struct FlowError	//�����������
{
	int framePos;								//֡λ�ã�>=0��	
	cv::Mat forwardErrorMap;					//ǰ�����		(��֡�����Կ�����һ����������Ϊ��������Ƕ����ģ��Ҳ���Ҫ����ģ�������Чλ�����ۻ����Ҫ���бȽϣ����Բ�����һ��)	
	cv::Mat backwardErrorMap;				//�������
	cv::Mat forwardAccumulatedError;		//ǰ���ۻ����
	cv::Mat backwardAccumulatedError;	//�����ۻ����
									//Mat validityBit;								//��Чλ��Edited at 2015.05.27��
};

class VedioMatting;

class GlobalData
{
public:
	static QSemaphore& getTrimapSema() { return trimapSemaphore; }
	static QSemaphore& getMattingSema() { return mattingSemaphore; }
	static QMutex    & getMutex() { return mutex; }
	static VedioMatting *dialog;
private:
	static QSemaphore trimapSemaphore;
	static QSemaphore mattingSemaphore;
	static QMutex     mutex;
};
enum CurrentStep
{
	STEP1,
	STEP2,
	STEP3
};
enum ShowMode
{
	IMAGEMODE,
    MANIPULATEMODE
};
extern CurrentStep g_curStep;
extern ShowMode    g_showmode;



QImage cvMat2QImage(const cv::Mat& mat);
cv::Mat QImage2cvMat(const QImage& image);