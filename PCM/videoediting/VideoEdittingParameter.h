#pragma once
#include <QSemaphore>
#include <QMutex>
#include <QDataStream>
#include <QImage>
#include <QDebug>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include "math.h"

#define fps	29.0	//帧率
#define perKF 15	//关键帧平均间隔
//红：0xffff0000       绿：0xff00ff00       蓝：0xff0000ff  for QImage
#define COMPUTE_AREA_VALUE			0xff00ff00		//(第一个0xff表示透明度100%，依次为A（0xff），R(00)，G(ff)，B(00))
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
#define ADJECENCY_N              255		//邻域大小	
static const double SIGMA = 8;		//σ
static const double SIGMA_C = 0.01;

#define MIN_VAR        0.05f
#define MAX_ITERATION  50
#define MIN_LIKE       1e-6	
enum GrabcutMode { GRABCUT_WITH_RECT = 0, GRABCUT_WITH_MASK, GRABCUT_EVAL };

#define PENALTY_TERM 50	//累积误差惩罚项
//#define VALIDITY_THRESHOLD 200		//误差阈值（Edited at 2015.05.28）
//#define VALID	1	//有效
//#define INVALID 0	//无效
struct FrameInfo	//每一帧相应的数据
{
	int framePos;								//帧位置（>=0）	
												/************************************************************************/
												/* 不另外保存srcFrame,通过帧位置来表示对应的原始帧，以免内存过大*/
												/************************************************************************/
	bool ifKeyFrame;							//是否关键帧
	cv::Mat trimap;									//相应的Trimap
	FrameInfo()
	{
		ifKeyFrame = false;		//默认不是关键帧
	}
	friend QDataStream& operator<<(QDataStream& out, const FrameInfo&info);
	friend QDataStream& operator >> (QDataStream& in, FrameInfo& infor);
};

struct FlowError	//光流估计误差
{
	int framePos;								//帧位置（>=0）	
	cv::Mat forwardErrorMap;					//前向误差		(单帧误差可以考虑用一个变量，因为两个误差是独立的，且不需要保存的，类似有效位，而累积误差要进行比较，所以不能用一个)	
	cv::Mat backwardErrorMap;				//后向误差
	cv::Mat forwardAccumulatedError;		//前向累积误差
	cv::Mat backwardAccumulatedError;	//后向累积误差
									//Mat validityBit;								//有效位（Edited at 2015.05.27）
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