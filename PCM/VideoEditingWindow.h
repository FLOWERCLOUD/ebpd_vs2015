#pragma once
#include "ui_VideoEditingScene.h"
#include "VideoEdittingParameter.h"
#include "GenerateTrimap.h"
#include <QtWidgets/QMainWindow>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <list>
#include <set>
#include <iterator>
#include <algorithm>
class VideoEditingWindow;
class PaintCanvas;
class CameraViewer;
class CameraWidget;
class ReadFrameWidget;
class VideoEditingWindow : public QMainWindow
{
	Q_OBJECT
public:
	static VideoEditingWindow& getInstance()
	{
		static VideoEditingWindow instance_;
		window_ = &instance_;
		return instance_;
	}
	~VideoEditingWindow();
private:
	VideoEditingWindow(QWidget *parent = 0);
	void setUp();
	void setUpSourceVideo();
	void setUpToolbox();
private slots:
    //file
	void openFile();
	void closeFile();
	void saveas();
	//tools
	void read_frame();
	void camera();
	void matting();
	void write_video();
	void alpha2Trimap();	
	void splitVideo();
	void computeGradient();

	//browser
	void nextFrame();
	void pause();
	void play();
	void preFrame();
	void turnToFrame(int curFrameNum);
	void nextInitKey();
	void setKeyFrame();


	//step1 tools
	void initKeyframe();
	void trimapInterpolation();
	void mattingVideo();
	void changeBackground();

	void increaseWidth(bool isIncrease);
	void setGaussianKernelSize(int kernel);
		
	void grabcutIteration();
	void showTrimap();
	void showKeyFrameNO();
	void mattingFrame();

	void reset()
	{
		updateMidImage();
	}
	void refresh();

	void cutInteract();
	void cutSelect();


	//step2 tools

	//step3 tools
private:
	void updateMidImage();
private:
	std::vector<FrameInfo> frame;	//��Ƶ֡����
	std::list<int> keyFrameNo;		//�ؼ�֡��λ��
	std::set<int> initKeyframeNo;		//֡��õ�������Ĺؼ�֡����
	long totalFrameNumber;	//��Ƶ��֡��

	std::string currentfilePath;	//��ǰ����Ƶ�ļ������ļ���
	PreprocessThread preprocessThread;
	GenerateTrimap preprocessor;

	CameraWidget *cameraWidget;
	ReadFrameWidget *rfWidget;


	FrameInfo * currentFrame;	//��ǰ֡
	QImage curImage;
	cv::Mat tempFrame;
	cv::Mat curframe;	//�����õ�ǰ֡
	cv::VideoCapture capture;
	cv::VideoWriter writer;
	QTimer *timer;
	double rate;		//֡��
	int delay;		//��֡���
	long currentframePos;	//��ǰ֡��
	cv::Mat back_ground;		//�µı���

private:
	Ui::VideoEditingWindow ui_;
	static VideoEditingWindow* window_;
	QLayout* opengllayout;
//	PaintCanvas* canvas;
	CameraViewer* camera_viewer;
};