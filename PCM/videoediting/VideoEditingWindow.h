#pragma once
#include "basic_types.h"
#include "ui_VideoEditingScene.h"
#include "VideoEdittingParameter.h"
#include "GenerateTrimap.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "videoediting\VEScene.h"
#include "videoediting\ObjectTransformWidget.h"
#include "GlobalObject.h"
#include "videoedit_serialization.h"
#include <vector>
#include <list>
#include <set>
#include <iterator>
#include <algorithm>
#include <QtWidgets/QMainWindow>

class VideoEditingWindow;
class PaintCanvas;
namespace videoEditting
{
	class OGL_widget_skin_hidden;
	class GLViewWidget;
	enum VImageType{VORI_IMAGE=0, VTRIMAP=1, VALPHA=2, VSILHOUETTE=3};
}

class CameraWidget;
class ReadFrameWidget;
class BulletInterface;
class VideoEditingWindow : public QMainWindow
{
	Q_OBJECT

public:

	friend class videoEditting::VSerialization;
//	static QSharedPointer<videoEditting::Scene> scene;
	static VideoEditingWindow& getInstance();
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

	void openVideoFromVSerialization();
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

	void changeStepTab(int idex);
	void changeShowMode(int idx);

	//step1 tools

	/*
	   operation 0 : use saved file ,if saved file doesn't exists, use from cur setting
	   operation 1 : use cur setting
	*/
	bool checkIfCurFileExist(videoEditting::VImageType type,QImage& image,int operation = 0);
	void saveCurFrame(videoEditting::VImageType type , QImage& image);
	void saveCurFrame(videoEditting::VImageType type, cv::Mat& image);


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
	void importModel();
	void selectTool();
	void selectFaceTool();
	void moveTool();
	void rotateTool();
	void scaleTool();
	void focusTool();

	void runCurrentFramPoseEstimation();
	void setCurFramePoseEstimationAsKeyFrame();
	void runWholeFramePoseEstimation();
	void showCorrespondence();
	void updateCurSceneObjectPose();
	void updateCurSceneObjectPose(QString& objname);
	//step3 tools
	void begin_simulate();
	void pause_simulate();
	void continuie_simulate();
	void restart();
	void step_simulate();


private:
	void updateMidImage();
private:
	std::vector<FrameInfo> frame;	//视频帧序列
	std::list<int> keyFrameNo;		//关键帧的位置
	std::set<int> initKeyframeNo;		//帧差法得到的有序的关键帧序列
	int totalFrameNumber;	//视频总帧数

	std::string currentfilePath;	//当前打开视频文件所在文件夹
	std::string videoFilename;
	PreprocessThread preprocessThread;
	GenerateTrimap preprocessor;

	CameraWidget *cameraWidget;
	ReadFrameWidget *rfWidget;


	FrameInfo * currentFrame;	//当前帧
	QImage curImage;
	cv::Mat tempFrame;
	cv::Mat curframe;	//处理用当前帧
	cv::VideoCapture capture;
	cv::VideoWriter writer;
	QTimer *timer;
	double rate;		//帧率
	float delay;		//两帧间隔
	int currentframePos;	//当前帧数
	cv::Mat back_ground;		//新的背景


	


private:
	Ui::VideoEditingWindow ui_;
	static VideoEditingWindow* window_;
	QLayout* cameraviewlayout;
	QLayout* worldviewlayout;
//	PaintCanvas* canvas;
	videoEditting::GLViewWidget* camera_viewer;
	videoEditting::GLViewWidget* world_viewer;
	videoEditting::GLViewWidget* cur_active_viewer;
	videoEditting::OGL_widget_skin_hidden* _hidden;
	QSharedPointer<BulletInterface>        bullet_wrapper;
public:
	videoEditting::ObjectInfoWidget* transformEditor;
	void updateGLView();
	videoEditting::GLViewWidget* activated_viewer()
	{
		return cur_active_viewer;
	}
	void clearScene();
	void activate_viewer();
	void updateGeometryImage();
	videoEditting::VSerialization serializer;
	void open_project(QString& filename);
	void save_project(QString& filename);
};