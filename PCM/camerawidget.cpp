#include "camerawidget.h"
#include "VideoEdittingParameter.h"
#include <time.h>
#include <QtWidgets/QApplication>
using namespace std;
using namespace cv;
CameraWidget::CameraWidget(QWidget *parent)
	: QWidget(parent)
{
//	setupUi(this);
	QWidget *Camera = this;
	if (Camera->objectName().isEmpty())
		Camera->setObjectName(QStringLiteral("Camera"));
	Camera->resize(400, 300);
	Camera->setWindowTitle(QApplication::translate("Camera", "Camera", 0));
	QMetaObject::connectSlotsByName(Camera);


	/*ui.setupUi(this);*/
	//初始化处理，建立QImage和frame的关联，开启定时器
	capture.open(0);
	if (capture.isOpened())
	{
		QMessageBox::information(this, "Information", "Camera is Opened!", QMessageBox::Ok);
		capture >> frame;
		//imwrite("frame.jpg",frame);


		// 		time_t t = time(0); 
		// 		char tmp[20];
		// 		strftime( tmp, sizeof(tmp), "%Y.%m.%d.%H.%M.%S",localtime(&t) ); 		

		time_t t = time(NULL);  //获取当前系统的日历时间
		tm *local = localtime(&t);
		char time_name[20];
		sprintf(time_name, "%d.%d.%d.%d.%d.%d", \
			local->tm_year + 1900, local->tm_mon + 1, \
			local->tm_mday, local->tm_hour, local->tm_min, local->tm_sec);

		string videoName;
		videoName = time_name;

		videoName.append(".avi");

		QString filepath = "./";
		QString videoFilepath = filepath;
		string vedioFilename = videoFilepath.toStdString();
		vedioFilename.append(videoName);

		writer = VideoWriter(vedioFilename.c_str(), CV_FOURCC('X', 'V', 'I', 'D'), fps, Size(640, 480), true);

		if (!frame.empty())
		{
			flip(frame, frame, 1); //直接将图像采集卡采集的图像cvShowImage出来的是反着的图像，也就是图像采集卡采集的图像是以左下角为原点的，而窗口显示的图像原点是左上角，相当于是关于X轴翻转了。在显示图像之前使用cvFlip()函数将图像翻转一下就可以了。
			writer << frame;
			cvtColor(frame, frame, CV_BGR2RGB);
			this->resize(frame.cols, frame.rows);

			image = new QImage((const unsigned char*)(frame.data), frame.cols, frame.rows, QImage::Format_RGB888);


			//			imwrite("frame.jpg",frame);
			// 			imshow("frame",frame);
			// 			waitKey(10);
			timer = new QTimer(this);
			timer->setInterval(30);
			connect(timer, SIGNAL(timeout()), this, SLOT(nextFrame()));
			timer->start();
		}
		else
		{
			QMessageBox::information(this, "Information", "Frame is empty!", QMessageBox::Ok);
		}
	}
}

void CameraWidget::paintEvent(QPaintEvent *e)
{
	//更新图像
	QPainter painter(this);
	painter.drawImage(QPoint(0, 0), *image);
}

void CameraWidget::nextFrame()
{
	capture >> frame;
	if (!frame.empty())
	{
		flip(frame, frame, 1);
		writer << frame;
		cvtColor(frame, frame, CV_BGR2RGB);
		this->update();
	}
}

void CameraWidget::closeEvent(QCloseEvent *event)
{
	capture.release();
	delete image;
	delete timer;
	image = NULL;
	timer = NULL;
	writer.release();
}
CameraWidget::~CameraWidget()
{
	/*cvReleaseCapture(capture);*/
	/*capture.release();
	delete image;
	delete timer;
	writer.release();*/
	if (image)
		delete image;
	if (timer)
		delete timer;
}