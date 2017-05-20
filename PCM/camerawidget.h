#pragma once
#include <QWidget>
#include <QPaintEvent>   
#include <QPainter>   
#include <QPoint> 
#include <QImage>   
#include <QtCore\QTimer>   
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"  
#include <QMessageBox>
#include <QCloseEvent>

using namespace cv;
using namespace std;

class CameraWidget : public QWidget
{
	Q_OBJECT

public:
	CameraWidget(QWidget *parent = 0);
	~CameraWidget();
protected:
	void paintEvent(QPaintEvent *e);
	void closeEvent(QCloseEvent *event);

	private slots:
	void nextFrame();

private:
	/*Ui::Camera ui;*/
	Mat frame;
	VideoCapture capture;
	QImage *image;
	QTimer *timer;
	VideoWriter writer;
};