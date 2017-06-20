#pragma once
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include <QWidget>
#include <QtGui\QPaintEvent>   
#include <QtGui\QPainter>   
#include <QtCore\QPoint> 
#include <QtGui\QImage>   
#include <QtCore\QTimer>   
#include <QMessageBox>
#include <QDir>
#include <QCloseEvent>



class ReadFrameWidget : public QWidget
{
	Q_OBJECT

public:
	ReadFrameWidget(QString filepath, QWidget *parent = 0);
	~ReadFrameWidget();

protected:
	void paintEvent(QPaintEvent *e);
	void closeEvent(QCloseEvent *event);
	private slots:
	void nextFrame();

private:
	/*Ui::readFrame ui;*/
	cv::Mat frame;
	QImage *image;
	QTimer *timer;

	QDir *dir;
	//struct _finddata_t file_info; //store file imformation
	//intptr_t  file_handle;	 //ÎÄ¼þµÄ¾ä±ú
	std::string filename;
	std::string filepath;
	int i;
	QFileInfoList list;
	QFileInfo fileInfo;
};