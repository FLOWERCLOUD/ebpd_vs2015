#include "readframe.h"
#include <QtWidgets/QApplication>
using namespace cv;
using namespace std;
ReadFrameWidget::ReadFrameWidget(QString file_path, QWidget *parent)
	: QWidget(parent)
{
	//setupUi(this);
	QWidget *readFrame = this;
	if (readFrame->objectName().isEmpty())
		readFrame->setObjectName(QStringLiteral("readFrame"));
	readFrame->resize(400, 300);
	readFrame->setWindowTitle(QApplication::translate("readFrame", "readFrame", 0));
	QMetaObject::connectSlotsByName(readFrame);


	//��ȡ�ļ����������ļ���������˳��
	dir = new QDir(file_path);
	dir->setFilter(QDir::Files/* | QDir::Hidden*/ | QDir::NoSymLinks);
	dir->setSorting(QDir::Name /* | QDir::Reversed*/);
	list = dir->entryInfoList();
	i = 0;
	fileInfo = list.at(i);

	filepath = file_path.toStdString();
	filename = filepath;
	filename.append(fileInfo.fileName().toLocal8Bit().constData());
	frame = imread(filename);

	if (!frame.empty())
	{
		cvtColor(frame, frame, CV_BGR2RGB);
		//imwrite("frame1.jpg",frame);
		this->resize(frame.cols, frame.rows);
		image = new QImage((const unsigned char*)(frame.data), frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
		//image->save("1.jpg");
		//���ö�ʱ������timeoutʱ������ȡ��һ֡
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


void ReadFrameWidget::paintEvent(QPaintEvent *e)
{
	QPainter painter(this);
	painter.drawImage(QPoint(0, 0), *image);	//���Ƶ�ǰ֡
}

void ReadFrameWidget::nextFrame()
{
	i++;
	if (i < list.size())
	{
		fileInfo = list.at(i);
		filename = filepath;
		filename.append(fileInfo.fileName().toLocal8Bit().constData());
		frame = imread(filename);
		//imwrite("frame.jpg",frame);
		if (!frame.empty())
		{
			cvtColor(frame, frame, CV_BGR2RGB);
			QImage img((const unsigned char*)(frame.data), frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
			*image = img;
			this->update();	//���µ�ǰ���ڣ����Ƶ�ǰ֡
		}
		//	image->save("2.jpg");
		//	imwrite("frame2.jpg",frame);
	}

}

void ReadFrameWidget::closeEvent(QCloseEvent *event)
{
	delete image;
	delete timer;
	delete dir;
	image = NULL;
	timer = NULL;
	dir   = NULL;
}

ReadFrameWidget::~ReadFrameWidget()
{
	if (image)
	{
		delete image;
	}
	if (timer)
	{
		delete timer;
	}
	if (dir)
	{
		delete dir;
	}

}