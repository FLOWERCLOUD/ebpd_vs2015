#include "VideoEditingWindow.h"
#include "videoedit_serialization.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include <list>
#include <vector>
#include <set>
using namespace std;

template<typename T> void convertSTDtoQT(vector<T>& v1, QVector<T>& v2)
{
	v2.clear();
	for (auto biter = v1.begin(); biter != v1.end(); ++biter)
	{
		v2.push_back(*biter);
	}
}
template<typename T> void convertSTDtoQT(list<T>& v1, QList<T>& v2)
{
	v2.clear();
	for (auto biter = v1.begin(); biter != v1.end(); ++biter)
	{
		v2.push_back(*biter);
	}
}
template<typename T> void convertSTDtoQT(set<T>& v1, QSet<T>& v2)
{
	v2.clear();
	for (auto biter = v1.begin(); biter != v1.end(); ++biter)
	{
		v2.insert(*biter);
	}
}


QImage cvMat2QImage(const cv::Mat& mat)
{
	// 8-bits unsigned, NO. OF CHANNELS = 1  
	if (mat.type() == CV_8UC1)
	{
		QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
		// Set the color table (used to translate colour indexes to qRgb values)  
		image.setColorCount(256);
		for (int i = 0; i < 256; i++)
		{
			image.setColor(i, qRgb(i, i, i));
		}
		// Copy input Mat  
		uchar *pSrc = mat.data;
		for (int row = 0; row < mat.rows; row++)
		{
			uchar *pDest = image.scanLine(row);
			memcpy(pDest, pSrc, mat.cols);
			pSrc += mat.step;
		}
		return image;
	}
	// 8-bits unsigned, NO. OF CHANNELS = 3  
	else if (mat.type() == CV_8UC3)
	{
		// Copy input Mat  
		const uchar *pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat  
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
		return image.rgbSwapped();
	}
	else if (mat.type() == CV_8UC4)
	{
		qDebug() << "CV_8UC4";
		// Copy input Mat  
		const uchar *pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat  
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
		return image.copy();
	}
	else
	{
		qDebug() << "ERROR: Mat could not be converted to QImage.";
		return QImage();
	}
}
cv::Mat QImage2cvMat(const QImage& image)
{
	cv::Mat mat;
	qDebug() << image.format();
	switch (image.format())
	{
	case QImage::Format_ARGB32:
	case QImage::Format_RGB32:
	case QImage::Format_ARGB32_Premultiplied:
		mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
		break;
	case QImage::Format_RGB888:
		mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
		cv::cvtColor(mat, mat, CV_BGR2RGB);
		break;
	case QImage::Format_Indexed8:
		mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
		break;
	}
	return mat;
}


QDataStream& operator<<(QDataStream& out, const FrameInfo&info)
{
	cv::Mat tempFrame;
	cvtColor(info.trimap, tempFrame, CV_BGR2RGB);
	QImage image = cvMat2QImage(tempFrame);
//	QImage image((const unsigned char*)(tempFrame.data), tempFrame.cols, tempFrame.rows, QImage::Format_RGB888);	
	out << info.framePos << info.ifKeyFrame << image;
	return out;
}
QDataStream& operator >> (QDataStream& in, FrameInfo& info)
{
	QImage image;
	int framePos;
	int ifKeyFrame;
	in >> framePos >> ifKeyFrame >> image;

	image = image.convertToFormat(QImage::Format_RGB32);
	info.framePos = framePos;
	info.ifKeyFrame = ifKeyFrame;
	info.trimap = QImage2cvMat(image);
	return in;
}
namespace videoEditting
{
	VSerialization::VSerialization()
	{
		currentframePos = totalFrameNumber = 0;
	}

	void VSerialization::open(const QString& fileName, VSerialization& v)
	{
		QFile file(fileName);
		if (file.open(QIODevice::ReadOnly))
		{
			QDataStream in(&file);
			in.setVersion(QDataStream::Qt_5_2);
			in >> v;
		}
	}

	void VSerialization::save(const QString& fileName, VSerialization& v)
	{
		QFile file(fileName);
		if (file.open(QIODevice::WriteOnly))
		{
			QDataStream out(&file);
			out.setVersion(QDataStream::Qt_5_2);
			out << v;
		}
	}

	void VSerialization::updateFrom()
	{
		VideoEditingWindow& window = VideoEditingWindow::getInstance();
		currentframePos =   window.currentframePos;
		totalFrameNumber = window.totalFrameNumber;
//		convertSTDtoQT(window.frame, frame);
//		convertSTDtoQT(window.keyFrameNo, keyFrameNo);
//		convertSTDtoQT(window.initKeyframeNo, initKeyframeNo);
		

	}

	QDataStream& operator<<(QDataStream& out, const VSerialization& v)
	{
		out << v.project_dir << v.Trimap_dir << v.Alpha_dir << v.ForeGround_dir << v.BackGround_dir
			<< v.currentframePos << v.totalFrameNumber << v.frame << v.keyFrameNo << v.initKeyframeNo
			<< v.comon_scene << v.object_orientaion << v.object_translation << v.corr_frame_idx
			<< v.camera_orientaion << v.camera_translation << v.corr_camera_frame_idx
			<< v.target << v.deltaAngle[0] << v.deltaAngle[1] << v.deltaLength << v.length
			<< v.projType << v.screenRes[0] << v.screenRes[1] << v.glwidgetRes[0] << v.viewoffeset[2]
			<< v.aspectRatio << v.fov_degree << v.nearplane << v.farplane;

		//	<< mesh.canvas;
		return out;
	}

	QDataStream& operator >> (QDataStream& in, VSerialization& v)
	{
		in >> v.project_dir >> v.Trimap_dir >> v.Alpha_dir >> v.ForeGround_dir >> v.BackGround_dir
			>> v.currentframePos >> v.totalFrameNumber >> v.frame >> v.keyFrameNo >> v.initKeyframeNo
			>> v.comon_scene >> v.object_orientaion >> v.object_translation >> v.corr_frame_idx
			>> v.camera_orientaion >> v.camera_translation >> v.corr_camera_frame_idx
			>> v.target >> v.deltaAngle[0] >> v.deltaAngle[1] >> v.deltaLength >> v.length
			>> v.projType >> v.screenRes[0] >> v.screenRes[1] >> v.glwidgetRes[0] >> v.viewoffeset[2]
			>> v.aspectRatio >> v.fov_degree >> v.nearplane >> v.farplane;
		return in;
	}

}

