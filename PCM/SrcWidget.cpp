#include "SrcWidget.h"
#include <QMessageBox>
#include <iostream>
using namespace cv;
SrcWidget::SrcWidget(QWidget *parent /*= 0*/, Qt::WindowFlags flags /*= 0*/) :
	QLabel(parent)
{
	//QMessageBox::information(this, "SrcWidget", "SrcWidget1",QMessageBox::Ok);
	srcImage = NULL;

	isPainting = false;
	brushSize = 15;
	brushValue = COMPUTE_AREA_VALUE;
	translateValue[0] = translateValue[1] = 0.0f;
	scaleValue = 1.f;
	isDragImage = false;
	curTool = TOOL_SELECT;
	hasRegion = false;

	strokePen = QPen(QColor((QRgb)brushValue));
	strokePen.setWidth(2);
	regionPen = QPen(QColor(Qt::yellow));
	regionPen.setWidth(2);
	regionPen.setStyle(Qt::DashLine);

	//this->update();
}

SrcWidget::~SrcWidget(void)
{
}

void SrcWidget::setImage(QImage* srcImage)
{
	//QMessageBox::information(this, "setImage", "setImage",QMessageBox::Ok);
	this->srcImage = srcImage;
	maskImage = QImage(srcImage->width(), srcImage->height(), QImage::Format_ARGB32);
	translateValue[0] = translateValue[1] = 0.0f;
	scaleValue = 1.f;
	float widthscale = (float) this->size().width() / (float)srcImage->width();
	float heightscale = (float)this->size().height() / (float)srcImage->height();
	scaleValue = widthscale < heightscale ? widthscale : heightscale;
	//
	//maskCutImage=QImage(srcImage->width(),srcImage->height(), QImage::Format_ARGB32);
}

void SrcWidget::updateDisplayImage()
{
	if ( !srcImage || srcImage->size().isEmpty() )
	{
		return;
	}

	resultImage = *srcImage;

	QPainter painter(&resultImage);
	painter.setBrush(QBrush(QColor::fromRgba(0x30ffffff)));
	painter.drawRect(0, 0, resultImage.width(), resultImage.height());
	painter.setCompositionMode(QPainter::CompositionMode_Multiply);
	painter.drawImage(0, 0, maskImage);
	painter.drawImage(0, 0, maskCutImage);
	painter.end();
	QPixmap scaledPixmap = QPixmap::fromImage(resultImage).scaled(this->size());
	std::cout << "srcwidget size " << this->size().width() << " " << this->size().height() << std::endl;
	this->setPixmap(scaledPixmap);
	std::cout << "updateDisplayImage(): " << scaledPixmap.width() << " " << scaledPixmap.height() << std::endl;
//	this->resize(QSize(resultImage.width(), resultImage.height()));
}

void SrcWidget::mousePressEvent(QMouseEvent *event)
{
	if (event->button() == Qt::LeftButton)
	{
		lastPoint = event->pos();
		if (curTool == TOOL_BRUSH)
		{
			isPainting = true;
		}
		else if (curTool == TOOL_SELECT)
		{
			beginPointLocal.setX((lastPoint.x() - translateValue[0]) / scaleValue);
			beginPointLocal.setY((lastPoint.y() - translateValue[1]) / scaleValue);
			rect = Rect(beginPointLocal.x(), beginPointLocal.y(), 1, 1);
			hasRegion = false;
		}
	}
	else if ((event->buttons() & Qt::MiddleButton))
	{
		isDragImage = true;
		initDragPoint = event->pos();

	}
	this->update();	//call paintEvent()
}

void SrcWidget::mouseMoveEvent(QMouseEvent *event)
{
	QPoint releasePos = event->pos();
	if ((event->buttons() & Qt::LeftButton))
	{
		if (isPainting && curTool == TOOL_BRUSH && (curBrush == BRUSH_BACKGROUND || curBrush == BRUSH_COMPUTE_AREA || curBrush == BRUSH_FOREGROUND))
		{
			paintTrimap(releasePos);
		}
		else if (isPainting && curTool == TOOL_BRUSH && (curBrush == BRUSH_FOREGROUND_CUT || curBrush == BRUSH_BACKGROUND_CUT))
		{
			paintMask(releasePos);
		}
		else if (curTool == TOOL_SELECT)
		{
			hasRegion = true;
			endPointLocal.setX((releasePos.x() - translateValue[0]) / scaleValue);
			endPointLocal.setY((releasePos.y() - translateValue[1]) / scaleValue);
		}
	}
	else if ((event->buttons() & Qt::MiddleButton))
	{
		if (isDragImage)
		{
			QPoint curpoint = event->pos();
			QPoint offset = curpoint - initDragPoint;
			translateValue[0] += offset.x();
			translateValue[1] += offset.y();
			initDragPoint = curpoint;
		}
		

	}
	curPointWorld = event->pos();

	this->update();	//call paintEvent()
}

void SrcWidget::mouseReleaseEvent(QMouseEvent *event)
{
	if (event->button() == Qt::LeftButton)
	{
		QPoint releasePos = event->pos();
		if (isPainting && curTool == TOOL_BRUSH && (curBrush == BRUSH_BACKGROUND || curBrush == BRUSH_COMPUTE_AREA || curBrush == BRUSH_FOREGROUND))
		{
			paintTrimap(releasePos);
			isPainting = false;
			emit changeTrimap();	//发射Trimap改变的信号
		}
		else if (isPainting && curTool == TOOL_BRUSH && (curBrush == BRUSH_FOREGROUND_CUT || curBrush == BRUSH_BACKGROUND_CUT))
		{
			paintMask(releasePos);
			isPainting = false;
			emit changeMask();
		}
		else if (curTool == TOOL_SELECT)
		{
			endPointLocal.setX((releasePos.x() - translateValue[0]) / scaleValue);
			endPointLocal.setY((releasePos.y() - translateValue[1]) / scaleValue);
			hasRegion = (endPointLocal != beginPointLocal);
			rect = Rect(Point(rect.x, rect.y), Point(endPointLocal.x(), endPointLocal.y()));
			emit changeRect();
		}
	}
	else if ((event->buttons() & Qt::MiddleButton))
	{
		isDragImage = false;
		initDragPoint = QPoint();

	}
	this->update();	//call paintEvent()
}

void SrcWidget::paintTrimap(const QPoint &endPoint)
{

	QPainter painter(&maskImage);
	painter.setPen(QPen(QBrush(QColor::fromRgba(brushValue)), brushSize * 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));

	painter.scale(1 / scaleValue, 1 / scaleValue);
	painter.translate(-translateValue[0], -translateValue[1]);
	painter.drawLine(lastPoint, endPoint);
	lastPoint = endPoint;
	//maskImage.save("G:\\Liya\\Task\\video matting\\Data\\Picture\\maskimage.jpg");
	updateDisplayImage();
}

void SrcWidget::paintMask(const QPoint &endPoint)
{
	QSize s = maskCutImage.size();
	QPainter painter(&maskCutImage);
	painter.setPen(QPen(QBrush(QColor::fromRgba(brushValue)), brushSize * 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));

	painter.scale(1 / scaleValue, 1 / scaleValue);
	painter.translate(-translateValue[0], -translateValue[1]);

	painter.drawLine(lastPoint, endPoint);
	lastPoint = endPoint;
	updateDisplayImage();
}

void SrcWidget::setBackgroundBrush()
{
	curTool = TOOL_BRUSH;
	curBrush = BRUSH_BACKGROUND;
	brushValue = BACKGROUND_AREA_VALUE;
	//brushSize = 25;
	strokePen.setColor(QColor((QRgb)brushValue).darker(600));
}

void SrcWidget::setComputeAreaBrush()
{
	curTool = TOOL_BRUSH;
	curBrush = BRUSH_COMPUTE_AREA;
	brushValue = COMPUTE_AREA_VALUE;
	//brushSize = 25;
	strokePen.setColor(QColor((QRgb)brushValue));
}

void SrcWidget::setForegroundBrush()
{
	curTool = TOOL_BRUSH;
	curBrush = BRUSH_FOREGROUND;
	brushValue = FOREGROUND_AREA_VALUE;
	//brushSize = 25;
	strokePen.setColor(QColor((QRgb)brushValue));
}

void SrcWidget::setForegroundCut()
{
	curTool = TOOL_BRUSH;
	curBrush = BRUSH_FOREGROUND_CUT;
	brushValue = FOREGROUND_CUT_VALUE;
	//brushSize = 5;
	strokePen.setColor(QColor((QRgb)brushValue));
}

void SrcWidget::setBackgroundCut()
{
	curTool = TOOL_BRUSH;
	curBrush = BRUSH_BACKGROUND_CUT;
	brushValue = BACKGROUND_CUT_VALUE;
	//brushSize = 5;
	strokePen.setColor(QColor((QRgb)brushValue));
}

void SrcWidget::setBrushSize(int size)
{
	brushSize = size;
	update();
}

void SrcWidget::paintEvent(QPaintEvent *event)
{
	QPainter painter(this);

	painter.translate(translateValue[0], translateValue[1]);
	painter.scale(scaleValue, scaleValue);

	switch (g_curStep)
	{
	case STEP1:
	{
		painter.drawImage(0, 0, resultImage);
		if (hasRegion)
		{
			painter.setPen(regionPen);
			painter.drawRect(QRect(beginPointLocal, endPointLocal));
		}
		painter.resetTransform();
		painter.setPen(strokePen);
		if (curTool == TOOL_BRUSH)
		{
			painter.drawEllipse(curPointWorld, brushSize, brushSize);	//在当前位置显示画笔
		}
	}
	break;
	case STEP2:
	{

	}
		break;
	case STEP3:
	{


	}
		break;
	default:
		break;
	}





}

void SrcWidget::resizeEvent(QResizeEvent* event)
{

	std::cout << "SrcWidget resize: " << event->size().width() << " " << event->size().height() << std::endl;
	QLabel::resizeEvent(event);
	if (srcImage && !srcImage->size().isEmpty())
	{
		float widthscale = (float) this->size().width() / (float)srcImage->width();
		float heightscale = (float)this->size().height() / (float)srcImage->height();
		scaleValue = widthscale < heightscale ? widthscale : heightscale;
	}

	updateDisplayImage();
}
/*
 [tx] = [curPointWorld.x()] * [r] *[-curPointWorld.x()]*[tx]
 [ty]   [curPointWorld.y()]   [r]  [-curPointWorld.y()] [ty]
*/
void SrcWidget::wheelEvent(QWheelEvent* event)
{
	float r = qMax(0.1f, 1.0f + event->delta() / 1000.0f);
	translateValue[0] = curPointWorld.x() * (1 - r) + r * translateValue[0];
	translateValue[1] = curPointWorld.y() * (1 - r) + r * translateValue[1];
	scaleValue = scaleValue * r;
	update();
}

// void SrcWidget::keyPressEvent(QKeyEvent *event)	//快捷键
// {
// 	switch (event->key())
// 	{
// 	case Qt::Key_A:
// 		emit selectionTool();
// 		break;
// 	case Qt::Key_B:
// 		emit changeBrushByKey(BRUSH_BACKGROUND);
// 		break;
// 	case Qt::Key_C:
// 		emit changeBrushByKey(BRUSH_COMPUTE_AREA);
// 		break;
// 	case Qt::Key_F:
// 		emit changeBrushByKey(BRUSH_FOREGROUND);
// 		break;
// 	case Qt::Key_W:
// 		emit increaseWidth(true);
// 		break;
// 	case Qt::Key_S:
// 		emit increaseWidth(false);
// 		break;
// 	}
// }

void SrcWidget::setSelectTool()
{
	curTool = TOOL_SELECT;
}

void SrcWidget::updateTrimap(const QImage& newMap)
{
	//if (hasRegion)
	//{
	//	QPainter painter(&maskImage);
	//	int x = qMin(beginPointLocal.x(), endPointLocal.x());
	//	int y = qMin(beginPointLocal.y(), endPointLocal.y());
	//	int w = abs(endPointLocal.x() - beginPointLocal.x());
	//	int h = abs(endPointLocal.y() - beginPointLocal.y());
	//	painter.drawImage(x,y,newMap.copy(x,y,w,h));
	//}
	//else
	maskImage = newMap;
	maskCutImage = QImage(srcImage->width(), srcImage->height(), QImage::Format_ARGB32);
}

void SrcWidget::setTrimap(const QImage& trimap)
{
	maskImage = trimap;
}