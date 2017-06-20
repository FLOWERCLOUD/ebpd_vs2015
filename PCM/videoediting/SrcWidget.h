#pragma once
#include <QtGui>
#include <QLabel>
#include <QWidget>
#include <QMouseEvent>
#include <QPoint>
#include <QWheelEvent>
#include <QImage>
#include "VideoEdittingParameter.h"
class SrcWidget : public QLabel
{
	Q_OBJECT
public:
	enum BrushMode { BRUSH_BACKGROUND = 0, BRUSH_COMPUTE_AREA, BRUSH_FOREGROUND, BRUSH_BACKGROUND_CUT, BRUSH_FOREGROUND_CUT };
	enum ToolMode { TOOL_SELECT = 0, TOOL_BRUSH };
	SrcWidget(QWidget *parent = 0, Qt::WindowFlags flags = 0);
	~SrcWidget(void);
	void clearRegion() { hasRegion = false; }
	void setImage(QImage* srcImage);
	void setTrimap(const QImage& trimap);
	void updateTrimap(const QImage& newMap);
	void updateDisplayImage();
	const QImage& getTrimap() { return maskImage; }
	const QImage& getMask() { return maskCutImage; }
	cv::Rect  getRect() { return rect; };
	bool getHasRegion() { return hasRegion; };
signals:
	void selectionTool();
	void increaseWidth(bool isIncrease);
	/*void changeBrushByKey(SrcWidget::BrushMode mode);*/
	void changeTrimap();
	void changeMask();
	void changeRect();
	private slots:
	void setSelectTool();
	void setBackgroundBrush();
	void setComputeAreaBrush();
	void setForegroundBrush();
	void setBrushSize(int size);
	void setForegroundCut();
	void setBackgroundCut();

protected:
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	/*void keyPressEvent(QKeyEvent *event);*/
	void wheelEvent(QWheelEvent* event);
	void paintEvent(QPaintEvent *event);
	void resizeEvent(QResizeEvent* event);

private:
	void paintTrimap(const QPoint &endPoint);
	void paintMask(const QPoint &endPoint);

	//关于选区的变量
	ToolMode curTool;	//当前的工具
	BrushMode curBrush;	//当前的刷子
	bool hasRegion;	//是否存在选区
	QPoint beginPointLocal, endPointLocal;	//选区的对角点在图像坐标系的坐标
	cv::Rect rect;	//rect中包含要分割的对象，rect外都被认为是（GC_FGD）明显的背景，这个参数只在mode==GC_INIT_WITH_RECT时使用; 

	QImage *srcImage;
	QImage maskImage;		//前景背景计算区域画刷结果
	QImage resultImage;	//当前图像
	QImage maskCutImage;//cut画刷结果

	QPoint lastPoint;
	QPoint curPointWorld;
	QPen strokePen, regionPen;

	int brushSize;
	bool isPainting;
	unsigned int brushValue;
	float translateValue[2];	//图片左上角相对于坐标原点的位移
	float scaleValue;	//缩放值
	//
	bool isDragImage;
	QPoint initDragPoint;

};