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

	//����ѡ���ı���
	ToolMode curTool;	//��ǰ�Ĺ���
	BrushMode curBrush;	//��ǰ��ˢ��
	bool hasRegion;	//�Ƿ����ѡ��
	QPoint beginPointLocal, endPointLocal;	//ѡ���ĶԽǵ���ͼ������ϵ������
	cv::Rect rect;	//rect�а���Ҫ�ָ�Ķ���rect�ⶼ����Ϊ�ǣ�GC_FGD�����Եı������������ֻ��mode==GC_INIT_WITH_RECTʱʹ��; 

	QImage *srcImage;
	QImage maskImage;		//ǰ��������������ˢ���
	QImage resultImage;	//��ǰͼ��
	QImage maskCutImage;//cut��ˢ���

	QPoint lastPoint;
	QPoint curPointWorld;
	QPen strokePen, regionPen;

	int brushSize;
	bool isPainting;
	unsigned int brushValue;
	float translateValue[2];	//ͼƬ���Ͻ����������ԭ���λ��
	float scaleValue;	//����ֵ
	//
	bool isDragImage;
	QPoint initDragPoint;

};