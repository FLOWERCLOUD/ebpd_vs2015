#pragma  once

#include <QGLViewer/qglviewer.h>

class CameraViewer : public QGLViewer
{
public :
	CameraViewer(qglviewer::Camera* camera = NULL);
	qglviewer::Camera* setCamera(qglviewer::Camera* camera)
	{
		qglviewer::Camera* prev = camera;
		c = camera;
		return prev;
	}
protected :
	virtual void draw();
	virtual void init();

private :
	qglviewer::Camera* c;
};