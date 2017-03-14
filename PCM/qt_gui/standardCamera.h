#pragma  once;
#include <QGLViewer/camera.h>

class StandardCamera : public qglviewer::Camera
{
	Q_OBJECT
public :
	StandardCamera();

	virtual qreal zNear() const;
	virtual qreal zFar() const;

	void toggleMode() { standard = !standard; }
	bool isStandard() { return standard; }

	void changeOrthoFrustumSize(int delta);
	virtual void getOrthoWidthHeight(GLdouble &halfWidth, GLdouble &halfHeight) const;
signals:
	void cameraChanged();
private :
	bool standard;
	float orthoSize;
};