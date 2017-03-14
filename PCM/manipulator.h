#ifndef _MANIPULATOR
#define  _MANIPULATOR
#include "QGLViewer//manipulatedFrame.h"
#include "QGLViewer/camera.h"
#include <vector>
class Sample;
class SelectedObj;
namespace qglviewer
{
	class Camera;
}
class Manipulator : qglviewer::ManipulatedFrame
{
public:
	static enum SelectType{ VERTEX , FACE ,OBJECT };
	Manipulator( SelectType st ,SelectedObj& _so );
	~Manipulator();
	void draw();
	void drawAxis(qreal length);
	void mousePressEvent(QMouseEvent* const event,qglviewer::Camera* const camera);
	void mouseReleaseEvent(QMouseEvent* const event,qglviewer::Camera* const camera);
	void mouseMoveEvent(QMouseEvent* const event, qglviewer::Camera* const camera);
	void checkIfGrabsMouse(int x, int y, const qglviewer::Camera* const camera);
	bool checkIfReady()
	{

	}

private:

	Sample*		m_sample;
	SelectType		m_st;
	SelectedObj&	m_so;
};
struct SelectedObj
{
	Manipulator::SelectType			   s;
	std::vector<int>		vertex_index;
	std::vector<int>		  face_index;
	std::vector<int>        sample_index;

};
#endif