#include "manipulator.h"
#include "sample_set.h"
#include "GlobalObject.h"
#include "QGLViewer/camera.h"
#include <iostream>
#include <QMouseEvent>
using std::cout;
using std::endl;

Manipulator::Manipulator(SelectType st ,SelectedObj& _so) :
	m_sample(NULL),m_st(st) ,m_so(_so)
{
	//s.setManipulator( this);
}

Manipulator::~Manipulator()
{

}

void Manipulator::draw()
{
	glPushMatrix();
	glMultMatrixd(matrix());
	drawAxis( 1/5.0f);
	glPopMatrix();
}

void Manipulator::drawAxis(qreal length)
{
	const qreal charWidth = length / 40.0;
	const qreal charHeight = length / 30.0;
	const qreal charShift = 1.04 * length;

	GLboolean lighting, colorMaterial;
	glGetBooleanv(GL_LIGHTING, &lighting);
	glGetBooleanv(GL_COLOR_MATERIAL, &colorMaterial);

	glDisable(GL_LIGHTING);

	glBegin(GL_LINES);
	// The X
	glVertex3d(charShift,  charWidth, -charHeight);
	glVertex3d(charShift, -charWidth,  charHeight);
	glVertex3d(charShift, -charWidth, -charHeight);
	glVertex3d(charShift,  charWidth,  charHeight);
	// The Y
	glVertex3d( charWidth, charShift, charHeight);
	glVertex3d(0.0,        charShift, 0.0);
	glVertex3d(-charWidth, charShift, charHeight);
	glVertex3d(0.0,        charShift, 0.0);
	glVertex3d(0.0,        charShift, 0.0);
	glVertex3d(0.0,        charShift, -charHeight);
	// The Z
	glVertex3d(-charWidth,  charHeight, charShift);
	glVertex3d( charWidth,  charHeight, charShift);
	glVertex3d( charWidth,  charHeight, charShift);
	glVertex3d(-charWidth, -charHeight, charShift);
	glVertex3d(-charWidth, -charHeight, charShift);
	glVertex3d( charWidth, -charHeight, charShift);
	glEnd();

	glEnable(GL_LIGHTING);
	glDisable(GL_COLOR_MATERIAL);

	float color[4];
	color[0] = 0.7f;  color[1] = 0.7f;  color[2] = 1.0f;  color[3] = 1.0f;
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	QGLViewer::drawArrow(length, 0.01*length);

	color[0] = 1.0f;  color[1] = 0.7f;  color[2] = 0.7f;  color[3] = 1.0f;
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	glPushMatrix();
	glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
	QGLViewer::drawArrow(length, 0.01*length);
	glPopMatrix();

	color[0] = 0.7f;  color[1] = 1.0f;  color[2] = 0.7f;  color[3] = 1.0f;
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	glPushMatrix();
	glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
	QGLViewer::drawArrow(length, 0.01*length);
	glPopMatrix();

	if (colorMaterial)
		glEnable(GL_COLOR_MATERIAL);
	if (!lighting)
		glDisable(GL_LIGHTING);
}

void Manipulator::mousePressEvent(QMouseEvent* const event,qglviewer::Camera* const camera)
{
	if( Global_SampleSet->size())
	{
		setGrabsMouse(true);
	}else
	{
		setGrabsMouse(false);
	}

	ManipulatedFrame::mousePressEvent(event , camera);
//	event->ignore();
}
void Manipulator::mouseReleaseEvent(QMouseEvent* const event,qglviewer::Camera* const camera)
{
	setGrabsMouse(false);
	ManipulatedFrame::mouseReleaseEvent(event , camera);
	event->ignore();
}

void Manipulator::mouseMoveEvent(QMouseEvent* const event, qglviewer::Camera* const camera)
{
	
	if( Global_SampleSet->size())
	{
		m_sample =   &(*Global_SampleSet)[0];
		setGrabsMouse(true);
	}else
	{
		m_sample = NULL;
	}
	ManipulatedFrame::mouseMoveEvent(event , camera);
//	event->ignore();
	if(!m_sample)
		return;
	qreal t_x,t_y,t_z;
	qreal q0,q1,q2,q3;
	getTranslation( t_x  ,t_y ,t_z);
	getRotation(q0, q1, q2, q3);
	cout<< " tx ty tz "<<t_x <<t_y<<t_z<<endl;
	cout<< " q0 q1 ,q2 ,q3 "<< q0 <<q1 <<q2<<q3<<endl;
	m_sample->getFrame().setTranslation( t_x ,t_y ,t_z);
	m_sample->getFrame().setRotation( q0 ,q1 ,q2 ,q3);
//	event->accept();
}

void Manipulator::checkIfGrabsMouse(int x, int y, const qglviewer::Camera* const camera)
{
	using namespace qglviewer;
	setGrabsMouse( Global_SampleSet->size());
	//const int thresold = 10;
	//const Vec proj = camera->projectedCoordinatesOf(position());
	//setGrabsMouse(keepsGrabbingMouse_ || ((fabs(x-proj.x) < thresold) && (fabs(y-proj.y) < thresold)));
}
