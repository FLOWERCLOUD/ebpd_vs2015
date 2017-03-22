#include "paint_canvas.h"
#include "manipulate_tool.h"
#include "manipulatedFrameSetConstraint.h"
#include "GlobalObject.h"
#include "globals.h"
#include "sample_set.h"
#include "sample.h"
#include "vertex.h"
#include <QMenu>
#include <QAction>


using namespace qglviewer;

const int SLICE = 60;
float ellipse[3 * (SLICE+1)];
#include "toolbox/gl_utils/glsave.hpp"
static void draw_arrow(float radius, float length) {
	if (0)
	{
		GLUquadricObj* quad = gluNewQuadric();

		glLineWidth(1.f);
		//gluCylinder(quad, radius/3.f, radius/3.f, 0.8f * length, 10, 10);
		glBegin(GL_LINES);
		glVertex3f(0.f, 0.f, 0.f);
		glVertex3f(0.f, 0.f, 0.8f*length);
		glAssert(glEnd());
		glTranslatef(0.f, 0.f, 0.8f * length);
		gluCylinder(quad, radius, 0.0f, 0.2f * length, 10, 10);

		gluDeleteQuadric(quad);
	}
	else
	{//gizmo2
		GLUquadricObj* quad = gluNewQuadric();
		glPushMatrix();
		glLineWidth(2.f);
		glTranslatef(0.f, 0.f, 0.3f * length);
		glBegin(GL_LINES);
		glVertex3f(0.f, 0.f, 0.f);
		glVertex3f(0.f, 0.f, 0.8f*length);
		glAssert(glEnd());

		glTranslatef(0.f, 0.f, 0.8f * length);
		gluCylinder(quad, radius, 0.0f, 0.2f * length, 10, 10);
		glPopMatrix();

		gluDeleteQuadric(quad);
	}

}
static void drawEllipse(qreal r)
{
	static int isInitailized = -1;
	if (isInitailized == -1)
	{
		for (int i = 0; i <= SLICE; i += 3) {
			ellipse[i + 0] = 0 + (float)cos(i*360.0f/SLICE*PI / 180.0f); // X
			ellipse[i + 1] = 0 + (float)sin(i*360.0f/ SLICE*PI / 180.0f); // Y
			ellipse[i + 2] = 0;                         // Z
		}
		isInitailized = 1;
	}
	glBegin(GL_LINE_LOOP);
	// This should generate a circle
	for (int i = 0; i <= SLICE; i += 3)
	{
		float x = ellipse[i + 0] * r; // keep the axes radius same
		float y = ellipse[i + 1] * r;
		float z = ellipse[i + 2] * r;
		glVertex3f(x, y, z);
	}

	glEnd();
}

//static void drawRotateAxis(qreal length, int select = -1)
//{
//	const qreal charWidth = length / 40.0;
//	const qreal charHeight = length / 30.0;
//	const qreal charShift = 1.04 * length;
//
//	GLboolean lighting, colorMaterial;
//	glGetBooleanv(GL_LIGHTING, &lighting);
//	glGetBooleanv(GL_COLOR_MATERIAL, &colorMaterial);
//
//	glDisable(GL_LIGHTING);
//
//	glBegin(GL_LINES);
//	// The X
//	glVertex3d(charShift, charWidth, -charHeight);
//	glVertex3d(charShift, -charWidth, charHeight);
//	glVertex3d(charShift, -charWidth, -charHeight);
//	glVertex3d(charShift, charWidth, charHeight);
//	// The Y
//	glVertex3d(charWidth, charShift, charHeight);
//	glVertex3d(0.0, charShift, 0.0);
//	glVertex3d(-charWidth, charShift, charHeight);
//	glVertex3d(0.0, charShift, 0.0);
//	glVertex3d(0.0, charShift, 0.0);
//	glVertex3d(0.0, charShift, -charHeight);
//	// The Z
//	glVertex3d(-charWidth, charHeight, charShift);
//	glVertex3d(charWidth, charHeight, charShift);
//	glVertex3d(charWidth, charHeight, charShift);
//	glVertex3d(-charWidth, -charHeight, charShift);
//	glVertex3d(-charWidth, -charHeight, charShift);
//	glVertex3d(charWidth, -charHeight, charShift);
//	glEnd();
//
//	glEnable(GL_LIGHTING);
//	glDisable(GL_COLOR_MATERIAL);
//
//	float color[4];
//
//	if (2 == select) //z 
//	{
//		color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//	}
//	else
//	{
//		color[0] = 0.7f;  color[1] = 0.7f;  color[2] = 1.0f;  color[3] = 1.0f;
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//	}
//	drawEllipse(length);
//
//
//	if (0 == select) //x 
//	{
//		color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//	}
//	else
//	{
//		color[0] = 1.0f;  color[1] = 0.7f;  color[2] = 0.7f;  color[3] = 1.0f;
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//	}
//	glPushMatrix();
//	glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
//	drawEllipse(length);
//	glPopMatrix();
//
//	if (1 == select) //y 
//	{
//		color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//	}
//	else
//	{
//		color[0] = 0.7f;  color[1] = 1.0f;  color[2] = 0.7f;  color[3] = 1.0f;
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//	}
//	glPushMatrix();
//	glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
//	drawEllipse(length);
//	glPopMatrix();
//
//	if (colorMaterial)
//		glEnable(GL_COLOR_MATERIAL);
//	if (!lighting)
//		glDisable(GL_LIGHTING);
//}

void drawRotateAxisWithNames(qreal length, int select = -1, bool isWithName = false)
{
	const qreal charWidth = length / 40.0;
	const qreal charHeight = length / 30.0;
	const qreal charShift = 1.04 * length;
	glLineWidth(1.0f);
	//GLboolean lighting, colorMaterial;
	//glGetBooleanv(GL_LIGHTING, &lighting);
	//glGetBooleanv(GL_COLOR_MATERIAL, &colorMaterial);

	//glDisable(GL_LIGHTING);

	//glBegin(GL_LINES);
	//// The X
	//glVertex3d(charShift, charWidth, -charHeight);
	//glVertex3d(charShift, -charWidth, charHeight);
	//glVertex3d(charShift, -charWidth, -charHeight);
	//glVertex3d(charShift, charWidth, charHeight);
	//// The Y
	//glVertex3d(charWidth, charShift, charHeight);
	//glVertex3d(0.0, charShift, 0.0);
	//glVertex3d(-charWidth, charShift, charHeight);
	//glVertex3d(0.0, charShift, 0.0);
	//glVertex3d(0.0, charShift, 0.0);
	//glVertex3d(0.0, charShift, -charHeight);
	//// The Z
	//glVertex3d(-charWidth, charHeight, charShift);
	//glVertex3d(charWidth, charHeight, charShift);
	//glVertex3d(charWidth, charHeight, charShift);
	//glVertex3d(-charWidth, -charHeight, charShift);
	//glVertex3d(-charWidth, -charHeight, charShift);
	//glVertex3d(charWidth, -charHeight, charShift);
	//glEnd();

	//glEnable(GL_LIGHTING);
	//glDisable(GL_COLOR_MATERIAL);

	double color[4];

	if (5 == select) //z 
	{
		glLineWidth(5.0f);
		color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
		glColor4dv(color);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	}
	else
	{
		glLineWidth(1.0f);
		color[0] = 0.0f;  color[1] = 0.0f;  color[2] = 1.0f;  color[3] = 1.0f;
		glColor4dv(color);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	}
	if (isWithName)glPushName(5);
	if (isWithName)
	{
		glLineWidth(10.0f);
		drawEllipse(length);
	}		
	else
		drawEllipse(length);
	if (isWithName)glPopName();


	if (3 == select) //x 
	{
		glLineWidth(5.0f);
		color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
		glColor4dv(color);
		//glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	}
	else
	{
		glLineWidth(1.0f);
		color[0] = 1.0f;  color[1] = 0.0f;  color[2] = 0.0f;  color[3] = 1.0f;
		glColor4dv(color);
		//glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	}
	glPushMatrix();
	glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
	if (isWithName)glPushName(3);
	if (isWithName)
	{
		glLineWidth(10.0f);
		drawEllipse(length);
	}
	else
		drawEllipse(length);
	if (isWithName)glPopName();
	glPopMatrix();

	if (4 == select) //y 
	{
		glLineWidth(5.0f);
		color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
		glColor4dv(color);
		//glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	}
	else
	{
		glLineWidth(1.0f);
		color[0] = 0.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
		glColor4dv(color);
		//glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	}
	glPushMatrix();
	glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
	if (isWithName)glPushName(4);
	if (isWithName)
	{
		glLineWidth(10.0f);
		drawEllipse(length);
	}
	else
		drawEllipse(length);
	if (isWithName)glPopName();
	glPopMatrix();

	//if (colorMaterial)
	//	glEnable(GL_COLOR_MATERIAL);
	//if (!lighting)
	//	glDisable(GL_LIGHTING);
}

void screen_quas(qglviewer::Vec position ,QGLViewer* _canvas ,int select = -1 ,bool withname = false, QPoint point = QPoint())
{
	int size = 10;
	Vec proj;
	//draw in screen coordinate 

//	_canvas->startScreenCoordinatesSystem();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
if(withname)  //should add this to clip 
{
	static GLint viewport[4];
	_canvas->camera()->getViewport(viewport);
	//glupickmatrix x,y should be  window coordinates.
	gluPickMatrix(point.x(), point.y(), _canvas->selectRegionWidth(), _canvas->selectRegionHeight(), viewport);
}
	glOrtho(0, _canvas->width(), _canvas->height(), 0, 0.0, -1.0);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();



	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	if (9 == select)
		glColor4f(1.0f, 1.0f, 0.0f, 0.2f);
	else
		glColor4f(0.0f, 1.0f, 1.0f, 0.2f);
	if (withname)
		glPushName(9);

	proj = _canvas->camera()->projectedCoordinatesOf(position);
	glBegin(GL_QUADS);
	glVertex3fv(proj + Vec(-size, -size, -0.001f));
	glVertex3fv(proj + Vec(size, -size, -0.001f));
	glVertex3fv(proj + Vec(size, size, -0.001f));
	glVertex3fv(proj + Vec(-size, size, -0.001f));

	glEnd();

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glLineWidth(1.0f);
	if( 9 == select)
		glColor4f(1.0f, 1.0f, 0.0f, 1.f);
	else
		glColor4f(0.0f, 1.0f, 1.0f, 1.f);

	glBegin(GL_LINES);
	// The small z offset makes the arrow slightly above the saucer, so that it is always visible
	glVertex3fv(proj + Vec(-size, -size, -0.001f));
	glVertex3fv(proj + Vec(size, -size, -0.001f));
	glVertex3fv(proj + Vec(size, -size, -0.001f));
	glVertex3fv(proj + Vec(size, size, -0.001f));
	glVertex3fv(proj + Vec(size, size, -0.001f));
	glVertex3fv(proj + Vec(-size, size, -0.001f));
	glVertex3fv(proj + Vec(-size, size, -0.001f));
	glVertex3fv(proj + Vec(-size, -size, -0.001f));
	glEnd();
	if (withname)
		glPopName();
	//the pop sequence is confuse
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();





//	_canvas->stopScreenCoordinatesSystem();


}




//static void drawAxis(qreal length, int select = -1)
//{
//	const qreal charWidth = length / 40.0;
//	const qreal charHeight = length / 30.0;
//	const qreal charShift = 1.04 * length;
//
//	GLboolean lighting, colorMaterial;
//	glGetBooleanv(GL_LIGHTING, &lighting);
//	glGetBooleanv(GL_COLOR_MATERIAL, &colorMaterial);
//
//	glDisable(GL_LIGHTING);
//
//	glBegin(GL_LINES);
//	// The X
//	glVertex3d(charShift, charWidth, -charHeight);
//	glVertex3d(charShift, -charWidth, charHeight);
//	glVertex3d(charShift, -charWidth, -charHeight);
//	glVertex3d(charShift, charWidth, charHeight);
//	// The Y
//	glVertex3d(charWidth, charShift, charHeight);
//	glVertex3d(0.0, charShift, 0.0);
//	glVertex3d(-charWidth, charShift, charHeight);
//	glVertex3d(0.0, charShift, 0.0);
//	glVertex3d(0.0, charShift, 0.0);
//	glVertex3d(0.0, charShift, -charHeight);
//	// The Z
//	glVertex3d(-charWidth, charHeight, charShift);
//	glVertex3d(charWidth, charHeight, charShift);
//	glVertex3d(charWidth, charHeight, charShift);
//	glVertex3d(-charWidth, -charHeight, charShift);
//	glVertex3d(-charWidth, -charHeight, charShift);
//	glVertex3d(charWidth, -charHeight, charShift);
//	glEnd();
//
//	glEnable(GL_LIGHTING);
//	glDisable(GL_COLOR_MATERIAL);
//
//	float color[4];
//
//	if (2 == select) //z 
//	{
//		color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//	}
//	else
//	{
//		color[0] = 0.7f;  color[1] = 0.7f;  color[2] = 1.0f;  color[3] = 1.0f;
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//	}
//	QGLViewer::drawArrow(length, 0.01*length);
//
//
//	if (0 == select) //x 
//	{
//		color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//	}
//	else
//	{
//		color[0] = 1.0f;  color[1] = 0.7f;  color[2] = 0.7f;  color[3] = 1.0f;
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//	}
//	glPushMatrix();
//	glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
//	QGLViewer::drawArrow(length, 0.01*length);
//	glPopMatrix();
//
//	if (1 == select) //y 
//	{
//		color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//	}
//	else
//	{
//		color[0] = 0.7f;  color[1] = 1.0f;  color[2] = 0.7f;  color[3] = 1.0f;
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//	}
//	glPushMatrix();
//	glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
//	QGLViewer::drawArrow(length, 0.01*length);
//	glPopMatrix();
//
//	if (colorMaterial)
//		glEnable(GL_COLOR_MATERIAL);
//	if (!lighting)
//		glDisable(GL_LIGHTING);
//}
void drawAxisWithNames(qreal length, int select , bool isWithName)
{
	const qreal charWidth = length / 40.0;
	const qreal charHeight = length / 30.0;
	const qreal charShift = 1.04 * length;

	//GLboolean lighting, colorMaterial;
	//glGetBooleanv(GL_LIGHTING, &lighting);
	//glGetBooleanv(GL_COLOR_MATERIAL, &colorMaterial);

	//glDisable(GL_LIGHTING);

	//glEnable(GL_LIGHTING);
	//glDisable(GL_COLOR_MATERIAL);

	double color[4];

	if (2 == select) //z 
	{
		color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
		glColor4dv(color);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	}
	else
	{
		color[0] = 0.0f;  color[1] = 0.0f;  color[2] = 1.0f;  color[3] = 1.0f;
		glColor4dv(color);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	}
	if(isWithName)glPushName(2);
	draw_arrow(0.05*length, length);//QGLViewer::drawArrow(length, 0.01*length);
	if (isWithName)glPopName();


	if (0 == select) //x 
	{
		color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
		glColor4dv(color);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	}
	else
	{
		color[0] = 1.0f;  color[1] = 0.0f;  color[2] = 0.0f;  color[3] = 1.0f;
		glColor4dv(color);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	}
	glPushMatrix();
	glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
	if (isWithName)glPushName(0);
	draw_arrow(0.05*length, length); //QGLViewer::drawArrow(length, 0.01*length);
	if (isWithName)glPopName();
	glPopMatrix();

	if (1 == select) //y 
	{
		color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
		glColor4dv(color);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	}
	else
	{
		color[0] = 0.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
		glColor4dv(color);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
	}
	glPushMatrix();
	glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
	if (isWithName)glPushName(1);
	draw_arrow(0.05*length, length); //QGLViewer::drawArrow(length, 0.01*length);
	if (isWithName)glPopName();
	glPopMatrix();

	//if (colorMaterial)
	//	glEnable(GL_COLOR_MATERIAL);
	//if (!lighting)
	//	glDisable(GL_LIGHTING);
}

static void draw_quad(float l)
{
	float currentColor[4];
	glGetFloatv(GL_CURRENT_COLOR, currentColor);
	glPushMatrix();
	glTranslatef(l*0.5, l*0.5, 0.0f);
	l *= 0.2f;

	glColor4f(currentColor[0], currentColor[1], currentColor[2], 1.0f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glLineWidth(1.0f);
	//	glColor4f(0.3f, 0.3f, 0.3f, 1.f);
	glBegin(GL_LINES);
	glVertex3f(l, 0.f, 0.f);
	glVertex3f(l, l, 0.f);
	glVertex3f(l, l, 0.f);
	glVertex3f(0.f, l, 0.f);
	glVertex3f(0.f, l, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(l, 0.f, 0.f);
	glEnd();

	glColor4f(currentColor[0], currentColor[1], currentColor[2], 0.5f);
	glBegin(GL_QUADS);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(l, 0.f, 0.f);
	glVertex3f(l, l, 0.f);
	glVertex3f(0.f, l, 0.f);
	glEnd();



	glPopMatrix();
}
void draw_quadsWithNames(qreal length, int select = -1, bool isWithName = false)
{
	//GLboolean lighting, colorMaterial;
	//glGetBooleanv(GL_LIGHTING, &lighting);
	//glGetBooleanv(GL_COLOR_MATERIAL, &colorMaterial);

	//glDisable(GL_LIGHTING);

	//glEnable(GL_LIGHTING);
	//glDisable(GL_COLOR_MATERIAL);

	double color[4];
	
	int XY = 8, XZ = 7, YZ = 6;
	glPushMatrix();
	{
		/* PLANE XY */
		if (select == XY)
		{
			color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
			glColor4dv(color);
//			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
		}
		else
		{
			color[0] = 0.0f;  color[1] = 0.0f;  color[2] = 1.0f;  color[3] = 1.0f;
			glColor4dv(color);
//			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
		}


		if (isWithName)glPushName(XY);
		draw_quad(length);
		if (isWithName)glPopName();
		/* PLANE XZ */
		if (select == XZ)
		{
			color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
			glColor4dv(color);
//			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
		}			
		else
		{
			color[0] = 0.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
			glColor4dv(color);
//			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
		}
			

		glRotatef(90.f, 1.f, 0.f, 0.f);
		if (isWithName)glPushName(XZ);
		draw_quad(length);
		if (isWithName)glPopName();
		/* PLANE YZ*/
		if (select == YZ)
		{
			color[0] = 1.0f;  color[1] = 1.0f;  color[2] = 0.0f;  color[3] = 1.0f;
			glColor4dv(color);
//			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
		}
		else
		{
			color[0] = 1.0f;  color[1] = 0.0f;  color[2] = 0.0f;  color[3] = 1.0f;
			glColor4dv(color);
//			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
		}


		glRotatef(90.f, 0.f, 1.f, 0.f);
		if (isWithName)glPushName(YZ);
		draw_quad(length);
		if (isWithName)glPopName();
	}
	glPopMatrix();
	//if (colorMaterial)
	//	glEnable(GL_COLOR_MATERIAL);
	//if (!lighting)
	//	glDisable(GL_LIGHTING);
}


ManipulateTool::ManipulateTool(PaintCanvas* canvas, 
	ManipulatedObject::ManipulatedObjectType _manipulateObjectType) :
	select_tool(canvas, _manipulateObjectType)
{

	axis_length = 0.5f;
	rotate_circle_radius = 0.5f;
	selected_name = NONE_OBJECT;
	activeConstraint = Local_Constraint;
//	cur_constraint = new LocalConstraint();
	//activeConstraint = World_Constraint;
	cur_constraint = new ManipulatedFrameSetLocalConstraint();
	manipulatedFrame()->setConstraint(cur_constraint);
	//manipulatedFrame()->setConstraint(new ManipulatedFrameSetConstraint());
	connect(&select_tool, SIGNAL(postSelect()), this, SLOT(postSelect()));
}

void ManipulateTool::move(QMouseEvent *e , qglviewer::Camera* camera)
{
	if (select_tool.left_mouse_button_ == true)
	{
		select_tool.move(e, camera);
	}
	this->mouseMoveEvent(e, camera);
	//for (std::vector<int>::const_iterator it = m_selected_idx.begin(), end = m_selected_idx.end(); it != end; ++it)
	//{
	//	qglviewer::Frame& cur_frame =  m_input_objs[*it]->getFrame();
	//	Mani
	//	cur_frame.set = *(qglviewer::Frame*) this;
	//}
}

void ManipulateTool::drag(QMouseEvent *e, qglviewer::Camera* camera)
{
	select_tool.drag(e, camera);


	this->mouseMoveEvent(e, camera);


}

void ManipulateTool::release(QMouseEvent *e , qglviewer::Camera* camera)
{

	select_tool.release(e, camera);

	if (e->button() == Qt::RightButton)
	{
		qglviewer::Vec release_pos(e->pos().x(), e->pos().y(), 0.0f);
		if (qglviewer::Vec(release_pos - select_tool.right_mouse_pressed_pos_).norm() < 0.001)
		{
		

			QAction* action_1 = new QAction("&Local Constraint", select_tool.popupMenu);
			QAction* action_2 = new QAction("&World Constraint", select_tool.popupMenu);
			QAction* action_3 = new QAction("&Camera Constraint", select_tool.popupMenu);
			action_1->setCheckable(true);
			action_2->setCheckable(true);
			action_3->setCheckable(true);
			switch (activeConstraint)
			{
			case  Local_Constraint:
			{
				action_1->setChecked(true);
			}
			break;
			case  World_Constraint:
			{
				action_2->setChecked(true);
			}
			break;
			case  Camera_Constraint:
			{
				action_3->setChecked(true);
			}
			break;

			}
			select_tool.popupMenu->addAction(action_1);
			select_tool.popupMenu->addAction(action_2);
			select_tool.popupMenu->addAction(action_3);

			QObject*b = (QObject*)this;
			connect(action_1, SIGNAL(triggered()), b, SLOT(slot_action_LocalConstraint()));
			connect(action_2, SIGNAL(triggered()), b, SLOT(slot_action_WorldConstraint()));
			connect(action_3, SIGNAL(triggered()), b, SLOT(slot_action_CameraConstraint()));

			QPoint& point = e->globalPos();
			point.setX(point.x() - 50);
			select_tool.popupMenu->exec(point);
		}
		delete select_tool.popupMenu;
		select_tool.popupMenu = NULL;
		select_tool.canvas_->updateGL();
		select_tool.right_mouse_button_ = false;

	}
	this->mouseReleaseEvent(e, camera);
	selected_name = NONE_OBJECT;
	setTranslationConstraintType(qglviewer::AxisPlaneConstraint::FREE);
//	setTranslationConstraintDirection(name);

}

void ManipulateTool::press(QMouseEvent* e, qglviewer::Camera* camera)
{

	
	QPoint targetpoint((float)e->x(), (float)e->y());
	select(targetpoint);

	switch (selected_name)
	{
	case X_AXIS:
		this->startAction(QGLViewer::TRANSLATE, true);
		break;
	case Y_AXIS:
		this->startAction(QGLViewer::TRANSLATE, true);
		break;
	case Z_AXIS:
		this->startAction(QGLViewer::TRANSLATE, true);
		break;
	case X_ROTATE:
		this->startAction(QGLViewer::ROTATE, true);
		break;
	case Y_ROTATE:
		this->startAction(QGLViewer::ROTATE, true);
		break;
	case Z_ROTATE:
		this->startAction(QGLViewer::ROTATE, true);
		break;
	case YZ_PLANE:
		this->startAction(QGLViewer::TRANSLATE, true);
		break;
	case XZ_PLAINE:
		this->startAction(QGLViewer::TRANSLATE, true);
		break;
	case XY_PLAIN:
		this->startAction(QGLViewer::TRANSLATE, true);
		break;
	case SCREEN_PLAIN:
		this->startAction(QGLViewer::TRANSLATE, true);
		break;
	}
	this->mousePressEvent(e, camera);
	if(NONE_OBJECT == selected_name)
		select_tool.press(e, camera);


	//GLdouble		model[16];
	//glGetDoublev(GL_MODELVIEW_MATRIX, model);

	//GLdouble proj[16];
	//glGetDoublev(GL_PROJECTION_MATRIX, proj);

	//GLint view[4];
	//glGetIntegerv(GL_VIEWPORT, view);

	//int winX = e->x();
	//int winY = view[3] - 1 - e->y();

	//float zValue;
	//glReadPixels(winX, winY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &zValue);

	//GLubyte stencilValue;
	//glReadPixels(winX, winY, 1, 1, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, &stencilValue);

	////GLdouble worldX, worldY, worldZ;
	////gluUnProject(winX, winY, zValue, model, proj, view, &worldX, &worldY, &worldZ);

	
	//bool isFound;
	//qglviewer::Vec target3dVec = camera->pointUnderPixel(targetpoint, isFound);







	//if (e->modifiers() == Qt::ControlModifier)
	//{

	//}



}

void ManipulateTool::keyPressEvent(QKeyEvent *e)
{
	switch (e->key())
	{

	case Qt::Key_Space:
	{
 		if (Local_Constraint == activeConstraint)
		{
			changeToConstraint(World_Constraint);
		}
		else if (World_Constraint == activeConstraint)
		{
			changeToConstraint(Camera_Constraint);
		}
		else if (Camera_Constraint == activeConstraint)
		{
			changeToConstraint(Local_Constraint);
		}
		break;
	}

	}


}
void ManipulateTool::draw()
{
	select_tool.draw();
//	glEnable(GL_DEPTH_TEST);  // 必须开启深度测试才能使用 unproject 方式选点
//	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
////	drawAxis(3,-1);
//	// Draws all the objects. Selected ones are not repainted because of GL depth test.
//	glColor3f(0.8f, 0.8f, 0.8f);
//	glPointSize(5.0f);
//	for (int i = 0; i<int( m_input_objs_.size()); i++)
//		m_input_objs_[i]->draw();
//
//	// Draws selected objects only.
//	glPointSize(20.0f);
//	glColor3f(0.9f, 0.0f, 0.0f);
	for (auto it = select_tool.m_selected_objs_.begin(), end = select_tool.m_selected_objs_.end(); it != end; ++it)
		(*it)->draw();
//	glDisable(GL_DEPTH_TEST);
	// Draws manipulatedFrame (the set's rotation center)
	//if (manipulatedFrame()->isManipulated())
	//{

	//}
	// Draws rectangular selection area. Could be done in postDraw() instead.
	//if (selectionMode_ != NONE)
	//	drawSelectionRectangle();

	//这里是为了让控制器大小不随照相机改变而改变
	camera_dist = select_tool.canvas_->camera()->type() == Camera::PERSPECTIVE ? (manipulatedFrame()->position() - select_tool.canvas_->camera()->position()).norm() : 1;  //注意是norm 不是square norm
	camera_dist *= 0.3;

	//Logger << "zNearCoefficient() " << canvas_->camera()->zNearCoefficient() << std::endl;
	//Logger << "zClippingCoefficient " << canvas_->camera()->zClippingCoefficient() << std::endl;
	//Logger << "zFar() " << canvas_->camera()->zFar() << std::endl;
	//Logger << "zNear() " << canvas_->camera()->zNear() << std::endl;
	//Logger << "canvas_->camera()->position() " << canvas_->camera()->position().x <<" "<< canvas_->camera()->position().y<< " " << canvas_->camera()->position().z << std::endl;
	//Logger << "manipulatedFrame()->position() " << manipulatedFrame()->position().x<< " " << manipulatedFrame()->position().y<< " " << manipulatedFrame()->position().z << std::endl;
	//Logger <<"camera to frame "<< dist << std::endl;

	//GLdouble mv_matraix[16];
	//canvas_->camera()->getModelViewMatrix(mv_matraix);
	//GLdouble tansposeOFTranslate[16] = { 0.0f };
	//tansposeOFTranslate[0] = tansposeOFTranslate[5] = tansposeOFTranslate[10] = tansposeOFTranslate[15]  =1.0f;
	//tansposeOFTranslate[12] = -mv_matraix[12]-0.5 ;
	//tansposeOFTranslate[13] = -mv_matraix[13]-0.5 ;
	//tansposeOFTranslate[14] = -mv_matraix[14]-0.5 ;

	Tbx::GLEnabledSave save_light(GL_LIGHTING, true, false);
	Tbx::GLEnabledSave save_depth(GL_DEPTH_TEST, true, false);
	Tbx::GLEnabledSave save_blend(GL_BLEND, true, true);
	Tbx::GLBlendSave save_blend_eq;
	glAssert(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
	Tbx::GLEnabledSave save_alpha(GL_ALPHA_TEST, true, true);
	Tbx::GLEnabledSave save_textu(GL_TEXTURE_2D, true, false);
	Tbx::GLLineWidthSave save_line;




	screen_quas(manipulatedFrame()->position(), select_tool.canvas_, selected_name);

	glPushMatrix();

//	glMultMatrixd( tansposeOFTranslate);
	switch (activeConstraint)
	{
	case Local_Constraint:
		glPushMatrix();
		glMultMatrixd(manipulatedFrame()->matrix());
		glScalef(camera_dist, camera_dist, camera_dist);
		drawAxisWithNames(axis_length, selected_name);
		drawRotateAxisWithNames(rotate_circle_radius, selected_name);
		draw_quadsWithNames(axis_length, selected_name);
		glPopMatrix();
		break;
	case World_Constraint:
	{
		glPushMatrix();
		GLdouble* frame_matrix = (GLdouble*)manipulatedFrame()->matrix();
		glTranslatef(frame_matrix[12], frame_matrix[13], frame_matrix[14]);
		glScalef(camera_dist, camera_dist, camera_dist);
		drawAxisWithNames(axis_length, selected_name);
		drawRotateAxisWithNames(rotate_circle_radius, selected_name);
		draw_quadsWithNames(axis_length, selected_name);
		glPopMatrix();
		break;
	}
	case Camera_Constraint:
		glPushMatrix();
		glMultMatrixd(manipulatedFrame()->matrix());
		glScalef(camera_dist, camera_dist, camera_dist);
		drawAxisWithNames(axis_length, selected_name);
		drawRotateAxisWithNames(rotate_circle_radius, selected_name);
		draw_quadsWithNames(axis_length, selected_name);
		glPopMatrix();
		break;
	default:
		glPushMatrix();
		glMultMatrixd(manipulatedFrame()->matrix());
		glScalef(camera_dist, camera_dist, camera_dist);
		drawAxisWithNames(axis_length, selected_name);
		drawRotateAxisWithNames(rotate_circle_radius, selected_name);
		glPopMatrix();
	}
	glPopMatrix();



}

void ManipulateTool::postSelect()
{
	if (selected_name == NONE_OBJECT)
	{
		startManipulation();

	}
}


void ManipulateTool::postManipulateToolSelection()
{
	int name = select_tool.canvas_->selectedName();
	selected_name = (EnumConstraintObj)name;
	Logger << "select " << name << std::endl;
	switch (selected_name)
	{
	case X_AXIS:
		setTranslationConstraintType(qglviewer::AxisPlaneConstraint::AXIS);
		setTranslationConstraintDirection(name);
		break;
	case Y_AXIS:
		setTranslationConstraintType(qglviewer::AxisPlaneConstraint::AXIS);
		setTranslationConstraintDirection(name);
		break;
	case Z_AXIS:
		setTranslationConstraintType(qglviewer::AxisPlaneConstraint::AXIS);
		setTranslationConstraintDirection(name);
		break;

	case X_ROTATE:
		setRotationConstraintType(qglviewer::AxisPlaneConstraint::AXIS);
		setRotationConstraintDirection(name -3);
		break;
	case Y_ROTATE:
		setRotationConstraintType(qglviewer::AxisPlaneConstraint::AXIS);
		setRotationConstraintDirection(name-3);
		break;
	case Z_ROTATE:
		setRotationConstraintType(qglviewer::AxisPlaneConstraint::AXIS);
		setRotationConstraintDirection(name-3);
		break;

	case YZ_PLANE:
		setTranslationConstraintType(qglviewer::AxisPlaneConstraint::PLANE);
		setTranslationConstraintDirection(name-6);
		break;
	case XZ_PLAINE:
		setTranslationConstraintType(qglviewer::AxisPlaneConstraint::PLANE);
		setTranslationConstraintDirection(name-6);
		break;
	case XY_PLAIN:
		setTranslationConstraintType(qglviewer::AxisPlaneConstraint::PLANE);
		setTranslationConstraintDirection(name-6);
		break;
	case SCREEN_PLAIN:
//		changeConstraint(Camera_Constraint);
		setTranslationConstraintType(qglviewer::AxisPlaneConstraint::FREE);
//		setTranslationConstraintDirection(name - 6);
		break;
	default:
		Logger << "select " << name << std::endl;
		break;
	}
}



void ManipulateTool::startManipulation()
{

	qglviewer::Vec averagePosition;
	switch (activeConstraint)
	{
	case Local_Constraint:
	{
		ManipulatedFrameSetLocalConstraint* mfsc = (ManipulatedFrameSetLocalConstraint*)(manipulatedFrame()->constraint());
		mfsc->clearSet();
		for (ManipulatedObject* obj : select_tool.m_selected_objs_)
		{
			mfsc->addObjectToSet(obj);
			averagePosition += obj->getWorldPosition();
		}
		break;
	}
	case World_Constraint:
	{
		ManipulatedFrameSetWorldConstraint* mfsc = (ManipulatedFrameSetWorldConstraint*)(manipulatedFrame()->constraint());
		mfsc->clearSet();
		for ( ManipulatedObject* obj: select_tool.m_selected_objs_)
		{
			mfsc->addObjectToSet(obj);
			averagePosition += obj->getWorldPosition();
		}
		break;
	}
	case Camera_Constraint:
	{
		ManipulatedFrameSetCameraConstraint* mfsc = (ManipulatedFrameSetCameraConstraint*)(manipulatedFrame()->constraint());
		mfsc->clearSet();
		for (ManipulatedObject* obj : select_tool.m_selected_objs_)
		{
			mfsc->addObjectToSet(obj);
			averagePosition += obj->getWorldPosition();
		}
		break;
	}

	}
//	ManipulatedFrameSetConstraint* mfsc = (ManipulatedFrameSetConstraint*)(manipulatedFrame()->constraint());
	if (select_tool.m_selected_objs_.size() > 0)
		manipulatedFrame()->setPosition(averagePosition / select_tool.m_selected_objs_.size());
}

void ManipulateTool::changeToConstraint(EnumConstraint curstraint)
{
	EnumConstraint previous = activeConstraint;
	AxisPlaneConstraint* previousConstraint = cur_constraint;
	std::vector<ManipulatedObject*> manipulateObjs;
	if (Local_Constraint == previous)
	{
		manipulateObjs = ((ManipulatedFrameSetLocalConstraint*)previousConstraint)->getManipulateObjs();
	}
	else if (World_Constraint == previous)
	{
		manipulateObjs = ((ManipulatedFrameSetWorldConstraint*)previousConstraint)->getManipulateObjs();
	}
	else if (Camera_Constraint == previous)
	{
		manipulateObjs = ((ManipulatedFrameSetCameraConstraint*)previousConstraint)->getManipulateObjs();
	}
	switch(curstraint)
	{
	case Local_Constraint:
		cur_constraint = new ManipulatedFrameSetLocalConstraint();
		((ManipulatedFrameSetLocalConstraint*)cur_constraint)->setManipulateObjs(manipulateObjs);
		activeConstraint = Local_Constraint;
		break;
	case World_Constraint:
		cur_constraint = new ManipulatedFrameSetWorldConstraint();
		((ManipulatedFrameSetWorldConstraint*)cur_constraint)->setManipulateObjs(manipulateObjs);
		activeConstraint = World_Constraint;
		break;
	case Camera_Constraint:
		cur_constraint = new ManipulatedFrameSetCameraConstraint(select_tool.canvas_->camera());
		((ManipulatedFrameSetCameraConstraint*)cur_constraint)->setManipulateObjs(manipulateObjs);
		activeConstraint = Camera_Constraint;
		break;
	default:
		cur_constraint = new ManipulatedFrameSetLocalConstraint();
		((ManipulatedFrameSetLocalConstraint*)cur_constraint)->setManipulateObjs(manipulateObjs);
		activeConstraint = Local_Constraint;
	}
	cur_constraint->setTranslationConstraintType(previousConstraint->translationConstraintType());
	cur_constraint->setTranslationConstraintDirection(previousConstraint->translationConstraintDirection());
	cur_constraint->setRotationConstraintType(previousConstraint->rotationConstraintType());
	cur_constraint->setRotationConstraintDirection(previousConstraint->rotationConstraintDirection());

	if (previousConstraint)
		delete previousConstraint;
	manipulatedFrame()->setConstraint(cur_constraint);
}

void ManipulateTool::setTranslationConstraintType(qglviewer::AxisPlaneConstraint::Type type)
{
	cur_constraint->setTranslationConstraintType(type);
}

void ManipulateTool::setRotationConstraintType(qglviewer::AxisPlaneConstraint::Type type)
{
	cur_constraint->setRotationConstraintType(type);
}

void ManipulateTool::setTranslationConstraintDirection(int transDir)
{
	Vec dir(0.0, 0.0, 0.0);
	dir[transDir] = 1.0;
	cur_constraint->setTranslationConstraintDirection(dir);


}

void ManipulateTool::setRotationConstraintDirection(int rotDir)
{
	Vec dir = Vec(0.0, 0.0, 0.0);
	dir[rotDir] = 1.0;
	cur_constraint->setRotationConstraintDirection(dir);
}

void ManipulateTool::select(const QPoint& point)
{

	select_tool.canvas_->beginSelection(point);
	switch (activeConstraint)
	{
	case Local_Constraint:
		glPushMatrix();
		glMultMatrixd(manipulatedFrame()->matrix());
		glScalef(camera_dist, camera_dist, camera_dist);
		drawAxisWithNames(axis_length, selected_name, true);
		drawRotateAxisWithNames(rotate_circle_radius, selected_name, true);
		draw_quadsWithNames(axis_length, selected_name, true);
		screen_quas(manipulatedFrame()->position(), select_tool.canvas_,-1, true, point);
		glPopMatrix();
		break;
	case World_Constraint:
	{	
		glPushMatrix();
		GLdouble* frame_matrix = (GLdouble*)manipulatedFrame()->matrix();
		//		glMultMatrixd(manipulatedFrame()->matrix());
		glTranslatef(frame_matrix[12], frame_matrix[13], frame_matrix[14]);
		glScalef(camera_dist, camera_dist, camera_dist);
		drawAxisWithNames(axis_length, selected_name, true);
		drawRotateAxisWithNames(rotate_circle_radius, selected_name, true);
		draw_quadsWithNames(axis_length, selected_name, true);
		screen_quas(manipulatedFrame()->position(), select_tool.canvas_, -1, true, point);
		glPopMatrix();
		break;
	}
	case Camera_Constraint:
		glPushMatrix();
		glMultMatrixd(manipulatedFrame()->matrix());
		glScalef(camera_dist, camera_dist, camera_dist);
		drawAxisWithNames(axis_length, selected_name, true);
		drawRotateAxisWithNames(rotate_circle_radius, selected_name, true);
		draw_quadsWithNames(axis_length, selected_name, true);
		screen_quas(manipulatedFrame()->position(), select_tool.canvas_, selected_name, true,point);
		glPopMatrix();
		break;
	default:
		drawAxisWithNames(3*axis_length, -1, true);
	}
	select_tool.canvas_->endSelection(point);
	select_tool.canvas_->postSelection(point);


}
