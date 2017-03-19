#include <qgl.h>
#include "manipulate_object.h"

using namespace qglviewer;

void ManipulatedObject::draw() const
{
	glPushMatrix();
	glMultMatrixd(frame.matrix());
	glBegin(GL_POINTS);
	qglviewer::Vec position = this->position;
	glNormal3f(normal.x, normal.y, normal.z);
	glVertex3f((GLfloat)position.x, (GLfloat)position.y, (GLfloat)position.z);
	glEnd();
	glPopMatrix();
}
