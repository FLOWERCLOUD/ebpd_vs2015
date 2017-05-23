#include "basic_types.h"
#include "HistoryCommand.h"
#include "Mesh.h"
#include "GLViewWidget.h"
#include "GlobalObject.h"
#include "VideoEditingWindow.h"
#include <QtGui>
#include <QtOpenGL>
#include <math.h>
using namespace std;
namespace videoEditting
{
	GLViewWidget::GLViewWidget(QWidget* parent) :QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
	{
		backgroundClr.setRgb(150, 150, 150);
		setFocusPolicy(Qt::ClickFocus);
		setMouseTracking(true);
		isAltDown = false;
		isCtrlDown = false;
		isShiftDown = false;
		curTool = NULL;
		curToolType = TOOL_SELECT;
		isPickMode = false;
		pickCallBackFun = NULL;
	}

	GLViewWidget::~GLViewWidget(void)
	{
	}

	void GLViewWidget::initializeGL()
	{
		qglClearColor(backgroundClr);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glShadeModel(GL_SMOOTH);

		Global_WideoEditing_Window->scene->init();
		// QMessageBox::information(NULL, QObject::tr("Info"), QObject::tr("All initialized."));
	}

	void GLViewWidget::paintGL()
	{
		makeCurrent();

		if (curToolType == TOOL_PAINT)
			glClearColor(94.0f / 255.0f, 111.0f / 255.0f, 138.0f / 255.0f, 1.0f);
		else if (curToolType == TOOL_FACE_SELECT)
			glClearColor(102.0f / 255.0f, 91.0f / 255.0f, 164.0f / 255.0f, 1.0f);
		else
			glClearColor(0.5, 0.5, 0.5, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// ��������
		Global_WideoEditing_Window->scene->draw();

		// ����������
		if (curTool)
		{
			curTool->draw(Global_WideoEditing_Window->scene->getCamera());
		}

		// ����ѡ��
		if (curToolType == TOOL_FACE_SELECT && isLMBDown)
		{
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();

			glDisable(GL_LIGHTING);
			glDisable(GL_DEPTH_TEST);
			glLineWidth(2.0f);
			glEnable(GL_LINE_STIPPLE);
			glLineStipple(1, 0x0F0F);

			float w = (float)this->width();
			float h = (float)this->height();
			float curPoint[2] = { curPos.x() / w * 2 - 1, 1 - curPos.y() / h * 2 };
			float pressPoint[2] = { pressPos.x() / w * 2 - 1, 1 - pressPos.y() / h * 2 };
			glColor3f(65.0f / 255.0f, 65.0f / 255.0f, 65.0f / 255.0f);
			glBegin(GL_LINE_LOOP);
			glVertex3f(curPoint[0], curPoint[1], 0.5);
			glVertex3f(pressPoint[0], curPoint[1], 0.5);
			glVertex3f(pressPoint[0], pressPoint[1], 0.5);
			glVertex3f(curPoint[0], pressPoint[1], 0.5);
			glEnd();

			glDisable(GL_LINE_STIPPLE);
			glEnable(GL_DEPTH_TEST);
		}
		glFlush();
	}

	void GLViewWidget::resizeGL(int width, int height)
	{
		Global_WideoEditing_Window->scene->resizeCamera(width, height);
		updateGL();
	}

	void GLViewWidget::mousePressEvent(QMouseEvent *event)
	{
		int x = event->x();
		int y = event->y();
		// ���ȴ���ʰȡģʽ
		if (isPickMode && pickCallBackFun)
		{
			pickCallBackFun(Global_WideoEditing_Window->scene->intersectObject(x, y));
			isPickMode = false;
			pickCallBackFun = NULL;
		}

		isLMBDown = event->buttons() & Qt::LeftButton;
		isMMBDown = event->buttons() & Qt::MidButton;
		isRMBDown = event->buttons() & Qt::RightButton;
		lastPos = event->pos();
		pressPos = event->pos();

		QSharedPointer<Scene> scene = Global_WideoEditing_Window->scene;
		if (!isAltDown)
		{
			if (isLMBDown)
			{
				curPos = event->pos();

				QVector3D ori, dir;
				scene->getCamera().getRay(x, y, ori, dir);
				if (curToolType == GLViewWidget::TOOL_SELECT)    // �����ǰΪѡ�񹤾ߣ�ֱ��ѡ������
				{
					curSelectObj = scene->selectObject(x, y);
				}
				else if (curToolType == GLViewWidget::TOOL_PAINT)
				{
//					scene->getPainter().beginPaint(QVector2D(x, y));
				}
				else if (curTool)
				{
					// ���Ϊ�ƶ�/��ת/���Ź��ߣ�
					// ���ȼ���Ƿ����˲����ᣬ
					// ����ǣ���ʼ���ݣ�������ǣ�˵������ѡ������
					QWeakPointer<RenderableObject> oldObject = curSelectObj;
					QWeakPointer<RenderableObject> newObject = scene->intersectObject(x, y);
					bool isSameObj = oldObject == newObject;
					char axis = curTool->intersect(ori, dir);
					bool isHitAxis = axis != -1;
					if (!oldObject && !newObject.isNull())
					{ // ԭ��û��ѡ�����壬����ѡ��һ��������
						curSelectObj = newObject;
						curSelectObj.data()->select();
						curTool->captureObject(QSharedPointer<RenderableObject>(curSelectObj).toWeakRef());
					}
					else if (newObject.isNull())
					{    // û��ѡ��������
						if (isHitAxis)
						{    // ����ѡ���˲�������һ���ᣬ���ǲ��ݵ�ǰ����

							if (curSelectObj)
							{
								curSelectObj.data()->select();
								curTool->selectAxis(axis);
								curTool->beginManipulate(ori, dir, axis);
							}
						}
						else
						{  // ʲô��ûѡ��
							curTool->releaseObject();
							if (!curSelectObj.isNull())
							{
								curSelectObj.data()->deselect();
								curSelectObj.clear();
							}
						}
					}
					else if (isHitAxis && oldObject)
					{    // ѡ��ԭ�������һ������
						 //m_curSelectObj = m_curTool->getCurObject();
						curTool->selectAxis(axis);
						curTool->beginManipulate(ori, dir, axis);
					}
					else if (!isSameObj && !isHitAxis)
					{    // ѡ��һ��������
						if (!curSelectObj.isNull())
							curSelectObj.data()->deselect();
						curSelectObj = newObject;
						curSelectObj.data()->select();
						curTool->captureObject(curSelectObj);
					}
				}
			}
		}
//		Global_WideoEditing_Window->layerEditor->updateList();
//		Global_WideoEditing_Window->actionPaint->setEnabled(!curSelectObj.isNull());
//		Global_WideoEditing_Window->actionSelectFace->setEnabled(!curSelectObj.isNull());
		Global_WideoEditing_Window->transformEditor->updateWidgets();
		updateGL();
	}

	void GLViewWidget::mouseMoveEvent(QMouseEvent *event)
	{
		int x = event->x();
		int y = event->y();
		int dx = event->x() - lastPos.x();
		int dy = event->y() - lastPos.y();
		QSharedPointer<Scene>scene = Global_WideoEditing_Window->scene;
		Camera& camera = scene->getCamera();

		curPos = event->pos();
		if (isAltDown)
		{
			if (event->buttons() & Qt::LeftButton)
			{
				scene->rotateCamera(dx, dy);
			}
			else if (event->buttons() & Qt::RightButton)
			{
				scene->zoomCamera(2 * (dx + dy));
			}
			else if (event->buttons() & Qt::MiddleButton)
			{
				scene->moveCamera(dx, dy);
			}
		}
		else
		{
			if (isLMBDown)
			{
				QVector3D ori, dir;
				scene->getCamera().getRay(x, y, ori, dir);
				if (curSelectObj && curTool)
				{
					if (curTool->isManipulating())
					{
						curTool->goOnManipulate(ori, dir);
					}
				}
				else if (curToolType == GLViewWidget::TOOL_PAINT)
				{
//					scene->getPainter().goOnPaint(QVector2D(x, y));
				}
			}
			else
			{
				
				QVector3D ori, dir;
				camera.getRay(x, y, ori, dir);
				if (curTool)
				{
					if (!curTool->isManipulating())
					{
						char axis = curTool->intersect(ori, dir);
						curTool->selectAxis(axis);
					}
				}
				else if (curToolType == GLViewWidget::TOOL_PAINT)
				{
//					scene->getPainter().onMouseHover(QVector2D(x, y));
				}
			}

		}
		lastPos = event->pos();
		updateGL();
	}

	void GLViewWidget::mouseReleaseEvent(QMouseEvent *event)
	{
		int x = event->x();
		int y = event->y();
		QSharedPointer<Scene>scene = Global_WideoEditing_Window->scene;
		Camera& camera = scene->getCamera();
		if (!isAltDown && isLMBDown)
		{
			if (curSelectObj && curTool)
			{	// ����������
				if (curTool->isManipulating())
				{
					curTool->endManipulate();
					ManipulateCommand::ManipulateCommandType type;
					switch (curToolType)
					{
					case TOOL_TRANSLATE:
						type = ManipulateCommand::MANCMD_TRANSLATE; break;
					case TOOL_ROTATE:
						type = ManipulateCommand::MANCMD_ROTATE; break;
					case TOOL_SCALE:
						type = ManipulateCommand::MANCMD_SCALE; break;
					}
					QUndoCommand* cmd = new ManipulateCommand(
						curSelectObj.data()->getObjectID(),
						curTool->getOldTransform(),
						curTool->getNewTransform(),
						type);
//					scene->getUndoStack().push(cmd);
					goto END_RELEASE;
				}
			}
			else if (curToolType == GLViewWidget::TOOL_PAINT)
			{	// ��������
//				scene->getPainter().endPaint(QVector2D(x, y));
			}
			else if (curToolType == GLViewWidget::TOOL_FACE_SELECT)
			{	// ��ѡ������
				GeometryExposer& exposer = scene->getGeometryExposer();
				float w = this->width();
				float h = this->height();
				QVector2D minRatio(min(x, pressPos.x()) / w, min(y, pressPos.y()) / h);
				QVector2D maxRatio(max(x, pressPos.x()) / w, max(y, pressPos.y()) / h);
				QSet<int>faceIDSet;
				Mesh* objPtr = (Mesh*)curSelectObj.data();
				exposer.getRegionFaceID(minRatio, maxRatio, objPtr->getObjectID(), faceIDSet);
				if (isCtrlDown)
					objPtr->addSelectedFaceID(faceIDSet);
				else if (isShiftDown)
					objPtr->removeSelectedFaceID(faceIDSet);
				else
					objPtr->setSelectedFaceID(faceIDSet);
			}
		}
	END_RELEASE:
		isLMBDown = false;
		Global_WideoEditing_Window->transformEditor->updateWidgets();
		updateGL();
		return;
	}
	void GLViewWidget::keyPressEvent(QKeyEvent *event)
	{
		switch (event->key())
		{
		case Qt::Key_Alt:
			isAltDown = true;
			break;
		case Qt::Key_Control:
			isCtrlDown = true;
			break;
		case Qt::Key_Shift:
			isShiftDown = true;
			break;
		case Qt::Key_Delete:
//			QSharedPointer<RenderableObject> renObj = Global_WideoEditing_Window->viewWidget->getSelectedObject();;
//			if (renObj)
			{
//				QUndoCommand* cmd = new RemoveObjectCommand(renObj);
//				Global_WideoEditing_Window->scene->getUndoStack().push(cmd);
//				Global_WideoEditing_Window->actionSelect->trigger();
				Global_WideoEditing_Window->transformEditor->updateWidgets();
			}
		}
	}

	void GLViewWidget::keyReleaseEvent(QKeyEvent *event)
	{
		switch (event->key())
		{
		case Qt::Key_Alt:
			isAltDown = false; break;
		case Qt::Key_Control:
			isCtrlDown = false; break;
		case Qt::Key_Shift:
			isShiftDown = false; break;
		}
	}


	void GLViewWidget::wheelEvent(QWheelEvent *event)
	{
		Global_WideoEditing_Window->scene->zoomCamera(event->delta());
		if (curToolType == TOOL_PAINT)
		{
//			Global_WideoEditing_Window->scene->getPainter().onMouseHover(QVector2D(lastPos.x(), lastPos.y()));
		}
		updateGL();
	}

	void GLViewWidget::setTool(ToolType type)
	{
		if (curToolType == TOOL_PAINT)
		{
//			Global_WideoEditing_Window->paintEditor->attachToCamera(false);
		}
		curToolType = type;
		QSharedPointer<Scene> scene = Global_WideoEditing_Window->scene;
		bool isPaintMode = type == TOOL_PAINT;
		Mesh::enableWireFrame(!isPaintMode);
//		scene->getBrush().activate(isPaintMode);
//		Global_WideoEditing_Window->actionImport->setEnabled(!isPaintMode);
//		Global_WideoEditing_Window->actionPlaneLocator->setEnabled(!isPaintMode);
		switch (type)
		{
		case TOOL_TRANSLATE:
			curTool = &translateTool;
			break;
		case TOOL_ROTATE:
			curTool = &rotateTool;
			break;
		case TOOL_SCALE:
			curTool = &scaleTool;
			break;
		case TOOL_PAINT:
		case TOOL_SELECT:
		case TOOL_FACE_SELECT:
			curTool = NULL;
		}
		if (curTool)
		{
			QSharedPointer<RenderableObject>cso(curSelectObj);
			curTool->captureObject(cso.toWeakRef());
		}
		if (curToolType == GLViewWidget::TOOL_PAINT && curSelectObj)
		{
			if (curSelectObj.data()->getType() & RenderableObject::OBJ_MESH)
			{
				QWeakPointer<Mesh> wm = qWeakPointerCast<Mesh>(QSharedPointer<RenderableObject>(curSelectObj));
//				scene->getPainter().setObject(wm);
//				scene->getBrush().setObject(wm);
			}
		}
		bool hasObject = !curSelectObj.isNull();
		isPaintMode = curToolType == GLViewWidget::TOOL_PAINT;
//		Global_WideoEditing_Window->paintEditor->setEnabled(hasObject && isPaintMode);
//		Global_WideoEditing_Window->paintEditor->changeColorPicker(true);
		if (!isPaintMode)
			Global_WideoEditing_Window->scene->setPickerObjectUnfrozen();
		updateGL();
	}

	bool GLViewWidget::focusCurSelected()
	{
		if (curSelectObj)
		{
			RenderableObject *obj = curSelectObj.data();
			Global_WideoEditing_Window->scene->getCamera().setCenter(obj->getCenter(), obj->getApproSize());
			Global_WideoEditing_Window->scene->updateGeometryImage();

			return true;
		}
		return false;
	}

	QSharedPointer<RenderableObject> GLViewWidget::getSelectedObject()
	{
		return curSelectObj;
	}

	void GLViewWidget::enterPickerMode(void(*callBackFun)(QSharedPointer<RenderableObject>))
	{
		pickCallBackFun = callBackFun;
		isPickMode = true;
		setCursor(Qt::PointingHandCursor);
	}

}

