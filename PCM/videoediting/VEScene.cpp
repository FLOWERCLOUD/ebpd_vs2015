#include "VEScene.h"
#include "ObjReader.h"
#include "camera.h"
#include "bulletInterface.h"
#include <QtWidgets\qmessagebox.h>
#include <qfile.h>
//extern Paint3DFrame* paint3DApp;
namespace videoEditting
{



	QSharedPointer<CommonScene> Scene::common_scene = QSharedPointer<CommonScene>(new CommonScene);

	Scene::Scene(QGLContext* qglcontex)
	{
		qglcontex->makeCurrent();
		camera = QSharedPointer<Camera>(new Camera);
		isGeometryImgInvalid = true;
		isEnvMapUpdated = false;
		
	}

	Scene::~Scene(void)
	{

	}

	void Scene::addPlanePicker()
	{
		//QSharedPointer<PlanePicker> picker(new PlanePicker);
		//picker->init();
		//objectArray.push_back(picker);
		refreshExposerObjectList();
	}



	void Scene::drawGrid()
	{
		glDisable(GL_LIGHTING);
		glLineWidth(1.0f);
		glBegin(GL_LINES);
		glColor3f(0.3, 0.3, 0.3);
		for (int i = -10; i <= 10; ++i)
		{
			glVertex3f(-10, i, 0);
			glVertex3f(10, i, 0);
			glVertex3f(i, -10, 0);
			glVertex3f(i, 10, 0);
		}
		glEnd();
		glEnable(GL_LIGHTING);
	}


	bool Scene::init()
	{
		unsigned err = glewInit();
		if (GLEW_OK != err)
		{
			const GLubyte *e = glewGetErrorString(err);
			QMessageBox::critical(NULL, QObject::tr("Error"), QObject::tr("GLEW initialization error."));
			qDebug() << e;
		}

		if (glewIsSupported("GL_EXT_framebuffer_object"))
			qDebug() << "Old EXT FBO available" << endl;
		else
		{
			qDebug() << "Old EXT FBO NOT available" << endl;
			QMessageBox::critical(NULL, QObject::tr("Error"), QObject::tr("Old EXT FBO NOT available"));
		}

		if (glewIsSupported("GL_ARB_framebuffer_object"))
			qDebug() << "Newer ARB FBO available" << endl;
		else
		{
			qDebug() << "Newer ARB FBO NOT available" << endl;
			//QMessageBox::critical(NULL, QObject::tr("Error"), QObject::tr("New ARB FBO NOT available"));
		}


		bool isfbo = GeometryExposer::isFBOSupported();
		int w, h;
		int x_offset, y_offset;
		camera->init();
//		camera->getScreenResolution(w, h);
		camera->getCameraViewport(x_offset, y_offset,w, h);
		exposer.init(w / SCREEN_BUFFER_RATIO, h / SCREEN_BUFFER_RATIO, x_offset, y_offset);

//		strokeLib.init();
//		brush = QSharedPointer<Brush>(new Brush(this));
//		brush->setStroke(strokeLib.getDefaultStroke());
		//QMessageBox::information(NULL, QObject::tr("Info"), QObject::tr("brush initialized."));
//		painter = QSharedPointer<Painter>(new SurfacePainter(this));
//		painter->setBrush(brush);
		//QMessageBox::information(NULL, QObject::tr("Info"), QObject::tr("painter initialized."));

		//importObj("expplane.obj");
//		undoStack.setUndoLimit(50);


//		envMap.load("background.png", 512);
//		envMap.initGLBuffer();
		// QMessageBox::information(NULL, QObject::tr("Info"), QObject::tr("env map initialized."));
		//envMap.saveCubeMap("background.jpg");


		isGeometryImgInvalid = true;
		return true;
	}

	void Scene::draw(bool  isdrawgrid)
	{
		
		camera->applyGLMatrices();

		//����������
		if (isdrawgrid)
			drawGrid();

		if (isGeometryImgInvalid)
		{
			exposer.exposeGeometry();
			isGeometryImgInvalid = false;
		}
		//brush->draw();

		// ��������shader
		QSharedPointer<QGLShaderProgram>meshShader = Mesh::getAppearanceShader();
		if (meshShader)
		{
			if (!meshShader->bind())
			{
				std::cout << "getAppearanceShader() bind  error" << std::endl;
			}

			// �󶨻�����ͼ,ֻ��һ��
			if (!isEnvMapUpdated)
			{
				int texRegBase = GL_TEXTURE0_ARB + SCENE_TEXTURE_REGISTER_OFFSET;
				int texRegOffset = SCENE_TEXTURE_REGISTER_OFFSET;
				glActiveTextureARB(texRegBase + 0);
//				glBindTexture(GL_TEXTURE_CUBE_MAP, envMap.getGLTexObj());
				meshShader->setUniformValue("envTex", texRegOffset + 0);
				isEnvMapUpdated = true;
			}

			// �趨ת������
			QMatrix4x4 viewMatrix = camera->getViewMatrix();
			meshShader->setUniformValue("viewMatrixTranspose", viewMatrix.transposed());


			// �Ȼ���͸��������
			for (int i = 0; i < common_scene->objectArray.size(); ++i)
			{
				if (common_scene->objectArray[i]->getType() == RenderableObject::OBJ_MESH)
				{
					common_scene->objectArray[i]->drawAppearance();
				}
				else if (common_scene->objectArray[i]->getType() == RenderableObject::OBJ_CAMERA)
				{
					common_scene->objectArray[i]->drawAppearance();
				}
			}
			if (oberveCamera)
			{
				oberveCamera.data()->drawAppearance();
			}

			// �ٻ�͸��������
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glDepthMask(GL_FALSE);
			for (int i = 0; i < common_scene->objectArray.size(); ++i)
			{
				if (common_scene->objectArray[i]->getType() == RenderableObject::OBJ_PICKER_OBJECT)
				{
					common_scene->objectArray[i]->drawAppearance();
				}
			}
			glDepthMask(GL_TRUE);
			glDisable(GL_BLEND);

			// �ͷ�shader
			meshShader->release();
		}


		//if (paint3DApp->viewWidget->getCurToolType() == GLViewWidget::TOOL_PAINT)
		//	brush->draw();


		GLenum errorMsg = glGetError();
		if (errorMsg != GL_NO_ERROR)
			qDebug() << "error occurs when rendering" << endl;

	}

	bool Scene::rename(const QString& oldName, const QString& newName)
	{
		QVector<QSharedPointer<RenderableObject>>::iterator pSel = common_scene->objectArray.end();
		for (QVector<QSharedPointer<RenderableObject>>::iterator pM = common_scene->objectArray.begin(); pM != common_scene->objectArray.end(); ++pM)
		{
			if ((*pM)->getName() == oldName)
			{
				pSel = pM;	break;
			}
		}
		if (pSel != common_scene->objectArray.end())
		{
			(*pSel)->setName(newName);
			return true;
		}
		if (oberveCamera)
		{
			if ( oberveCamera.data()->getName() == oldName)
			{
				oberveCamera.data()->setName(newName);
			}
		}
		return false;
	}




	QWeakPointer<RenderableObject> Scene::selectObject(int x, int y)
	{
		//int width, height;
		//camera->getScreenResolution(width, height);
		//float xRatio = x / float(width);
		//float yRatio = y / float(height);

		int width, height;
		int x_offset, y_offset;//y_offset ��������½�
		int glwidget_width, glwidget_height;
		//	camera->getScreenResolution(width, height);
		camera->getCameraViewport(x_offset, y_offset, width, height);
		camera->getGLWidgetResoluiont(glwidget_width, glwidget_height);
		//float xRatio = x / float(width);
		//float yRatio = y / float(height);
		float xRatio = (x - x_offset) / float(width);
		float y_offset_reverse = glwidget_height - y_offset - height;
		float yRatio = (y - y_offset_reverse) / float(height);

		unsigned char objID;
		exposer.getObjectID(QVector2D(xRatio, yRatio), objID);
		QWeakPointer<RenderableObject> curSelectObj;
		QSharedPointer<RenderableObject> bestObj;
		for (QVector<QSharedPointer<RenderableObject>>::iterator pM = common_scene->objectArray.begin();
			pM != common_scene->objectArray.end(); ++pM)
		{
			float t;
			if (objID == (*pM)->getObjectID())
			{
				(*pM)->select();
				curSelectObj = pM->toWeakRef();
				//brush.setObject(m_curSelectObj);
			}
			else
				(*pM)->deselect();
		}
		if (oberveCamera)
		{
			if (oberveCamera.data()->getObjectID() == objID)
			{
				oberveCamera.data()->select();
				curSelectObj = oberveCamera;
			}else
				oberveCamera.data()->deselect();
		}
		return curSelectObj;
	}

	QWeakPointer<RenderableObject> Scene::selectObject(int objID)
	{
		QSharedPointer<RenderableObject> bestObj;
		for (QVector<QSharedPointer<RenderableObject>>::iterator pM = common_scene->objectArray.begin();
			pM != common_scene->objectArray.end(); ++pM)
		{
			float t;
			if (objID == (*pM)->getObjectID())
			{
				(*pM)->select();
				return pM->toWeakRef();
			}
			else
				(*pM)->deselect();
		}
		if (oberveCamera)
		{
			if (oberveCamera.data()->getObjectID() == objID)
			{
				oberveCamera.data()->select();
				return oberveCamera;
			}
			else
				oberveCamera.data()->deselect();
		}
		return QWeakPointer<RenderableObject>();
	}

	QWeakPointer<RenderableObject> Scene::intersectObject(int x, int y)
	{
		int width, height;
		int x_offset, y_offset;//y_offset ��������½�
		int glwidget_width, glwidget_height;
	//	camera->getScreenResolution(width, height);
		camera->getCameraViewport(x_offset, y_offset, width, height);
		camera->getGLWidgetResoluiont(glwidget_width, glwidget_height);
		//float xRatio = x / float(width);
		//float yRatio = y / float(height);
		float xRatio = (x- x_offset) / float(width);
		float y_offset_reverse = glwidget_height - y_offset - height;
		float yRatio =  (y - y_offset_reverse) / float(height);

		unsigned char objID;
		exposer.getObjectID(QVector2D(xRatio, yRatio), objID);

		for (QVector<QSharedPointer<RenderableObject>>::iterator pM = common_scene->objectArray.begin();
			pM != common_scene->objectArray.end(); ++pM)
		{
			float t;
			if (objID == (*pM)->getObjectID())
			{
				return (*pM).toWeakRef();
			}
		}
		if (oberveCamera)
		{
			if (oberveCamera.data()->getObjectID() == objID)
			{	
				return oberveCamera;
			}
		}
		return QWeakPointer<RenderableObject>();
	}

	void Scene::resizeCamera(int width, int height, int glwidget_width , int glwidget_height)
	{
		camera->setScreenResolution(width, height, glwidget_width, glwidget_height);
		int x_offset, yoffset, cwidth, cheight;
		camera->getCameraViewport(x_offset, yoffset, cwidth, cheight);
		exposer.setResolution(cwidth / SCREEN_BUFFER_RATIO, cheight / SCREEN_BUFFER_RATIO,
			0, 0);
		isGeometryImgInvalid = true;
	}

	void Scene::refreshExposerObjectList()
	{
		QVector<QWeakPointer<RenderableObject>> v;
		for (int i = 0; i < common_scene->objectArray.size(); i++)
		{
			QWeakPointer<RenderableObject> pO = common_scene->objectArray[i].toWeakRef();
			if (!common_scene->frozenObjectSet.contains(pO))
			{
				v.push_back(common_scene->objectArray[i].toWeakRef());
			}
		}
		if (oberveCamera)
		{
			v.push_back(oberveCamera);
		}
		exposer.setRenderObject(v);
		isGeometryImgInvalid = true;
	}

	void Scene::rotateCamera(float dx, float dy)
	{
		camera->rotateCamera(dx, dy);
		isGeometryImgInvalid = true;
	}

	void Scene::moveCamera(float dx, float dy)
	{
		camera->moveCamera(dx, dy);
		isGeometryImgInvalid = true;

	}

	void Scene::zoomCamera(float dz)
	{
		camera->zoomCamera(dz);
		isGeometryImgInvalid = true;
	}

	Mesh* Scene::getMesh(int objID)
	{
		Mesh* mesh;
		for (QVector<QSharedPointer<RenderableObject>>::iterator pO = common_scene->objectArray.begin();
			pO != common_scene->objectArray.end(); ++pO)
		{
			if (((*pO)->getType() & RenderableObject::OBJ_MESH) &&
				(*pO)->getObjectID() == objID)
			{
				mesh = (Mesh*)((*pO).data());
				return mesh;
			}
		}

		return NULL;
	}

	QSharedPointer<RenderableObject> Scene::getObject(int objID)
	{
		for (QVector<QSharedPointer<RenderableObject>>::iterator pO = common_scene->objectArray.begin();
			pO != common_scene->objectArray.end(); ++pO)
		{
			if ((*pO)->getObjectID() == objID)
			{
				return ((*pO));
			}
		}
		if (oberveCamera)
		{
			if(oberveCamera.data()->getObjectID() == objID)
			{
				return oberveCamera;
			}
		}
		return QSharedPointer<RenderableObject>();
	}

	QSharedPointer<RenderableObject> Scene::getObject(const QString& objName)
	{
		for (QVector<QSharedPointer<RenderableObject>>::iterator pO = common_scene->objectArray.begin();
			pO != common_scene->objectArray.end(); ++pO)
		{
			if ((*pO)->getName() == objName)
			{
				return ((*pO));
			}
		}
		if (oberveCamera)
		{
			if (oberveCamera.data()->getName() == objName)
			{
				return oberveCamera;
			}
		}
		return QSharedPointer<RenderableObject>();
	}

	void Scene::getObjectNames(RenderableObject::ObjectType type, QStringList& names)
	{
		for (QVector<QSharedPointer<RenderableObject>>::iterator pO = common_scene->objectArray.begin();
			pO != common_scene->objectArray.end(); ++pO)
		{
			if ((*pO)->getType() == RenderableObject::OBJ_PICKER_OBJECT)
			{
				names.push_back((*pO)->getName());
			}
		}

	}



	bool Scene::importEnvMap(const QString& fileName)
	{
//		envMap.releaseGLBuffer();
//		envMap.load(fileName, 512);
//		envMap.initGLBuffer();
		isEnvMapUpdated = false;
		return true;
	}

	bool Scene::removeObject(QSharedPointer<RenderableObject>& obj)
	{
		for (QVector<QSharedPointer<RenderableObject>>::iterator pO = common_scene->objectArray.begin();
			pO != common_scene->objectArray.end(); ++pO)
		{
			if (*pO == obj)
			{
				common_scene->objectArray.erase(pO);
				refreshExposerObjectList();
				isGeometryImgInvalid = true;
				return true;
			}
		}

		return false;
	}

	void Scene::insertObject(const QSharedPointer<RenderableObject>& obj)
	{
		if (obj->getType() == RenderableObject::OBJ_CAMERA)
		{
			common_scene->objectArray.push_back(obj);
		}
		
		selectObject(obj->getObjectID());
		refreshExposerObjectList();
		isGeometryImgInvalid = true;
	}

	void Scene::setObjectFrozen(QSet<QWeakPointer<RenderableObject>> obj)
	{
		common_scene->frozenObjectSet += obj;
		refreshExposerObjectList();
	}

	void Scene::setObjectUnfrozen(QSet<QWeakPointer<RenderableObject>> obj)
	{
		common_scene->frozenObjectSet -= obj;
		refreshExposerObjectList();
	}

	void Scene::setOtherObjectFrozen(QSet<QWeakPointer<RenderableObject>> obj)
	{
		for (int i = 0; i < common_scene->objectArray.size(); ++i)
		{
			if (!obj.contains(common_scene->objectArray[i]))
			{
				common_scene->frozenObjectSet.insert(common_scene->objectArray[i].toWeakRef());
			}
		}
		refreshExposerObjectList();
	}

	void Scene::setAllObjectUnfrozen()
	{
		common_scene->frozenObjectSet.clear();
		refreshExposerObjectList();
	}

	void Scene::setPickerObjectFrozen()
	{
		for (int i = 0; i < common_scene->objectArray.size(); ++i)
		{
			if (common_scene->objectArray[i]->getType() & RenderableObject::CMP_PICKER)
			{
				common_scene->frozenObjectSet.insert(common_scene->objectArray[i].toWeakRef());
			}
		}
		refreshExposerObjectList();
	}



	void Scene::setPickerObjectUnfrozen()
	{
		for (int i = 0; i < common_scene->objectArray.size(); ++i)
		{
			if (common_scene->objectArray[i]->getType() & RenderableObject::CMP_PICKER)
			{
				common_scene->frozenObjectSet.remove(common_scene->objectArray[i].toWeakRef());
			}
		}
		refreshExposerObjectList();
	}

	const QSet<QWeakPointer<RenderableObject>>& Scene::getFrozenObjectSet() { return common_scene->frozenObjectSet; }
	void Scene::save(const QString& fileName)
	{
		QFile file(fileName);
		if (file.open(QIODevice::WriteOnly))
		{
			QDataStream out(&file);
			out.setVersion(QDataStream::Qt_5_2);
			out << common_scene->objectArray;
			out << getCamera();
		}
	}
	void Scene::open(const QString& fileName)
	{

		QFile file(fileName);
		if (file.open(QIODevice::ReadOnly))
		{
			clear();
			QDataStream in(&file);
			in.setVersion(QDataStream::Qt_5_2);
			in >> common_scene->objectArray;
			in>> getCamera();
		}
		QVector<QWeakPointer<RenderableObject>> v;
		for (int i = 0; i < common_scene->objectArray.size(); i++)
		{
			QWeakPointer<RenderableObject> pO = common_scene->objectArray[i].toWeakRef();
			pO.data()->updateTransformMatrix();
		}
		if (oberveCamera)
		{
			oberveCamera.data()->updateTransformMatrix();
		}
		refreshExposerObjectList();
		isGeometryImgInvalid = true;
	}
	bool Scene::importObj(const QString& fileName)
	{
		ObjReader reader;
		if (!reader.read(fileName))
			return false;
		for (int i = 0; i < reader.getNumMeshes(); ++i)
		{
			QSharedPointer<Mesh> pM(reader.getMesh(i));
			pM->init();
			common_scene->objectArray.push_back(pM);
			g_init_vertices = pM->getVertices().toStdVector();
			g_faces_ = pM->getFacesIdxs().toStdVector();
		}
		refreshExposerObjectList();
		isGeometryImgInvalid = true;
		return true;
	}



	void Scene::clear()
	{
		common_scene->objectArray.clear();
		common_scene->frozenObjectSet.clear();
			
		//undoStack.clear();
		refreshExposerObjectList();
		//brush->setObject(QWeakPointer<Mesh>());
		//painter->setObject(QWeakPointer<Mesh>());
	}


	uint qHash(const QWeakPointer<RenderableObject>& key)
	{
		return (uint)key.data();
	}

}

