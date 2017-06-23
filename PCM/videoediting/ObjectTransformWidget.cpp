#include "basic_types.h"
#include "ObjectTransformWidget.h"
#include "RenderableObject.h"
#include "GLViewWidget.h"
#include "HistoryCommand.h"
#include "VideoEditingWindow.h"
#include "bulletInterface.h"
#include "QGLViewer\quaternion.h"
namespace videoEditting
{
	ObjectInfoWidget::ObjectInfoWidget(Ui::VideoEditingWindow& ui) :ui_(ui)
	{

		connect(ui_.translate_x, SIGNAL(valueChanged(double)), this, SLOT(updateSceneObject()));
		connect(ui_.translate_y, SIGNAL(valueChanged(double)), this, SLOT(updateSceneObject()));
		connect(ui_.translate_z, SIGNAL(valueChanged(double)), this, SLOT(updateSceneObject()));
		connect(ui_.rotate_x, SIGNAL(valueChanged(double)), this, SLOT(updateSceneObject()));
		connect(ui_.rotate_y, SIGNAL(valueChanged(double)), this, SLOT(updateSceneObject()));
		connect(ui_.rotate_z, SIGNAL(valueChanged(double)), this, SLOT(updateSceneObject()));
		connect(ui_.scale_x, SIGNAL(valueChanged(double)), this, SLOT(updateSceneObject()));
		connect(ui_.scale_y, SIGNAL(valueChanged(double)), this, SLOT(updateSceneObject()));
		connect(ui_.scale_z, SIGNAL(valueChanged(double)), this, SLOT(updateSceneObject()));
		connect(ui_.object_name, SIGNAL(editingFinished()), this, SLOT(updateObjectName()));
		connect(ui_.object_type, SIGNAL(editingFinished()), this, SLOT(updateObjectName()));

		ui_.translate_x->setValue(0.0f);
		ui_.translate_y->setValue(0.0f);
		ui_.translate_z->setValue(0.0f);
		ui_.rotate_x->setValue(0.0f);
		ui_.rotate_y->setValue(0.0f);
		ui_.rotate_z->setValue(0.0f);
		ui_.scale_x->setValue(1.0f);
		ui_.scale_y->setValue(1.0f);
		ui_.scale_z->setValue(1.0f);

		ui_.translate_x->setRange(-1000, 1000);
		ui_.translate_y->setRange(-1000, 1000);
		ui_.translate_z->setRange(-1000, 1000);
		ui_.rotate_x->setRange(-360, 360);
		ui_.rotate_y->setRange(-360, 360);
		ui_.rotate_z->setRange(-360, 360);
		ui_.scale_x->setRange(-10, 10);
		ui_.scale_y->setRange(-10, 10);
		ui_.scale_z->setRange(-10, 10);


		//camera param
		ui_.doubleSpinBox_farplane->setValue(100);
		ui_.doubleSpinBox_nearplane->setValue(0.2);
		ui_.doubleSpinBox_fov->setValue(60);
		ui_.doubleSpinBox_width_divide_height->setValue(1);
		
		ui_.doubleSpinBox_farplane->setRange(10, 1000);
		ui_.doubleSpinBox_farplane->setSingleStep(1.0);
		ui_.doubleSpinBox_nearplane->setRange(0.01, 10);
		ui_.doubleSpinBox_nearplane->setSingleStep(0.01);
		ui_.doubleSpinBox_fov->setRange(0.01, 89);
		ui_.doubleSpinBox_fov->setSingleStep(0.1);
		ui_.doubleSpinBox_width_divide_height->setRange(0.01, 100);
		ui_.doubleSpinBox_width_divide_height->setSingleStep(0.1);
		connect(ui_.doubleSpinBox_fov, SIGNAL(valueChanged(double)), this, SLOT(updateCameraParam()));
		connect(ui_.doubleSpinBox_width_divide_height, SIGNAL(valueChanged(double)), this, SLOT(updateCameraParam()));
		connect(ui_.doubleSpinBox_nearplane, SIGNAL(valueChanged(double)), this, SLOT(updateCameraParam()));
		connect(ui_.doubleSpinBox_farplane, SIGNAL(valueChanged(double)), this, SLOT(updateCameraParam()));
		
	}

	ObjectInfoWidget::~ObjectInfoWidget(void)
	{
	}

	void ObjectInfoWidget::updateWidgets()
	{
		Global_WideoEditing_Window->updateGLView();
		QSharedPointer<RenderableObject> pO = Global_WideoEditing_Window->activated_viewer()->getSelectedObject();
		if (pO)
		{
			ObjectTransform& trans = pO->getTransform();
			const QVector3D& pos = trans.getTranslate();
			ui_.translate_x->setValue(pos.x());
			ui_.translate_y->setValue(pos.y());
			ui_.translate_z->setValue(pos.z());
			const QQuaternion& q = trans.getRotate();


			float w = q.scalar();
			float x = q.x();
			float y = q.y();
			float z = q.z();
			float a = atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y));
			float b = asin(2 * (w*y - z*x));
			float c = atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z));
			ui_.rotate_x->setValue(a / M_PI * 180);
			ui_.rotate_y->setValue(b / M_PI * 180);
			ui_.rotate_z->setValue(c / M_PI * 180);
			const QVector3D& sca = trans.getScale();
			ui_.scale_x->setValue(sca.x());
			ui_.scale_y->setValue(sca.y());
			ui_.scale_z->setValue(sca.z());
			ui_.object_name->setText(pO->getName());
			switch (pO->getType())
			{
			case RenderableObject::OBJ_MESH:
			{
				ui_.object_type->setText("MESH");
				break;
			}
			case RenderableObject::OBJ_PICKER_OBJECT:
			{
				ui_.object_type->setText("PICKER_OBJECT");
				break;
			}
			case RenderableObject::OBJ_CAMERA:
			{
				ui_.object_type->setText("OBJ_CAMERA");
				break;
			}
			

			}
			if (pO->getType() == RenderableObject::OBJ_CAMERA)
			{
				Camera* camera = (Camera*)pO.data();
				ui_.doubleSpinBox_fov->setValue(camera->getFovAngle());
				ui_.doubleSpinBox_width_divide_height->setValue(camera->getAspectRatio());
				ui_.doubleSpinBox_nearplane->setValue(camera->getNearPlane());
				ui_.doubleSpinBox_farplane->setValue(camera->getFarPlane());
				int x, y;
				camera->getScreenResolution(x, y);
				ui_.lineEdit_viewport_width->setText(QString("%1").arg(x));
				ui_.lineEdit_viewport_height->setText(QString("%1").arg(y));
				ui_.lineEdit_viewport_width->setFocusPolicy(Qt::NoFocus);
				ui_.lineEdit_viewport_height->setFocusPolicy(Qt::NoFocus);
				switch(camera->getProjtype())
				{
					case Camera::GLCAMERA_PERSP:
						ui_.lineEdit_project_type->setText( "perspective");
					break;
					case Camera::GLCAMERA_ORTHO:
						ui_.lineEdit_project_type->setText("otho");
					break;

				}
				ui_.lineEdit_project_type->setFocusPolicy(Qt::NoFocus);
				
			}
		}

		ui_.object_pose->setEnabled(!pO.isNull());
		if (pO)
		{
			ui_.prespectuve_param->setEnabled(pO->getType() == RenderableObject::OBJ_CAMERA);
			ui_.cameraintrisic->setEnabled(pO->getType() == RenderableObject::OBJ_CAMERA);
		}
		else
		{
			ui_.prespectuve_param->setEnabled(false);
			ui_.cameraintrisic->setEnabled(false);
		}

	}

	void ObjectInfoWidget::updateSceneObject()
	{
		if (!Global_WideoEditing_Window)
			return;
		if (!Global_WideoEditing_Window->activated_viewer())
			return;
		QSharedPointer<RenderableObject> pO = Global_WideoEditing_Window->activated_viewer()->getSelectedObject();
		if (pO)
		{
			ObjectTransform  trans;// = pO->getTransform();
			trans.setTranslate(QVector3D(ui_.translate_x->value(), ui_.translate_y->value(), ui_.translate_z->value()));
			float a = ui_.rotate_x->value() / 180.0*M_PI;
			float b = ui_.rotate_y->value() / 180.0*M_PI;
			float c = ui_.rotate_z->value() / 180.0*M_PI;
			float cosA2 = cos(a / 2);
			float sinA2 = sin(a / 2);
			float cosB2 = cos(b / 2);
			float sinB2 = sin(b / 2);
			float cosC2 = cos(c / 2);
			float sinC2 = sin(c / 2);
			QQuaternion newQuat;
			newQuat.setScalar(cosA2*cosB2*cosC2 + sinA2*sinB2*sinC2);
			newQuat.setX(sinA2*cosB2*cosC2 - cosA2*sinB2*sinC2);
			newQuat.setY(cosA2*sinB2*cosC2 + sinA2*cosB2*sinC2);
			newQuat.setZ(cosA2*cosB2*sinC2 - sinA2*sinB2*cosC2);
			trans.setRotate(newQuat);
			trans.setScale(QVector3D(ui_.scale_x->value(), ui_.scale_y->value(), ui_.scale_z->value()));
			QUndoCommand* cmd = new ManipulateCommand(pO->getObjectID(), pO->getTransform(), trans, ManipulateCommand::MANCMD_ALL);
			if (pO->getType() == RenderableObject::OBJ_MESH)
			{
				if (g_current_frame < g_total_frame)
				{
					//g_translations[g_current_frame] = trans.getTranslate();
					//g_rotations[g_current_frame] = trans.getRotate();
				}
			}
//			paint3DApp->scene->getUndoStack().push(cmd);
		}
	}

	void ObjectInfoWidget::updateCameraParam()
	{
		QSharedPointer<RenderableObject> pO = Global_WideoEditing_Window->activated_viewer()->getSelectedObject();
		if (pO && pO->getType() == RenderableObject::OBJ_CAMERA)
		{
			 
			Camera* camera = (Camera*)pO.data();
			camera->setFovAngle(ui_.doubleSpinBox_fov->value());
			camera->setAspectRatio(ui_.doubleSpinBox_width_divide_height->value());
			camera->setNearPlane(ui_.doubleSpinBox_nearplane->value());
			camera->setFarPlane(ui_.doubleSpinBox_farplane->value());
			Global_WideoEditing_Window->updateGeometryImage();
			Global_WideoEditing_Window->updateGLView();
		}

	}

	void ObjectInfoWidget::updateObjectName()
	{
		QString oldName = Global_WideoEditing_Window->activated_viewer()->getSelectedObject()->getName();
		Global_WideoEditing_Window->activated_viewer()->getScene().rename(oldName, ui_.object_name->text());
		ui_.object_name->setText(Global_WideoEditing_Window->activated_viewer()->getSelectedObject()->getName());
	}
	void ObjectInfoWidget::updateObjectType()
	{
		//ObjectType oldtype = Global_WideoEditing_Window->active_viewer()->getSelectedObject()->getType()
		//VideoEditingWindow::scene->rename(oldName, ui_.object_name->text());
		//ui_.object_name->setText(Global_WideoEditing_Window->active_viewer()->getSelectedObject()->getName());
	}
}

