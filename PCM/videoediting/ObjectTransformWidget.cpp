#include "basic_types.h"
#include "ObjectTransformWidget.h"
#include "RenderableObject.h"
#include "GLViewWidget.h"
#include "HistoryCommand.h"
#include "VideoEditingWindow.h"


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
	}

	ObjectInfoWidget::~ObjectInfoWidget(void)
	{
	}

	void ObjectInfoWidget::updateWidgets()
	{
		Global_WideoEditing_Window->updateGLView();
		QSharedPointer<RenderableObject> pO = Global_WideoEditing_Window->active_viewer()->getSelectedObject();
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
			}
			case RenderableObject::OBJ_PICKER_OBJECT:
			{
				ui_.object_type->setText("PICKER_OBJECT");
			}

			}
		}
		ui_.object_pose->setEnabled(!pO.isNull());
	}

	void ObjectInfoWidget::updateSceneObject()
	{
		QSharedPointer<RenderableObject> pO = Global_WideoEditing_Window->active_viewer()->getSelectedObject();
		if (pO)
		{
			ObjectTransform  trans;
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
//			paint3DApp->scene->getUndoStack().push(cmd);
		}
	}

	void ObjectInfoWidget::updateObjectName()
	{
		QString oldName = Global_WideoEditing_Window->active_viewer()->getSelectedObject()->getName();
		VideoEditingWindow::scene->rename(oldName, ui_.object_name->text());
		ui_.object_name->setText(Global_WideoEditing_Window->active_viewer()->getSelectedObject()->getName());
	}
	void ObjectInfoWidget::updateObjectType()
	{
		//ObjectType oldtype = Global_WideoEditing_Window->active_viewer()->getSelectedObject()->getType()
		//VideoEditingWindow::scene->rename(oldName, ui_.object_name->text());
		//ui_.object_name->setText(Global_WideoEditing_Window->active_viewer()->getSelectedObject()->getName());
	}
}

