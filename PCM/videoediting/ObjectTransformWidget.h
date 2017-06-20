#pragma once
#include "ui_VideoEditingScene.h"
#include <QObject>
namespace videoEditting
{
	class ObjectInfoWidget : QObject
	{
		Q_OBJECT
	public:
		ObjectInfoWidget(Ui::VideoEditingWindow& ui);
		~ObjectInfoWidget(void);

		public slots:
		void updateWidgets();
		void updateSceneObject();
		void updateCameraParam();
		void updateObjectName();
		void updateObjectType();
	private:
		Ui::VideoEditingWindow& ui_;
	};
}

