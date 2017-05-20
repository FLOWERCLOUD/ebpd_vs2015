#include "parsers/loader.hpp"
#include "parsers/fbx_loader.hpp"
#include "parsers/obj_loader.hpp"
#include "control/cuda_ctrl.hpp"
#include "meshes/mesh_utils_loader.hpp"
#include "animation/skeleton.hpp"
#include "global_datas/cuda_globals.hpp"
#include "global_datas/toolglobals.hpp"
#include "qt_gui/OGL_viewports_skin2.hpp"
#include "main_window.h"
#include <QFileDialog>
#include <QColorDialog>
#include <QMessageBox>
#include "qt_gui/tools/popup_ok_cancel.hpp"
#include "VideoEditingWindow.h"

void main_window::load_fbx_mesh( Loader::Fbx_file& loader)
{
	Mesh* ptr_mesh = new Mesh();
	Loader::Abs_mesh abs_mesh;
	loader.get_mesh( abs_mesh );
	if(abs_mesh._vertices.size() > 0)
	{
		Mesh_utils::load_mesh(*ptr_mesh, abs_mesh);
		Cuda_ctrl::load_mesh( ptr_mesh );
	}else
		QMessageBox::information(this, "Warning", "no mesh found");
}

// -----------------------------------------------------------------------------

bool main_window::load_custom_skeleton(QString name)
{
	QString skel_name = name;
	skel_name.append(".skel");
	if( QFile::exists(skel_name) )
	{
		Cuda_ctrl::_graph.load_from_file( skel_name.toLatin1() );
		Cuda_ctrl::_skeleton.load( *g_graph );
		return true;
	}
	else
	{
		QMessageBox::information(this, "Error", "Can't' find "+name+".skel");
		return false;
	}
}

// -----------------------------------------------------------------------------

bool main_window::load_custom_weights(QString name)
{
	QString ssd_name = name;
	ssd_name.append(".weights");
	if( QFile::exists(ssd_name) )
	{
		Cuda_ctrl::load_animesh_and_ssd_weights(ssd_name.toLatin1());
		return true;
	}
	else
	{
		QMessageBox::information(this, "Error", "Can't' find "+name+".weights");
		return false;
	}
}

// -----------------------------------------------------------------------------

bool main_window::load_fbx_skeleton_anims(const Loader::Fbx_file& loader)
{
	//cout<<" not support fbx"<<endl;
	//return false;
	// Extract skeleton data
	Loader::Abs_skeleton skel;
	loader.get_skeleton(skel);
	if(skel._bones.size() == 0) return false;

	// Convert to our skeleton representation
	Cuda_ctrl::_skeleton.load( skel );
	Cuda_ctrl::_skeleton.set_offset_scale( g_mesh->get_offset(), g_mesh->get_scale());

	Cuda_ctrl::load_animesh(); // Bind animated mesh to skel
	enable_animesh( false );   // enable gui for animated mesh0
	// Convert bones weights to our representation
	Cuda_ctrl::_anim_mesh->set_ssd_weight( skel );

	// Load first animation
	std::vector<Loader::Base_anim_eval*> anims;
	loader.get_animations( anims );
	ui.toolBar_frame->set_anim_list( anims );

	return true;
}

void main_window::load_ism(const QString& fileName)
{
	if( fileName.size() != 0)
	{
		PaintCanvas* wgl = _viewports->active_viewport();
		//wgl->makeCurrent();

		QFileInfo fi(fileName);
		QString name         = fi.canonicalPath() + "/" + fi.completeBaseName();
		QString skel_name    = name;
		QString weights_name = name;
		QString ism_name     = name;
		weights_name.append(".weights");
		skel_name.   append(".skel"   );
		ism_name.    append(".ism"    );

		// Load mesh
		bool skel_loaded = false;
		QString mesh_name = name;
		mesh_name.append(".off");
		if( QFile::exists(mesh_name) )
			Cuda_ctrl::load_mesh(mesh_name.toLocal8Bit().constData()); //replace toStdString()
		else if( QFile::exists((mesh_name = name).append(".obj")) )
			Cuda_ctrl::load_mesh(mesh_name.toLocal8Bit().constData());
		else if( QFile::exists((mesh_name = name).append(".fbx")) )
		{
			Loader::Fbx_file loader( mesh_name.toLocal8Bit().constData());
			load_fbx_mesh( loader );
			skel_loaded = load_fbx_skeleton_anims( loader );
			if( skel_loaded )
			{
				_viewports->set_io(EOGL_widget::MESH_EDIT);
				enable_animesh( true );
			}
		}
		else
		{
			QMessageBox::information(this, "Error !", "Can't' find "+name+"'.obj/.off/.fbx'\n");
			return;
		}
		// Enable GUI for mesh
		enable_mesh( true );

		if( !skel_loaded )
		{
			// Load skeleton graph
			if( !load_custom_skeleton( name ) ) return;
			// Load ssd weights
			if( !load_custom_weights( name ) ) return;
		}

		Cuda_ctrl::_anim_mesh->load_ism(fileName.toLatin1());

		// Enable GUI for animesh
		enable_animesh( true );
		_viewports->set_io(EOGL_widget::MESH_EDIT);
	}
	update_viewports();
}

void main_window::on_actionLoad_model_triggered(bool checked)
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load model"),
		"./resource/meshes",
		tr("*.ism") );

	if(fileName.size())
		load_ism(fileName);
}

void main_window::on_actionLoad_ISM_triggered()
{
	if( !Cuda_ctrl::is_animesh_loaded() ){
		QMessageBox::information(this, "Error", "No animated mesh loaded");
		return;
	}

	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load ism"),
		"./resource/meshes",
		tr("*.ism") );

	if( fileName.size() != 0)
		Cuda_ctrl::_anim_mesh->load_ism(fileName.toLatin1());

	update_viewports();
}

void main_window::on_actionLoad_FBX_triggered()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load from FBX"),
		"./resource/meshes",
		tr("*.fbx") );
	if( fileName.size() != 0)
	{
		QGLWidget* wgl = _viewports->shared_viewport();
		//wgl->makeCurrent();

		QString name = fileName.section('.',0,0);

		// Load mesh
		QString mesh_name = name;
		mesh_name.append(".fbx");
		if( QFile::exists(mesh_name) )
		{
			Loader::Fbx_file loader( mesh_name.toLocal8Bit().constData());
			load_fbx_mesh( loader );
			enable_mesh( true );
			_viewports->set_io(EOGL_widget::GRAPH);

			if( load_fbx_skeleton_anims( loader ) )
			{
				_viewports->set_io(EOGL_widget::MESH_EDIT);
				enable_animesh( true );
			}
		}
		else
		{
			QMessageBox::information(this, "Error !", "Can't' find "+name+"'.fbx'\n");
			return;
		}
	}

	update_viewports();
}

void main_window::on_actionLoad_mesh_triggered()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load mesh"),
		"./resource/meshes",
		tr("*.off *.obj *.fbx") );
	if( fileName.size() != 0)
	{
		Cuda_ctrl::load_mesh(fileName.toLocal8Bit().constData());
		Cuda_ctrl::erase_graph();
		enable_animesh( false );
		enable_mesh( true );
		_viewports->set_io(EOGL_widget::GRAPH);
	}

	update_viewports();
}

void main_window::on_actionLoad_skeleton_triggered()
{
	if( !Cuda_ctrl::is_mesh_loaded() ){
		QMessageBox::information(this, "Error", "You must load a mesh before.");
		return;
	}

	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load skeleton"),
		"./resource/meshes",
		tr("*.skel *.fbx") );
	if( fileName.size() != 0)
	{
		QFileInfo fi(fileName);
		QString ext = fi.suffix().toLower();

		if(ext == "fbx")
		{
			// Parse file
			Loader::Fbx_file loader( fileName.toLocal8Bit().constData() );

			// Load into our data representation
			if( load_fbx_skeleton_anims( loader ) )
			{
				enable_animesh( true );
				_viewports->set_io(EOGL_widget::MESH_EDIT);
			}
		}
		else if( ext == "skel")
		{
			Cuda_ctrl::_graph.load_from_file(fileName.toLatin1());
			Cuda_ctrl::_skeleton.load( *g_graph );
			_viewports->set_io(EOGL_widget::GRAPH);
		}
		else
		{
			QMessageBox::information(this, "Error", "Unsupported file type: '"+ext+"'");
		}
	}

	update_viewports();
}

void main_window::on_actionLoad_weights_triggered()
{
	if( !Cuda_ctrl::is_animesh_loaded() ){
		QMessageBox::information(this, "Error", "No animated mesh loaded");
		return;
	}

	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load skinning weights"),
		"./resource/meshes",
		tr("*.weights *.csv") );
	if( fileName.size() != 0)
	{
		if(Cuda_ctrl::is_animesh_loaded())
			Cuda_ctrl::_anim_mesh->load_weights(fileName.toLatin1());
		else
			Cuda_ctrl::load_animesh_and_ssd_weights( fileName.toLatin1() );

		_viewports->set_io(EOGL_widget::MESH_EDIT);
		enable_animesh( true );
	}

	update_viewports();
}


void main_window::on_actionLoad_keyframes_triggered()
{
	if( !Cuda_ctrl::is_animesh_loaded() ){
		QMessageBox::information(this, "Error", "No animated mesh loaded");
		return;
	}

	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load keyframes"),
		"./resource/meshes",
		tr("*.fbx") );
	if( fileName.size() != 0)
	{
		QFileInfo fi(fileName);
		QString ext = fi.suffix().toLower();

		if(ext == "fbx")
		{
			// Parse file
			Loader::Fbx_file loader( fileName.toLocal8Bit().constData() );
			// Load into our data representation
			std::vector<Loader::Base_anim_eval*> anims;
			loader.get_animations( anims );

			Diag_ok_cancel diag("Add or replace keyframes",
				"Do you want to replace the current animation tracks",
				this);

			if(diag.exec()) ui.toolBar_frame->set_anim_list( anims );
			else            ui.toolBar_frame->add_anims( anims );
		}
		else
		{
			QMessageBox::information(this, "Error", "Unsupported file type: '"+ext+"'");
		}
	}
	update_viewports();
}

void main_window::on_actionLoad_cluster_triggered()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Save cluster"),
		"./resource/meshes",
		tr("*.cluster") );
	if( fileName.size() != 0){
		Cuda_ctrl::_anim_mesh->load_cluster(fileName.toLatin1());
		update_viewports();
	}

}

void main_window::on_actionLoad_pose_triggered()
{

	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load pose"),
		"./resource/meshes",
		tr("*.skel_pose") );
	if( fileName.size() != 0){
		Cuda_ctrl::_skeleton.load_pose( fileName.toLocal8Bit().constData() );
		update_viewports();
	}
}

void main_window::on_actionLoad_camera_triggered()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load pose"),
		"./resource/meshes",
		tr("*.cam") );
	if( fileName.size() != 0){

		PaintCanvas* wgl = _viewports->active_viewport();
		//Tbx::load_class(wgl->camera(), fileName.toLocal8Bit().constData());
		update_viewports();
	}
}

void main_window::on_actionLoad_exampleMesh_triggered()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load model"),
		"./resource/meshes",
		tr("*.rig") );

	QFileInfo fi(fileName);
	resetSampleSet();
	if( fileName.size() != 0)
	{
		Cuda_ctrl::genertateVertices(
			(fi.canonicalPath() + "/").toLocal8Bit().constData(),
			fi.completeBaseName().toLocal8Bit().constData());
	}
	else
		return;
	createTreeWidgetItems();
	m_layer->updateTable();
	QApplication::setOverrideCursor(Qt::WaitCursor);
	//textEdit->setPlainText(in.readAll());
	QApplication::restoreOverrideCursor();
	//	setCurrentFile(fileName);
	statusBar()->showMessage(tr("File loaded"), 2000);
}
void main_window::on_actionLoad_depthImage_triggered()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load depth image"),
		"./resource/meshes",
		tr("*.depth"));

	QFileInfo fi(fileName);
//	resetSampleSet();
	if (fileName.size() != 0)
	{
		Cuda_ctrl::_image_ctrl.add_depthImage(
			(fi.canonicalPath() + "/").toLocal8Bit().constData(),
			fi.completeBaseName().toLocal8Bit().constData());
		statusBar()->showMessage(tr("depthImage loaded"), 2000);
	}
	else
		return;
	createTreeWidgetItems();
	m_layer->updateTable();
	QApplication::setOverrideCursor(Qt::WaitCursor);
	//textEdit->setPlainText(in.readAll());
	QApplication::restoreOverrideCursor();
}
void main_window::on_actionLoad_sampleIamge_triggered()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Load sample image"),
		"./resource/meshes",
		tr("*.png *.jpg *.bmp"));

	QFileInfo fi(fileName);
	//	resetSampleSet();
	if (fileName.size() != 0)
	{
		Cuda_ctrl::_image_ctrl.add_example_image(
			(fi.canonicalPath() + "/").toLocal8Bit().constData(),
			fi.completeBaseName().toLocal8Bit().constData());
		statusBar()->showMessage(tr("sampleIamge loaded"), 2000);
	}
	else
		return;
	createTreeWidgetItems();
	m_layer->updateTable();
	QApplication::setOverrideCursor(Qt::WaitCursor);
	//textEdit->setPlainText(in.readAll());
	QApplication::restoreOverrideCursor();
}

void main_window::on_actionSave_as_ISM_triggered(){}
void main_window::on_actionSave_as_FBX_triggered(){}
void main_window::on_actionSave_as_mesh_triggered()
{
	if( !Cuda_ctrl::is_mesh_loaded() ){
		QMessageBox::information(this, "Error", "No mesh to be saved.");
		return;
	}

	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Save mesh"),
		"./resource/meshes",
		tr("*.off *obj") );

	Diag_ok_cancel diag("Invert index ?",
		"Do you want to invert the mesh index",
		this);

	if( fileName.size() != 0 )
	{
		QFileInfo fi(fileName);
		QString ext = fi.suffix().toLower();
		if(ext == "off")
			g_mesh->export_off(fileName.toLatin1(), diag.exec());
		else if( ext == "obj" )
		{
			Loader::Abs_mesh abs_mesh;
			Mesh_utils::save_mesh(*g_mesh, abs_mesh );
			Loader::Obj_file loader;
			loader.set_mesh( abs_mesh );
			loader.export_file( fileName.toLocal8Bit().constData() );
		}
		else
			QMessageBox::information(this, "Error !", "unsupported ext: '"+ext+"' \n");
	}


}
void main_window::on_actionSave_as_skeleton_triggered()
{
	if( !Cuda_ctrl::is_skeleton_loaded() ){
		QMessageBox::information(this, "Error", "No skeleton to be saved");
		return;
	}

	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Save skeleton"),
		"./resource/meshes",
		tr("*.skel") );
	if( fileName.size() != 0)
		Cuda_ctrl::_graph.save_to_file(fileName.toLatin1());


}
void main_window::on_actionSave_weights_triggered()
{
	if( !Cuda_ctrl::is_animesh_loaded() ){
		QMessageBox::information(this, "Error", "No animated mesh loaded");
		return;
	}

	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Save weights"),
		"./resource/meshes",
		tr("*.weights") );
	if( fileName.size() != 0)
		Cuda_ctrl::_anim_mesh->save_weights(fileName.toLatin1());


}
void main_window::on_actionSave_keyframes_triggered(){}
void main_window::on_actionSave_cluster_triggered(){}
void main_window::on_actionSave_pose_triggered(){}
void main_window::on_actionSave_camera_triggered(){}

void main_window::on_actionSkeleton_triggered()
{
	if(g_mesh != 0)
	{
		Skeleton_ctrl& skel = Cuda_ctrl::_skeleton;
		QMessageBox::information(this,
			"Skeleton informations",
			"Nb joint: "+QString::number(skel.get_nb_joints())+"\n"+
			"Hierachy: \n"+
			QString( g_skel->to_string().c_str() )
			);
	}
	else
	{
		QMessageBox::information(this, "Skeleton informations", "No skeleton to get infos from");
	}
}

void main_window::on_actionNew_VideoEditing_Scene_triggered()
{
	VideoEditingWindow& videoEditingWindow = VideoEditingWindow::getInstance();
	videoEditingWindow.show();
}
void main_window::on_actionOpen_VideoEditing_Scene_triggered()
{
	VideoEditingWindow& videoEditingWindow = VideoEditingWindow::getInstance();
	videoEditingWindow.show();
}
void main_window::on_actionSave_VideoEditing_Scene_triggered()
{
	VideoEditingWindow& videoEditingWindow = VideoEditingWindow::getInstance();
	videoEditingWindow.show();
}