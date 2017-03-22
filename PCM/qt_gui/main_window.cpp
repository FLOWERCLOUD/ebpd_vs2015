
#include "main_window.h"
#include "paint_canvas.h"
#include "manipulate_tool.h"
#include "toolbars/widget_selection.hpp"
#include "toolbars/widget_viewports.hpp"
#include "toolbars/widget_gizmo.hpp"
#include "toolbars/widget_render_mode.hpp"
#include "toolbars/widget_fitting.hpp"
#include "OGL_viewports_skin2.hpp"
#include "toolbars/OGL_widget_enum.hpp"
#include "file_system.h"
#include "sample.h"
#include "sample_set.h"
#include "vertex.h"
#include "file_io.h"
#include "color_table.h"
#include "time.h"
#include "tracer.h"
#include "sample_properity.h"
#include "maching_state.h"
#include "saveSnapshotDialog.h"
#include "savePLYDialog.h"
#include "GLLogStream.h"
#include "rendering/render_types.h"
#include "SSDR.h"
#include "GlobalObject.h"
#include "control/cuda_ctrl.hpp"
#include <QMessageBox>
#include <QFileDialog>
#include <QLabel>
#include <QStatusBar>
#include <QSettings>
#include <QCloseEvent>
#include <QPlainTextEdit>
#include <QAbstractItemModel>
#include <QStandardItemModel>
#include <QGroupBox>
#include <QColorDialog>
#include <QComboBox>
using namespace qglviewer;
using namespace std;
using namespace ANIMATION;
//add by huayun
bool ifGraphBoxVisible = 0;
bool ifEdgeVertexVisible = 0;
IndexType LabelGraphDepth = 0;
#include <QtWidgets/QSplitter>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QSizePolicy>
#include <QtWidgets/QProgressDialog>
#include <sstream>
#include <QWaitCondition>
// 全局条件变量
QWaitCondition mWaitcond;
GLLogStream logstream;
int REAL_TIME_RENDER = 0;
extern RenderMode::WhichColorMode	which_color_mode_;
extern RenderMode::RenderType which_render_mode;
extern bool isShowNoraml ;

bool isBulletRun = false;
bool isSSDRRun = false;
bool isAnimationRun = false;
bool isCameraViewerOn = false;
main_window::main_window(QWidget *parent)
	: QMainWindow(parent),
	cur_select_sample_idx_(-1),
	last_select_sample_idx_(-1),
	is_mutiview(false),
	frameTimer( NULL),
	_viewports(NULL),
	single_operate_tool_( NULL)
{
	globalObjectsInit();	
	ui.setupUi(this);
	// Desactivate GUI parts
	enable_animesh( false );
	enable_mesh   ( false );

	//QGLFormat format = QGLFormat::defaultFormat();
	//format.setSampleBuffers(true);
	//format.setSamples(8);
//	main_canvas_ = new PaintCanvas(format, this);

//	setCentralWidget(main_canvas_);
	//add Multiple widgets on a QDockWidget

	setupDockWidget();	
	createAction();
	createStatusBar();
	setup_toolbar();
	setup_toolbar_painting();
	setup_toolbar_frame();
	setup_viewports();


	setWindowTitle("PCM");

	setContextMenuPolicy(Qt::CustomContextMenu);
//	setWindowState(Qt::WindowMaximized);

	setFocusPolicy(Qt::ClickFocus);

}


void main_window::resetSampleSet()
{
	cur_import_files_attr_.clear();
	cur_select_sample_idx_ = last_select_sample_idx_ = -1;
	for( int i = 0 ;i<(*Global_SampleSet).size();++i)
	{

	}
	(*Global_SampleSet).clear();
}


void main_window::setupDockWidget()
{
	QDockWidget* layerDock = new QDockWidget(  QString(" LAYERDOCK") , this);
	ui.LayerSpinBox->setMinimumSize(QSize(50, 38));
	ui.LayerSpinBox->setMaximumWidth(300);
	ui.centerframe->setMinimumSize(QSize(50, 38));
	ui.centerframe->setMaximumWidth(300);
	//layerDock->setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding ) );

	QWidget* multiwidget = new QWidget();
	multiwidget->setMaximumHeight(150);
	QVBoxLayout* layerlayout = new QVBoxLayout();
	QHBoxLayout* LayerDialoglayout = new QHBoxLayout();
	QHBoxLayout* centetframelayout = new QHBoxLayout();
	QHBoxLayout* trj_labelLayout = new QHBoxLayout();
	QHBoxLayout* frame_labelLayout = new QHBoxLayout();
	LayerDialoglayout->addWidget(new QLabel("layer") );
	LayerDialoglayout->addWidget(ui.LayerSpinBox);
	centetframelayout->addWidget(new QLabel("center frame"));
	centetframelayout->addWidget(ui.centerframe);
	trj_labelLayout->addWidget(ui.text_trajectory_label);
	trj_labelLayout->addWidget(ui.button_traj_label);
	frame_labelLayout->addWidget(ui.showlabel_lineEdit);
	frame_labelLayout->addWidget(ui.show_label_Button);

	layerlayout->addLayout(LayerDialoglayout);
	layerlayout->addLayout(centetframelayout);
	layerlayout->addLayout(trj_labelLayout);
	layerlayout->addLayout(frame_labelLayout);
	multiwidget->setLayout(layerlayout);
	layerDock->setWidget(multiwidget);
	this->addDockWidget(static_cast<Qt::DockWidgetArea>(1), layerDock);

	//Codes for initializing treeWidget
	QStringList headerLabels;
	headerLabels <<"Index"<< "Name" << "VtxNum" ;//<< "#Point";
	ui.treeWidget->setHeaderLabels(headerLabels);
	ui.treeWidget->setColumnWidth( 0 ,50);
	ui.treeWidget->setColumnWidth( 1,50 );
	ui.treeWidget->setColumnWidth( 2,50 );
	connect(ui.treeWidget, SIGNAL(itemClicked( QTreeWidgetItem * , int  ) ),
		this, SLOT(selectedSampleChanged( QTreeWidgetItem * , int  )) );

	//added by huayun
	m_layer = new LayerDialog(this);
	m_layer->setAllowedAreas( Qt::RightDockWidgetArea | Qt::LeftDockWidgetArea);
	addDockWidget( Qt::RightDockWidgetArea, m_layer);
	m_layer->setVisible(true);
}



void main_window::createAction()
{
	createFileMenuAction();
	createPaintSettingAction();
	createAlgorithmAction();
	createToolAction();

}


void main_window::createAlgorithmAction()
{
	connect(ui.actionClustering, SIGNAL(triggered()), this, SLOT(doClustering()));
	connect(ui.actionRegister,SIGNAL(triggered()),this,SLOT(doRegister()));
	//connect(ui.actionSpectral_Cluster,SIGNAL(triggered()),this,SLOT(doSpectralCluster()));
	connect(ui.actionGraphCut,SIGNAL(triggered()),this,SLOT(doGraphCut()));
	connect(ui.actionCalculateNorm,SIGNAL(triggered()),this,SLOT(computeSampleNormal()));
	connect(ui.actionClusterAll,SIGNAL(triggered()),this,SLOT(batchTrajClustering()));
	connect(ui.actionVisDistortion,SIGNAL(triggered()),this,SLOT(visDistor()));

	connect(ui.actionGCopti,SIGNAL(triggered()),this,SLOT(doGCOptimization()));

	connect(ui.actionPlanFit,SIGNAL(triggered()), this ,SLOT(doPlanFit() ) );

	connect( ui.actionPropagate , SIGNAL(triggered()) ,this , SLOT( doPropagate()) );
	connect( ui.actionBullet , SIGNAL(triggered()) ,this , SLOT( doBulletPhysics() ) );
	connect( ui.actionSSDR , SIGNAL(triggered()) ,this , SLOT( doSSDR()) );
	
}

void main_window::createPaintSettingAction()
{
	connect(ui.actionSet_Visible, SIGNAL(triggered()), this, SLOT(setSampleVisible()) );
	connect(ui.actionSet_Invisible, SIGNAL(triggered()), this, SLOT(setSampleInvisible()));
	connect(ui.actionObject_Color, SIGNAL(triggered()), this, SLOT(setObjectColorMode()));
	connect(ui.actionVertex_Color, SIGNAL(triggered()), this, SLOT(setVertexColorMode()));
	connect(ui.actionLabel_Color, SIGNAL(triggered()), this, SLOT(setLabelColorMode()));
	connect(ui.actionShow_Tracjectory, SIGNAL(triggered()), this, SLOT(showTracer()));
	connect(ui.actionBallvertex ,SIGNAL(triggered()), this, SLOT(setBallvertexMode()));
	connect(ui.actionShow_normal ,SIGNAL(triggered()), this, SLOT(show_normal()));
	connect(ui.actionShow_camera_viewer ,SIGNAL(triggered()), this, SLOT(toggle_camera_viewer()));
	//render mode
	connect(ui.actionPoint_mode, SIGNAL(triggered()), this, SLOT(setPointMode()) );
	connect(ui.actionFlat_mode, SIGNAL(triggered()), this, SLOT(setFlatMode()) );
	connect(ui.actionWire_mode, SIGNAL(triggered()), this, SLOT(setWireMode()) );
	connect(ui.actionFlatWire_mode, SIGNAL(triggered()), this, SLOT(setFlatWireMode()) );
	connect(ui.actionSmooth_mode, SIGNAL(triggered()), this, SLOT(setSmoothMode()) );
	connect(ui.actionTexture_mode, SIGNAL(triggered()), this, SLOT(setTextureMode()) );
	connect(ui.actionSelect_Mode_render, SIGNAL(triggered()), this, SLOT(setSelectMode()) );
	connect(ui.actionAnimate_Mode, SIGNAL(triggered()), this, SLOT(setAnimateMode()) );


	connect(ui.actionDont_Trace,SIGNAL(triggered()), this, SLOT(clearTracer()));
	connect(ui.actionShow_Graph_WrapBox ,SIGNAL(triggered()) ,this ,SLOT(chooseGraph_WrapBox_visiable() ));
	connect(ui.actionShow_EdgeVertexs ,SIGNAL(triggered()) ,this ,SLOT(chooseEdgeVertexs__visiable() ));
	//added by huayun
	//connect( ui.LayerSpinBox ,SIGNAL( triggered() ) ,this ,SLOT(layerSpinBoxChanged( 2)) );
	connect( ui.LayerSpinBox ,SIGNAL( valueChanged(int)) ,this ,SLOT(layerSpinBoxChanged(int)) );
	//connect( ui.actionButtonback, SIGNAL(triggered() ) , this ,SLOT(layerSpinBoxChanged(int)) );
	connect( ui.centerframe ,SIGNAL( valueChanged(int)) ,this ,SLOT(centerframeChanged(int)) );

	connect(ui.button_traj_label, SIGNAL(clicked()), this, SLOT(dealtarjlabel()) );
	connect(ui.show_label_Button, SIGNAL(clicked()), this, SLOT(Framelabel()) );
	connect(ui.actionWakeWorkThread ,  SIGNAL(triggered()), this, SLOT(wakeUpThread() ) );
	//connect( ui.actionButton2stop, SIGNAL(triggered() ) , this ,SLOT(layerSpinBoxChanged(int)) );
	StateManager::getInstance().setWindowrefer(this);
	connect( ui.actionButtonRunOrPause, SIGNAL(triggered() ) , this ,SLOT( runOrPause() ) );

	connect( ui.actionButton2stop, SIGNAL(triggered() ) , this ,SLOT( stop() ) );

	connect( &(StateManager::getInstance()) , SIGNAL( stateChanged(ANIMATION::AState,ANIMATION::AState ) ) ,
		this ,SLOT( mediaStateChanged(ANIMATION::AState ,ANIMATION::AState) ) );

	connect( ui.actionButtonback, SIGNAL(triggered() ) , this ,SLOT( increaseInterval() ) );
	connect( ui.actionButtonadvance, SIGNAL(triggered() ) , this ,SLOT( decreaseInterval()) );
	//connect( ui.actionButtonadvance, SIGNAL(triggered() ) , this ,SLOT(layerSpinBoxChanged(int)) );

}

void main_window::createToolAction()
{
	connect( ui.actionSelect_Mode, SIGNAL(triggered()), this, SLOT(setSelectToolMode()) );
	connect( ui.actionScene_Mode, SIGNAL(triggered()),this, SLOT(setSceneToolMode()));
	connect(ui.actionPaint_Mode, SIGNAL(triggered()), this, SLOT(setPaintMode()));
	connect(ui.actionEbpd_hand_mode, SIGNAL(triggered()), this, SLOT(setEbpd_hand_Mode()));
}

void main_window::setObjectColorMode()
{
	which_color_mode_ =  RenderMode::OBJECT_COLOR;
	updateGL();
}

void main_window::setVertexColorMode()
{
	which_color_mode_ =  RenderMode::VERTEX_COLOR;
	updateGL();
}

void main_window::setLabelColorMode()
{
	which_color_mode_ = RenderMode::LABEL_COLOR;
	updateGL();
}
void main_window::setBallvertexMode()
{
	which_color_mode_ = RenderMode::SphereMode;
	updateGL();
}
void main_window::show_normal()
{
	if(isShowNoraml) 
	{
		isShowNoraml = false;
	}else
	{
		isShowNoraml = true;
	}
	updateGL();
}

void main_window::toggle_camera_viewer()
{
	isCameraViewerOn = !isCameraViewerOn;
	if( isCameraViewerOn)
	{
		ui.actionShow_camera_viewer->setText("CameraViewer On");
		_viewports->toggleCameraViewer();
	}else
	{
		ui.actionShow_camera_viewer->setText("isCameraViewer Off");
		_viewports->toggleCameraViewer();
	}
	

}
void main_window::setSelectToolMode()
{
	if (cur_select_sample_idx_==-1)
	{
		cout << " not select frame" << std::endl;
		return;
	}

	if (_viewports->active_viewport()->single_operate_tool_)
	{
		delete _viewports->active_viewport()->single_operate_tool_;
	}
	_viewports->active_viewport()->single_operate_tool_ = new SelectTool(_viewports->active_viewport());
	((SelectTool*)_viewports->active_viewport()->single_operate_tool_)->set_tool_type(Tool::SELECT_TOOL);
	((SelectTool*)_viewports->active_viewport()->single_operate_tool_)->set_cur_smaple_to_operate(cur_select_sample_idx_);

	updateGL();

}

void main_window::setSceneToolMode()
{
	_viewports->active_viewport()->single_operate_tool_->set_tool_type(Tool::EMPTY_TOOL);
	updateGL();
}

void main_window::setPaintMode()
{


}
void main_window::setEbpd_hand_Mode()
{
	if (cur_select_sample_idx_ == -1)
	{
		cout << " not select frame" << std::endl;
		return;
	}
	if (_viewports->active_viewport()->single_operate_tool_)
	{
		delete _viewports->active_viewport()->single_operate_tool_;
	}
	_viewports->active_viewport()->single_operate_tool_ = new ManipulateTool(_viewports->active_viewport());
	((ManipulateTool*)_viewports->active_viewport()->single_operate_tool_)->set_tool_type(Tool::MANIPULATE_TOOL);
	((ManipulateTool*)_viewports->active_viewport()->single_operate_tool_)->set_cur_smaple_to_operate(cur_select_sample_idx_);
	updateGL();

}

void main_window::setPointMode()
{
	which_render_mode = RenderMode::PointMode;
	updateGL();
};
void main_window::setFlatMode()
{
	which_render_mode = RenderMode::FlatMode;
	updateGL();
};
void main_window::setWireMode()
{
	which_render_mode = RenderMode::WireMode;
	updateGL();
};
void main_window::setFlatWireMode()
{
	which_render_mode = RenderMode::FlatWireMode;
	updateGL();
};
void main_window::setSmoothMode()
{
	which_render_mode = RenderMode::PointMode;
	updateGL();
};
void main_window::setTextureMode()
{
	which_render_mode = RenderMode::TextureMode;
	updateGL();
};
void main_window::setSelectMode()
{
	which_render_mode = RenderMode::SelectMode;
	updateGL();
};

void main_window::setAnimateMode()
{
	isAnimationRun = !isAnimationRun;
	if( isAnimationRun)
	{
		ui.actionAnimate_Mode->setText("animation is run");
	}else
	{
		ui.actionAnimate_Mode->setText("animation is stop");
	}
	viewport_toolb_single();
}

void main_window::createFileMenuAction()
{
	for (int i = 0; i < MaxRecentFiles; ++i) {
		recentFileActs[i] = new QAction(this);
		recentFileActs[i]->setVisible(false);
		connect(recentFileActs[i], SIGNAL(triggered()),
			this, SLOT(openRecentFile()));
	}
	separatorAct = ui.menuFiles->addSeparator();
	for (int i = 0; i < MaxRecentFiles; ++i)
		ui.menuFiles->addAction(recentFileActs[i]);
	ui.menuFiles->addSeparator();

	updateRecentFileActions();
	connect(ui.actionImportFiles, SIGNAL(triggered()),this, SLOT(openFiles()));
	connect(ui.actionImportFiles_Lazy, SIGNAL(triggered()),this, SLOT(openFilesLazy()));
	connect(ui.actionSaveSnapshot ,SIGNAL(triggered()) , this ,SLOT(saveSnapshot()));
	connect(ui.actionSavePly ,SIGNAL(triggered()) , this ,SLOT(savePLY()));
	connect(ui.actionsaveLabelFile ,SIGNAL(triggered()) , this ,SLOT(saveLabelFile()));
	connect(ui.actionGetlabel_from_file ,SIGNAL(triggered()) , this ,SLOT(getLabelFromFile()));
}

bool main_window::openFile()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Import point cloud from file"), ".",
		tr(
		"Ascii Point cloud (*.lab *.xyz *.pwn *.pcd)\n"
		"All files (*.*)")
		);

	if (fileName.isEmpty())
		return false;

	return true;
}


bool main_window::setSampleVisible()
{
	(*Global_SampleSet)[cur_select_sample_idx_].set_visble(true);
	m_layer->updateTable(cur_select_sample_idx_);
	updateGL();
	return true;
}

bool main_window::setSampleInvisible()
{
	(*Global_SampleSet)[cur_select_sample_idx_].set_visble(false);
	m_layer->updateTable(cur_select_sample_idx_);
	updateGL();
	return true;
}

void main_window::selectedSampleChanged(QTreeWidgetItem * item, int column)
{
	last_select_sample_idx_ = cur_select_sample_idx_;
	cur_select_sample_idx_ = item->text(0).toInt();

	//change the active frame


	if (last_select_sample_idx_!= -1)
	{
		(*Global_SampleSet)[last_select_sample_idx_].set_selected(false);
	}
	if ( cur_select_sample_idx_ != -1)
	{
		(*Global_SampleSet)[cur_select_sample_idx_].set_selected(true);
	}
	if (cur_select_sample_idx_ != -1)
	{
		setSampleSelectedIndex(cur_select_sample_idx_);
	}		
	updateGL();

}

void main_window::showTracer()
{
	_viewports->active_viewport()->setTracerShowOrNot(true);
	updateGL();
}

void main_window::clearTracer()
{
	_viewports->active_viewport()->setTracerShowOrNot(false);
	Tracer::get_instance().clear_records();
	updateGL();
}
//bool ifGraphBoxVisible = 0;
//bool ifEdgeVertixVisible = 0;

//QIcon icon;
//icon.addFile(QString::fromUtf8("Resources/openFile.png"), QSize(), QIcon::Normal, QIcon::Off);
//actionImportFiles->setIcon(icon);
void main_window::layerSpinBoxChanged(int i)
{
//	Logger<<"layer 值改变  "<< i <<std::endl;
	if( i != LabelGraphDepth){
		LabelGraphDepth = i;
		(*Global_SampleSet)[cur_select_sample_idx_].clayerDepth_ = i;
		emit LabelGraphDepthChanged(i);
	} 
	updateGL();

}
void main_window::centerframeChanged(int i)
{
//	Logger<<"centerframe 值改变  "<< i <<std::endl;
	_viewports->active_viewport()->centerframeNum = i;
	updateGL();
}
void main_window::setSampleSelectedIndex(int i)
{
	last_select_sample_idx_ = cur_select_sample_idx_;
	cur_select_sample_idx_ =  i;


	if (last_select_sample_idx_!= -1)
	{
		(*Global_SampleSet)[last_select_sample_idx_].set_selected(false);
	}
	if ( cur_select_sample_idx_ != -1)
	{
		(*Global_SampleSet)[cur_select_sample_idx_].set_selected(true);
	}
	if (_viewports->active_viewport()->single_operate_tool_ != NULL &&_viewports->active_viewport()->single_operate_tool_->tool_type() == Tool::MANIPULATE_TOOL)
	{
		((ManipulateTool*)_viewports->active_viewport()->single_operate_tool_)->set_cur_smaple_to_operate(cur_select_sample_idx_);
	}
	updateGL();
}
void main_window::dealtarjlabel()
{
	std::stringstream istring(ui.text_trajectory_label->text().toLocal8Bit().constData());
	static int labelidx = 0;
	Logger<<"dealtarjlabel  "<<std::endl;

	std::vector<int> frame;

	frame.push_back(_viewports->active_viewport()->centerframeNum);
	/*frame.push_back(52);
	frame.push_back(53);
	frame.push_back(54);*/
	//std::vector<int> traj_label_vec;
	
	int label;
	traj_label_vec.clear();
	while(istring>>label){
		traj_label_vec.push_back(label);
	};

	updateGL();
}
void main_window::Framelabel()
{
	std::stringstream istring(ui.showlabel_lineEdit->text().toLocal8Bit().constData());
	static int labelidx = 0;
	Logger<<"framelabel  "<<std::endl;

//	std::vector<int> frame;

//	frame.push_back(main_canvas_->centerframeNum);
	/*frame.push_back(52);
	frame.push_back(53);
	frame.push_back(54);*/
	//std::vector<int> traj_label_vec;
	
	int label;
	std::vector<int> frame_label;
	frame_label.clear();
	while(istring>>label){
		frame_label.push_back(label);
	};

	_viewports->active_viewport()->showSelectedFrameLabel(frame_label ,cur_select_sample_idx_);

	updateGL();
}
//connect( ui.actionButtonback, SIGNAL(triggered() ) , this ,SLOT(layerSpinBoxChanged(int)) );
//connect( ui.actionButton2stop, SIGNAL(triggered() ) , this ,SLOT(layerSpinBoxChanged(int)) );
//connect( ui.actionButtonRunOrPause, SIGNAL(triggered() ) , this ,SLOT(layerSpinBoxChanged(int)) );
//connect( ui.actionButtonadvance, SIGNAL(triggered() ) , this ,SLOT(layerSpinBoxChanged(int)) );

void main_window::processButton()
{

}

void main_window::processButtonBack()
{

}
void main_window::processButton2Stop()
{

}

void main_window::processButtonadvance()
{

}





void  main_window::chooseGraph_WrapBox_visiable()
{
	std::cout<<"Show_Graph_WrapBox"<<std::endl;
	if( ifGraphBoxVisible){
		std::cout<<"Visible"<<ifGraphBoxVisible<<std::endl;
		ifGraphBoxVisible = 0;
		QIcon icon;
		icon.addFile(QString::fromUtf8("Resources/nnolinkNode.png") ,QSize() , QIcon::Normal ,QIcon::Off);
		ui.actionShow_Graph_WrapBox->setIcon(icon);
		ui.actionShow_Graph_WrapBox->setText(QApplication::translate("main_windowClass", "Show Graph WrapBox", 0));
		unShow_Graph_WrapBox();
	}else{
		std::cout<<"Visible"<<ifGraphBoxVisible<<std::endl;
		ifGraphBoxVisible =1;
		QIcon icon;
		icon.addFile(QString::fromUtf8("Resources/linkNode.png") ,QSize() , QIcon::Normal ,QIcon::Off);
		ui.actionShow_Graph_WrapBox->setIcon(icon);
		ui.actionShow_Graph_WrapBox->setText(QApplication::translate("main_windowClass", "unShow Graph WrapBox", 0));
		Show_Graph_WrapBox();
	}
}
void  main_window::Show_Graph_WrapBox()
{

	std::cout<<"Show_Graph_WrapBox"<<std::endl;

	which_color_mode_ =  RenderMode::WrapBoxColorMode;
	//main_canvas_->which_color_mode_ = PaintCanvas::SphereMode;
	_viewports->active_viewport()->setGraph_WrapBoxShowOrNot(true);

	updateGL();
}
void  main_window::unShow_Graph_WrapBox()
{
	std::cout<<"unShow_Graph_WrapBox"<<std::endl;

	which_color_mode_ =  RenderMode::LABEL_COLOR;
	_viewports->active_viewport()->setGraph_WrapBoxShowOrNot(false);
	//Tracer::get_instance().clear_records();
	updateGL();
}
//  actionSet_Invisible->setText(QApplication::translate("main_windowClass", "Set Invisible", 0, QApplication::UnicodeUTF8));
void  main_window::chooseEdgeVertexs__visiable()
{
	std::cout<<"Show_EdgeVertexs"<<std::endl;
	if( ifEdgeVertexVisible){
		std::cout<<"Visible"<<ifEdgeVertexVisible<<std::endl;
		ifEdgeVertexVisible = 0;
		QIcon icon;
		icon.addFile(QString::fromUtf8("Resources/NoedgeVertexs.png") ,QSize() , QIcon::Normal ,QIcon::Off);
		ui.actionShow_EdgeVertexs->setIcon(icon);
		ui.actionShow_EdgeVertexs->setText(QApplication::translate("main_windowClass", "Show EdgeVertexs", 0));
		unShow_EdgeVertexs();
	}else{
		std::cout<<"Visible"<<ifEdgeVertexVisible<<std::endl;
		ifEdgeVertexVisible =1;
		QIcon icon;
		icon.addFile(QString::fromUtf8("Resources/edgeVertexs.png") ,QSize() , QIcon::Normal ,QIcon::Off);
		ui.actionShow_EdgeVertexs->setIcon(icon);
		ui.actionShow_EdgeVertexs->setText(QApplication::translate("main_windowClass", "unShow EdgeVertexs", 0));
		Show_EdgeVertexs();
	}
}
void  main_window::Show_EdgeVertexs()
{
	//	main_canvas_->which_color_mode_ = PaintCanvas::LABEL_COLOR;
	//updateGL();
	which_color_mode_ =  RenderMode::EdgePointColorMode;
	_viewports->active_viewport()->setEdgeVertexsShowOrNot(true);
	updateGL();
}
void  main_window::unShow_EdgeVertexs()
{	
	which_color_mode_ =  RenderMode::LABEL_COLOR;
	_viewports->active_viewport()->setEdgeVertexsShowOrNot(false);
	//Tracer::get_instance().clear_records();
	updateGL();

}

bool main_window::openFiles()
{
	QSettings settings;
	QStringList files = settings.value("recentFileList").toStringList();
	QString dir;
	if(files.size() >0){
		dir = QFileDialog::getExistingDirectory(this,tr("Import point cloud files"), files[0]);
	}
	else{
		dir = QFileDialog::getExistingDirectory(this,tr("Import point cloud files") ,QString("./") );
	}

	if (dir.isEmpty())
		return false;
	setCurrentFile(dir);

	resetSampleSet();

	QDir file_dir(dir);
	if ( !file_dir.exists() )
	{
		return false;
	}
	file_dir.setFilter(QDir::Files);

	QFileInfoList file_list = file_dir.entryInfoList();
	loadFileToSample(file_list ,false);

	createTreeWidgetItems();
	m_layer->updateTable();
	QApplication::setOverrideCursor(Qt::WaitCursor);
	//textEdit->setPlainText(in.readAll());
	QApplication::restoreOverrideCursor();
	//	setCurrentFile(fileName);
	statusBar()->showMessage(tr("File loaded"), 2000);

	return true;
}

bool main_window::openFilesLazy()
{
	QSettings settings;
	QStringList files = settings.value("recentFileList").toStringList();
	QString dir;
	if(files.size() >0){
		dir = QFileDialog::getExistingDirectory(this,tr("Import point cloud files"), files[0]);
	}
	else{
		dir = QFileDialog::getExistingDirectory(this,tr("Import point cloud files") ,QString("./") );
	}

	if (dir.isEmpty())
		return false;
	setCurrentFile(dir);

	resetSampleSet();

	QDir file_dir(dir);
	if ( !file_dir.exists() )
	{
		return false;
	}
	file_dir.setFilter(QDir::Files);

	QFileInfoList file_list = file_dir.entryInfoList();

	loadFileToSample(file_list ,true);

	createTreeWidgetItems();
	m_layer->updateTable();
	QApplication::setOverrideCursor(Qt::WaitCursor);
	//textEdit->setPlainText(in.readAll());
	QApplication::restoreOverrideCursor();
	//	setCurrentFile(fileName);
	statusBar()->showMessage(tr("File loaded"), 2000);
	return true;
}
void main_window::setup_toolbar()
{
	// Connect toolButtons for selection
	Widget_selection* select = ui.toolBar->_wgt_select;
	QObject::connect(select->toolB_select_point, SIGNAL(pressed()), this, SLOT(selection_toolb_point()));
	QObject::connect(select->toolB_select_circle, SIGNAL(pressed()), this, SLOT(selection_toolb_circle()));
	QObject::connect(select->toolB_select_square, SIGNAL(pressed()), this, SLOT(selection_toolb_square()));

	Widget_viewports* view = ui.toolBar->_wgt_viewport;
	QObject::connect(view->toolB_single, SIGNAL(pressed()), this, SLOT(viewport_toolb_single()));
	QObject::connect(view->toolB_doubleV, SIGNAL(pressed()), this, SLOT(viewport_toolb_doubleV()));
	QObject::connect(view->toolB_doubleH, SIGNAL(pressed()), this, SLOT(viewport_toolb_doubleH()));
	QObject::connect(view->toolB_four, SIGNAL(pressed()), this, SLOT(viewport_toolb_four()));

	// Connect toolButtons for gizmo
	Widget_gizmo* gizmo = ui.toolBar->_wgt_gizmo;
	QObject::connect(gizmo->toolB_show_gizmo, SIGNAL(toggled(bool)), this, SLOT(show_all_gizmo(bool)));
	QObject::connect(gizmo->toolB_translate, SIGNAL(pressed()), this, SLOT(set_gizmo_trans()));
	QObject::connect(gizmo->toolB_rotate, SIGNAL(pressed()), this, SLOT(set_gizmo_rot()));
	QObject::connect(gizmo->toolB_trackball, SIGNAL(pressed()), this, SLOT(set_gizmo_trackball()));
	QObject::connect(gizmo->toolB_scale, SIGNAL(pressed()), this, SLOT(set_gizmo_scale()));

	// Connect toolButtons for the rendering mode
	Widget_render_mode* render = ui.toolBar->_wgt_rd_mode;
	QObject::connect(render->toolB_wire_transc, SIGNAL(pressed()), this, SLOT(rd_mode_toolb_wire_transc()));
	QObject::connect(render->toolB_wire, SIGNAL(pressed()), this, SLOT(rd_mode_toolb_wire()));
	QObject::connect(render->toolB_solid, SIGNAL(pressed()), this, SLOT(rd_mode_toolb_solid()));
	QObject::connect(render->toolB_tex, SIGNAL(pressed()), this, SLOT(rd_mode_toolb_tex()));

	// Fitting buton
	QObject::connect(ui.toolBar->_wgt_fit->toolB_fitting, SIGNAL(toggled(bool)), this, SLOT(toggle_fitting(bool)));

	// Connect combo box for the pivot mode
	QObject::connect(ui.toolBar->_pivot_comboBox, SIGNAL(currentIndexChanged(int)),
		this           , SLOT(pivot_comboBox_currentIndexChanged(int)));


}

void main_window::setup_toolbar_painting()
{
	/*	QObject::connect(ui.toolBar_painting->_enable_paint, SIGNAL(toggled(bool)),
	this                           , SLOT(paint_toggled(bool))); */  
}
void main_window::setup_toolbar_frame()
{
	QObject::connect(ui.toolBar_frame, SIGNAL(update_gl()),
		this         , SLOT(update_viewports()));
}

void main_window::createStatusBar()
{
	coord_underMouse_label_ = new QLabel(this);
	coord_underMouse_label_->setAlignment(Qt::AlignLeft);

	vtx_idx_underMouse_label_ = new QLabel(this);
	coord_underMouse_label_->setAlignment(Qt::AlignRight);
	
	ui.statusBar->addWidget( coord_underMouse_label_ , 1 );
	ui.statusBar->addWidget( vtx_idx_underMouse_label_, 0 );
}

void main_window::setup_viewports()
{
	_viewports = new OGL_viewports_skin2(ui.viewports_frame, this);
	ui.viewports_frame->layout()->addWidget(_viewports);

}

void main_window::createTreeWidgetItems()
{
	ui.treeWidget->clear();
	
	SampleSet& set = (*Global_SampleSet);
	for ( int sample_idx=0; sample_idx < set.size(); sample_idx++ )
	{
		//QTreeWidgetItem* item = new QTreeWidgetItem(ui.treeWidget); 
		//QTreeWidgetItem* itemA = new QTreeWidgetItem( ui.treeWidget);
		//QTreeWidgetItem* itemB = new QTreeWidgetItem(ui.treeWidget);
		//QIcon icon;
		//icon.addFile(QString::fromUtf8("Resources/invisible.png") ,QSize() , QIcon::Normal ,QIcon::Off);
		//ui.actionShow_Graph_WrapBox->setIcon(icon);

		//ColorType color = set[ sample_idx ].color();
		//itemA->setData(0,Qt::DecorationRole, icon);
		//itemB->setData(0, Qt::DisplayRole, sample_idx);
		//itemB->setData(1,Qt::DecorationRole, QColor(color(0)*255, color(1)*255, color(2)*255) );
		//itemB->setData(2, Qt::DisplayRole, set[sample_idx].num_vertices() );
		///*item->insertChild(sample_idx ,itemA);
		//item->insertChild( sample_idx ,itemB);*/
		//ui.treeWidget->insertTopLevelItem(sample_idx, itemA);
		//ui.treeWidget->insertTopLevelItem(sample_idx, itemB);

		QTreeWidgetItem* item = new QTreeWidgetItem(ui.treeWidget); 
	//	QIcon icon2;
	//	icon2.addFile(QString::fromUtf8("Resources/invisible.png") ,QSize() , QIcon::Normal ,QIcon::Off);
	//	ui.actionShow_Graph_WrapBox->setIcon(icon2);

		ColorType color = set[ sample_idx ].color();
	//	item->setData(0,Qt::DecorationRole, icon2);
		item->setData(0, Qt::DisplayRole, sample_idx);
		item->setData(1,Qt::DecorationRole, QColor(color(0)*255, color(1)*255, color(2)*255) );
		item->setData(2, Qt::DisplayRole, set[sample_idx].num_vertices() );

		ui.treeWidget->insertTopLevelItem(sample_idx, item);
	}
}

void main_window::showCoordinateAndIndexUnderMouse( const QPoint& point )
{
	
	//Mouse point info come from canvas
	bool found = false;

    qglviewer::Vec v = _viewports->active_viewport()->camera()->pointUnderPixel(point, found);
	if ( !found )
	{
		v = qglviewer::Vec();
	}
	QString coord_str = QString("XYZ = [%1, %2, %3]").arg(v.x).arg(v.y).arg(v.z);
	coord_underMouse_label_->setText(coord_str);

	IndexType idx;
	IndexType label;
	if ( !found || cur_select_sample_idx_==-1 )
	{
		idx = -1;
		label = -1;
	}
	else
	{
		Sample& cur_selected_sample = (*Global_SampleSet)[cur_select_sample_idx_];
		if(!cur_selected_sample.isLoaded())return;
		Vec4 v_pre(v.x - Paint_Param::g_step_size(0)*(cur_select_sample_idx_-_viewports->active_viewport()->centerframeNum),
			v.y - Paint_Param::g_step_size(1)*(cur_select_sample_idx_-_viewports->active_viewport()->centerframeNum),
			v.z - Paint_Param::g_step_size(2)*(cur_select_sample_idx_-_viewports->active_viewport()->centerframeNum) ,1.);
		//Necessary to do this step, convert view-sample space to world-sample space
		v_pre = cur_selected_sample.matrix_to_scene_coord().inverse() * v_pre;
		idx = cur_selected_sample.closest_vtx( PointType(v_pre(0), v_pre(1), v_pre(2)) );
		label = cur_selected_sample[idx].label();
	}
	QString idx_str = QString("VERTEX INDEX = [%1],LABEL = [%2]").arg(idx).arg(label);
	vtx_idx_underMouse_label_->setText( idx_str );

	return;
}

void main_window::doClustering()
{





}
void main_window::doRegister()
{

}
void main_window::doGraphCut()
{

}

void main_window::doGCOptimization()
{

}

void main_window::doPlanFit()
{


}

void main_window::doPropagate()
{

}

void main_window::doBulletPhysics()
{

	isBulletRun = !isBulletRun;
	if( isBulletRun )
	{
		ui.actionBullet->setText("bullet is run");
	}else
	{
		ui.actionBullet->setText("bullet is stop");
	}
	viewport_toolb_single();



}
void main_window::doSSDR()
{

	isSSDRRun = !isSSDRRun;
	if( isSSDRRun)
	{
		ui.actionSSDR->setText("SSDR is run");
	}else
	{
		ui.actionSSDR->setText("SSDR is stop");
	}
	return;
	Sample* refSample = new Sample();
	std::string ref_filepath = " ";
 
	FileIO::load_point_cloud_file(refSample , ref_filepath,FileIO::OBJ);
	using namespace  SSDR;
	SSDR::Output output;
	std::string filepath = "";
//	SSDR::GetFromFile(output,filepath);
	int vec_size = output.boneTrans.size();
	std::vector<RTransform> transfoms(vec_size);
	for (int i = 0; i < vec_size; i++)
	{
//		rtRigidToCom(output.boneTrans[i] , transfoms[i]);
	}


}

// void main_window::doSpectralCluster()
// {
// 	SpectralClusteringThread* specCla = new SpectralClusteringThread();
// 	connect(specCla,SIGNAL(finish_compute()),this,SLOT(finishSpectralCluster()));
// 	connect(specCla,SIGNAL(finished()),specCla,SLOT(deleteLater()));
// 	specCla->start();
// }
void main_window::finishClustering()
{

}
void main_window::finishRegister()
{

}
void main_window::finishGraphCut()
{

}

void main_window::finishGCotimization()
{

}


// MANUAL SLOTS ################################################################

void main_window::selection_toolb_point()
{
	Vec_viewports& list = _viewports->get_viewports();

	for(unsigned i = 0; i < list.size(); i++)
	{
		list[i]->setMouseTracking(false);
 		list[i]->set_selection(EOGL_widget::MOUSE);
	}
	update_viewports();
}

void main_window::selection_toolb_circle()
{
	Vec_viewports& list = _viewports->get_viewports();

	for(unsigned i = 0; i < list.size(); i++)
	{
		list[i]->setMouseTracking(false);

		list[i]->set_selection(EOGL_widget::CIRCLE);
		list[i]->setMouseTracking(true);
	}
	update_viewports();
}

void main_window::selection_toolb_square(){

}

void main_window::viewport_toolb_single(){
	_viewports->set_viewports_layout(OGL_viewports_skin2::SINGLE);
	_viewports->updateGL();
}

void main_window::viewport_toolb_doubleV(){
	_viewports->set_viewports_layout(OGL_viewports_skin2::VDOUBLE);
	_viewports->updateGL();
}

void main_window::viewport_toolb_doubleH(){
	_viewports->set_viewports_layout(OGL_viewports_skin2::HDOUBLE);
	_viewports->updateGL();
}

void main_window::viewport_toolb_four(){
	_viewports->set_viewports_layout(OGL_viewports_skin2::FOUR);
	_viewports->updateGL();
}

void main_window::rd_mode_toolb_tex()
{
	PaintCanvas* wgl = _viewports->active_viewport();
	wgl->set_phong( true );
	wgl->set_textures( true );
	update_viewports();
}

void main_window::rd_mode_toolb_solid()
{
	PaintCanvas* wgl = _viewports->active_viewport();
	wgl->set_phong( true );
	wgl->set_textures( false );
	update_viewports();
}

void main_window::rd_mode_toolb_wire()
{
	PaintCanvas* wgl = _viewports->active_viewport();
	wgl->set_phong( false );
	Cuda_ctrl::_display.set_transparency_factor( 1.f);
	update_viewports();
}

void main_window::rd_mode_toolb_wire_transc()
{
	PaintCanvas* wgl = _viewports->active_viewport();
	Cuda_ctrl::_display.set_transparency_factor( 0.5f );
	wgl->set_phong( false );
	update_viewports();
}

void main_window::show_all_gizmo(bool checked)
{
	_viewports->show_gizmo(checked);
	update_viewports();
}

void main_window::set_gizmo_trans(){
	_viewports->set_gizmo(Gizmo::TRANSLATION);
}

void main_window::set_gizmo_rot(){
	_viewports->set_gizmo(Gizmo::ROTATION);
}

void main_window::set_gizmo_trackball(){
	_viewports->set_gizmo(Gizmo::TRACKBALL);
}

void main_window::set_gizmo_scale(){
	_viewports->set_gizmo(Gizmo::SCALE);
}

void main_window::toggle_fitting(bool checked){
	//Cuda_ctrl::_anim_mesh->set_implicit_skinning(checked);
	update_viewports();
}

void main_window::pivot_comboBox_currentIndexChanged(int idx)
{
	//int val = toolBar->_pivot_comboBox->itemData( idx ).toInt();
	//_viewports->set_pivot_mode((EOGL_widget::Pivot_t)val);
}

void main_window::active_viewport(int id)
{
	static int id_prev = -1;
	// Update necessary only if the active viewport is changed
	if(id_prev == id) return;

	PaintCanvas* wgl = _viewports->active_viewport();

	// update the pannel buttons:
	//settings_raytracing->enable_raytracing->setChecked( wgl->raytrace() );

	id_prev = id;
}
// void main_window::finishDoPlanFit()
// {
// 
// }

// void main_window::finishSpectralCluster()
// {
// 
// }
main_window::~main_window()
{
	if(_viewports) delete _viewports;
	if(single_operate_tool_) delete single_operate_tool_;

	(*Global_SampleSet).clear();
	if( frameTimer != NULL){
		delete frameTimer;
		frameTimer = NULL;
	}
	globalObjectsDelete();

}
void main_window::computeSampleNormal()
{
	auto camera_look_at = _viewports->active_viewport()->camera()->viewDirection();
	//SampleManipulation::computerMinMax(cur_select_sample_idx_);
	SampleManipulation::compute_normal_all( NormalType(-camera_look_at.x, -camera_look_at.y, -camera_look_at.z));
	//SampleManipulation::compute_normal( cur_select_sample_idx_ ,NormalType(-camera_look_at.x, -camera_look_at.y, -camera_look_at.z));
	//SampleManipulation::compute_normal( cur_select_sample_idx_ ,NormalType(-1.0,0.0,0.0));
}

void main_window::batchTrajClustering()
{

}

void main_window::visDistor()
{

}

void main_window::iterateTrajClustering()
{

}
using namespace ANIMATION;
bool main_window::runOrPause()
{
	if(  StateManager::getInstance().state() ==  RUNSTATE){
//	Logger<<" old state is runstate"<<std::endl;
		StateManager::getInstance().pause();
		 

	}else if(StateManager::getInstance().state() ==  PAUSESTATE){
//		Logger<<" old state is pause state"<<std::endl;
		StateManager::getInstance().run();
	}else if(StateManager::getInstance().state() ==  STOPSTATE){
//		Logger<<" old state is stop state"<<std::endl;
		StateManager::getInstance().run();
	}
	return false;
}
bool main_window::stop()
{
	if(StateManager::getInstance().state() ==  RUNSTATE){
//		Logger<<" old state is runstate"<<std::endl;
		StateManager::getInstance().stop();
	}else if (StateManager::getInstance().state() ==  PAUSESTATE){
//		Logger<<" old state is pause state"<<std::endl;

	}else{


	}
	return false;
}



bool main_window::mediaStateChanged(ANIMATION::AState _newState ,ANIMATION::AState _oldState)
{
	Logger<<"mediaStateChanged"<< std::endl;
	switch( _newState)
	{
	
	case ANIMATION::RUNSTATE:{
		QIcon icon;
		icon.addFile(QString::fromUtf8("Resources/buttonrun2pause.png") ,QSize() , QIcon::Normal ,QIcon::Off);
		ui.actionButtonRunOrPause->setIcon(icon);
	 
		
		break;
							 }
	case ANIMATION::PAUSESTATE:{
		QIcon icon;
		icon.addFile(QString::fromUtf8("Resources/buttonstop2run.png") ,QSize() , QIcon::Normal ,QIcon::Off);
		ui.actionButtonRunOrPause->setIcon(icon);
 

		break;
							   }
	case ANIMATION::STOPSTATE:{
		QIcon icon;
		icon.addFile(QString::fromUtf8("Resources/buttonstop2run.png") ,QSize() , QIcon::Normal ,QIcon::Off);
		ui.actionButtonRunOrPause->setIcon(icon);
		
		break;
							  }
	case ANIMATION::ERRORSTATE:{
		break;
							   }
	default:
		break;


	}
	return false;
}

void main_window::processButtonRunOrPuase()
{
	bool i;
	//创建定时器
	if( NULL == frameTimer ){
		frameTimer = new QTimer(this);

		i =connect( frameTimer, SIGNAL(timeout()), this, SLOT( excFrameAnimation()) );
		frameTimer->start(50);
	}
	Logger<<"曹"<<i<<std::endl;
	//if(!REAL_TIME_RENDER)
}

void main_window::excFrameAnimation()
{

	updateGL();
}

bool main_window::increaseInterval()
{
	if(  StateManager::getInstance().state() ==  RUNSTATE){
//		Logger<<" add duration"<<std::endl;
		StateManager::getInstance().increaseInterval();


	}else if(StateManager::getInstance().state() ==  PAUSESTATE){
//		Logger<<" add duration"<<std::endl;
		StateManager::getInstance().increaseInterval();
	}else if(StateManager::getInstance().state() ==  STOPSTATE){
//		Logger<<" old state is stop state"<<std::endl;
	}
	return false;
}

bool main_window::decreaseInterval()
{
	std::cout<<"decreaseInterval()"<<std::endl;
	if(  StateManager::getInstance().state() ==  RUNSTATE){
//		Logger<<" lessenduration"<<std::endl;
		StateManager::getInstance().decreaseInterval();


	}else if(StateManager::getInstance().state() ==  PAUSESTATE){
//		Logger<<" lessenduration"<<std::endl;
		StateManager::getInstance().decreaseInterval();
	}else if(StateManager::getInstance().state() ==  STOPSTATE){
//		Logger<<"lessenduration"<<std::endl;
		
	}
	return false;
}

bool main_window::saveSnapshot()
{ 
	SaveSnapshotDialog dialog(this);

	dialog.setValues(*getActivedCanvas()->ss);

	if (dialog.exec()==QDialog::Accepted)
	{
	*getActivedCanvas()->ss=dialog.getValues();
	getActivedCanvas()->saveSnapshot();

	// if user ask to add the snapshot to raster layers
	/*
	if(dialog.addToRasters())
	{
	  QString savedfile = QString("%1/%2%3.png")
	.arg(GLA()->ss.outdir).arg(GLA()->ss.basename)
	.arg(GLA()->ss.counter,2,10,QChar('0'));

	  importRaster(savedfile);
	}
	*/
	return true;
	}

	return false;
}
bool main_window::savePLY()
{
	SavePlyDialog dialog(this);

	dialog.setValues(*getActivedCanvas()->splys);

	if (dialog.exec()==QDialog::Accepted)
	{
		
		getActivedCanvas()->savePLY( dialog.getValues());

	// if user ask to add the snapshot to raster layers
	/*
	if(dialog.addToRasters())
	{
	  QString savedfile = QString("%1/%2%3.png")
	.arg(GLA()->ss.outdir).arg(GLA()->ss.basename)
	.arg(GLA()->ss.counter,2,10,QChar('0'));

	  importRaster(savedfile);
	}
	*/
	return true;
	}

	return false; 
}
bool main_window::saveLabelFile()
{
	QSettings settings;
	QString path ;
	QStringList files = settings.value("recentFileList").toStringList();
	QString dir;
	if(files.size() >0){
		path = QFileDialog::getSaveFileName(this,tr("save label file"),files[0]+QString("./label.seg"),tr("label Files (*.seg)"));
		//dir = QFileDialog::getExistingDirectory(this,tr("save label file"), files[0]);
	}
	else{
		path = QFileDialog::getSaveFileName(this,tr("save label file"),QString("./label.seg"),tr("label Files (*.seg)"));
	}

	if (path.isEmpty())
		return false;
	getActivedCanvas()->saveLabelFile(path.toLocal8Bit().constData(),cur_select_sample_idx_);
	return true;
}
bool main_window::getLabelFromFile()
{
	QSettings settings;
	QString path ;
	QStringList files = settings.value("recentFileList").toStringList();
	QString dir;
	if(files.size() >0){
		path = QFileDialog::getOpenFileName(this,tr("get label file"),files[0],tr("label Files (*.seg)"));
		//dir = QFileDialog::getExistingDirectory(this,tr("save label file"), files[0]);
	}
	else{
		path = QFileDialog::getOpenFileName(this,tr("get label file"),QString("./"),tr("label Files (*.seg)"));
	}

	if (path.isEmpty())
		return false;


	getActivedCanvas()->getLabelFromFile(path.toLocal8Bit().constData(),cur_select_sample_idx_);
	return true;
}
bool main_window::wakeUpThread()
{

	mWaitcond.wakeOne();
	//REAL_TIME_RENDER =1;
	return true;

}
void main_window::logText(QString as,int level)
{
	//std::cout<<"logText"<<as.toStdString()<<std::endl;
	string tmp = as.toLocal8Bit().constData();
	char* text = (char*)tmp.c_str();
	int mode = level;
	GLLogStream::Levels lveltype;
	switch(mode){
		case 0 : lveltype  = GLLogStream::SYSTEM;break;
		case 1 : lveltype  = GLLogStream::WARNING;break;
		case 2 : lveltype  = GLLogStream::FILTER;break;
		case 3 : lveltype  = GLLogStream::DEBUG;break;
		default: lveltype  = GLLogStream::DEBUG;
	}
	logstream.Log( lveltype,text);
	m_layer->updateLog(logstream);
	//return true;

}

void main_window::openRecentFile()
{
	QAction *action = qobject_cast<QAction *>(sender());
	if (action)
		loadFile(action->data().toString());

}
void main_window::updateRecentFileActions()
{
	QSettings settings;
	QStringList files = settings.value("recentFileList").toStringList();

	int numRecentFiles = qMin(files.size(), (int)MaxRecentFiles);

	for (int i = 0; i < numRecentFiles; ++i) {
		QString text = tr("&%1 %2").arg(i + 1).arg(strippedName(files[i]));
		recentFileActs[i]->setText(text);
		recentFileActs[i]->setData(files[i]);
		recentFileActs[i]->setVisible(true);
	}
	for (int j = numRecentFiles; j < MaxRecentFiles; ++j)
		recentFileActs[j]->setVisible(false);

	separatorAct->setVisible(numRecentFiles > 0);



}
QString main_window::strippedName(const QString &fullFileName)
{
	return QFileInfo(fullFileName).fileName();
}
void main_window::loadFile(const QString &dir)
{
	//QString dir = QFileDialog::getExistingDirectory(this,tr("Import point cloud files"),".");
	if (dir.isEmpty())
		return ;
	setCurrentFile(dir);

	resetSampleSet();

	QDir file_dir(dir);
	if ( !file_dir.exists() )
	{
		QMessageBox::warning(this, tr("Recent Files"),
			tr("Cannot read file %1:\n%2.")
			.arg("ss")
			.arg("ss"));
		return;
	}
	file_dir.setFilter(QDir::Files);

	QFileInfoList file_list = file_dir.entryInfoList();
	loadFileToSample(file_list ,false);

	createTreeWidgetItems();
	m_layer->updateTable();

	//QTextStream in(&file);
	QApplication::setOverrideCursor(Qt::WaitCursor);
	//textEdit->setPlainText(in.readAll());
	QApplication::restoreOverrideCursor();
//	setCurrentFile(fileName);
	statusBar()->showMessage(tr("File loaded"), 2000);

}


void main_window::loadFileToSample( const QFileInfoList& file_list ,bool isLazy)
{
	IndexType sample_idx = 0;
	QProgressDialog progress(this);
	progress.setLabelText(tr("loading files: %1").arg(  file_list.at(0).baseName()));
	progress.setRange(0 ,file_list.size());
	progress.setModal(true);

	for (IndexType file_idx = 0; file_idx < file_list.size(); file_idx++)
	{
		QFileInfo file_info = file_list.at(file_idx);
		progress.setLabelText(tr("loading files: %1").arg(  file_list.at(file_idx).baseName()));
		progress.setValue(file_idx);
		qApp->processEvents( QEventLoop::ExcludeUserInputEvents);
		if( progress.wasCanceled())
		{
			resetSampleSet();
			createTreeWidgetItems();
			m_layer->updateTable();
			return ;
		}
		FileIO::FILE_TYPE file_type;

		if (file_info.suffix() == "xyz")
		{
			file_type = FileIO::XYZ;
		}
		else if(file_info.suffix() == "ply")
		{
			file_type = FileIO::PLY;
		}
		else if(file_info.suffix() == "obj")
		{
			file_type = FileIO::OBJ;
		}
		else
		{
			continue;
		}

		string file_path = file_info.filePath().toLocal8Bit().constData();
		cur_import_files_attr_.push_back( make_pair(FileSystem::base_name(file_path), 
			FileSystem::extension(file_path)) );

		Sample* new_sample;
		if(isLazy)
		{
			new_sample = FileIO::lazy_load_point_cloud_file(file_path, file_type);
		}else
		{
			new_sample = FileIO::load_point_cloud_file(file_path, file_type);
		}

		if (new_sample != nullptr)
		{
			if(isLazy)
			{
				new_sample->setLoaded(false);
			}else
			{
				new_sample->setLoaded(true);
			}
			new_sample->set_color( Color_Utility::span_color_from_table( file_idx ) );
			SampleSet& smpset = (*Global_SampleSet);
			smpset.push_back(new_sample);
			new_sample->smpId = sample_idx;
			sample_idx++;
		}
	}


}
void main_window::setCurrentFile(const QString fileName)
{
	curFile = fileName;
//	curFile.toStdString();
//	std::cout<<curFile.toStdString()<<std::endl;
	//QDir::setCurrent(curFile);
	//setWindowFilePath(curFile);

	QSettings settings;
	QStringList files = settings.value("recentFileList").toStringList();
	files.removeAll(fileName);
	files.prepend(fileName);
	while (files.size() > MaxRecentFiles)
		files.removeLast();

	settings.setValue("recentFileList", files);

	foreach (QWidget *widget, QApplication::topLevelWidgets()) {
		main_window *mainWin = qobject_cast<main_window *>(widget);
		if (mainWin)
			mainWin->updateRecentFileActions();
	}

}

void main_window::setMutiView(bool b)
{
	QGLFormat format = QGLFormat::defaultFormat();
	format.setSampleBuffers(true);
	format.setSamples(8);
	// Create Splitters


}

void main_window::updateGL()
{
	update_viewports();
}

void main_window::update_viewports()
{
	 _viewports->updateGL();
}

PaintCanvas* main_window::getActivedCanvas()
{
	return _viewports->active_viewport();
}
