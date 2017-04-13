#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QtWidgets/QMainWindow>

#include "ui_main_window.h"
//#include "paint_canvas.h"
#include "globals.h"
#include "CustomGL/glew.h"
#include "tool.h"
#include "maching_state.h"
#include "dlg_propagate.h"
#include "LayerDialog.h"
#include <string>
#include <map>
#include <QLabel>
#include <QFileInfoList>
//#include "dlg_fitPlan.h"

using namespace std;
class PaintCanvas;
class OGL_viewports_skin2;
namespace Loader
{
	class Fbx_file;
}
class main_window : public QMainWindow
{
	Q_OBJECT

public:
	//static main_window& getInstance()
	//{
	//	static main_window instance_;
	//	//mv_ = &instance_;
	//	return instance_;
	//}

	main_window(QWidget *parent = 0);
	~main_window();
	static std::string title() { return "[MainWindow]: "; }

	void showCoordinateAndIndexUnderMouse( const QPoint& point );
	signals:
		bool LabelGraphDepthChanged( int i);

	public slots:
		bool openFile();
		bool openFiles();
		bool openFilesLazy();
		void selectedSampleChanged(QTreeWidgetItem * item, int column);

		bool setSampleVisible();
		bool setSampleInvisible();

		void doClustering();
		void finishClustering();

		void doRegister();
		void finishRegister();

// 		void doSpectralCluster();
// 		void finishSpectralCluster();

		void doGraphCut();
		void finishGraphCut();

		void doGCOptimization();
		void finishGCotimization();

		void setObjectColorMode();
		void setVertexColorMode();
		void setLabelColorMode();
		void setBallvertexMode();
		void show_normal();
		void show_kdtree();
		void toggle_camera_viewer();

		void setSelectToolMode();
		void setSceneToolMode();
		void setPaintMode();
		void setEbpd_hand_Mode();

		//render mode
		void setPointMode();
		void setFlatMode();
		void setWireMode();
		void setFlatWireMode();
		void setSmoothMode();
		void setTextureMode();
		void setSelectMode();
		void setAnimateMode();

		void showTracer();
		void clearTracer();
		//add by huayun
		//snapshot
		bool saveSnapshot();
		bool savePLY();
		bool saveLabelFile();
		bool getLabelFromFile();
		void dealtarjlabel();
		void Framelabel();
		bool wakeUpThread();
		void logText(QString as,int level);/*char* text ,int mode*/

		bool runOrPause();
		bool stop();
		bool mediaStateChanged(ANIMATION::AState _newSate  ,ANIMATION::AState _oldState);
		bool increaseInterval();
		bool decreaseInterval();



		void layerSpinBoxChanged(int i);
		void centerframeChanged(int i);
		void setSampleSelectedIndex(int i);
		void chooseGraph_WrapBox_visiable();
		void Show_Graph_WrapBox();
		void unShow_Graph_WrapBox();

		void chooseEdgeVertexs__visiable();
		void Show_EdgeVertexs();
		void unShow_EdgeVertexs();
		void processButton();
		void processButtonBack();
		void processButton2Stop();
		void processButtonRunOrPuase();
		void processButtonadvance();

		void excFrameAnimation();


		void createTreeWidgetItems();

		void computeSampleNormal();

		void batchTrajClustering();
		void iterateTrajClustering();
		void visDistor();

		void doPlanFit();

		void doPropagate();
		void doBulletPhysics();
		void doSSDR();
		void doRaycast();
		//void finishDoPlanFit();
public slots:

	    void active_viewport(int id);
		void selection_toolb_point();
		void selection_toolb_circle();
		void selection_toolb_square();
		void viewport_toolb_single();
		void viewport_toolb_doubleV();
		void viewport_toolb_doubleH();
		void viewport_toolb_four();
		void rd_mode_toolb_tex();
		void rd_mode_toolb_solid();
		void rd_mode_toolb_wire();
		void rd_mode_toolb_wire_transc();
		void show_all_gizmo(bool checked);
		void set_gizmo_trans();
		void set_gizmo_rot();
		void set_gizmo_trackball();
		void set_gizmo_scale();
		void toggle_fitting(bool checked);
		void pivot_comboBox_currentIndexChanged(int index);

		// -------------------------------------------------------------------------
		/// @name Tools
		// -------------------------------------------------------------------------
		/// Load skeleton and animation from a given fbx data structure
		bool load_fbx_skeleton_anims(const Loader::Fbx_file& loader);
		void load_ism(const QString& fileName);
		void load_fbx_mesh( Loader::Fbx_file& loader);
		/// load skeleton '.skel'
		bool load_custom_skeleton(QString name);
		/// load ssd weights '.weights'
		bool load_custom_weights(QString name);

		/// Wether activate/or deactivate GUI related to animesh
		void enable_animesh(bool state);

		/// Wether activate/or deactivate GUI related to mesh
		void enable_mesh(bool state);


//AUTO SLOTS

		void on_actionLoad_model_triggered(bool checked);


		void on_actionLoad_ISM_triggered();
		void on_actionLoad_FBX_triggered();
		void on_actionLoad_mesh_triggered();
		void on_actionLoad_skeleton_triggered();
		void on_actionLoad_weights_triggered();
		void on_actionLoad_keyframes_triggered();
		void on_actionLoad_cluster_triggered();
		void on_actionLoad_pose_triggered();
		void on_actionLoad_camera_triggered();
		void on_actionLoad_exampleMesh_triggered();
		void on_actionLoad_depthImage_triggered();
		void on_actionLoad_sampleIamge_triggered();

		void on_actionSave_as_ISM_triggered();
		void on_actionSave_as_FBX_triggered();
		void on_actionSave_as_mesh_triggered();
		void on_actionSave_as_skeleton_triggered();
		void on_actionSave_weights_triggered();
		void on_actionSave_keyframes_triggered();
		void on_actionSave_cluster_triggered();
		void on_actionSave_pose_triggered();
		void on_actionSave_camera_triggered();


		//display toolbox
		void on_gaussian_curvature_toggled(bool checked);
        void on_ssd_interpolation_toggled(bool checked);
        void on_ssd_weights_toggled(bool checked);
		void on_color_smoothing_toggled(bool checked);
		void on_color_smoothing_conservative_toggled(bool checked);
		void on_color_smoothing_laplacian_toggled(bool checked);
		void on_base_potential_toggled(bool checked);
		void on_vertices_state_toggled(bool checked);
		void on_implicit_gradient_toggled(bool checked);
		void on_cluster_toggled(bool checked);
		void on_color_nearest_joint_toggled(bool checked);
		void on_color_normals_toggled(bool checked);
		void on_color_grey_toggled(bool checked);
		void on_color_free_vertices_toggled(bool checked);
		void on_color_edge_stress_toggled(bool checked);
		void on_color_area_stress_toggled(bool checked);

		void on_display_skeleton_toggled(bool checked);
		void on_wireframe_toggled(bool checked);
		void on_display_oriented_bbox_toggled(bool checked);
		void on_horizontalSlider_sliderMoved(int position);
		void on_ssd_raio_toggled(bool checked);
		void on_dual_quaternion_radio_toggled(bool checked);
		void on_actionSkeleton_triggered();
private:
		void createAction();
		void createFileMenuAction();
		void createPaintSettingAction();
		void createAlgorithmAction();
		void createToolAction();

		void setupDockWidget();
		void setup_toolbar();
		void setup_toolbar_painting();
		void setup_toolbar_frame();
		void createStatusBar();
		void setup_viewports();
public:
	void resetSampleSet();
	PaintCanvas* getActivedCanvas();
	LayerDialog* getLayerdialog()
	{
		return m_layer;

	}
	QString curFile;
	void setCurrentFile(const QString fileName);
	void updateRecentFileActions();
	QString strippedName(const QString &fullFileName);
	
	void loadFile(const QString &fileName);
	void loadFileToSample( const QFileInfoList& filelist ,bool isLazy);
	QAction *separatorAct;
	enum { MaxRecentFiles = 5 };
	QAction *recentFileActs[MaxRecentFiles];
public:
	Ui::main_windowClass* getUI()
	{
		return &ui;
	}
public
slots:
	void openRecentFile();
	void setMutiView(bool b);
	void updateGL();
	void update_viewports();
	
private:



	//UI
	OGL_viewports_skin2* _viewports;

	Ui::main_windowClass ui;
	//std::vector<PaintCanvas*>		main_canvas_;

	QLabel*			coord_underMouse_label_;
	QLabel*			vtx_idx_underMouse_label_;

	//Samples Info
	vector< pair<string,string> >		cur_import_files_attr_;
	int			cur_select_sample_idx_;
	int			last_select_sample_idx_;

	Tool*			single_operate_tool_;
	QTimer * frameTimer;   //added by huayun

private:
	IndexType iterate_sample_idx_;

	bool is_mutiview;

	//UI
private:
	//JLinkageUI * m_linkageUi;
	//GraphCutUI * m_graphCutUi;
	//PlanFitUI* m_planFitUi;
	PropagateUI* m_propagateUi;
	LayerDialog* m_layer;
public:
	//static main_window* mv_;
	std::vector<int> traj_label_vec;

};


#endif // MAIN_WINDOW_H
