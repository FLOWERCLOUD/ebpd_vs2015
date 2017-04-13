/********************************************************************************
** Form generated from reading UI file 'main_window.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAIN_WINDOW_H
#define UI_MAIN_WINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLayout>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QToolBox>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "toolbars/toolbar.hpp"
#include "toolbars/toolbar_frames.hpp"
#include "toolbars/toolbar_painting.hpp"

QT_BEGIN_NAMESPACE

class Ui_main_windowClass
{
public:
    QAction *actionImportFiles;
    QAction *actionSet_Visible;
    QAction *actionSet_Invisible;
    QAction *actionScene_Mode;
    QAction *actionSelect_Mode;
    QAction *actionClustering;
    QAction *actionObject_Color;
    QAction *actionVertex_Color;
    QAction *actionLabel_Color;
    QAction *actionOriginal_Location;
    QAction *actionShow_Tracjectory;
    QAction *actionDont_Trace;
    QAction *actionRegister;
    QAction *actionSpectral_Cluster;
    QAction *actionGraphCut;
    QAction *actionCalculateNorm;
    QAction *actionClusterAll;
    QAction *actionVisDistortion;
    QAction *actionGCopti;
    QAction *actionPlanFit;
    QAction *actionShow_Graph_WrapBox;
    QAction *actionShow_EdgeVertexs;
    QAction *actionButtonback;
    QAction *actionButton2stop;
    QAction *actionButtonRunOrPause;
    QAction *actionButtonadvance;
    QAction *actionPropagate;
    QAction *actionSaveSnapshot;
    QAction *actionSavePly;
    QAction *actionWakeWorkThread;
    QAction *actionPoint_mode;
    QAction *actionFlat_mode;
    QAction *actionWire_mode;
    QAction *actionSmooth_mode;
    QAction *actionTexture_mode;
    QAction *actionSelect_Mode_render;
    QAction *actionFlatWire_mode;
    QAction *actionPaint_Mode;
    QAction *actionsaveLabelFile;
    QAction *actionGetlabel_from_file;
    QAction *actionBallvertex;
    QAction *actionShow_normal;
    QAction *actionImportFiles_Lazy;
    QAction *actionAbout;
    QAction *actionOn_Screen_Quick_Help;
    QAction *actionSSDR;
    QAction *actionBullet;
    QAction *actionAnimate_Mode;
    QAction *actionShow_camera_viewer;
    QAction *actionLoad_ISM;
    QAction *actionLoad_FBX;
    QAction *actionLoad_mesh;
    QAction *actionLoad_skeleton;
    QAction *actionLoad_weights;
    QAction *actionLoad_keyframes;
    QAction *actionLoad_cluster;
    QAction *actionSave_as_ISM;
    QAction *actionSave_as_FBX;
    QAction *actionSave_as_mesh;
    QAction *actionSave_as_skeleton;
    QAction *actionSave_weights;
    QAction *actionSave_cluster;
    QAction *actionLoad_model;
    QAction *actionShotcut;
    QAction *actionLoad_pose;
    QAction *actionSave_pose;
    QAction *actionLoad_camera;
    QAction *actionSave_camera;
    QAction *actionMesh;
    QAction *actionSkeleton;
    QAction *actionResouce_usage;
    QAction *actionColor;
    QAction *actionMaterial;
    QAction *actionSetting;
    QAction *actionSave_keyframes;
    QAction *actionLoad_exampleMesh;
    QAction *actionEbpd_hand_mode;
    QAction *actionLoad_depthImage;
    QAction *actionLoad_sampleIamge;
    QAction *actionRaycast;
    QAction *actionShow_kdtree;
    QWidget *centralWidget;
    QSplitter *splitter;
    QToolBox *toolBoxMenu;
    QWidget *Display_settings;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *vert_layout_display_tool_box;
    QGroupBox *box_camera;
    QVBoxLayout *verticalLayout_7;
    QVBoxLayout *verticalLayout_2;
    QWidget *widget_15;
    QVBoxLayout *verticalLayout_5;
    QWidget *widget_26;
    QHBoxLayout *horizontalLayout_32;
    QLabel *lbl_camera_aperture;
    QSpinBox *spinB_aperture;
    QWidget *widget_27;
    QHBoxLayout *horizontalLayout_33;
    QLabel *lbl_near_plane;
    QDoubleSpinBox *dSpinB_near_plane;
    QWidget *widget_29;
    QHBoxLayout *horizontalLayout_35;
    QLabel *lbl_far_plane;
    QDoubleSpinBox *dSpinB_far_plane;
    QWidget *widget_28;
    QHBoxLayout *horizontalLayout_34;
    QPushButton *pushB_reset_camera;
    QCheckBox *checkB_camera_tracking;
    QGroupBox *box_animesh_color;
    QVBoxLayout *verticalLayout_3;
    QVBoxLayout *vert_layout_box_mesh;
    QWidget *radio_grp_mesh_color;
    QVBoxLayout *verticalLayout_6;
    QRadioButton *gaussian_curvature;
    QRadioButton *ssd_interpolation;
    QRadioButton *ssd_weights;
    QRadioButton *color_smoothing;
    QRadioButton *color_smoothing_conservative;
    QRadioButton *color_smoothing_laplacian;
    QRadioButton *base_potential;
    QRadioButton *vertices_state;
    QRadioButton *implicit_gradient;
    QRadioButton *cluster;
    QRadioButton *color_nearest_joint;
    QRadioButton *color_normals;
    QRadioButton *color_grey;
    QRadioButton *color_free_vertices;
    QRadioButton *color_edge_stress;
    QRadioButton *color_area_stress;
    QCheckBox *show_rbf_samples;
    QFrame *line_12;
    QLabel *lbl_select_vert;
    QWidget *select_vert_widget;
    QHBoxLayout *horizontalLayout_25;
    QSpinBox *spinB_vert_id;
    QPushButton *pButton_do_select_vert;
    QGroupBox *box_mesh_color;
    QVBoxLayout *verticalLayout_36;
    QLabel *lbl_points_color;
    QWidget *widget_6;
    QVBoxLayout *verticalLayout_25;
    QRadioButton *buton_uniform_point_cl;
    QRadioButton *button_defects_point_cl;
    QWidget *widget;
    QVBoxLayout *verticalLayout_21;
    QLabel *label;
    QSlider *horizontalSlider;
    QFrame *line;
    QCheckBox *wireframe;
    QGroupBox *box_skeleton;
    QVBoxLayout *verticalLayout_12;
    QCheckBox *display_skeleton;
    QCheckBox *display_oriented_bbox;
    QCheckBox *checkB_aa_bbox;
    QCheckBox *checkB_draw_grid;
    QGroupBox *grpBox_operators;
    QVBoxLayout *verticalLayout_13;
    QWidget *widget_12;
    QVBoxLayout *verticalLayout_32;
    QComboBox *comboB_operators;
    QCheckBox *display_operator;
    QWidget *widget_13;
    QHBoxLayout *horizontalLayout_20;
    QLabel *lbl_size_operator;
    QSpinBox *spinBox;
    QWidget *widget_11;
    QHBoxLayout *horizontalLayout_19;
    QLabel *lbl_opening_angle;
    QDoubleSpinBox *dSpinB_opening_value;
    QCheckBox *display_controller;
    QSpacerItem *verticalSpacer_2;
    QSpacerItem *horizontal_spacer_2;
    QWidget *graph_edition;
    QHBoxLayout *horizontalLayout_8;
    QVBoxLayout *vert_layout_graph_edit;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout_19;
    QPushButton *pushB_attached_skeleton;
    QPushButton *center_graph_node;
    QSpacerItem *verticalSpacer_4;
    QSpacerItem *horizontalSpacer_4;
    QWidget *Animation_settings;
    QHBoxLayout *horizontalLayout_2;
    QVBoxLayout *vert_layout_anim_toolbox;
    QGroupBox *box_anim_type;
    QVBoxLayout *verticalLayout_10;
    QWidget *radio_grp_3;
    QVBoxLayout *verticalLayout_11;
    QRadioButton *ssd_raio;
    QRadioButton *dual_quaternion_radio;
    QCheckBox *implicit_skinning_checkBox;
    QCheckBox *checkBox_incremental;
    QPushButton *reset_anim;
    QGroupBox *grpBox_smoothing_weights;
    QVBoxLayout *verticalLayout_23;
    QWidget *widget_smooth_weights_diffuse_iter;
    QFormLayout *formLayout_2;
    QSpinBox *spinBox_diffuse_smoothing_weights_iter;
    QLabel *lbl_nb_iter_smoothing_weight_diffusion;
    QGroupBox *grpBox_weights;
    QVBoxLayout *verticalLayout_18;
    QPushButton *pushB_set_rigid_weights;
    QFrame *line_11;
    QLabel *lbl_auto_weights_experimental;
    QWidget *grp_widget_auto_weight;
    QHBoxLayout *horizontalLayout_13;
    QWidget *widget_5;
    QVBoxLayout *verticalLayout_24;
    QDoubleSpinBox *dSpinB_diff_w_alpha_exp;
    QSpinBox *spinB_auto_w_nb_iter_exp;
    QPushButton *pushB_diff_w_exp;
    QFrame *line_7;
    QLabel *lbl_diffuse_weights;
    QWidget *widget_9;
    QHBoxLayout *horizontalLayout_16;
    QWidget *widget_10;
    QVBoxLayout *verticalLayout_28;
    QDoubleSpinBox *dSpinB_diff_w_alpha;
    QSpinBox *spinB_diff_w_nb_iter;
    QPushButton *pushB_diffuse_curr_weights;
    QFrame *line_10;
    QLabel *lbl_heat_diffusion_weights;
    QWidget *grp_widget_heat_diffusion;
    QHBoxLayout *horizontalLayout_23;
    QDoubleSpinBox *dSpinBox_heat_coeff;
    QPushButton *pButton_compute_heat_difusion;
    QSpacerItem *verticalSpacer;
    QSpacerItem *horizontalSpacer;
    QWidget *blending_settings;
    QHBoxLayout *horizontalLayout_3;
    QVBoxLayout *vert_layout_blending;
    QGroupBox *grpBox_bulge_in_contact;
    QVBoxLayout *verticalLayout_14;
    QWidget *widget_bulge_in_contact;
    QHBoxLayout *horizontalLayout_6;
    QLabel *lbl_force;
    QDoubleSpinBox *spinBox_bulge_in_contact_force;
    QPushButton *update_bulge_in_contact;
    QGroupBox *grpBox_controller;
    QVBoxLayout *verticalLayout_17;
    QWidget *preset_ctrl;
    QGroupBox *gBox_controller_values;
    QVBoxLayout *verticalLayout_16;
    QHBoxLayout *layout_ctrl_p0;
    QLabel *lbl_p0;
    QDoubleSpinBox *dSpinB_ctrl_p0_x;
    QDoubleSpinBox *dSpinB_ctrl_p0_y;
    QHBoxLayout *layout_ctrl_p1;
    QLabel *lbl_p1;
    QDoubleSpinBox *dSpinB_ctrl_p1_x;
    QDoubleSpinBox *dSpinB_ctrl_p1_y;
    QHBoxLayout *layout_ctrl_p2;
    QLabel *lbl_p2;
    QDoubleSpinBox *dSpinB_ctrl_p2_x;
    QDoubleSpinBox *dSpinB_ctrl_p2_y;
    QHBoxLayout *layout_ctrl_slopes;
    QLabel *lbl_slopes;
    QDoubleSpinBox *dSpinB_ctrl_slope0;
    QDoubleSpinBox *dSpinB_ctrl_slope1;
    QPushButton *pushB_edit_spline;
    QSpacerItem *blending_vertical_spacer;
    QSpacerItem *blending_horizontal_spacer;
    QWidget *bone_editor;
    QHBoxLayout *horizontalLayout_4;
    QVBoxLayout *vert_layout_bone_editor;
    QGroupBox *box_edit_RBF;
    QVBoxLayout *verticalLayout_15;
    QCheckBox *rbf_edition;
    QCheckBox *cBox_always_precompute;
    QCheckBox *checkB_factor_siblings;
    QCheckBox *local_frame;
    QCheckBox *checkB_align_with_normal;
    QCheckBox *move_joints;
    QCheckBox *checkB_show_junction;
    QPushButton *pushB_empty_bone;
    QWidget *widget_24;
    QHBoxLayout *horizontalLayout_30;
    QPushButton *pButton_add_caps;
    QPushButton *pButton_supr_caps;
    QWidget *widget_16;
    QHBoxLayout *horizontalLayout_7;
    QLabel *lbl_hrbf_radius;
    QDoubleSpinBox *dSpinB_hrbf_radius;
    QGroupBox *groupBx_auto_sampling;
    QVBoxLayout *verticalLayout_29;
    QLabel *lbl_per_max_dist_from_joints;
    QWidget *widget_grp_max_dist_from_joints;
    QHBoxLayout *horizontalLayout_17;
    QWidget *widget_grp_joint;
    QVBoxLayout *verticalLayout_31;
    QLabel *lbl_max_dist_joint;
    QDoubleSpinBox *dSpinB_max_dist_joint;
    QCheckBox *checkB_cap_joint;
    QWidget *widget_grp_parent;
    QVBoxLayout *verticalLayout_30;
    QLabel *lbl_max_dist_parent;
    QDoubleSpinBox *dSpinB_max_dist_parent;
    QCheckBox *checkB_capparent;
    QWidget *widget_grp_max_fold;
    QHBoxLayout *horizontalLayout_18;
    QLabel *lbl_max_fold;
    QDoubleSpinBox *dSpinB_max_fold;
    QWidget *widget_14;
    QHBoxLayout *horizontalLayout_21;
    QLabel *label_2;
    QDoubleSpinBox *dSpinB_min_dist_samples;
    QWidget *widget_25;
    QHBoxLayout *horizontalLayout_31;
    QLabel *lbl_nb_samples;
    QSpinBox *spinB_nb_samples_psd;
    QCheckBox *checkB_auto_sample;
    QComboBox *cBox_sampling_type;
    QPushButton *choose_hrbf_samples;
    QSpacerItem *verticalSpacer_3;
    QSpacerItem *horizontalSpacer_3;
    QWidget *debug_tools;
    QHBoxLayout *horizontalLayout_9;
    QVBoxLayout *vert_layout_graph_edit_2;
    QLabel *lbl_fitting_steps;
    QWidget *widget_2;
    QHBoxLayout *horizontalLayout_10;
    QSpinBox *spinBox_nb_step_fitting;
    QCheckBox *enable_partial_fit;
    QSlider *slider_nb_step_fit;
    QFrame *line_3;
    QCheckBox *debug_show_gradient;
    QCheckBox *debug_show_normal;
    QFrame *line_2;
    QLabel *lbl_collisions_threshold;
    QDoubleSpinBox *doubleSpinBox;
    QFrame *line_4;
    QLabel *lbl_step_length;
    QDoubleSpinBox *spinB_step_length;
    QCheckBox *checkB_enable_raphson;
    QFrame *line_8;
    QLabel *lbl_collision_depth;
    QDoubleSpinBox *dSpinB_collision_depth;
    QCheckBox *box_potential_pit;
    QCheckBox *checkBox_collsion_on;
    QCheckBox *checkBox_update_base_potential;
    QCheckBox *checkB_filter_relax;
    QFrame *line_9;
    QWidget *widget_17;
    QHBoxLayout *horizontalLayout_22;
    QPushButton *pButton_invert_propagation;
    QPushButton *pButton_rst_invert_propagation;
    QFrame *line_13;
    QCheckBox *checkB_enable_smoothing;
    QLabel *lbl_smoothing_first_step;
    QWidget *widget_21;
    QHBoxLayout *horizontalLayout_27;
    QSpinBox *spinB_nb_iter_smooth1;
    QDoubleSpinBox *dSpinB_lambda_smooth1;
    QLabel *lbl_smoothing_second_step;
    QWidget *widget_22;
    QHBoxLayout *horizontalLayout_28;
    QSpinBox *spinB_nb_iter_smooth2;
    QDoubleSpinBox *dSpinB_lambda_smooth2;
    QWidget *widget_20;
    QHBoxLayout *horizontalLayout_26;
    QLabel *lbl_grid_res;
    QSpinBox *spinB_grid_res;
    QWidget *widget_30;
    QHBoxLayout *horizontalLayout_36;
    QLabel *label_3;
    QDoubleSpinBox *dSpinBox_spare_box;
    QWidget *widget_31;
    QHBoxLayout *horizontalLayout_37;
    QLabel *label_4;
    QDoubleSpinBox *doubleSpinBox_2;
    QWidget *widget_32;
    QHBoxLayout *horizontalLayout_38;
    QLabel *label_5;
    QSpinBox *spinB_tab_val_idx;
    QDoubleSpinBox *dSpin_spare_vals_array;
    QCheckBox *checkBox;
    QSpacerItem *horizontalSpacer_5;
    QFrame *viewports_frame;
    QVBoxLayout *verticalLayout_35;
    QMenuBar *menuBar;
    QMenu *menuFiles;
    QMenu *menuImport;
    QMenu *menuExport;
    QMenu *menuPaint;
    QMenu *menuSelect;
    QMenu *menuAlgorithm;
    QMenu *menuRenderMode;
    QMenu *menuHelp;
    QMenu *menuInfo;
    QMenu *menuCutstomize;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents;
    QTreeWidget *treeWidget;
    QSpinBox *LayerSpinBox;
    QSpinBox *centerframe;
    QPushButton *button_traj_label;
    QLineEdit *text_trajectory_label;
    QLineEdit *showlabel_lineEdit;
    QPushButton *show_label_Button;
    Toolbar *toolBar;
    Toolbar_frames *toolBar_frame;
    Toolbar_painting *toolBar_painting;

    void setupUi(QMainWindow *main_windowClass)
    {
        if (main_windowClass->objectName().isEmpty())
            main_windowClass->setObjectName(QStringLiteral("main_windowClass"));
        main_windowClass->resize(1122, 837);
        actionImportFiles = new QAction(main_windowClass);
        actionImportFiles->setObjectName(QStringLiteral("actionImportFiles"));
        QIcon icon;
        icon.addFile(QStringLiteral("Resources/openFile.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionImportFiles->setIcon(icon);
        actionSet_Visible = new QAction(main_windowClass);
        actionSet_Visible->setObjectName(QStringLiteral("actionSet_Visible"));
        QIcon icon1;
        icon1.addFile(QStringLiteral("Resources/visible.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSet_Visible->setIcon(icon1);
        actionSet_Invisible = new QAction(main_windowClass);
        actionSet_Invisible->setObjectName(QStringLiteral("actionSet_Invisible"));
        QIcon icon2;
        icon2.addFile(QStringLiteral("Resources/invisible.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSet_Invisible->setIcon(icon2);
        actionScene_Mode = new QAction(main_windowClass);
        actionScene_Mode->setObjectName(QStringLiteral("actionScene_Mode"));
        QIcon icon3;
        icon3.addFile(QStringLiteral("Resources/scene.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionScene_Mode->setIcon(icon3);
        actionSelect_Mode = new QAction(main_windowClass);
        actionSelect_Mode->setObjectName(QStringLiteral("actionSelect_Mode"));
        QIcon icon4;
        icon4.addFile(QStringLiteral("Resources/select.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSelect_Mode->setIcon(icon4);
        actionClustering = new QAction(main_windowClass);
        actionClustering->setObjectName(QStringLiteral("actionClustering"));
        QIcon icon5;
        icon5.addFile(QStringLiteral("Resources/categorize.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionClustering->setIcon(icon5);
        actionObject_Color = new QAction(main_windowClass);
        actionObject_Color->setObjectName(QStringLiteral("actionObject_Color"));
        QIcon icon6;
        icon6.addFile(QStringLiteral("Resources/tree.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionObject_Color->setIcon(icon6);
        actionVertex_Color = new QAction(main_windowClass);
        actionVertex_Color->setObjectName(QStringLiteral("actionVertex_Color"));
        QIcon icon7;
        icon7.addFile(QStringLiteral("Resources/leaf.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionVertex_Color->setIcon(icon7);
        actionLabel_Color = new QAction(main_windowClass);
        actionLabel_Color->setObjectName(QStringLiteral("actionLabel_Color"));
        QIcon icon8;
        icon8.addFile(QStringLiteral("Resources/label.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionLabel_Color->setIcon(icon8);
        actionOriginal_Location = new QAction(main_windowClass);
        actionOriginal_Location->setObjectName(QStringLiteral("actionOriginal_Location"));
        QIcon icon9;
        icon9.addFile(QStringLiteral("Resources/tree2.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionOriginal_Location->setIcon(icon9);
        actionShow_Tracjectory = new QAction(main_windowClass);
        actionShow_Tracjectory->setObjectName(QStringLiteral("actionShow_Tracjectory"));
        QIcon icon10;
        icon10.addFile(QStringLiteral("Resources/show_trace.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionShow_Tracjectory->setIcon(icon10);
        actionDont_Trace = new QAction(main_windowClass);
        actionDont_Trace->setObjectName(QStringLiteral("actionDont_Trace"));
        QIcon icon11;
        icon11.addFile(QStringLiteral("Resources/dont_trace.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionDont_Trace->setIcon(icon11);
        actionRegister = new QAction(main_windowClass);
        actionRegister->setObjectName(QStringLiteral("actionRegister"));
        actionSpectral_Cluster = new QAction(main_windowClass);
        actionSpectral_Cluster->setObjectName(QStringLiteral("actionSpectral_Cluster"));
        actionGraphCut = new QAction(main_windowClass);
        actionGraphCut->setObjectName(QStringLiteral("actionGraphCut"));
        actionCalculateNorm = new QAction(main_windowClass);
        actionCalculateNorm->setObjectName(QStringLiteral("actionCalculateNorm"));
        actionClusterAll = new QAction(main_windowClass);
        actionClusterAll->setObjectName(QStringLiteral("actionClusterAll"));
        actionVisDistortion = new QAction(main_windowClass);
        actionVisDistortion->setObjectName(QStringLiteral("actionVisDistortion"));
        actionGCopti = new QAction(main_windowClass);
        actionGCopti->setObjectName(QStringLiteral("actionGCopti"));
        actionPlanFit = new QAction(main_windowClass);
        actionPlanFit->setObjectName(QStringLiteral("actionPlanFit"));
        actionShow_Graph_WrapBox = new QAction(main_windowClass);
        actionShow_Graph_WrapBox->setObjectName(QStringLiteral("actionShow_Graph_WrapBox"));
        QIcon icon12;
        icon12.addFile(QStringLiteral("Resources/nnolinkNode.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionShow_Graph_WrapBox->setIcon(icon12);
        actionShow_EdgeVertexs = new QAction(main_windowClass);
        actionShow_EdgeVertexs->setObjectName(QStringLiteral("actionShow_EdgeVertexs"));
        QIcon icon13;
        icon13.addFile(QStringLiteral("Resources/NoedgeVertexs.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionShow_EdgeVertexs->setIcon(icon13);
        actionButtonback = new QAction(main_windowClass);
        actionButtonback->setObjectName(QStringLiteral("actionButtonback"));
        QIcon icon14;
        icon14.addFile(QStringLiteral("Resources/buttonback.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionButtonback->setIcon(icon14);
        actionButton2stop = new QAction(main_windowClass);
        actionButton2stop->setObjectName(QStringLiteral("actionButton2stop"));
        QIcon icon15;
        icon15.addFile(QStringLiteral("Resources/button2stop.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionButton2stop->setIcon(icon15);
        actionButtonRunOrPause = new QAction(main_windowClass);
        actionButtonRunOrPause->setObjectName(QStringLiteral("actionButtonRunOrPause"));
        QIcon icon16;
        icon16.addFile(QStringLiteral("Resources/buttonstop2run.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionButtonRunOrPause->setIcon(icon16);
        actionButtonadvance = new QAction(main_windowClass);
        actionButtonadvance->setObjectName(QStringLiteral("actionButtonadvance"));
        QIcon icon17;
        icon17.addFile(QStringLiteral("Resources/buttonadvance.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionButtonadvance->setIcon(icon17);
        actionPropagate = new QAction(main_windowClass);
        actionPropagate->setObjectName(QStringLiteral("actionPropagate"));
        QIcon icon18;
        icon18.addFile(QStringLiteral("Resources/propagate.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPropagate->setIcon(icon18);
        actionSaveSnapshot = new QAction(main_windowClass);
        actionSaveSnapshot->setObjectName(QStringLiteral("actionSaveSnapshot"));
        QIcon icon19;
        icon19.addFile(QStringLiteral("Resources/screenshot.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSaveSnapshot->setIcon(icon19);
        actionSavePly = new QAction(main_windowClass);
        actionSavePly->setObjectName(QStringLiteral("actionSavePly"));
        QIcon icon20;
        icon20.addFile(QStringLiteral("Resources/save.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSavePly->setIcon(icon20);
        actionWakeWorkThread = new QAction(main_windowClass);
        actionWakeWorkThread->setObjectName(QStringLiteral("actionWakeWorkThread"));
        QIcon icon21;
        icon21.addFile(QStringLiteral("Resources/wake_up.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionWakeWorkThread->setIcon(icon21);
        actionPoint_mode = new QAction(main_windowClass);
        actionPoint_mode->setObjectName(QStringLiteral("actionPoint_mode"));
        actionFlat_mode = new QAction(main_windowClass);
        actionFlat_mode->setObjectName(QStringLiteral("actionFlat_mode"));
        actionWire_mode = new QAction(main_windowClass);
        actionWire_mode->setObjectName(QStringLiteral("actionWire_mode"));
        actionSmooth_mode = new QAction(main_windowClass);
        actionSmooth_mode->setObjectName(QStringLiteral("actionSmooth_mode"));
        actionTexture_mode = new QAction(main_windowClass);
        actionTexture_mode->setObjectName(QStringLiteral("actionTexture_mode"));
        actionSelect_Mode_render = new QAction(main_windowClass);
        actionSelect_Mode_render->setObjectName(QStringLiteral("actionSelect_Mode_render"));
        actionFlatWire_mode = new QAction(main_windowClass);
        actionFlatWire_mode->setObjectName(QStringLiteral("actionFlatWire_mode"));
        actionPaint_Mode = new QAction(main_windowClass);
        actionPaint_Mode->setObjectName(QStringLiteral("actionPaint_Mode"));
        actionsaveLabelFile = new QAction(main_windowClass);
        actionsaveLabelFile->setObjectName(QStringLiteral("actionsaveLabelFile"));
        actionGetlabel_from_file = new QAction(main_windowClass);
        actionGetlabel_from_file->setObjectName(QStringLiteral("actionGetlabel_from_file"));
        actionBallvertex = new QAction(main_windowClass);
        actionBallvertex->setObjectName(QStringLiteral("actionBallvertex"));
        actionShow_normal = new QAction(main_windowClass);
        actionShow_normal->setObjectName(QStringLiteral("actionShow_normal"));
        actionImportFiles_Lazy = new QAction(main_windowClass);
        actionImportFiles_Lazy->setObjectName(QStringLiteral("actionImportFiles_Lazy"));
        actionAbout = new QAction(main_windowClass);
        actionAbout->setObjectName(QStringLiteral("actionAbout"));
        actionOn_Screen_Quick_Help = new QAction(main_windowClass);
        actionOn_Screen_Quick_Help->setObjectName(QStringLiteral("actionOn_Screen_Quick_Help"));
        actionSSDR = new QAction(main_windowClass);
        actionSSDR->setObjectName(QStringLiteral("actionSSDR"));
        actionBullet = new QAction(main_windowClass);
        actionBullet->setObjectName(QStringLiteral("actionBullet"));
        actionAnimate_Mode = new QAction(main_windowClass);
        actionAnimate_Mode->setObjectName(QStringLiteral("actionAnimate_Mode"));
        actionShow_camera_viewer = new QAction(main_windowClass);
        actionShow_camera_viewer->setObjectName(QStringLiteral("actionShow_camera_viewer"));
        actionLoad_ISM = new QAction(main_windowClass);
        actionLoad_ISM->setObjectName(QStringLiteral("actionLoad_ISM"));
        actionLoad_FBX = new QAction(main_windowClass);
        actionLoad_FBX->setObjectName(QStringLiteral("actionLoad_FBX"));
        actionLoad_mesh = new QAction(main_windowClass);
        actionLoad_mesh->setObjectName(QStringLiteral("actionLoad_mesh"));
        actionLoad_skeleton = new QAction(main_windowClass);
        actionLoad_skeleton->setObjectName(QStringLiteral("actionLoad_skeleton"));
        actionLoad_weights = new QAction(main_windowClass);
        actionLoad_weights->setObjectName(QStringLiteral("actionLoad_weights"));
        actionLoad_keyframes = new QAction(main_windowClass);
        actionLoad_keyframes->setObjectName(QStringLiteral("actionLoad_keyframes"));
        actionLoad_cluster = new QAction(main_windowClass);
        actionLoad_cluster->setObjectName(QStringLiteral("actionLoad_cluster"));
        actionSave_as_ISM = new QAction(main_windowClass);
        actionSave_as_ISM->setObjectName(QStringLiteral("actionSave_as_ISM"));
        actionSave_as_FBX = new QAction(main_windowClass);
        actionSave_as_FBX->setObjectName(QStringLiteral("actionSave_as_FBX"));
        actionSave_as_mesh = new QAction(main_windowClass);
        actionSave_as_mesh->setObjectName(QStringLiteral("actionSave_as_mesh"));
        actionSave_as_skeleton = new QAction(main_windowClass);
        actionSave_as_skeleton->setObjectName(QStringLiteral("actionSave_as_skeleton"));
        actionSave_weights = new QAction(main_windowClass);
        actionSave_weights->setObjectName(QStringLiteral("actionSave_weights"));
        actionSave_cluster = new QAction(main_windowClass);
        actionSave_cluster->setObjectName(QStringLiteral("actionSave_cluster"));
        actionLoad_model = new QAction(main_windowClass);
        actionLoad_model->setObjectName(QStringLiteral("actionLoad_model"));
        actionShotcut = new QAction(main_windowClass);
        actionShotcut->setObjectName(QStringLiteral("actionShotcut"));
        actionLoad_pose = new QAction(main_windowClass);
        actionLoad_pose->setObjectName(QStringLiteral("actionLoad_pose"));
        actionSave_pose = new QAction(main_windowClass);
        actionSave_pose->setObjectName(QStringLiteral("actionSave_pose"));
        actionLoad_camera = new QAction(main_windowClass);
        actionLoad_camera->setObjectName(QStringLiteral("actionLoad_camera"));
        actionSave_camera = new QAction(main_windowClass);
        actionSave_camera->setObjectName(QStringLiteral("actionSave_camera"));
        actionMesh = new QAction(main_windowClass);
        actionMesh->setObjectName(QStringLiteral("actionMesh"));
        actionSkeleton = new QAction(main_windowClass);
        actionSkeleton->setObjectName(QStringLiteral("actionSkeleton"));
        actionResouce_usage = new QAction(main_windowClass);
        actionResouce_usage->setObjectName(QStringLiteral("actionResouce_usage"));
        actionColor = new QAction(main_windowClass);
        actionColor->setObjectName(QStringLiteral("actionColor"));
        actionMaterial = new QAction(main_windowClass);
        actionMaterial->setObjectName(QStringLiteral("actionMaterial"));
        actionSetting = new QAction(main_windowClass);
        actionSetting->setObjectName(QStringLiteral("actionSetting"));
        QIcon icon22;
        icon22.addFile(QStringLiteral("Resources/setting.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSetting->setIcon(icon22);
        actionSave_keyframes = new QAction(main_windowClass);
        actionSave_keyframes->setObjectName(QStringLiteral("actionSave_keyframes"));
        actionLoad_exampleMesh = new QAction(main_windowClass);
        actionLoad_exampleMesh->setObjectName(QStringLiteral("actionLoad_exampleMesh"));
        actionEbpd_hand_mode = new QAction(main_windowClass);
        actionEbpd_hand_mode->setObjectName(QStringLiteral("actionEbpd_hand_mode"));
        actionLoad_depthImage = new QAction(main_windowClass);
        actionLoad_depthImage->setObjectName(QStringLiteral("actionLoad_depthImage"));
        actionLoad_sampleIamge = new QAction(main_windowClass);
        actionLoad_sampleIamge->setObjectName(QStringLiteral("actionLoad_sampleIamge"));
        actionRaycast = new QAction(main_windowClass);
        actionRaycast->setObjectName(QStringLiteral("actionRaycast"));
        actionShow_kdtree = new QAction(main_windowClass);
        actionShow_kdtree->setObjectName(QStringLiteral("actionShow_kdtree"));
        centralWidget = new QWidget(main_windowClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        centralWidget->setEnabled(true);
        splitter = new QSplitter(centralWidget);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setGeometry(QRect(10, 10, 1522, 744));
        splitter->setOrientation(Qt::Horizontal);
        toolBoxMenu = new QToolBox(splitter);
        toolBoxMenu->setObjectName(QStringLiteral("toolBoxMenu"));
        toolBoxMenu->setMinimumSize(QSize(215, 0));
        toolBoxMenu->setMaximumSize(QSize(265, 16777215));
        toolBoxMenu->setAutoFillBackground(false);
        toolBoxMenu->setFrameShape(QFrame::NoFrame);
        Display_settings = new QWidget();
        Display_settings->setObjectName(QStringLiteral("Display_settings"));
        Display_settings->setGeometry(QRect(0, 0, 248, 1147));
        horizontalLayout = new QHBoxLayout(Display_settings);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        vert_layout_display_tool_box = new QVBoxLayout();
        vert_layout_display_tool_box->setSpacing(6);
        vert_layout_display_tool_box->setObjectName(QStringLiteral("vert_layout_display_tool_box"));
        box_camera = new QGroupBox(Display_settings);
        box_camera->setObjectName(QStringLiteral("box_camera"));
        verticalLayout_7 = new QVBoxLayout(box_camera);
        verticalLayout_7->setSpacing(6);
        verticalLayout_7->setContentsMargins(11, 11, 11, 11);
        verticalLayout_7->setObjectName(QStringLiteral("verticalLayout_7"));
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        widget_15 = new QWidget(box_camera);
        widget_15->setObjectName(QStringLiteral("widget_15"));
        verticalLayout_5 = new QVBoxLayout(widget_15);
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setContentsMargins(11, 11, 11, 11);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(0, -1, 0, -1);
        widget_26 = new QWidget(widget_15);
        widget_26->setObjectName(QStringLiteral("widget_26"));
        horizontalLayout_32 = new QHBoxLayout(widget_26);
        horizontalLayout_32->setSpacing(6);
        horizontalLayout_32->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_32->setObjectName(QStringLiteral("horizontalLayout_32"));
        horizontalLayout_32->setContentsMargins(-1, 0, -1, 0);
        lbl_camera_aperture = new QLabel(widget_26);
        lbl_camera_aperture->setObjectName(QStringLiteral("lbl_camera_aperture"));

        horizontalLayout_32->addWidget(lbl_camera_aperture);

        spinB_aperture = new QSpinBox(widget_26);
        spinB_aperture->setObjectName(QStringLiteral("spinB_aperture"));
        spinB_aperture->setMinimum(5);
        spinB_aperture->setMaximum(170);
        spinB_aperture->setSingleStep(5);
        spinB_aperture->setValue(50);

        horizontalLayout_32->addWidget(spinB_aperture);


        verticalLayout_5->addWidget(widget_26);

        widget_27 = new QWidget(widget_15);
        widget_27->setObjectName(QStringLiteral("widget_27"));
        horizontalLayout_33 = new QHBoxLayout(widget_27);
        horizontalLayout_33->setSpacing(6);
        horizontalLayout_33->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_33->setObjectName(QStringLiteral("horizontalLayout_33"));
        horizontalLayout_33->setContentsMargins(-1, 0, -1, 0);
        lbl_near_plane = new QLabel(widget_27);
        lbl_near_plane->setObjectName(QStringLiteral("lbl_near_plane"));

        horizontalLayout_33->addWidget(lbl_near_plane);

        dSpinB_near_plane = new QDoubleSpinBox(widget_27);
        dSpinB_near_plane->setObjectName(QStringLiteral("dSpinB_near_plane"));
        dSpinB_near_plane->setDecimals(3);
        dSpinB_near_plane->setMinimum(0.01);
        dSpinB_near_plane->setValue(1);

        horizontalLayout_33->addWidget(dSpinB_near_plane);


        verticalLayout_5->addWidget(widget_27);

        widget_29 = new QWidget(widget_15);
        widget_29->setObjectName(QStringLiteral("widget_29"));
        horizontalLayout_35 = new QHBoxLayout(widget_29);
        horizontalLayout_35->setSpacing(6);
        horizontalLayout_35->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_35->setObjectName(QStringLiteral("horizontalLayout_35"));
        horizontalLayout_35->setContentsMargins(-1, 0, -1, 0);
        lbl_far_plane = new QLabel(widget_29);
        lbl_far_plane->setObjectName(QStringLiteral("lbl_far_plane"));

        horizontalLayout_35->addWidget(lbl_far_plane);

        dSpinB_far_plane = new QDoubleSpinBox(widget_29);
        dSpinB_far_plane->setObjectName(QStringLiteral("dSpinB_far_plane"));
        dSpinB_far_plane->setDecimals(2);
        dSpinB_far_plane->setMinimum(1);
        dSpinB_far_plane->setMaximum(100000);
        dSpinB_far_plane->setSingleStep(10);
        dSpinB_far_plane->setValue(300);

        horizontalLayout_35->addWidget(dSpinB_far_plane);


        verticalLayout_5->addWidget(widget_29);

        widget_28 = new QWidget(widget_15);
        widget_28->setObjectName(QStringLiteral("widget_28"));
        horizontalLayout_34 = new QHBoxLayout(widget_28);
        horizontalLayout_34->setSpacing(6);
        horizontalLayout_34->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_34->setObjectName(QStringLiteral("horizontalLayout_34"));
        horizontalLayout_34->setContentsMargins(-1, 0, -1, 0);
        pushB_reset_camera = new QPushButton(widget_28);
        pushB_reset_camera->setObjectName(QStringLiteral("pushB_reset_camera"));

        horizontalLayout_34->addWidget(pushB_reset_camera);

        checkB_camera_tracking = new QCheckBox(widget_28);
        checkB_camera_tracking->setObjectName(QStringLiteral("checkB_camera_tracking"));

        horizontalLayout_34->addWidget(checkB_camera_tracking);


        verticalLayout_5->addWidget(widget_28);


        verticalLayout_2->addWidget(widget_15);


        verticalLayout_7->addLayout(verticalLayout_2);


        vert_layout_display_tool_box->addWidget(box_camera);

        box_animesh_color = new QGroupBox(Display_settings);
        box_animesh_color->setObjectName(QStringLiteral("box_animesh_color"));
        verticalLayout_3 = new QVBoxLayout(box_animesh_color);
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        vert_layout_box_mesh = new QVBoxLayout();
        vert_layout_box_mesh->setSpacing(6);
        vert_layout_box_mesh->setObjectName(QStringLiteral("vert_layout_box_mesh"));
        radio_grp_mesh_color = new QWidget(box_animesh_color);
        radio_grp_mesh_color->setObjectName(QStringLiteral("radio_grp_mesh_color"));
        radio_grp_mesh_color->setEnabled(true);
        verticalLayout_6 = new QVBoxLayout(radio_grp_mesh_color);
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setContentsMargins(11, 11, 11, 11);
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        gaussian_curvature = new QRadioButton(radio_grp_mesh_color);
        gaussian_curvature->setObjectName(QStringLiteral("gaussian_curvature"));

        verticalLayout_6->addWidget(gaussian_curvature);

        ssd_interpolation = new QRadioButton(radio_grp_mesh_color);
        ssd_interpolation->setObjectName(QStringLiteral("ssd_interpolation"));

        verticalLayout_6->addWidget(ssd_interpolation);

        ssd_weights = new QRadioButton(radio_grp_mesh_color);
        ssd_weights->setObjectName(QStringLiteral("ssd_weights"));

        verticalLayout_6->addWidget(ssd_weights);

        color_smoothing = new QRadioButton(radio_grp_mesh_color);
        color_smoothing->setObjectName(QStringLiteral("color_smoothing"));

        verticalLayout_6->addWidget(color_smoothing);

        color_smoothing_conservative = new QRadioButton(radio_grp_mesh_color);
        color_smoothing_conservative->setObjectName(QStringLiteral("color_smoothing_conservative"));

        verticalLayout_6->addWidget(color_smoothing_conservative);

        color_smoothing_laplacian = new QRadioButton(radio_grp_mesh_color);
        color_smoothing_laplacian->setObjectName(QStringLiteral("color_smoothing_laplacian"));

        verticalLayout_6->addWidget(color_smoothing_laplacian);

        base_potential = new QRadioButton(radio_grp_mesh_color);
        base_potential->setObjectName(QStringLiteral("base_potential"));
        base_potential->setChecked(true);

        verticalLayout_6->addWidget(base_potential);

        vertices_state = new QRadioButton(radio_grp_mesh_color);
        vertices_state->setObjectName(QStringLiteral("vertices_state"));

        verticalLayout_6->addWidget(vertices_state);

        implicit_gradient = new QRadioButton(radio_grp_mesh_color);
        implicit_gradient->setObjectName(QStringLiteral("implicit_gradient"));

        verticalLayout_6->addWidget(implicit_gradient);

        cluster = new QRadioButton(radio_grp_mesh_color);
        cluster->setObjectName(QStringLiteral("cluster"));

        verticalLayout_6->addWidget(cluster);

        color_nearest_joint = new QRadioButton(radio_grp_mesh_color);
        color_nearest_joint->setObjectName(QStringLiteral("color_nearest_joint"));

        verticalLayout_6->addWidget(color_nearest_joint);

        color_normals = new QRadioButton(radio_grp_mesh_color);
        color_normals->setObjectName(QStringLiteral("color_normals"));

        verticalLayout_6->addWidget(color_normals);

        color_grey = new QRadioButton(radio_grp_mesh_color);
        color_grey->setObjectName(QStringLiteral("color_grey"));

        verticalLayout_6->addWidget(color_grey);

        color_free_vertices = new QRadioButton(radio_grp_mesh_color);
        color_free_vertices->setObjectName(QStringLiteral("color_free_vertices"));

        verticalLayout_6->addWidget(color_free_vertices);

        color_edge_stress = new QRadioButton(radio_grp_mesh_color);
        color_edge_stress->setObjectName(QStringLiteral("color_edge_stress"));

        verticalLayout_6->addWidget(color_edge_stress);

        color_area_stress = new QRadioButton(radio_grp_mesh_color);
        color_area_stress->setObjectName(QStringLiteral("color_area_stress"));

        verticalLayout_6->addWidget(color_area_stress);


        vert_layout_box_mesh->addWidget(radio_grp_mesh_color);

        show_rbf_samples = new QCheckBox(box_animesh_color);
        show_rbf_samples->setObjectName(QStringLiteral("show_rbf_samples"));
        show_rbf_samples->setChecked(false);

        vert_layout_box_mesh->addWidget(show_rbf_samples);

        line_12 = new QFrame(box_animesh_color);
        line_12->setObjectName(QStringLiteral("line_12"));
        line_12->setFrameShape(QFrame::HLine);
        line_12->setFrameShadow(QFrame::Sunken);

        vert_layout_box_mesh->addWidget(line_12);

        lbl_select_vert = new QLabel(box_animesh_color);
        lbl_select_vert->setObjectName(QStringLiteral("lbl_select_vert"));

        vert_layout_box_mesh->addWidget(lbl_select_vert);

        select_vert_widget = new QWidget(box_animesh_color);
        select_vert_widget->setObjectName(QStringLiteral("select_vert_widget"));
        horizontalLayout_25 = new QHBoxLayout(select_vert_widget);
        horizontalLayout_25->setSpacing(6);
        horizontalLayout_25->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_25->setObjectName(QStringLiteral("horizontalLayout_25"));
        horizontalLayout_25->setContentsMargins(-1, 0, -1, 0);
        spinB_vert_id = new QSpinBox(select_vert_widget);
        spinB_vert_id->setObjectName(QStringLiteral("spinB_vert_id"));
        spinB_vert_id->setMaximum(1000000);

        horizontalLayout_25->addWidget(spinB_vert_id);

        pButton_do_select_vert = new QPushButton(select_vert_widget);
        pButton_do_select_vert->setObjectName(QStringLiteral("pButton_do_select_vert"));
        pButton_do_select_vert->setMaximumSize(QSize(35, 16777215));

        horizontalLayout_25->addWidget(pButton_do_select_vert);


        vert_layout_box_mesh->addWidget(select_vert_widget);


        verticalLayout_3->addLayout(vert_layout_box_mesh);


        vert_layout_display_tool_box->addWidget(box_animesh_color);

        box_mesh_color = new QGroupBox(Display_settings);
        box_mesh_color->setObjectName(QStringLiteral("box_mesh_color"));
        verticalLayout_36 = new QVBoxLayout(box_mesh_color);
        verticalLayout_36->setSpacing(6);
        verticalLayout_36->setContentsMargins(11, 11, 11, 11);
        verticalLayout_36->setObjectName(QStringLiteral("verticalLayout_36"));
        lbl_points_color = new QLabel(box_mesh_color);
        lbl_points_color->setObjectName(QStringLiteral("lbl_points_color"));
        lbl_points_color->setAlignment(Qt::AlignCenter);

        verticalLayout_36->addWidget(lbl_points_color);

        widget_6 = new QWidget(box_mesh_color);
        widget_6->setObjectName(QStringLiteral("widget_6"));
        verticalLayout_25 = new QVBoxLayout(widget_6);
        verticalLayout_25->setSpacing(6);
        verticalLayout_25->setContentsMargins(11, 11, 11, 11);
        verticalLayout_25->setObjectName(QStringLiteral("verticalLayout_25"));
        verticalLayout_25->setContentsMargins(-1, 0, -1, 0);
        buton_uniform_point_cl = new QRadioButton(widget_6);
        buton_uniform_point_cl->setObjectName(QStringLiteral("buton_uniform_point_cl"));
        buton_uniform_point_cl->setChecked(true);

        verticalLayout_25->addWidget(buton_uniform_point_cl);

        button_defects_point_cl = new QRadioButton(widget_6);
        button_defects_point_cl->setObjectName(QStringLiteral("button_defects_point_cl"));

        verticalLayout_25->addWidget(button_defects_point_cl);


        verticalLayout_36->addWidget(widget_6);

        widget = new QWidget(box_mesh_color);
        widget->setObjectName(QStringLiteral("widget"));
        verticalLayout_21 = new QVBoxLayout(widget);
        verticalLayout_21->setSpacing(6);
        verticalLayout_21->setContentsMargins(11, 11, 11, 11);
        verticalLayout_21->setObjectName(QStringLiteral("verticalLayout_21"));
        label = new QLabel(widget);
        label->setObjectName(QStringLiteral("label"));
        label->setAlignment(Qt::AlignCenter);

        verticalLayout_21->addWidget(label);

        horizontalSlider = new QSlider(widget);
        horizontalSlider->setObjectName(QStringLiteral("horizontalSlider"));
        horizontalSlider->setValue(50);
        horizontalSlider->setOrientation(Qt::Horizontal);
        horizontalSlider->setTickPosition(QSlider::TicksBelow);
        horizontalSlider->setTickInterval(10);

        verticalLayout_21->addWidget(horizontalSlider);


        verticalLayout_36->addWidget(widget);

        line = new QFrame(box_mesh_color);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        verticalLayout_36->addWidget(line);

        wireframe = new QCheckBox(box_mesh_color);
        wireframe->setObjectName(QStringLiteral("wireframe"));
        wireframe->setChecked(true);

        verticalLayout_36->addWidget(wireframe);


        vert_layout_display_tool_box->addWidget(box_mesh_color);

        box_skeleton = new QGroupBox(Display_settings);
        box_skeleton->setObjectName(QStringLiteral("box_skeleton"));
        verticalLayout_12 = new QVBoxLayout(box_skeleton);
        verticalLayout_12->setSpacing(6);
        verticalLayout_12->setContentsMargins(11, 11, 11, 11);
        verticalLayout_12->setObjectName(QStringLiteral("verticalLayout_12"));
        display_skeleton = new QCheckBox(box_skeleton);
        display_skeleton->setObjectName(QStringLiteral("display_skeleton"));
        display_skeleton->setChecked(true);

        verticalLayout_12->addWidget(display_skeleton);

        display_oriented_bbox = new QCheckBox(box_skeleton);
        display_oriented_bbox->setObjectName(QStringLiteral("display_oriented_bbox"));
        display_oriented_bbox->setChecked(false);

        verticalLayout_12->addWidget(display_oriented_bbox);

        checkB_aa_bbox = new QCheckBox(box_skeleton);
        checkB_aa_bbox->setObjectName(QStringLiteral("checkB_aa_bbox"));

        verticalLayout_12->addWidget(checkB_aa_bbox);

        checkB_draw_grid = new QCheckBox(box_skeleton);
        checkB_draw_grid->setObjectName(QStringLiteral("checkB_draw_grid"));

        verticalLayout_12->addWidget(checkB_draw_grid);


        vert_layout_display_tool_box->addWidget(box_skeleton);

        grpBox_operators = new QGroupBox(Display_settings);
        grpBox_operators->setObjectName(QStringLiteral("grpBox_operators"));
        grpBox_operators->setFlat(false);
        grpBox_operators->setCheckable(false);
        verticalLayout_13 = new QVBoxLayout(grpBox_operators);
        verticalLayout_13->setSpacing(6);
        verticalLayout_13->setContentsMargins(11, 11, 11, 11);
        verticalLayout_13->setObjectName(QStringLiteral("verticalLayout_13"));
        verticalLayout_13->setSizeConstraint(QLayout::SetDefaultConstraint);
        widget_12 = new QWidget(grpBox_operators);
        widget_12->setObjectName(QStringLiteral("widget_12"));
        verticalLayout_32 = new QVBoxLayout(widget_12);
        verticalLayout_32->setSpacing(6);
        verticalLayout_32->setContentsMargins(11, 11, 11, 11);
        verticalLayout_32->setObjectName(QStringLiteral("verticalLayout_32"));
        comboB_operators = new QComboBox(widget_12);
        comboB_operators->setObjectName(QStringLiteral("comboB_operators"));

        verticalLayout_32->addWidget(comboB_operators);

        display_operator = new QCheckBox(widget_12);
        display_operator->setObjectName(QStringLiteral("display_operator"));
        display_operator->setChecked(false);

        verticalLayout_32->addWidget(display_operator);

        widget_13 = new QWidget(widget_12);
        widget_13->setObjectName(QStringLiteral("widget_13"));
        horizontalLayout_20 = new QHBoxLayout(widget_13);
        horizontalLayout_20->setSpacing(6);
        horizontalLayout_20->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_20->setObjectName(QStringLiteral("horizontalLayout_20"));
        horizontalLayout_20->setContentsMargins(-1, 0, -1, 0);
        lbl_size_operator = new QLabel(widget_13);
        lbl_size_operator->setObjectName(QStringLiteral("lbl_size_operator"));

        horizontalLayout_20->addWidget(lbl_size_operator);

        spinBox = new QSpinBox(widget_13);
        spinBox->setObjectName(QStringLiteral("spinBox"));
        spinBox->setMinimum(16);
        spinBox->setMaximum(1024);
        spinBox->setSingleStep(30);
        spinBox->setValue(240);

        horizontalLayout_20->addWidget(spinBox);


        verticalLayout_32->addWidget(widget_13);

        widget_11 = new QWidget(widget_12);
        widget_11->setObjectName(QStringLiteral("widget_11"));
        horizontalLayout_19 = new QHBoxLayout(widget_11);
        horizontalLayout_19->setSpacing(6);
        horizontalLayout_19->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_19->setObjectName(QStringLiteral("horizontalLayout_19"));
        horizontalLayout_19->setContentsMargins(-1, 0, -1, 0);
        lbl_opening_angle = new QLabel(widget_11);
        lbl_opening_angle->setObjectName(QStringLiteral("lbl_opening_angle"));

        horizontalLayout_19->addWidget(lbl_opening_angle);

        dSpinB_opening_value = new QDoubleSpinBox(widget_11);
        dSpinB_opening_value->setObjectName(QStringLiteral("dSpinB_opening_value"));
        dSpinB_opening_value->setDecimals(2);
        dSpinB_opening_value->setMaximum(1);
        dSpinB_opening_value->setSingleStep(0.05);
        dSpinB_opening_value->setValue(0.5);

        horizontalLayout_19->addWidget(dSpinB_opening_value);


        verticalLayout_32->addWidget(widget_11);


        verticalLayout_13->addWidget(widget_12);

        display_controller = new QCheckBox(grpBox_operators);
        display_controller->setObjectName(QStringLiteral("display_controller"));
        display_controller->setChecked(false);

        verticalLayout_13->addWidget(display_controller);


        vert_layout_display_tool_box->addWidget(grpBox_operators);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        vert_layout_display_tool_box->addItem(verticalSpacer_2);


        horizontalLayout->addLayout(vert_layout_display_tool_box);

        horizontal_spacer_2 = new QSpacerItem(13, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontal_spacer_2);

        toolBoxMenu->addItem(Display_settings, QStringLiteral("Display"));
        graph_edition = new QWidget();
        graph_edition->setObjectName(QStringLiteral("graph_edition"));
        graph_edition->setGeometry(QRect(0, 0, 265, 588));
        horizontalLayout_8 = new QHBoxLayout(graph_edition);
        horizontalLayout_8->setSpacing(0);
        horizontalLayout_8->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_8->setObjectName(QStringLiteral("horizontalLayout_8"));
        horizontalLayout_8->setContentsMargins(0, -1, 0, -1);
        vert_layout_graph_edit = new QVBoxLayout();
        vert_layout_graph_edit->setSpacing(0);
        vert_layout_graph_edit->setObjectName(QStringLiteral("vert_layout_graph_edit"));
        groupBox_2 = new QGroupBox(graph_edition);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        verticalLayout_19 = new QVBoxLayout(groupBox_2);
        verticalLayout_19->setSpacing(6);
        verticalLayout_19->setContentsMargins(11, 11, 11, 11);
        verticalLayout_19->setObjectName(QStringLiteral("verticalLayout_19"));
        pushB_attached_skeleton = new QPushButton(groupBox_2);
        pushB_attached_skeleton->setObjectName(QStringLiteral("pushB_attached_skeleton"));

        verticalLayout_19->addWidget(pushB_attached_skeleton);

        center_graph_node = new QPushButton(groupBox_2);
        center_graph_node->setObjectName(QStringLiteral("center_graph_node"));

        verticalLayout_19->addWidget(center_graph_node);


        vert_layout_graph_edit->addWidget(groupBox_2);

        verticalSpacer_4 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        vert_layout_graph_edit->addItem(verticalSpacer_4);


        horizontalLayout_8->addLayout(vert_layout_graph_edit);

        horizontalSpacer_4 = new QSpacerItem(101, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_8->addItem(horizontalSpacer_4);

        toolBoxMenu->addItem(graph_edition, QStringLiteral("Graph edition"));
        Animation_settings = new QWidget();
        Animation_settings->setObjectName(QStringLiteral("Animation_settings"));
        Animation_settings->setGeometry(QRect(0, 0, 265, 588));
        horizontalLayout_2 = new QHBoxLayout(Animation_settings);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        vert_layout_anim_toolbox = new QVBoxLayout();
        vert_layout_anim_toolbox->setSpacing(0);
        vert_layout_anim_toolbox->setObjectName(QStringLiteral("vert_layout_anim_toolbox"));
        box_anim_type = new QGroupBox(Animation_settings);
        box_anim_type->setObjectName(QStringLiteral("box_anim_type"));
        verticalLayout_10 = new QVBoxLayout(box_anim_type);
        verticalLayout_10->setSpacing(0);
        verticalLayout_10->setContentsMargins(11, 11, 11, 11);
        verticalLayout_10->setObjectName(QStringLiteral("verticalLayout_10"));
        verticalLayout_10->setContentsMargins(0, -1, 0, -1);
        radio_grp_3 = new QWidget(box_anim_type);
        radio_grp_3->setObjectName(QStringLiteral("radio_grp_3"));
        verticalLayout_11 = new QVBoxLayout(radio_grp_3);
        verticalLayout_11->setSpacing(0);
        verticalLayout_11->setContentsMargins(11, 11, 11, 11);
        verticalLayout_11->setObjectName(QStringLiteral("verticalLayout_11"));
        ssd_raio = new QRadioButton(radio_grp_3);
        ssd_raio->setObjectName(QStringLiteral("ssd_raio"));
        ssd_raio->setChecked(false);

        verticalLayout_11->addWidget(ssd_raio);

        dual_quaternion_radio = new QRadioButton(radio_grp_3);
        dual_quaternion_radio->setObjectName(QStringLiteral("dual_quaternion_radio"));
        dual_quaternion_radio->setChecked(true);

        verticalLayout_11->addWidget(dual_quaternion_radio);

        implicit_skinning_checkBox = new QCheckBox(radio_grp_3);
        implicit_skinning_checkBox->setObjectName(QStringLiteral("implicit_skinning_checkBox"));

        verticalLayout_11->addWidget(implicit_skinning_checkBox);

        checkBox_incremental = new QCheckBox(radio_grp_3);
        checkBox_incremental->setObjectName(QStringLiteral("checkBox_incremental"));
        checkBox_incremental->setChecked(false);

        verticalLayout_11->addWidget(checkBox_incremental);


        verticalLayout_10->addWidget(radio_grp_3);

        reset_anim = new QPushButton(box_anim_type);
        reset_anim->setObjectName(QStringLiteral("reset_anim"));

        verticalLayout_10->addWidget(reset_anim);


        vert_layout_anim_toolbox->addWidget(box_anim_type);

        grpBox_smoothing_weights = new QGroupBox(Animation_settings);
        grpBox_smoothing_weights->setObjectName(QStringLiteral("grpBox_smoothing_weights"));
        verticalLayout_23 = new QVBoxLayout(grpBox_smoothing_weights);
        verticalLayout_23->setSpacing(6);
        verticalLayout_23->setContentsMargins(11, 11, 11, 11);
        verticalLayout_23->setObjectName(QStringLiteral("verticalLayout_23"));
        verticalLayout_23->setContentsMargins(-1, 5, -1, -1);
        widget_smooth_weights_diffuse_iter = new QWidget(grpBox_smoothing_weights);
        widget_smooth_weights_diffuse_iter->setObjectName(QStringLiteral("widget_smooth_weights_diffuse_iter"));
        formLayout_2 = new QFormLayout(widget_smooth_weights_diffuse_iter);
        formLayout_2->setSpacing(6);
        formLayout_2->setContentsMargins(11, 11, 11, 11);
        formLayout_2->setObjectName(QStringLiteral("formLayout_2"));
        formLayout_2->setHorizontalSpacing(0);
        formLayout_2->setContentsMargins(-1, 0, -1, 0);
        spinBox_diffuse_smoothing_weights_iter = new QSpinBox(widget_smooth_weights_diffuse_iter);
        spinBox_diffuse_smoothing_weights_iter->setObjectName(QStringLiteral("spinBox_diffuse_smoothing_weights_iter"));
        spinBox_diffuse_smoothing_weights_iter->setMinimum(1);
        spinBox_diffuse_smoothing_weights_iter->setMaximum(200);
        spinBox_diffuse_smoothing_weights_iter->setValue(6);

        formLayout_2->setWidget(0, QFormLayout::FieldRole, spinBox_diffuse_smoothing_weights_iter);

        lbl_nb_iter_smoothing_weight_diffusion = new QLabel(widget_smooth_weights_diffuse_iter);
        lbl_nb_iter_smoothing_weight_diffusion->setObjectName(QStringLiteral("lbl_nb_iter_smoothing_weight_diffusion"));

        formLayout_2->setWidget(0, QFormLayout::LabelRole, lbl_nb_iter_smoothing_weight_diffusion);


        verticalLayout_23->addWidget(widget_smooth_weights_diffuse_iter);


        vert_layout_anim_toolbox->addWidget(grpBox_smoothing_weights);

        grpBox_weights = new QGroupBox(Animation_settings);
        grpBox_weights->setObjectName(QStringLiteral("grpBox_weights"));
        verticalLayout_18 = new QVBoxLayout(grpBox_weights);
        verticalLayout_18->setSpacing(6);
        verticalLayout_18->setContentsMargins(11, 11, 11, 11);
        verticalLayout_18->setObjectName(QStringLiteral("verticalLayout_18"));
        verticalLayout_18->setContentsMargins(0, -1, 0, -1);
        pushB_set_rigid_weights = new QPushButton(grpBox_weights);
        pushB_set_rigid_weights->setObjectName(QStringLiteral("pushB_set_rigid_weights"));

        verticalLayout_18->addWidget(pushB_set_rigid_weights);

        line_11 = new QFrame(grpBox_weights);
        line_11->setObjectName(QStringLiteral("line_11"));
        line_11->setFrameShape(QFrame::HLine);
        line_11->setFrameShadow(QFrame::Sunken);

        verticalLayout_18->addWidget(line_11);

        lbl_auto_weights_experimental = new QLabel(grpBox_weights);
        lbl_auto_weights_experimental->setObjectName(QStringLiteral("lbl_auto_weights_experimental"));

        verticalLayout_18->addWidget(lbl_auto_weights_experimental);

        grp_widget_auto_weight = new QWidget(grpBox_weights);
        grp_widget_auto_weight->setObjectName(QStringLiteral("grp_widget_auto_weight"));
        horizontalLayout_13 = new QHBoxLayout(grp_widget_auto_weight);
        horizontalLayout_13->setSpacing(6);
        horizontalLayout_13->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_13->setObjectName(QStringLiteral("horizontalLayout_13"));
        widget_5 = new QWidget(grp_widget_auto_weight);
        widget_5->setObjectName(QStringLiteral("widget_5"));
        verticalLayout_24 = new QVBoxLayout(widget_5);
        verticalLayout_24->setSpacing(6);
        verticalLayout_24->setContentsMargins(11, 11, 11, 11);
        verticalLayout_24->setObjectName(QStringLiteral("verticalLayout_24"));
        verticalLayout_24->setContentsMargins(0, -1, 0, -1);
        dSpinB_diff_w_alpha_exp = new QDoubleSpinBox(widget_5);
        dSpinB_diff_w_alpha_exp->setObjectName(QStringLiteral("dSpinB_diff_w_alpha_exp"));
        dSpinB_diff_w_alpha_exp->setMinimum(-100);
        dSpinB_diff_w_alpha_exp->setMaximum(100);
        dSpinB_diff_w_alpha_exp->setSingleStep(0.2);
        dSpinB_diff_w_alpha_exp->setValue(1);

        verticalLayout_24->addWidget(dSpinB_diff_w_alpha_exp);

        spinB_auto_w_nb_iter_exp = new QSpinBox(widget_5);
        spinB_auto_w_nb_iter_exp->setObjectName(QStringLiteral("spinB_auto_w_nb_iter_exp"));
        spinB_auto_w_nb_iter_exp->setMaximum(200);
        spinB_auto_w_nb_iter_exp->setValue(1);

        verticalLayout_24->addWidget(spinB_auto_w_nb_iter_exp);


        horizontalLayout_13->addWidget(widget_5);

        pushB_diff_w_exp = new QPushButton(grp_widget_auto_weight);
        pushB_diff_w_exp->setObjectName(QStringLiteral("pushB_diff_w_exp"));

        horizontalLayout_13->addWidget(pushB_diff_w_exp);


        verticalLayout_18->addWidget(grp_widget_auto_weight);

        line_7 = new QFrame(grpBox_weights);
        line_7->setObjectName(QStringLiteral("line_7"));
        line_7->setFrameShape(QFrame::HLine);
        line_7->setFrameShadow(QFrame::Sunken);

        verticalLayout_18->addWidget(line_7);

        lbl_diffuse_weights = new QLabel(grpBox_weights);
        lbl_diffuse_weights->setObjectName(QStringLiteral("lbl_diffuse_weights"));

        verticalLayout_18->addWidget(lbl_diffuse_weights);

        widget_9 = new QWidget(grpBox_weights);
        widget_9->setObjectName(QStringLiteral("widget_9"));
        horizontalLayout_16 = new QHBoxLayout(widget_9);
        horizontalLayout_16->setSpacing(6);
        horizontalLayout_16->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_16->setObjectName(QStringLiteral("horizontalLayout_16"));
        horizontalLayout_16->setContentsMargins(0, -1, 0, -1);
        widget_10 = new QWidget(widget_9);
        widget_10->setObjectName(QStringLiteral("widget_10"));
        verticalLayout_28 = new QVBoxLayout(widget_10);
        verticalLayout_28->setSpacing(6);
        verticalLayout_28->setContentsMargins(11, 11, 11, 11);
        verticalLayout_28->setObjectName(QStringLiteral("verticalLayout_28"));
        dSpinB_diff_w_alpha = new QDoubleSpinBox(widget_10);
        dSpinB_diff_w_alpha->setObjectName(QStringLiteral("dSpinB_diff_w_alpha"));
        dSpinB_diff_w_alpha->setMaximum(1);
        dSpinB_diff_w_alpha->setSingleStep(0.1);
        dSpinB_diff_w_alpha->setValue(1);

        verticalLayout_28->addWidget(dSpinB_diff_w_alpha);

        spinB_diff_w_nb_iter = new QSpinBox(widget_10);
        spinB_diff_w_nb_iter->setObjectName(QStringLiteral("spinB_diff_w_nb_iter"));
        spinB_diff_w_nb_iter->setMaximum(30);
        spinB_diff_w_nb_iter->setValue(3);

        verticalLayout_28->addWidget(spinB_diff_w_nb_iter);


        horizontalLayout_16->addWidget(widget_10);

        pushB_diffuse_curr_weights = new QPushButton(widget_9);
        pushB_diffuse_curr_weights->setObjectName(QStringLiteral("pushB_diffuse_curr_weights"));

        horizontalLayout_16->addWidget(pushB_diffuse_curr_weights);


        verticalLayout_18->addWidget(widget_9);

        line_10 = new QFrame(grpBox_weights);
        line_10->setObjectName(QStringLiteral("line_10"));
        line_10->setFrameShape(QFrame::HLine);
        line_10->setFrameShadow(QFrame::Sunken);

        verticalLayout_18->addWidget(line_10);

        lbl_heat_diffusion_weights = new QLabel(grpBox_weights);
        lbl_heat_diffusion_weights->setObjectName(QStringLiteral("lbl_heat_diffusion_weights"));

        verticalLayout_18->addWidget(lbl_heat_diffusion_weights);

        grp_widget_heat_diffusion = new QWidget(grpBox_weights);
        grp_widget_heat_diffusion->setObjectName(QStringLiteral("grp_widget_heat_diffusion"));
        horizontalLayout_23 = new QHBoxLayout(grp_widget_heat_diffusion);
        horizontalLayout_23->setSpacing(6);
        horizontalLayout_23->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_23->setObjectName(QStringLiteral("horizontalLayout_23"));
        dSpinBox_heat_coeff = new QDoubleSpinBox(grp_widget_heat_diffusion);
        dSpinBox_heat_coeff->setObjectName(QStringLiteral("dSpinBox_heat_coeff"));
        dSpinBox_heat_coeff->setMaximum(10);
        dSpinBox_heat_coeff->setSingleStep(0.1);
        dSpinBox_heat_coeff->setValue(1);

        horizontalLayout_23->addWidget(dSpinBox_heat_coeff);

        pButton_compute_heat_difusion = new QPushButton(grp_widget_heat_diffusion);
        pButton_compute_heat_difusion->setObjectName(QStringLiteral("pButton_compute_heat_difusion"));

        horizontalLayout_23->addWidget(pButton_compute_heat_difusion);


        verticalLayout_18->addWidget(grp_widget_heat_diffusion);


        vert_layout_anim_toolbox->addWidget(grpBox_weights);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        vert_layout_anim_toolbox->addItem(verticalSpacer);


        horizontalLayout_2->addLayout(vert_layout_anim_toolbox);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);

        toolBoxMenu->addItem(Animation_settings, QStringLiteral("Animation"));
        blending_settings = new QWidget();
        blending_settings->setObjectName(QStringLiteral("blending_settings"));
        blending_settings->setGeometry(QRect(0, 0, 265, 588));
        horizontalLayout_3 = new QHBoxLayout(blending_settings);
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        vert_layout_blending = new QVBoxLayout();
        vert_layout_blending->setSpacing(6);
        vert_layout_blending->setObjectName(QStringLiteral("vert_layout_blending"));
        grpBox_bulge_in_contact = new QGroupBox(blending_settings);
        grpBox_bulge_in_contact->setObjectName(QStringLiteral("grpBox_bulge_in_contact"));
        verticalLayout_14 = new QVBoxLayout(grpBox_bulge_in_contact);
        verticalLayout_14->setSpacing(0);
        verticalLayout_14->setContentsMargins(11, 11, 11, 11);
        verticalLayout_14->setObjectName(QStringLiteral("verticalLayout_14"));
        verticalLayout_14->setContentsMargins(0, -1, 0, -1);
        widget_bulge_in_contact = new QWidget(grpBox_bulge_in_contact);
        widget_bulge_in_contact->setObjectName(QStringLiteral("widget_bulge_in_contact"));
        horizontalLayout_6 = new QHBoxLayout(widget_bulge_in_contact);
        horizontalLayout_6->setSpacing(6);
        horizontalLayout_6->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        lbl_force = new QLabel(widget_bulge_in_contact);
        lbl_force->setObjectName(QStringLiteral("lbl_force"));

        horizontalLayout_6->addWidget(lbl_force);

        spinBox_bulge_in_contact_force = new QDoubleSpinBox(widget_bulge_in_contact);
        spinBox_bulge_in_contact_force->setObjectName(QStringLiteral("spinBox_bulge_in_contact_force"));
        spinBox_bulge_in_contact_force->setDecimals(2);
        spinBox_bulge_in_contact_force->setMaximum(1);
        spinBox_bulge_in_contact_force->setSingleStep(0.05);
        spinBox_bulge_in_contact_force->setValue(0.7);

        horizontalLayout_6->addWidget(spinBox_bulge_in_contact_force);


        verticalLayout_14->addWidget(widget_bulge_in_contact);

        update_bulge_in_contact = new QPushButton(grpBox_bulge_in_contact);
        update_bulge_in_contact->setObjectName(QStringLiteral("update_bulge_in_contact"));

        verticalLayout_14->addWidget(update_bulge_in_contact);


        vert_layout_blending->addWidget(grpBox_bulge_in_contact);

        grpBox_controller = new QGroupBox(blending_settings);
        grpBox_controller->setObjectName(QStringLiteral("grpBox_controller"));
        verticalLayout_17 = new QVBoxLayout(grpBox_controller);
        verticalLayout_17->setSpacing(0);
        verticalLayout_17->setContentsMargins(11, 11, 11, 11);
        verticalLayout_17->setObjectName(QStringLiteral("verticalLayout_17"));
        verticalLayout_17->setContentsMargins(0, 7, 0, 7);
        preset_ctrl = new QWidget(grpBox_controller);
        preset_ctrl->setObjectName(QStringLiteral("preset_ctrl"));

        verticalLayout_17->addWidget(preset_ctrl);


        vert_layout_blending->addWidget(grpBox_controller);

        gBox_controller_values = new QGroupBox(blending_settings);
        gBox_controller_values->setObjectName(QStringLiteral("gBox_controller_values"));
        verticalLayout_16 = new QVBoxLayout(gBox_controller_values);
        verticalLayout_16->setSpacing(6);
        verticalLayout_16->setContentsMargins(11, 11, 11, 11);
        verticalLayout_16->setObjectName(QStringLiteral("verticalLayout_16"));
        verticalLayout_16->setContentsMargins(0, -1, 0, -1);
        layout_ctrl_p0 = new QHBoxLayout();
        layout_ctrl_p0->setSpacing(0);
        layout_ctrl_p0->setObjectName(QStringLiteral("layout_ctrl_p0"));
        lbl_p0 = new QLabel(gBox_controller_values);
        lbl_p0->setObjectName(QStringLiteral("lbl_p0"));

        layout_ctrl_p0->addWidget(lbl_p0);

        dSpinB_ctrl_p0_x = new QDoubleSpinBox(gBox_controller_values);
        dSpinB_ctrl_p0_x->setObjectName(QStringLiteral("dSpinB_ctrl_p0_x"));
        dSpinB_ctrl_p0_x->setMinimum(-3.14);
        dSpinB_ctrl_p0_x->setMaximum(6.28);
        dSpinB_ctrl_p0_x->setSingleStep(0.05);

        layout_ctrl_p0->addWidget(dSpinB_ctrl_p0_x);

        dSpinB_ctrl_p0_y = new QDoubleSpinBox(gBox_controller_values);
        dSpinB_ctrl_p0_y->setObjectName(QStringLiteral("dSpinB_ctrl_p0_y"));
        dSpinB_ctrl_p0_y->setMaximum(0.79);
        dSpinB_ctrl_p0_y->setSingleStep(0.05);

        layout_ctrl_p0->addWidget(dSpinB_ctrl_p0_y);


        verticalLayout_16->addLayout(layout_ctrl_p0);

        layout_ctrl_p1 = new QHBoxLayout();
        layout_ctrl_p1->setSpacing(0);
        layout_ctrl_p1->setObjectName(QStringLiteral("layout_ctrl_p1"));
        layout_ctrl_p1->setContentsMargins(0, 0, -1, -1);
        lbl_p1 = new QLabel(gBox_controller_values);
        lbl_p1->setObjectName(QStringLiteral("lbl_p1"));

        layout_ctrl_p1->addWidget(lbl_p1);

        dSpinB_ctrl_p1_x = new QDoubleSpinBox(gBox_controller_values);
        dSpinB_ctrl_p1_x->setObjectName(QStringLiteral("dSpinB_ctrl_p1_x"));
        dSpinB_ctrl_p1_x->setMinimum(-3.14);
        dSpinB_ctrl_p1_x->setMaximum(6.28);
        dSpinB_ctrl_p1_x->setSingleStep(0.05);

        layout_ctrl_p1->addWidget(dSpinB_ctrl_p1_x);

        dSpinB_ctrl_p1_y = new QDoubleSpinBox(gBox_controller_values);
        dSpinB_ctrl_p1_y->setObjectName(QStringLiteral("dSpinB_ctrl_p1_y"));
        dSpinB_ctrl_p1_y->setMaximum(0.79);
        dSpinB_ctrl_p1_y->setSingleStep(0.05);

        layout_ctrl_p1->addWidget(dSpinB_ctrl_p1_y);


        verticalLayout_16->addLayout(layout_ctrl_p1);

        layout_ctrl_p2 = new QHBoxLayout();
        layout_ctrl_p2->setSpacing(0);
        layout_ctrl_p2->setObjectName(QStringLiteral("layout_ctrl_p2"));
        layout_ctrl_p2->setContentsMargins(-1, 0, 0, -1);
        lbl_p2 = new QLabel(gBox_controller_values);
        lbl_p2->setObjectName(QStringLiteral("lbl_p2"));

        layout_ctrl_p2->addWidget(lbl_p2);

        dSpinB_ctrl_p2_x = new QDoubleSpinBox(gBox_controller_values);
        dSpinB_ctrl_p2_x->setObjectName(QStringLiteral("dSpinB_ctrl_p2_x"));
        dSpinB_ctrl_p2_x->setMinimum(-3.14);
        dSpinB_ctrl_p2_x->setMaximum(6.28);
        dSpinB_ctrl_p2_x->setSingleStep(0.05);

        layout_ctrl_p2->addWidget(dSpinB_ctrl_p2_x);

        dSpinB_ctrl_p2_y = new QDoubleSpinBox(gBox_controller_values);
        dSpinB_ctrl_p2_y->setObjectName(QStringLiteral("dSpinB_ctrl_p2_y"));
        dSpinB_ctrl_p2_y->setMaximum(0.79);
        dSpinB_ctrl_p2_y->setSingleStep(0.05);

        layout_ctrl_p2->addWidget(dSpinB_ctrl_p2_y);


        verticalLayout_16->addLayout(layout_ctrl_p2);

        layout_ctrl_slopes = new QHBoxLayout();
        layout_ctrl_slopes->setSpacing(0);
        layout_ctrl_slopes->setObjectName(QStringLiteral("layout_ctrl_slopes"));
        layout_ctrl_slopes->setContentsMargins(0, 0, -1, -1);
        lbl_slopes = new QLabel(gBox_controller_values);
        lbl_slopes->setObjectName(QStringLiteral("lbl_slopes"));

        layout_ctrl_slopes->addWidget(lbl_slopes);

        dSpinB_ctrl_slope0 = new QDoubleSpinBox(gBox_controller_values);
        dSpinB_ctrl_slope0->setObjectName(QStringLiteral("dSpinB_ctrl_slope0"));
        dSpinB_ctrl_slope0->setDecimals(2);
        dSpinB_ctrl_slope0->setMinimum(-4);
        dSpinB_ctrl_slope0->setMaximum(12.99);
        dSpinB_ctrl_slope0->setSingleStep(0.05);

        layout_ctrl_slopes->addWidget(dSpinB_ctrl_slope0);

        dSpinB_ctrl_slope1 = new QDoubleSpinBox(gBox_controller_values);
        dSpinB_ctrl_slope1->setObjectName(QStringLiteral("dSpinB_ctrl_slope1"));
        dSpinB_ctrl_slope1->setMinimum(-4);
        dSpinB_ctrl_slope1->setMaximum(12.99);
        dSpinB_ctrl_slope1->setSingleStep(0.05);

        layout_ctrl_slopes->addWidget(dSpinB_ctrl_slope1);


        verticalLayout_16->addLayout(layout_ctrl_slopes);


        vert_layout_blending->addWidget(gBox_controller_values);

        pushB_edit_spline = new QPushButton(blending_settings);
        pushB_edit_spline->setObjectName(QStringLiteral("pushB_edit_spline"));

        vert_layout_blending->addWidget(pushB_edit_spline);

        blending_vertical_spacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        vert_layout_blending->addItem(blending_vertical_spacer);


        horizontalLayout_3->addLayout(vert_layout_blending);

        blending_horizontal_spacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(blending_horizontal_spacer);

        toolBoxMenu->addItem(blending_settings, QStringLiteral("Blending"));
        bone_editor = new QWidget();
        bone_editor->setObjectName(QStringLiteral("bone_editor"));
        bone_editor->setGeometry(QRect(0, 0, 248, 633));
        horizontalLayout_4 = new QHBoxLayout(bone_editor);
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        vert_layout_bone_editor = new QVBoxLayout();
        vert_layout_bone_editor->setSpacing(6);
        vert_layout_bone_editor->setObjectName(QStringLiteral("vert_layout_bone_editor"));
        box_edit_RBF = new QGroupBox(bone_editor);
        box_edit_RBF->setObjectName(QStringLiteral("box_edit_RBF"));
        verticalLayout_15 = new QVBoxLayout(box_edit_RBF);
        verticalLayout_15->setSpacing(6);
        verticalLayout_15->setContentsMargins(11, 11, 11, 11);
        verticalLayout_15->setObjectName(QStringLiteral("verticalLayout_15"));
        rbf_edition = new QCheckBox(box_edit_RBF);
        rbf_edition->setObjectName(QStringLiteral("rbf_edition"));

        verticalLayout_15->addWidget(rbf_edition);

        cBox_always_precompute = new QCheckBox(box_edit_RBF);
        cBox_always_precompute->setObjectName(QStringLiteral("cBox_always_precompute"));
        cBox_always_precompute->setChecked(true);

        verticalLayout_15->addWidget(cBox_always_precompute);

        checkB_factor_siblings = new QCheckBox(box_edit_RBF);
        checkB_factor_siblings->setObjectName(QStringLiteral("checkB_factor_siblings"));

        verticalLayout_15->addWidget(checkB_factor_siblings);

        local_frame = new QCheckBox(box_edit_RBF);
        local_frame->setObjectName(QStringLiteral("local_frame"));

        verticalLayout_15->addWidget(local_frame);

        checkB_align_with_normal = new QCheckBox(box_edit_RBF);
        checkB_align_with_normal->setObjectName(QStringLiteral("checkB_align_with_normal"));
        checkB_align_with_normal->setChecked(true);

        verticalLayout_15->addWidget(checkB_align_with_normal);

        move_joints = new QCheckBox(box_edit_RBF);
        move_joints->setObjectName(QStringLiteral("move_joints"));

        verticalLayout_15->addWidget(move_joints);

        checkB_show_junction = new QCheckBox(box_edit_RBF);
        checkB_show_junction->setObjectName(QStringLiteral("checkB_show_junction"));

        verticalLayout_15->addWidget(checkB_show_junction);

        pushB_empty_bone = new QPushButton(box_edit_RBF);
        pushB_empty_bone->setObjectName(QStringLiteral("pushB_empty_bone"));

        verticalLayout_15->addWidget(pushB_empty_bone);

        widget_24 = new QWidget(box_edit_RBF);
        widget_24->setObjectName(QStringLiteral("widget_24"));
        horizontalLayout_30 = new QHBoxLayout(widget_24);
        horizontalLayout_30->setSpacing(0);
        horizontalLayout_30->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_30->setObjectName(QStringLiteral("horizontalLayout_30"));
        horizontalLayout_30->setContentsMargins(0, -1, 0, -1);
        pButton_add_caps = new QPushButton(widget_24);
        pButton_add_caps->setObjectName(QStringLiteral("pButton_add_caps"));

        horizontalLayout_30->addWidget(pButton_add_caps);

        pButton_supr_caps = new QPushButton(widget_24);
        pButton_supr_caps->setObjectName(QStringLiteral("pButton_supr_caps"));

        horizontalLayout_30->addWidget(pButton_supr_caps);


        verticalLayout_15->addWidget(widget_24);

        widget_16 = new QWidget(box_edit_RBF);
        widget_16->setObjectName(QStringLiteral("widget_16"));
        horizontalLayout_7 = new QHBoxLayout(widget_16);
        horizontalLayout_7->setSpacing(0);
        horizontalLayout_7->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        horizontalLayout_7->setContentsMargins(0, -1, 0, -1);
        lbl_hrbf_radius = new QLabel(widget_16);
        lbl_hrbf_radius->setObjectName(QStringLiteral("lbl_hrbf_radius"));

        horizontalLayout_7->addWidget(lbl_hrbf_radius);

        dSpinB_hrbf_radius = new QDoubleSpinBox(widget_16);
        dSpinB_hrbf_radius->setObjectName(QStringLiteral("dSpinB_hrbf_radius"));
        dSpinB_hrbf_radius->setSingleStep(0.5);
        dSpinB_hrbf_radius->setValue(1);

        horizontalLayout_7->addWidget(dSpinB_hrbf_radius);


        verticalLayout_15->addWidget(widget_16);

        groupBx_auto_sampling = new QGroupBox(box_edit_RBF);
        groupBx_auto_sampling->setObjectName(QStringLiteral("groupBx_auto_sampling"));
        verticalLayout_29 = new QVBoxLayout(groupBx_auto_sampling);
        verticalLayout_29->setSpacing(0);
        verticalLayout_29->setContentsMargins(11, 11, 11, 11);
        verticalLayout_29->setObjectName(QStringLiteral("verticalLayout_29"));
        verticalLayout_29->setContentsMargins(0, -1, 0, -1);
        lbl_per_max_dist_from_joints = new QLabel(groupBx_auto_sampling);
        lbl_per_max_dist_from_joints->setObjectName(QStringLiteral("lbl_per_max_dist_from_joints"));

        verticalLayout_29->addWidget(lbl_per_max_dist_from_joints);

        widget_grp_max_dist_from_joints = new QWidget(groupBx_auto_sampling);
        widget_grp_max_dist_from_joints->setObjectName(QStringLiteral("widget_grp_max_dist_from_joints"));
        horizontalLayout_17 = new QHBoxLayout(widget_grp_max_dist_from_joints);
        horizontalLayout_17->setSpacing(6);
        horizontalLayout_17->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_17->setObjectName(QStringLiteral("horizontalLayout_17"));
        horizontalLayout_17->setContentsMargins(0, -1, 0, -1);
        widget_grp_joint = new QWidget(widget_grp_max_dist_from_joints);
        widget_grp_joint->setObjectName(QStringLiteral("widget_grp_joint"));
        verticalLayout_31 = new QVBoxLayout(widget_grp_joint);
        verticalLayout_31->setSpacing(0);
        verticalLayout_31->setContentsMargins(11, 11, 11, 11);
        verticalLayout_31->setObjectName(QStringLiteral("verticalLayout_31"));
        verticalLayout_31->setContentsMargins(0, -1, 0, -1);
        lbl_max_dist_joint = new QLabel(widget_grp_joint);
        lbl_max_dist_joint->setObjectName(QStringLiteral("lbl_max_dist_joint"));

        verticalLayout_31->addWidget(lbl_max_dist_joint);

        dSpinB_max_dist_joint = new QDoubleSpinBox(widget_grp_joint);
        dSpinB_max_dist_joint->setObjectName(QStringLiteral("dSpinB_max_dist_joint"));
        dSpinB_max_dist_joint->setMinimum(-1);
        dSpinB_max_dist_joint->setMaximum(1);
        dSpinB_max_dist_joint->setSingleStep(0.1);
        dSpinB_max_dist_joint->setValue(-0.02);

        verticalLayout_31->addWidget(dSpinB_max_dist_joint);

        checkB_cap_joint = new QCheckBox(widget_grp_joint);
        checkB_cap_joint->setObjectName(QStringLiteral("checkB_cap_joint"));
        checkB_cap_joint->setChecked(false);

        verticalLayout_31->addWidget(checkB_cap_joint);


        horizontalLayout_17->addWidget(widget_grp_joint);

        widget_grp_parent = new QWidget(widget_grp_max_dist_from_joints);
        widget_grp_parent->setObjectName(QStringLiteral("widget_grp_parent"));
        verticalLayout_30 = new QVBoxLayout(widget_grp_parent);
        verticalLayout_30->setSpacing(0);
        verticalLayout_30->setContentsMargins(11, 11, 11, 11);
        verticalLayout_30->setObjectName(QStringLiteral("verticalLayout_30"));
        verticalLayout_30->setContentsMargins(0, -1, 0, -1);
        lbl_max_dist_parent = new QLabel(widget_grp_parent);
        lbl_max_dist_parent->setObjectName(QStringLiteral("lbl_max_dist_parent"));

        verticalLayout_30->addWidget(lbl_max_dist_parent);

        dSpinB_max_dist_parent = new QDoubleSpinBox(widget_grp_parent);
        dSpinB_max_dist_parent->setObjectName(QStringLiteral("dSpinB_max_dist_parent"));
        dSpinB_max_dist_parent->setMinimum(-1);
        dSpinB_max_dist_parent->setMaximum(1);
        dSpinB_max_dist_parent->setSingleStep(0.1);
        dSpinB_max_dist_parent->setValue(-0.02);

        verticalLayout_30->addWidget(dSpinB_max_dist_parent);

        checkB_capparent = new QCheckBox(widget_grp_parent);
        checkB_capparent->setObjectName(QStringLiteral("checkB_capparent"));
        checkB_capparent->setChecked(false);

        verticalLayout_30->addWidget(checkB_capparent);


        horizontalLayout_17->addWidget(widget_grp_parent);


        verticalLayout_29->addWidget(widget_grp_max_dist_from_joints);

        widget_grp_max_fold = new QWidget(groupBx_auto_sampling);
        widget_grp_max_fold->setObjectName(QStringLiteral("widget_grp_max_fold"));
        horizontalLayout_18 = new QHBoxLayout(widget_grp_max_fold);
        horizontalLayout_18->setSpacing(0);
        horizontalLayout_18->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_18->setObjectName(QStringLiteral("horizontalLayout_18"));
        horizontalLayout_18->setContentsMargins(0, -1, 0, -1);
        lbl_max_fold = new QLabel(widget_grp_max_fold);
        lbl_max_fold->setObjectName(QStringLiteral("lbl_max_fold"));

        horizontalLayout_18->addWidget(lbl_max_fold);

        dSpinB_max_fold = new QDoubleSpinBox(widget_grp_max_fold);
        dSpinB_max_fold->setObjectName(QStringLiteral("dSpinB_max_fold"));
        dSpinB_max_fold->setMinimum(-1);
        dSpinB_max_fold->setMaximum(1);
        dSpinB_max_fold->setSingleStep(0.1);

        horizontalLayout_18->addWidget(dSpinB_max_fold);


        verticalLayout_29->addWidget(widget_grp_max_fold);

        widget_14 = new QWidget(groupBx_auto_sampling);
        widget_14->setObjectName(QStringLiteral("widget_14"));
        horizontalLayout_21 = new QHBoxLayout(widget_14);
        horizontalLayout_21->setSpacing(0);
        horizontalLayout_21->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_21->setObjectName(QStringLiteral("horizontalLayout_21"));
        horizontalLayout_21->setContentsMargins(0, -1, 0, -1);
        label_2 = new QLabel(widget_14);
        label_2->setObjectName(QStringLiteral("label_2"));

        horizontalLayout_21->addWidget(label_2);

        dSpinB_min_dist_samples = new QDoubleSpinBox(widget_14);
        dSpinB_min_dist_samples->setObjectName(QStringLiteral("dSpinB_min_dist_samples"));
        dSpinB_min_dist_samples->setMaximum(999.99);
        dSpinB_min_dist_samples->setSingleStep(0.01);
        dSpinB_min_dist_samples->setValue(0);

        horizontalLayout_21->addWidget(dSpinB_min_dist_samples);


        verticalLayout_29->addWidget(widget_14);

        widget_25 = new QWidget(groupBx_auto_sampling);
        widget_25->setObjectName(QStringLiteral("widget_25"));
        horizontalLayout_31 = new QHBoxLayout(widget_25);
        horizontalLayout_31->setSpacing(6);
        horizontalLayout_31->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_31->setObjectName(QStringLiteral("horizontalLayout_31"));
        lbl_nb_samples = new QLabel(widget_25);
        lbl_nb_samples->setObjectName(QStringLiteral("lbl_nb_samples"));

        horizontalLayout_31->addWidget(lbl_nb_samples);

        spinB_nb_samples_psd = new QSpinBox(widget_25);
        spinB_nb_samples_psd->setObjectName(QStringLiteral("spinB_nb_samples_psd"));
        spinB_nb_samples_psd->setMaximum(1000);
        spinB_nb_samples_psd->setSingleStep(20);
        spinB_nb_samples_psd->setValue(50);

        horizontalLayout_31->addWidget(spinB_nb_samples_psd);


        verticalLayout_29->addWidget(widget_25);

        checkB_auto_sample = new QCheckBox(groupBx_auto_sampling);
        checkB_auto_sample->setObjectName(QStringLiteral("checkB_auto_sample"));

        verticalLayout_29->addWidget(checkB_auto_sample);

        cBox_sampling_type = new QComboBox(groupBx_auto_sampling);
        cBox_sampling_type->setObjectName(QStringLiteral("cBox_sampling_type"));

        verticalLayout_29->addWidget(cBox_sampling_type);

        choose_hrbf_samples = new QPushButton(groupBx_auto_sampling);
        choose_hrbf_samples->setObjectName(QStringLiteral("choose_hrbf_samples"));

        verticalLayout_29->addWidget(choose_hrbf_samples);


        verticalLayout_15->addWidget(groupBx_auto_sampling);


        vert_layout_bone_editor->addWidget(box_edit_RBF);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        vert_layout_bone_editor->addItem(verticalSpacer_3);


        horizontalLayout_4->addLayout(vert_layout_bone_editor);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_3);

        toolBoxMenu->addItem(bone_editor, QStringLiteral("Bone edition"));
        debug_tools = new QWidget();
        debug_tools->setObjectName(QStringLiteral("debug_tools"));
        debug_tools->setGeometry(QRect(0, 0, 248, 844));
        horizontalLayout_9 = new QHBoxLayout(debug_tools);
        horizontalLayout_9->setSpacing(0);
        horizontalLayout_9->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_9->setObjectName(QStringLiteral("horizontalLayout_9"));
        horizontalLayout_9->setContentsMargins(0, -1, 0, -1);
        vert_layout_graph_edit_2 = new QVBoxLayout();
        vert_layout_graph_edit_2->setSpacing(6);
        vert_layout_graph_edit_2->setObjectName(QStringLiteral("vert_layout_graph_edit_2"));
        vert_layout_graph_edit_2->setSizeConstraint(QLayout::SetMinAndMaxSize);
        vert_layout_graph_edit_2->setContentsMargins(9, 9, 9, 9);
        lbl_fitting_steps = new QLabel(debug_tools);
        lbl_fitting_steps->setObjectName(QStringLiteral("lbl_fitting_steps"));

        vert_layout_graph_edit_2->addWidget(lbl_fitting_steps);

        widget_2 = new QWidget(debug_tools);
        widget_2->setObjectName(QStringLiteral("widget_2"));
        horizontalLayout_10 = new QHBoxLayout(widget_2);
        horizontalLayout_10->setSpacing(0);
        horizontalLayout_10->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_10->setObjectName(QStringLiteral("horizontalLayout_10"));
        horizontalLayout_10->setContentsMargins(0, -1, 0, -1);
        spinBox_nb_step_fitting = new QSpinBox(widget_2);
        spinBox_nb_step_fitting->setObjectName(QStringLiteral("spinBox_nb_step_fitting"));
        spinBox_nb_step_fitting->setMaximum(1000);
        spinBox_nb_step_fitting->setSingleStep(1);
        spinBox_nb_step_fitting->setValue(250);

        horizontalLayout_10->addWidget(spinBox_nb_step_fitting);

        enable_partial_fit = new QCheckBox(widget_2);
        enable_partial_fit->setObjectName(QStringLiteral("enable_partial_fit"));
        enable_partial_fit->setEnabled(false);

        horizontalLayout_10->addWidget(enable_partial_fit);


        vert_layout_graph_edit_2->addWidget(widget_2);

        slider_nb_step_fit = new QSlider(debug_tools);
        slider_nb_step_fit->setObjectName(QStringLiteral("slider_nb_step_fit"));
        slider_nb_step_fit->setMaximum(250);
        slider_nb_step_fit->setValue(250);
        slider_nb_step_fit->setOrientation(Qt::Horizontal);
        slider_nb_step_fit->setTickPosition(QSlider::TicksAbove);

        vert_layout_graph_edit_2->addWidget(slider_nb_step_fit);

        line_3 = new QFrame(debug_tools);
        line_3->setObjectName(QStringLiteral("line_3"));
        line_3->setFrameShape(QFrame::HLine);
        line_3->setFrameShadow(QFrame::Sunken);

        vert_layout_graph_edit_2->addWidget(line_3);

        debug_show_gradient = new QCheckBox(debug_tools);
        debug_show_gradient->setObjectName(QStringLiteral("debug_show_gradient"));

        vert_layout_graph_edit_2->addWidget(debug_show_gradient);

        debug_show_normal = new QCheckBox(debug_tools);
        debug_show_normal->setObjectName(QStringLiteral("debug_show_normal"));

        vert_layout_graph_edit_2->addWidget(debug_show_normal);

        line_2 = new QFrame(debug_tools);
        line_2->setObjectName(QStringLiteral("line_2"));
        line_2->setFrameShape(QFrame::HLine);
        line_2->setFrameShadow(QFrame::Sunken);

        vert_layout_graph_edit_2->addWidget(line_2);

        lbl_collisions_threshold = new QLabel(debug_tools);
        lbl_collisions_threshold->setObjectName(QStringLiteral("lbl_collisions_threshold"));

        vert_layout_graph_edit_2->addWidget(lbl_collisions_threshold);

        doubleSpinBox = new QDoubleSpinBox(debug_tools);
        doubleSpinBox->setObjectName(QStringLiteral("doubleSpinBox"));
        doubleSpinBox->setMinimum(-1);
        doubleSpinBox->setMaximum(1.1);
        doubleSpinBox->setSingleStep(0.05);
        doubleSpinBox->setValue(0.9);

        vert_layout_graph_edit_2->addWidget(doubleSpinBox);

        line_4 = new QFrame(debug_tools);
        line_4->setObjectName(QStringLiteral("line_4"));
        line_4->setFrameShape(QFrame::HLine);
        line_4->setFrameShadow(QFrame::Sunken);

        vert_layout_graph_edit_2->addWidget(line_4);

        lbl_step_length = new QLabel(debug_tools);
        lbl_step_length->setObjectName(QStringLiteral("lbl_step_length"));

        vert_layout_graph_edit_2->addWidget(lbl_step_length);

        spinB_step_length = new QDoubleSpinBox(debug_tools);
        spinB_step_length->setObjectName(QStringLiteral("spinB_step_length"));
        spinB_step_length->setDecimals(3);
        spinB_step_length->setMinimum(0);
        spinB_step_length->setSingleStep(0.01);
        spinB_step_length->setValue(0.05);

        vert_layout_graph_edit_2->addWidget(spinB_step_length);

        checkB_enable_raphson = new QCheckBox(debug_tools);
        checkB_enable_raphson->setObjectName(QStringLiteral("checkB_enable_raphson"));

        vert_layout_graph_edit_2->addWidget(checkB_enable_raphson);

        line_8 = new QFrame(debug_tools);
        line_8->setObjectName(QStringLiteral("line_8"));
        line_8->setFrameShape(QFrame::HLine);
        line_8->setFrameShadow(QFrame::Sunken);

        vert_layout_graph_edit_2->addWidget(line_8);

        lbl_collision_depth = new QLabel(debug_tools);
        lbl_collision_depth->setObjectName(QStringLiteral("lbl_collision_depth"));

        vert_layout_graph_edit_2->addWidget(lbl_collision_depth);

        dSpinB_collision_depth = new QDoubleSpinBox(debug_tools);
        dSpinB_collision_depth->setObjectName(QStringLiteral("dSpinB_collision_depth"));
        dSpinB_collision_depth->setSingleStep(0.01);

        vert_layout_graph_edit_2->addWidget(dSpinB_collision_depth);

        box_potential_pit = new QCheckBox(debug_tools);
        box_potential_pit->setObjectName(QStringLiteral("box_potential_pit"));
        box_potential_pit->setChecked(true);

        vert_layout_graph_edit_2->addWidget(box_potential_pit);

        checkBox_collsion_on = new QCheckBox(debug_tools);
        checkBox_collsion_on->setObjectName(QStringLiteral("checkBox_collsion_on"));
        checkBox_collsion_on->setChecked(true);

        vert_layout_graph_edit_2->addWidget(checkBox_collsion_on);

        checkBox_update_base_potential = new QCheckBox(debug_tools);
        checkBox_update_base_potential->setObjectName(QStringLiteral("checkBox_update_base_potential"));
        checkBox_update_base_potential->setChecked(true);

        vert_layout_graph_edit_2->addWidget(checkBox_update_base_potential);

        checkB_filter_relax = new QCheckBox(debug_tools);
        checkB_filter_relax->setObjectName(QStringLiteral("checkB_filter_relax"));
        checkB_filter_relax->setChecked(false);

        vert_layout_graph_edit_2->addWidget(checkB_filter_relax);

        line_9 = new QFrame(debug_tools);
        line_9->setObjectName(QStringLiteral("line_9"));
        line_9->setFrameShape(QFrame::HLine);
        line_9->setFrameShadow(QFrame::Sunken);

        vert_layout_graph_edit_2->addWidget(line_9);

        widget_17 = new QWidget(debug_tools);
        widget_17->setObjectName(QStringLiteral("widget_17"));
        horizontalLayout_22 = new QHBoxLayout(widget_17);
        horizontalLayout_22->setSpacing(6);
        horizontalLayout_22->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_22->setObjectName(QStringLiteral("horizontalLayout_22"));
        horizontalLayout_22->setContentsMargins(0, -1, 0, -1);
        pButton_invert_propagation = new QPushButton(widget_17);
        pButton_invert_propagation->setObjectName(QStringLiteral("pButton_invert_propagation"));

        horizontalLayout_22->addWidget(pButton_invert_propagation);

        pButton_rst_invert_propagation = new QPushButton(widget_17);
        pButton_rst_invert_propagation->setObjectName(QStringLiteral("pButton_rst_invert_propagation"));

        horizontalLayout_22->addWidget(pButton_rst_invert_propagation);


        vert_layout_graph_edit_2->addWidget(widget_17);

        line_13 = new QFrame(debug_tools);
        line_13->setObjectName(QStringLiteral("line_13"));
        line_13->setFrameShape(QFrame::HLine);
        line_13->setFrameShadow(QFrame::Sunken);

        vert_layout_graph_edit_2->addWidget(line_13);

        checkB_enable_smoothing = new QCheckBox(debug_tools);
        checkB_enable_smoothing->setObjectName(QStringLiteral("checkB_enable_smoothing"));
        checkB_enable_smoothing->setChecked(true);

        vert_layout_graph_edit_2->addWidget(checkB_enable_smoothing);

        lbl_smoothing_first_step = new QLabel(debug_tools);
        lbl_smoothing_first_step->setObjectName(QStringLiteral("lbl_smoothing_first_step"));

        vert_layout_graph_edit_2->addWidget(lbl_smoothing_first_step);

        widget_21 = new QWidget(debug_tools);
        widget_21->setObjectName(QStringLiteral("widget_21"));
        horizontalLayout_27 = new QHBoxLayout(widget_21);
        horizontalLayout_27->setSpacing(6);
        horizontalLayout_27->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_27->setObjectName(QStringLiteral("horizontalLayout_27"));
        horizontalLayout_27->setContentsMargins(-1, -1, -1, 11);
        spinB_nb_iter_smooth1 = new QSpinBox(widget_21);
        spinB_nb_iter_smooth1->setObjectName(QStringLiteral("spinB_nb_iter_smooth1"));
        spinB_nb_iter_smooth1->setMaximum(30);
        spinB_nb_iter_smooth1->setSingleStep(1);
        spinB_nb_iter_smooth1->setValue(7);

        horizontalLayout_27->addWidget(spinB_nb_iter_smooth1);

        dSpinB_lambda_smooth1 = new QDoubleSpinBox(widget_21);
        dSpinB_lambda_smooth1->setObjectName(QStringLiteral("dSpinB_lambda_smooth1"));
        dSpinB_lambda_smooth1->setMaximum(1);
        dSpinB_lambda_smooth1->setSingleStep(0.1);
        dSpinB_lambda_smooth1->setValue(1);

        horizontalLayout_27->addWidget(dSpinB_lambda_smooth1);


        vert_layout_graph_edit_2->addWidget(widget_21);

        lbl_smoothing_second_step = new QLabel(debug_tools);
        lbl_smoothing_second_step->setObjectName(QStringLiteral("lbl_smoothing_second_step"));

        vert_layout_graph_edit_2->addWidget(lbl_smoothing_second_step);

        widget_22 = new QWidget(debug_tools);
        widget_22->setObjectName(QStringLiteral("widget_22"));
        horizontalLayout_28 = new QHBoxLayout(widget_22);
        horizontalLayout_28->setSpacing(6);
        horizontalLayout_28->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_28->setObjectName(QStringLiteral("horizontalLayout_28"));
        spinB_nb_iter_smooth2 = new QSpinBox(widget_22);
        spinB_nb_iter_smooth2->setObjectName(QStringLiteral("spinB_nb_iter_smooth2"));
        spinB_nb_iter_smooth2->setMaximum(30);
        spinB_nb_iter_smooth2->setValue(1);

        horizontalLayout_28->addWidget(spinB_nb_iter_smooth2);

        dSpinB_lambda_smooth2 = new QDoubleSpinBox(widget_22);
        dSpinB_lambda_smooth2->setObjectName(QStringLiteral("dSpinB_lambda_smooth2"));
        dSpinB_lambda_smooth2->setMaximum(1);
        dSpinB_lambda_smooth2->setSingleStep(0.1);
        dSpinB_lambda_smooth2->setValue(0.5);

        horizontalLayout_28->addWidget(dSpinB_lambda_smooth2);


        vert_layout_graph_edit_2->addWidget(widget_22);

        widget_20 = new QWidget(debug_tools);
        widget_20->setObjectName(QStringLiteral("widget_20"));
        horizontalLayout_26 = new QHBoxLayout(widget_20);
        horizontalLayout_26->setSpacing(6);
        horizontalLayout_26->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_26->setObjectName(QStringLiteral("horizontalLayout_26"));
        horizontalLayout_26->setContentsMargins(-1, 2, -1, 2);
        lbl_grid_res = new QLabel(widget_20);
        lbl_grid_res->setObjectName(QStringLiteral("lbl_grid_res"));

        horizontalLayout_26->addWidget(lbl_grid_res);

        spinB_grid_res = new QSpinBox(widget_20);
        spinB_grid_res->setObjectName(QStringLiteral("spinB_grid_res"));
        spinB_grid_res->setMinimum(1);
        spinB_grid_res->setMaximum(128);
        spinB_grid_res->setValue(20);

        horizontalLayout_26->addWidget(spinB_grid_res);


        vert_layout_graph_edit_2->addWidget(widget_20);

        widget_30 = new QWidget(debug_tools);
        widget_30->setObjectName(QStringLiteral("widget_30"));
        horizontalLayout_36 = new QHBoxLayout(widget_30);
        horizontalLayout_36->setSpacing(6);
        horizontalLayout_36->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_36->setObjectName(QStringLiteral("horizontalLayout_36"));
        label_3 = new QLabel(widget_30);
        label_3->setObjectName(QStringLiteral("label_3"));

        horizontalLayout_36->addWidget(label_3);

        dSpinBox_spare_box = new QDoubleSpinBox(widget_30);
        dSpinBox_spare_box->setObjectName(QStringLiteral("dSpinBox_spare_box"));
        dSpinBox_spare_box->setDecimals(12);
        dSpinBox_spare_box->setMinimum(-100);
        dSpinBox_spare_box->setMaximum(100);

        horizontalLayout_36->addWidget(dSpinBox_spare_box);


        vert_layout_graph_edit_2->addWidget(widget_30);

        widget_31 = new QWidget(debug_tools);
        widget_31->setObjectName(QStringLiteral("widget_31"));
        horizontalLayout_37 = new QHBoxLayout(widget_31);
        horizontalLayout_37->setSpacing(6);
        horizontalLayout_37->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_37->setObjectName(QStringLiteral("horizontalLayout_37"));
        label_4 = new QLabel(widget_31);
        label_4->setObjectName(QStringLiteral("label_4"));

        horizontalLayout_37->addWidget(label_4);

        doubleSpinBox_2 = new QDoubleSpinBox(widget_31);
        doubleSpinBox_2->setObjectName(QStringLiteral("doubleSpinBox_2"));
        doubleSpinBox_2->setDecimals(12);
        doubleSpinBox_2->setMinimum(-100);
        doubleSpinBox_2->setMaximum(100);

        horizontalLayout_37->addWidget(doubleSpinBox_2);


        vert_layout_graph_edit_2->addWidget(widget_31);

        widget_32 = new QWidget(debug_tools);
        widget_32->setObjectName(QStringLiteral("widget_32"));
        horizontalLayout_38 = new QHBoxLayout(widget_32);
        horizontalLayout_38->setSpacing(6);
        horizontalLayout_38->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_38->setObjectName(QStringLiteral("horizontalLayout_38"));
        label_5 = new QLabel(widget_32);
        label_5->setObjectName(QStringLiteral("label_5"));

        horizontalLayout_38->addWidget(label_5);

        spinB_tab_val_idx = new QSpinBox(widget_32);
        spinB_tab_val_idx->setObjectName(QStringLiteral("spinB_tab_val_idx"));

        horizontalLayout_38->addWidget(spinB_tab_val_idx);

        dSpin_spare_vals_array = new QDoubleSpinBox(widget_32);
        dSpin_spare_vals_array->setObjectName(QStringLiteral("dSpin_spare_vals_array"));
        dSpin_spare_vals_array->setDecimals(4);
        dSpin_spare_vals_array->setMinimum(-100);
        dSpin_spare_vals_array->setMaximum(100);

        horizontalLayout_38->addWidget(dSpin_spare_vals_array);


        vert_layout_graph_edit_2->addWidget(widget_32);

        checkBox = new QCheckBox(debug_tools);
        checkBox->setObjectName(QStringLiteral("checkBox"));

        vert_layout_graph_edit_2->addWidget(checkBox);


        horizontalLayout_9->addLayout(vert_layout_graph_edit_2);

        horizontalSpacer_5 = new QSpacerItem(45, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_9->addItem(horizontalSpacer_5);

        toolBoxMenu->addItem(debug_tools, QStringLiteral("Debug"));
        splitter->addWidget(toolBoxMenu);
        viewports_frame = new QFrame(splitter);
        viewports_frame->setObjectName(QStringLiteral("viewports_frame"));
        viewports_frame->setStyleSheet(QStringLiteral("color: rgb(157, 157, 157);"));
        verticalLayout_35 = new QVBoxLayout(viewports_frame);
        verticalLayout_35->setSpacing(6);
        verticalLayout_35->setContentsMargins(11, 11, 11, 11);
        verticalLayout_35->setObjectName(QStringLiteral("verticalLayout_35"));
        splitter->addWidget(viewports_frame);
        main_windowClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(main_windowClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1122, 23));
        menuFiles = new QMenu(menuBar);
        menuFiles->setObjectName(QStringLiteral("menuFiles"));
        menuImport = new QMenu(menuFiles);
        menuImport->setObjectName(QStringLiteral("menuImport"));
        menuExport = new QMenu(menuFiles);
        menuExport->setObjectName(QStringLiteral("menuExport"));
        menuPaint = new QMenu(menuBar);
        menuPaint->setObjectName(QStringLiteral("menuPaint"));
        menuSelect = new QMenu(menuBar);
        menuSelect->setObjectName(QStringLiteral("menuSelect"));
        menuAlgorithm = new QMenu(menuBar);
        menuAlgorithm->setObjectName(QStringLiteral("menuAlgorithm"));
        menuRenderMode = new QMenu(menuBar);
        menuRenderMode->setObjectName(QStringLiteral("menuRenderMode"));
        menuHelp = new QMenu(menuBar);
        menuHelp->setObjectName(QStringLiteral("menuHelp"));
        menuInfo = new QMenu(menuHelp);
        menuInfo->setObjectName(QStringLiteral("menuInfo"));
        menuCutstomize = new QMenu(menuHelp);
        menuCutstomize->setObjectName(QStringLiteral("menuCutstomize"));
        main_windowClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(main_windowClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        main_windowClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(main_windowClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        main_windowClass->setStatusBar(statusBar);
        dockWidget = new QDockWidget(main_windowClass);
        dockWidget->setObjectName(QStringLiteral("dockWidget"));
        dockWidget->setMinimumSize(QSize(200, 38));
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QStringLiteral("dockWidgetContents"));
        treeWidget = new QTreeWidget(dockWidgetContents);
        QTreeWidgetItem *__qtreewidgetitem = new QTreeWidgetItem();
        __qtreewidgetitem->setText(0, QStringLiteral("1"));
        treeWidget->setHeaderItem(__qtreewidgetitem);
        treeWidget->setObjectName(QStringLiteral("treeWidget"));
        treeWidget->setGeometry(QRect(0, 0, 300, 431));
        treeWidget->setMinimumSize(QSize(300, 0));
        LayerSpinBox = new QSpinBox(dockWidgetContents);
        LayerSpinBox->setObjectName(QStringLiteral("LayerSpinBox"));
        LayerSpinBox->setGeometry(QRect(0, 430, 301, 41));
        centerframe = new QSpinBox(dockWidgetContents);
        centerframe->setObjectName(QStringLiteral("centerframe"));
        centerframe->setGeometry(QRect(0, 470, 201, 22));
        button_traj_label = new QPushButton(dockWidgetContents);
        button_traj_label->setObjectName(QStringLiteral("button_traj_label"));
        button_traj_label->setGeometry(QRect(130, 500, 71, 31));
        text_trajectory_label = new QLineEdit(dockWidgetContents);
        text_trajectory_label->setObjectName(QStringLiteral("text_trajectory_label"));
        text_trajectory_label->setGeometry(QRect(0, 500, 131, 31));
        showlabel_lineEdit = new QLineEdit(dockWidgetContents);
        showlabel_lineEdit->setObjectName(QStringLiteral("showlabel_lineEdit"));
        showlabel_lineEdit->setGeometry(QRect(0, 540, 131, 31));
        show_label_Button = new QPushButton(dockWidgetContents);
        show_label_Button->setObjectName(QStringLiteral("show_label_Button"));
        show_label_Button->setGeometry(QRect(130, 540, 75, 31));
        dockWidget->setWidget(dockWidgetContents);
        main_windowClass->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget);
        toolBar = new Toolbar(main_windowClass);
        toolBar->setObjectName(QStringLiteral("toolBar"));
        main_windowClass->addToolBar(Qt::TopToolBarArea, toolBar);
        toolBar_frame = new Toolbar_frames(main_windowClass);
        toolBar_frame->setObjectName(QStringLiteral("toolBar_frame"));
        main_windowClass->addToolBar(Qt::BottomToolBarArea, toolBar_frame);
        toolBar_painting = new Toolbar_painting(main_windowClass);
        toolBar_painting->setObjectName(QStringLiteral("toolBar_painting"));
        main_windowClass->addToolBar(Qt::RightToolBarArea, toolBar_painting);

        menuBar->addAction(menuFiles->menuAction());
        menuBar->addAction(menuPaint->menuAction());
        menuBar->addAction(menuSelect->menuAction());
        menuBar->addAction(menuAlgorithm->menuAction());
        menuBar->addAction(menuRenderMode->menuAction());
        menuBar->addAction(menuHelp->menuAction());
        menuFiles->addAction(actionImportFiles);
        menuFiles->addAction(actionSaveSnapshot);
        menuFiles->addAction(actionSavePly);
        menuFiles->addAction(actionsaveLabelFile);
        menuFiles->addAction(actionGetlabel_from_file);
        menuFiles->addAction(actionImportFiles_Lazy);
        menuFiles->addAction(menuImport->menuAction());
        menuFiles->addAction(menuExport->menuAction());
        menuFiles->addAction(actionLoad_model);
        menuImport->addAction(actionLoad_ISM);
        menuImport->addAction(actionLoad_FBX);
        menuImport->addAction(actionLoad_mesh);
        menuImport->addAction(actionLoad_skeleton);
        menuImport->addAction(actionLoad_weights);
        menuImport->addAction(actionLoad_keyframes);
        menuImport->addAction(actionLoad_cluster);
        menuImport->addAction(actionLoad_pose);
        menuImport->addAction(actionLoad_camera);
        menuImport->addAction(actionLoad_exampleMesh);
        menuImport->addAction(actionLoad_depthImage);
        menuImport->addAction(actionLoad_sampleIamge);
        menuExport->addSeparator();
        menuExport->addAction(actionSave_as_ISM);
        menuExport->addAction(actionSave_as_FBX);
        menuExport->addAction(actionSave_as_mesh);
        menuExport->addAction(actionSave_as_skeleton);
        menuExport->addAction(actionSave_weights);
        menuExport->addAction(actionSave_keyframes);
        menuExport->addAction(actionSave_cluster);
        menuExport->addAction(actionSave_pose);
        menuExport->addAction(actionSave_camera);
        menuPaint->addAction(actionSet_Visible);
        menuPaint->addAction(actionSet_Invisible);
        menuPaint->addAction(actionObject_Color);
        menuPaint->addAction(actionVertex_Color);
        menuPaint->addAction(actionLabel_Color);
        menuPaint->addAction(actionOriginal_Location);
        menuPaint->addAction(actionShow_Tracjectory);
        menuPaint->addAction(actionDont_Trace);
        menuPaint->addAction(actionShow_Graph_WrapBox);
        menuPaint->addAction(actionShow_EdgeVertexs);
        menuPaint->addAction(actionBallvertex);
        menuPaint->addAction(actionShow_normal);
        menuPaint->addAction(actionShow_kdtree);
        menuPaint->addAction(actionShow_camera_viewer);
        menuSelect->addAction(actionScene_Mode);
        menuSelect->addAction(actionSelect_Mode);
        menuSelect->addAction(actionPaint_Mode);
        menuSelect->addAction(actionEbpd_hand_mode);
        menuAlgorithm->addAction(actionClustering);
        menuAlgorithm->addAction(actionRegister);
        menuAlgorithm->addAction(actionSpectral_Cluster);
        menuAlgorithm->addAction(actionGraphCut);
        menuAlgorithm->addAction(actionCalculateNorm);
        menuAlgorithm->addAction(actionClusterAll);
        menuAlgorithm->addAction(actionVisDistortion);
        menuAlgorithm->addAction(actionGCopti);
        menuAlgorithm->addAction(actionPlanFit);
        menuAlgorithm->addAction(actionPropagate);
        menuAlgorithm->addAction(actionBullet);
        menuAlgorithm->addAction(actionSSDR);
        menuAlgorithm->addAction(actionRaycast);
        menuRenderMode->addAction(actionPoint_mode);
        menuRenderMode->addAction(actionFlat_mode);
        menuRenderMode->addAction(actionWire_mode);
        menuRenderMode->addAction(actionFlatWire_mode);
        menuRenderMode->addAction(actionSmooth_mode);
        menuRenderMode->addAction(actionTexture_mode);
        menuRenderMode->addAction(actionSelect_Mode_render);
        menuRenderMode->addAction(actionAnimate_Mode);
        menuHelp->addAction(actionAbout);
        menuHelp->addAction(actionOn_Screen_Quick_Help);
        menuHelp->addAction(actionShotcut);
        menuHelp->addAction(menuInfo->menuAction());
        menuHelp->addAction(menuCutstomize->menuAction());
        menuHelp->addAction(actionSetting);
        menuInfo->addAction(actionMesh);
        menuInfo->addAction(actionSkeleton);
        menuInfo->addAction(actionResouce_usage);
        menuCutstomize->addAction(actionColor);
        menuCutstomize->addAction(actionMaterial);
        mainToolBar->addAction(actionImportFiles);
        mainToolBar->addAction(actionSet_Visible);
        mainToolBar->addAction(actionSet_Invisible);
        mainToolBar->addAction(actionShow_Tracjectory);
        mainToolBar->addAction(actionDont_Trace);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionObject_Color);
        mainToolBar->addAction(actionVertex_Color);
        mainToolBar->addAction(actionLabel_Color);
        mainToolBar->addAction(actionOriginal_Location);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionSelect_Mode);
        mainToolBar->addAction(actionScene_Mode);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionClustering);
        mainToolBar->addAction(actionPropagate);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionShow_Graph_WrapBox);
        mainToolBar->addAction(actionShow_EdgeVertexs);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionButtonback);
        mainToolBar->addAction(actionButton2stop);
        mainToolBar->addAction(actionButtonRunOrPause);
        mainToolBar->addAction(actionButtonadvance);
        mainToolBar->addAction(actionWakeWorkThread);

        retranslateUi(main_windowClass);

        toolBoxMenu->setCurrentIndex(5);
        toolBoxMenu->layout()->setSpacing(6);


        QMetaObject::connectSlotsByName(main_windowClass);
    } // setupUi

    void retranslateUi(QMainWindow *main_windowClass)
    {
        main_windowClass->setWindowTitle(QApplication::translate("main_windowClass", "main_window", 0));
        actionImportFiles->setText(QApplication::translate("main_windowClass", "ImportFiles", 0));
        actionSet_Visible->setText(QApplication::translate("main_windowClass", "Set Visible", 0));
        actionSet_Invisible->setText(QApplication::translate("main_windowClass", "Set Invisible", 0));
        actionScene_Mode->setText(QApplication::translate("main_windowClass", "Scene Mode", 0));
        actionSelect_Mode->setText(QApplication::translate("main_windowClass", "Select Mode", 0));
        actionClustering->setText(QApplication::translate("main_windowClass", "Clustering", 0));
        actionObject_Color->setText(QApplication::translate("main_windowClass", "Object Color", 0));
        actionVertex_Color->setText(QApplication::translate("main_windowClass", "Vertex Color", 0));
        actionLabel_Color->setText(QApplication::translate("main_windowClass", "Label Color", 0));
        actionOriginal_Location->setText(QApplication::translate("main_windowClass", "Original Location", 0));
        actionShow_Tracjectory->setText(QApplication::translate("main_windowClass", "Show Tracjectory", 0));
        actionDont_Trace->setText(QApplication::translate("main_windowClass", "Dont Trace", 0));
        actionRegister->setText(QApplication::translate("main_windowClass", "Register", 0));
        actionSpectral_Cluster->setText(QApplication::translate("main_windowClass", "Spectral Cluster", 0));
        actionGraphCut->setText(QApplication::translate("main_windowClass", "GraphCut", 0));
        actionCalculateNorm->setText(QApplication::translate("main_windowClass", "CalculateNorm", 0));
        actionClusterAll->setText(QApplication::translate("main_windowClass", "ClusterAll", 0));
        actionVisDistortion->setText(QApplication::translate("main_windowClass", "VisDistortion", 0));
        actionGCopti->setText(QApplication::translate("main_windowClass", "GCopti", 0));
        actionPlanFit->setText(QApplication::translate("main_windowClass", "PlanFit", 0));
        actionShow_Graph_WrapBox->setText(QApplication::translate("main_windowClass", "Show Graph WrapBox", 0));
        actionShow_EdgeVertexs->setText(QApplication::translate("main_windowClass", "Show EdgeVertexs", 0));
        actionButtonback->setText(QApplication::translate("main_windowClass", "buttonback", 0));
        actionButton2stop->setText(QApplication::translate("main_windowClass", "button2stop", 0));
        actionButtonRunOrPause->setText(QApplication::translate("main_windowClass", "buttonstopORrun", 0));
        actionButtonadvance->setText(QApplication::translate("main_windowClass", "buttonadvance", 0));
        actionPropagate->setText(QApplication::translate("main_windowClass", "propagate", 0));
        actionSaveSnapshot->setText(QApplication::translate("main_windowClass", "SaveSnapshot", 0));
        actionSavePly->setText(QApplication::translate("main_windowClass", "SavePly", 0));
        actionWakeWorkThread->setText(QApplication::translate("main_windowClass", "wakeWorkThread", 0));
        actionPoint_mode->setText(QApplication::translate("main_windowClass", "Point mode", 0));
        actionFlat_mode->setText(QApplication::translate("main_windowClass", "Flat mode", 0));
        actionWire_mode->setText(QApplication::translate("main_windowClass", "Wire mode", 0));
        actionSmooth_mode->setText(QApplication::translate("main_windowClass", "Smooth mode", 0));
        actionTexture_mode->setText(QApplication::translate("main_windowClass", "Texture mode", 0));
        actionSelect_Mode_render->setText(QApplication::translate("main_windowClass", "Select Mode", 0));
        actionFlatWire_mode->setText(QApplication::translate("main_windowClass", "FlatWire mode", 0));
        actionPaint_Mode->setText(QApplication::translate("main_windowClass", "Paint Mode", 0));
        actionsaveLabelFile->setText(QApplication::translate("main_windowClass", "saveLabelFile", 0));
        actionGetlabel_from_file->setText(QApplication::translate("main_windowClass", "getlabel from file", 0));
        actionBallvertex->setText(QApplication::translate("main_windowClass", "ballvertex", 0));
        actionShow_normal->setText(QApplication::translate("main_windowClass", "show normal", 0));
        actionImportFiles_Lazy->setText(QApplication::translate("main_windowClass", "importFiles_Lazy", 0));
        actionAbout->setText(QApplication::translate("main_windowClass", "About", 0));
        actionOn_Screen_Quick_Help->setText(QApplication::translate("main_windowClass", "On screen quick help", 0));
        actionSSDR->setText(QApplication::translate("main_windowClass", "SSDR", 0));
        actionBullet->setText(QApplication::translate("main_windowClass", "Bullet Physics", 0));
        actionAnimate_Mode->setText(QApplication::translate("main_windowClass", "Animate Mode", 0));
        actionShow_camera_viewer->setText(QApplication::translate("main_windowClass", "show camera viewer", 0));
        actionLoad_ISM->setText(QApplication::translate("main_windowClass", "load ISM", 0));
        actionLoad_FBX->setText(QApplication::translate("main_windowClass", "Load FBX", 0));
        actionLoad_mesh->setText(QApplication::translate("main_windowClass", "Load mesh", 0));
        actionLoad_skeleton->setText(QApplication::translate("main_windowClass", "Load skeleton", 0));
        actionLoad_weights->setText(QApplication::translate("main_windowClass", "Load weights", 0));
        actionLoad_keyframes->setText(QApplication::translate("main_windowClass", "Load keyframes", 0));
        actionLoad_cluster->setText(QApplication::translate("main_windowClass", "Load cluster", 0));
        actionSave_as_ISM->setText(QApplication::translate("main_windowClass", "Save as ISM", 0));
        actionSave_as_FBX->setText(QApplication::translate("main_windowClass", "Save as FBX", 0));
        actionSave_as_mesh->setText(QApplication::translate("main_windowClass", "Save as mesh", 0));
        actionSave_as_skeleton->setText(QApplication::translate("main_windowClass", "Save as skeleton", 0));
        actionSave_weights->setText(QApplication::translate("main_windowClass", "Save weights", 0));
        actionSave_cluster->setText(QApplication::translate("main_windowClass", "Save cluster", 0));
        actionLoad_model->setText(QApplication::translate("main_windowClass", "Load model", 0));
        actionShotcut->setText(QApplication::translate("main_windowClass", "Shotcuts", 0));
        actionLoad_pose->setText(QApplication::translate("main_windowClass", "Load pose", 0));
        actionSave_pose->setText(QApplication::translate("main_windowClass", "Save pose", 0));
        actionLoad_camera->setText(QApplication::translate("main_windowClass", "Load camera", 0));
        actionSave_camera->setText(QApplication::translate("main_windowClass", "Save camera", 0));
        actionMesh->setText(QApplication::translate("main_windowClass", "mesh", 0));
        actionSkeleton->setText(QApplication::translate("main_windowClass", "skeleton", 0));
        actionResouce_usage->setText(QApplication::translate("main_windowClass", "resouce usage", 0));
        actionColor->setText(QApplication::translate("main_windowClass", "color", 0));
        actionMaterial->setText(QApplication::translate("main_windowClass", "material", 0));
        actionSetting->setText(QApplication::translate("main_windowClass", "Settings", 0));
        actionSave_keyframes->setText(QApplication::translate("main_windowClass", "Save keyframes", 0));
        actionLoad_exampleMesh->setText(QApplication::translate("main_windowClass", "Load exampleMesh", 0));
        actionEbpd_hand_mode->setText(QApplication::translate("main_windowClass", "ebpd hand mode", 0));
        actionLoad_depthImage->setText(QApplication::translate("main_windowClass", "Load depthImage", 0));
        actionLoad_sampleIamge->setText(QApplication::translate("main_windowClass", "Load sampleIamge", 0));
        actionRaycast->setText(QApplication::translate("main_windowClass", "Raycast", 0));
        actionShow_kdtree->setText(QApplication::translate("main_windowClass", "show kdtree", 0));
        box_camera->setTitle(QApplication::translate("main_windowClass", "Camera", 0));
        lbl_camera_aperture->setText(QApplication::translate("main_windowClass", "fov: ", 0));
        lbl_near_plane->setText(QApplication::translate("main_windowClass", "Near: ", 0));
        lbl_far_plane->setText(QApplication::translate("main_windowClass", " Far: ", 0));
#ifndef QT_NO_TOOLTIP
        pushB_reset_camera->setToolTip(QApplication::translate("main_windowClass", "<html><head/><body><p>Set camera position to orgin and points torwards z.</p></body></html>", 0));
#endif // QT_NO_TOOLTIP
        pushB_reset_camera->setText(QApplication::translate("main_windowClass", "Reset", 0));
#ifndef QT_NO_TOOLTIP
        checkB_camera_tracking->setToolTip(QApplication::translate("main_windowClass", "<html><head/><body><p>Track the pivot point</p></body></html>", 0));
#endif // QT_NO_TOOLTIP
        checkB_camera_tracking->setText(QApplication::translate("main_windowClass", "track", 0));
        box_animesh_color->setTitle(QApplication::translate("main_windowClass", "Animated mesh colors", 0));
        gaussian_curvature->setText(QApplication::translate("main_windowClass", "Gaussian curvature", 0));
#ifndef QT_NO_TOOLTIP
        ssd_interpolation->setToolTip(QApplication::translate("main_windowClass", "When vertices are outside the support of the implicit surface we\n"
"interpolate between SSD and implicit skinning. \n"
"Interpolation weights are shown on the mesh \n"
"(Red for full SSD yellow for full implicit skinning).", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_STATUSTIP
        ssd_interpolation->setStatusTip(QString());
#endif // QT_NO_STATUSTIP
#ifndef QT_NO_WHATSTHIS
        ssd_interpolation->setWhatsThis(QString());
#endif // QT_NO_WHATSTHIS
#ifndef QT_NO_ACCESSIBILITY
        ssd_interpolation->setAccessibleDescription(QString());
#endif // QT_NO_ACCESSIBILITY
        ssd_interpolation->setText(QApplication::translate("main_windowClass", "SSD interpolation", 0));
#ifndef QT_NO_TOOLTIP
        ssd_weights->setToolTip(QApplication::translate("main_windowClass", "Show ssd weights associated to the vertices for the selected\n"
"joint. No weight is black, nan values are white, null weights are\n"
"yellow and turns red for weights equal to 1.\n"
"", 0));
#endif // QT_NO_TOOLTIP
        ssd_weights->setText(QApplication::translate("main_windowClass", "SSD weights", 0));
#ifndef QT_NO_TOOLTIP
        color_smoothing->setToolTip(QApplication::translate("main_windowClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Show areas which are candidates to be smooth in red.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">To see currently smoothed areas enable &quot;Animated smoothing&quot;</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
        color_smoothing->setText(QApplication::translate("main_windowClass", "Smoothing weights", 0));
#ifndef QT_NO_TOOLTIP
        color_smoothing_conservative->setToolTip(QApplication::translate("main_windowClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Show smoothed areas in red and the other in yellow.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">smoothing occurs only in canditates areas showed with </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">&quot;Smoothing weights&quot;. Smoothing strength depends on the</span></p>\n"
"<p st"
                        "yle=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">bone angle relative to the rest position.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
        color_smoothing_conservative->setText(QApplication::translate("main_windowClass", "Smoothing conservative", 0));
        color_smoothing_laplacian->setText(QApplication::translate("main_windowClass", "Smoothing laplace", 0));
#ifndef QT_NO_TOOLTIP
        base_potential->setToolTip(QApplication::translate("main_windowClass", "Color the mesh according to the potential associated \n"
"at each vertices. Smooth transition is done through from \n"
"red to green and then blue. Red is 0,\n"
"green 0.5 and blue 1.\n"
"Nan numbers are painted in white", 0));
#endif // QT_NO_TOOLTIP
        base_potential->setText(QApplication::translate("main_windowClass", "Base potential", 0));
#ifndef QT_NO_TOOLTIP
        vertices_state->setToolTip(QApplication::translate("main_windowClass", "Paint the vertices with different colors depending\n"
"on what condition stopped them during the fitting process.\n"
"Red : colision (gradient divergence)\n"
"Magenta : potential pit (going away from the initial iso-surface)\n"
"Blue : stopped because fitting has reached its maximal number of steps\n"
"Yellow : Vertex already located on th right iso-surface\n"
"Green : Successfully fitted on the initial iso-surface.\n"
"black: gradient norm is null we can't know were to march.", 0));
#endif // QT_NO_TOOLTIP
        vertices_state->setText(QApplication::translate("main_windowClass", "Vertices state", 0));
#ifndef QT_NO_TOOLTIP
        implicit_gradient->setToolTip(QApplication::translate("main_windowClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Show the gradient of the implicit primitives at each </span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">mesh vertex.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Nan values are white</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
        implicit_gradient->setText(QApplication::translate("main_windowClass", "Implicit gradient", 0));
#ifndef QT_NO_TOOLTIP
        cluster->setToolTip(QApplication::translate("main_windowClass", "Each vertex is colored according to the bone it belongs to.", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_WHATSTHIS
        cluster->setWhatsThis(QApplication::translate("main_windowClass", "Gives a color for each bone and their associated vertices", 0));
#endif // QT_NO_WHATSTHIS
        cluster->setText(QApplication::translate("main_windowClass", "Clusters", 0));
        color_nearest_joint->setText(QApplication::translate("main_windowClass", "Nearest joint", 0));
#ifndef QT_NO_TOOLTIP
        color_normals->setToolTip(QApplication::translate("main_windowClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Draw mesh normals (white are nan values)</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
        color_normals->setText(QApplication::translate("main_windowClass", "Normals", 0));
        color_grey->setText(QApplication::translate("main_windowClass", "Uniform grey", 0));
#ifndef QT_NO_TOOLTIP
        color_free_vertices->setToolTip(QApplication::translate("main_windowClass", "<html><head/><body><p>Color vertex to be deformed with incr algorithm.</p></body></html>", 0));
#endif // QT_NO_TOOLTIP
        color_free_vertices->setText(QApplication::translate("main_windowClass", "Free vertices", 0));
        color_edge_stress->setText(QApplication::translate("main_windowClass", "Edge stress", 0));
        color_area_stress->setText(QApplication::translate("main_windowClass", "Area stress", 0));
        show_rbf_samples->setText(QApplication::translate("main_windowClass", "Show RBF samples", 0));
        lbl_select_vert->setText(QApplication::translate("main_windowClass", "Paint vertex id: ", 0));
        pButton_do_select_vert->setText(QApplication::translate("main_windowClass", "do", 0));
        box_mesh_color->setTitle(QApplication::translate("main_windowClass", "Mesh color", 0));
        lbl_points_color->setText(QApplication::translate("main_windowClass", "Points color", 0));
#ifndef QT_NO_TOOLTIP
        buton_uniform_point_cl->setToolTip(QApplication::translate("main_windowClass", "Every point has the same color specified in :\n"
"Customize->Colors->point color", 0));
#endif // QT_NO_TOOLTIP
        buton_uniform_point_cl->setText(QApplication::translate("main_windowClass", "Uniform", 0));
#ifndef QT_NO_TOOLTIP
        button_defects_point_cl->setToolTip(QApplication::translate("main_windowClass", "Mesh defects are colored such as side and non-manifold vertices", 0));
#endif // QT_NO_TOOLTIP
        button_defects_point_cl->setText(QApplication::translate("main_windowClass", "Show defects", 0));
        label->setText(QApplication::translate("main_windowClass", "Transparency  ", 0));
        wireframe->setText(QApplication::translate("main_windowClass", "Wireframe", 0));
        box_skeleton->setTitle(QApplication::translate("main_windowClass", "Skeleton", 0));
        display_skeleton->setText(QApplication::translate("main_windowClass", "Display skeleton", 0));
        display_oriented_bbox->setText(QApplication::translate("main_windowClass", "Display oriented bbox", 0));
        checkB_aa_bbox->setText(QApplication::translate("main_windowClass", "Display aa bbox", 0));
#ifndef QT_NO_TOOLTIP
        checkB_draw_grid->setToolTip(QApplication::translate("main_windowClass", "The skeleton evaluation is accelerated with a grid structure.\n"
"Each cell contains a sub tree ready to be evaluated.\n"
"", 0));
#endif // QT_NO_TOOLTIP
        checkB_draw_grid->setText(QApplication::translate("main_windowClass", "grid", 0));
        grpBox_operators->setTitle(QApplication::translate("main_windowClass", "Operators", 0));
        display_operator->setText(QApplication::translate("main_windowClass", "show operator", 0));
        lbl_size_operator->setText(QApplication::translate("main_windowClass", "Size tex", 0));
        lbl_opening_angle->setText(QApplication::translate("main_windowClass", "Opening", 0));
        display_controller->setText(QApplication::translate("main_windowClass", "show controller", 0));
        toolBoxMenu->setItemText(toolBoxMenu->indexOf(Display_settings), QApplication::translate("main_windowClass", "Display", 0));
        groupBox_2->setTitle(QApplication::translate("main_windowClass", "Tools", 0));
#ifndef QT_NO_TOOLTIP
        pushB_attached_skeleton->setToolTip(QApplication::translate("main_windowClass", "<html><head/><body><p>Bind the skeleton to the mesh.</p><p>The skeleton edition mode is left</p><p>to go to animation mode.</p><p>You can edit skinning weights (diffuse them/paint)</p><p>or edit the skeleton by pressing j to move a joint.</p><p><br/></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
        pushB_attached_skeleton->setText(QApplication::translate("main_windowClass", "Bind skeleton", 0));
#ifndef QT_NO_TOOLTIP
        center_graph_node->setToolTip(QApplication::translate("main_windowClass", "Try to center the selected node inside the mesh", 0));
#endif // QT_NO_TOOLTIP
        center_graph_node->setText(QApplication::translate("main_windowClass", "center", 0));
        toolBoxMenu->setItemText(toolBoxMenu->indexOf(graph_edition), QApplication::translate("main_windowClass", "Graph edition", 0));
        box_anim_type->setTitle(QApplication::translate("main_windowClass", "Type", 0));
        ssd_raio->setText(QApplication::translate("main_windowClass", "SSD", 0));
        dual_quaternion_radio->setText(QApplication::translate("main_windowClass", "Dual quaternions", 0));
        implicit_skinning_checkBox->setText(QApplication::translate("main_windowClass", "Implicit skinning", 0));
        checkBox_incremental->setText(QApplication::translate("main_windowClass", "Incremental", 0));
#ifndef QT_NO_TOOLTIP
        reset_anim->setToolTip(QApplication::translate("main_windowClass", "Set skeleton to the dress pose", 0));
#endif // QT_NO_TOOLTIP
        reset_anim->setText(QApplication::translate("main_windowClass", "reset", 0));
        grpBox_smoothing_weights->setTitle(QApplication::translate("main_windowClass", "Smoothing contact area", 0));
        lbl_nb_iter_smoothing_weight_diffusion->setText(QApplication::translate("main_windowClass", "diffusion iters", 0));
        grpBox_weights->setTitle(QApplication::translate("main_windowClass", "SSD weights", 0));
#ifndef QT_NO_TOOLTIP
        pushB_set_rigid_weights->setToolTip(QApplication::translate("main_windowClass", "Associates rigid weights to every bones.", 0));
#endif // QT_NO_TOOLTIP
        pushB_set_rigid_weights->setText(QApplication::translate("main_windowClass", "Set rigid weights", 0));
        lbl_auto_weights_experimental->setText(QApplication::translate("main_windowClass", "Diffuse weight (Experimental)", 0));
#ifndef QT_NO_TOOLTIP
        pushB_diff_w_exp->setToolTip(QApplication::translate("main_windowClass", "Compute ssd weights", 0));
#endif // QT_NO_TOOLTIP
        pushB_diff_w_exp->setText(QApplication::translate("main_windowClass", "diffuse", 0));
        lbl_diffuse_weights->setText(QApplication::translate("main_windowClass", "Diffuse current weights", 0));
        pushB_diffuse_curr_weights->setText(QApplication::translate("main_windowClass", "diffuse", 0));
        lbl_heat_diffusion_weights->setText(QApplication::translate("main_windowClass", "heat diffusion weights", 0));
        pButton_compute_heat_difusion->setText(QApplication::translate("main_windowClass", "compute", 0));
        toolBoxMenu->setItemText(toolBoxMenu->indexOf(Animation_settings), QApplication::translate("main_windowClass", "Animation", 0));
        grpBox_bulge_in_contact->setTitle(QApplication::translate("main_windowClass", "Bulge in contact", 0));
        lbl_force->setText(QApplication::translate("main_windowClass", "Force", 0));
        update_bulge_in_contact->setText(QApplication::translate("main_windowClass", "Update", 0));
        grpBox_controller->setTitle(QApplication::translate("main_windowClass", "Controller preset", 0));
        gBox_controller_values->setTitle(QApplication::translate("main_windowClass", "Controller values", 0));
        lbl_p0->setText(QApplication::translate("main_windowClass", "P0", 0));
        lbl_p1->setText(QApplication::translate("main_windowClass", "P1", 0));
        lbl_p2->setText(QApplication::translate("main_windowClass", "P2", 0));
        lbl_slopes->setText(QApplication::translate("main_windowClass", "Slopes", 0));
        pushB_edit_spline->setText(QApplication::translate("main_windowClass", "Edit/Add operators", 0));
        toolBoxMenu->setItemText(toolBoxMenu->indexOf(blending_settings), QApplication::translate("main_windowClass", "Blending", 0));
        box_edit_RBF->setTitle(QApplication::translate("main_windowClass", "RBFs", 0));
#ifndef QT_NO_TOOLTIP
        rbf_edition->setToolTip(QApplication::translate("main_windowClass", "Enable hrbf point selection you can delete or move a point", 0));
#endif // QT_NO_TOOLTIP
        rbf_edition->setText(QApplication::translate("main_windowClass", "activate edition", 0));
#ifndef QT_NO_TOOLTIP
        cBox_always_precompute->setToolTip(QApplication::translate("main_windowClass", "Keeps the bones to a precomputed type.\n"
"When editing bones the resulting HRBF will \n"
"be automatically converted to a precomputed primitive.", 0));
#endif // QT_NO_TOOLTIP
        cBox_always_precompute->setText(QApplication::translate("main_windowClass", "always precompute", 0));
#ifndef QT_NO_TOOLTIP
        checkB_factor_siblings->setToolTip(QApplication::translate("main_windowClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">When sampling a bone which has several siblings (same bone parent)</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">samples for every children will be added to the first child.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">The other children will be ignored.</span></p>\n"
"<p style=\"-q"
                        "t-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8pt;\"></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
        checkB_factor_siblings->setText(QApplication::translate("main_windowClass", "always factor bones", 0));
#ifndef QT_NO_TOOLTIP
        local_frame->setToolTip(QApplication::translate("main_windowClass", "Set the frame to the local frame of the curently\n"
"selected bone", 0));
#endif // QT_NO_TOOLTIP
        local_frame->setText(QApplication::translate("main_windowClass", "local frame", 0));
        checkB_align_with_normal->setText(QApplication::translate("main_windowClass", "Align with normal", 0));
        move_joints->setText(QApplication::translate("main_windowClass", "Move joints", 0));
        checkB_show_junction->setText(QApplication::translate("main_windowClass", "Show junctions", 0));
#ifndef QT_NO_TOOLTIP
        pushB_empty_bone->setToolTip(QApplication::translate("main_windowClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Erase hrbf samples from the selected bones.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
        pushB_empty_bone->setText(QApplication::translate("main_windowClass", "Empty bone", 0));
#ifndef QT_NO_TOOLTIP
        pButton_add_caps->setToolTip(QApplication::translate("main_windowClass", "Close HRBF with caps at both ends of the selected joints", 0));
#endif // QT_NO_TOOLTIP
        pButton_add_caps->setText(QApplication::translate("main_windowClass", "add caps", 0));
        pButton_supr_caps->setText(QApplication::translate("main_windowClass", "supr caps", 0));
        lbl_hrbf_radius->setText(QApplication::translate("main_windowClass", "HRBF radius", 0));
        groupBx_auto_sampling->setTitle(QApplication::translate("main_windowClass", "Sampling", 0));
#ifndef QT_NO_TOOLTIP
        lbl_per_max_dist_from_joints->setToolTip(QApplication::translate("main_windowClass", "Set a distance threshold from sample to the joints\n"
"to choose them.", 0));
#endif // QT_NO_TOOLTIP
        lbl_per_max_dist_from_joints->setText(QApplication::translate("main_windowClass", "% Max dist from joints", 0));
        lbl_max_dist_joint->setText(QApplication::translate("main_windowClass", "joint", 0));
#ifndef QT_NO_TOOLTIP
        checkB_cap_joint->setToolTip(QApplication::translate("main_windowClass", "Add hrbf sample to the tip of the selected joint", 0));
#endif // QT_NO_TOOLTIP
        checkB_cap_joint->setText(QApplication::translate("main_windowClass", "cap", 0));
        lbl_max_dist_parent->setText(QApplication::translate("main_windowClass", "parent", 0));
#ifndef QT_NO_TOOLTIP
        checkB_capparent->setToolTip(QApplication::translate("main_windowClass", "Add hrbf sample to the tip of the parent of the selected joint.", 0));
#endif // QT_NO_TOOLTIP
        checkB_capparent->setText(QApplication::translate("main_windowClass", "cap", 0));
#ifndef QT_NO_TOOLTIP
        lbl_max_fold->setToolTip(QApplication::translate("main_windowClass", "We choose a sample if:\n"
"max fold > (vertex orthogonal dir to the bone) dot (vertex normal)", 0));
#endif // QT_NO_TOOLTIP
        lbl_max_fold->setText(QApplication::translate("main_windowClass", "max fold", 0));
#ifndef QT_NO_TOOLTIP
        label_2->setToolTip(QApplication::translate("main_windowClass", "Minimal distance between two HRBF sample", 0));
#endif // QT_NO_TOOLTIP
        label_2->setText(QApplication::translate("main_windowClass", "min dist", 0));
#ifndef QT_NO_TOOLTIP
        lbl_nb_samples->setToolTip(QApplication::translate("main_windowClass", "Minimal number of samples.\n"
"(this value is used only whe the value min dist is zero)", 0));
#endif // QT_NO_TOOLTIP
        lbl_nb_samples->setText(QApplication::translate("main_windowClass", "nb samples", 0));
#ifndef QT_NO_TOOLTIP
        checkB_auto_sample->setToolTip(QApplication::translate("main_windowClass", "When activated samples are choosen automatically \n"
"as the parameters changes.", 0));
#endif // QT_NO_TOOLTIP
        checkB_auto_sample->setText(QApplication::translate("main_windowClass", "auto sampling", 0));
        cBox_sampling_type->clear();
        cBox_sampling_type->insertItems(0, QStringList()
         << QApplication::translate("main_windowClass", "poisson disk", 0)
         << QApplication::translate("main_windowClass", "adhoc sampling", 0)
         << QApplication::translate("main_windowClass", "gael optimal", 0)
        );
#ifndef QT_NO_TOOLTIP
        choose_hrbf_samples->setToolTip(QApplication::translate("main_windowClass", "Choose the samples and reconstruct implicit surfaces \n"
"bounds to each bone (solve the linear system).", 0));
#endif // QT_NO_TOOLTIP
        choose_hrbf_samples->setText(QApplication::translate("main_windowClass", "sample", 0));
        toolBoxMenu->setItemText(toolBoxMenu->indexOf(bone_editor), QApplication::translate("main_windowClass", "Bone edition", 0));
        lbl_fitting_steps->setText(QApplication::translate("main_windowClass", "Fitting steps", 0));
#ifndef QT_NO_TOOLTIP
        spinBox_nb_step_fitting->setToolTip(QApplication::translate("main_windowClass", "Maximum number of steps to fit vertices to the implicit surface.\n"
"Max range 0-1000.", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        enable_partial_fit->setToolTip(QApplication::translate("main_windowClass", "[Deprecated]\n"
"Select the vertices you want to move to match their potentials.\n"
"Blue : vertex has reach the maximum step.\n"
"Green : vertex has been stopped to its iso\n"
"Red : vertex has been stopped because it was marching away from its iso.", 0));
#endif // QT_NO_TOOLTIP
        enable_partial_fit->setText(QApplication::translate("main_windowClass", "partial fit", 0));
#ifndef QT_NO_TOOLTIP
        slider_nb_step_fit->setToolTip(QApplication::translate("main_windowClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Slide back to diminish the max number of steps to fit the vertices.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Slider range is from 0 to 250. For a higher value use the spinBox</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">above the slider.</span></p></body></html>", 0));
#endif // QT_NO_TOOLTIP
        debug_show_gradient->setText(QApplication::translate("main_windowClass", "show gradient", 0));
        debug_show_normal->setText(QApplication::translate("main_windowClass", "show normal", 0));
#ifndef QT_NO_TOOLTIP
        lbl_collisions_threshold->setToolTip(QApplication::translate("main_windowClass", "When the scalar product between the gradient of\n"
"step n and n-1 exceed this threshold we stop the \n"
"vertex.", 0));
#endif // QT_NO_TOOLTIP
        lbl_collisions_threshold->setText(QApplication::translate("main_windowClass", "Collision threshold", 0));
        lbl_step_length->setText(QApplication::translate("main_windowClass", "Step length", 0));
#ifndef QT_NO_TOOLTIP
        spinB_step_length->setToolTip(QApplication::translate("main_windowClass", "Length of the steps made for the gradient walk \n"
"(mesh fitting onto implicit primitives)", 0));
#endif // QT_NO_TOOLTIP
        checkB_enable_raphson->setText(QApplication::translate("main_windowClass", "raphson", 0));
        lbl_collision_depth->setText(QApplication::translate("main_windowClass", "Collision depth", 0));
#ifndef QT_NO_TOOLTIP
        dSpinB_collision_depth->setToolTip(QApplication::translate("main_windowClass", "When a vertex interpenetrate another primitive \n"
"how far do we put it from the collision plane.\n"
"0 place it at the collision plane (before smoothing)\n"
"higher values place it more inside the primitive (before smoothing)", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        box_potential_pit->setToolTip(QApplication::translate("main_windowClass", "When a vertices is going away from its \n"
"initial iso-value we stop it.\n"
"(shown in magenta)", 0));
#endif // QT_NO_TOOLTIP
        box_potential_pit->setText(QApplication::translate("main_windowClass", "potential pit", 0));
#ifndef QT_NO_TOOLTIP
        checkBox_collsion_on->setToolTip(QApplication::translate("main_windowClass", "When activated vertices are fit against the entire skeletion\n"
"potential field. And not just potential of the 3 nearest bones.\n"
"This is a post process step.", 0));
#endif // QT_NO_TOOLTIP
        checkBox_collsion_on->setText(QApplication::translate("main_windowClass", "collision everywhere", 0));
        checkBox_update_base_potential->setText(QApplication::translate("main_windowClass", "update base potential", 0));
        checkB_filter_relax->setText(QApplication::translate("main_windowClass", "Filter relaxation", 0));
#ifndef QT_NO_TOOLTIP
        pButton_invert_propagation->setToolTip(QApplication::translate("main_windowClass", "Invert the propagation direction for the selected vertices.", 0));
#endif // QT_NO_TOOLTIP
        pButton_invert_propagation->setText(QApplication::translate("main_windowClass", "invert", 0));
#ifndef QT_NO_TOOLTIP
        pButton_rst_invert_propagation->setToolTip(QApplication::translate("main_windowClass", "Restore the default propagation direction of the vertices.", 0));
#endif // QT_NO_TOOLTIP
        pButton_rst_invert_propagation->setText(QApplication::translate("main_windowClass", "reset invert", 0));
        checkB_enable_smoothing->setText(QApplication::translate("main_windowClass", "Enable smoothing", 0));
        lbl_smoothing_first_step->setText(QApplication::translate("main_windowClass", "Smooting first step", 0));
        lbl_smoothing_second_step->setText(QApplication::translate("main_windowClass", "Smoothing second step", 0));
        lbl_grid_res->setText(QApplication::translate("main_windowClass", "Grid res:", 0));
        label_3->setText(QApplication::translate("main_windowClass", "spare0:", 0));
        label_4->setText(QApplication::translate("main_windowClass", "spare1:", 0));
        label_5->setText(QApplication::translate("main_windowClass", "sparen:", 0));
        checkBox->setText(QApplication::translate("main_windowClass", "CheckBox", 0));
        toolBoxMenu->setItemText(toolBoxMenu->indexOf(debug_tools), QApplication::translate("main_windowClass", "Debug", 0));
        menuFiles->setTitle(QApplication::translate("main_windowClass", "Files", 0));
        menuImport->setTitle(QApplication::translate("main_windowClass", "import", 0));
        menuExport->setTitle(QApplication::translate("main_windowClass", "export", 0));
        menuPaint->setTitle(QApplication::translate("main_windowClass", "View", 0));
        menuSelect->setTitle(QApplication::translate("main_windowClass", "Select", 0));
        menuAlgorithm->setTitle(QApplication::translate("main_windowClass", "Algorithm", 0));
        menuRenderMode->setTitle(QApplication::translate("main_windowClass", "RenderMode", 0));
        menuHelp->setTitle(QApplication::translate("main_windowClass", "Help", 0));
        menuInfo->setTitle(QApplication::translate("main_windowClass", "Info", 0));
        menuCutstomize->setTitle(QApplication::translate("main_windowClass", "Cutstomize", 0));
        button_traj_label->setText(QApplication::translate("main_windowClass", "traj_label", 0));
        show_label_Button->setText(QApplication::translate("main_windowClass", "show_label", 0));
        toolBar->setWindowTitle(QApplication::translate("main_windowClass", "toolBar", 0));
        toolBar_frame->setWindowTitle(QApplication::translate("main_windowClass", "Timeline", 0));
        toolBar_painting->setWindowTitle(QApplication::translate("main_windowClass", "Paint", 0));
    } // retranslateUi

};

namespace Ui {
    class main_windowClass: public Ui_main_windowClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAIN_WINDOW_H
