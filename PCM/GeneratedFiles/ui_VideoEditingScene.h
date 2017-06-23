/********************************************************************************
** Form generated from reading UI file 'VideoEditingScene.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_VIDEOEDITINGSCENE_H
#define UI_VIDEOEDITINGSCENE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "srcwidget.h"

QT_BEGIN_NAMESPACE

class Ui_VideoEditingWindow
{
public:
    QAction *actionOpen;
    QAction *actionSave_as;
    QAction *actionClose;
    QAction *actionRead_frame;
    QAction *actionCamera;
    QAction *actionMatting;
    QAction *actionWrite_video;
    QAction *actionAlpha2trimap;
    QAction *actionSplit_Video;
    QAction *actionCompute_gradient;
    QAction *actionLoad_model;
    QAction *actionLoad_poses;
    QAction *actionSave_poses;
    QAction *actionWrite_CameraViewer_to_video;
    QAction *actionRender_CameraViewer_To_Image_array;
    QAction *actionWrite_CameraViewer_to_Image_array;
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout_5;
    QVBoxLayout *verticalLayout;
    QGroupBox *SourceVideo;
    QVBoxLayout *verticalLayout_6;
    QTabWidget *source_video_tab;
    QWidget *tab_imagemode;
    QVBoxLayout *verticalLayout_2;
    SrcWidget *videoFrame;
    QWidget *tab_manipulatemode;
    QVBoxLayout *verticalLayout_7;
    QFrame *frame_manipulate;
    QHBoxLayout *horizontalLayout_2;
    QLabel *cur_frame_idx;
    QLabel *label_5;
    QLabel *total_framenum;
    QHBoxLayout *horizontalLayout;
    QPushButton *pushButton_previousframe;
    QPushButton *pushButton_play;
    QPushButton *pushButton_pause;
    QPushButton *pushButton_nextframe;
    QLabel *label_7;
    QSpinBox *spinBox_turntoframe;
    QSpacerItem *horizontalSpacer;
    QPushButton *NextInitKey;
    QCheckBox *SetKey;
    QTabWidget *tabWidget_algorithom;
    QWidget *tab_fore_extract;
    QHBoxLayout *horizontalLayout_5;
    QGroupBox *toolsforstep1;
    QHBoxLayout *horizontalLayout_6;
    QGridLayout *gridLayout_3;
    QRadioButton *SelectArea;
    QRadioButton *BrushBack;
    QRadioButton *BrushCompute;
    QRadioButton *BrushFore;
    QRadioButton *ForeCut;
    QRadioButton *radioButton_7;
    QRadioButton *BackCut;
    QVBoxLayout *verticalLayout_8;
    QGridLayout *gridLayout;
    QLabel *label_2;
    QSlider *brushSizeSlider;
    QSlider *kernelSizeSlider;
    QLabel *label_3;
    QGridLayout *gridLayout_2;
    QPushButton *Grabcut;
    QCheckBox *ifshow_silhouette;
    QPushButton *MattingSingleFrame;
    QPushButton *ShowSilhouette;
    QPushButton *ShowTrimap;
    QPushButton *ShowAlphamap;
    QPushButton *showKeyFrameNo;
    QCheckBox *ifshowimagemask;
    QSpacerItem *horizontalSpacer_2;
    QVBoxLayout *verticalLayout_3;
    QPushButton *Init_Key_Frame_by_Diff;
    QPushButton *TrimapInterpolation;
    QPushButton *MattingVideo;
    QPushButton *ChangeBackground;
    QWidget *tab_pose_estimation;
    QHBoxLayout *horizontalLayout_7;
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout_3;
    QScrollArea *scrollArea_object_info;
    QWidget *scrollAreaWidgetContents;
    QHBoxLayout *horizontalLayout_8;
    QHBoxLayout *horizontalLayout_4;
    QGroupBox *object_pose;
    QGridLayout *gridLayout_4;
    QLabel *object_name_label;
    QLineEdit *object_name;
    QLabel *object_type_label;
    QLineEdit *object_type;
    QLabel *label;
    QDoubleSpinBox *translate_x;
    QDoubleSpinBox *translate_y;
    QDoubleSpinBox *translate_z;
    QLabel *label_8;
    QDoubleSpinBox *rotate_x;
    QDoubleSpinBox *rotate_y;
    QDoubleSpinBox *rotate_z;
    QLabel *label_11;
    QDoubleSpinBox *scale_x;
    QDoubleSpinBox *scale_y;
    QDoubleSpinBox *scale_z;
    QGroupBox *prespectuve_param;
    QGridLayout *gridLayout_5;
    QLabel *label_6;
    QLineEdit *lineEdit_viewport_width;
    QLabel *label_10;
    QLabel *label_9;
    QLabel *label_13;
    QLineEdit *lineEdit_viewport_height;
    QLabel *label_12;
    QLabel *label_14;
    QLineEdit *lineEdit_project_type;
    QDoubleSpinBox *doubleSpinBox_width_divide_height;
    QLabel *label_4;
    QDoubleSpinBox *doubleSpinBox_fov;
    QDoubleSpinBox *doubleSpinBox_nearplane;
    QDoubleSpinBox *doubleSpinBox_farplane;
    QGroupBox *cameraintrisic;
    QGridLayout *gridLayout_6;
    QLineEdit *alpha;
    QLineEdit *rotate_x_3;
    QLineEdit *gama;
    QLineEdit *v0;
    QLineEdit *scale_z_3;
    QLineEdit *beta;
    QLineEdit *scale_x_3;
    QLineEdit *u0;
    QLineEdit *scale_y_3;
    QSpacerItem *horizontalSpacer_5;
    QGroupBox *groupBox_manipulate;
    QVBoxLayout *verticalLayout_4;
    QRadioButton *radioButton_select;
    QRadioButton *radioButton_selectface;
    QRadioButton *radioButton_translate;
    QRadioButton *radioButton_rotate;
    QRadioButton *radioButton_scale;
    QRadioButton *radioButton_fouces;
    QGroupBox *groupBox_3;
    QVBoxLayout *verticalLayout_9;
    QPushButton *pushButtoncur_pose_estimation;
    QPushButton *set_curframe_as_key_frame_of_pose;
    QPushButton *pushButton_whole_pose_estimation;
    QPushButton *pushButton_correspondence;
    QPushButton *caculateCorredTexture;
    QPushButton *caculateAllCorredTexture;
    QWidget *tab_simulate;
    QHBoxLayout *horizontalLayout_9;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout_10;
    QPushButton *begin_simulate;
    QPushButton *pause_simulate;
    QPushButton *continue_simulate;
    QPushButton *restart;
    QPushButton *step_forward;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout_7;
    QLabel *label_15;
    QLabel *label_16;
    QLineEdit *lineEdit;
    QLineEdit *lineEdit_2;
    QGroupBox *constraint;
    QVBoxLayout *verticalLayout_11;
    QPushButton *setStrongFaceConstraint;
    QPushButton *setWeakFaceConstraint;
    QPushButton *unsetFaceConstraint;
    QPushButton *showConstraint;
    QPushButton *unshowConstraint;
    QSpacerItem *horizontalSpacer_3;
    QMenuBar *menubar;
    QMenu *menuFile;
    QMenu *menuTools;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *VideoEditingWindow)
    {
        if (VideoEditingWindow->objectName().isEmpty())
            VideoEditingWindow->setObjectName(QStringLiteral("VideoEditingWindow"));
        VideoEditingWindow->resize(856, 939);
        actionOpen = new QAction(VideoEditingWindow);
        actionOpen->setObjectName(QStringLiteral("actionOpen"));
        actionSave_as = new QAction(VideoEditingWindow);
        actionSave_as->setObjectName(QStringLiteral("actionSave_as"));
        actionClose = new QAction(VideoEditingWindow);
        actionClose->setObjectName(QStringLiteral("actionClose"));
        actionRead_frame = new QAction(VideoEditingWindow);
        actionRead_frame->setObjectName(QStringLiteral("actionRead_frame"));
        actionCamera = new QAction(VideoEditingWindow);
        actionCamera->setObjectName(QStringLiteral("actionCamera"));
        actionMatting = new QAction(VideoEditingWindow);
        actionMatting->setObjectName(QStringLiteral("actionMatting"));
        actionWrite_video = new QAction(VideoEditingWindow);
        actionWrite_video->setObjectName(QStringLiteral("actionWrite_video"));
        actionAlpha2trimap = new QAction(VideoEditingWindow);
        actionAlpha2trimap->setObjectName(QStringLiteral("actionAlpha2trimap"));
        actionSplit_Video = new QAction(VideoEditingWindow);
        actionSplit_Video->setObjectName(QStringLiteral("actionSplit_Video"));
        actionCompute_gradient = new QAction(VideoEditingWindow);
        actionCompute_gradient->setObjectName(QStringLiteral("actionCompute_gradient"));
        actionLoad_model = new QAction(VideoEditingWindow);
        actionLoad_model->setObjectName(QStringLiteral("actionLoad_model"));
        actionLoad_poses = new QAction(VideoEditingWindow);
        actionLoad_poses->setObjectName(QStringLiteral("actionLoad_poses"));
        actionSave_poses = new QAction(VideoEditingWindow);
        actionSave_poses->setObjectName(QStringLiteral("actionSave_poses"));
        actionWrite_CameraViewer_to_video = new QAction(VideoEditingWindow);
        actionWrite_CameraViewer_to_video->setObjectName(QStringLiteral("actionWrite_CameraViewer_to_video"));
        actionRender_CameraViewer_To_Image_array = new QAction(VideoEditingWindow);
        actionRender_CameraViewer_To_Image_array->setObjectName(QStringLiteral("actionRender_CameraViewer_To_Image_array"));
        actionWrite_CameraViewer_to_Image_array = new QAction(VideoEditingWindow);
        actionWrite_CameraViewer_to_Image_array->setObjectName(QStringLiteral("actionWrite_CameraViewer_to_Image_array"));
        centralwidget = new QWidget(VideoEditingWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        verticalLayout_5 = new QVBoxLayout(centralwidget);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        SourceVideo = new QGroupBox(centralwidget);
        SourceVideo->setObjectName(QStringLiteral("SourceVideo"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(SourceVideo->sizePolicy().hasHeightForWidth());
        SourceVideo->setSizePolicy(sizePolicy);
        SourceVideo->setMinimumSize(QSize(0, 550));
        verticalLayout_6 = new QVBoxLayout(SourceVideo);
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        source_video_tab = new QTabWidget(SourceVideo);
        source_video_tab->setObjectName(QStringLiteral("source_video_tab"));
        source_video_tab->setEnabled(true);
        source_video_tab->setMinimumSize(QSize(198, 0));
        tab_imagemode = new QWidget();
        tab_imagemode->setObjectName(QStringLiteral("tab_imagemode"));
        sizePolicy.setHeightForWidth(tab_imagemode->sizePolicy().hasHeightForWidth());
        tab_imagemode->setSizePolicy(sizePolicy);
        verticalLayout_2 = new QVBoxLayout(tab_imagemode);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        videoFrame = new SrcWidget(tab_imagemode);
        videoFrame->setObjectName(QStringLiteral("videoFrame"));
        videoFrame->setEnabled(true);
        sizePolicy.setHeightForWidth(videoFrame->sizePolicy().hasHeightForWidth());
        videoFrame->setSizePolicy(sizePolicy);
        videoFrame->setMinimumSize(QSize(321, 20));
        videoFrame->setMaximumSize(QSize(2480, 1020));
        videoFrame->setFrameShape(QFrame::Box);
        videoFrame->setScaledContents(false);

        verticalLayout_2->addWidget(videoFrame);

        source_video_tab->addTab(tab_imagemode, QString());
        tab_manipulatemode = new QWidget();
        tab_manipulatemode->setObjectName(QStringLiteral("tab_manipulatemode"));
        verticalLayout_7 = new QVBoxLayout(tab_manipulatemode);
        verticalLayout_7->setSpacing(0);
        verticalLayout_7->setObjectName(QStringLiteral("verticalLayout_7"));
        verticalLayout_7->setContentsMargins(0, 0, 0, 0);
        frame_manipulate = new QFrame(tab_manipulatemode);
        frame_manipulate->setObjectName(QStringLiteral("frame_manipulate"));
        sizePolicy.setHeightForWidth(frame_manipulate->sizePolicy().hasHeightForWidth());
        frame_manipulate->setSizePolicy(sizePolicy);
        frame_manipulate->setFrameShape(QFrame::Box);
        frame_manipulate->setFrameShadow(QFrame::Plain);
        frame_manipulate->setLineWidth(1);

        verticalLayout_7->addWidget(frame_manipulate);

        source_video_tab->addTab(tab_manipulatemode, QString());

        verticalLayout_6->addWidget(source_video_tab);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        cur_frame_idx = new QLabel(SourceVideo);
        cur_frame_idx->setObjectName(QStringLiteral("cur_frame_idx"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(cur_frame_idx->sizePolicy().hasHeightForWidth());
        cur_frame_idx->setSizePolicy(sizePolicy1);
        cur_frame_idx->setMinimumSize(QSize(60, 0));
        QFont font;
        font.setFamily(QString::fromUtf8("\351\273\221\344\275\223"));
        font.setPointSize(20);
        cur_frame_idx->setFont(font);
        cur_frame_idx->setAlignment(Qt::AlignCenter);

        horizontalLayout_2->addWidget(cur_frame_idx);

        label_5 = new QLabel(SourceVideo);
        label_5->setObjectName(QStringLiteral("label_5"));
        sizePolicy1.setHeightForWidth(label_5->sizePolicy().hasHeightForWidth());
        label_5->setSizePolicy(sizePolicy1);
        label_5->setMinimumSize(QSize(40, 0));
        label_5->setFont(font);
        label_5->setAlignment(Qt::AlignCenter);

        horizontalLayout_2->addWidget(label_5);

        total_framenum = new QLabel(SourceVideo);
        total_framenum->setObjectName(QStringLiteral("total_framenum"));
        sizePolicy1.setHeightForWidth(total_framenum->sizePolicy().hasHeightForWidth());
        total_framenum->setSizePolicy(sizePolicy1);
        total_framenum->setMinimumSize(QSize(60, 0));
        total_framenum->setFont(font);
        total_framenum->setAlignment(Qt::AlignCenter);
        total_framenum->setMargin(7);

        horizontalLayout_2->addWidget(total_framenum);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        pushButton_previousframe = new QPushButton(SourceVideo);
        pushButton_previousframe->setObjectName(QStringLiteral("pushButton_previousframe"));

        horizontalLayout->addWidget(pushButton_previousframe);

        pushButton_play = new QPushButton(SourceVideo);
        pushButton_play->setObjectName(QStringLiteral("pushButton_play"));

        horizontalLayout->addWidget(pushButton_play);

        pushButton_pause = new QPushButton(SourceVideo);
        pushButton_pause->setObjectName(QStringLiteral("pushButton_pause"));

        horizontalLayout->addWidget(pushButton_pause);

        pushButton_nextframe = new QPushButton(SourceVideo);
        pushButton_nextframe->setObjectName(QStringLiteral("pushButton_nextframe"));

        horizontalLayout->addWidget(pushButton_nextframe);

        label_7 = new QLabel(SourceVideo);
        label_7->setObjectName(QStringLiteral("label_7"));

        horizontalLayout->addWidget(label_7);

        spinBox_turntoframe = new QSpinBox(SourceVideo);
        spinBox_turntoframe->setObjectName(QStringLiteral("spinBox_turntoframe"));

        horizontalLayout->addWidget(spinBox_turntoframe);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        NextInitKey = new QPushButton(SourceVideo);
        NextInitKey->setObjectName(QStringLiteral("NextInitKey"));

        horizontalLayout->addWidget(NextInitKey);

        SetKey = new QCheckBox(SourceVideo);
        SetKey->setObjectName(QStringLiteral("SetKey"));
        QSizePolicy sizePolicy2(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(SetKey->sizePolicy().hasHeightForWidth());
        SetKey->setSizePolicy(sizePolicy2);

        horizontalLayout->addWidget(SetKey);


        horizontalLayout_2->addLayout(horizontalLayout);


        verticalLayout_6->addLayout(horizontalLayout_2);


        verticalLayout->addWidget(SourceVideo);

        tabWidget_algorithom = new QTabWidget(centralwidget);
        tabWidget_algorithom->setObjectName(QStringLiteral("tabWidget_algorithom"));
        QSizePolicy sizePolicy3(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(tabWidget_algorithom->sizePolicy().hasHeightForWidth());
        tabWidget_algorithom->setSizePolicy(sizePolicy3);
        tabWidget_algorithom->setMinimumSize(QSize(0, 230));
        tab_fore_extract = new QWidget();
        tab_fore_extract->setObjectName(QStringLiteral("tab_fore_extract"));
        horizontalLayout_5 = new QHBoxLayout(tab_fore_extract);
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        toolsforstep1 = new QGroupBox(tab_fore_extract);
        toolsforstep1->setObjectName(QStringLiteral("toolsforstep1"));
        horizontalLayout_6 = new QHBoxLayout(toolsforstep1);
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setSpacing(0);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        SelectArea = new QRadioButton(toolsforstep1);
        SelectArea->setObjectName(QStringLiteral("SelectArea"));
        SelectArea->setChecked(true);

        gridLayout_3->addWidget(SelectArea, 0, 0, 1, 1);

        BrushBack = new QRadioButton(toolsforstep1);
        BrushBack->setObjectName(QStringLiteral("BrushBack"));

        gridLayout_3->addWidget(BrushBack, 2, 0, 1, 1);

        BrushCompute = new QRadioButton(toolsforstep1);
        BrushCompute->setObjectName(QStringLiteral("BrushCompute"));

        gridLayout_3->addWidget(BrushCompute, 3, 0, 1, 1);

        BrushFore = new QRadioButton(toolsforstep1);
        BrushFore->setObjectName(QStringLiteral("BrushFore"));

        gridLayout_3->addWidget(BrushFore, 4, 0, 1, 1);

        ForeCut = new QRadioButton(toolsforstep1);
        ForeCut->setObjectName(QStringLiteral("ForeCut"));

        gridLayout_3->addWidget(ForeCut, 0, 1, 1, 1);

        radioButton_7 = new QRadioButton(toolsforstep1);
        radioButton_7->setObjectName(QStringLiteral("radioButton_7"));

        gridLayout_3->addWidget(radioButton_7, 3, 1, 1, 1);

        BackCut = new QRadioButton(toolsforstep1);
        BackCut->setObjectName(QStringLiteral("BackCut"));

        gridLayout_3->addWidget(BackCut, 2, 1, 1, 1);


        horizontalLayout_6->addLayout(gridLayout_3);

        verticalLayout_8 = new QVBoxLayout();
        verticalLayout_8->setSpacing(0);
        verticalLayout_8->setObjectName(QStringLiteral("verticalLayout_8"));
        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        label_2 = new QLabel(toolsforstep1);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        brushSizeSlider = new QSlider(toolsforstep1);
        brushSizeSlider->setObjectName(QStringLiteral("brushSizeSlider"));
        sizePolicy3.setHeightForWidth(brushSizeSlider->sizePolicy().hasHeightForWidth());
        brushSizeSlider->setSizePolicy(sizePolicy3);
        brushSizeSlider->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(brushSizeSlider, 1, 1, 1, 1);

        kernelSizeSlider = new QSlider(toolsforstep1);
        kernelSizeSlider->setObjectName(QStringLiteral("kernelSizeSlider"));
        QSizePolicy sizePolicy4(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(kernelSizeSlider->sizePolicy().hasHeightForWidth());
        kernelSizeSlider->setSizePolicy(sizePolicy4);
        kernelSizeSlider->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(kernelSizeSlider, 2, 1, 1, 1);

        label_3 = new QLabel(toolsforstep1);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);


        verticalLayout_8->addLayout(gridLayout);

        gridLayout_2 = new QGridLayout();
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        Grabcut = new QPushButton(toolsforstep1);
        Grabcut->setObjectName(QStringLiteral("Grabcut"));

        gridLayout_2->addWidget(Grabcut, 2, 0, 1, 1);

        ifshow_silhouette = new QCheckBox(toolsforstep1);
        ifshow_silhouette->setObjectName(QStringLiteral("ifshow_silhouette"));
        ifshow_silhouette->setChecked(true);

        gridLayout_2->addWidget(ifshow_silhouette, 4, 2, 1, 1);

        MattingSingleFrame = new QPushButton(toolsforstep1);
        MattingSingleFrame->setObjectName(QStringLiteral("MattingSingleFrame"));

        gridLayout_2->addWidget(MattingSingleFrame, 2, 1, 1, 1);

        ShowSilhouette = new QPushButton(toolsforstep1);
        ShowSilhouette->setObjectName(QStringLiteral("ShowSilhouette"));

        gridLayout_2->addWidget(ShowSilhouette, 3, 2, 1, 1);

        ShowTrimap = new QPushButton(toolsforstep1);
        ShowTrimap->setObjectName(QStringLiteral("ShowTrimap"));

        gridLayout_2->addWidget(ShowTrimap, 3, 0, 1, 1);

        ShowAlphamap = new QPushButton(toolsforstep1);
        ShowAlphamap->setObjectName(QStringLiteral("ShowAlphamap"));

        gridLayout_2->addWidget(ShowAlphamap, 3, 1, 1, 1);

        showKeyFrameNo = new QPushButton(toolsforstep1);
        showKeyFrameNo->setObjectName(QStringLiteral("showKeyFrameNo"));

        gridLayout_2->addWidget(showKeyFrameNo, 2, 2, 1, 1);

        ifshowimagemask = new QCheckBox(toolsforstep1);
        ifshowimagemask->setObjectName(QStringLiteral("ifshowimagemask"));

        gridLayout_2->addWidget(ifshowimagemask, 4, 0, 1, 1);


        verticalLayout_8->addLayout(gridLayout_2);

        verticalLayout_8->setStretch(0, 1);
        verticalLayout_8->setStretch(1, 1);

        horizontalLayout_6->addLayout(verticalLayout_8);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_2);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        Init_Key_Frame_by_Diff = new QPushButton(toolsforstep1);
        Init_Key_Frame_by_Diff->setObjectName(QStringLiteral("Init_Key_Frame_by_Diff"));

        verticalLayout_3->addWidget(Init_Key_Frame_by_Diff);

        TrimapInterpolation = new QPushButton(toolsforstep1);
        TrimapInterpolation->setObjectName(QStringLiteral("TrimapInterpolation"));

        verticalLayout_3->addWidget(TrimapInterpolation);

        MattingVideo = new QPushButton(toolsforstep1);
        MattingVideo->setObjectName(QStringLiteral("MattingVideo"));

        verticalLayout_3->addWidget(MattingVideo);

        ChangeBackground = new QPushButton(toolsforstep1);
        ChangeBackground->setObjectName(QStringLiteral("ChangeBackground"));

        verticalLayout_3->addWidget(ChangeBackground);


        horizontalLayout_6->addLayout(verticalLayout_3);

        horizontalLayout_6->setStretch(0, 2);
        horizontalLayout_6->setStretch(1, 3);
        horizontalLayout_6->setStretch(2, 4);
        horizontalLayout_6->setStretch(3, 1);

        horizontalLayout_5->addWidget(toolsforstep1);

        tabWidget_algorithom->addTab(tab_fore_extract, QString());
        tab_pose_estimation = new QWidget();
        tab_pose_estimation->setObjectName(QStringLiteral("tab_pose_estimation"));
        horizontalLayout_7 = new QHBoxLayout(tab_pose_estimation);
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        groupBox = new QGroupBox(tab_pose_estimation);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        horizontalLayout_3 = new QHBoxLayout(groupBox);
        horizontalLayout_3->setSpacing(1);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(1, 1, 1, 1);
        scrollArea_object_info = new QScrollArea(groupBox);
        scrollArea_object_info->setObjectName(QStringLiteral("scrollArea_object_info"));
        scrollArea_object_info->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QStringLiteral("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 1000, 161));
        scrollAreaWidgetContents->setMinimumSize(QSize(1000, 0));
        horizontalLayout_8 = new QHBoxLayout(scrollAreaWidgetContents);
        horizontalLayout_8->setSpacing(0);
        horizontalLayout_8->setObjectName(QStringLiteral("horizontalLayout_8"));
        horizontalLayout_8->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        object_pose = new QGroupBox(scrollAreaWidgetContents);
        object_pose->setObjectName(QStringLiteral("object_pose"));
        object_pose->setMinimumSize(QSize(0, 0));
        gridLayout_4 = new QGridLayout(object_pose);
        gridLayout_4->setSpacing(0);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        gridLayout_4->setContentsMargins(5, 0, 5, 0);
        object_name_label = new QLabel(object_pose);
        object_name_label->setObjectName(QStringLiteral("object_name_label"));

        gridLayout_4->addWidget(object_name_label, 0, 0, 1, 1);

        object_name = new QLineEdit(object_pose);
        object_name->setObjectName(QStringLiteral("object_name"));

        gridLayout_4->addWidget(object_name, 0, 1, 1, 3);

        object_type_label = new QLabel(object_pose);
        object_type_label->setObjectName(QStringLiteral("object_type_label"));

        gridLayout_4->addWidget(object_type_label, 1, 0, 1, 1);

        object_type = new QLineEdit(object_pose);
        object_type->setObjectName(QStringLiteral("object_type"));

        gridLayout_4->addWidget(object_type, 1, 1, 1, 3);

        label = new QLabel(object_pose);
        label->setObjectName(QStringLiteral("label"));

        gridLayout_4->addWidget(label, 2, 0, 1, 1);

        translate_x = new QDoubleSpinBox(object_pose);
        translate_x->setObjectName(QStringLiteral("translate_x"));

        gridLayout_4->addWidget(translate_x, 2, 1, 1, 1);

        translate_y = new QDoubleSpinBox(object_pose);
        translate_y->setObjectName(QStringLiteral("translate_y"));

        gridLayout_4->addWidget(translate_y, 2, 2, 1, 1);

        translate_z = new QDoubleSpinBox(object_pose);
        translate_z->setObjectName(QStringLiteral("translate_z"));

        gridLayout_4->addWidget(translate_z, 2, 3, 1, 1);

        label_8 = new QLabel(object_pose);
        label_8->setObjectName(QStringLiteral("label_8"));

        gridLayout_4->addWidget(label_8, 3, 0, 1, 1);

        rotate_x = new QDoubleSpinBox(object_pose);
        rotate_x->setObjectName(QStringLiteral("rotate_x"));

        gridLayout_4->addWidget(rotate_x, 3, 1, 1, 1);

        rotate_y = new QDoubleSpinBox(object_pose);
        rotate_y->setObjectName(QStringLiteral("rotate_y"));

        gridLayout_4->addWidget(rotate_y, 3, 2, 1, 1);

        rotate_z = new QDoubleSpinBox(object_pose);
        rotate_z->setObjectName(QStringLiteral("rotate_z"));

        gridLayout_4->addWidget(rotate_z, 3, 3, 1, 1);

        label_11 = new QLabel(object_pose);
        label_11->setObjectName(QStringLiteral("label_11"));

        gridLayout_4->addWidget(label_11, 4, 0, 1, 1);

        scale_x = new QDoubleSpinBox(object_pose);
        scale_x->setObjectName(QStringLiteral("scale_x"));

        gridLayout_4->addWidget(scale_x, 4, 1, 1, 1);

        scale_y = new QDoubleSpinBox(object_pose);
        scale_y->setObjectName(QStringLiteral("scale_y"));

        gridLayout_4->addWidget(scale_y, 4, 2, 1, 1);

        scale_z = new QDoubleSpinBox(object_pose);
        scale_z->setObjectName(QStringLiteral("scale_z"));

        gridLayout_4->addWidget(scale_z, 4, 3, 1, 1);

        label_11->raise();
        label_8->raise();
        label->raise();
        translate_x->raise();
        translate_z->raise();
        scale_x->raise();
        scale_z->raise();
        rotate_z->raise();
        translate_y->raise();
        rotate_y->raise();
        object_name_label->raise();
        scale_y->raise();
        rotate_x->raise();
        object_type_label->raise();
        object_type->raise();
        object_name->raise();

        horizontalLayout_4->addWidget(object_pose);

        prespectuve_param = new QGroupBox(scrollAreaWidgetContents);
        prespectuve_param->setObjectName(QStringLiteral("prespectuve_param"));
        gridLayout_5 = new QGridLayout(prespectuve_param);
        gridLayout_5->setSpacing(0);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        gridLayout_5->setContentsMargins(10, 5, 5, 0);
        label_6 = new QLabel(prespectuve_param);
        label_6->setObjectName(QStringLiteral("label_6"));

        gridLayout_5->addWidget(label_6, 2, 0, 1, 1);

        lineEdit_viewport_width = new QLineEdit(prespectuve_param);
        lineEdit_viewport_width->setObjectName(QStringLiteral("lineEdit_viewport_width"));

        gridLayout_5->addWidget(lineEdit_viewport_width, 2, 1, 1, 1);

        label_10 = new QLabel(prespectuve_param);
        label_10->setObjectName(QStringLiteral("label_10"));

        gridLayout_5->addWidget(label_10, 5, 0, 1, 1);

        label_9 = new QLabel(prespectuve_param);
        label_9->setObjectName(QStringLiteral("label_9"));

        gridLayout_5->addWidget(label_9, 4, 0, 1, 1);

        label_13 = new QLabel(prespectuve_param);
        label_13->setObjectName(QStringLiteral("label_13"));

        gridLayout_5->addWidget(label_13, 3, 0, 1, 1);

        lineEdit_viewport_height = new QLineEdit(prespectuve_param);
        lineEdit_viewport_height->setObjectName(QStringLiteral("lineEdit_viewport_height"));

        gridLayout_5->addWidget(lineEdit_viewport_height, 3, 1, 1, 1);

        label_12 = new QLabel(prespectuve_param);
        label_12->setObjectName(QStringLiteral("label_12"));

        gridLayout_5->addWidget(label_12, 6, 0, 1, 1);

        label_14 = new QLabel(prespectuve_param);
        label_14->setObjectName(QStringLiteral("label_14"));

        gridLayout_5->addWidget(label_14, 0, 0, 1, 1);

        lineEdit_project_type = new QLineEdit(prespectuve_param);
        lineEdit_project_type->setObjectName(QStringLiteral("lineEdit_project_type"));

        gridLayout_5->addWidget(lineEdit_project_type, 0, 1, 1, 1);

        doubleSpinBox_width_divide_height = new QDoubleSpinBox(prespectuve_param);
        doubleSpinBox_width_divide_height->setObjectName(QStringLiteral("doubleSpinBox_width_divide_height"));

        gridLayout_5->addWidget(doubleSpinBox_width_divide_height, 4, 1, 1, 1);

        label_4 = new QLabel(prespectuve_param);
        label_4->setObjectName(QStringLiteral("label_4"));

        gridLayout_5->addWidget(label_4, 1, 0, 1, 1);

        doubleSpinBox_fov = new QDoubleSpinBox(prespectuve_param);
        doubleSpinBox_fov->setObjectName(QStringLiteral("doubleSpinBox_fov"));

        gridLayout_5->addWidget(doubleSpinBox_fov, 1, 1, 1, 1);

        doubleSpinBox_nearplane = new QDoubleSpinBox(prespectuve_param);
        doubleSpinBox_nearplane->setObjectName(QStringLiteral("doubleSpinBox_nearplane"));

        gridLayout_5->addWidget(doubleSpinBox_nearplane, 5, 1, 1, 1);

        doubleSpinBox_farplane = new QDoubleSpinBox(prespectuve_param);
        doubleSpinBox_farplane->setObjectName(QStringLiteral("doubleSpinBox_farplane"));

        gridLayout_5->addWidget(doubleSpinBox_farplane, 6, 1, 1, 1);


        horizontalLayout_4->addWidget(prespectuve_param);

        cameraintrisic = new QGroupBox(scrollAreaWidgetContents);
        cameraintrisic->setObjectName(QStringLiteral("cameraintrisic"));
        gridLayout_6 = new QGridLayout(cameraintrisic);
        gridLayout_6->setSpacing(0);
        gridLayout_6->setObjectName(QStringLiteral("gridLayout_6"));
        gridLayout_6->setContentsMargins(5, 0, 5, 0);
        alpha = new QLineEdit(cameraintrisic);
        alpha->setObjectName(QStringLiteral("alpha"));

        gridLayout_6->addWidget(alpha, 0, 0, 1, 1);

        rotate_x_3 = new QLineEdit(cameraintrisic);
        rotate_x_3->setObjectName(QStringLiteral("rotate_x_3"));

        gridLayout_6->addWidget(rotate_x_3, 1, 0, 1, 1);

        gama = new QLineEdit(cameraintrisic);
        gama->setObjectName(QStringLiteral("gama"));

        gridLayout_6->addWidget(gama, 0, 1, 1, 1);

        v0 = new QLineEdit(cameraintrisic);
        v0->setObjectName(QStringLiteral("v0"));

        gridLayout_6->addWidget(v0, 1, 2, 1, 1);

        scale_z_3 = new QLineEdit(cameraintrisic);
        scale_z_3->setObjectName(QStringLiteral("scale_z_3"));

        gridLayout_6->addWidget(scale_z_3, 2, 2, 1, 1);

        beta = new QLineEdit(cameraintrisic);
        beta->setObjectName(QStringLiteral("beta"));

        gridLayout_6->addWidget(beta, 1, 1, 1, 1);

        scale_x_3 = new QLineEdit(cameraintrisic);
        scale_x_3->setObjectName(QStringLiteral("scale_x_3"));

        gridLayout_6->addWidget(scale_x_3, 2, 0, 1, 1);

        u0 = new QLineEdit(cameraintrisic);
        u0->setObjectName(QStringLiteral("u0"));

        gridLayout_6->addWidget(u0, 0, 2, 1, 1);

        scale_y_3 = new QLineEdit(cameraintrisic);
        scale_y_3->setObjectName(QStringLiteral("scale_y_3"));

        gridLayout_6->addWidget(scale_y_3, 2, 1, 1, 1);


        horizontalLayout_4->addWidget(cameraintrisic);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_5);

        horizontalLayout_4->setStretch(0, 2);
        horizontalLayout_4->setStretch(1, 2);
        horizontalLayout_4->setStretch(2, 2);
        horizontalLayout_4->setStretch(3, 2);

        horizontalLayout_8->addLayout(horizontalLayout_4);

        scrollArea_object_info->setWidget(scrollAreaWidgetContents);

        horizontalLayout_3->addWidget(scrollArea_object_info);

        groupBox_manipulate = new QGroupBox(groupBox);
        groupBox_manipulate->setObjectName(QStringLiteral("groupBox_manipulate"));
        verticalLayout_4 = new QVBoxLayout(groupBox_manipulate);
        verticalLayout_4->setSpacing(1);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(1, 1, 1, 1);
        radioButton_select = new QRadioButton(groupBox_manipulate);
        radioButton_select->setObjectName(QStringLiteral("radioButton_select"));
        radioButton_select->setChecked(true);

        verticalLayout_4->addWidget(radioButton_select);

        radioButton_selectface = new QRadioButton(groupBox_manipulate);
        radioButton_selectface->setObjectName(QStringLiteral("radioButton_selectface"));

        verticalLayout_4->addWidget(radioButton_selectface);

        radioButton_translate = new QRadioButton(groupBox_manipulate);
        radioButton_translate->setObjectName(QStringLiteral("radioButton_translate"));

        verticalLayout_4->addWidget(radioButton_translate);

        radioButton_rotate = new QRadioButton(groupBox_manipulate);
        radioButton_rotate->setObjectName(QStringLiteral("radioButton_rotate"));

        verticalLayout_4->addWidget(radioButton_rotate);

        radioButton_scale = new QRadioButton(groupBox_manipulate);
        radioButton_scale->setObjectName(QStringLiteral("radioButton_scale"));

        verticalLayout_4->addWidget(radioButton_scale);

        radioButton_fouces = new QRadioButton(groupBox_manipulate);
        radioButton_fouces->setObjectName(QStringLiteral("radioButton_fouces"));

        verticalLayout_4->addWidget(radioButton_fouces);


        horizontalLayout_3->addWidget(groupBox_manipulate);

        groupBox_3 = new QGroupBox(groupBox);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        verticalLayout_9 = new QVBoxLayout(groupBox_3);
        verticalLayout_9->setSpacing(0);
        verticalLayout_9->setObjectName(QStringLiteral("verticalLayout_9"));
        verticalLayout_9->setContentsMargins(0, 0, 0, 0);
        pushButtoncur_pose_estimation = new QPushButton(groupBox_3);
        pushButtoncur_pose_estimation->setObjectName(QStringLiteral("pushButtoncur_pose_estimation"));

        verticalLayout_9->addWidget(pushButtoncur_pose_estimation);

        set_curframe_as_key_frame_of_pose = new QPushButton(groupBox_3);
        set_curframe_as_key_frame_of_pose->setObjectName(QStringLiteral("set_curframe_as_key_frame_of_pose"));

        verticalLayout_9->addWidget(set_curframe_as_key_frame_of_pose);

        pushButton_whole_pose_estimation = new QPushButton(groupBox_3);
        pushButton_whole_pose_estimation->setObjectName(QStringLiteral("pushButton_whole_pose_estimation"));

        verticalLayout_9->addWidget(pushButton_whole_pose_estimation);

        pushButton_correspondence = new QPushButton(groupBox_3);
        pushButton_correspondence->setObjectName(QStringLiteral("pushButton_correspondence"));

        verticalLayout_9->addWidget(pushButton_correspondence);

        caculateCorredTexture = new QPushButton(groupBox_3);
        caculateCorredTexture->setObjectName(QStringLiteral("caculateCorredTexture"));

        verticalLayout_9->addWidget(caculateCorredTexture);

        caculateAllCorredTexture = new QPushButton(groupBox_3);
        caculateAllCorredTexture->setObjectName(QStringLiteral("caculateAllCorredTexture"));

        verticalLayout_9->addWidget(caculateAllCorredTexture);


        horizontalLayout_3->addWidget(groupBox_3);


        horizontalLayout_7->addWidget(groupBox);

        tabWidget_algorithom->addTab(tab_pose_estimation, QString());
        tab_simulate = new QWidget();
        tab_simulate->setObjectName(QStringLiteral("tab_simulate"));
        horizontalLayout_9 = new QHBoxLayout(tab_simulate);
        horizontalLayout_9->setObjectName(QStringLiteral("horizontalLayout_9"));
        groupBox_2 = new QGroupBox(tab_simulate);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        verticalLayout_10 = new QVBoxLayout(groupBox_2);
        verticalLayout_10->setObjectName(QStringLiteral("verticalLayout_10"));
        begin_simulate = new QPushButton(groupBox_2);
        begin_simulate->setObjectName(QStringLiteral("begin_simulate"));

        verticalLayout_10->addWidget(begin_simulate);

        pause_simulate = new QPushButton(groupBox_2);
        pause_simulate->setObjectName(QStringLiteral("pause_simulate"));

        verticalLayout_10->addWidget(pause_simulate);

        continue_simulate = new QPushButton(groupBox_2);
        continue_simulate->setObjectName(QStringLiteral("continue_simulate"));

        verticalLayout_10->addWidget(continue_simulate);

        restart = new QPushButton(groupBox_2);
        restart->setObjectName(QStringLiteral("restart"));

        verticalLayout_10->addWidget(restart);

        step_forward = new QPushButton(groupBox_2);
        step_forward->setObjectName(QStringLiteral("step_forward"));

        verticalLayout_10->addWidget(step_forward);


        horizontalLayout_9->addWidget(groupBox_2);

        groupBox_4 = new QGroupBox(tab_simulate);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        gridLayout_7 = new QGridLayout(groupBox_4);
        gridLayout_7->setObjectName(QStringLiteral("gridLayout_7"));
        label_15 = new QLabel(groupBox_4);
        label_15->setObjectName(QStringLiteral("label_15"));

        gridLayout_7->addWidget(label_15, 0, 0, 1, 1);

        label_16 = new QLabel(groupBox_4);
        label_16->setObjectName(QStringLiteral("label_16"));

        gridLayout_7->addWidget(label_16, 1, 0, 1, 1);

        lineEdit = new QLineEdit(groupBox_4);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));

        gridLayout_7->addWidget(lineEdit, 0, 1, 1, 1);

        lineEdit_2 = new QLineEdit(groupBox_4);
        lineEdit_2->setObjectName(QStringLiteral("lineEdit_2"));

        gridLayout_7->addWidget(lineEdit_2, 1, 1, 1, 1);


        horizontalLayout_9->addWidget(groupBox_4);

        constraint = new QGroupBox(tab_simulate);
        constraint->setObjectName(QStringLiteral("constraint"));
        verticalLayout_11 = new QVBoxLayout(constraint);
        verticalLayout_11->setObjectName(QStringLiteral("verticalLayout_11"));
        setStrongFaceConstraint = new QPushButton(constraint);
        setStrongFaceConstraint->setObjectName(QStringLiteral("setStrongFaceConstraint"));

        verticalLayout_11->addWidget(setStrongFaceConstraint);

        setWeakFaceConstraint = new QPushButton(constraint);
        setWeakFaceConstraint->setObjectName(QStringLiteral("setWeakFaceConstraint"));

        verticalLayout_11->addWidget(setWeakFaceConstraint);

        unsetFaceConstraint = new QPushButton(constraint);
        unsetFaceConstraint->setObjectName(QStringLiteral("unsetFaceConstraint"));

        verticalLayout_11->addWidget(unsetFaceConstraint);

        showConstraint = new QPushButton(constraint);
        showConstraint->setObjectName(QStringLiteral("showConstraint"));

        verticalLayout_11->addWidget(showConstraint);

        unshowConstraint = new QPushButton(constraint);
        unshowConstraint->setObjectName(QStringLiteral("unshowConstraint"));

        verticalLayout_11->addWidget(unshowConstraint);


        horizontalLayout_9->addWidget(constraint);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_9->addItem(horizontalSpacer_3);

        horizontalLayout_9->setStretch(0, 1);
        horizontalLayout_9->setStretch(1, 1);
        horizontalLayout_9->setStretch(3, 3);
        tabWidget_algorithom->addTab(tab_simulate, QString());

        verticalLayout->addWidget(tabWidget_algorithom);

        verticalLayout->setStretch(0, 1);
        verticalLayout->setStretch(1, 2);

        verticalLayout_5->addLayout(verticalLayout);

        VideoEditingWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(VideoEditingWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 856, 23));
        menuFile = new QMenu(menubar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuTools = new QMenu(menubar);
        menuTools->setObjectName(QStringLiteral("menuTools"));
        VideoEditingWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(VideoEditingWindow);
        statusbar->setObjectName(QStringLiteral("statusbar"));
        VideoEditingWindow->setStatusBar(statusbar);

        menubar->addAction(menuFile->menuAction());
        menubar->addAction(menuTools->menuAction());
        menuFile->addAction(actionOpen);
        menuFile->addAction(actionSave_as);
        menuFile->addAction(actionClose);
        menuFile->addAction(actionLoad_model);
        menuFile->addAction(actionLoad_poses);
        menuFile->addAction(actionSave_poses);
        menuTools->addAction(actionRead_frame);
        menuTools->addAction(actionCamera);
        menuTools->addAction(actionMatting);
        menuTools->addAction(actionWrite_video);
        menuTools->addAction(actionAlpha2trimap);
        menuTools->addAction(actionSplit_Video);
        menuTools->addAction(actionCompute_gradient);
        menuTools->addAction(actionWrite_CameraViewer_to_video);
        menuTools->addAction(actionWrite_CameraViewer_to_Image_array);
        menuTools->addAction(actionRender_CameraViewer_To_Image_array);

        retranslateUi(VideoEditingWindow);

        source_video_tab->setCurrentIndex(1);
        tabWidget_algorithom->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(VideoEditingWindow);
    } // setupUi

    void retranslateUi(QMainWindow *VideoEditingWindow)
    {
        VideoEditingWindow->setWindowTitle(QApplication::translate("VideoEditingWindow", "MainWindow", 0));
        actionOpen->setText(QApplication::translate("VideoEditingWindow", "openVideo", 0));
        actionSave_as->setText(QApplication::translate("VideoEditingWindow", "save as", 0));
        actionClose->setText(QApplication::translate("VideoEditingWindow", "close", 0));
        actionRead_frame->setText(QApplication::translate("VideoEditingWindow", "Read frame", 0));
        actionCamera->setText(QApplication::translate("VideoEditingWindow", "Camera", 0));
        actionMatting->setText(QApplication::translate("VideoEditingWindow", "Matting", 0));
        actionWrite_video->setText(QApplication::translate("VideoEditingWindow", "Write video", 0));
        actionAlpha2trimap->setText(QApplication::translate("VideoEditingWindow", "Alpha2Trimap", 0));
        actionSplit_Video->setText(QApplication::translate("VideoEditingWindow", "Split Video", 0));
        actionCompute_gradient->setText(QApplication::translate("VideoEditingWindow", "Compute gradient", 0));
        actionLoad_model->setText(QApplication::translate("VideoEditingWindow", "load model", 0));
        actionLoad_poses->setText(QApplication::translate("VideoEditingWindow", "load poses", 0));
        actionSave_poses->setText(QApplication::translate("VideoEditingWindow", "save poses", 0));
        actionWrite_CameraViewer_to_video->setText(QApplication::translate("VideoEditingWindow", "write CameraViewer to video", 0));
        actionRender_CameraViewer_To_Image_array->setText(QApplication::translate("VideoEditingWindow", "Render CameraViewer To Image array", 0));
        actionWrite_CameraViewer_to_Image_array->setText(QApplication::translate("VideoEditingWindow", "write CameraViewer to Image array", 0));
        SourceVideo->setTitle(QApplication::translate("VideoEditingWindow", "Source Video", 0));
        videoFrame->setText(QString());
        source_video_tab->setTabText(source_video_tab->indexOf(tab_imagemode), QApplication::translate("VideoEditingWindow", "Image Mode", 0));
        source_video_tab->setTabText(source_video_tab->indexOf(tab_manipulatemode), QApplication::translate("VideoEditingWindow", "Manipulate Mode", 0));
        cur_frame_idx->setText(QApplication::translate("VideoEditingWindow", "0", 0));
        label_5->setText(QApplication::translate("VideoEditingWindow", "/", 0));
        total_framenum->setText(QApplication::translate("VideoEditingWindow", "0", 0));
        pushButton_previousframe->setText(QApplication::translate("VideoEditingWindow", "Previous Frame", 0));
        pushButton_play->setText(QApplication::translate("VideoEditingWindow", "Play", 0));
        pushButton_pause->setText(QApplication::translate("VideoEditingWindow", "Pause", 0));
        pushButton_nextframe->setText(QApplication::translate("VideoEditingWindow", "Next Frame", 0));
        label_7->setText(QApplication::translate("VideoEditingWindow", "Turn to", 0));
        NextInitKey->setText(QApplication::translate("VideoEditingWindow", "Next Init Key", 0));
        SetKey->setText(QApplication::translate("VideoEditingWindow", "Set Key Frame", 0));
        toolsforstep1->setTitle(QApplication::translate("VideoEditingWindow", "Tools for Step1", 0));
        SelectArea->setText(QApplication::translate("VideoEditingWindow", "SelectArea", 0));
        BrushBack->setText(QApplication::translate("VideoEditingWindow", "Brush Background", 0));
        BrushCompute->setText(QApplication::translate("VideoEditingWindow", "Brush Compute Area", 0));
        BrushFore->setText(QApplication::translate("VideoEditingWindow", "Brush Foreground", 0));
        ForeCut->setText(QApplication::translate("VideoEditingWindow", "Foreground cut", 0));
        radioButton_7->setText(QApplication::translate("VideoEditingWindow", "RadioButton", 0));
        BackCut->setText(QApplication::translate("VideoEditingWindow", "Background cut", 0));
        label_2->setText(QApplication::translate("VideoEditingWindow", "Brush size:", 0));
        label_3->setText(QApplication::translate("VideoEditingWindow", "Trimap width:", 0));
        Grabcut->setText(QApplication::translate("VideoEditingWindow", "Grabcut Iteration", 0));
        ifshow_silhouette->setText(QApplication::translate("VideoEditingWindow", "show silhouette", 0));
        MattingSingleFrame->setText(QApplication::translate("VideoEditingWindow", "MattingSingleFrame", 0));
        ShowSilhouette->setText(QApplication::translate("VideoEditingWindow", "Show Current silhouette", 0));
        ShowTrimap->setText(QApplication::translate("VideoEditingWindow", "Show Current Trimap", 0));
        ShowAlphamap->setText(QApplication::translate("VideoEditingWindow", "Show Current Alphamap", 0));
        showKeyFrameNo->setText(QApplication::translate("VideoEditingWindow", "Show key Frame No.", 0));
        ifshowimagemask->setText(QApplication::translate("VideoEditingWindow", "show image mask", 0));
        Init_Key_Frame_by_Diff->setText(QApplication::translate("VideoEditingWindow", "Init Key \n"
" Frame by Diff", 0));
        TrimapInterpolation->setText(QApplication::translate("VideoEditingWindow", "Trimap \n"
" Interpolation", 0));
        MattingVideo->setText(QApplication::translate("VideoEditingWindow", "Matting \n"
" Video", 0));
        ChangeBackground->setText(QApplication::translate("VideoEditingWindow", "Change \n"
" background", 0));
        tabWidget_algorithom->setTabText(tabWidget_algorithom->indexOf(tab_fore_extract), QApplication::translate("VideoEditingWindow", "1)\345\211\215\346\231\257\346\243\200\346\265\213\357\274\214\350\275\256\345\273\223\346\217\220\345\217\226", 0));
        groupBox->setTitle(QApplication::translate("VideoEditingWindow", "Toolbox for Step2", 0));
        object_pose->setTitle(QApplication::translate("VideoEditingWindow", "object pose", 0));
        object_name_label->setText(QApplication::translate("VideoEditingWindow", "\347\211\251\344\275\223\345\220\215\347\247\260", 0));
        object_type_label->setText(QApplication::translate("VideoEditingWindow", "\347\211\251\344\275\223\347\261\273\345\236\213", 0));
        label->setText(QApplication::translate("VideoEditingWindow", "\345\271\263\347\247\273", 0));
        label_8->setText(QApplication::translate("VideoEditingWindow", "\346\227\213\350\275\254", 0));
        label_11->setText(QApplication::translate("VideoEditingWindow", "\347\274\251\346\224\276", 0));
        prespectuve_param->setTitle(QApplication::translate("VideoEditingWindow", "\351\200\217\350\247\206\347\237\251\351\230\265", 0));
        label_6->setText(QApplication::translate("VideoEditingWindow", "viewport \345\256\275\357\274\210\345\203\217\347\264\240\357\274\211", 0));
        label_10->setText(QApplication::translate("VideoEditingWindow", "\350\277\221\345\271\263\351\235\242", 0));
        label_9->setText(QApplication::translate("VideoEditingWindow", "\345\256\275/\351\253\230", 0));
        label_13->setText(QApplication::translate("VideoEditingWindow", "viewport \351\253\230\357\274\210\345\203\217\347\264\240\357\274\211", 0));
        label_12->setText(QApplication::translate("VideoEditingWindow", "\350\277\234\345\271\263\351\235\242", 0));
        label_14->setText(QApplication::translate("VideoEditingWindow", "\346\212\225\345\275\261\347\261\273\345\236\213", 0));
        label_4->setText(QApplication::translate("VideoEditingWindow", "fov_y(\345\272\246\357\274\211", 0));
        cameraintrisic->setTitle(QApplication::translate("VideoEditingWindow", "\346\221\204\345\203\217\346\234\272\345\206\205\345\217\202", 0));
        alpha->setText(QApplication::translate("VideoEditingWindow", "2.1848e+04", 0));
        rotate_x_3->setText(QApplication::translate("VideoEditingWindow", "0", 0));
        gama->setText(QApplication::translate("VideoEditingWindow", "0.0", 0));
        v0->setText(QApplication::translate("VideoEditingWindow", "1.3402e+03", 0));
        scale_z_3->setText(QApplication::translate("VideoEditingWindow", "1", 0));
        beta->setText(QApplication::translate("VideoEditingWindow", "1.2049e+04", 0));
        scale_x_3->setText(QApplication::translate("VideoEditingWindow", "0", 0));
        u0->setText(QApplication::translate("VideoEditingWindow", "-241.417", 0));
        scale_y_3->setText(QApplication::translate("VideoEditingWindow", "0", 0));
        groupBox_manipulate->setTitle(QApplication::translate("VideoEditingWindow", "\346\223\215\344\275\234\345\267\245\345\205\267", 0));
        radioButton_select->setText(QApplication::translate("VideoEditingWindow", "\351\200\211\346\213\251", 0));
        radioButton_selectface->setText(QApplication::translate("VideoEditingWindow", "\351\200\211\346\213\251\351\235\242", 0));
        radioButton_translate->setText(QApplication::translate("VideoEditingWindow", "\345\271\263\347\247\273", 0));
        radioButton_rotate->setText(QApplication::translate("VideoEditingWindow", "\346\227\213\350\275\254", 0));
        radioButton_scale->setText(QApplication::translate("VideoEditingWindow", "\347\274\251\346\224\276", 0));
        radioButton_fouces->setText(QApplication::translate("VideoEditingWindow", "\347\204\246\347\202\271\345\267\245\345\205\267", 0));
        groupBox_3->setTitle(QApplication::translate("VideoEditingWindow", "\345\247\277\346\200\201\344\274\260\350\256\241\347\256\227\346\263\225", 0));
        pushButtoncur_pose_estimation->setText(QApplication::translate("VideoEditingWindow", "\345\275\223\345\211\215\345\270\247\345\247\277\346\200\201\344\274\260\350\256\241", 0));
        set_curframe_as_key_frame_of_pose->setText(QApplication::translate("VideoEditingWindow", "\347\241\256\345\256\232\345\275\223\345\211\215\345\270\247\344\274\260\350\256\241", 0));
        pushButton_whole_pose_estimation->setText(QApplication::translate("VideoEditingWindow", "\345\205\250\345\270\247\345\247\277\346\200\201\344\274\260\350\256\241", 0));
        pushButton_correspondence->setText(QApplication::translate("VideoEditingWindow", "\345\257\271\345\272\224\345\205\263\347\263\273", 0));
        caculateCorredTexture->setText(QApplication::translate("VideoEditingWindow", "\350\256\241\347\256\227\345\275\223\345\211\215\345\270\247\n"
"\345\257\271\345\272\224\347\272\271\347\220\206\345\235\220\346\240\207", 0));
        caculateAllCorredTexture->setText(QApplication::translate("VideoEditingWindow", "\350\256\241\347\256\227\346\211\200\346\234\211\345\270\247\n"
"\345\257\271\345\272\224\347\272\271\347\220\206\345\235\220\346\240\207", 0));
        tabWidget_algorithom->setTabText(tabWidget_algorithom->indexOf(tab_pose_estimation), QApplication::translate("VideoEditingWindow", "2)\345\247\277\346\200\201\344\274\260\350\256\241", 0));
        groupBox_2->setTitle(QApplication::translate("VideoEditingWindow", "simulate", 0));
        begin_simulate->setText(QApplication::translate("VideoEditingWindow", "\345\274\200\345\247\213\346\250\241\346\213\237", 0));
        pause_simulate->setText(QApplication::translate("VideoEditingWindow", "\346\232\202\345\201\234\346\250\241\346\213\237", 0));
        continue_simulate->setText(QApplication::translate("VideoEditingWindow", "\347\273\247\347\273\255\346\250\241\346\213\237", 0));
        restart->setText(QApplication::translate("VideoEditingWindow", "\350\277\230\345\216\237", 0));
        step_forward->setText(QApplication::translate("VideoEditingWindow", "\345\215\225\346\255\245\346\211\247\350\241\214", 0));
        groupBox_4->setTitle(QApplication::translate("VideoEditingWindow", "info", 0));
        label_15->setText(QApplication::translate("VideoEditingWindow", "\345\275\223\345\211\215\346\250\241\346\213\237\345\270\247", 0));
        label_16->setText(QApplication::translate("VideoEditingWindow", "\346\200\273\345\270\247\346\225\260", 0));
        constraint->setTitle(QApplication::translate("VideoEditingWindow", "GroupBox", 0));
        setStrongFaceConstraint->setText(QApplication::translate("VideoEditingWindow", "strong constraint \n"
"selected face", 0));
        setWeakFaceConstraint->setText(QApplication::translate("VideoEditingWindow", "weak constraint \n"
" selected face", 0));
        unsetFaceConstraint->setText(QApplication::translate("VideoEditingWindow", "unconstraint selected face", 0));
        showConstraint->setText(QApplication::translate("VideoEditingWindow", "showConstraint", 0));
        unshowConstraint->setText(QApplication::translate("VideoEditingWindow", "unshowConstraint", 0));
        tabWidget_algorithom->setTabText(tabWidget_algorithom->indexOf(tab_simulate), QApplication::translate("VideoEditingWindow", "3)\345\275\242\347\212\266\347\274\226\350\276\221\357\274\210\347\211\251\347\220\206\346\250\241\346\213\237\357\274\214\345\237\272\344\272\216\346\240\267\344\276\213\347\232\204\345\217\230\345\275\242\357\274\211", 0));
        menuFile->setTitle(QApplication::translate("VideoEditingWindow", "File", 0));
        menuTools->setTitle(QApplication::translate("VideoEditingWindow", "Tools", 0));
    } // retranslateUi

};

namespace Ui {
    class VideoEditingWindow: public Ui_VideoEditingWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VIDEOEDITINGSCENE_H
