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
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolButton>
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
    QHBoxLayout *horizontalLayout_9;
    QSplitter *splitter;
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout_10;
    QScrollArea *scrollArea_2;
    QWidget *scrollAreaWidgetContents_3;
    QHBoxLayout *horizontalLayout_10;
    QFrame *frame;
    QHBoxLayout *horizontalLayout_13;
    QToolButton *solid1;
    QToolButton *wireframe1;
    QToolButton *transparent1;
    QToolButton *texture1;
    QSpacerItem *horizontalSpacer_3;
    QFrame *frame_cameraview;
    QWidget *verticalLayoutWidget_2;
    QVBoxLayout *verticalLayout_11;
    QScrollArea *scrollArea_3;
    QWidget *scrollAreaWidgetContents_4;
    QHBoxLayout *horizontalLayout_11;
    QFrame *frame_2;
    QHBoxLayout *horizontalLayout_12;
    QToolButton *solid2;
    QToolButton *wireframe2;
    QToolButton *transparent2;
    QToolButton *texture2;
    QSpacerItem *horizontalSpacer_4;
    QFrame *frame_worldview;
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
    QPushButton *pushButton_5;
    QPushButton *Grabcut;
    QPushButton *showKeyFrameNo;
    QPushButton *MattingSingleFrame;
    QPushButton *pushButton_6;
    QPushButton *ShowTrimap;
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
    QLabel *label_11;
    QLabel *label_8;
    QLabel *label;
    QDoubleSpinBox *translate_x;
    QDoubleSpinBox *translate_z;
    QDoubleSpinBox *scale_x;
    QDoubleSpinBox *scale_z;
    QDoubleSpinBox *rotate_z;
    QDoubleSpinBox *translate_y;
    QDoubleSpinBox *rotate_y;
    QLabel *object_name_label;
    QDoubleSpinBox *scale_y;
    QDoubleSpinBox *rotate_x;
    QLabel *object_type_label;
    QLineEdit *object_type;
    QLineEdit *object_name;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout_6;
    QLineEdit *alpha;
    QLineEdit *gama;
    QLineEdit *u0;
    QLineEdit *rotate_x_3;
    QLineEdit *beta;
    QLineEdit *v0;
    QLineEdit *scale_x_3;
    QLineEdit *scale_y_3;
    QLineEdit *scale_z_3;
    QGroupBox *groupBox_4;
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
    QPushButton *pushButton_whole_pose_estimation;
    QPushButton *pushButton_correspondence;
    QWidget *tab_simulate;
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
        horizontalLayout_9 = new QHBoxLayout(frame_manipulate);
        horizontalLayout_9->setSpacing(0);
        horizontalLayout_9->setObjectName(QStringLiteral("horizontalLayout_9"));
        horizontalLayout_9->setContentsMargins(0, 0, 0, 0);
        splitter = new QSplitter(frame_manipulate);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        verticalLayoutWidget = new QWidget(splitter);
        verticalLayoutWidget->setObjectName(QStringLiteral("verticalLayoutWidget"));
        verticalLayout_10 = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout_10->setSpacing(0);
        verticalLayout_10->setObjectName(QStringLiteral("verticalLayout_10"));
        verticalLayout_10->setContentsMargins(0, 0, 0, 0);
        scrollArea_2 = new QScrollArea(verticalLayoutWidget);
        scrollArea_2->setObjectName(QStringLiteral("scrollArea_2"));
        scrollArea_2->setWidgetResizable(true);
        scrollAreaWidgetContents_3 = new QWidget();
        scrollAreaWidgetContents_3->setObjectName(QStringLiteral("scrollAreaWidgetContents_3"));
        scrollAreaWidgetContents_3->setGeometry(QRect(0, 0, 600, 52));
        scrollAreaWidgetContents_3->setMinimumSize(QSize(600, 0));
        horizontalLayout_10 = new QHBoxLayout(scrollAreaWidgetContents_3);
        horizontalLayout_10->setSpacing(0);
        horizontalLayout_10->setObjectName(QStringLiteral("horizontalLayout_10"));
        horizontalLayout_10->setContentsMargins(0, 0, -1, 0);
        frame = new QFrame(scrollAreaWidgetContents_3);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        horizontalLayout_13 = new QHBoxLayout(frame);
        horizontalLayout_13->setSpacing(0);
        horizontalLayout_13->setObjectName(QStringLiteral("horizontalLayout_13"));
        horizontalLayout_13->setContentsMargins(0, 0, 0, 0);
        solid1 = new QToolButton(frame);
        solid1->setObjectName(QStringLiteral("solid1"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(solid1->sizePolicy().hasHeightForWidth());
        solid1->setSizePolicy(sizePolicy1);
        solid1->setIconSize(QSize(32, 32));
        solid1->setAutoExclusive(true);

        horizontalLayout_13->addWidget(solid1);

        wireframe1 = new QToolButton(frame);
        wireframe1->setObjectName(QStringLiteral("wireframe1"));
        sizePolicy1.setHeightForWidth(wireframe1->sizePolicy().hasHeightForWidth());
        wireframe1->setSizePolicy(sizePolicy1);
        wireframe1->setIconSize(QSize(32, 32));
        wireframe1->setAutoExclusive(true);

        horizontalLayout_13->addWidget(wireframe1);

        transparent1 = new QToolButton(frame);
        transparent1->setObjectName(QStringLiteral("transparent1"));
        sizePolicy1.setHeightForWidth(transparent1->sizePolicy().hasHeightForWidth());
        transparent1->setSizePolicy(sizePolicy1);
        transparent1->setIconSize(QSize(32, 32));
        transparent1->setAutoExclusive(true);

        horizontalLayout_13->addWidget(transparent1);

        texture1 = new QToolButton(frame);
        texture1->setObjectName(QStringLiteral("texture1"));
        sizePolicy1.setHeightForWidth(texture1->sizePolicy().hasHeightForWidth());
        texture1->setSizePolicy(sizePolicy1);
        texture1->setIconSize(QSize(32, 32));
        texture1->setAutoExclusive(true);

        horizontalLayout_13->addWidget(texture1);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_13->addItem(horizontalSpacer_3);


        horizontalLayout_10->addWidget(frame);

        scrollArea_2->setWidget(scrollAreaWidgetContents_3);

        verticalLayout_10->addWidget(scrollArea_2);

        frame_cameraview = new QFrame(verticalLayoutWidget);
        frame_cameraview->setObjectName(QStringLiteral("frame_cameraview"));
        frame_cameraview->setFrameShape(QFrame::Box);
        frame_cameraview->setFrameShadow(QFrame::Raised);

        verticalLayout_10->addWidget(frame_cameraview);

        verticalLayout_10->setStretch(0, 1);
        verticalLayout_10->setStretch(1, 20);
        splitter->addWidget(verticalLayoutWidget);
        verticalLayoutWidget_2 = new QWidget(splitter);
        verticalLayoutWidget_2->setObjectName(QStringLiteral("verticalLayoutWidget_2"));
        verticalLayout_11 = new QVBoxLayout(verticalLayoutWidget_2);
        verticalLayout_11->setSpacing(0);
        verticalLayout_11->setObjectName(QStringLiteral("verticalLayout_11"));
        verticalLayout_11->setContentsMargins(0, 0, 0, 0);
        scrollArea_3 = new QScrollArea(verticalLayoutWidget_2);
        scrollArea_3->setObjectName(QStringLiteral("scrollArea_3"));
        scrollArea_3->setWidgetResizable(true);
        scrollAreaWidgetContents_4 = new QWidget();
        scrollAreaWidgetContents_4->setObjectName(QStringLiteral("scrollAreaWidgetContents_4"));
        scrollAreaWidgetContents_4->setGeometry(QRect(0, 0, 600, 52));
        scrollAreaWidgetContents_4->setMinimumSize(QSize(600, 0));
        horizontalLayout_11 = new QHBoxLayout(scrollAreaWidgetContents_4);
        horizontalLayout_11->setSpacing(0);
        horizontalLayout_11->setObjectName(QStringLiteral("horizontalLayout_11"));
        horizontalLayout_11->setContentsMargins(-1, 0, 0, 0);
        frame_2 = new QFrame(scrollAreaWidgetContents_4);
        frame_2->setObjectName(QStringLiteral("frame_2"));
        frame_2->setFrameShape(QFrame::StyledPanel);
        frame_2->setFrameShadow(QFrame::Raised);
        horizontalLayout_12 = new QHBoxLayout(frame_2);
        horizontalLayout_12->setSpacing(0);
        horizontalLayout_12->setObjectName(QStringLiteral("horizontalLayout_12"));
        horizontalLayout_12->setContentsMargins(0, 0, 0, 0);
        solid2 = new QToolButton(frame_2);
        solid2->setObjectName(QStringLiteral("solid2"));
        sizePolicy1.setHeightForWidth(solid2->sizePolicy().hasHeightForWidth());
        solid2->setSizePolicy(sizePolicy1);
        solid2->setIconSize(QSize(32, 32));
        solid2->setAutoExclusive(true);

        horizontalLayout_12->addWidget(solid2);

        wireframe2 = new QToolButton(frame_2);
        wireframe2->setObjectName(QStringLiteral("wireframe2"));
        sizePolicy1.setHeightForWidth(wireframe2->sizePolicy().hasHeightForWidth());
        wireframe2->setSizePolicy(sizePolicy1);
        wireframe2->setIconSize(QSize(32, 32));
        wireframe2->setAutoExclusive(true);

        horizontalLayout_12->addWidget(wireframe2);

        transparent2 = new QToolButton(frame_2);
        transparent2->setObjectName(QStringLiteral("transparent2"));
        sizePolicy1.setHeightForWidth(transparent2->sizePolicy().hasHeightForWidth());
        transparent2->setSizePolicy(sizePolicy1);
        transparent2->setIconSize(QSize(32, 32));
        transparent2->setAutoExclusive(true);

        horizontalLayout_12->addWidget(transparent2);

        texture2 = new QToolButton(frame_2);
        texture2->setObjectName(QStringLiteral("texture2"));
        sizePolicy1.setHeightForWidth(texture2->sizePolicy().hasHeightForWidth());
        texture2->setSizePolicy(sizePolicy1);
        texture2->setIconSize(QSize(32, 32));
        texture2->setAutoExclusive(true);

        horizontalLayout_12->addWidget(texture2);

        horizontalSpacer_4 = new QSpacerItem(430, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_12->addItem(horizontalSpacer_4);


        horizontalLayout_11->addWidget(frame_2);

        scrollArea_3->setWidget(scrollAreaWidgetContents_4);

        verticalLayout_11->addWidget(scrollArea_3);

        frame_worldview = new QFrame(verticalLayoutWidget_2);
        frame_worldview->setObjectName(QStringLiteral("frame_worldview"));
        frame_worldview->setFrameShape(QFrame::Box);
        frame_worldview->setFrameShadow(QFrame::Raised);

        verticalLayout_11->addWidget(frame_worldview);

        verticalLayout_11->setStretch(0, 1);
        verticalLayout_11->setStretch(1, 20);
        splitter->addWidget(verticalLayoutWidget_2);

        horizontalLayout_9->addWidget(splitter);


        verticalLayout_7->addWidget(frame_manipulate);

        source_video_tab->addTab(tab_manipulatemode, QString());

        verticalLayout_6->addWidget(source_video_tab);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        cur_frame_idx = new QLabel(SourceVideo);
        cur_frame_idx->setObjectName(QStringLiteral("cur_frame_idx"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(cur_frame_idx->sizePolicy().hasHeightForWidth());
        cur_frame_idx->setSizePolicy(sizePolicy2);
        cur_frame_idx->setMinimumSize(QSize(60, 0));
        QFont font;
        font.setFamily(QString::fromUtf8("\351\273\221\344\275\223"));
        font.setPointSize(20);
        cur_frame_idx->setFont(font);
        cur_frame_idx->setAlignment(Qt::AlignCenter);

        horizontalLayout_2->addWidget(cur_frame_idx);

        label_5 = new QLabel(SourceVideo);
        label_5->setObjectName(QStringLiteral("label_5"));
        sizePolicy2.setHeightForWidth(label_5->sizePolicy().hasHeightForWidth());
        label_5->setSizePolicy(sizePolicy2);
        label_5->setMinimumSize(QSize(40, 0));
        label_5->setFont(font);
        label_5->setAlignment(Qt::AlignCenter);

        horizontalLayout_2->addWidget(label_5);

        total_framenum = new QLabel(SourceVideo);
        total_framenum->setObjectName(QStringLiteral("total_framenum"));
        sizePolicy2.setHeightForWidth(total_framenum->sizePolicy().hasHeightForWidth());
        total_framenum->setSizePolicy(sizePolicy2);
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
        QSizePolicy sizePolicy3(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(SetKey->sizePolicy().hasHeightForWidth());
        SetKey->setSizePolicy(sizePolicy3);

        horizontalLayout->addWidget(SetKey);


        horizontalLayout_2->addLayout(horizontalLayout);


        verticalLayout_6->addLayout(horizontalLayout_2);


        verticalLayout->addWidget(SourceVideo);

        tabWidget_algorithom = new QTabWidget(centralwidget);
        tabWidget_algorithom->setObjectName(QStringLiteral("tabWidget_algorithom"));
        QSizePolicy sizePolicy4(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(tabWidget_algorithom->sizePolicy().hasHeightForWidth());
        tabWidget_algorithom->setSizePolicy(sizePolicy4);
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
        verticalLayout_8->setObjectName(QStringLiteral("verticalLayout_8"));
        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        label_2 = new QLabel(toolsforstep1);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        brushSizeSlider = new QSlider(toolsforstep1);
        brushSizeSlider->setObjectName(QStringLiteral("brushSizeSlider"));
        sizePolicy4.setHeightForWidth(brushSizeSlider->sizePolicy().hasHeightForWidth());
        brushSizeSlider->setSizePolicy(sizePolicy4);
        brushSizeSlider->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(brushSizeSlider, 1, 1, 1, 1);

        kernelSizeSlider = new QSlider(toolsforstep1);
        kernelSizeSlider->setObjectName(QStringLiteral("kernelSizeSlider"));
        QSizePolicy sizePolicy5(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(kernelSizeSlider->sizePolicy().hasHeightForWidth());
        kernelSizeSlider->setSizePolicy(sizePolicy5);
        kernelSizeSlider->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(kernelSizeSlider, 2, 1, 1, 1);

        label_3 = new QLabel(toolsforstep1);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);


        verticalLayout_8->addLayout(gridLayout);

        gridLayout_2 = new QGridLayout();
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        pushButton_5 = new QPushButton(toolsforstep1);
        pushButton_5->setObjectName(QStringLiteral("pushButton_5"));

        gridLayout_2->addWidget(pushButton_5, 1, 2, 1, 1);

        Grabcut = new QPushButton(toolsforstep1);
        Grabcut->setObjectName(QStringLiteral("Grabcut"));

        gridLayout_2->addWidget(Grabcut, 1, 0, 1, 1);

        showKeyFrameNo = new QPushButton(toolsforstep1);
        showKeyFrameNo->setObjectName(QStringLiteral("showKeyFrameNo"));

        gridLayout_2->addWidget(showKeyFrameNo, 2, 0, 1, 1);

        MattingSingleFrame = new QPushButton(toolsforstep1);
        MattingSingleFrame->setObjectName(QStringLiteral("MattingSingleFrame"));

        gridLayout_2->addWidget(MattingSingleFrame, 2, 1, 1, 1);

        pushButton_6 = new QPushButton(toolsforstep1);
        pushButton_6->setObjectName(QStringLiteral("pushButton_6"));

        gridLayout_2->addWidget(pushButton_6, 2, 2, 1, 1);

        ShowTrimap = new QPushButton(toolsforstep1);
        ShowTrimap->setObjectName(QStringLiteral("ShowTrimap"));

        gridLayout_2->addWidget(ShowTrimap, 1, 1, 1, 1);


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
        horizontalLayout_3->setSpacing(0);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        scrollArea_object_info = new QScrollArea(groupBox);
        scrollArea_object_info->setObjectName(QStringLiteral("scrollArea_object_info"));
        scrollArea_object_info->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QStringLiteral("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 1000, 158));
        scrollAreaWidgetContents->setMinimumSize(QSize(1000, 0));
        horizontalLayout_8 = new QHBoxLayout(scrollAreaWidgetContents);
        horizontalLayout_8->setSpacing(0);
        horizontalLayout_8->setObjectName(QStringLiteral("horizontalLayout_8"));
        horizontalLayout_8->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(1);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalLayout_4->setSizeConstraint(QLayout::SetFixedSize);
        object_pose = new QGroupBox(scrollAreaWidgetContents);
        object_pose->setObjectName(QStringLiteral("object_pose"));
        object_pose->setMinimumSize(QSize(0, 0));
        gridLayout_4 = new QGridLayout(object_pose);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        gridLayout_4->setSizeConstraint(QLayout::SetDefaultConstraint);
        label_11 = new QLabel(object_pose);
        label_11->setObjectName(QStringLiteral("label_11"));

        gridLayout_4->addWidget(label_11, 4, 0, 1, 1);

        label_8 = new QLabel(object_pose);
        label_8->setObjectName(QStringLiteral("label_8"));

        gridLayout_4->addWidget(label_8, 3, 0, 1, 1);

        label = new QLabel(object_pose);
        label->setObjectName(QStringLiteral("label"));

        gridLayout_4->addWidget(label, 2, 0, 1, 1);

        translate_x = new QDoubleSpinBox(object_pose);
        translate_x->setObjectName(QStringLiteral("translate_x"));

        gridLayout_4->addWidget(translate_x, 2, 1, 1, 1);

        translate_z = new QDoubleSpinBox(object_pose);
        translate_z->setObjectName(QStringLiteral("translate_z"));

        gridLayout_4->addWidget(translate_z, 2, 3, 1, 1);

        scale_x = new QDoubleSpinBox(object_pose);
        scale_x->setObjectName(QStringLiteral("scale_x"));

        gridLayout_4->addWidget(scale_x, 4, 1, 1, 1);

        scale_z = new QDoubleSpinBox(object_pose);
        scale_z->setObjectName(QStringLiteral("scale_z"));

        gridLayout_4->addWidget(scale_z, 4, 3, 1, 1);

        rotate_z = new QDoubleSpinBox(object_pose);
        rotate_z->setObjectName(QStringLiteral("rotate_z"));

        gridLayout_4->addWidget(rotate_z, 3, 3, 1, 1);

        translate_y = new QDoubleSpinBox(object_pose);
        translate_y->setObjectName(QStringLiteral("translate_y"));

        gridLayout_4->addWidget(translate_y, 2, 2, 1, 1);

        rotate_y = new QDoubleSpinBox(object_pose);
        rotate_y->setObjectName(QStringLiteral("rotate_y"));

        gridLayout_4->addWidget(rotate_y, 3, 2, 1, 1);

        object_name_label = new QLabel(object_pose);
        object_name_label->setObjectName(QStringLiteral("object_name_label"));

        gridLayout_4->addWidget(object_name_label, 0, 0, 1, 1);

        scale_y = new QDoubleSpinBox(object_pose);
        scale_y->setObjectName(QStringLiteral("scale_y"));

        gridLayout_4->addWidget(scale_y, 4, 2, 1, 1);

        rotate_x = new QDoubleSpinBox(object_pose);
        rotate_x->setObjectName(QStringLiteral("rotate_x"));

        gridLayout_4->addWidget(rotate_x, 3, 1, 1, 1);

        object_type_label = new QLabel(object_pose);
        object_type_label->setObjectName(QStringLiteral("object_type_label"));

        gridLayout_4->addWidget(object_type_label, 1, 0, 1, 1);

        object_type = new QLineEdit(object_pose);
        object_type->setObjectName(QStringLiteral("object_type"));

        gridLayout_4->addWidget(object_type, 1, 1, 1, 3);

        object_name = new QLineEdit(object_pose);
        object_name->setObjectName(QStringLiteral("object_name"));

        gridLayout_4->addWidget(object_name, 0, 1, 1, 3);


        horizontalLayout_4->addWidget(object_pose);


        horizontalLayout_8->addLayout(horizontalLayout_4);

        groupBox_2 = new QGroupBox(scrollAreaWidgetContents);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        gridLayout_6 = new QGridLayout(groupBox_2);
        gridLayout_6->setObjectName(QStringLiteral("gridLayout_6"));
        alpha = new QLineEdit(groupBox_2);
        alpha->setObjectName(QStringLiteral("alpha"));

        gridLayout_6->addWidget(alpha, 0, 0, 1, 1);

        gama = new QLineEdit(groupBox_2);
        gama->setObjectName(QStringLiteral("gama"));

        gridLayout_6->addWidget(gama, 0, 1, 1, 1);

        u0 = new QLineEdit(groupBox_2);
        u0->setObjectName(QStringLiteral("u0"));

        gridLayout_6->addWidget(u0, 0, 2, 1, 1);

        rotate_x_3 = new QLineEdit(groupBox_2);
        rotate_x_3->setObjectName(QStringLiteral("rotate_x_3"));

        gridLayout_6->addWidget(rotate_x_3, 1, 0, 1, 1);

        beta = new QLineEdit(groupBox_2);
        beta->setObjectName(QStringLiteral("beta"));

        gridLayout_6->addWidget(beta, 1, 1, 1, 1);

        v0 = new QLineEdit(groupBox_2);
        v0->setObjectName(QStringLiteral("v0"));

        gridLayout_6->addWidget(v0, 1, 2, 1, 1);

        scale_x_3 = new QLineEdit(groupBox_2);
        scale_x_3->setObjectName(QStringLiteral("scale_x_3"));

        gridLayout_6->addWidget(scale_x_3, 2, 0, 1, 1);

        scale_y_3 = new QLineEdit(groupBox_2);
        scale_y_3->setObjectName(QStringLiteral("scale_y_3"));

        gridLayout_6->addWidget(scale_y_3, 2, 1, 1, 1);

        scale_z_3 = new QLineEdit(groupBox_2);
        scale_z_3->setObjectName(QStringLiteral("scale_z_3"));

        gridLayout_6->addWidget(scale_z_3, 2, 2, 1, 1);


        horizontalLayout_8->addWidget(groupBox_2);

        groupBox_4 = new QGroupBox(scrollAreaWidgetContents);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));

        horizontalLayout_8->addWidget(groupBox_4);

        horizontalLayout_8->setStretch(0, 1);
        horizontalLayout_8->setStretch(1, 1);
        horizontalLayout_8->setStretch(2, 1);
        scrollArea_object_info->setWidget(scrollAreaWidgetContents);

        horizontalLayout_3->addWidget(scrollArea_object_info);

        groupBox_manipulate = new QGroupBox(groupBox);
        groupBox_manipulate->setObjectName(QStringLiteral("groupBox_manipulate"));
        verticalLayout_4 = new QVBoxLayout(groupBox_manipulate);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
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
        verticalLayout_9->setObjectName(QStringLiteral("verticalLayout_9"));
        pushButtoncur_pose_estimation = new QPushButton(groupBox_3);
        pushButtoncur_pose_estimation->setObjectName(QStringLiteral("pushButtoncur_pose_estimation"));

        verticalLayout_9->addWidget(pushButtoncur_pose_estimation);

        pushButton_whole_pose_estimation = new QPushButton(groupBox_3);
        pushButton_whole_pose_estimation->setObjectName(QStringLiteral("pushButton_whole_pose_estimation"));

        verticalLayout_9->addWidget(pushButton_whole_pose_estimation);

        pushButton_correspondence = new QPushButton(groupBox_3);
        pushButton_correspondence->setObjectName(QStringLiteral("pushButton_correspondence"));

        verticalLayout_9->addWidget(pushButton_correspondence);


        horizontalLayout_3->addWidget(groupBox_3);

        horizontalLayout_3->setStretch(0, 2);
        horizontalLayout_3->setStretch(1, 1);
        horizontalLayout_3->setStretch(2, 1);

        horizontalLayout_7->addWidget(groupBox);

        tabWidget_algorithom->addTab(tab_pose_estimation, QString());
        tab_simulate = new QWidget();
        tab_simulate->setObjectName(QStringLiteral("tab_simulate"));
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
        menuTools->addAction(actionRead_frame);
        menuTools->addAction(actionCamera);
        menuTools->addAction(actionMatting);
        menuTools->addAction(actionWrite_video);
        menuTools->addAction(actionAlpha2trimap);
        menuTools->addAction(actionSplit_Video);
        menuTools->addAction(actionCompute_gradient);

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
        SourceVideo->setTitle(QApplication::translate("VideoEditingWindow", "Source Video", 0));
        videoFrame->setText(QString());
        source_video_tab->setTabText(source_video_tab->indexOf(tab_imagemode), QApplication::translate("VideoEditingWindow", "Image Mode", 0));
        solid1->setText(QApplication::translate("VideoEditingWindow", "...", 0));
        wireframe1->setText(QApplication::translate("VideoEditingWindow", "...", 0));
        transparent1->setText(QApplication::translate("VideoEditingWindow", "...", 0));
        texture1->setText(QApplication::translate("VideoEditingWindow", "...", 0));
        solid2->setText(QApplication::translate("VideoEditingWindow", "...", 0));
        wireframe2->setText(QApplication::translate("VideoEditingWindow", "...", 0));
        transparent2->setText(QApplication::translate("VideoEditingWindow", "...", 0));
        texture2->setText(QApplication::translate("VideoEditingWindow", "...", 0));
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
        pushButton_5->setText(QApplication::translate("VideoEditingWindow", "PushButton", 0));
        Grabcut->setText(QApplication::translate("VideoEditingWindow", "Grabcut Iteration", 0));
        showKeyFrameNo->setText(QApplication::translate("VideoEditingWindow", "Show key Frame No.", 0));
        MattingSingleFrame->setText(QApplication::translate("VideoEditingWindow", "MattingSingleFrame", 0));
        pushButton_6->setText(QApplication::translate("VideoEditingWindow", "PushButton", 0));
        ShowTrimap->setText(QApplication::translate("VideoEditingWindow", "Show Current Trimap", 0));
        Init_Key_Frame_by_Diff->setText(QApplication::translate("VideoEditingWindow", "Init Key Frame by Diff", 0));
        TrimapInterpolation->setText(QApplication::translate("VideoEditingWindow", "TrimapInterpolation", 0));
        MattingVideo->setText(QApplication::translate("VideoEditingWindow", "MattingVideo", 0));
        ChangeBackground->setText(QApplication::translate("VideoEditingWindow", "Change background", 0));
        tabWidget_algorithom->setTabText(tabWidget_algorithom->indexOf(tab_fore_extract), QApplication::translate("VideoEditingWindow", "1)\345\211\215\346\231\257\346\243\200\346\265\213\357\274\214\350\275\256\345\273\223\346\217\220\345\217\226", 0));
        groupBox->setTitle(QApplication::translate("VideoEditingWindow", "Toolbox for Step2", 0));
        object_pose->setTitle(QApplication::translate("VideoEditingWindow", "object pose", 0));
        label_11->setText(QApplication::translate("VideoEditingWindow", "\347\274\251\346\224\276", 0));
        label_8->setText(QApplication::translate("VideoEditingWindow", "\346\227\213\350\275\254", 0));
        label->setText(QApplication::translate("VideoEditingWindow", "\345\271\263\347\247\273", 0));
        object_name_label->setText(QApplication::translate("VideoEditingWindow", "\347\211\251\344\275\223\345\220\215\347\247\260", 0));
        object_type_label->setText(QApplication::translate("VideoEditingWindow", "\347\211\251\344\275\223\347\261\273\345\236\213", 0));
        groupBox_2->setTitle(QApplication::translate("VideoEditingWindow", "\346\221\204\345\203\217\346\234\272\345\206\205\345\217\202", 0));
        alpha->setText(QApplication::translate("VideoEditingWindow", "2.1848e+04", 0));
        gama->setText(QApplication::translate("VideoEditingWindow", "0.0", 0));
        u0->setText(QApplication::translate("VideoEditingWindow", "-241.417", 0));
        rotate_x_3->setText(QApplication::translate("VideoEditingWindow", "0", 0));
        beta->setText(QApplication::translate("VideoEditingWindow", "1.2049e+04", 0));
        v0->setText(QApplication::translate("VideoEditingWindow", "1.3402e+03", 0));
        scale_x_3->setText(QApplication::translate("VideoEditingWindow", "0", 0));
        scale_y_3->setText(QApplication::translate("VideoEditingWindow", "0", 0));
        scale_z_3->setText(QApplication::translate("VideoEditingWindow", "1", 0));
        groupBox_4->setTitle(QApplication::translate("VideoEditingWindow", "GroupBox", 0));
        groupBox_manipulate->setTitle(QApplication::translate("VideoEditingWindow", "\346\223\215\344\275\234\345\267\245\345\205\267", 0));
        radioButton_select->setText(QApplication::translate("VideoEditingWindow", "\351\200\211\346\213\251", 0));
        radioButton_selectface->setText(QApplication::translate("VideoEditingWindow", "\351\200\211\346\213\251\351\235\242", 0));
        radioButton_translate->setText(QApplication::translate("VideoEditingWindow", "\345\271\263\347\247\273", 0));
        radioButton_rotate->setText(QApplication::translate("VideoEditingWindow", "\346\227\213\350\275\254", 0));
        radioButton_scale->setText(QApplication::translate("VideoEditingWindow", "\347\274\251\346\224\276", 0));
        radioButton_fouces->setText(QApplication::translate("VideoEditingWindow", "\347\204\246\347\202\271\345\267\245\345\205\267", 0));
        groupBox_3->setTitle(QApplication::translate("VideoEditingWindow", "\345\247\277\346\200\201\344\274\260\350\256\241\347\256\227\346\263\225", 0));
        pushButtoncur_pose_estimation->setText(QApplication::translate("VideoEditingWindow", "\345\275\223\345\211\215\345\270\247\345\247\277\346\200\201\344\274\260\350\256\241", 0));
        pushButton_whole_pose_estimation->setText(QApplication::translate("VideoEditingWindow", "\345\205\250\345\270\247\345\247\277\346\200\201\344\274\260\350\256\241", 0));
        pushButton_correspondence->setText(QApplication::translate("VideoEditingWindow", "\345\257\271\345\272\224\345\205\263\347\263\273", 0));
        tabWidget_algorithom->setTabText(tabWidget_algorithom->indexOf(tab_pose_estimation), QApplication::translate("VideoEditingWindow", "2)\345\247\277\346\200\201\344\274\260\350\256\241", 0));
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
