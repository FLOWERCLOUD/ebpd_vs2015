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
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
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
        verticalLayout_7->setObjectName(QStringLiteral("verticalLayout_7"));
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
        menuTools->addAction(actionRead_frame);
        menuTools->addAction(actionCamera);
        menuTools->addAction(actionMatting);
        menuTools->addAction(actionWrite_video);
        menuTools->addAction(actionAlpha2trimap);
        menuTools->addAction(actionSplit_Video);
        menuTools->addAction(actionCompute_gradient);

        retranslateUi(VideoEditingWindow);

        source_video_tab->setCurrentIndex(0);
        tabWidget_algorithom->setCurrentIndex(0);


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
