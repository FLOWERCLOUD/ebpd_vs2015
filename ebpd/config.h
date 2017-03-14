///////////////////////////////////////////////////////////////////
//               3DViewer configuration file                 //
//  Modify these settings according to your local configuration  //
///////////////////////////////////////////////////////////////////

#ifndef TDVIEWER_CONFIG_H
#define TDVIEWER_CONFIG_H
#define TDVIEWER_VERSION 0x010001

// Get QT_VERSION and other Qt flags
#include <qglobal.h>

#if QT_VERSION < 0x040000
Error : 3DViewer requires a minimum Qt version of 4.0
#endif

# include <QGLWidget>
# include <GL/glu.h>
#include <QList>
#include <QVector>
#define M_PI 3.14159265358979323846

#endif // !3DVIWER_CONFIG_H
