
///////////////////////////////////////////////////////////////////////////////
// cameraSimple.h
// ==============
// A simple camera model (192 triangles)
// Use "drawCamera()" to draw this model.
//
// 3D model is converted by the PolyTrans from Okino Computer Graphics, Inc.
// Bounding box of geometry = (-0.5,-0.35,-0.3) to (0.5,0.37,0.3).
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2008-09-18
// UPDATED: 2008-09-18
///////////////////////////////////////////////////////////////////////////////

#ifndef CAMERA_SIMPLE_H
#define CAMERA_SIMPLE_H
#include "basic_types.h"
#include <qvector3d.h>
///////////////////////////////////////////////////////////////////////////////
// cameraSimple.h
// ==============
// A simple camera model (192 triangles)
// Use "drawCamera()" to draw this model.
//
// 3D model is converted by the PolyTrans from Okino Computer Graphics, Inc.
// Bounding box of geometry = (-0.5,-0.35,-0.3) to (0.5,0.37,0.3).
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2008-09-18
// UPDATED: 2008-09-18
///////////////////////////////////////////////////////////////////////////////

void drawCamera();


///////////////////////////////////////////////////////////////////////////////
// draw frustum with 6 params (left, right, bottom, top, near, far)
///////////////////////////////////////////////////////////////////////////////
void drawFrustum(int projectionMode,float l, float r, float b, float t, float n, float f);


#endif
