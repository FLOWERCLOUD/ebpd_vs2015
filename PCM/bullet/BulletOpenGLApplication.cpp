#include "BulletOpenGLApplication.h"
#include "Bullet3Common/b3Vector3.h"
#include "qt_gui/paint_canvas.h"
#include "QGLViewer/camera.h"
#include "QGLViewer/manipulatedCameraFrame.h"
#include "GLInstancingRenderer.h"
#include "LinearMath/btQuickprof.h"
#include <iostream>
// Some constants for 3D math and the camera speed
#define RADIANS_PER_DEGREE 0.01745329f
#define CAMERA_STEP_SIZE 5.0f
static int pre_x = 0;
static int pre_y = 0;
static int mouse_type;
static int cur_key;
static btVector3 VecTobtVector3( const qglviewer::Vec& vec)
{
	return btVector3(vec.x ,vec.y , vec.z);
}
static qglviewer::Vec btVector3ToVec(const btVector3& vec )
{
	return qglviewer::Vec(vec.x() ,vec.y() , vec.z());
}

btBroadphaseInterface* BulletOpenGLApplication::s_pBroadphase(0);
btCollisionConfiguration* BulletOpenGLApplication::s_pCollisionConfiguration(0);
btCollisionDispatcher* BulletOpenGLApplication::s_pDispatcher(0);
btConstraintSolver* BulletOpenGLApplication::s_pSolver(0);
btDynamicsWorld* BulletOpenGLApplication::s_pWorld(0);
btRigidBody* BulletOpenGLApplication::s_pPickedBody(0);
btTypedConstraint* BulletOpenGLApplication::s_pPickConstraint(0);
btClock BulletOpenGLApplication::s_clock;
btRigidBody*	BulletOpenGLApplication::s_pickedBody;
btTypedConstraint* BulletOpenGLApplication::s_pickedConstraint(0);
btVector3 BulletOpenGLApplication::m_oldPickingPos;
int	BulletOpenGLApplication::m_savedState;
btVector3 BulletOpenGLApplication::m_hitPos;

BulletOpenGLApplication::BulletOpenGLApplication(PaintCanvas* qgl) 
:
m_cameraPosition(10.0f, 5.0f, 0.0f),
m_cameraTarget(0.0f, 0.0f, 0.0f),
m_cameraDistance(15.0f),
m_cameraPitch(20.0f),
m_cameraYaw(0.0f),
m_upVector(0.0f, 1.0f, 0.0f),
m_nearPlane(1.0f),
m_farPlane(1000.0f),
//s_pBroadphase(0),
//s_pCollisionConfiguration(0),
//s_pDispatcher(0),
//s_pSolver(0),
//s_pWorld(0),
//s_pPickedBody(0),
//s_pPickConstraint(0),
m_qgl(qgl),
m_renderer(qgl->m_instancingRenderer)
//s_pickedBody(0),
//s_pickedConstraint(0)
{
	std::cout<<"BulletOpenGLApplication";

}

BulletOpenGLApplication::~BulletOpenGLApplication() {
	// shutdown the physics system
	ShutdownPhysics();
}

void BulletOpenGLApplication::Initialize() {
	// this function is called inside glutmain() after
	// creating the window, but before handing control
	// to FreeGLUT

	// create some floats for our ambient, diffuse, specular and position
	GLfloat ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f }; // dark grey
	GLfloat diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f }; // white
	GLfloat specular[] = { 1.0f, 1.0f, 1.0f, 1.0f }; // white
	GLfloat position[] = { 5.0f, 10.0f, 1.0f, 0.0f };
	
	// set the ambient, diffuse, specular and position for LIGHT0
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
	glLightfv(GL_LIGHT0, GL_POSITION, position);

	glEnable(GL_LIGHTING); // enables lighting
	glEnable(GL_LIGHT0); // enables the 0th light
	glEnable(GL_COLOR_MATERIAL); // colors materials when lighting is enabled
		
	// enable specular lighting via materials
	glMaterialfv(GL_FRONT, GL_SPECULAR, specular);
	glMateriali(GL_FRONT, GL_SHININESS, 15);
	
	// enable smooth shading
	glShadeModel(GL_SMOOTH);
	
	// enable depth testing to be 'less than'
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	// set the backbuffer clearing color to a lightish blue
	glClearColor(0.6, 0.65, 0.85, 0);

	// initialize the physics system
	InitializePhysics();

	// create the debug drawer
	m_pDebugDrawer = new DebugDrawer();
	// set the initial debug level to 0
	m_pDebugDrawer->setDebugMode(0);
	// add the debug drawer to the world
	s_pWorld->setDebugDrawer(m_pDebugDrawer);
}


void BulletOpenGLApplication::InitializeNaive()
{
 	m_renderer->init();
	b3Assert(glGetError() ==GL_NO_ERROR);
 	m_renderer->InitShaders();
	// initialize the physics system
	InitializePhysics();

	// create the debug drawer
	m_pDebugDrawer = new DebugDrawer();
	// set the initial debug level to 0
	m_pDebugDrawer->setDebugMode(
		btIDebugDraw::DBG_DrawWireframe|
		//btIDebugDraw::DBG_DrawAabb|
		//btIDebugDraw::DBG_DrawFeaturesText
		//|
//		btIDebugDraw::DBG_DrawContactPoints|
		btIDebugDraw::DBG_DrawConstraints
		//|
//		btIDebugDraw::DBG_DrawNormals
		);
	// add the debug drawer to the world
	s_pWorld->setDebugDrawer(m_pDebugDrawer);
}
void BulletOpenGLApplication::Keyboard(unsigned char key, int x, int y) {
	// This function is called by FreeGLUT whenever
	// generic keys are pressed down.
	switch(key) {
		// 'z' zooms in
	case 'z': ZoomCamera(+CAMERA_STEP_SIZE); break;
		// 'x' zoom out
	case 'x': ZoomCamera(-CAMERA_STEP_SIZE); break;
	case 'w':
		// toggle wireframe debug drawing
		m_pDebugDrawer->ToggleDebugFlag(btIDebugDraw::DBG_DrawWireframe);
		break;

	case 'b':
		// toggle AABB debug drawing
		m_pDebugDrawer->ToggleDebugFlag(btIDebugDraw::DBG_DrawAabb);
		break;
 	case 'd':
 		{
 			// create a temp object to store the raycast result
 			RayResult result;
 			// perform the raycast
 			if (!Raycast(m_cameraPosition, GetPickingRay(x, y), result))
 				return; // return if the test failed
 			// destroy the corresponding game object
 			DestroyGameObject(result.pBody);
 			break;
 		}
	}
}

void BulletOpenGLApplication::KeyboardUp(unsigned char key, int x, int y) 
{
	cur_key = -1;

}

void BulletOpenGLApplication::Special(int key, int x, int y) {
	// This function is called by FreeGLUT whenever special keys
	// are pressed down, like the arrow keys, or Insert, Delete etc.
	switch(key) {
		// the arrow keys rotate the camera up/down/left/right
	case 0x0064: //GLUT_KEY_LEFT: 
		RotateCamera(m_cameraYaw, +CAMERA_STEP_SIZE); break;
	case 0x0066: //GLUT_KEY_RIGHT:
		RotateCamera(m_cameraYaw, -CAMERA_STEP_SIZE); break;
	case 0x0065: //GLUT_KEY_UP:	
		RotateCamera(m_cameraPitch, +CAMERA_STEP_SIZE); break;
	case 0x0067: //GLUT_KEY_DOWN:	
		RotateCamera(m_cameraPitch, -CAMERA_STEP_SIZE); break;
	case 0x0004: //alt
		cur_key = 0x0004;

	}
}

void BulletOpenGLApplication::SpecialUp(int key, int x, int y) {

	cur_key = -1;
}

void BulletOpenGLApplication::Reshape(int w, int h) {
	// this function is called once during application intialization
	// and again every time we resize the window

	// grab the screen width/height
	m_screenWidth = w;
	m_screenHeight = h;
	// set the viewport
	glViewport(0, 0, w, h);
	// update the camera
	UpdateCamera();
	m_renderer->resize( w ,h);
}

void BulletOpenGLApplication::Idle() {



	m_renderer->init();
	float projM[16];
	float viewM[16];
	m_qgl->camera()->getProjectionMatrix(projM);
	m_qgl->camera()->getModelViewMatrix(viewM);
	m_renderer->updateCamera( projM ,viewM);
	// this function is called frequently, whenever FreeGlut
	// isn't busy processing its own events. It should be used
	// to perform any updating and rendering tasks

	// clear the backbuffer
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	// get the time since the last iteration
	float dt = s_clock.getTimeMilliseconds();
	// reset the clock to 0
	s_clock.reset();
	// update the scene (convert ms to s)
	UpdateScene(dt / 1000.0f);


	// update the camera
//	UpdateCamera();
	m_cameraPosition[0 ] = m_qgl->camera()->position().x;
	m_cameraPosition[1 ] = m_qgl->camera()->position().y;
	m_cameraPosition[2 ] = m_qgl->camera()->position().z;

	// render the scene
	renderSceneNew();
//	RenderScene();
	//std::cout<<" Idle()"<<std::endl;
	// swap the front and back buffers
	//glutSwapBuffers();
}

void BulletOpenGLApplication::Mouse(int button, int state, int x, int y) {

	if(cur_key == 0x0004) //OMIT ALT
		return;
	pre_x = x;
	pre_y = y;
 	switch(button) {
	case 0:  // left mouse button
		{
			mouse_type = 0;
			if (state == 0) { // button down
				// create the picking constraint when we click the LMB
				CreatePickingConstraint(x, y);

				qglviewer::Vec camPos = m_qgl->camera()->position();
				qglviewer::Vec rayFrom = camPos;
				bool isFound = false;
				qglviewer::Vec rayTo = m_qgl->camera()->pointUnderPixel( QPoint(x,y) ,isFound);
				pickBody( VecTobtVector3(rayFrom) , VecTobtVector3(rayTo));

			} else { // button up
				// remove the picking constraint when we release the LMB
				RemovePickingConstraint();
			}
			break;
		}

	case 1: // middle mouse button
		{
			mouse_type = 1;
			break;
		}

	case 2: // right mouse button
 		{
			mouse_type = 2;
 			if (state == 0) { // pressed down
 				// shoot a box
 				ShootBox(GetPickingRay(x, y));
 			}
 		
 		break;
 		}
 	}
}

void BulletOpenGLApplication::PassiveMotion(int x, int y) {}

void BulletOpenGLApplication::Motion(int x, int y) {

	if(cur_key == 0x0004) //OMIT ALT
		return;

	pre_x = x;
	pre_y = y;
	bool isfound;
	m_qgl->camera()->pointUnderPixel( QPoint(x,y),isfound);

	qglviewer::Vec rayTo = m_qgl->camera()->pointUnderPixel( QPoint(x,y),isfound);
	qglviewer::Vec rayFrom = m_qgl->camera()->position() ;

	movePickedBody( VecTobtVector3(rayFrom), VecTobtVector3(rayTo));

	if(m_renderer)
	{

		CommonCameraInterface* camera = m_renderer->getActiveCamera();
		if(0){
			float xDelta = x-pre_x;
			float yDelta = y-pre_y;
			//float cameraDistance = camera->getCameraDistance();
			float cameraDistance = m_qgl->camera()->frame()->position().norm();
			float pitch = camera->getCameraPitch();
			float yaw = camera->getCameraYaw();

			float targPos[3];
			float	camPos[3];

			camera->getCameraTargetPosition(targPos);
			camera->getCameraPosition(camPos);

			b3Vector3 cameraPosition = b3MakeVector3(b3Scalar(camPos[0]),
				b3Scalar(camPos[1]),
				b3Scalar(camPos[2]));

			b3Vector3 cameraTargetPosition = b3MakeVector3(	b3Scalar(targPos[0]),
				b3Scalar(targPos[1]),
				b3Scalar(targPos[2]));
			b3Vector3 cameraUp = b3MakeVector3(0,0,0);
			cameraUp[camera->getCameraUpAxis()] = 1.f;
			m_qgl->camera();
			if ( 0 == mouse_type ) //m_leftMouseButton
			{
				//			if (b3Fabs(xDelta)>b3Fabs(yDelta))
				//			{
				pitch -= xDelta*  m_qgl->camera()->frame()->rotationSensitivity();
				//			} else
				//			{
				yaw += yDelta*m_qgl->camera()->frame()->rotationSensitivity();
				//			}
			}

			if (0) //m_middleMouseButton
			{
				cameraTargetPosition += cameraUp * yDelta*0.01;


				b3Vector3 fwd = cameraTargetPosition-cameraPosition;
				b3Vector3 side = cameraUp.cross(fwd);
				side.normalize();
				cameraTargetPosition += side * xDelta*0.01;

			}
			if (0) //m_rightMouseButton
			{
				cameraDistance -= xDelta*0.01f;
				cameraDistance -= yDelta*0.01f;
				if (cameraDistance<1)
					cameraDistance=1;
				if (cameraDistance>1000)
					cameraDistance=1000;
			}
			camera->setCameraDistance(cameraDistance);
			camera->setCameraPitch(pitch);
			camera->setCameraYaw(yaw);
			camera->setCameraTargetPosition(cameraTargetPosition[0],cameraTargetPosition[1],cameraTargetPosition[2]);






		}

		float projM[16];
		float viewM[16];
		m_qgl->camera()->getProjectionMatrix(projM);
		m_qgl->camera()->getModelViewMatrix(viewM);
		m_renderer->updateCamera( projM ,viewM);


	}
	



	// did we pick a body with the LMB?
	if (s_pPickedBody) {
		btGeneric6DofConstraint* pickCon = static_cast<btGeneric6DofConstraint*>(s_pPickConstraint);
		if (!pickCon)
			return;

		// use another picking ray to get the target direction
		btVector3 dir = GetPickingRay(x,y) - m_cameraPosition;
		dir.normalize();

		// use the same distance as when we originally picked the object
		dir *= m_oldPickingDist;
		btVector3 newPivot = m_cameraPosition + dir;

		// set the position of the constraint
		pickCon->getFrameOffsetA().setOrigin(newPivot);
	}
}
void BulletOpenGLApplication::Wheel( float deltax, float deltay)
{
	static const qreal WHEEL_SENSITIVITY_COEF = 8E-4;
	float test_delta = deltay* m_qgl->camera()->frame()->wheelSensitivity()*WHEEL_SENSITIVITY_COEF;
	deltay = deltay/120.0*6;
	if (m_renderer)
	{
		b3Vector3 cameraTargetPosition, cameraPosition, cameraUp = b3MakeVector3(0,0,0);
		//cameraUp[getUpAxis()] = 1;
		cameraUp[1] = 1;
		CommonCameraInterface* camera = m_renderer->getActiveCamera();

		camera->getCameraPosition(cameraPosition);
		camera->getCameraTargetPosition(cameraTargetPosition);

		if (1)
		{

			float cameraDistance = 	camera->getCameraDistance();
			if (deltay<0 || cameraDistance>1)
			{
				cameraDistance -= deltay*0.01f;
				if (cameraDistance<1)
					cameraDistance=1;
				camera->setCameraDistance(cameraDistance);

			} else
			{

				b3Vector3 fwd = cameraTargetPosition-cameraPosition;
				fwd.normalize();
				//cameraTargetPosition += fwd*deltay*m_wheelMultiplier;//todo: expose it in the GUI?
				cameraTargetPosition += fwd*deltay*m_qgl->camera()->frame()->wheelSensitivity();
			}
		} else
		{
			if (b3Fabs(deltax)>b3Fabs(deltay))
			{
				b3Vector3 fwd = cameraTargetPosition-cameraPosition;
				b3Vector3 side = cameraUp.cross(fwd);
				side.normalize();
				cameraTargetPosition += side * deltax*m_qgl->camera()->frame()->wheelSensitivity();

			} else
			{
				//cameraTargetPosition -= cameraUp * deltay*m_wheelMultiplier;
				cameraTargetPosition -= cameraUp * deltay*m_qgl->camera()->frame()->wheelSensitivity();

			}
		}

		camera->setCameraTargetPosition(cameraTargetPosition[0],cameraTargetPosition[1],cameraTargetPosition[2]);
	}
}
void BulletOpenGLApplication::Display() {}

void BulletOpenGLApplication::UpdateCamera() {
	// exit in erroneous situations
	if (m_screenWidth == 0 && m_screenHeight == 0)
		return;
	
	// select the projection matrix
	glMatrixMode(GL_PROJECTION);
	// set it to the matrix-equivalent of 1
	glLoadIdentity();
	// determine the aspect ratio of the screen
	float aspectRatio = m_screenWidth / (float)m_screenHeight;
	// create a viewing frustum based on the aspect ratio and the
	// boundaries of the camera
	glFrustum (-aspectRatio * m_nearPlane, aspectRatio * m_nearPlane, -m_nearPlane, m_nearPlane, m_nearPlane, m_farPlane);
	// the projection matrix is now set

	// select the view matrix
	glMatrixMode(GL_MODELVIEW);
	// set it to '1'
	glLoadIdentity();

	// our values represent the angles in degrees, but 3D 
	// math typically demands angular values are in radians.
	float pitch = m_cameraPitch * RADIANS_PER_DEGREE;
	float yaw = m_cameraYaw * RADIANS_PER_DEGREE;

	// create a quaternion defining the angular rotation 
	// around the up vector
	btQuaternion rotation(m_upVector, yaw);

	// set the camera's position to 0,0,0, then move the 'z' 
	// position to the current value of m_cameraDistance.
	btVector3 cameraPosition(0,0,0);
	cameraPosition[2] = -m_cameraDistance;

	// create a Bullet Vector3 to represent the camera 
	// position and scale it up if its value is too small.
	btVector3 forward(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
	if (forward.length2() < SIMD_EPSILON) {
		forward.setValue(1.f,0.f,0.f);
	}

	// figure out the 'right' vector by using the cross 
	// product on the 'forward' and 'up' vectors
	btVector3 right = m_upVector.cross(forward);

	// create a quaternion that represents the camera's roll
	btQuaternion roll(right, - pitch);

	// turn the rotation (around the Y-axis) and roll (around 
	// the forward axis) into transformation matrices and 
	// apply them to the camera position. This gives us the 
	// final position
	cameraPosition = btMatrix3x3(rotation) * btMatrix3x3(roll) * cameraPosition;

	// save our new position in the member variable, and 
	// shift it relative to the target position (so that we 
	// orbit it)
	m_cameraPosition[0] = cameraPosition.getX();
	m_cameraPosition[1] = cameraPosition.getY();
	m_cameraPosition[2] = cameraPosition.getZ();
	m_cameraPosition += m_cameraTarget;

	// create a view matrix based on the camera's position and where it's
	// looking
//	gluLookAt(m_cameraPosition[0], m_cameraPosition[1], m_cameraPosition[2], m_cameraTarget[0], m_cameraTarget[1], m_cameraTarget[2], m_upVector.getX(), m_upVector.getY(), m_upVector.getZ());
	// the view matrix is now set
}

void BulletOpenGLApplication::DrawBox(const btVector3 &halfSize) {
	
	float halfWidth = halfSize.x();
	float halfHeight = halfSize.y();
	float halfDepth = halfSize.z();

	// create the vertex positions
	btVector3 vertices[8]={	
	btVector3(halfWidth,halfHeight,halfDepth),
	btVector3(-halfWidth,halfHeight,halfDepth),
	btVector3(halfWidth,-halfHeight,halfDepth),	
	btVector3(-halfWidth,-halfHeight,halfDepth),	
	btVector3(halfWidth,halfHeight,-halfDepth),
	btVector3(-halfWidth,halfHeight,-halfDepth),	
	btVector3(halfWidth,-halfHeight,-halfDepth),	
	btVector3(-halfWidth,-halfHeight,-halfDepth)};

	// create the indexes for each triangle, using the 
	// vertices above. Make it static so we don't waste 
	// processing time recreating it over and over again
	static int indices[36] = {
		0,1,2,
		3,2,1,
		4,0,6,
		6,0,2,
		5,1,4,
		4,1,0,
		7,3,1,
		7,1,5,
		5,4,7,
		7,4,6,
		7,2,3,
		7,6,2};

	// start processing vertices as triangles
	glBegin (GL_TRIANGLES);

	// increment the loop by 3 each time since we create a 
	// triangle with 3 vertices at a time.

	for (int i = 0; i < 36; i += 3) {
		// get the three vertices for the triangle based
		// on the index values set above
		// use const references so we don't copy the object
		// (a good rule of thumb is to never allocate/deallocate
		// memory during *every* render/update call. This should 
		// only happen sporadically)
		const btVector3 &vert1 = vertices[indices[i]];
		const btVector3 &vert2 = vertices[indices[i+1]];
		const btVector3 &vert3 = vertices[indices[i+2]];

		// create a normal that is perpendicular to the 
		// face (use the cross product)
		btVector3 normal = (vert3-vert1).cross(vert2-vert1);
		normal.normalize ();

		// set the normal for the subsequent vertices
		glNormal3f(normal.getX(),normal.getY(),normal.getZ());

		// create the vertices
		glVertex3f (vert1.x(), vert1.y(), vert1.z());
		glVertex3f (vert2.x(), vert2.y(), vert2.z());
		glVertex3f (vert3.x(), vert3.y(), vert3.z());
	}

	// stop processing vertices
	glEnd();
}

void BulletOpenGLApplication::RotateCamera(float &angle, float value) {
	// change the value (it is passed by reference, so we
	// can edit it here)
	angle -= value; 
	// keep the value within bounds
	if (angle < 0) angle += 360; 
	if (angle >= 360) angle -= 360;
	// update the camera since we changed the angular value
	UpdateCamera(); 
}

void BulletOpenGLApplication::ZoomCamera(float distance) {
	// change the distance value
	m_cameraDistance -= distance;
	// prevent it from zooming in too far
	if (m_cameraDistance < 0.1f) m_cameraDistance = 0.1f;
	// update the camera since we changed the zoom distance
	UpdateCamera();
}

void BulletOpenGLApplication::RenderScene() {

	glPushAttrib(GL_ALL_ATTRIB_BITS);
	// create some floats for our ambient, diffuse, specular and position
	GLfloat ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f }; // dark grey
	GLfloat diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f }; // white
	GLfloat specular[] = { 1.0f, 1.0f, 1.0f, 1.0f }; // white
	GLfloat position[] = { 5.0f, 10.0f, 1.0f, 0.0f };

	// set the ambient, diffuse, specular and position for LIGHT0
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
	glLightfv(GL_LIGHT0, GL_POSITION, position);

	glEnable(GL_LIGHTING); // enables lighting
	glEnable(GL_LIGHT0); // enables the 0th light
	glEnable(GL_COLOR_MATERIAL); // colors materials when lighting is enabled

	// enable specular lighting via materials
	glMaterialfv(GL_FRONT, GL_SPECULAR, specular);
	glMateriali(GL_FRONT, GL_SHININESS, 15);

	// enable smooth shading
	glShadeModel(GL_SMOOTH);

	// enable depth testing to be 'less than'
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	// set the backbuffer clearing color to a lightish blue
	glClearColor(0.6, 0.65, 0.85, 0);


	// create an array of 16 floats (representing a 4x4 matrix)
	btScalar transform[16];

	// iterate through all of the objects in our world
	for(GameObjects::iterator i = m_objects.begin(); i != m_objects.end(); ++i) {
		// get the object from the iterator
		GameObject* pObj = *i;

		// read the transform
		pObj->GetTransform(transform);

		// get data from the object and draw it
		DrawShape(transform, pObj->GetShape(), pObj->GetColor());
	}

	// after rendering all game objects, perform debug rendering
	// Bullet will figure out what needs to be drawn then call to
	// our DebugDrawer class to do the rendering for us
	s_pWorld->debugDrawWorld();

	glPopAttrib();
}

void BulletOpenGLApplication::UpdateScene(float dt) {
	// check if the world object exists
	if (s_pWorld) {
		// step the simulation through time. This is called
		// every update and the amount of elasped time was 
		// determined back in ::Idle() by our clock object.

		btCollisionObjectArray& objects = s_pWorld->getCollisionObjectArray();
		for (int i = 0 ;i < objects.size(); ++i)
		{
			btRigidBody *rigidBody = btRigidBody::upcast(objects[i]);
			if (!rigidBody) {
				continue;
			}
			rigidBody->applyGravity();
		}



		s_pWorld->stepSimulation(dt);

		// check for any new collisions/separations
// 		CheckForCollisionEvents();
	}
	//CProfileManager::dumpAll();
}

void BulletOpenGLApplication::DrawShape(btScalar* transform, const btCollisionShape* pShape, const btVector3 &color) {
	// set the color
	glColor3f(color.x(), color.y(), color.z());

	// push the matrix stack
	glPushMatrix();
	glMultMatrixf(transform);

	// make a different draw call based on the object type
	switch(pShape->getShapeType()) {
		// an internal enum used by Bullet for boxes
	case BOX_SHAPE_PROXYTYPE:
		{
			// assume the shape is a box, and typecast it
			const btBoxShape* box = static_cast<const btBoxShape*>(pShape);
			// get the 'halfSize' of the box
			btVector3 halfSize = box->getHalfExtentsWithMargin();
			// draw the box
			DrawBox(halfSize);
			break;
		}
	default:
		// unsupported type
		break;
	}

	// pop the stack
	glPopMatrix();
}

GameObject* BulletOpenGLApplication::CreateGameObject(btCollisionShape* pShape, const float &mass, const btVector3 &color, const btVector3 &initialPosition, const btQuaternion &initialRotation) {
	// create a new game object
	GameObject* pObject = new GameObject(pShape, mass, color, initialPosition, initialRotation);

	// push it to the back of the list
	m_objects.push_back(pObject);

	// check if the world object is valid
	if (s_pWorld) {
		// add the object's rigid body to the world
		s_pWorld->addRigidBody(pObject->GetRigidBody());
	}
	return pObject;
}

btVector3 BulletOpenGLApplication::GetPickingRay(int x, int y) {
 	// calculate the field-of-view
 	float tanFov = 1.0f / m_nearPlane;
 	float fov = btScalar(2.0) * btAtan(tanFov);
 	
 	// get a ray pointing forward from the 
 	// camera and extend it to the far plane
 	btVector3 rayFrom = m_cameraPosition;
 	btVector3 rayForward = (m_cameraTarget - m_cameraPosition);
 	rayForward.normalize();
 	rayForward*= m_farPlane;
 	
 	// find the horizontal and vertical vectors 
 	// relative to the current camera view
 	btVector3 ver = m_upVector;
 	btVector3 hor = rayForward.cross(ver);
 	hor.normalize();
 	ver = hor.cross(rayForward);
 	ver.normalize();
 	hor *= 2.f * m_farPlane * tanFov;
 	ver *= 2.f * m_farPlane * tanFov;
 	
 	// calculate the aspect ratio
 	btScalar aspect = m_screenWidth / (btScalar)m_screenHeight;
 	
 	// adjust the forward-ray based on
 	// the X/Y coordinates that were clicked
 	hor*=aspect;
 	btVector3 rayToCenter = rayFrom + rayForward;
 	btVector3 dHor = hor * 1.f/float(m_screenWidth);
 	btVector3 dVert = ver * 1.f/float(m_screenHeight);
 	btVector3 rayTo = rayToCenter - 0.5f * hor + 0.5f * ver;
 	rayTo += btScalar(x) * dHor;
 	rayTo -= btScalar(y) * dVert;
 	
 	// return the final result
 	return rayTo;
}


bool BulletOpenGLApplication::pickBody(const btVector3& rayFromWorld , const btVector3& rayToWorld)
{
	if (s_pWorld==0)
		return false;

	btCollisionWorld::ClosestRayResultCallback rayCallback(rayFromWorld, rayToWorld);

	s_pWorld->rayTest(rayFromWorld, rayToWorld, rayCallback);
	if (rayCallback.hasHit())
	{

		btVector3 pickPos = rayCallback.m_hitPointWorld;
		btRigidBody* body = (btRigidBody*)btRigidBody::upcast(rayCallback.m_collisionObject);
		if (body)
		{



			//other exclusions?
			if (!(body->isStaticObject() || body->isKinematicObject()))
			{
				s_pickedBody = body;
				m_savedState = s_pickedBody->getActivationState();
				s_pickedBody->setActivationState(DISABLE_DEACTIVATION);
				//printf("pickPos=%f,%f,%f\n",pickPos.getX(),pickPos.getY(),pickPos.getZ());
				btVector3 localPivot = body->getCenterOfMassTransform().inverse() * pickPos;
				btPoint2PointConstraint* p2p = new btPoint2PointConstraint(*body, localPivot);
				s_pWorld->addConstraint(p2p, true);
				s_pickedConstraint = p2p;
				btScalar mousePickClamping = 30.f;
				p2p->m_setting.m_impulseClamp = mousePickClamping;
				//very weak constraint for picking
				p2p->m_setting.m_tau = 0.001f;
			}
		}


		//					pickObject(pickPos, rayCallback.m_collisionObject);
		m_oldPickingPos = rayToWorld;
		m_hitPos = pickPos;
		m_oldPickingDist = (pickPos - rayFromWorld).length();
		//					printf("hit !\n");
		//add p2p
	}
	return false;

}

bool BulletOpenGLApplication::movePickedBody(const btVector3& rayFromWorld, const btVector3& rayToWorld)
{
	if (s_pickedBody  && s_pickedConstraint)
	{
		btPoint2PointConstraint* pickCon = static_cast<btPoint2PointConstraint*>(s_pickedConstraint);
		if (pickCon)
		{
			//keep it at the same picking distance

			btVector3 newPivotB;

			btVector3 dir = rayToWorld - rayFromWorld;
			dir.normalize();
			dir *= m_oldPickingDist;

			newPivotB = rayFromWorld + dir;
			pickCon->setPivotB(newPivotB);
			return true;
		}
	}
	return false;



} 	
void BulletOpenGLApplication::ShootBox(const btVector3 &direction) {
 	// create a new box object
 	GameObject* pObject = CreateGameObject(new btBoxShape(btVector3(1, 1, 1)), 1, btVector3(0.4f, 0.f, 0.4f), m_cameraPosition);
 		
 	// calculate the velocity
 	btVector3 velocity = direction; 
 	velocity.normalize();
 	velocity *= 25.0f;
 		
 	// set the linear velocity of the box
 	pObject->GetRigidBody()->setLinearVelocity(velocity);
}
 	
/*REM**	bool BulletOpenGLApplication::Raycast(const btVector3 &startPosition, const btVector3 &direction, RayResult &output) { **/
  	bool BulletOpenGLApplication::Raycast(const btVector3 &startPosition, const btVector3 &direction, RayResult &output, bool includeStatic) {
 	if (!s_pWorld) 
 		return false;
 		
 	// get the picking ray from where we clicked
 	btVector3 rayTo = direction;
 	btVector3 rayFrom = m_cameraPosition;
 		
 	// create our raycast callback object
 	btCollisionWorld::ClosestRayResultCallback rayCallback(rayFrom,rayTo);
 		
 	// perform the raycast
 	s_pWorld->rayTest(rayFrom,rayTo,rayCallback);
 		
 	// did we hit something?
 	if (rayCallback.hasHit())
 	{
 		// if so, get the rigid body we hit
 		btRigidBody* pBody = (btRigidBody*)btRigidBody::upcast(rayCallback.m_collisionObject);
 		if (!pBody)
 			return false;
 		
 		// prevent us from picking objects 
 		// like the ground plane
  		if (!includeStatic) // skip this check if we want it to hit static objects
				if (pBody->isStaticObject() || pBody->isKinematicObject()) 
					return false;
 	    
 		// set the result data
 		output.pBody = pBody;
 		output.hitPoint = rayCallback.m_hitPointWorld;
 		return true;
 	}
 	
 	// we didn't hit anything
 	return false;
}
 	
void BulletOpenGLApplication::DestroyGameObject(btRigidBody* pBody) {
 	// we need to search through the objects in order to 
 	// find the corresponding iterator (can only erase from 
 	// an std::vector by passing an iterator)
 	for (GameObjects::iterator iter = m_objects.begin(); iter != m_objects.end(); ++iter) {
 		if ((*iter)->GetRigidBody() == pBody) {
 			GameObject* pObject = *iter;
 			// remove the rigid body from the world
 			s_pWorld->removeRigidBody(pObject->GetRigidBody());
 			// erase the object from the list
 			m_objects.erase(iter);
 			// delete the object from memory
 			delete pObject;
 			// done
 			return;
 		}
 	}
}

void BulletOpenGLApplication::CreatePickingConstraint(int x, int y) {
	if (!s_pWorld) 
		return;

	// perform a raycast and return if it fails
	RayResult output;
	if (!Raycast(m_cameraPosition, GetPickingRay(x, y), output))
		return;

	// store the body for future reference
	s_pPickedBody = output.pBody;

	// prevent the picked object from falling asleep
	s_pPickedBody->setActivationState(DISABLE_DEACTIVATION);

	// get the hit position relative to the body we hit 
	btVector3 localPivot = s_pPickedBody->getCenterOfMassTransform().inverse() * output.hitPoint;

	// create a transform for the pivot point
	btTransform pivot;
	pivot.setIdentity();
	pivot.setOrigin(localPivot);

	// create our constraint object
	btGeneric6DofConstraint* dof6 = new btGeneric6DofConstraint(*s_pPickedBody, pivot, true);
	bool bLimitAngularMotion = true;
	if (bLimitAngularMotion) {
		dof6->setAngularLowerLimit(btVector3(0,0,0));
		dof6->setAngularUpperLimit(btVector3(0,0,0));
	}

	// add the constraint to the world
	s_pWorld->addConstraint(dof6,true);

	// store a pointer to our constraint
	s_pPickConstraint = dof6;

	// define the 'strength' of our constraint (each axis)
	float cfm = 0.5f;
	dof6->setParam(BT_CONSTRAINT_STOP_CFM,cfm,0);
	dof6->setParam(BT_CONSTRAINT_STOP_CFM,cfm,1);
	dof6->setParam(BT_CONSTRAINT_STOP_CFM,cfm,2);
	dof6->setParam(BT_CONSTRAINT_STOP_CFM,cfm,3);
	dof6->setParam(BT_CONSTRAINT_STOP_CFM,cfm,4);
	dof6->setParam(BT_CONSTRAINT_STOP_CFM,cfm,5);

	// define the 'error reduction' of our constraint (each axis)
	float erp = 0.5f;
	dof6->setParam(BT_CONSTRAINT_STOP_ERP,erp,0);
	dof6->setParam(BT_CONSTRAINT_STOP_ERP,erp,1);
	dof6->setParam(BT_CONSTRAINT_STOP_ERP,erp,2);
	dof6->setParam(BT_CONSTRAINT_STOP_ERP,erp,3);
	dof6->setParam(BT_CONSTRAINT_STOP_ERP,erp,4);
	dof6->setParam(BT_CONSTRAINT_STOP_ERP,erp,5);

	// save this data for future reference
	m_oldPickingDist  = (output.hitPoint - m_cameraPosition).length();
}

void BulletOpenGLApplication::RemovePickingConstraint() {

	if (s_pickedConstraint)
	{
		s_pickedBody->forceActivationState(m_savedState);
		s_pickedBody->activate();
		s_pWorld->removeConstraint(s_pickedConstraint);
		delete s_pickedConstraint;
		s_pickedConstraint = 0;
		s_pickedBody = 0;
	}

	// exit in erroneous situations
	if (!s_pPickConstraint || !s_pWorld) 
		return;

	// remove the constraint from the world
	s_pWorld->removeConstraint(s_pPickConstraint);

	// delete the constraint object
	delete s_pPickConstraint;

	// reactivate the body
	s_pPickedBody->forceActivationState(ACTIVE_TAG);
	s_pPickedBody->setDeactivationTime( 0.f );

	// clear the pointers
	s_pPickConstraint = 0;
	s_pPickedBody = 0;
}

void BulletOpenGLApplication::CheckForCollisionEvents() {
	// keep a list of the collision pairs we
	// found during the current update
	CollisionPairs pairsThisUpdate;

	// iterate through all of the manifolds in the dispatcher
	for (int i = 0; i < s_pDispatcher->getNumManifolds(); ++i) {
		
		// get the manifold
		btPersistentManifold* pManifold = s_pDispatcher->getManifoldByIndexInternal(i);
		
		// ignore manifolds that have 
		// no contact points.
		if (pManifold->getNumContacts() > 0) {
			// get the two rigid bodies involved in the collision
			const btRigidBody* pBody0 = static_cast<const btRigidBody*>(pManifold->getBody0());
			const btRigidBody* pBody1 = static_cast<const btRigidBody*>(pManifold->getBody1());
    
			// always create the pair in a predictable order
			// (use the pointer value..)
			bool const swapped = pBody0 > pBody1;
			const btRigidBody* pSortedBodyA = swapped ? pBody1 : pBody0;
			const btRigidBody* pSortedBodyB = swapped ? pBody0 : pBody1;
			
			// create the pair
			CollisionPair thisPair = std::make_pair(pSortedBodyA, pSortedBodyB);
			
			// insert the pair into the current list
			pairsThisUpdate.insert(thisPair);

			// if this pair doesn't exist in the list
			// from the previous update, it is a new
			// pair and we must send a collision event
			if (m_pairsLastUpdate.find(thisPair) == m_pairsLastUpdate.end()) {
				CollisionEvent((btRigidBody*)pBody0, (btRigidBody*)pBody1);
			}
		}
	}
	
	// create another list for pairs that
	// were removed this update
	CollisionPairs removedPairs;
	
	// this handy function gets the difference beween
	// two sets. It takes the difference between
	// collision pairs from the last update, and this 
	// update and pushes them into the removed pairs list
	std::set_difference( m_pairsLastUpdate.begin(), m_pairsLastUpdate.end(),
	pairsThisUpdate.begin(), pairsThisUpdate.end(),
	std::inserter(removedPairs, removedPairs.begin()));
	
	// iterate through all of the removed pairs
	// sending separation events for them
	for (CollisionPairs::const_iterator iter = removedPairs.begin(); iter != removedPairs.end(); ++iter) {
		SeparationEvent((btRigidBody*)iter->first, (btRigidBody*)iter->second);
	}
	
	// in the next iteration we'll want to
	// compare against the pairs we found
	// in this iteration
	m_pairsLastUpdate = pairsThisUpdate;
}

void BulletOpenGLApplication::CollisionEvent(btRigidBody * pBody0, btRigidBody * pBody1) {

}

void BulletOpenGLApplication::SeparationEvent(btRigidBody * pBody0, btRigidBody * pBody1) {

}

GameObject* BulletOpenGLApplication::FindGameObject(btRigidBody* pBody) {
	// search through our list of gameobjects finding
	// the one with a rigid body that matches the given one
	for (GameObjects::iterator iter = m_objects.begin(); iter != m_objects.end(); ++iter) {
		if ((*iter)->GetRigidBody() == pBody) {
			// found the body, so return the corresponding game object
			return *iter;
		}
	}
	return 0;
}

void BulletOpenGLApplication::syncPhysicsToGraphics()
{
	int numCollisionObjects = s_pWorld->getNumCollisionObjects();
	{
		B3_PROFILE("write all InstanceTransformToCPU");
		for (int i = 0; i<numCollisionObjects; i++)
		{
			B3_PROFILE("writeSingleInstanceTransformToCPU");
			btCollisionObject* colObj = s_pWorld->getCollisionObjectArray()[i];
			btVector3 pos = colObj->getWorldTransform().getOrigin();
			btQuaternion orn = colObj->getWorldTransform().getRotation();
			int index = colObj->getUserIndex();
			if (index >= 0)
			{
				m_renderer->writeSingleInstanceTransformToCPU(pos, orn, index);
				//float c[4] = {1.0f ,0.0f, 0.0f ,0.3f};
				//m_renderer->writeSingleInstanceColorToCPU(c, index);
			}
		}
	}
	{
		B3_PROFILE("writeTransforms");
		m_renderer->writeTransforms();
	}
}

void BulletOpenGLApplication::renderWorld()
{
	 m_renderer->renderScene();
	 s_pWorld->debugDrawWorld();
}

void BulletOpenGLApplication::renderSceneNew()
{
	syncPhysicsToGraphics();
	renderWorld();
	DrawGridData dg;
	dg.upAxis = 1;
	drawGrid( dg);
}

void BulletOpenGLApplication::drawGrid(DrawGridData data)
{
	int gridSize = data.gridSize;
	float upOffset = data.upOffset;
	int upAxis = data.upAxis;
	float gridColor[4];
	gridColor[0] = data.gridColor[0];
	gridColor[1] = data.gridColor[1];
	gridColor[2] = data.gridColor[2];
	gridColor[3] = data.gridColor[3];

	int sideAxis=-1;
	int forwardAxis=-1;

	switch (upAxis)
	{
	case 1:
		forwardAxis=2;
		sideAxis=0;
		break;
	case 2:
		forwardAxis=1;
		sideAxis=0;
		break;
	default:
		b3Assert(0);
	};
	//b3Vector3 gridColor = b3MakeVector3(0.5,0.5,0.5);

	b3AlignedObjectArray<unsigned int> indices;
	b3AlignedObjectArray<b3Vector3> vertices;
	int lineIndex=0;
	for(int i=-gridSize;i<=gridSize;i++)
	{
		{
			b3Assert(glGetError() ==GL_NO_ERROR);
			b3Vector3 from = b3MakeVector3(0,0,0);
			from[sideAxis] = float(i);
			from[upAxis] = upOffset;
			from[forwardAxis] = float(-gridSize);
			b3Vector3 to=b3MakeVector3(0,0,0);
			to[sideAxis] = float(i);
			to[upAxis] = upOffset;
			to[forwardAxis] = float(gridSize);
			vertices.push_back(from);
			indices.push_back(lineIndex++);
			vertices.push_back(to);
			indices.push_back(lineIndex++);
			// m_renderer->drawLine(from,to,gridColor);
		}

		b3Assert(glGetError() ==GL_NO_ERROR);
		{

			b3Assert(glGetError() ==GL_NO_ERROR);
			b3Vector3 from=b3MakeVector3(0,0,0);
			from[sideAxis] = float(-gridSize);
			from[upAxis] = upOffset;
			from[forwardAxis] = float(i);
			b3Vector3 to=b3MakeVector3(0,0,0);
			to[sideAxis] = float(gridSize);
			to[upAxis] = upOffset;
			to[forwardAxis] = float(i);
			vertices.push_back(from);
			indices.push_back(lineIndex++);
			vertices.push_back(to);
			indices.push_back(lineIndex++);
			// m_renderer->drawLine(from,to,gridColor);
		}

	}


	m_renderer->drawLines(&vertices[0].x,
		gridColor,
		vertices.size(),sizeof(b3Vector3),&indices[0],indices.size(),1);


	m_renderer->drawLine(b3MakeVector3(0,0,0),b3MakeVector3(1,0,0),b3MakeVector3(1,0,0),3);
	m_renderer->drawLine(b3MakeVector3(0,0,0),b3MakeVector3(0,1,0),b3MakeVector3(0,1,0),3);
	m_renderer->drawLine(b3MakeVector3(0,0,0),b3MakeVector3(0,0,1),b3MakeVector3(0,0,1),3);

	//	void GLInstancingRenderer::drawPoints(const float* positions, const float color[4], int numPoints, int pointStrideInBytes, float pointDrawSize)

	//we don't use drawPoints because all points would have the same color
	//	b3Vector3 points[3] = { b3MakeVector3(1, 0, 0), b3MakeVector3(0, 1, 0), b3MakeVector3(0, 0, 1) };
	//	m_renderer->drawPoints(&points[0].x, b3MakeVector3(1, 0, 0), 3, sizeof(b3Vector3), 6);

	m_renderer->drawPoint(b3MakeVector3(1,0,0),b3MakeVector3(1,0,0),6);
	m_renderer->drawPoint(b3MakeVector3(0,1,0),b3MakeVector3(0,1,0),6);
	m_renderer->drawPoint(b3MakeVector3(0,0,1),b3MakeVector3(0,0,1),6);
}
