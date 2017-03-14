#ifndef _BULLETOPENGLAPP_H_
#define _BULLETOPENGLAPP_H_

//#include <Windows.h>
//#include <GL/GL.h>


#include "BulletDynamics/Dynamics/btDynamicsWorld.h"

// include our custom Motion State object
#include "OpenGLMotionState.h"

// Our custom debug renderer
#include "DebugDrawer.h"

#include "GameObject.h"
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

// a convenient typedef to reference an STL vector of GameObjects
typedef std::vector<GameObject*> GameObjects;

// convenient typedefs for collision events
typedef std::pair<const btRigidBody*, const btRigidBody*> CollisionPair;
typedef std::set<CollisionPair> CollisionPairs;

// struct to store our raycasting results

struct DrawGridData
{
	int gridSize;
	float upOffset;
	int upAxis;
	float gridColor[4];

	DrawGridData(int upAxis=1)
		:gridSize(10),
		upOffset(0.001f),
		upAxis(upAxis)
	{
		gridColor[0] = 0.6f;
		gridColor[1] = 0.6f;
		gridColor[2] = 0.6f;
		gridColor[3] = 1.f;
	}
};


struct RayResult {
 	btRigidBody* pBody;
 	btVector3 hitPoint;
};
class PaintCanvas;
class GLInstancingRenderer;

class BulletOpenGLApplication {
public:
	BulletOpenGLApplication(PaintCanvas* qgl);
	~BulletOpenGLApplication();
	void Initialize();
	void InitializeNaive();
	// FreeGLUT callbacks //
	virtual void Keyboard(unsigned char key, int x, int y);
	virtual void KeyboardUp(unsigned char key, int x, int y);
	virtual void Special(int key, int x, int y);
	virtual void SpecialUp(int key, int x, int y);
	virtual void Reshape(int w, int h);
	virtual void Idle();
	virtual void Mouse(int button, int state, int x, int y);
	virtual void PassiveMotion(int x, int y);
	virtual void Motion(int x, int y);
	virtual void Wheel( float deltax, float deltaY);
	virtual void Display();

	
	// rendering. Can be overrideen by derived classes
	virtual void RenderScene();

	// scene updating. Can be overridden by derived classes
	virtual void UpdateScene(float dt);

	// physics functions. Can be overrideen by derived classes (like BasicDemo)
	virtual void InitializePhysics() {};
	virtual void ShutdownPhysics() {};

	// camera functions
	void UpdateCamera();
	void RotateCamera(float &angle, float value);
	void ZoomCamera(float distance);

	// drawing functions
	void DrawBox(const btVector3 &halfSize);
	void DrawShape(btScalar* transform, const btCollisionShape* pShape, const btVector3 &color);

	// object functions
	GameObject* CreateGameObject(btCollisionShape* pShape, 
			const float &mass, 
			const btVector3 &color = btVector3(1.0f,1.0f,1.0f), 
			const btVector3 &initialPosition = btVector3(0.0f,0.0f,0.0f), 
			const btQuaternion &initialRotation = btQuaternion(0,0,1,1));

 	void ShootBox(const btVector3 &direction);
 	void DestroyGameObject(btRigidBody* pBody);
	GameObject* FindGameObject(btRigidBody* pBody);

 	// picking functions
 	btVector3 GetPickingRay(int x, int y);
/*REM*	bool Raycast(const btVector3 &startPosition, const btVector3 &direction, RayResult &output); **/
  	bool Raycast(const btVector3 &startPosition, const btVector3 &direction, RayResult &output, bool includeStatic = false);
	
	// constraint functions
	void CreatePickingConstraint(int x, int y);
	void RemovePickingConstraint();

	virtual bool pickBody(const btVector3& rayFromWorld , const btVector3& rayToWorld);
	virtual bool movePickedBody(const btVector3& rayFromWorld, const btVector3& rayToWorld) ;
	// collision event functions
	void CheckForCollisionEvents();
	virtual void CollisionEvent(btRigidBody* pBody0, btRigidBody * pBody1);
	virtual void SeparationEvent(btRigidBody * pBody0, btRigidBody * pBody1);

	void renderSceneNew();
	void drawGrid(DrawGridData data);
protected:

	void syncPhysicsToGraphics();
	void renderWorld();
	// camera control
	btVector3 m_cameraPosition; // the camera's current position
	btVector3 m_cameraTarget;	 // the camera's lookAt target
	float m_nearPlane; // minimum distance the camera will render
	float m_farPlane; // farthest distance the camera will render
	btVector3 m_upVector; // keeps the camera rotated correctly
	float m_cameraDistance; // distance from the camera to its target
	float m_cameraPitch; // pitch of the camera 
	float m_cameraYaw; // yaw of the camera

	int m_screenWidth;
	int m_screenHeight;

	// core Bullet components
	static btBroadphaseInterface* s_pBroadphase;
	static btCollisionConfiguration* s_pCollisionConfiguration;
	static btCollisionDispatcher* s_pDispatcher;
	static btConstraintSolver* s_pSolver;
	static btDynamicsWorld* s_pWorld;

	// a simple clock for counting time
	static btClock s_clock;

	// an array of our game objects
	GameObjects m_objects;

	// debug renderer
	DebugDrawer* m_pDebugDrawer;

	// constraint variables
	static btRigidBody* s_pPickedBody;				// the body we picked up
	static btTypedConstraint*  s_pPickConstraint;	// the constraint the body is attached to
	btScalar m_oldPickingDist;				// the distance from the camera to the hit point (so we can move the object up, down, left and right from our view)

	// collision event variables
	CollisionPairs m_pairsLastUpdate;

	PaintCanvas* m_qgl;
	GLInstancingRenderer* m_renderer;
	//
	//data for picking objects
	static btRigidBody*	s_pickedBody;
	static btTypedConstraint* s_pickedConstraint;
	static int	m_savedState;
	static btVector3 m_oldPickingPos;
	static btVector3 m_hitPos;
//	btScalar m_oldPickingDist;

};
#endif
