#pragma once
#include <QVector>
#include <QVector2D>
#include <QVector3D>
#include <QQuaternion>
#include <QSharedPointer>
#include <QImage>
#include <vector>
#include <set>
#include <unordered_map>
#include "BulletDynamics/Dynamics/btDynamicsWorld.h"
class btSoftRigidDynamicsWorld;
class btSoftBodyRigidBodyCollisionConfiguration;
class btSoftBodyWorldInfo;
class btDispatcher;
class btBroadphaseInterface;
class btConstraintSolver;
class btSoftBody;

namespace videoEditting
{
	extern std::set<int> g_constrainted_nodes;
	extern std::vector<QVector3D>     g_init_vertices; 
	extern std::vector<QVector3D>	  g_init_normals;
	extern std::vector<int>           g_faces_;
	extern QVector3D                  g_init_translation;
	extern QQuaternion                g_init_rotation;
	extern std::vector<QVector3D>     g_translations;
	extern std::vector<QQuaternion>   g_rotations;
	extern std::set<int>			  g_pose_key_frame;
	extern std::vector<std::vector<QVector3D> > g_simulated_vertices;
	extern std::vector<std::vector<QVector3D> > g_simulated_normals;
	extern std::vector<std::vector<QVector2D> > g_tracked_textures;
	extern std::vector<std::vector<int> >		g_tracked_isVisiable;  //1 if visiable 0 if not
	extern std::vector<std::vector<int> >		g_simulated_isVisiable;
	extern std::vector<std::unordered_map<int, QVector3D>> g_position_constraint; //this constraint the position of vertices of frames
	extern int g_total_frame;//total frame 
	extern int g_current_frame;
	extern float g_time_step;
	extern std::vector<QImage>        g_cameraviewer_image_array;
}
class BulletInterface
{
public:
	friend void pickingPreTickCallback(btDynamicsWorld *world, btScalar timeStep);
	BulletInterface();
//	void initWorld();
	void setUpWorld();
	void begin_simulate();
	btRigidBody*localCreateRigidBody(float mass, const btTransform& startTransform, btCollisionShape* shape);
private:
	bool isWorldSetup;
	QSharedPointer<btSoftRigidDynamicsWorld> pWorld;
//	QSharedPointer<btSoftBodyRigidBodyCollisionConfiguration>  pCollisionConfiguration;
	//QSharedPointer<btSoftBodyWorldInfo>                        pWorldinfo;
	//QSharedPointer<btDispatcher>                               pDispatcher;
	//QSharedPointer<btBroadphaseInterface>                      pBroadphase;
	//QSharedPointer<btConstraintSolver>                         pSolver;
	btSoftBody*								   pSoftBody;
	QVector < btRigidBody*>				       pRidgidBody;
};