#include "bulletInterface.h"
#include "BulletDynamics/Dynamics/btDynamicsWorld.h"
#include "BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h"
#include "BulletSoftBody/btSoftRigidDynamicsWorld.h"
#include "BulletSoftBody/btSoftSoftCollisionAlgorithm.h"
#include "BulletSoftBody/btSoftBodyHelpers.h"
#include "BulletCollision/BroadphaseCollision/btDbvtBroadphase.h"
#include "BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.h"
#include "BulletCollision/CollisionShapes/btBoxShape.h"
#include "LinearMath/btDefaultMotionState.h"
#include "videoediting\BunnyMesh.h"
#include <iostream>
#include <QMatrix4x4>
namespace videoEditting
{
	std::set<int> g_constrainted_nodes;
	std::vector<QVector3D>   g_init_vertices;
	std::vector<QVector3D>   g_init_normals;
	std::vector<int>         g_faces_;
	QVector3D                g_init_translation;
	QQuaternion              g_init_rotation;
	std::vector<QVector3D>     g_translations;
	std::vector<QQuaternion>   g_rotations;
	std::set<int>            g_pose_key_frame;
	std::vector<std::vector<QVector3D> > g_simulated_vertices;
	std::vector<std::vector<QVector3D> > g_simulated_normals;
	std::vector<std::vector<QVector2D> > g_tracked_textures;
	std::vector<std::vector<int> >		 g_tracked_isVisiable;  
	std::vector<std::vector<int> >		 g_simulated_isVisiable;
	std::vector<std::unordered_map<int, QVector3D>> g_position_constraint; //this constraint the position of vertices of frames
	int g_total_frame = 0;//total frame 
	int g_current_frame = 0;
	float g_time_step = 0.001f;
	std::vector<QImage>       g_cameraviewer_image_array;
}

using namespace videoEditting;

///for mouse picking
void pickingPreTickCallback(btDynamicsWorld *world, btScalar timeStep)
{
	//SoftDemo* softDemo = (SoftDemo*)world->getWorldUserInfo();

	//if (softDemo->m_drag)
	//{
	//	const int				x = softDemo->m_lastmousepos[0];
	//	const int				y = softDemo->m_lastmousepos[1];
	//	const btVector3			rayFrom = softDemo->getCameraPosition();
	//	const btVector3			rayTo = softDemo->getRayTo(x, y);
	//	const btVector3			rayDir = (rayTo - rayFrom).normalized();
	//	const btVector3			N = (softDemo->getCameraTargetPosition() - softDemo->getCameraPosition()).normalized();
	//	const btScalar			O = btDot(softDemo->m_impact, N);
	//	const btScalar			den = btDot(N, rayDir);
	//	if ((den*den) > 0)
	//	{
	//		const btScalar			num = O - btDot(N, rayFrom);
	//		const btScalar			hit = num / den;
	//		if ((hit > 0) && (hit < 1500))
	//		{
	//			softDemo->m_goal = rayFrom + rayDir*hit;
	//		}
	//	}
	//	btVector3				delta = softDemo->m_goal - softDemo->m_node->m_x;
	//	static const btScalar	maxdrag = 10;
	//	if (delta.length2() > (maxdrag*maxdrag))
	//	{
	//		delta = delta.normalized()*maxdrag;
	//	}
	//	softDemo->m_node->m_v += delta / timeStep;
	//}
	BulletInterface* softDemo = (BulletInterface*)world->getWorldUserInfo();
	btSoftBody*  pSoftBody = (btSoftBody*)softDemo->pWorld->getSoftBodyArray()[0];
	//static int test_count = 0;
	//if (test_count < 4)
	//{
	//	for (int i = 0; i < g_constrainted_nodes.size(); ++i)
	//	{
	//		pSoftBody->m_nodes[g_constrainted_nodes[i]].m_v = btVector3(1.1, 0.1f, 0.1);
	//	}
	//}
	//else if (test_count < 8)
	//{
	//	for (int i = 0; i < g_constrainted_nodes.size(); ++i)
	//	{
	//		pSoftBody->m_nodes[g_constrainted_nodes[i]].m_v = btVector3(1.1, 0.1f, 0.1);
	//	}
	//}
	//else if (test_count >= 8)
	//{
	//	test_count = 0;
	//	for (int i = 0; i < g_constrainted_nodes.size(); ++i)
	//	{
	//		pSoftBody->m_nodes[g_constrainted_nodes[i]].m_v = btVector3(1.1, 0.1f, 0.1);
	//	}
	//}


//	test_count++;

}



BulletInterface::BulletInterface():isWorldSetup(false),pSoftBody(NULL)
{

}


btRigidBody* BulletInterface::localCreateRigidBody(float mass, const btTransform& startTransform, btCollisionShape* shape)
{
//	btAssert((!shape || shape->getShapeType() != INVALID_SHAPE_PROXYTYPE));
	//rigidbody is dynamic if and only if mass is non zero, otherwise static
	bool isDynamic = (mass != 0.f);

	btVector3 localInertia(0, 0, 0);
	if (isDynamic)
		shape->calculateLocalInertia(mass, localInertia);
	btDefaultMotionState* myMotionState = new btDefaultMotionState(startTransform);

	btRigidBody::btRigidBodyConstructionInfo cInfo(mass, myMotionState, shape, localInertia);

	btRigidBody* body = new btRigidBody(cInfo);

	float defaultContactProcessingThreshold = 1e18f;
	body->setContactProcessingThreshold(defaultContactProcessingThreshold);
	if(pWorld)
		pWorld->addRigidBody(body);
	return body;
}


void BulletInterface::setUpWorld()
{
	if (g_total_frame < 1)
		return;
	btSoftBodyRigidBodyCollisionConfiguration* pCollisionConfiguration =
		new btSoftBodyRigidBodyCollisionConfiguration();
	// create the dispatcher
	btDispatcher* pDispatcher =  new btCollisionDispatcher(pCollisionConfiguration);
	// create the broadphase
	btBroadphaseInterface* pBroadphase = new btDbvtBroadphase();
	// create the constraint solver
	btConstraintSolver* pSolver = new btSequentialImpulseConstraintSolver();

	pWorld = QSharedPointer<btSoftRigidDynamicsWorld>(new btSoftRigidDynamicsWorld(pDispatcher,
		pBroadphase, pSolver, pCollisionConfiguration));
	btSoftBodyWorldInfo* pWorldinfo = new btSoftBodyWorldInfo();
	pWorldinfo->m_dispatcher = pDispatcher;
	pWorldinfo->m_broadphase = pBroadphase;
	pWorldinfo->m_sparsesdf.Initialize();


	pWorld->setInternalTickCallback(pickingPreTickCallback, this, true);

//	pWorld->getDispatchInfo().m_enableSPU = true;
	pWorld->setGravity(btVector3(0, -10, 0));
	pWorldinfo->m_gravity.setValue(0, -10, 0);




	btScalar* vertices = new btScalar[3*g_init_vertices.size()];
	//
	if( g_translations.size() && g_rotations.size())
	{
		QVector3D& trans = g_translations[0];
		QQuaternion& rot = g_rotations[0];
		QMatrix4x4 tr;
		QMatrix4x4 m_rotMatrix;
		m_rotMatrix.setToIdentity();
		m_rotMatrix.rotate(rot);
		tr.translate(trans);
		tr *= m_rotMatrix;
		//tr.rotate(rot);  //这样不对
		//tr.translate(trans);
		for (int i = 0; i < g_init_vertices.size(); ++i)
		{
			QVector3D converted_vertexs = tr * g_init_vertices[i];
			vertices[3 * i    ] = converted_vertexs.x();
			vertices[3 * i + 1] = converted_vertexs.y();
			vertices[3 * i + 2] = converted_vertexs.z();
		}	
	}
	else
	{
		std::cout << "g_translation or g_rotation size == 0" << std::endl;
	}
	std::vector<QVector3D> tmp_vertex = g_init_vertices;

	std::vector<int> tmp_face = g_faces_;
	int* faces = new int[g_faces_.size()];
	for (int i = 0; i < g_faces_.size(); ++i)
	{
		faces[i] = g_faces_[i];
	}
	int n_triangle = g_faces_.size()/3;
	pSoftBody = btSoftBodyHelpers::CreateFromTriMesh(*(pWorldinfo),vertices, faces, n_triangle);

	//pSoftBody = btSoftBodyHelpers::CreateFromTriMesh(*(pWorldinfo), gVerticesBunny,
	//	&gIndicesBunny[0][0],
	//	BUNNY_NUM_TRIANGLES);

	g_init_translation = g_translations[0];
	g_init_rotation = g_rotations[0];

	btSoftBody::Material*	pm = pSoftBody->appendMaterial();
	g_init_translation = g_translations[0];
	g_init_rotation = g_rotations[0];
	// set the body's pose
	//在soft body 中这个 translate 和 rotate 没用
	//pSoftBody->translate(btVector3(g_init_translation.x(), g_init_translation.y(), g_init_translation.z()));
	//pSoftBody->rotate(btQuaternion(g_init_rotation.x(), g_init_rotation.y(), g_init_rotation.z(), g_init_rotation.scalar()));
/*	
	pSoftBody->generateBendingConstraints(2, pm);
	pSoftBody->m_cfg.piterations = 2;
	pSoftBody->m_cfg.kDF = 0.5;
	// set the 'volume conservation coefficient'
	pSoftBody->m_cfg.kVC = 0.5;
	// set the 'linear stiffness'
	pSoftBody->m_materials[0]->m_kLST = 0.5;
	// set the total mass of the soft body
	pSoftBody->setTotalMass(10);
	*/
//	btSoftBody::Material*	pm = pSoftBody->appendMaterial();
	pm->m_kLST = 1.5;//before 0.5
	pm->m_flags -= btSoftBody::fMaterial::DebugDraw;
	pSoftBody->generateBendingConstraints(2, pm);
	pSoftBody->m_cfg.piterations = 2;
	pSoftBody->m_cfg.kDF = 0.5;
	pSoftBody->randomizeConstraints();

	pSoftBody->setTotalMass(100, true);
	// set the 'volume conservation coefficient'
	pSoftBody->m_cfg.kVC = 0.001; //add by huayun
	

	if ( !g_position_constraint.size())
	{
		std::cout << "constraint not set up" << std::endl;
		return;
	}
	//设置初始的位置约束
	pRidgidBody.clear();
	btTransform startTransform;
	startTransform.setIdentity();
	btRigidBody* body = NULL; // localCreateRigidBody(20, startTransform, new btBoxShape(btVector3(10.01, 10.01, 10.01)));
	if(body)
		body->setMassProps(0, btVector3(1, 1, 1)); //设置为不受重力影响
	QVector3D constraint_center;
	for (auto bitr = g_position_constraint[0].begin(); bitr != g_position_constraint[0].end(); ++bitr)
	{
/* 每个 约束点 生成一个 rigid body

		int vtx_id = bitr->first;
		QVector3D pos = bitr->second;
		std::unordered_map<int, QVector3D>& map = g_position_constraint[0];
		//int vertx_idx = bitr->first;
		//QVector3D& vec = bitr->second;
		//pBody->setMass(vertx_idx, 0);
		//pBody->m_nodes[vertx_idx].m_x = btVector3(vec.x(), vec.y(), vec.z());
		btTransform startTransform;
		startTransform.setOrigin(btVector3(pos.x(), pos.y(), pos.z()));
		btRigidBody* body = localCreateRigidBody(20, startTransform, new btBoxShape(btVector3(0.01, 0.01, 0.01)));
		body->setMassProps(0, btVector3(1, 1, 1)); //设置为不受重力影响
		pRidgidBody.push_back( body);
		pSoftBody->appendAnchor(vtx_id, body, true, 1);
		*/
/*一组约束点生成一个rigid body*/
/*
		int vtx_id = bitr->first;
		QVector3D pos = bitr->second;
		constraint_center += pos;

		pSoftBody->appendAnchor(vtx_id, body, true, 1);
*/
		int vtx_id = bitr->first;
		QVector3D pos = bitr->second;
		pSoftBody->setMass(vtx_id, 0);
		pSoftBody->m_nodes[vtx_id].m_x = btVector3(pos.x(), pos.y(), pos.z());
	}
	constraint_center /= g_position_constraint.size();
	startTransform.setIdentity();
	startTransform.setOrigin(btVector3(constraint_center.x(), constraint_center.y(), constraint_center.z()));
	if (body)
	{
		pRidgidBody.push_back(body);
		body->setWorldTransform(startTransform);

	}
		
	//g_constrainted_nodes.push_back(120);
	//g_constrainted_nodes.push_back(105);
	//g_constrainted_nodes.push_back(94);
	//g_constrainted_nodes.push_back(71);
//	g_constrainted_nodes.push_back(0);
	//g_constrainted_nodes.push_back(1);
	//g_constrainted_nodes.push_back(2);
	//g_constrainted_nodes.push_back(3);
	//g_constrainted_nodes.push_back(4);
	//for (int i = 0; i < g_constrainted_nodes.size(); ++i)
	//{
	//	pSoftBody->setMass(g_constrainted_nodes[i], 0);
	//}

	// tell the soft body to initialize and
	// attempt to maintain the current pose
//	pSoftBody->setPose(true, false);
	pWorld->addSoftBody(pSoftBody);

	delete[] vertices;
	delete[] faces;
	isWorldSetup = true;
}

void BulletInterface::begin_simulate()
{

	static int test_count = 0;
	if (!isWorldSetup)
	{
		setUpWorld();
		
	}
	if (pWorld)
	{
		g_simulated_vertices.clear();
		g_simulated_vertices.resize(g_total_frame);
		g_simulated_normals.clear();
		g_simulated_normals.resize(g_total_frame);

		if (g_simulated_vertices.size() > 0 && g_simulated_normals.size() > 0)
		{
			for (int i = 0; i < pWorld->getSoftBodyArray().size(); i++)
			{
				// get the body
				btSoftBody*  pBody = (btSoftBody*)pWorld->getSoftBodyArray()[i];
				pBody->updateNormals();
				// is it possible to render?
				//					if (pWorld->getDebugDrawer() && !(pWorld->getDebugDrawer()->getDebugMode() & (btIDebugDraw::DBG_DrawWireframe))) {
				// draw it
				//					btSoftBodyHelpers::Draw(pBody, pWorld->getDebugDrawer(), pWorld->getDrawFlags());
				for (int j = 0; j < pBody->m_nodes.size(); ++j)
				{
					const btSoftBody::Node&	n = pBody->m_nodes[j];
					const btVector3& vertex = n.m_x;
					const btVector3& normal = n.m_n;
					g_simulated_vertices[0].push_back(QVector3D(vertex.x(), vertex.y(), vertex.z()));
					g_simulated_normals[0].push_back(QVector3D(normal.x(), normal.y(), normal.z()));
				}
				break;
			}
		}


		for (int cur_frameid = 1; cur_frameid < g_total_frame; ++cur_frameid)
		{
			btSoftBody*  pBody = (btSoftBody*)pWorld->getSoftBodyArray()[0];
			// prevent the picked object from falling asleep
			pBody->setActivationState(DISABLE_DEACTIVATION);
/*			
			for (auto bitr = g_position_constraint[cur_frameid].begin(); bitr != g_position_constraint[cur_frameid].end(); ++bitr)
			{
				int vertx_idx = bitr->first;
				QVector3D pos = bitr->second;
				for (size_t i = 0; i < pRidgidBody.size(); i++)
				{
					pRidgidBody[i]->getWorldTransform().setOrigin( btVector3(pos.x(), pos.y(), pos.z()));
				}
				//pBody->setMass(vertx_idx, 1);
			}
		

*/			QVector3D constraint_center;
			for (auto bitr = g_position_constraint[cur_frameid].begin(); bitr != g_position_constraint[cur_frameid].end(); ++bitr)
			{
				//int vertx_idx = bitr->first;
				//QVector3D& vec = bitr->second;
				//constraint_center += vec;
				int vtx_id = bitr->first;
				QVector3D pos = bitr->second;
//				pBody->setMass(vtx_id, 0);
				btVector3 prev = pBody->m_nodes[vtx_id].m_x;
				btVector3 target = btVector3(pos.x(), pos.y(), pos.z());
				btVector3 velocity = (target - prev)/ g_time_step;
//				pBody->m_nodes[vtx_id].m_x = btVector3(pos.x(), pos.y(), pos.z());
		//		pBody->m_nodes[vtx_id].m_v += velocity;
				pBody->m_nodes[vtx_id].m_x = target;

			}
			constraint_center /= g_position_constraint.size();
			btTransform startTransform;
			startTransform.setIdentity();
			startTransform.setOrigin(btVector3(constraint_center.x(), constraint_center.y(), constraint_center.z()));

			for (size_t i = 0; i < pRidgidBody.size(); i++)
			{
				pRidgidBody[i]->setWorldTransform(startTransform);
			}
			

			//test 
//			pBody->setMass(446, 0);
//			pBody->m_nodes[50].m_v += btVector3(10.1, 1.f, 0.1);

			float tmp_time_step = g_time_step;
			pWorld->stepSimulation(g_time_step);
//			g_current_frame++;


			for (int i = 0; i < pWorld->getSoftBodyArray().size(); i++)
			{
				// get the body
				btSoftBody*  pBody = (btSoftBody*)pWorld->getSoftBodyArray()[i];
				// is it possible to render?
//					if (pWorld->getDebugDrawer() && !(pWorld->getDebugDrawer()->getDebugMode() & (btIDebugDraw::DBG_DrawWireframe))) {
				// draw it
//					btSoftBodyHelpers::Draw(pBody, pWorld->getDebugDrawer(), pWorld->getDrawFlags());
				g_simulated_vertices[cur_frameid].clear();
				g_simulated_normals[cur_frameid].clear();
				for (int j = 0; j < pBody->m_nodes.size(); ++j)
				{
					const btSoftBody::Node&	n = pBody->m_nodes[j];
					const btVector3& vertex = n.m_x;
					const btVector3& normal = n.m_n;
					g_simulated_vertices[cur_frameid].push_back(QVector3D(vertex.x(), vertex.y(), vertex.z()));
					g_simulated_normals[cur_frameid].push_back(QVector3D(normal.x(), normal.y(), normal.z()));
				}
			}

		}




	}

}
