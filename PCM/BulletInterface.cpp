#include "bulletInterface.h"
#include "BulletDynamics/Dynamics/btDynamicsWorld.h"
#include "BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h"
#include "BulletSoftBody/btSoftRigidDynamicsWorld.h"
#include "BulletSoftBody/btSoftSoftCollisionAlgorithm.h"
#include "BulletSoftBody/btSoftBodyHelpers.h"
#include "BulletCollision/BroadphaseCollision/btDbvtBroadphase.h"
#include "BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.h"

namespace videoEditting
{
	std::vector<int> g_constrainted_nodes;
	std::vector<QVector3D>   g_init_vertices;
	std::vector<int>         g_faces_;
	QVector3D                g_init_translation;
	QQuaternion              g_init_rotation;
	std::vector<QVector3D>     g_translations;
	std::vector<QQuaternion>   g_rotations;
	std::set<int>           g_pose_key_frame;
	std::vector<std::vector<QVector3D> > g_simulated_vertices;
	std::vector<std::vector<QVector3D> > g_simulated_normals;
	std::vector<std::unordered_map<int, QVector3D>> g_position_constraint; //this constraint the position of vertices of frames
	int g_total_frame = 0;//total frame 
	int g_current_frame = 0;
	float g_time_step = 0.001f;
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
	static int test_count = 0;
	if (test_count < 4)
	{
		for (int i = 0; i < g_constrainted_nodes.size(); ++i)
		{
			pSoftBody->m_nodes[g_constrainted_nodes[i]].m_v = btVector3(1.1, 0.1f, 0.1);
		}
	}
	else if (test_count < 8)
	{
		for (int i = 0; i < g_constrainted_nodes.size(); ++i)
		{
			pSoftBody->m_nodes[g_constrainted_nodes[i]].m_v = btVector3(1.1, 0.1f, 0.1);
		}
	}
	else if (test_count >= 8)
	{
		test_count = 0;
		for (int i = 0; i < g_constrainted_nodes.size(); ++i)
		{
			pSoftBody->m_nodes[g_constrainted_nodes[i]].m_v = btVector3(1.1, 0.1f, 0.1);
		}
	}


	test_count++;

}



BulletInterface::BulletInterface():isWorldSetup(false)
{

}





void BulletInterface::setUpWorld()
{
	if (g_total_frame < 1)
		return;
	pCollisionConfiguration =
		QSharedPointer<btSoftBodyRigidBodyCollisionConfiguration>(new btSoftBodyRigidBodyCollisionConfiguration());
	// create the dispatcher
	 pDispatcher = QSharedPointer<btDispatcher>(new btCollisionDispatcher(pCollisionConfiguration.data()));
	// create the broadphase
	pBroadphase = QSharedPointer<btBroadphaseInterface>(new btDbvtBroadphase());
	// create the constraint solver
	pSolver = QSharedPointer<btConstraintSolver>(new btSequentialImpulseConstraintSolver());

	pWorld = QSharedPointer<btSoftRigidDynamicsWorld>(new btSoftRigidDynamicsWorld(pDispatcher.data(),
		pBroadphase.data(), pSolver.data(), pCollisionConfiguration.data()));
	pWorldinfo = QSharedPointer<btSoftBodyWorldInfo>(new btSoftBodyWorldInfo());
	pWorldinfo->m_dispatcher = pDispatcher.data();
	pWorldinfo->m_broadphase = pBroadphase.data();
	pWorldinfo->m_sparsesdf.Initialize();

	pWorld->setInternalTickCallback(pickingPreTickCallback, this, true);
	btScalar* vertices = new btScalar[3*g_init_vertices.size()];
	//
	std::vector<QVector3D> tmp_vertex = g_init_vertices;
	for (int i = 0; i < g_init_vertices.size();++i)
	{
		vertices[3 * i]   = g_init_vertices[i].x();
		vertices[3 * i+1] = g_init_vertices[i].y();
		vertices[3 * i+2] = g_init_vertices[i].z();
	}
	std::vector<int> tmp_face = g_faces_;
	int* faces = new int[g_faces_.size()];
	for (int i = 0; i < g_faces_.size(); ++i)
	{
		faces[i] = g_faces_[i];
	}
	int n_triangle = g_faces_.size()/3;
	pSoftBody = QSharedPointer<btSoftBody>(
		btSoftBodyHelpers::CreateFromTriMesh(*(pWorldinfo.value),vertices, faces, n_triangle)
		);

	btSoftBody::Material*	pm = pSoftBody->appendMaterial();
	g_init_translation = g_translations[0];
	g_init_rotation = g_rotations[0];
	// set the body's pose
	pSoftBody->translate(btVector3(g_init_translation.x(), g_init_translation.y(), g_init_translation.z()));
	pSoftBody->rotate(btQuaternion(g_init_rotation.x(), g_init_rotation.y(), g_init_rotation.z(), g_init_rotation.scalar()));
	
	pSoftBody->generateBendingConstraints(2, pm);
	pSoftBody->m_cfg.piterations = 2;
	pSoftBody->m_cfg.kDF = 0.5;
	// set the 'volume conservation coefficient'
	pSoftBody->m_cfg.kVC = 0.5;
	// set the 'linear stiffness'
	pSoftBody->m_materials[0]->m_kLST = 0.5;
	// set the total mass of the soft body
	pSoftBody->setTotalMass(5);
	for (int i = 0; i < pSoftBody->m_nodes.size(); ++i)
	{
//		pSoftBody->setMass(i, 1);
	}
	//g_constrainted_nodes.push_back(120);
	//g_constrainted_nodes.push_back(105);
	//g_constrainted_nodes.push_back(94);
	//g_constrainted_nodes.push_back(71);
	g_constrainted_nodes.push_back(0);
	//g_constrainted_nodes.push_back(1);
	//g_constrainted_nodes.push_back(2);
	//g_constrainted_nodes.push_back(3);
	//g_constrainted_nodes.push_back(4);
	for (int i = 0; i < g_constrainted_nodes.size(); ++i)
	{
		pSoftBody->setMass(g_constrainted_nodes[i], 0);
	}



	// tell the soft body to initialize and
	// attempt to maintain the current pose
	pSoftBody->setPose(true, false);
	pWorld->addSoftBody(pSoftBody.data());

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
		for (int cur_frameid = 0; cur_frameid < g_total_frame; ++cur_frameid)
		{
			btSoftBody*  pBody = (btSoftBody*)pWorld->getSoftBodyArray()[0];
			// prevent the picked object from falling asleep
			pBody->setActivationState(DISABLE_DEACTIVATION);
			for (auto bitr = g_position_constraint[cur_frameid].begin(); bitr != g_position_constraint[cur_frameid].end(); ++bitr)
			{
				int vertx_idx = bitr->first;
				pBody->setMass(vertx_idx, 1);
			}
			for (auto bitr = g_position_constraint[cur_frameid].begin(); bitr != g_position_constraint[cur_frameid].end(); ++bitr)
			{
				int vertx_idx = bitr->first;
				QVector3D& vec = bitr->second;
				pBody->setMass(vertx_idx, 0);
				pBody->m_nodes[vertx_idx].m_x = btVector3(vec.x(), vec.y(), vec.z());
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
