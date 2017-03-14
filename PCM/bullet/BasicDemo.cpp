#include "BasicDemo.h"
#include "GLInstancingRenderer.h"
#include "CollisionShape2TriangleMesh.h"
#include "GLInstanceGraphicsShape.h"
#include <iostream>
#define ARRAY_SIZE_Y 5
#define ARRAY_SIZE_X 5
#define ARRAY_SIZE_Z 5


static btVector4 sColors[4] =
{
	btVector4(0.3,0.3,1,0.1),
	btVector4(0.6,0.6,1,0.1),
	btVector4(0,1,0,0.1),
	btVector4(0,1,1,0.1),
	//btVector4(1,1,0,1),
};
 BasicDemo::BasicDemo(PaintCanvas* qgl):
  	BulletOpenGLApplication(qgl),
	m_bApplyForce(false),
  	m_pExplosion(0),
	m_bCanExplode(true)
   {
		std::cout<<"BulletOpenGLApplication";
   }

 void BasicDemo::Initialize()
 {
	 
	 BulletOpenGLApplication::InitializeNaive();
 }
void BasicDemo::InitializePhysics() {

	createEmptyDynamicsWorld();
	// create our scene's physics objects
	//CreateObjects();
	CreateObjects2();
}

void BasicDemo::createEmptyDynamicsWorld()
{
	// create the collision configuration
	s_pCollisionConfiguration = new btDefaultCollisionConfiguration();
	// create the dispatcher
	s_pDispatcher = new btCollisionDispatcher(s_pCollisionConfiguration);
	// create the broadphase
	s_pBroadphase = new btDbvtBroadphase();
	// create the constraint solver
	s_pSolver = new btSequentialImpulseConstraintSolver();
	// create the world
	s_pWorld = new btDiscreteDynamicsWorld(s_pDispatcher, s_pBroadphase, s_pSolver, s_pCollisionConfiguration);
	s_pWorld->setGravity(btVector3(0, -10, 0));
}
void BasicDemo::ShutdownPhysics() {
	delete s_pWorld;
	delete s_pSolver;
	delete s_pBroadphase;
	delete s_pDispatcher;
	delete s_pCollisionConfiguration;
}

void BasicDemo::CreateObjects() {
	// create a ground plane
	CreateGameObject(new btBoxShape(btVector3(1,50,50)), 0, btVector3(0.2f, 0.6f, 0.6f), btVector3(0.0f, 0.0f, 0.0f));

	// create our red box, but store the pointer for future usage
	m_pBox = CreateGameObject(new btBoxShape(btVector3(1,1,1)), 1.0, btVector3(1.0f, 0.2f, 0.2f), btVector3(0.0f, 10.0f, 0.0f));

	// create a second box
	CreateGameObject(new btBoxShape(btVector3(1,1,1)), 1.0, btVector3(0.0f, 0.2f, 0.8f), btVector3(1.25f, 20.0f, 0.0f));

	// create a trigger volume
	m_pTrigger = new btCollisionObject();
	// create a box for the trigger's shape
	m_pTrigger->setCollisionShape(new btBoxShape(btVector3(1,0.25,1)));
	// set the trigger's position
	btTransform triggerTrans;
	triggerTrans.setIdentity();
	triggerTrans.setOrigin(btVector3(0,1.5,0));
	m_pTrigger->setWorldTransform(triggerTrans);
	// flag the trigger to ignore contact responses
	m_pTrigger->setCollisionFlags(btCollisionObject::CF_NO_CONTACT_RESPONSE);
	// add the trigger to our world
	s_pWorld->addCollisionObject(m_pTrigger);
}

void BasicDemo::CreateObjects2()
{
	///create a few basic rigid bodies
	btBoxShape* groundShape = createBoxShape(btVector3(btScalar(50.),btScalar(50.),btScalar(50.)));


	//groundShape->initializePolyhedralFeatures();
	//btCollisionShape* groundShape = new btStaticPlaneShape(btVector3(0,1,0),50);

	m_collisionShapes.push_back(groundShape);

	btTransform groundTransform;
	groundTransform.setIdentity();
	groundTransform.setOrigin(btVector3(0,-50,0));

	{
		btScalar mass(0.);
		createRigidBody(mass,groundTransform,groundShape, btVector4(0,0,1,1));
	}


	{
		//create a few dynamic rigidbodies
		// Re-using the same collision is better for memory usage and performance

		btBoxShape* colShape = createBoxShape(btVector3(.1,.1,.1));


		//btCollisionShape* colShape = new btSphereShape(btScalar(1.));
		m_collisionShapes.push_back(colShape);

		/// Create Dynamic Objects
		btTransform startTransform;
		startTransform.setIdentity();

		btScalar	mass(1.f);

		//rigidbody is dynamic if and only if mass is non zero, otherwise static
		bool isDynamic = (mass != 0.f);

		btVector3 localInertia(0,0,0);
		if (isDynamic)
			colShape->calculateLocalInertia(mass,localInertia);


		for (int k=0;k<ARRAY_SIZE_Y;k++)
		{
			for (int i=0;i<ARRAY_SIZE_X;i++)
			{
				for(int j = 0;j<ARRAY_SIZE_Z;j++)
				{
					startTransform.setOrigin(btVector3(
						btScalar(0.2*i),
						btScalar(2+.2*k),
						btScalar(0.2*j)));


					createRigidBody(mass,startTransform,colShape);


				}
			}
		}
	}

	autogenerateGraphicsObjects( s_pWorld);

}

btBoxShape* BasicDemo::createBoxShape(const btVector3& halfExtents)
{
	btBoxShape* box = new btBoxShape(halfExtents);
	return box;
}

btRigidBody*	BasicDemo::createRigidBody(float mass, const btTransform& startTransform, btCollisionShape* shape,  const btVector4& color ,int user_idex ,btMotionState* motion_state )
{
	btAssert((!shape || shape->getShapeType() != INVALID_SHAPE_PROXYTYPE));

	//rigidbody is dynamic if and only if mass is non zero, otherwise static
	bool isDynamic = (mass != 0.f);

	btVector3 localInertia(0, 0, 0);
	if (isDynamic)
		shape->calculateLocalInertia(mass, localInertia);

	//using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects

#define USE_MOTIONSTATE 1
#ifdef USE_MOTIONSTATE
	btMotionState* myMotionState ; 
	if(motion_state)
	{
		myMotionState = motion_state;
	}else
	{
		myMotionState = new btDefaultMotionState(startTransform);
	}


	btRigidBody::btRigidBodyConstructionInfo cInfo(mass, myMotionState, shape, localInertia);

	btRigidBody* body = new btRigidBody(cInfo);
	//body->setContactProcessingThreshold(m_defaultContactProcessingThreshold);

#else
	btRigidBody* body = new btRigidBody(mass, 0, shape, localInertia);
	body->setWorldTransform(startTransform);
#endif//

	body->setUserIndex(user_idex); //default -1
	s_pWorld->addRigidBody(body);
	return body;
}
void BasicDemo::CollisionEvent(btRigidBody* pBody0, btRigidBody* pBody1) {
	// did the box collide with the trigger?
	if (pBody0 == m_pBox->GetRigidBody() && pBody1 == m_pTrigger ||
		pBody1 == m_pBox->GetRigidBody() && pBody0 == m_pTrigger) {
			// if yes, create a big green box nearby
			CreateGameObject(new btBoxShape(btVector3(2,2,2)), 2.0, btVector3(0.3, 0.7, 0.3), btVector3(5, 10, 0));
	}

  		// Impulse testing
  	if (pBody0 == m_pExplosion || pBody1 == m_pExplosion) {
  		// get the pointer of the other object
  		btRigidBody* pOther;
  		pBody0 == m_pExplosion ? pOther = (btRigidBody*)pBody1 : pOther = (btRigidBody*)pBody0;
  		// wake the object up
  		pOther->setActivationState(ACTIVE_TAG);
  		// calculate the vector between the object and
  		// the center of the explosion
  		btVector3 dir = pOther->getWorldTransform().getOrigin() - m_pExplosion->getWorldTransform().getOrigin();
  		// get the distance
  		float dist = dir.length();
  		// calculate the impulse strength
  		float strength = EXPLOSION_STRENGTH;
  		// follow an inverse-distance rule
  		if (dist != 0.0) strength /= dist;
  		// normalize the direction vector
  		dir.normalize();
  		// apply the impulse
  		pOther->applyCentralImpulse(dir * strength);
  	}
}

struct MyConvertPointerSizeT
{
	union 
	{
		const void* m_ptr;
		size_t m_int;
	};
};

bool shapePointerCompareFunc(const btCollisionObject* colA, const btCollisionObject* colB)
{
	MyConvertPointerSizeT a,b;
	a.m_ptr = colA->getCollisionShape();
	b.m_ptr = colB->getCollisionShape();
	return (a.m_int<b.m_int);
}
void BasicDemo::autogenerateGraphicsObjects(btDynamicsWorld* rbWorld)
{
	//sort the collision objects based on collision shape, the gfx library requires instances that re-use a shape to be added after eachother

	btAlignedObjectArray<btCollisionObject*> sortedObjects;
	sortedObjects.reserve(rbWorld->getNumCollisionObjects());
	for (int i=0;i<rbWorld->getNumCollisionObjects();i++)
	{
		btCollisionObject* colObj = rbWorld->getCollisionObjectArray()[i];
		sortedObjects.push_back(colObj);
	}
	sortedObjects.quickSort(shapePointerCompareFunc);  //to ensure that collisionObjct with same shape be aligned together
	for (int i=0;i<sortedObjects.size();i++)
	{
		btCollisionObject* colObj = sortedObjects[i];
		//btRigidBody* body = btRigidBody::upcast(colObj);
		//does this also work for btMultiBody/btMultiBodyLinkCollider?
		createCollisionShapeGraphicsObject(colObj->getCollisionShape());
		int colorIndex = colObj->getBroadphaseHandle()->getUid() & 3;

		btVector3 color= sColors[colorIndex];
		createCollisionObjectGraphicsObject(colObj,color);
	}
}

void BasicDemo::createCollisionShapeGraphicsObject(btCollisionShape* collisionShape)
{
	//already has a graphics object?
	if (collisionShape->getUserIndex()>=0)
		return;

	btAlignedObjectArray<GLInstanceVertex> gfxVertices;

	btAlignedObjectArray<int> indices;
	btTransform startTrans;startTrans.setIdentity();

	{
		btAlignedObjectArray<btVector3> vertexPositions;
		btAlignedObjectArray<btVector3> vertexNormals;
		CollisionShape2TriangleMesh(collisionShape,startTrans,vertexPositions,vertexNormals,indices);
		gfxVertices.resize(vertexPositions.size());
		for (int i=0;i<vertexPositions.size();i++)
		{
			for (int j=0;j<4;j++)
			{
				gfxVertices[i].xyzw[j] = vertexPositions[i][j];
			}
			for (int j=0;j<3;j++)
			{
				gfxVertices[i].normal[j] = vertexNormals[i][j];
			}
			for (int j=0;j<2;j++)
			{
				gfxVertices[i].uv[j] = 0.5;//we don't have UV info...
			}
		}
	}


	if (gfxVertices.size() && indices.size())
	{
		int shapeId = registerGraphicsShape(&gfxVertices[0].xyzw[0],gfxVertices.size(),&indices[0],indices.size(),B3_GL_TRIANGLES,-1);
		collisionShape->setUserIndex(shapeId);
	}

}

void BasicDemo::createCollisionObjectGraphicsObject(btCollisionObject* body, const btVector3& color)
{
	if (body->getUserIndex()<0)
	{
		btCollisionShape* shape = body->getCollisionShape();
		btTransform startTransform = body->getWorldTransform();
		int graphicsShapeId = shape->getUserIndex();
		if (graphicsShapeId>=0)
		{
			//	btAssert(graphicsShapeId >= 0);
			//the graphics shape is already scaled
			btVector3 localScaling(1,1,1);

			int graphicsInstanceId = m_renderer->registerGraphicsInstance(graphicsShapeId, startTransform.getOrigin(), startTransform.getRotation(), color, localScaling);
			body->setUserIndex(graphicsInstanceId);
		}
	}
}

void BasicDemo::updateGraphyObj(btCollisionObject* body)
{
	if (body->getUserIndex() >= 0)
	{
		btCollisionShape* shape = body->getCollisionShape();
		btTransform startTransform;
		startTransform.setIdentity();
		int graphicsShapeId = shape->getUserIndex();
		if (graphicsShapeId >=0 )
		{
			//	btAssert(graphicsShapeId >= 0);
			//the graphics shape is already scaled
			btVector3 localScaling(1,1,1);
			int colorIndex = body->getBroadphaseHandle()->getUid() & 3;
			btVector3 color= sColors[colorIndex];
			m_renderer->updateGraphObj(graphicsShapeId ,body->getUserIndex(), startTransform.getOrigin(), startTransform.getRotation(), color, localScaling);
		}
	}
}

int BasicDemo::registerGraphicsShape(const float* vertices, int numvertices, const int* indices, int numIndices,int primitiveType, int textureId)
{
	int shapeId = m_renderer->registerShape(vertices, numvertices,indices,numIndices,primitiveType, textureId);
	return shapeId;
}

  	void BasicDemo::Keyboard(unsigned char key, int x, int y) {
  		// call the base implementation first
  		BulletOpenGLApplication::Keyboard(key, x, y);
  		switch(key) {
  		// Force testing
  		case 'g': 
  			{
  				// if 'g' is held down, apply a force
  				m_bApplyForce = true; 
  				// prevent the box from deactivating
  				m_pBox->GetRigidBody()->setActivationState(DISABLE_DEACTIVATION);
  				break;
  			}
  		// Impulse testing
  		case 'e':
  			{
  				// don't create a new explosion if one already exists
  				// or we haven't released the key, yet
  				if (m_pExplosion || !m_bCanExplode) break;
  				// don't let us create another explosion until the key is released
  				m_bCanExplode = false;
  				// create a collision object for our explosion
  				m_pExplosion = new btCollisionObject();
  				m_pExplosion->setCollisionShape(new btSphereShape(3.0f));
  				// get the position that we clicked
  				RayResult result;
  				Raycast(m_cameraPosition, GetPickingRay(x, y), result, true);
  				// create a transform from the hit point
  				btTransform explodeTrans;
  				explodeTrans.setIdentity();
  				explodeTrans.setOrigin(result.hitPoint);
  				m_pExplosion->setWorldTransform(explodeTrans);
  				// set the collision flag
  				m_pExplosion->setCollisionFlags(btCollisionObject::CF_NO_CONTACT_RESPONSE);
  				// add the explosion trigger to our world
  				s_pWorld->addCollisionObject(m_pExplosion);
  				break;
  			}
  
  		}
  	}
  	
  	void BasicDemo::KeyboardUp(unsigned char key, int x, int y) {
  		// call the base implementation first
  		BulletOpenGLApplication::KeyboardUp(key, x, y);
  		switch(key) {
  		// Force testing
  		case 'g': 
  			{
  				// if 'g' is let go, stop applying the force
  				m_bApplyForce = false; 
  				// allow the object to deactivate again
  				m_pBox->GetRigidBody()->forceActivationState(ACTIVE_TAG); 
  				break;
  			}
  		// Impulse testing
  		case 'e': m_bCanExplode = true; break;
  		}
  	}
  	
  	void BasicDemo::UpdateScene(float dt) {
  		// call the base implementation first
  		BulletOpenGLApplication::UpdateScene(dt);
  return;
  		// Force testing
  		if (m_bApplyForce) {
  			if (!m_pBox) return;
  			// apply a central upwards force that exceeds gravity
  			m_pBox->GetRigidBody()->applyCentralForce(btVector3(0, 20, 0));
  		}
  
  		// Impulse testing
  		if (m_pExplosion) {
  			// destroy the explosion object after one iteration
  			s_pWorld->removeCollisionObject(m_pExplosion);
  			delete m_pExplosion;
  			m_pExplosion = 0;
  		}
	}


