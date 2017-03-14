#pragma  once
#include "BulletOpenGLApplication.h"
#include "btBulletDynamicsCommon.h"

#define EXPLOSION_STRENGTH 50.0f
class PaintCanvas;
class BasicDemo : public BulletOpenGLApplication {
public:
   	BasicDemo(PaintCanvas* qgl);
	void Initialize();
	virtual void InitializePhysics() override;
	virtual void ShutdownPhysics() override;
	void createEmptyDynamicsWorld();
	void CreateObjects();
	void CreateObjects2();
	btBoxShape* createBoxShape(const btVector3& halfExtents);
	btRigidBody*	createRigidBody(float mass, const btTransform& startTransform, btCollisionShape* shape,  const btVector4& color = btVector4(1, 0, 0, 1),int user_idex = -1,btMotionState* motion_state =NULL);
  	void autogenerateGraphicsObjects(btDynamicsWorld* rbWorld); 
	void createCollisionShapeGraphicsObject(btCollisionShape* collisionShape);
	void createCollisionObjectGraphicsObject(btCollisionObject* body, const btVector3& color);
	void updateGraphyObj(btCollisionObject* body);
	int registerGraphicsShape(const float* vertices, int numvertices, const int* indices, int numIndices,int primitiveType, int textureId);
	virtual void Keyboard(unsigned char key, int x, int y) override;
  	virtual void KeyboardUp(unsigned char key, int x, int y) override;
  	virtual void UpdateScene(float dt);

	virtual void CollisionEvent(btRigidBody* pBody0, btRigidBody* pBody1) override;

protected:
	// our box to lift
	GameObject* m_pBox;

	// a simple trigger volume
	btCollisionObject* m_pTrigger;

  		// keeps track of whether we're holding down the 'g' key
  	bool m_bApplyForce;

  		// explosion variables
  	btCollisionObject* m_pExplosion;
  	bool m_bCanExplode;

	btAlignedObjectArray<btCollisionShape*>	m_collisionShapes;
	btAlignedObjectArray<btCollisionObject*> m_RigidBodys;

};