#pragma once
#include "BasicDemo.h"
class RigidbodyDemo :public BasicDemo 
{
public:
	friend class MyConvexDecomposition;
	enum ObjToRigidBodyOptionsEnum
	{
		ObjUseConvexHullForRendering=1,
		OptimizeConvexObj=2,
		ComputePolyhedralFeatures=4,
	};
	RigidbodyDemo(PaintCanvas* qgl ,const std::string _filename);
	~RigidbodyDemo();
	virtual void InitializePhysics() override;
	virtual void ShutdownPhysics() override;
	virtual void UpdateScene(float dt);
protected:
	void CreateObjObject();
	void CreateObjObject2();
	void CreateObjObjectConvexDecomp();
	void CreateGimpactObject();
	int loadAndRegisterMeshFromFile2(const std::string& fileName);

	std::string m_fileName;
	int m_options;
	btAlignedObjectArray<btTriangleMesh*> m_trimeshes;
};