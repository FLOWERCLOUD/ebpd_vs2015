#pragma once
#include "VideoEDMesh.h"
#include "videoEditingCommonType.h"
#include <qstring.h>
#include <qvector.h>
#include <qvector3d.h>
#include <qdatastream.h>
//#include "Primitive.h"
//#include "DisplaySystem.h"
namespace videoEditting
{
	



	struct ObjMaterial
	{
		QString name;				//name of material
		QVector3D diffuse;			//��������ɫ
	};

	class ObjReader
	{
	public:
		enum coordType { VERTCOORDS, TEXCOORDS, NORCOORDS };
		enum indexType { VERTINDICES, TEXINDICES, NORINDICES };

		ObjReader();
		~ObjReader();

		QVector<QSharedPointer<Mesh>>	meshes;
		QVector<ObjMaterial>			materials;

		//��ȡobj�Լ�������mtl�ļ�
		//obj��mtl�ļ�������ͬһĿ¼
		bool read(const QString& fileName);
		QSharedPointer<Mesh> getMesh(int ithMesh) { return meshes[ithMesh]; }
		int  getNumMeshes() { return meshes.size(); }

		//return vertex buffer
		bool getWireArray(int primitive, GLWireArray*vB);

		void showObj();
		void showMtl();
		void clear();

	private:
		//ֻ��ȡobj�ļ�
		bool readObjFile(const QString& fileName);
		//ֻ��ȡmtl�ļ�
		bool readMtlFile(const QString& fileName);
		//��vector���͵���������ת����float��������
		//pArrayΪ����ָ�룬nFloatsΪ����Ԫ�ظ���
		//type��3��ȡֵ��OBJ_VERTICES,OBJ_TEXCOORDS,OBJ_NORMALS
		//��ʾ���ض�Ӧ���͵�����
		//nthPrimitiveָ��������һ�����������
		void getCoordArray(float*&pArray, unsigned int&nFloats,
			unsigned nthPrimitive, coordType type);
		//��vector���͵���������ת����unsigned int��������
		//������getCoordArray���ƣ�nInt����������Ϊ����*3
		//ע������ֵ��3�����Ƕ�Ӧ���㣬������������±�
		//ע������ֵ��2�����Ƕ�Ӧ��ͼ����������±�
		void getIndexArray(unsigned int*&pArray, unsigned int &nInts,
			unsigned nthPrimitive, indexType type);

	};
}

