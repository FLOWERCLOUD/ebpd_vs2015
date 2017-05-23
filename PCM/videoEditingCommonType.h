#pragma once
#include "qdatastream.h"
namespace videoEditting
{
	// ����ṹ��¼��obj�ļ���ȡ�õ�������
	struct ObjTriangle
	{
		unsigned int vertIndex[3];			//vertex index
		unsigned int texcoordIndex[3];		//texture coordinate index
		unsigned int norIndex[3];			//normal index
		unsigned int mtlIndex;				//material index
		friend QDataStream& operator<<(QDataStream& out, const ObjTriangle&tri);
		friend QDataStream& operator<<(QDataStream& in, ObjTriangle&tri);
	};
}