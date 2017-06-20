#pragma once
#include "qdatastream.h"
namespace videoEditting
{
	// 这个结构记录从obj文件读取得到的数据
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