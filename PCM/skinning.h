#ifndef _SKINNING_H
#define _SKINNING_H
#include "QGLViewer/vec.h"
#include "QGLViewer/quaternion.h"
class Vertex;
namespace PCM
{
	class RigidTransform;
	class SkinWeight;
	class Skining
	{
	public:
		static void caculateOneSample( std::vector<qglviewer::Vec>& final_vertices , const std::vector<qglviewer::Vec>& ori_vertices ,const std::vector<RigidTransform>& rts ,const SkinWeight& sk)
		{
			int numVertices;
			int numBones;
			final_vertices.resize(ori_vertices.size());
			for( int i = 0 ; i < numVertices; ++ i)
			{
				for( int j = 0 ; j < numBones ; j++)
				{
					final_vertices[i] +=  rts[j].rotation()*ori_vertices[i]* sk( i, j) + rts[j].translation();

				}			
			}
		}
	};
	class RigidTransform
	{
	public:
		qglviewer::Quaternion rotation() const
		{
			return m_rotation;
		}
		qglviewer::Vec translation() const
		{
			return m_translation;
		}
	private:
		qglviewer::Quaternion m_rotation;
		qglviewer::Vec		  m_translation;

	};
	class Transforms
	{
	public:
		Transforms( int _numExamp , int _numBone)
			:numExample(_numExamp),numBones(_numBone)
		{
			if( _numExamp >0 &&  _numBone >0)
				boneTrans.resize( numBones * numBones);
		}
		std::vector<RigidTransform> boneTrans;
		RigidTransform operator()( int i ,int j) const
		{
			return boneTrans[ i*numBones+j];
		}
	private:
		int numExample;
		int numBones;
	};
	

	class SkinWeight
	{
	public:
		SkinWeight( int _numVertices , int _numBoneIndix )
			:numVertices(_numVertices),numIndices(_numBoneIndix)
		{
			if( numVertices >0 &&  numIndices >0)
			{
				weight.resize( numVertices*numIndices );
				index.resize( numVertices*numIndices );
			}

		}
		float operator()( int i_vertice ,int i_bone) const
		{
			
			return findIFboneInVertice( i_vertice, i_bone);
		}
		float findIFboneInVertice( int i_vertice ,int i_bone) const
		{
			for (int i = 0; i < numIndices; i++)
			{
				if( index[i_vertice * numIndices + i]  == i_bone)
					return weight[i_vertice * numIndices + i];
			}
			return 0.0f;

		}
	private:
		int numVertices;
		int numIndices;
		//! Skinning weight (number of vertices x index number)
		std::vector<float> weight;
		//! Index (number of vertices x number of indexes)
		std::vector<int> index;
		int index; //number of indexes 
	};
}






#endif