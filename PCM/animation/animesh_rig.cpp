#include "animesh_rig.h"
#include <iostream>

namespace Animesh_kers
{
	void transform_rigid(const std::vector<Tbx::Vec3>& in_verts,
		const std::vector<Tbx::Vec3>& in_normals,
		int nb_verts,
		std::vector<Tbx::Vec3>& out_verts,
		std::vector<Tbx::Vec3>& out_verts2,
		std::vector<Tbx::Vec3>& out_normals,
		const std::vector<Tbx::Transfo>& transfos,
		Mat3 *rots,
		const int* nearest_bone)
	{
		out_verts.resize( nb_verts);
		out_verts2.resize(nb_verts);
		out_normals.resize(nb_verts);

		for( int i = 0 ; i< in_verts.size(); ++i)
		{
			Transfo t = transfos[ nearest_bone[i] ];
			Vec3 vi = t*in_verts[i];
			out_verts[i] = vi;
			out_verts2[i] = vi;
			out_normals[i] = t.fast_invert().transpose()*in_normals[i];
		}
	}
	/// Transform each vertex with SSD
	/// @param in_verts : vertices in rest position
	/// @param out_verts : animated vertices
	/// @param out_verts_2 : animated vertices (same as out_verts)
	void transform_SSD(const std::vector<Tbx::Vec3>& in_verts,
		const std::vector<Tbx::Vec3>& in_normals,
		int nb_verts,
		std::vector<Tbx::Vec3>& out_verts,
		std::vector<Tbx::Vec3>& out_verts2,
		std::vector<Tbx::Vec3>& out_normals,
		const std::vector<Tbx::Transfo>& transfos,
		const std::vector<std::map<int, float> >& h_weights )
	{
		//for (int i = 0; i < transfos.size(); i++)
		//{
		//	const Tbx::Transfo& tr = transfos[i];
		//	 Vec3& trans = tr.get_translation();
		//	 std::cout<<"bone "<<i<<" tr "<< trans.x<<" "<<trans.y<<" "<<trans.z<<std::endl;
		//}


		out_verts.resize( nb_verts);
		out_verts2.resize(nb_verts);
		out_normals.resize(nb_verts);
		for (int i_vtx = 0; i_vtx < in_verts.size(); ++i_vtx)
		{
			Transfo t;
			auto iter = h_weights[i_vtx].begin();
			t = (  h_weights[i_vtx].size() > 0) ? transfos[ iter->first ] * iter->second:
				Transfo::identity();
			for (int i_bone = 0; i_bone < h_weights[i_vtx].size(); ++i_bone ,++iter)
			{	
				if ( 0 == i_bone )
				{
					continue;
				}
				const int   k = iter->first;
				const float w = iter->second;
				t = t + transfos[k] * w;
			}
			Vec3 vi = (t * in_verts[i_vtx].to_point3()).to_vec3();

			out_verts[i_vtx] = vi;
			out_verts2[i_vtx] = vi;
			out_normals[i_vtx] = t.fast_invert().transpose()*in_normals[i_vtx];
		}







	}
	/// Transform each vertex with dual quaternions
	/// @param in_verts : vertices in rest position
	/// @param out_verts : animated vertices
	/// @param out_verts_2 : animated vertices (same as out_verts)
	void transform_dual_quat(const std::vector<Tbx::Vec3>& in_verts,
		const std::vector<Tbx::Vec3>& in_normals,
		int nb_verts,
		std::vector<Tbx::Vec3>& out_verts,
		std::vector<Tbx::Vec3>& out_verts2,
		std::vector<Tbx::Vec3>& out_normals,
		const std::vector<Dual_quat_cu>& dual_quat,
		const std::vector<std::map<int, float> >& h_weights)
	{

		out_verts.resize( nb_verts);
		out_verts2.resize(nb_verts);
		out_normals.resize(nb_verts);

		for (int i_vtx = 0; i_vtx < in_verts.size(); ++i_vtx)
		{
			int   k0 = -1;
			float w0 = 0.f;
			Dual_quat_cu dq_blend;
			Quat_cu q0;

			if( h_weights[i_vtx].size() )
			{
				k0 = h_weights[i_vtx].begin()->first;
				w0 = h_weights[i_vtx].begin()->second;
			}else
			{
				std::cout<<"weight empty "<<i_vtx<<std::endl;
				dq_blend = Dual_quat_cu::identity();
				k0 = 0;
				w0 = 1.0f;
			}

			if( k0 != -1)
				dq_blend =dual_quat[k0]*w0;
			int pivot = k0;
			q0 = dual_quat[pivot].rotation();
			auto iter = h_weights[i_vtx].begin();
			for (int i_bone = 0; i_bone < h_weights[i_vtx].size(); ++i_bone ,++iter)
			{	
				if ( 0 == i_bone )
				{
					continue;
				}
				int bone_id = iter->first;
				float w = iter->second;
				const Dual_quat_cu& dq = (bone_id == -1) ? Dual_quat_cu::identity() : dual_quat[bone_id];
				// Seek shortest rotation:
				if( dq.rotation().dot( q0 ) < 0.f )
					w *= -1.f;
				dq_blend = dq_blend + dq * w;
			}
			Point3 p(in_verts[i_vtx].x,in_verts[i_vtx].y,in_verts[i_vtx].z);
			Vec3 vi = dq_blend.transform( p);
			out_verts[i_vtx] = vi;
			out_verts2[i_vtx] = vi;
			out_normals[i_vtx] = dq_blend.rotate( in_normals[i_vtx]);
		}



	}
	void transform_arap_dual_quat( std::vector<Tbx::Mat3>& rots,
		int nb_verts,
		const std::vector<Dual_quat_cu>& dual_quat,
		const std::vector<std::map<int, float> >& h_weights)
	{

	}
}
