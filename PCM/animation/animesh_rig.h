#ifndef _ANIMESH_RIG_
#define _ANIMESH_RIG_

#include "toolbox/maths/vec3.hpp"
#include "toolbox/maths/transfo.hpp"
#include "toolbox/maths/dual_quat_cu.hpp"
#include <vector>
#include <map>
using namespace Tbx;
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
		const int* nearest_bone);

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
		const std::vector<std::map<int, float> >& h_weights );

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
		const std::vector<Tbx::Dual_quat_cu>& d_transform,
		const std::vector<std::map<int, float> >& h_weights);

	void transform_arap_dual_quat( std::vector<Tbx::Mat3>& rots,
		int nb_verts,
		const std::vector<Tbx::Dual_quat_cu>& dual_quat,
		const std::vector<std::map<int, float> >& h_weights);
}


#endif