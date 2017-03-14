#ifndef __ANIMESH_COLOR__
#define __ANIMESH_COLOR__

#include "toolbox/maths/vec4.hpp"
#include "meshes/mesh_types.hpp"
#include <vector>
#include <map>
namespace Animesh_colors {
	//void ssd_weights_colors_kernel( std::vector<Tbx::Vec4>& d_colors,
	//							   const EMesh::Packed_data* d_map, 
	//							   int joint_id,
	//							   std::vector<int> d_jpv,
	//							   std::vector<float> d_weights,
	//							   std::vector<int> d_joints );
	void ssd_weights_colors_kernel(
		std::vector<Tbx::Vec4>& d_colors, 
		const std::vector<EMesh::Packed_data>& d_map,
		int joint_id,
		const std::vector<std::map<int, float> >& h_weights
		);
	void base_potential_colors_kernel(
		std::vector<Tbx::Vec4>& d_colors,
		const std::vector<EMesh::Packed_data>& d_map,
	    const std::vector<float >& d_base_potential);

}
#endif