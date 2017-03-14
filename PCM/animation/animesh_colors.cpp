#include "animesh_colors.h"
#include "toolbox/maths/color.hpp"
#include "meshes/mesh_types.hpp"
#include <iostream>
#include <map>
using namespace Tbx;
using std::cout;
using std::endl;

namespace Animesh_colors {

	void paint(const std::vector<int>&  map, const std::vector<Vec4>& color, std::vector<Vec4>& colors)
	{
		for(int i=0; i < colors.size(); i++) {
			colors[i] = color[map[i] ];
		}
	}
	//void ssd_weights_colors_kernel(std::vector<Tbx::Vec4>& d_colors, 
	//	const EMesh::Packed_data* d_map, 
	//	int joint_id,
	//	std::vector<int> d_jpv, 
	//	std::vector<float> d_weights, 
	//	std::vector<int> d_joints)
	//{


	//	for( int i = 0 ; i < d_joints.size() ; ++i)
	//	{
	//		if( joint_id == d_joints[i])
	//		{
	//			Color c;
	//			float w = d_weights[i];
	//			if(w < 0) c.set(1.f, 1.f,   1.f, 0.99f);
	//			else
	//			{
	//				c = Color::heat_color( w );
	//			}
	//			c = (c + 0.5f) / 1.5f;
	//			d_colors[i] = Vec4(c.r , c.g ,c.b ,0.99f);
	//		}

	//	}

	//}
	void ssd_weights_colors_kernel(std::vector<Tbx::Vec4>& d_colors, 
		const std::vector<EMesh::Packed_data>& d_map,
		int joint_id,
		const std::vector<std::map<int, float> >& h_weights)
	{

		for( int i = 0 ; i < d_map.size() ; ++i)
		{
			int idx = d_map[i]._idx_data_unpacked;
			for (int j = 0; j < d_map[i]._nb_ocurrence; j++)
			{
				Color c;
				//				std::pair<int ,float> pari = h_weights[i].find(joint_id);
				float w  = 0.0f;
				auto iter = h_weights[i].find(joint_id);
				if( iter !=  h_weights[i].end())
				{
					w = iter->second;
				}else
				{
					//cout<<"error: ssd_weights_colors_kernel no bone"<<endl;
					//cout<<"vertex id "<<i<<endl;
				}
				if(w < 0)
				{
					c = Color::heat_color( -w );
					cout<<"error: ssd_weights_colors_kernel w < 0"<<endl;
					cout<<"vertex id "<<i<<endl;
				}
				else
				{
					c = Color::heat_color( w );
				}
				//				c = (c + 0.5f) / 1.5f;
				d_colors[idx + j] = Vec4(c.r , c.g ,c.b ,0.99f);
			}

		}

	}

	void base_potential_colors_kernel(std::vector<Tbx::Vec4>& d_colors,
				const std::vector<EMesh::Packed_data>& d_map,
				const std::vector<float>& d_base_potential)
	{

		for( int i = 0 ; i < d_map.size() ; ++i)
		{
			int idx = d_map[i]._idx_data_unpacked;
			for (int j = 0; j < d_map[i]._nb_ocurrence; j++)
			{
				Color c(0.f, 0.f, 0.f, 0.99f);
				float pot = d_base_potential[i];
				if( pot < 0)
				{
					c.r = c.g = c.b = 0.f;
				}else
				{
					pot = 1.f - max(min((pot - 0.4f) * 5.f, 1.f), 0.f);
					c = Color::heat_color(pot);
				}
				d_colors[idx + j] = Vec4(c.r , c.g ,c.b ,0.99f);
			}

		}

	}

}