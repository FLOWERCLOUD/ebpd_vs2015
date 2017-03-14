#include "mesh_buffer_objects.hpp"
#include "mesh.hpp"
using namespace Tbx;
// -----------------------------------------------------------------------------

Mesh_gl::Mesh_gl( Mesh& m ) :
	_mesh( m ),
	_registered(false),
	_vbo(GL_ARRAY_BUFFER),
	_normals_bo(GL_ARRAY_BUFFER),
	_tangents_bo(GL_ARRAY_BUFFER),
	_color_bo(GL_ARRAY_BUFFER),
	_tex_bo(GL_ARRAY_BUFFER),
	_point_color_bo(GL_ARRAY_BUFFER),
	_index_bo_tri(GL_ELEMENT_ARRAY_BUFFER),
	_index_bo_quad(GL_ELEMENT_ARRAY_BUFFER),
	_index_bo_point(GL_ELEMENT_ARRAY_BUFFER)
{
}

// -----------------------------------------------------------------------------

//Mesh_gl::Mesh_gl( const Mesh_gl& mesh_gl ){ }

// -----------------------------------------------------------------------------

Mesh_gl::~Mesh_gl(){
	
}

// -----------------------------------------------------------------------------



// -----------------------------------------------------------------------------

void Mesh_gl::update_vertex_buffer_object()
{
	Tbx::Vec3* vert_gpu = 0;
	_vbo.map_to(vert_gpu, GL_WRITE_ONLY);
	for(int i = 0; i < _mesh.get_nb_vertices(); i++)
	{
		const EMesh::Packed_data d = _mesh.get_packed_vert_map()[i];
		for(int j = 0; j < d._nb_ocurrence; j++)
		{
			const int p_idx = d._idx_data_unpacked + j;
			vert_gpu[p_idx] = _mesh.get_vertex( i );
		}
	}
	_vbo.unmap();
}

// -----------------------------------------------------------------------------

void Mesh_gl::alloc_gl_buffer_objects()
{
	// Some of these data from _mesh should be moved in Mesh_gl I think like the unpack map
	int size_unpacked_vert_array = _mesh.get_vbos_size();
	assert(size_unpacked_vert_array != 0);
	assert(_mesh.get_nb_vertices() > 0);

	_vbo.set_data(size_unpacked_vert_array, 0, GL_STATIC_DRAW);
	update_vertex_buffer_object();

	if(_mesh.get_nb_tris() > 0)
		_index_bo_tri. set_data( 3 * _mesh.get_nb_tris() , _mesh._mesh_attr._unpacked_tri.data() , GL_STATIC_DRAW);

	if(_mesh.get_nb_quads() > 0)
		_index_bo_quad.set_data( 4 * _mesh.get_nb_quads(), _mesh._mesh_attr._unpacked_quad.data(), GL_STATIC_DRAW);

	const int size = size_unpacked_vert_array;
	if(_mesh.has_normals()){
		_normals_bo. set_data(size, (Tbx::Vec3*)_mesh._mesh_attr._normals.data(), GL_STATIC_DRAW);
		_tangents_bo.set_data(size, (Tbx::Vec3*)_mesh._mesh_attr._tangents.data(), GL_STATIC_DRAW);
	}

	if( _mesh.has_tex_coords() )
		_tex_bo.set_data(size, (Vec2*)_mesh._mesh_attr._tex_coords.data(), GL_STATIC_DRAW);

	float* colors      = new float[ 4 * size];
	int*   point_index = new int  [ _mesh.get_nb_vertices() ];
	for(int i = 0; i < size; i++)
	{
		colors[i*4] = colors[i*4+1] = colors[i*4+2] = colors[i*4+3] = 1.f;
		if( i < _mesh.get_nb_vertices() )
			point_index[i] = _mesh.get_packed_vert_map()[i]._idx_data_unpacked;
	}
	_index_bo_point.set_data( _mesh.get_nb_vertices(), point_index, GL_STATIC_DRAW);
	_point_color_bo.set_data(size, (Tbx::Vec4*)colors, GL_STATIC_DRAW);

	for(int i = 0; i < size; i++) colors[i*4+3] = 0.99f;

	_color_bo.set_data(size, (Tbx::Vec4*)colors, GL_STATIC_DRAW);

	delete[] colors;
	delete[] point_index;

}
