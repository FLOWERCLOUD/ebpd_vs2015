#include <stdio.h>
#include <string.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <limits>
#include <deque>
#include <set>
#include <map>

#include "toolbox/gl_utils/glsave.hpp"
#include "mesh.hpp"
#include "mesh_utils.hpp"
#include "toolbox/utils.hpp"
#include "toolbox/std_utils/set.hpp"
	using namespace Tbx;
// -----------------------------------------------------------------------------

Mesh::Mesh() :
    _is_initialized(false),
    _has_tex_coords(false),
    _has_normals(false),
    _has_materials(false),
    _has_bumpmap(false),
    _offset(0.f,0.f,0.f),
    _scale(1.f),
    _mesh_attr( *this ),
    _mesh_gl( *this )
{
}

// -----------------------------------------------------------------------------

Mesh::Mesh(const std::vector<EMesh::Vert_idx>& tri, const std::vector<float>& vert) :
    _is_initialized(true),
    _has_tex_coords(false),
    _has_normals(false),
    _has_materials(false),
    _has_bumpmap(false),
    _offset(0.f,0.f,0.f),
    _scale(1.f),
    _mesh_attr( *this ),
    _mesh_gl( *this )
{
    _mesh_attr._size_unpacked_verts = vert.size() / 3;
    _mesh_static.resize_tris( tri.size() );
    _mesh_static.resize_vertices( vert.size() / 3 );

    if(get_nb_vertices() > 0)
    {
        _mesh_attr._packed_vert_map.resize( get_nb_vertices() );
    }

    for(int i = 0; i < get_nb_vertices(); i++)
    {
        EMesh::Packed_data d = {i, 1};
        _mesh_attr._packed_vert_map[i] = d;

        Tbx::Vec3 p(vert[i*3  ], vert[i*3+1], vert[i*3+2]);
        _mesh_static.set_vertex(i, p);
    }

    // Copy to packed and unpacked arrays :
    if( _mesh_static.get_nb_tris() > 0)
    {
        _mesh_attr._unpacked_tri.resize( _mesh_static.get_nb_tris()  * 3 );

        for(int i = 0; i < _mesh_static.get_nb_tris(); i++)
        {
            EMesh::Tri_face face(tri[i*3], tri[i*3 + 1], tri[i*3 + 2]);
            _mesh_static.set_tri(i, face);
            _mesh_attr.set_unpacked_tri(i, face);
        }
    }

    _mesh_gl.alloc_gl_buffer_objects();
    _mesh_he.update(_mesh_static);
    compute_normals();
}

// -----------------------------------------------------------------------------

Mesh::Mesh(const Mesh& m) :
    _is_initialized(m._is_initialized),
    _has_tex_coords(m._has_tex_coords),
    _has_normals(m._has_normals),
    _has_materials(m._has_materials),
    _has_bumpmap(m._has_bumpmap),
    _offset(m._offset),
    _scale(m._scale),
    _mesh_static( m._mesh_static ),
    _mesh_he( m._mesh_he ),
    _mesh_attr( m._mesh_attr ),
    _mesh_gl( *this )
{
    _mesh_gl.alloc_gl_buffer_objects();
}

// -----------------------------------------------------------------------------

Mesh::Mesh(const char* filename) :
    _is_initialized(false),
    _has_tex_coords(false),
    _has_normals(false),
    _has_materials(false),
    _has_bumpmap(false),
    _offset(0.f,0.f,0.f),
    _scale(1.f),
    _mesh_attr( *this ),
    _mesh_gl( *this )
{
    _mesh_static.import_off( filename );

    _mesh_attr._size_unpacked_verts = _mesh_static.get_nb_vertices();
    _mesh_attr._packed_vert_map.resize( _mesh_static.get_nb_vertices() );

    for(int i = 0; i < _mesh_static.get_nb_vertices(); i++) {
        EMesh::Packed_data d = {i, 1};
        _mesh_attr._packed_vert_map[i] = d;
    }

    // Copy to packed and unpacked arrays :
    if( _mesh_static.get_nb_tris() > 0){
        _mesh_attr._unpacked_tri.resize( _mesh_static.get_nb_tris() * 3 );
        for(int i = 0; i < _mesh_static.get_nb_tris(); i++)
            for(int j = 0; j < 3; ++j) {
                _mesh_attr._unpacked_tri[i*3 + j] = _mesh_static.get_tri(i)[j];
            }
    }

    if( _mesh_static.get_nb_quads() > 0){
        _mesh_attr._unpacked_quad.resize( _mesh_static.get_nb_quads() * 4 );
        for(int i = 0; i < _mesh_static.get_nb_quads(); i++)
            for(int j = 0; j < 4; ++j) {
                _mesh_attr._unpacked_quad[i*4 + j] = _mesh_static.get_quad(i)[j];
            }
    }

    _mesh_he.update( _mesh_static );
    compute_normals();
    _mesh_gl.alloc_gl_buffer_objects();

    _is_initialized = true;
}

// -----------------------------------------------------------------------------

Mesh::~Mesh(){
    clear_data();
}

// -----------------------------------------------------------------------------

void Mesh::clear_data()
{
    _is_initialized = false;

    _mesh_static.clear_data();
    _mesh_attr.clear_data();
    _mesh_he.clear_data();
}

// -----------------------------------------------------------------------------

void Mesh::center_and_resize(float max_size)
{
    float xmin, ymin, zmin;
    float xmax, ymax, zmax;
    xmin = ymin = zmin =  std::numeric_limits<float>::infinity();
    xmax = ymax = zmax = -std::numeric_limits<float>::infinity();

    for(int i = 0; i < get_nb_vertices(); i++)
    {
        Tbx::Vec3 p = _mesh_static.get_vertex(i);
        //printf("%f %f %f\n",x,y,z);
        xmin = (p.x < xmin)? p.x : xmin;
        xmax = (p.x > xmax)? p.x : xmax;
        ymin = (p.y < ymin)? p.y : ymin;
        ymax = (p.y > ymax)? p.y : ymax;
        zmin = (p.z < zmin)? p.z : zmin;
        zmax = (p.z > zmax)? p.z : zmax;
    }
    float dx = xmax - xmin;
    float dy = ymax - ymin;
    float dz = zmax - zmin;
    float du = (dx > dy)?((dx > dz)?dx:dz):((dy>dz)?dy:dz);
    float scale_f = max_size / du;
    _scale = scale_f;
    _offset.x = - 0.5f * (xmax + xmin);
    _offset.y = - 0.5f * (ymax + ymin);
    _offset.z = - 0.5f * (zmax + zmin);
    for(int i = 0; i < get_nb_vertices(); i++)
    {
        Vec3 p = _mesh_static.get_vertex( i );
        p = (p + _offset) * scale_f;
        _mesh_static.set_vertex(i, p);
    }

    // update the vbo
    _mesh_gl.update_vertex_buffer_object();
}

// -----------------------------------------------------------------------------

#include "toolbox/maths/color.hpp"

void Mesh::debug_draw_edges() const
{

#if 0
    ///////////
    // Draw normals using 1st ring neighborhood
    //static int i = 0;

    glBegin(GL_LINES);
    for(int i = 0; i < _mesh.get_nb_vertices(); i++)
    {
        Vec3 v0 = _mesh.get_vertex( i );
        int dep      = _mesh.get_1st_ring_offset(i*2    );
        int nb_neigh = _mesh.get_1st_ring_offset(i*2 + 1);
        int end      = dep + nb_neigh;
        for(int n = dep; n < (end-1); n++)
        {
            int neigh = _mesh.get_1st_ring(n);
            int next  = _mesh.get_1st_ring( (n+1) >= end  ? dep : n+1 );

            Vec3 v1 = _mesh.get_vertex( neigh );
            Vec3 v2 = _mesh.get_vertex( next );

            Vec3 n = (v1 - v0).cross(v2 - v0).normalized();

            Color::pseudo_rand( i ).set_gl_state();
            Vec3 cog = (v0 + v1 + v2) / 3.f;
            glVertex3f(cog.x, cog.y, cog.z);
            cog += n * 0.1f;
            glVertex3f(cog.x, cog.y, cog.z);
        }
    }
    glEnd();
    //i = (i+1) % get_nb_vertices();
#elif 0
    ///////////
    // Draw Triangles edges using the edge list
    static int i = 0;
    static int u = 0;
    glBegin(GL_LINES);
    //for(unsigned i = 0; i < get_nb_tri(); ++i)
    {
        //for(int u = 0; u < 3; ++u)
        {
            const int edge_idx = get_tri_edges(i)[u];
            const Edge e = _edge_list[edge_idx];
            Vec3 v0 = get_vertex( e.a );
            Vec3 v1 = get_vertex( e.b );
            Color::pseudo_rand( i ).set_gl_state();
            glVertex3f(v0.x, v0.y, v0.z);
            glVertex3f(v1.x, v1.y, v1.z);
        }
    }
    glEnd();
    u = (u+1) %3;
    if( u == 0 ){
        i = (i+1) % get_nb_tri();
    }
#elif 0
    ///////////
    // Draw list of edges
    static int i = 0;
    glBegin(GL_LINES);
    for(unsigned i = 0; i < _edge_list.size(); ++i)
    {
        const Edge e = _edge_list[i];
        Vec3 v0 = get_vertex( e.a );
        Vec3 v1 = get_vertex( e.b );
        Color::pseudo_rand( i ).set_gl_state();
        glVertex3f(v0.x, v0.y, v0.z);
        glVertex3f(v1.x, v1.y, v1.z);
    }
    glEnd();
    i = (i+1) % _edge_list.size();
#elif 0
    ///////////
    // Draw 1st ring neighborhood of edges
    static int i = 0;
    glBegin(GL_LINES);
    //for(int i = 0; i < get_nb_vertices(); i++)
    {
        for(unsigned n = 0; n < _edge_list_per_vert[i].size(); n++)
        {
            const Edge e = _edge_list[ _edge_list_per_vert[i][n] ];
            Vec3 v0 = get_vertex( e.a );
            Vec3 v1 = get_vertex( e.b );
            Color::pseudo_rand( i ).set_gl_state();
            glVertex3f(v0.x, v0.y, v0.z);
            glVertex3f(v1.x, v1.y, v1.z);
        }
    }
    glEnd();
    i = (i+1) % get_nb_vertices();
#elif 0
    ///////////
    // Draw 2 ring neighborhood of edges
    static int i = 150;
    static int n = 0;
    glBegin(GL_LINES);
    //for(int i = 0; i < _mesh.get_nb_vertices(); i++)
    {
        Vec3 c = _mesh.get_vertex( i );
        //for(unsigned n = 0; n < _2nd_ring_neigh[i].size(); n++)
        {
            const Vec3 v = _mesh.get_vertex( _2nd_ring_neigh[i][n] );
            Color::pseudo_rand( i ).set_gl_state();
            glVertex3f(c.x, c.y, c.z);
            glVertex3f(v.x, v.y, v.z);
        }
    }
    glEnd();
    n = (n+1) % _2nd_ring_neigh[i].size();
    //i = (i+1) % get_nb_vertices();
#elif 0
    ///////////
    // Draw 1st ring neighborhood of vertices
    static int i = 0;

    glBegin(GL_LINES);
    int dep      = get_1st_ring_offset(i*2    );
    int nb_neigh = get_1st_ring_offset(i*2 + 1);
    for(int n = dep; n < (dep+nb_neigh); n++)
    {
        int neigh = get_1st_ring(n);

        Vec3 v0 = get_vertex( i     );
        Vec3 v1 = get_vertex( neigh );
        Color::pseudo_rand( i ).set_gl_state();

        glVertex3f(v0.x, v0.y, v0.z);
        glVertex3f(v1.x, v1.y, v1.z);
    }
    glEnd();
    i = (i+1) % get_nb_vertices();
#elif 0
    ///////////
    // Draw triangles linked to edges
    static int i = 0;
    glBegin(GL_LINES);
    //for(unsigned i = 0; i < _edge_list.size(); ++i)
    {
        for(unsigned j = 0; j < _tri_list_per_edge[i].size(); ++j)
        {
            const EMesh::Tri_face tri_idx = _mesh.get_tri( _tri_list_per_edge[i][j] );

            Vec3 v0 = _mesh.get_vertex( tri_idx.a );
            Vec3 v1 = _mesh.get_vertex( tri_idx.b );
            Vec3 v2 = _mesh.get_vertex( tri_idx.c );

            Color::pseudo_rand( i ).set_gl_state();

            glVertex3f(v0.x, v0.y, v0.z);
            glVertex3f(v1.x, v1.y, v1.z);

            glVertex3f(v1.x, v1.y, v1.z);
            glVertex3f(v2.x, v2.y, v2.z);

            glVertex3f(v2.x, v2.y, v2.z);
            glVertex3f(v0.x, v0.y, v0.z);
        }
    }
    glEnd();
    i = (i+1) % _edge_list.size();
#endif
}

// -----------------------------------------------------------------------------

void Mesh::draw_using_buffer_object(const GlBuffer_obj<Vec3>& new_vbo,
                                    const GlBuffer_obj<Vec3>& n_bo,
                                    const GlBuffer_obj<Vec4>& c_bo,
                                    bool use_color_array) const
{
    if(get_nb_vertices() == 0 || !_is_initialized) return;

    glAssert( glEnableClientState(GL_VERTEX_ARRAY) );
    new_vbo.bind();
    glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );

    assert(_has_normals);
    n_bo.bind();
    glAssert( glEnableClientState(GL_NORMAL_ARRAY) );
    glAssert( glNormalPointer(GL_FLOAT, 0, 0) );

    // TODO: when use_color_array true disable materials
    if(use_color_array)
    {
        c_bo.bind();
        glAssert( glEnableClientState(GL_COLOR_ARRAY) );
        glAssert( glColorPointer(4,GL_FLOAT,0,0) );
    }

    _mesh_gl._index_bo_tri.bind();
    glAssert( glDrawElements(GL_TRIANGLES, get_nb_tris() * 3, GL_UNSIGNED_INT, 0) );
    _mesh_gl._index_bo_tri.unbind();


    _mesh_gl._index_bo_quad.bind();
    glAssert( glDrawElements(GL_QUADS, 4 * get_nb_quads(), GL_UNSIGNED_INT, 0) );
    _mesh_gl._index_bo_quad.unbind();

    c_bo.unbind();

    glAssert( glBindBuffer(GL_ARRAY_BUFFER, 0) );
    glAssert( glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0) );
    glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );
    glAssert( glNormalPointer(GL_FLOAT, 0, 0) );
    glAssert( glDrawElements(GL_TRIANGLES, 0, GL_UNSIGNED_INT, 0) );
    glAssert( glDisableClientState(GL_VERTEX_ARRAY) );
    glAssert( glDisableClientState(GL_NORMAL_ARRAY) );

    if(use_color_array){
        glAssert( glColorPointer(4,GL_FLOAT,0,0) );
        glAssert( glDisableClientState(GL_COLOR_ARRAY) );
    }
}

// -----------------------------------------------------------------------------

void Mesh::draw_points_using_buffer_object(const GlBuffer_obj<Vec3>& new_vbo,
                                           const GlBuffer_obj<Vec3>& n_bo,
                                           const GlBuffer_obj<Vec4>& c_bo,
                                           bool use_color_array) const
{
    GLEnabledSave save_tex(GL_TEXTURE_2D, true, false);

    glAssert( glEnableClientState(GL_VERTEX_ARRAY) );
    new_vbo.bind();
    glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );

    n_bo.bind();

    glAssert( glEnableClientState(GL_NORMAL_ARRAY) );
    glAssert( glNormalPointer(GL_FLOAT, 0, 0) );

    if(use_color_array){
        c_bo.bind();
        glAssert( glEnableClientState(GL_COLOR_ARRAY) );
        glAssert( glColorPointer(4,GL_FLOAT,0,0) );
    }

    _mesh_gl._index_bo_point.bind();
    glAssert( glDrawElements(GL_POINTS, get_nb_vertices(), GL_UNSIGNED_INT, 0) );
    _mesh_gl._index_bo_point.unbind();

    c_bo.unbind();

    glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );
    glAssert( glNormalPointer(GL_FLOAT, 0, 0) );
    glAssert( glDisableClientState(GL_VERTEX_ARRAY) );
    glAssert( glDisableClientState(GL_NORMAL_ARRAY) );
    if(use_color_array){
        glAssert( glColorPointer(4,GL_FLOAT,0,0) );
        glAssert( glDisableClientState(GL_COLOR_ARRAY) );
    }

}

// -----------------------------------------------------------------------------

void Mesh::draw_points() const{
    draw_points_using_buffer_object(_mesh_gl._vbo, _mesh_gl._normals_bo, _mesh_gl._point_color_bo, true);
}

// -----------------------------------------------------------------------------

void Mesh::draw(bool use_color_array, bool use_point_color) const
{
    if(use_point_color)
        draw_using_buffer_object( _mesh_gl._vbo, _mesh_gl._normals_bo, _mesh_gl._point_color_bo, use_color_array );
    else
        draw_using_buffer_object( _mesh_gl._vbo, _mesh_gl._normals_bo, use_color_array );
}

// -----------------------------------------------------------------------------

void Mesh::draw_using_buffer_object(const GlBuffer_obj<Vec3>& vbo,
                                    const GlBuffer_obj<Vec3>& n_bo,
                                    bool use_color_array) const
{
    draw_using_buffer_object(vbo, n_bo, _mesh_gl._color_bo, use_color_array);
}

// -----------------------------------------------------------------------------

void Mesh::enable_client_state() const
{
    glAssert( glEnableClientState(GL_VERTEX_ARRAY) );
    _mesh_gl._vbo.bind();
    glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );

    if(_has_normals)
    {
        _mesh_gl._normals_bo.bind();
        glAssert( glEnableClientState(GL_NORMAL_ARRAY) );
        glAssert( glNormalPointer(GL_FLOAT, 0, 0) );
    }

    GLEnabledSave save_color_mat(GL_COLOR_MATERIAL, true, true);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    if(_has_tex_coords)
    {
        glAssert( glEnableClientState(GL_TEXTURE_COORD_ARRAY) );
        _mesh_gl._tex_bo.bind();
        glAssert( glTexCoordPointer(2, GL_FLOAT, 0, 0) );
    }
}

// -----------------------------------------------------------------------------

void Mesh::disable_client_state() const
{
    _mesh_gl._vbo.unbind();
    glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );
    glAssert( glDisableClientState(GL_VERTEX_ARRAY) );
    if(_has_normals){
        glAssert( glNormalPointer(GL_FLOAT, 0, 0) );
        glAssert( glDisableClientState(GL_NORMAL_ARRAY) );
    }

    if(_has_tex_coords)
    {
        glAssert( glTexCoordPointer(2, GL_FLOAT, 0, 0) );
        glAssert( glDisableClientState(GL_TEXTURE_COORD_ARRAY) );
    }
}

// -----------------------------------------------------------------------------

void Mesh::compute_normals()
{
    std::vector<Tbx::Vec3> new_normals( get_nb_vertices() );
    _has_normals = true;

    Mesh_utils::normals(*this, new_normals);

    _mesh_attr.set_normals( new_normals );
    _mesh_gl.  set_normals( new_normals );
}

// -----------------------------------------------------------------------------

void Mesh::compute_tangents()
{
    std::vector<Tbx::Vec3> new_tangents;
    _has_normals = true;

    Mesh_utils::tangents(*this, new_tangents);

    _mesh_attr.set_tangents( new_tangents );
    _mesh_gl.  set_tangents( new_tangents );
}

// -----------------------------------------------------------------------------

void Mesh::smoothen_mesh(float smooth_factor,
                         int nb_iter,
                         int nb_min_neighbours)
{

    // TODO check it works for odd and even numbers of iterations
    if(nb_iter == 0) return;

    std::vector<Vec3> vector_verts_buffer( get_nb_vertices() );

    Vec3* vertices_data     = _mesh_static.get_vertices().data();
    Vec3* new_vertices_data = vector_verts_buffer.data();

    std::vector<int> vert_neig( get_nb_vertices() );

    for(int k = 0; k < nb_iter; k++)
    {
        for(int i = 0; i < get_nb_vertices(); i++)
        {
            new_vertices_data[i] = 0.f;
            vert_neig[i] = 0;
        }

        for(int i = 0; i < get_nb_tris(); i++)
        {
            EMesh::Tri_face tri = _mesh_static.get_tri( i );
            new_vertices_data[tri.a] += vertices_data[tri.b] + vertices_data[tri.c];
            new_vertices_data[tri.b] += vertices_data[tri.a] + vertices_data[tri.c];
            new_vertices_data[tri.c] += vertices_data[tri.b] + vertices_data[tri.a];

            vert_neig[tri.a] += 2;
            vert_neig[tri.b] += 2;
            vert_neig[tri.c] += 2;
        }

        for(int i = 0; i < get_nb_quads(); i++)
        {
            EMesh::Quad_face quad =_mesh_static.get_quad(i);
            new_vertices_data[quad.a] += vertices_data[quad.b] + vertices_data[quad.d];
            new_vertices_data[quad.b] += vertices_data[quad.a] + vertices_data[quad.c];
            new_vertices_data[quad.c] += vertices_data[quad.b] + vertices_data[quad.d];
            new_vertices_data[quad.d] += vertices_data[quad.a] + vertices_data[quad.c];

            vert_neig[quad.a] += 2;
            vert_neig[quad.b] += 2;
            vert_neig[quad.c] += 2;
            vert_neig[quad.d] += 2;
        }

        for(int i = 0; i < get_nb_vertices(); i++)
        {
            if(vert_neig[i] > nb_min_neighbours && !is_vert_on_side(i) )
            {
                float iv = 1.f / vert_neig[i];
                new_vertices_data[i] *= iv;
            }
            else
            {
                new_vertices_data[i] = vertices_data[i];
            }

            float u = smooth_factor;
            vertices_data[i] = vertices_data[i] * (1.f - u) + new_vertices_data[i] * u;
        }

        Vec3* tmp = new_vertices_data;
        new_vertices_data = vertices_data;
        vertices_data = tmp;
    }

    if((nb_iter%2) == 0)
        _mesh_static.get_vertices().swap( vector_verts_buffer );

    // update the vbo
    _mesh_gl.update_vertex_buffer_object();
}

// -----------------------------------------------------------------------------

void Mesh::check_integrity()
{
#ifndef NDEBUG
    for(int i = 0; i < get_nb_vertices(); ++i)
    {
        int dep = get_1st_ring_offset(i*2  );
        int off = get_1st_ring_offset(i*2+1);
        // Disconnected vertices do not have edges
        if(is_disconnect(i)) assert( off == 0 );
        //std::cout << off << std::endl;
        for(int j = dep; j < (dep+off); ++j)
        {
            int edge_id = get_1st_ring(j);
            if(edge_id == i){
                assert(false); // no self edges
            }
            // Check edge ids;
            if(edge_id >= 0 && edge_id >= get_nb_vertices()){
                assert(false);
            }

            // Check edge loop integrity
            // (every edge must point to a unique vertex)
            // FIXME: quad pairs are bugged and this condition will fail
/*
            for(int n=dep; n< (dep+off); ++n ) {
                if(n == j) continue;
                int nedge_id = get_1st_ring(n);
                assert(nedge_id != edge_id);
            }
*/
        }
    }
#endif
}

// -----------------------------------------------------------------------------

void Mesh::diffuse_along_mesh(float* vertices_attributes,
                              float locked_value,
                              int nb_iter) const
{
    std::vector<float> new_attribs  ( get_nb_vertices() );
    std::vector<int>   nb_neighbours( get_nb_vertices() );
    for(int k = 0; k < nb_iter; k++)
    {
        for(int i = 0; i < get_nb_vertices(); i++){
            new_attribs[i] = 0.f;
            nb_neighbours[i] = 0;
        }

        for(int i = 0; i < get_nb_tris(); i++)
        {
            EMesh::Tri_face tri = _mesh_static.get_tri(i);
            new_attribs[tri.a] += vertices_attributes[tri.b] + vertices_attributes[tri.c];
            new_attribs[tri.b] += vertices_attributes[tri.c] + vertices_attributes[tri.a];
            new_attribs[tri.c] += vertices_attributes[tri.a] + vertices_attributes[tri.b];
            nb_neighbours[tri.a] += 2;
            nb_neighbours[tri.b] += 2;
            nb_neighbours[tri.c] += 2;
        }
        for(int i = 0; i < get_nb_quads(); i++)
        {
            EMesh::Quad_face quad = _mesh_static.get_quad(i);
            new_attribs[quad.a] += vertices_attributes[quad.b] + vertices_attributes[quad.d];
            new_attribs[quad.b] += vertices_attributes[quad.c] + vertices_attributes[quad.a];
            new_attribs[quad.c] += vertices_attributes[quad.b] + vertices_attributes[quad.d];
            new_attribs[quad.d] += vertices_attributes[quad.c] + vertices_attributes[quad.a];
            nb_neighbours[quad.a] += 2;
            nb_neighbours[quad.b] += 2;
            nb_neighbours[quad.c] += 2;
            nb_neighbours[quad.d] += 2;
        }
        for(int i = 0; i < get_nb_vertices(); i++){
            if(vertices_attributes[i] != locked_value){
                vertices_attributes[i] = new_attribs[i] / nb_neighbours[i];
            }
        }
    }
}

// -----------------------------------------------------------------------------

void Mesh::diffuse_along_mesh(float* vertices_attributes, int nb_iter) const
{
    std::vector<float> new_attribs  ( get_nb_vertices() );
    std::vector<int>   nb_neighbours( get_nb_vertices() );
    for(int k = 0; k < nb_iter; k++)
    {
        for(int i = 0; i < get_nb_vertices(); i++){
            new_attribs  [i] = 0.f;
            nb_neighbours[i] = 0;
        }
        for(int i = 0; i < get_nb_tris(); i++)
        {
            EMesh::Tri_face tri = _mesh_static.get_tri(i);
            new_attribs[tri.a] += vertices_attributes[tri.b] + vertices_attributes[tri.c];
            new_attribs[tri.b] += vertices_attributes[tri.c] + vertices_attributes[tri.a];
            new_attribs[tri.c] += vertices_attributes[tri.a] + vertices_attributes[tri.b];
            nb_neighbours[tri.a] += 2;
            nb_neighbours[tri.b] += 2;
            nb_neighbours[tri.c] += 2;
        }
        for(int i = 0; i < get_nb_quads(); i++)
        {
            EMesh::Quad_face quad = _mesh_static.get_quad(i);
            new_attribs[quad.a] += vertices_attributes[quad.b] + vertices_attributes[quad.d];
            new_attribs[quad.b] += vertices_attributes[quad.c] + vertices_attributes[quad.a];
            new_attribs[quad.c] += vertices_attributes[quad.b] + vertices_attributes[quad.d];
            new_attribs[quad.d] += vertices_attributes[quad.c] + vertices_attributes[quad.a];
            nb_neighbours[quad.a] += 2;
            nb_neighbours[quad.b] += 2;
            nb_neighbours[quad.c] += 2;
            nb_neighbours[quad.d] += 2;
        }
        for(int i = 0; i < get_nb_vertices(); i++)
            vertices_attributes[i] = new_attribs[i] / nb_neighbours[i];
    }
}

// -----------------------------------------------------------------------------

void Mesh::add_noise(int fq, float amp)
{
    for(int i = 0; i < get_nb_vertices(); i++)
    {
        const Vec3 c = get_vertex(i);
        float u = c.x;
        float r = sqrtf(c.y * c.y + c.z * c.z);
        float v = atan2(c.y, c.z);
        float dr = cosf(u*fq/r) * cosf(v*fq) * amp;

        _mesh_static.set_vertex(i, get_vertex(i) + (c * dr));
    }
    // update the vbo
    _mesh_gl.update_vertex_buffer_object();
}

// -----------------------------------------------------------------------------

void Mesh::set_point_color_bo(int i, float r, float g, float b, float a)
{
    assert(i < get_nb_vertices());
    Vec4* color_ptr;
    _mesh_gl._point_color_bo.map_to(color_ptr, GL_WRITE_ONLY);
    const EMesh::Packed_data d = _mesh_attr._packed_vert_map[i];
    for(int j = 0; j < d._nb_ocurrence; j++)
    {
        const int p_idx = d._idx_data_unpacked + j;
        color_ptr[p_idx].set(r, g, b, a);
    }
    _mesh_gl._point_color_bo.unmap();
}

// -----------------------------------------------------------------------------

void Mesh::set_color_bo(float r, float g, float b, float a)
{
    const int size = _mesh_attr._size_unpacked_verts;
    std::vector<Vec4> colors( size, Vec4(r, g, b, a));
    _mesh_gl._color_bo.set_data(size, &colors.front(), GL_STATIC_DRAW);
}

// -----------------------------------------------------------------------------

void Mesh::set_point_color_bo(float r, float g, float b, float a)
{
    const int size = _mesh_attr._size_unpacked_verts;
    if( size < 0 ) return;

    std::vector<Vec4> colors( size, Vec4(r,g,b,a) );
    _mesh_gl._point_color_bo.set_data(size, &colors.front(), GL_STATIC_DRAW);
}

// -----------------------------------------------------------------------------

Vec3 Mesh::get_normal(EMesh::Vert_idx i, int n) const {
    assert(_has_normals);
    EMesh::Packed_data d = _mesh_attr._packed_vert_map[i];
    int idx = d._idx_data_unpacked + n;
    if(d._nb_ocurrence == 0)
        return Vec3(0.f, 0.f, 0.f);
    assert(n < d._nb_ocurrence);
    return Vec3(_mesh_attr._normals[3*idx], _mesh_attr._normals[3*idx+1], _mesh_attr._normals[3*idx+2]);
}

// -----------------------------------------------------------------------------

Vec3 Mesh::get_mean_normal(EMesh::Vert_idx i) const {
    assert(_has_normals);
    EMesh::Packed_data d = _mesh_attr._packed_vert_map[i];
    Vec3 mean(0.f, 0.f, 0.f);
    for (int n = 0; n < d._nb_ocurrence; ++n) {
        int idx = d._idx_data_unpacked + n;
        Vec3 nor(_mesh_attr._normals[3*idx], _mesh_attr._normals[3*idx+1], _mesh_attr._normals[3*idx+2]);
        mean += nor;
    }
    return mean;
}

// -----------------------------------------------------------------------------

EMesh::Tex_coords Mesh::get_tex_coords(EMesh::Vert_idx i, int n) const {
    assert(_has_tex_coords);
    int idx = _mesh_attr._packed_vert_map[i]._idx_data_unpacked + n;
    assert(n < _mesh_attr._packed_vert_map[i]._nb_ocurrence);
    return EMesh::Tex_coords(_mesh_attr._tex_coords[2*idx], _mesh_attr._tex_coords[2*idx+1]);
}

// -----------------------------------------------------------------------------

void Mesh::get_vertices( std::vector<Vec3>& verts ) const
{
    verts.clear();
    verts.resize( get_nb_vertices() );
    for(int i = 0; i < get_nb_vertices(); i++)
        verts[i] = _mesh_static.get_vertex(i);
}

// -----------------------------------------------------------------------------

void Mesh::get_mean_normals( std::vector<Vec3>& normals ) const
{
    normals.clear();
    normals.resize( get_nb_vertices() );
    for(int i = 0; i < get_nb_vertices(); i++)
        normals[i] = get_mean_normal(i);
}