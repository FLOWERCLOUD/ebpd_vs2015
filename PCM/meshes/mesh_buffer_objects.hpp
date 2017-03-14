#ifndef MESH_BUFFER_OBJECTS_HPP__
#define MESH_BUFFER_OBJECTS_HPP__

#include "toolbox/maths/vec2.hpp"
#include "toolbox/maths/vec3.hpp"
#include "toolbox/maths/vec4.hpp"
#include "toolbox/gl_utils/glbuffer_object.hpp"

class Mesh;


/**
 * @class Mesh_gl
 * @brief Build Opengl buffer objects from a static mesh
 *
 * @note OpenGL context must be initiallized before creating a Mesh_gl object
 */
struct Mesh_gl {

    // Todo we should build from Mesh_unpack
    Mesh_gl( Mesh& m );

    //Mesh_gl( const Mesh_gl& mesh_gl ){ }

    ~Mesh_gl();


    void update_vertex_buffer_object();

    //TODO:
    // init_triangles()
    // init_vertices()
    // etc for each buffer

    /// Set the buffer object with an array of normals.
    void set_normals(const std::vector<Tbx::Vec3>& normals) {
        _normals_bo.set_data(normals, GL_STATIC_DRAW);
    }

    /// Set the buffer object with an array of tangents.
    void set_tangents(const std::vector<Tbx::Vec3>& tangents) {
        _tangents_bo.set_data(tangents, GL_STATIC_DRAW);
    }

    void resize_tris(int nb_tris) {
        _index_bo_tri.set_data(nb_tris, 0, GL_STATIC_DRAW);
    }

    void alloc_gl_buffer_objects();

    Mesh& _mesh; // <- todo to be deleted

    bool _registered; ///< Does buffer objects are cuda registered

    /// @note every buffer objects are registered in cuda context
    /// @name Buffer object data
    /// @brief size of these buffers are 'size_unpacked_vert_array'
    /// @{
    Tbx::GlBuffer_obj<Tbx::Vec3> _vbo;
    Tbx::GlBuffer_obj<Tbx::Vec3> _normals_bo;
    Tbx::GlBuffer_obj<Tbx::Vec3> _tangents_bo;
    Tbx::GlBuffer_obj<Tbx::Vec4> _color_bo;
    Tbx::GlBuffer_obj<Tbx::Vec2> _tex_bo;
    /// color of mesh's points when points are displayed
    Tbx::GlBuffer_obj<Tbx::Vec4> _point_color_bo;
    /// @}

    /// @name Buffer object index
    /// @{
    /// Size of this buffer is 'nb_tri' * 3
    Tbx::GlBuffer_obj<int> _index_bo_tri;
    /// Size of this buffer is 'nb_quad' * 4
    Tbx::GlBuffer_obj<int> _index_bo_quad;
    /// Size of this buffer is 'nb_vert'
    Tbx::GlBuffer_obj<int> _index_bo_point;
    /// @}
};

#endif // MESH_BUFFER_OBJECTS_HPP__
