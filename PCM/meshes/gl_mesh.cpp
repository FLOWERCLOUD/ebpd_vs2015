#include "gl_mesh.hpp"

#include "toolbox/gl_utils/glbuffer_object.hpp"
#include "toolbox/gl_utils/glsave.hpp"

#include <iostream>

using namespace Tbx;
// -----------------------------------------------------------------------------

void Gl_mesh_quad::init_bo()
{
    _vertex_bo     = new GlBuffer_obj<float>(GL_ARRAY_BUFFER);
    _index_bo_quad = new GlBuffer_obj<int>(GL_ELEMENT_ARRAY_BUFFER);
}

// -----------------------------------------------------------------------------

Gl_mesh_quad::Gl_mesh_quad()
{
    init_bo();
}

// -----------------------------------------------------------------------------

Gl_mesh_quad::~Gl_mesh_quad()
{
    delete _vertex_bo;
    delete _index_bo_quad;

    _vertex_bo     = 0;
    _index_bo_quad = 0;
}

// -----------------------------------------------------------------------------

Gl_mesh_quad::Gl_mesh_quad(const std::vector<float>& points,
                           const std::vector<int>& quad_idx)
{
    init_bo();
    set_points( &(points  [0]), points.  size() );
    set_index ( &(quad_idx[0]), quad_idx.size() );
}

// -----------------------------------------------------------------------------

void Gl_mesh_quad::set_points(const float* points, int size)
{
    _points.clear();
    _points.resize( size );

    for(int i = 0; i < size; i++)
        _points[i] = points[i];
}

// -----------------------------------------------------------------------------

void Gl_mesh_quad::set_index(const int* quad_idx, int size)
{
    _quad_idx.clear();
    _quad_idx.resize( size );

    for(int i = 0; i < size; i++)
        _quad_idx[i] = quad_idx[i];
}

// -----------------------------------------------------------------------------

/// initialize GPU memory (VBO, VAO ...)
void Gl_mesh_quad::compileGl()
{
     _vertex_bo->set_data(_points, GL_STATIC_DRAW);
    _index_bo_quad->set_data(_quad_idx, GL_STATIC_DRAW);
}

// -----------------------------------------------------------------------------

void Gl_mesh_quad::draw() const
{
    GLEnabledSave save_tex(GL_TEXTURE_2D, true, false);

    glAssert( glEnableClientState(GL_VERTEX_ARRAY) );
    _vertex_bo->bind();
    glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );

    _index_bo_quad->bind();
    glAssert( glDrawElements(GL_QUADS, _quad_idx.size(), GL_UNSIGNED_INT, 0) );
    _index_bo_quad->unbind();

    _vertex_bo->unbind();
    glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );
    glAssert( glDisableClientState(GL_VERTEX_ARRAY) );
}

// -----------------------------------------------------------------------------
