#ifndef GL_MESH_HPP__
#define GL_MESH_HPP__

#include <vector>
#include "toolbox/portable_includes/port_glew.h"

namespace Tbx {
template<typename T>
class GlBuffer_obj;
}

/** @brief Utility to display quad mesh with Opengl

*/
class Gl_mesh_quad {
public:

    Gl_mesh_quad();

    ~Gl_mesh_quad();

    /// Build the gl mesh from a vector of points and quad index
    /// @param points   List of points with contigus coordinates (x,y,z)
    /// @param quad_idx List of quad faces
    Gl_mesh_quad(const std::vector<float>& points,
                 const std::vector<int>& quad_idx);

    void set_points(const float* points, int size);

    void set_index(const int* quad_idx, int size);

    /// initialize GPU memory (VBO, VAO ...)
    void compileGl();

    void draw() const;


private:
    void init_bo();

    std::vector<float> _points;   ///< list of points with contigus (x,y,z)
    std::vector<int>   _quad_idx; ///< list of quad faces

    Tbx::GlBuffer_obj<float>* _vertex_bo;
    // Possible extenssion to be done:
    //BufferObject<GL_ARRAY_BUFFER> _normals_bo;
    //BufferObject<GL_ARRAY_BUFFER> _color_bo;
    //BufferObject<GL_ARRAY_BUFFER> _tex_bo;

    Tbx::GlBuffer_obj<int>* _index_bo_quad;
};

#endif // GL_MESH_HPP__
