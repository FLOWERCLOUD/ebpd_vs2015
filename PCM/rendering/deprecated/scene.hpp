#ifndef SCENE_HPP__
#define SCENE_HPP__

#include "toolbox/gl_utils/glbuffer_object.hpp"
#include "toolbox/gl_utils/glbuffer_object.hpp"
#include "toolbox/maths/vec3.hpp"

class Mesh;

/// Draw the animated character
void draw_mesh(const Mesh& mesh,
               const GlBuffer_obj<Vec3>& vbo,
               const GlBuffer_obj<Vec3>& nbo,
               bool enable_texture);


#endif // SCENE_HPP__
