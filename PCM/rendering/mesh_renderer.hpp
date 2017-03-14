#ifndef MESH_RENDERER_HPP__
#define MESH_RENDERER_HPP__

#include "toolbox/maths/vec3.hpp"

class Mesh;
namespace Tbx {
template<class T>
class GlBuffer_obj;
}

// =============================================================================
namespace Mesh_renderer {
// =============================================================================

/// Draw the animated character
void draw(const Mesh& mesh,
          const Tbx::GlBuffer_obj<Tbx::Vec3>& vbo,
          const Tbx::GlBuffer_obj<Tbx::Vec3>& nbo,
          bool enable_texture);

}// END Mesh_renderer ==========================================================


#endif // MESH_RENDERER_HPP__
