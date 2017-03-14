#ifndef DEPTH_PEELING_HPP__
#define DEPTH_PEELING_HPP__

#include "toolbox/gl_utils/glassert.h"
#include "toolbox/gl_utils/glbuffer_object.hpp"
#include "toolbox/gl_utils/shader.hpp"

namespace Tbx {
template<class T>
class GlBuffer_obj;
}

// =============================================================================
namespace Depth_peeling{
// =============================================================================

class Render{
public:
    virtual ~Render() {}
    /// Method to override in order to do the depth peeling of some objects
    virtual void draw_transc_objs() = 0;
};

}// END Depth_peeling NAMESPACE ================================================

/**
 * These function performs depth peeling so that transparency is rendered in
   the right way
 * The class that performs depth peeling with alpha blending
   @note : hardware antialiasing may not work with this because of
   the use of FBOs
*/
struct Peeler{

    Peeler();
    ~Peeler();

    /// Sets the function used to draw the objects with depth peeling
    void set_render_func(Depth_peeling::Render* r);

    /// Draw a fullscreen quad
    static void draw_quad();

    /// Initialize color and depth buffers prior to depth peeling
    void set_background(int width, int height,
                        const Tbx::GlBuffer_obj<GLint>* pbo_color,
                        const Tbx::GlBuffer_obj<GLuint>* pbo_depth);

    /// Do depth peelign with alpha blending
    void peel(float base_alpha);

    /// Initialize buffers
    void init_depth_peeling(int width, int height);

    /// Initialize buffers again (when window is resized)
    void reinit_depth_peeling(int width, int height);

    /// Erase textures FBO etc.
    void erase_gl_mem();

    // buffers used for depth peeling
    bool   _is_init;
    GLuint _depthTexId[2];
    GLuint _colorTexId[2];
    GLuint _fboId[2];
    GLuint _colorBlenderTexId;
    GLuint _colorBlenderFboId;
    GLuint _backgroundColorTexId;
    GLuint _backgroundDepthTexId;

    Depth_peeling::Render* _renderer;
};

#endif // DEPTH_PEELING_HPP__
