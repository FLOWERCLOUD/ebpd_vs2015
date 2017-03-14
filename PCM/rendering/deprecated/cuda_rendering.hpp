#ifndef CUDA_RENDERING_HPP__
#define CUDA_RENDERING_HPP__

#include "toolbox/gl_utils/glbuffer_object.hpp"
#include "depth_peeling.hpp"
#include "camera.hpp"

namespace Tbx { class GlTex2D; }

// -----------------------------------------------------------------------------

/// @class Render_context_cu
/// @brief Holds rendering context (image buffers pbos textures) for one viewport
class Render_context_cu {
public:
    #ifndef __CUDACC__
    struct float4 { float x,y,z,w; };
    #endif

    Render_context_cu(int w, int h);

    ~Render_context_cu();

    void reshape(int w, int h);
    void allocate(int width, int height);

    /*------------------*
    | Getters & setters |
    *------------------*/

    int width()  const { return _width;  }
    int height() const { return _height; }

    GlBuffer_obj<GLint>*  pbo_color(){ return _pbo_color; }
    GlBuffer_obj<GLuint>* pbo_depth(){ return _pbo_depth; }

    float4* d_render_buff(){ return _d_rendu_buf;       }
    float*  d_depth_buff (){ return _d_rendu_depth_buf; }

    Tbx::GlTex2D* frame_tex(){ return _frame_tex; }

    Peeler* peeler(){ return _peeler; }

    /*-----------------*
    | Rendering states |
    *-----------------*/
    /// deactivate transparency when true
    bool _plain_phong;

    /// enable textures in plain phong
    bool _textures;

    /// Draw the mesh when true
    bool _draw_mesh;

    /// activate raytracing of implicit surface
    bool _raytrace;

    /// Draw skeleton or graph
    bool _skeleton;

    /// Draw the mesh in rest pose
    bool _rest_pose;

private:
    /*------*
    | Datas |
    *------*/
    int _width, _height;

    /// The pbo that contains the result of the raytracing done by cuda
    GlBuffer_obj<GLint>*  _pbo_color;
    GlBuffer_obj<GLuint>* _pbo_depth;

    Tbx::GlTex2D* _frame_tex;

    float4* _d_img_buffer;
    float4* _d_bloom_buffer;
    float4* _d_rendu_buf;
    float*  _d_rendu_depth_buf;

    Peeler* _peeler;
};

// -----------------------------------------------------------------------------

class RenderFuncWireframe : public Depth_peeling::Render {
public:

    RenderFuncWireframe(const Camera* cam, const Render_context_cu* ctx) :
        Depth_peeling::Render(),
        _cam(cam),
        _ctx(ctx)
    {  }

    void f();

    static void render(const Camera* cam, const Render_context_cu* ctx);

    void draw_transc_objs();

    const Camera* _cam;
    const Render_context_cu* _ctx;
};

// -----------------------------------------------------------------------------

/// GLUT's display function
/// @return wether 'display_loop()' needs to be call again to complete the
/// rendering
bool display_loop(Render_context_cu* ctx, const Camera* cam);

#endif // CUDA_RENDERING_HPP__
