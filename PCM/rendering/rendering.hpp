#ifndef __RENDERING_H
#define __RENDERING_H



#include "toolbox/gl_utils/glbuffer_object.hpp"
#include "depth_peeling.hpp"
namespace qglviewer
{
	class Camera;
}
class PaintCanvas;
/// @class Render_context_cu
/// @brief Holds rendering context (image buffers pbos textures) for one viewport
class Render_context {
public:

    struct float4 { float x,y,z,w; };


    Render_context(int w, int h);

    ~Render_context();

    void reshape(int w, int h);
    void allocate(int width, int height);

    /*------------------*
    | Getters & setters |
    *------------------*/

   inline int width()  const { return _width;  }
   inline int height() const { return _height; }

   inline Tbx::GlBuffer_obj<GLint>*  pbo_color(){ return _pbo_color; }
   inline Tbx::GlBuffer_obj<GLuint>* pbo_depth(){ return _pbo_depth; }


   inline Peeler* peeler(){ return _peeler; }

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
    Tbx::GlBuffer_obj<GLint>*  _pbo_color;
    Tbx::GlBuffer_obj<GLuint>* _pbo_depth;

    Peeler* _peeler;
};



class RenderFuncWireframe : public Depth_peeling::Render {
public:

	RenderFuncWireframe(const qglviewer::Camera* cam ,const Render_context* ctx) :
		Depth_peeling::Render(),
		_cam(cam),
		_ctx(ctx)
	{  }

	void f();

	static void render(const qglviewer:: Camera* cam, const Render_context* ctx);

	void draw_transc_objs();

	const qglviewer::Camera* _cam;
	const Render_context* _ctx;
};

bool display_loop(const qglviewer:: Camera* _cam ,Render_context* _render_ctx);

void drawSampleSet( PaintCanvas* _curCanvas);

#endif // !__RENDERING_H