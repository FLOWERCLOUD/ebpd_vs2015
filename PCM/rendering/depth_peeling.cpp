#include "toolbox/gl_utils/shader.hpp"
#include "depth_peeling.hpp"
#include "toolbox/gl_utils/glsave.hpp"
#include "global_datas/g_paths.hpp"

#include <cassert>
using namespace Tbx;
// =============================================================================
namespace Depth_peeling{
	// =============================================================================

	Tbx::Shader_prog* g_shader_init  = 0;
	Tbx::Shader_prog* g_shader_peel  = 0;
	Tbx::Shader_prog* g_shader_blend = 0;
	Tbx::Shader_prog* g_shader_final = 0;

	bool is_shaders_init = false;

	/// Helps keeping track of the number of Peeler class instances ->
	/// last one should erase the shader
	int count_peeler_inst = 0;

	// -----------------------------------------------------------------------------

	void delete_shaders()
	{
		delete g_shader_init;
		delete g_shader_blend;
		delete g_shader_peel;
		delete g_shader_final;
		g_shader_init  = 0;
		g_shader_blend = 0;
		g_shader_peel  = 0;
		g_shader_final = 0;

		is_shaders_init = false;
	}

	// -----------------------------------------------------------------------------

	// Init or reload shaders
	void init_shaders()
	{
		delete_shaders();
		Shader vs_init(g_shaders_dir+"/peeling/front_peeling_init_vertex.glsl", GL_VERTEX_SHADER);
		Shader fs_init(g_shaders_dir+"/peeling/front_peeling_init_fragment.glsl", GL_FRAGMENT_SHADER);

		g_shader_init = new Shader_prog(vs_init, fs_init);
		g_shader_init->link();

		Shader vs_blend(g_shaders_dir+"/peeling/front_peeling_blend_vertex.glsl", GL_VERTEX_SHADER);
		Shader fs_blend(g_shaders_dir+"/peeling/front_peeling_blend_fragment.glsl", GL_FRAGMENT_SHADER);

		g_shader_blend = new Shader_prog(vs_blend, fs_blend);
		g_shader_blend->link();

		Shader vs_peel(g_shaders_dir+"/peeling/front_peeling_peel_vertex.glsl", GL_VERTEX_SHADER);
		Shader fs_peel(g_shaders_dir+"/peeling/front_peeling_peel_fragment.glsl", GL_FRAGMENT_SHADER);

		g_shader_peel = new Shader_prog(vs_peel, fs_peel);
		g_shader_peel->link();

		Shader vs_final(g_shaders_dir+"/peeling/front_peeling_final_vertex.glsl", GL_VERTEX_SHADER);
		Shader fs_final(g_shaders_dir+"/peeling/front_peeling_final_fragment.glsl", GL_FRAGMENT_SHADER);

		g_shader_final = new Shader_prog(vs_final, fs_final);
		g_shader_final->link();

		is_shaders_init = true;
	}

}// END Depth_peeling NAMESPACE ================================================

Peeler::Peeler():
	_is_init(false)
{
	Depth_peeling::count_peeler_inst++;
	if(!Depth_peeling::is_shaders_init)
		Depth_peeling::init_shaders();
}

// -----------------------------------------------------------------------------

Peeler::~Peeler(){
	Depth_peeling::count_peeler_inst--;
	if( Depth_peeling::count_peeler_inst == 0 ) // Last instance ? We erase the shader then
		Depth_peeling::delete_shaders();

	erase_gl_mem();
}

// -----------------------------------------------------------------------------

void Peeler::set_render_func(Depth_peeling::Render* r){
	_renderer = r;
}

// -----------------------------------------------------------------------------

void Peeler::draw_quad()
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glBegin(GL_QUADS);
	{
		glTexCoord2f(0.f,0.f);
		glVertex2f(-1.0, -1.0);
		glTexCoord2f(1.f,0.f);
		glVertex2f(1.0, -1.0);
		glTexCoord2f(1.f,1.f);
		glVertex2f(1.0, 1.0);
		glTexCoord2f(0.f,1.f);
		glVertex2f(-1.0, 1.0);
	}
	glEnd();

	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

// -----------------------------------------------------------------------------

void Peeler::set_background(int width, int height,
							const GlBuffer_obj<GLint>* pbo_color,
							const GlBuffer_obj<GLuint>* pbo_depth)
{
	using namespace Depth_peeling;
	GLEnabledSave save_tex(GL_TEXTURE_2D, true, true);

	//glClearColor(0, 0, 0, 1);
	//glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);

	glAssert( glActiveTexture(GL_TEXTURE1));
	pbo_depth->bind();
	glAssert( glBindTexture(GL_TEXTURE_RECTANGLE, _backgroundDepthTexId) );
	glAssert( glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, width, height,
		GL_DEPTH_COMPONENT, GL_FLOAT, 0) );
	//draw_quad();
	pbo_depth->unbind();

	pbo_color->bind();
	glAssert( glBindTexture(GL_TEXTURE_RECTANGLE, _backgroundColorTexId) );
	glAssert( glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, width, height,
		GL_RGBA, GL_UNSIGNED_BYTE, 0) );
	//draw_quad();
	pbo_color->unbind();

	glAssert( glBindTexture(GL_TEXTURE_2D, 0) );
	glAssert( glActiveTexture(GL_TEXTURE0));
}

// -----------------------------------------------------------------------------

void Peeler::peel(float base_alpha)
{
	using namespace Depth_peeling;
	GLEnabledSave save_tex(GL_TEXTURE_2D, true, true);

	// ---------------------------------------------------------------------
	// 1. Initialize Min Depth Buffer
	// ---------------------------------------------------------------------
	glAssert( glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _colorBlenderFboId) );
	glAssert( glDrawBuffer(GL_COLOR_ATTACHMENT0) );

	glAssert( glClearColor(0, 0, 0, 1) );
	glAssert( glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) );

	glAssert( glEnable(GL_DEPTH_TEST) );

	glAssert( g_shader_init->use() );

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_RECTANGLE, _backgroundDepthTexId);
	g_shader_init->set_uniform("BaseDepthTex", 1);

	glActiveTexture(GL_TEXTURE0);

	g_shader_init->set_uniform("Alpha", 0.3f);
	g_shader_init->set_uniform("base_alpha", base_alpha);

	_renderer->draw_transc_objs();
	Shader_prog::unuse();


	// ---------------------------------------------------------------------
	// 2. Depth Peeling + Blending
	// ---------------------------------------------------------------------

	int numLayers = 6;
	for (int layer = 1; layer < numLayers; layer++) {
		int currId = layer % 2;
		int prevId = 1 - currId;

		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _fboId[currId]);
		glDrawBuffer(GL_COLOR_ATTACHMENT0);

		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);


		g_shader_peel->use();
		glBindTexture(GL_TEXTURE_RECTANGLE, _depthTexId[prevId]);
		g_shader_peel->set_uniform("DepthTex", 0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_RECTANGLE, _backgroundDepthTexId);
		g_shader_peel->set_uniform("BaseDepthTex", 1);

		glActiveTexture(GL_TEXTURE0);
		g_shader_peel->set_uniform("base_alpha", base_alpha);

		_renderer->draw_transc_objs();
		Shader_prog::unuse();

		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _colorBlenderFboId);
		glDrawBuffer(GL_COLOR_ATTACHMENT0);

		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);

		glBlendEquation(GL_FUNC_ADD);
		glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);

		g_shader_blend->use();
		glBindTexture(GL_TEXTURE_RECTANGLE, _colorTexId[currId]);
		g_shader_blend->set_uniform("TempTex", 0);
		draw_quad();
		Shader_prog::unuse();

		glDisable(GL_BLEND);


	}

	// ---------------------------------------------------------------------
	// 3. Final Pass
	// ---------------------------------------------------------------------

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glDrawBuffer(GL_BACK);
	glDisable(GL_DEPTH_TEST);

	g_shader_final->use();
	//g_shaderFrontFinal->set_uniform("BackgroundColor", 0.f, 0.f, 0.f);
	glBindTexture(GL_TEXTURE_RECTANGLE, _colorBlenderTexId);
	g_shader_final->set_uniform("ColorTex", 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_RECTANGLE, _backgroundColorTexId);
	g_shader_final->set_uniform("BackgroundTex", 1);

	glActiveTexture(GL_TEXTURE0);
	//g_shaderFrontFinal.setUniform("BackgroundColor", g_backgroundColor, 3);
	//g_shaderFrontFinal.bindTextureRECT("ColorTex", g_frontColorBlenderTexId, 0);

	draw_quad();
	Shader_prog::unuse();

}

// -----------------------------------------------------------------------------

void Peeler::init_depth_peeling(int width, int height)
{
	assert(!_is_init);
	glAssert(glGenTextures(1, &_backgroundColorTexId));
	glAssert(glGenTextures(1, &_backgroundDepthTexId));
	glAssert(glActiveTexture(GL_TEXTURE1));

	glAssert(glBindTexture(GL_TEXTURE_RECTANGLE, _backgroundDepthTexId));
	glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP));
	glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP));
	glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	glAssert(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_DEPTH_COMPONENT32F_NV,
		width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));

	glAssert(glBindTexture(GL_TEXTURE_RECTANGLE, _backgroundColorTexId));
	glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP));
	glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP));
	glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	glAssert(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, width, height,
		0, GL_RGBA, GL_UNSIGNED_BYTE, 0));

	glAssert(glActiveTexture(GL_TEXTURE0));
	/////////////////////////


	glAssert(glGenTextures(2, _depthTexId));
	glAssert(glGenTextures(2, _colorTexId));
	glAssert(glGenFramebuffersEXT(2, _fboId));

	for (int i = 0; i < 2; i++)
	{
		glAssert(glBindTexture(GL_TEXTURE_RECTANGLE, _depthTexId[i]));
		glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP));
		glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP));
		glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST));

		glAssert(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_DEPTH_COMPONENT32F_NV,
			width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));

		glAssert(glBindTexture(GL_TEXTURE_RECTANGLE, _colorTexId[i]));
		glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP));
		glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP));
		glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		glAssert(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, width, height,
			0, GL_RGBA, GL_FLOAT, 0));

		glAssert(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _fboId[i]));
		glAssert(glAssert( glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT,
			GL_TEXTURE_RECTANGLE, _depthTexId[i], 0) ));
		glAssert(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_RECTANGLE, _colorTexId[i], 0));

	}

	glAssert(glGenTextures(1, &_colorBlenderTexId));
	glAssert(glBindTexture(GL_TEXTURE_RECTANGLE, _colorBlenderTexId));
	glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP));
	glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP));
	glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	glAssert(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	glAssert(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, width, height,
		0, GL_RGBA, GL_FLOAT, 0));

	glAssert(glGenFramebuffersEXT(1, &_colorBlenderFboId));
	glAssert(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _colorBlenderFboId));
	glAssert(glAssert( glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT,
		GL_TEXTURE_RECTANGLE, _depthTexId[0], 0) ));
	glAssert(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0,
		GL_TEXTURE_RECTANGLE, _colorBlenderTexId, 0));
	glAssert(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));

	_is_init = true;
}

// -----------------------------------------------------------------------------

void Peeler::reinit_depth_peeling(int width, int height)
{
	if(_is_init) erase_gl_mem();
	init_depth_peeling(width, height);
}

// -----------------------------------------------------------------------------

void Peeler::erase_gl_mem()
{
	assert(_is_init);
	glDeleteTextures(2, _depthTexId);
	glDeleteTextures(2, _colorTexId);
	glDeleteFramebuffersEXT(2, _fboId);
	glDeleteTextures(1, &_colorBlenderTexId);
	glDeleteTextures(1, &_backgroundColorTexId);
	glDeleteTextures(1, &_backgroundDepthTexId);
	glDeleteFramebuffersEXT(1, &_colorBlenderFboId);
	_is_init = false;
}

// -----------------------------------------------------------------------------

