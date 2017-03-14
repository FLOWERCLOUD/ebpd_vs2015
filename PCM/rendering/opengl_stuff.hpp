#ifndef OPENGL_STUFF_HPP__
#define OPENGL_STUFF_HPP__

#include "toolbox/portable_includes/port_glew.h"
namespace qglviewer
{
	class Camera;
}

/** Various OpenGL functions
 */

/// Enable the program used to display a textured quad
void EnableProgram();
/// Disable that program
void DisableProgram();

/// Initialize various OpenGL states, buffer objects, textures...
void init_opengl();

/// Draw a quad to paint on
void draw_quad();

/// Display the representation of a blending operatoron the screen at (x,y)
void draw_operator(int x, int y, int w, int h);

/// Draw the controller at position (x,y) in a w x h frame
void draw_global_controller(int x, int y, int w, int h);

GLuint init_tex_operator( int width, int height );

void load_shaders();

void erase_shaders();

/// Draw a circle on screen at position (x, y)  of radius rad
void draw_circle(int width, int height, int x, int y, float rad);

void draw_grid_lines(const qglviewer::Camera* cam);

#endif // OPENGL_STUFF_HPP__
