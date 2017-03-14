#ifndef G_TEXTURES_HPP_
#define G_TEXTURES_HPP_

#include "toolbox/portable_includes/port_glew.h"

// -----------------------------------------------------------------------------

/// Vbo id of the quad used to display the pbo that contains the result
/// of the raytracing done by cuda
extern GLuint g_gl_quad;

// -----------------------------------------------------------------------------

/// Textures used to display controller and operator frames
extern GLuint g_ctrl_frame_tex;
extern GLuint g_op_frame_tex;
extern GLuint g_op_tex;

/// use these to index pbos & textures:
enum{ COLOR = 0,
      DEPTH,
      MAP,
      NORMAL_MAP, ///< scene normals
      NOISE,
      NB_TEX      // Keep that at the end
     };

/// Textures that store color and depth buffer + map
extern GLuint g_gl_Tex[NB_TEX];

// -----------------------------------------------------------------------------

#endif // G_TEXTURES_HPP_
