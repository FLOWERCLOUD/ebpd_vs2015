#ifndef G_SHADERS_HPP__
#define G_SHADERS_HPP__

#include "toolbox/gl_utils/shader.hpp"

/** @file g_shaders.hpp
    @brief export the global variables variables related to shader programs
    @see globals.cpp
*/

/// Various shader programs
extern Tbx::Shader_prog* g_dummy_quad_shader;
extern Tbx::Shader_prog* g_points_shader;
extern Tbx::Shader_prog* g_normal_map_shader;
extern Tbx::Shader_prog* g_ssao_shader;

/// 'phong_list' is a list of shaders generated from a single file source
/// the enum field specifies which type of shader is in the list
enum {
    NO_TEX,            ///< phong shading no texture
    MAP_KD,            ///< phong shading only diffuse texture
    MAP_KD_BUMP,       ///< phong shading only bump and diffuse textures
    MAP_KD_KS,         ///< phong shading only diffuse and specular textures
    NB_PHONG_SHADERS
};

/// @namespace Tex_units
/// @brief specifies which textures units are used in 'phong_list'
namespace Tex_units{
    extern const int KD;
    extern const int KS;
    extern const int BUMP;
}

extern Tbx::Shader_prog* g_phong_list[NB_PHONG_SHADERS];

#endif // G_SHADERS_HPP__
