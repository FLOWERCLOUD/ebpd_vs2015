#ifndef TEXTURES_ENV_TEX_HPP__
#define TEXTURES_ENV_TEX_HPP__

#include "textures_env.hpp"

//texture for environment map:
texture<uchar4, 2, cudaReadModeNormalizedFloat> g_tex_envmap;
texture<uchar4, 2, cudaReadModeNormalizedFloat> g_tex_light_envmap;
texture<uchar4, 2, cudaReadModeNormalizedFloat> g_tex_blob;
texture<float, 2, cudaReadModeElementType> g_tex_extrusion;
texture<float2, 2, cudaReadModeElementType> g_tex_extrusion_gradient;
texture<float, 2, cudaReadModeElementType> g_tex_extrusion2;
texture<float2, 2, cudaReadModeElementType> g_tex_extrusion2_gradient;

// -----------------------------------------------------------------------------
namespace Textures_env{
// -----------------------------------------------------------------------------

    __device__
    float4 sample_envmap(float u, float v);

    __device__
    float4 sample_envmap(const Vec3& d);

    __device__
    float4 sample_light_envmap(float u, float v);

    __device__
    float4 sample_envmap(const Vec3& d);

    __device__
    float sample_extrusion(int n, float u, float v);

    __device__
    float2 sample_extrusion_gradient(int n, float u, float v);

    void init();

}// END Textures_env -----------------------------------------------------------

#include "textures_env_tex.inl"

#endif // TEXTURES_ENV_TEX_HPP__
