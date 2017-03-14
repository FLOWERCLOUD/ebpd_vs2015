#ifndef TEXTURES_ENV_HPP__
#define TEXTURES_ENV_HPP__

#include "toolbox/maths/vec3.hpp"

/**
 * @namespace Textures_env
 * @brief Holds textures for the raytracing (env maps, bump etc.)
 *
 */
// =============================================================================
namespace Textures_env {
// =============================================================================

extern const int light_envmap_downscale;
extern cudaArray* array_envmap;
extern cudaArray* array_light_envmap;
extern cudaArray* blob_tex;
extern cudaArray* array_extrusion_mask[];
extern cudaArray* array_extrusion_gradient[];
extern int envmapx, envmapy;
extern int blobx, bloby;
extern int extrusionx[];
extern int extrusiony[];
extern int* img_envmap;
extern int* img_light_envmap;
extern int* blob_img;
extern float* img_extrusion_mask[2];
extern float2* img_extrusion_gradient[2];
extern bool img_ok;
extern bool blob_ok;
extern bool extrusion_ok;
extern bool bind_ok;

// -----------------------------------------------------------------------------

void clean_env();
/// @warning don't know what will happen if you try to load a map twice
/// code to be checked
void load_envmap(char* filename);
void load_blob_tex(char* filename);
void load_extrusion_tex(char* filename, int n);

}// END NAMESPACE IMG_TEXTURES =================================================


#endif // TEXTURES_ENV_HPP__
