#ifndef TEX_LOADER_HPP__
#define TEX_LOADER_HPP__

#include "toolbox/gl_utils/gltex2D.hpp"
#include <string>

// =============================================================================
namespace Loader {
// =============================================================================

/**
  @namespace Tex_loader
  @brief Loading/writting openGL textures utilities (using Qt4 QImage class)

*/
// =============================================================================
namespace Tex_loader{
// =============================================================================

/// @param file_path : path to the texture image. We use Qt to parse the image
/// file, so what QImage can open this function can too.
/// @return An openGL textures
Tbx::GlTex2D* load(const std::string& file_path);

}// END TEX_LOADER NAMESPACE ===================================================

}// END LOADER NAMESPACE =======================================================

#endif // TEX_LOADER_HPP__
