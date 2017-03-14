#ifndef PPM_LOADER_HPP__
#define PPM_LOADER_HPP__

#include <string>

// =============================================================================
namespace Loader {
// =============================================================================

/**
  @namespace Ppm_loader
  @brief Reading/writting '.ppm' images

  .PPM is an easy to read and write file format. It's an ASCII format.
  First line stores the width/height of the image the RGB channels are listed
  for every pixels.

  @note Due to it's ASCII nature .PPM is an highly unefficient format for both
  memory and speed performances.
*/
// =============================================================================
namespace Ppm_loader {
// =============================================================================

/// Quick and dirty ppm loader
/// writes dimensions in 'width' & 'height'
/// writes image data in 'data'
bool read(const std::string& path_name, int& width, int& height, int*& data);

bool read_with_alpha(const std::string& path_name, int& width, int& height, int*& data);

}// END PPM LOADER =============================================================

}// END Loader =================================================================

#endif // PPM_LOADER_HPP__
