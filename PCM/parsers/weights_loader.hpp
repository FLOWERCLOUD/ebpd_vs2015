#ifndef WEIGHTS_LOADER_HPP__
#define WEIGHTS_LOADER_HPP__

#include <vector>
#include <map>

// =============================================================================
namespace Loader {
// =============================================================================

/**
 * @namespace Weights_loader
 * @brief Load weights of influence for ssd skinning (custom data format)
 */
// =============================================================================
namespace Weights_loader {
// =============================================================================

/// @param nb_vert : number of vertices of the mesh the weights belongs to
/// @param weights : ssd weights per vertices per bones.
/// weights[ith_vert][bone_id] = ssd_weight
void load(const char* filename,
          int nb_vert,
          std::vector< std::map<int, float> >& weights);

}// END Weights_loader =========================================================

}// END Loader =================================================================

#endif // WEIGHTS_LOADER_HPP__
