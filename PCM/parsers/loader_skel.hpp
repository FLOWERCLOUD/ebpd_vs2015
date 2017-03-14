#ifndef LOADER_SKEL_HPP__
#define LOADER_SKEL_HPP__

#include "toolbox/maths/transfo.hpp"
#include <string>
#include <vector>

// =============================================================================
namespace Loader {
// =============================================================================

/// @brief intermediate representation of a bone for file loading
struct Abs_bone {
    float       _length; ///< The bone length
    Tbx::Transfo     _frame;  ///< The bone position and orientation
    std::string _name;   ///< The bone name
};

//------------------------------------------------------------------------------

/// @brief intermediate representation of a skeleton for file loading
struct Abs_skeleton {

    int _root; ///< index of the root bone in _bones

    /// List of bones
    std::vector<Abs_bone> _bones;
    /// _sons[bone_id] == vec_sons
    std::vector< std::vector<int> > _sons;
    /// _parents[bone_id] == parent_bone_id
    std::vector<int> _parents;

    /// _weights[vert_idx][ith_bone].first  == bone_idx
    /// _weights[vert_idx][ith_bone].second == bone_weight
    std::vector< std::vector< std::pair<int, float> > > _weights;
};

//------------------------------------------------------------------------------

/// Compute the bone lengths of every bones.
/// The hierachy of the skeleton and bones positions must be correctly filled.
/// The bone length equal the mean length between the joint and its sons.
/// Leaves are of length zero
void compute_bone_lengths(Abs_skeleton& skel);

} // END LOADER NAMESPACE ======================================================

#endif // LOADER_SKEL_HPP__
