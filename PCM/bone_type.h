#ifndef BONE_TYPE_HPP__
#define BONE_TYPE_HPP__

/** @namespace Bone_type
 *  @brief This namespace holds an enum field related to Bone class
 *
 *  @see Bone
*/
// =============================================================================
namespace EBone{
// =============================================================================

enum Bone_t {
    SSD,
    HRBF,
    PRECOMPUTED,
    CYLINDER
};

/// A bone identifier
typedef int Id;

} // END BONE_TYPE NAMESPACE ===================================================

#endif // BONE_TYPE_HPP__
