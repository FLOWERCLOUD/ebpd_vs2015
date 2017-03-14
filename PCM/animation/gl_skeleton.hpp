#ifndef GL_SKELETON_HPP__
#define GL_SKELETON_HPP__

#include "toolbox/gl_utils/glpick.hpp"
#include "toolbox/maths/color.hpp"
#include "../rendering/camera.hpp"
#include <vector>

struct Skeleton;

/// @brief drawing skeleton with opengl
/// @see Skeleton
class GL_skeleton {
public:
    GL_skeleton(const Skeleton* skel);

    void draw_bone(int i, const Tbx::Color& c, bool rest_pose, bool use_material, bool use_circle = false);

    int nb_joints() const;

    const Skeleton* _skel;
};


#endif // GL_SKELETON_HPP__
