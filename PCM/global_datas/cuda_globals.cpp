#include "cuda_globals.hpp"

#include "../rendering/camera.hpp"
#include "../global_datas/toolglobals.hpp"
#include "../control/cuda_ctrl.hpp"
#include "../animation/skeleton.hpp"

using namespace Tbx;
Material_cu g_ray_material;

Graph*    g_graph   = 0;
Skeleton* g_skel    = 0;
Animesh*  g_animesh = 0;

bool g_cuda_context_is_init = false;

/// Return true if one of the given parameters has changed or some other scene
/// parameters since the last call of this function.
/// N.B: the first call returns always true
bool has_changed(const Tbx::Camera& cam,
                 const Vec3& v_plane,
                 const Point3& p_plane,
                 int width,
                 int height)
{
    using namespace Cuda_ctrl;
	using namespace Tbx;
    static bool begin = true;       // First call is a changed
    static Tbx::Camera prev_cam;
    static Vec3  prev_v_plane;
    static Point3   prev_p_plane;
    static int   prev_width       = -1;
    static int   prev_height      = -1;
    static int prev_selected_joint = -1;
    static int prev_selection_size = -1;
    static Vec3 prev_joint_pos(0.f, 0.f, 0.f);

    // Sum every joint coordinates, not very accurate but enough to
    // detect small changes between two frame
    Vec3 joint_pos(0.f, 0.f, 0.f);
    if (g_skel != 0)
        for(int i=0; i<g_skel->nb_joints(); i++)
            joint_pos += g_skel->joint_pos(i);

    // extract last selected joints
    const std::vector<int>& joint_set = _skeleton.get_selection_set();
    int s = (int)joint_set.size();
    int selected_joint =  (s != 0) ? joint_set[s-1] : -1;

    // Detect differences between this frame and the last one
    float delta = 0.f;
    delta += (prev_cam.get_pos()-cam.get_pos()).norm();
    delta += prev_cam.get_dir().cross(cam.get_dir()).norm();
    delta += prev_cam.get_y().cross(cam.get_y()).norm();
    delta += fabs(prev_cam.get_ortho_zoom() - cam.get_ortho_zoom());
    delta += fabs(prev_cam.get_near() - cam.get_near());
    delta += fabs(prev_cam.get_far() - cam.get_far());
    delta += (float)(prev_cam.is_ortho() != cam.is_ortho());
    delta += (prev_p_plane-p_plane).norm();
    delta += prev_v_plane.cross(v_plane).norm();
    delta += abs(prev_width-width);
    delta += abs(prev_height-height);
    delta += (float)(prev_selected_joint != selected_joint);
    delta += (float)(prev_selection_size != s);
    delta += (joint_pos != prev_joint_pos);

    if( delta > 0.0001f || begin)
    {
        prev_cam         = cam;
        prev_v_plane     = v_plane;
        prev_p_plane     = p_plane;
        prev_width       = width;
        prev_height      = height;
        prev_selected_joint = selected_joint;
        prev_selection_size = s;
        prev_joint_pos = joint_pos;
        begin = false;

        return true;
    }

    return false;
}

// -----------------------------------------------------------------------------
