#ifndef SKELETON_CTRL_HPP_
#define SKELETON_CTRL_HPP_

#include <string>
#include <vector>
#include "toolbox/maths/vec3.hpp"
#include "../animation/bone_type.hpp"
#include "../animation/joint_type.hpp"
#include "../rendering/camera.hpp"
#include "toolbox/maths/transfo.hpp"
#include "../parsers/loader_skel.hpp"
#include "../parsers/loader_anims.hpp"
#include "../animation/graph.hpp"


/**
    @brief
*/
class Skeleton_ctrl {
public:
    Skeleton_ctrl() :
        _display(true)
    {
    }

    void load_pose(const std::string& filepath);
    void save_pose(const std::string& filepath);

    void load(const Graph& g_graph);

    void load(const Loader::Abs_skeleton& abs_skel);

    bool is_loaded();

    bool is_displayed  (){ return _display;      }
    void switch_display(){ _display = !_display; }

    /// reset the skelton position in the resting pose
    void reset();

    /// @return the bone_id associated to the hrbf_id or -1
    int find_associated_bone(int hrbf_id);

    // -------------------------------------------------------------------------
    /// @name Setters
    // -------------------------------------------------------------------------

    /// Given an animation evaluator and a time, set the skeleton to the
    /// corresponding pose.
    void set_pose(Loader::Base_anim_eval* evaluator, int frame);

    void set_joint_blending(int i, EJoint::Joint_t type);

    void set_joint_bulge_mag(int i, float m);

    //void set_joint_controller(int id_joint, const IBL::Ctrl_shape& shape);

    void set_joint_pos(int joint_id, const Tbx::Vec3& pos);

    void set_offset_scale(const Tbx::Vec3& off, float scale);

    // -------------------------------------------------------------------------
    /// @name Getters
    // -------------------------------------------------------------------------

    /// @return identifier of the root bone
    int root();

    Tbx::Vec3 joint_pos(int idx);

    /// Get local frame of the bone ( fx will  be oriented along the bone)
    /// no asumption can be done for fy and fz
    void joint_anim_frame(int id_bone,
                          Tbx::Vec3& fx,
                          Tbx::Vec3& fy,
                          Tbx::Vec3& fz);

    Tbx::Transfo joint_anim_frame(int id_bone);

    Tbx::Transfo bone_anim_frame(int id_bone);

    /// @return the hrbf indentifier in HRBF_Env or -1 if the designated bone
    /// is not a hrbf
    int get_hrbf_id(int bone_id);

    /// @return the bone id associated to the hrbf -1 if the hrbf_id does not
    /// exists
    int get_bone_id(int hrbf_id);

    /// @return the parent id of the bone of index 'bone_id'
    int get_parent(int bone_id);

    int get_bone_type(int bone_id);

    EJoint::Joint_t get_joint_blending(int id);

    //IBL::Ctrl_shape get_joint_controller(int id_joint);

    int get_nb_joints();

    const std::vector<int>& get_sons(int joint_id);

    // -------------------------------------------------------------------------
    /// @name Selection
    // -------------------------------------------------------------------------

    const std::vector<int>& get_selection_set(){ return _selected_joints; }

    int get_last_selected() const {
        return _selected_joints.size() > 0 ? _selected_joints[_selected_joints.size() - 1] : -1;
    }

    /// Try to select a joint and add it to the previous selection set
    /// @return if a vertex has been selected
    bool select_joint(const Tbx::Camera &cam, int x, int y, bool rest_pose);

    /// Try to select a joint if succeed replace the previous set.
    /// if it fails do nothing (the previous selection is kept).
    /// @return wether a joint has been selected or not
    bool select_safely(const Tbx::Camera &cam, int x, int y, bool rest_pose);

    /// Try to unselect a joint and remove it from the previous selection set
    bool unselect(const Tbx::Camera &cam, int x, int y, bool rest_pose);

    /// Select a joint given its ID
    /// @return if it was already selected
    bool select_joint(int joint_id);

    /// Select every joints and add them to the selection list
    /// @return number of joints selected
    int select_all();

    void reset_selection();

private:
    // -------------------------------------------------------------------------
    /// @name Tools
    // -------------------------------------------------------------------------

    void add_to_selection(int id);
    void remove_from_selection(int id);

    // -------------------------------------------------------------------------
    /// @name Attributes
    // -------------------------------------------------------------------------

    bool _display;
    std::vector<int> _selected_joints; ///< set of selected skeleton joints
};


#endif // SKELETON_CTRL_HPP_
