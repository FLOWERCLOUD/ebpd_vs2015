#ifndef KINEMATIC_HPP__
#define KINEMATIC_HPP__

#include <vector>
#include "toolbox/maths/transfo.hpp"

struct Skeleton;

/**
  @name Kinematic
  @brief Handling forward and inverse kinematic of the skeleton

  This class aims to handle the skeleton kinematic. While the skeleton
  only gives means to change each joint transformation this class
  provides more advanced functionnalities like passing on a user defined
  transformations to every children.
*/
class Kinematic {
public:
    Kinematic(Skeleton &s);

    /// Set the skeleton pose given a vector of local transformations at
    /// each joints. Usually we use this to load a saved animation
    /// @note this updates the skeleton pose
    void set_pose_lcl( const std::vector<Tbx::Transfo>& poses );

    /// Set fot the ith joint a local transformation expressed in the
    /// <b> parent </b> joint frame coordinate system
    /// @note this updates the skeleton pose
    void set_user_lcl_parent(int id_joint, const Tbx::Transfo& tr);

    Tbx::Transfo get_user_lcl_parent(int id_joint) const {
        return _user_lcl[id_joint];
    }

    /// Given the current skeleton state the various local transformations
    /// at each joints we compute the global transformations used to deform
    /// the mesh's vertices. This method is called automatically by the skeleton
    /// when needed
    void compute_transfo_gl(Tbx::Transfo *tr);

    void reset();

    /// Previous global transformations prior to a user modification done with
    /// set_xxx() functions
    Tbx::Transfo get_prev_transfo( int i ) const { return _prev_global[i]; }

private:
    void rec_compute_tr(Tbx::Transfo* transfos, int root, const Tbx::Transfo& parent);

    void save_prev_transfos();

    Skeleton& _skel;

    /// Locale transformations of the skeleton's joints defined by the user.
    /// Applied on top of the pose transformations.
    /// These transformations are expressed in their <b>parent</b> joint frame
    std::vector<Tbx::Transfo> _user_lcl;

    /// Locale transformations of the skeleton's joints. This is always applied
    /// first. It's Usually defined by the current animation frame pose
    std::vector<Tbx::Transfo> _pose_lcl;

    /// Previous globol transformations prior to a user modification done with
    /// set_xxx() functions
    std::vector<Tbx::Transfo> _prev_global;
};

#endif // KINEMATIC_HPP__
