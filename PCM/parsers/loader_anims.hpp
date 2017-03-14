#ifndef LOADER_ANIMS_HPP__
#define LOADER_ANIMS_HPP__

#include <vector>
#include "toolbox/maths/transfo.hpp"

// =============================================================================
namespace Loader {
// =============================================================================

/// @class Base_anim_eval
/// @brief Abstract class to evaluate a skeleton animation
class Base_anim_eval {
public:
    Base_anim_eval(const std::string& name) :
        _name( name ),
        _frame_rate(30.f)
    { }

    virtual ~Base_anim_eval(){ }

    /// @return the local frame of the ith bone for the ith frame
    virtual Tbx::Transfo eval_lcl(int bone_id, int frame) = 0;
    virtual int nb_frames () const = 0;

    /// frame rate in seconds
    float frame_rate() const { return _frame_rate; }

    std::string _name;
    float _frame_rate;
};

//------------------------------------------------------------------------------

/// @class Sampled_anim_eval
/// @brief Implementation of animation evaluator based on matrix samples
/// For each frame and each bone this class stores the associated matrix
class Sampled_anim_eval : public  Base_anim_eval {
public:
    Sampled_anim_eval(const std::string& name) :
        Base_anim_eval( name )
    { }

    Tbx::Transfo eval_lcl(int bone_id, int frame){
        return _lcl_frames[frame][bone_id];
    }

    int nb_frames () const { return _lcl_frames.size();}

    /// Stores every bones local transformations for each frame
    /// _gl_frames[ith_frame][bone_id] == local_transformation
    std::vector< std::vector<Tbx::Transfo> > _lcl_frames;
};


} // END namespace Loader ======================================================

#endif // LOADER_ANIMS_HPP__
