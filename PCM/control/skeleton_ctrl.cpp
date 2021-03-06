#include "skeleton_ctrl.hpp"

#include "../animation/animesh.hpp"
#include "../global_datas/cuda_globals.hpp"
#include "../global_datas/toolglobals.hpp"
#include "../control/cuda_ctrl.hpp"
#include "../animation/skeleton.hpp"
#include <iostream>
using std::cout;
using std::endl;

using namespace Tbx;
// -----------------------------------------------------------------------------

void Skeleton_ctrl::load_pose(const std::string& filepath)
{
    g_skel->load_pose( filepath );
}

// -----------------------------------------------------------------------------

void Skeleton_ctrl::save_pose(const std::string& filepath)
{
    g_skel->save_pose( filepath );
}

// -----------------------------------------------------------------------------

int Skeleton_ctrl::root()
{
    return g_skel->root();
}

// -----------------------------------------------------------------------------

void Skeleton_ctrl::load( const Graph& g_graph )
{
    delete g_skel;
    g_skel = new Skeleton(g_graph, 0);

    reset_selection();
}

// -----------------------------------------------------------------------------

void Skeleton_ctrl::load(const Loader::Abs_skeleton& abs_skel)
{
    delete g_skel;
    g_skel = new Skeleton(abs_skel);

    reset_selection();
}

// -----------------------------------------------------------------------------

bool Skeleton_ctrl::is_loaded(){ return g_skel != 0; }

// -----------------------------------------------------------------------------

void Skeleton_ctrl::reset(){
    g_skel->reset();
}

// -----------------------------------------------------------------------------

Vec3 Skeleton_ctrl::joint_pos(int idx) {
    return g_skel->joint_pos(idx);
}

// -----------------------------------------------------------------------------

void Skeleton_ctrl::joint_anim_frame(int id_bone,
                              Vec3& fx,
                              Vec3& fy,
                              Vec3& fz)
{
    Mat3 m = g_skel->joint_anim_frame(id_bone).get_mat3();
    fx = m.x();
    fy = m.y();
    fz = m.z();
}

// -----------------------------------------------------------------------------

Transfo Skeleton_ctrl::joint_anim_frame(int id_joint)
{
    return g_skel->joint_anim_frame( id_joint );
}

// -----------------------------------------------------------------------------

Transfo Skeleton_ctrl::bone_anim_frame(int id_bone)
{
    return g_skel->bone_anim_frame( id_bone );
}

// -----------------------------------------------------------------------------

bool Skeleton_ctrl::select_joint(const Camera &cam, int x, int y, bool rest_pose)
{
    //y = Cuda_ctrl::_display._height - y;
    int nearest = g_skel->select_joint( cam, (float)x, (float)y, rest_pose );
    if( nearest > -1 )
    {
		cout<<"select joint "<<nearest<<endl;
        add_to_selection( nearest );
        if(g_animesh->get_color_type() == EAnimesh::SSD_WEIGHTS)
            g_animesh->set_color_ssd_weight(nearest);

        return true;
    }
    return false;
}

// -----------------------------------------------------------------------------

bool Skeleton_ctrl::select_safely(const Camera &cam, int x, int y, bool rest_pose)
{
    //y = Cuda_ctrl::_display._height - y;
    int nearest = g_skel->select_joint( cam, (float)x, (float)y, rest_pose );
    if(nearest > -1)
    {
        reset_selection();
        add_to_selection( nearest );
        //DEBUG
        std::cout << "bone type : " << EBone::type_to_string(g_skel->bone_type(nearest)) << std::endl;//DEBUG
        std::cout << "bone id : " << nearest << std::endl;//DEBUG
        //DEBUG
        if(g_animesh->get_color_type() == EAnimesh::SSD_WEIGHTS)
            g_animesh->set_color_ssd_weight(nearest);

        return true;
    }
    return false;
}

// -----------------------------------------------------------------------------

bool Skeleton_ctrl::unselect(const Camera &cam, int x, int y, bool rest_pose)
{
    //y = Cuda_ctrl::_display._height - y;
    int nearest = g_skel->select_joint( cam, (float)x, (float)y, rest_pose );
    if(nearest > -1){
        remove_from_selection( nearest );
        return true;
    }
    return false;
}

// -----------------------------------------------------------------------------

bool Skeleton_ctrl::select_joint(int joint_id)
{
    bool state = false;
    for(unsigned i = 0; i < _selected_joints.size(); ++i){
        if(_selected_joints[i] == joint_id){
            state = true;
            break;
        }
    }
    add_to_selection( joint_id );
    return state;
}

// -----------------------------------------------------------------------------

int Skeleton_ctrl::select_all()
{
    // Select all joints but keep the last selection last in the vector
    int id = -1;
    if( _selected_joints.size() > 0)
        id = _selected_joints[ _selected_joints.size()-1 ];

    _selected_joints.clear();
    int nb_joints = g_skel->nb_joints();
    for(int i = 0; i < nb_joints; i++){
        if( i == id ) continue;
        _selected_joints.push_back(i);
    }

    if(id != -1) _selected_joints.push_back(id);

    return nb_joints;
}

// -----------------------------------------------------------------------------

void Skeleton_ctrl::reset_selection()
{
    _selected_joints.clear();
}

// -----------------------------------------------------------------------------

void Skeleton_ctrl::add_to_selection(int id)
{
    // Check for doubles
    bool state = false;
    for(unsigned int i=0; i<_selected_joints.size(); i++)
        state = state || (_selected_joints[i] == id);

    if(!state) _selected_joints.push_back(id);
}

// -----------------------------------------------------------------------------

void Skeleton_ctrl::remove_from_selection(int id)
{
    std::vector<int>::iterator it = _selected_joints.begin();
    unsigned int i = 0;
    for(; it<_selected_joints.end(); ++it, ++i)
        if( (*it) == id )
            break;

    if(i < _selected_joints.size())
        _selected_joints.erase(it);
}

// -----------------------------------------------------------------------------

int Skeleton_ctrl::get_hrbf_id(int bone_id)
{

    //if( g_skel->bone_type(bone_id) == EBone::HRBF)
    //{
    //    Bone_hrbf* b = ((Bone_hrbf*)g_skel->get_bone(bone_id));
    //    return b->get_hrbf().get_id();
    //}
    //else
        return -1;

}

// -----------------------------------------------------------------------------

int Skeleton_ctrl::get_bone_id(int hrbf_id)
{
    //for (int i = 0; i < g_skel->nb_joints(); ++i)
    //{
    //    if( g_skel->bone_type(i) == EBone::HRBF)
    //    {
    //        Bone_hrbf* b = ((Bone_hrbf*)g_skel->get_bone(i));
    //        if(b->get_hrbf().get_id() == hrbf_id)
    //            return i;
    //    }
    //}

    // hrbf id does not exists or hrbf are not even used
    return -1;
}

// -----------------------------------------------------------------------------

int Skeleton_ctrl::get_parent(int bone_id){
    return g_skel->parent(bone_id);
}

// -----------------------------------------------------------------------------

int Skeleton_ctrl::get_bone_type(int bone_id){
    return g_skel->bone_type(bone_id);
}

// -----------------------------------------------------------------------------

void Skeleton_ctrl::set_pose(Loader::Base_anim_eval* evaluator, int frame)
{
    if( g_skel == 0) return;

    std::vector<Transfo> trs( g_skel->nb_joints() );
    for(int i = 0; i < g_skel->nb_joints(); i++)
        trs[i] = evaluator->eval_lcl( i, frame );

    g_skel->_kinec->set_pose_lcl( trs );
}

// -----------------------------------------------------------------------------

void Skeleton_ctrl::set_joint_pos(int joint_id, const Vec3& pos)
{
   g_skel->set_joint_rest_pos(joint_id, Point3(pos) );
}

// -----------------------------------------------------------------------------

void Skeleton_ctrl::set_offset_scale(const Vec3& off, float scale)
{
    g_skel->set_offset_scale(off, scale);
}

// -----------------------------------------------------------------------------
//
//IBL::Ctrl_shape Skeleton_ctrl::get_joint_controller(int id_joint){
//    int pt = g_skel->parent( id_joint );
//    if( pt > -1)
//        return g_skel->get_joint_controller(/*id_joint*/pt);
//    else
//        return IBL::Ctrl_shape();
//}

// -----------------------------------------------------------------------------
//
//void Skeleton_ctrl::set_joint_controller(int id_joint, const IBL::Ctrl_shape& shape){
//    int pt = g_skel->parent( id_joint );
//    if( pt > -1)
//        g_skel->set_joint_controller(/*id_joint*/pt, shape);
//}

// -----------------------------------------------------------------------------

void Skeleton_ctrl::set_joint_blending(int i, EJoint::Joint_t type){
    int pt = g_skel->parent( i );
    if(pt > -1)
        g_skel->set_joint_blending(pt, type);
    //g_animesh->update_base_potential();
}

// -----------------------------------------------------------------------------

//EJoint::Joint_t Skeleton_ctrl::get_joint_blending(int id)
//{
//    int pt = g_skel->parent( id );
//    if(pt > -1)
//        return g_skel->joint_blending(pt);
//
//    return EJoint::NONE;
//    //g_animesh->update_base_potential();
//}

// -----------------------------------------------------------------------------

void Skeleton_ctrl::set_joint_bulge_mag(int i, float m){
    g_skel->set_joint_bulge_mag(i, m);
}

// -----------------------------------------------------------------------------

int Skeleton_ctrl::get_nb_joints(){
    return g_skel->nb_joints();
}

// -----------------------------------------------------------------------------

const std::vector<int>& Skeleton_ctrl::get_sons(int joint_id)
{
    return g_skel->get_sons( joint_id );
}

// -----------------------------------------------------------------------------

int Skeleton_ctrl::find_associated_bone(int hrbf_id)
{
    //for(int i = 0; i < g_skel->nb_joints(); i++)
    //{
    //    const Bone* b = g_skel->get_bone(i);
    //    if(b->get_type() == EBone::HRBF)
    //    {
    //        const HermiteRBF& hrbf = ((const Bone_hrbf*)b)->get_hrbf();
    //        if(hrbf.get_id() == hrbf_id)
    //            return i;
    //    }
    //}
    return -1;
}

// -----------------------------------------------------------------------------
