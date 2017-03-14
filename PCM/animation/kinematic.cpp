#include "kinematic.hpp"
#include "skeleton.hpp"
#include <iostream>
using std::cout;
using std::endl;
static bool debug = 0;
// -----------------------------------------------------------------------------
using namespace Tbx;
Kinematic::Kinematic(Skeleton& s) :
    _skel(s),
    _user_lcl(s.nb_joints()),
    _pose_lcl(s.nb_joints())
{
    for (int i = 0; i < _skel.nb_joints(); ++i) {
        _user_lcl[i] = Transfo::identity();
        _pose_lcl[i] = Transfo::identity();
    }
    save_prev_transfos();
}

// -----------------------------------------------------------------------------


void Kinematic::set_pose_lcl( const std::vector<Transfo>& poses )
{
    save_prev_transfos();
    _pose_lcl = poses;
    _skel.update_anim_pose();
}

// -----------------------------------------------------------------------------

void Kinematic::set_user_lcl_parent(int id_joint, const Transfo& tr)
{
    save_prev_transfos();
    _user_lcl[id_joint] = tr;
    _skel.update_anim_pose();
}

// -----------------------------------------------------------------------------

void Kinematic::reset()
{
    save_prev_transfos();
    for (int i = 0; i < _skel.nb_joints(); ++i) {
        _user_lcl[i] = Transfo::identity();
        _pose_lcl[i] = Transfo::identity();
    }
    _skel.update_anim_pose();
}

// -----------------------------------------------------------------------------

void Kinematic::compute_transfo_gl( Transfo* tr)
{
    rec_compute_tr( tr, _skel.root(), Transfo::identity() );
}

// -----------------------------------------------------------------------------

void Kinematic::rec_compute_tr(Transfo* transfos,
                              int root,
                              const Transfo& parent)
{
    // Joint frame in rest pose (global and local)
    int pid = _skel.parent(root) > -1  ? _skel.parent(root) : root;

    const Transfo f    = _skel.joint_frame    ( pid );



    //const Transfo finv = _skel.joint_frame_lcl( pid );

    // Global transfo of the pose for the current joint
    const Transfo pose = _skel.joint_frame( root ) * _pose_lcl[root] * _skel.joint_frame_lcl( root );

    // Frame of the parent joint with applied pose
    const Transfo ppose_frame = f * _pose_lcl[pid];

    // Global user transfo :
    // The user transfo is based on the parent pose frame
    const Transfo usr = ppose_frame * _user_lcl[root] * ppose_frame.fast_invert();

    // Vertex deformation matrix with repercussion of the user transformations
    // throughout the skeleton's tree
    const Transfo tr = parent *  usr * pose;

    transfos[root] = tr;
if(debug)
{
	cout<<"root "<<root<<endl;
	cout<<"_skel.joint_frame("<<root<<")"<<endl;
	_skel.joint_frame( root ).print();
	cout<<"_pose_lcl("<<root<<")"<<endl;
	_pose_lcl[root].print();
	cout<<"_user_lcl("<<root<<")"<<endl;
	_user_lcl[root].print();
	cout<<"parent "<<endl;
	parent.print();
	cout<<"usr "<<endl;
	usr.print();
	cout<<"pose "<<endl;
	pose.print();

}
    for(unsigned i = 0; i < _skel.get_sons( root ).size(); i++)
        rec_compute_tr(transfos, _skel.get_sons( root )[i], parent * usr);
}

// -----------------------------------------------------------------------------

void Kinematic::save_prev_transfos()
{
    _prev_global.resize( _skel._h_transfos.size() );
    for(unsigned i = 0; i < _prev_global.size(); ++i) {
        _prev_global[i] = _skel._h_transfos[i];
    }
}

// -----------------------------------------------------------------------------
