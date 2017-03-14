#include "skeleton.hpp"

#include <iostream>
#include <fstream>

#include "toolbox/std_utils/string.hpp"
#include "toolbox/std_utils/vector.hpp"
#include "toolbox/maths/color.hpp"
//#include "blending_env.hpp"
//#include "blending_env_ctrl.hpp"
//#include "skeleton_env.hpp"
//#include "hrbf_kernels.hpp"
#include <iostream>
#include "toolbox/gl_utils/glsave.hpp"
#include "toolbox/class_saver.hpp"
#include "../rendering/camera.hpp"
#include "graph.hpp"
#include "../parsers/loader_skel.hpp"
#include "toolbox/gl_utils/glpick.hpp"
#include "../global_datas/g_vbo_primitives.hpp"// <- TODO: to be deleted as well as drawing funcs -> use gl_skeleton instead
using std::cout;
using std::endl;
// -----------------------------------------------------------------------------

#ifndef PI
#define PI (3.14159265358979323846f)
#endif

using namespace Tbx;
const float default_bone_radius = 1.f;

// -----------------------------------------------------------------------------

/// @param g : graph to look up
/// @param curr : current node of 'g'
/// @param prev : previous (i.e parent) node of g(curr)
/// @param res : newly created tree with nodes duplicated when several sons
/// @param pt : current parent in res
static void rec_gen_bone_graph(const Graph& g,
                               int curr,
                               int prev,
                               Graph& res,
                               int pt)
{
    // Current is a leaf:
    if( g._neighs[curr].size() < 2  && prev != -1)
    {
        int id = res.push_vertex( g._vertices[curr] );
        if(pt != -1) res.push_edge( Graph::Edge(pt, id) );
        return;
    }

    for(unsigned i = 0; i < g._neighs[curr].size(); ++i)
    {
        const int arc = g._neighs[curr][i];

        if(arc == prev) continue; // skip parent

        int id = res.push_vertex( g._vertices[curr] );
        if(pt != -1) res.push_edge( Graph::Edge(pt, id) );

        rec_gen_bone_graph(g, arc, curr, res, id);
    }

}

// -----------------------------------------------------------------------------

static Graph gen_bone_graph(const Graph& g, int root)
{
    Graph res( g._offset, g._scale);
    if( g._neighs[root].size() > 1)
    {
        // When root has more than 1 son it will be duplicated, hence no root
        // node will exists. We need to create it
        Vec3 v = g._vertices[root];
        v.x += 0.1f; // offset so that it has not zero length
        int id = res.push_vertex( v );
        rec_gen_bone_graph(g, root, -1, res, id);

    }
    else if ( g._neighs[root].size() == 0)
        res.push_vertex( g._vertices[root] );
    else
        rec_gen_bone_graph(g, root, -1, res, -1);

    return  res;
}

// -----------------------------------------------------------------------------

void Skeleton::init(int nb_joints)
{
    _pick = Tbx::GLPick(nb_joints);
    _pick._pick_size = 3.0f;
    _nb_joints = nb_joints;
    _children.resize(nb_joints);
    _parents.resize(nb_joints);
    _frames.resize(nb_joints);
    _lcl_frames.resize(nb_joints);
    _anim_frames.resize(nb_joints);
    _saved_transfos.resize(nb_joints);
    _h_transfos.resize(nb_joints);
    _d_transfos.resize(nb_joints);
    _h_dual_quat.resize(nb_joints);
    _d_dual_quat.resize(nb_joints);
    //_joints_data.resize(nb_joints);
    //_controllers.malloc(nb_joints);
    _anim_bones.resize(nb_joints);
    _bones.resize(nb_joints);
    _hrbf_radius.resize(nb_joints, 1.f);

    for(int i = 0; i < nb_joints; i++)
    {
        _anim_bones[i] = new Bone_ssd();

        //Skeleton_env::Joint_data d;
        //d._blend_type     = EJoint::MAX;
        //d._ctrl_id        = Blending_env::new_ctrl_instance();
        //d._bulge_strength = 0.7f;

        //_controllers[i] = IBL::Ctrl_shape::caml();
        //_joints_data[i] = d;
        //Blending_env::update_controller(d._ctrl_id, _controllers[i]);

        _anim_bones[i]->set_radius(default_bone_radius);
        _anim_bones[i]->_bone_id = i;

        _h_transfos[i] = Transfo::identity();
    }

    _scale = 1.f;
    _offset = Tbx::Vec3::zero();
    _kinec  = new Kinematic(*this);
}

// -----------------------------------------------------------------------------

void Skeleton::init_skel_env()
{
    //_skel_id = Skeleton_env::new_skel_instance(_root, _anim_bones, _parents);
    //update_anim_pose();
    //Skeleton_env::update_joints_data(_skel_id, _joints_data);
    //Skeleton_env::update_bones_data (_skel_id, _anim_bones );
}

// -----------------------------------------------------------------------------

Skeleton::Skeleton(const Graph& graph, int root)
{
    assert( !graph.is_cycles(root) );

    // from the graph we generate another graph were nodes with more than one
    // son are duplicated to obtain as many nodes as sons. The new nodes keeps
    // the position and arcs of the old node.
    // Later each node will give rise to a single bone.
//    Graph g = gen_bone_graph(graph, root);
	Graph g = graph;
    // The generated graph has always root set to node 0
    _root = root = 0;

    init( g.nb_vertices() );

    _parents[_root] = -1;
    Mat3 id = Mat3::identity();
    _frames     [_root] = Transfo( id, g._vertices[_root] );
    _lcl_frames [_root] = _frames[_root].fast_invert();

    Graph tmp = g;
    fill_children( tmp, _root );
    fill_frames( g );
    fill_bones();
    // must be called last
    init_skel_env();
}

// -----------------------------------------------------------------------------

Skeleton::Skeleton(const Loader::Abs_skeleton& skel) : _root(skel._root)
{
    init( skel._bones.size() );

    for(unsigned i = 0; i < skel._bones.size(); i++ )
    {
        Transfo tr = skel._bones[i]._frame;
        _frames    [i] = tr;
        _lcl_frames[i] = tr.fast_invert();
        _parents   [i] = skel._parents[i];

        _anim_bones[i]->set_length( skel._bones[i]._length );

    }
    _children = skel._sons;

    fill_bones();
    // must be called last
    init_skel_env();
}

// -----------------------------------------------------------------------------

Skeleton::~Skeleton()
{
    for(unsigned i = 0; i < _anim_bones.size(); i++){
        _children[i].clear();
        delete _anim_bones[i];
        //const int ctrl_id = _joints_data[i]._ctrl_id;
        //if( ctrl_id >= 0)
        //    Blending_env::delete_ctrl_instance(ctrl_id);
    }
    delete _kinec;

    //Skeleton_env::delete_skel_instance( _skel_id );
}

// -----------------------------------------------------------------------------

void Skeleton::reset()
{
    for(int i = 0; i < nb_joints(); i++){
        _saved_transfos[i] = _h_transfos[i];
        _h_transfos[i] = Transfo::identity();
    }
    update_bones_pose( _h_transfos );
}

// -----------------------------------------------------------------------------

void Skeleton::unreset()
{
    for(int i = 0; i < nb_joints(); i++)
        _h_transfos[i] = _saved_transfos[i];
    update_bones_pose( _h_transfos );
}

// -----------------------------------------------------------------------------

void Skeleton::compute_joints_half_angles(HA_Vec3& half_angles,
                                          HA_Vec3& orthos)
{
    for(int i = 0; i < nb_joints(); i++)
    {
        if(i == _root  || is_leaf(i))
        {
            half_angles[i] = Vec3(0.f, 0.f, 0.f);
            orthos     [i] = Vec3(0.f, 0.f, 0.f);
        }
        else
        {
            Vec3 null  = Vec3::zero();
            Vec3 half  = Vec3::zero();
            Vec3 ortho = Vec3::zero();
            Vec3 v0    = joint_pos(_parents[i]) - joint_pos(i);

            const std::vector<int>& sons = get_sons(i);
            for(unsigned p = 0; p < sons.size(); ++p)
            {
                Vec3 v1 = joint_pos(sons[p]) - joint_pos(i);

                Vec3 temp = v0.cross(v1);
                if(temp.norm() >= 0.0001f)
                {
                    half  = half  + (v0 + v1);
                    ortho = ortho + temp;
                }
            }

            half  = half  * (1.f / (float)sons.size());
            ortho = ortho * (1.f / (float)sons.size());

            half_angles[i] = half. norm() > 0.0001f ? half. normalized() : null;
            orthos     [i] = ortho.norm() > 0.0001f ? ortho.normalized() : null;
        }
    }
}

// -----------------------------------------------------------------------------

void Skeleton::rec_to_string(int id, int depth, std::string& str)
{
    for (int i = 0; i < depth; ++i)
        str += "    ";

    str += "Bone: " + Std_utils::to_string(id) + " ";
    str += EBone::type_to_string( bone_type(id) ) + "\n";

    for(unsigned i = 0; i < _children[id].size(); ++i)
        rec_to_string( _children[id][i], depth+1, str);
}

// -----------------------------------------------------------------------------

std::string Skeleton::to_string()
{
    std::string str;
    rec_to_string(root(), 0, str);
    return str;
}

// -----------------------------------------------------------------------------

//void Skeleton::set_joint_controller(Blending_env::Ctrl_id i,
//                                    const IBL::Ctrl_shape& shape)
//{
//    assert( i >= 0);
//    assert( i < _nb_joints);
//
//    _controllers[i] = shape;
//    Blending_env::update_controller(_joints_data[i]._ctrl_id, shape);
//}

// -----------------------------------------------------------------------------

void Skeleton::set_joint_blending(int i, EJoint::Joint_t type)
{
    assert( i >= 0);
    assert( i < _nb_joints);

    //_joints_data[i]._blend_type = type;
    //Skeleton_env::update_joints_data(_skel_id, _joints_data);
}

// -----------------------------------------------------------------------------

void Skeleton::set_joint_bulge_mag(int i, float m)
{
    assert( i >= 0);
    assert( i < _nb_joints);

    //_joints_data[i]._bulge_strength = std::min(std::max(m, 0.f), 1.f);
    //Skeleton_env::update_joints_data(_skel_id, _joints_data);
}

// -----------------------------------------------------------------------------

void Skeleton::set_bone(int i, Bone* b)
{
    assert(i < _nb_joints);
    assert(i >= 0);

    b->_bone_id = i;

    delete _anim_bones[i];
    _anim_bones[i] = b;

    update_anim_pose();
}

// -----------------------------------------------------------------------------

void Skeleton::set_bone_radius(int i, float radius)
{
    _anim_bones[i]->set_radius(radius);
    update_anim_pose();
}

// -----------------------------------------------------------------------------

//IBL::Ctrl_shape Skeleton::get_joint_controller(int i)
//{
//    assert( i >= 0);
//    assert( i < _nb_joints);
//    return _controllers[i];
//}

// -----------------------------------------------------------------------------

void Skeleton::set_bone_hrbf_radius(int i, float radius)
{
	cout<<"set_bone_hrbf_radius"<<endl;
	//_hrbf_radius[i] = radius;

	//if(bone_type(i) == EBone::HRBF)
	//{
	//	((Bone_hrbf*)_anim_bones[i])->set_hrbf_radius(radius);
	//	update_anim_pose();
	//}
}

// -----------------------------------------------------------------------------

//int Skeleton::get_hrbf_id(EBone::Id bone_id) const
//{
//    assert(bone_id >= 0);
//    assert(bone_id < _nb_joints);
//    if(bone_type(bone_id) == EBone::HRBF)
//        return ((const Bone_hrbf*)_anim_bones[bone_id])->get_hrbf().get_id();
//    else
//        return -1;
//}

// -----------------------------------------------------------------------------

float Skeleton::get_hrbf_radius(EBone::Id bone_id)
{
    return _hrbf_radius[bone_id];
}

// -----------------------------------------------------------------------------

void Skeleton::set_joint_rest_pos(int joint_id, const Point3& pt)
{
    _frames[joint_id].set_translation( pt.to_vec3() );
    _lcl_frames[joint_id] = _frames[joint_id].fast_invert();
    fill_bones();
    update_anim_pose();
}

// -----------------------------------------------------------------------------

void Skeleton::set_offset_scale(const Vec3& offset, float scale)
{
    _offset = offset;
    _scale  = scale;
    for(int i = 0; i < _nb_joints; i++ )
    {
        Transfo tr = Transfo::scale(scale) * Transfo::translate(offset) * _frames[i];
        _frames[i] = tr;
        _lcl_frames[i] = tr.fast_invert();
        _anim_bones[i]->set_length( _anim_bones[i]->length() * scale );
    }
    fill_bones();
    update_anim_pose();
}

// -----------------------------------------------------------------------------

void Skeleton::draw(const Tbx::Camera& cam,
                    const std::vector<int>& selected_joints,
                    bool rest_pose)
{
    subdraw( cam, selected_joints, rest_pose);

    glColor3f(0.f, 0.f, 0.7f);
    glLineStipple(5, 0xAAAA);
    GLEnabledSave save_line(GL_LINE_STIPPLE, true, true);
    for(int i = 0; i < _nb_joints; i++)
    {
        if(i == _root) continue;
        const Point3 p0 = _anim_bones[i]->org();
        const Point3 p1 = _anim_bones[_parents[i]]->end();
        glBegin(GL_LINES);
        glVertex3f(p0.x, p0.y, p0.z);
        glVertex3f(p1.x, p1.y, p1.z);
        glEnd();
    }
}

// -----------------------------------------------------------------------------

Vec3 Skeleton::joint_pos(int joint) const {
    assert(joint >= 0        );
    assert(joint <  _nb_joints);
    return _anim_frames[joint].get_translation();
}

// -----------------------------------------------------------------------------

Vec3 Skeleton::joint_rest_pos(int joint){
    assert(joint >= 0        );
    assert(joint <  _nb_joints);
    return _frames[joint].get_translation();
}

// -----------------------------------------------------------------------------

int Skeleton::select_joint(const Tbx::Camera &cam,
                           float x,
                           float y,
                           bool rest_pose)
{

    GLfloat m[16];
    glGetFloatv(GL_PROJECTION_MATRIX, m);
    _pick.begin(m, (GLfloat)x, (GLfloat)y);
    subdraw( cam, std::vector<int>(), rest_pose);
    return _pick.end();
}

// -----------------------------------------------------------------------------


void Skeleton::save_pose(const std::string& filepath)
{
    write_array( &(_h_transfos[0]), _h_transfos.size(), filepath );
}

// -----------------------------------------------------------------------------

void Skeleton::load_pose(const std::string& filepath)
{
    read_array( &(_h_transfos[0]), _h_transfos.size(), filepath );
    update_bones_pose( _h_transfos );
}

// -----------------------------------------------------------------------------

void Skeleton::transform_hrbf(const std::vector<Transfo>& d_global_transfos)
{
	//for (int i = 0; i < nb_joints(); ++i)
	//{
	//	const int id = get_hrbf_id(i);
	//	if( id > -1) HRBF_env::set_transfo(id, _h_transfos[i]);
	//}

	//HRBF_env::apply_hrbf_transfos();
}

// -----------------------------------------------------------------------------

void Skeleton::transform_precomputed_prim(const HPLA_tr &global_transfos )
{

    //for( int i = 0; i < _nb_joints; i++)
    //    if(bone_type(i) == EBone::PRECOMPUTED)
    //    {
    //        Precomputed_prim& prim = ((Bone_precomputed*)_anim_bones[i])->get_primitive();
    //        Precomputed_env::set_transform(prim.get_id(), global_transfos[i]);
    //    }

    //Precomputed_env::update_device_transformations();
}

// -----------------------------------------------------------------------------

void Skeleton::update_anim_pose()
{
    // Compute the global transformation of each joint
    _kinec->compute_transfo_gl( &_h_transfos[0] );
    // update animated pose
    update_bones_pose( _h_transfos );
}

// -----------------------------------------------------------------------------

void Skeleton::update_bones_pose(const HPLA_tr& global_transfos)
{
    // Update joints position in animated position and the associated
    // transformations
    subupdate_vertices( _root, global_transfos );

    // Update the dual quaternion representation of the transformations from
    // the matrices
    for(int i = 0; i < nb_joints(); i++)
    {
        Dual_quat_cu dq = Dual_quat_cu(global_transfos[i]);
        _h_dual_quat[i] = dq;
    }

    // Update joint positions in texture.
	//std::copy(global_transfos.begin() ,global_transfos.end(), _d_transfos.begin() );
	_d_transfos = global_transfos;
//	std::copy( _h_dual_quat.begin() , _h_dual_quat.end(), _d_dual_quat.begin() );
	_d_dual_quat = _h_dual_quat;
    //_d_transfos. copy_from( global_transfos );
    //_d_dual_quat.copy_from( _h_dual_quat    );

    transform_hrbf( _d_transfos );
    transform_precomputed_prim( global_transfos );

    // In order to this call to take effect correctly it MUST be done after
    // transform_hrbf() and transform_precomputed_prim() otherwise bones
    // positions will not be updated correctly within the Skeleton_env.
    //Skeleton_env::update_bones_data(_skel_id, _anim_bones);
    // --------
}

// -----------------------------------------------------------------------------

void Skeleton::subupdate_vertices( int root,
                                   const HPLA_tr& global_transfos)
{
    const Transfo tr = global_transfos[root];
    _anim_frames[root] = tr * _frames[root];

    Bone_cu b = _bones[root];
    _anim_bones[root]->set_length( b.length() );
    _anim_bones[root]->set_orientation(tr * b.org(), tr * b.dir());

    for(unsigned i = 0; i < _children[root].size(); i++)
        subupdate_vertices(_children[root][i], global_transfos);
}

// -----------------------------------------------------------------------------

void Skeleton::fill_children(Graph& g, int root)
{
    std::vector<int> to_pop;
    to_pop.reserve( 2 * g.nb_edges() );
    for(int i = 0; i < (int)g.nb_edges(); i++)
    {
        const Graph::Edge& e = g._edges[i];
        if(e.a == root){
            to_pop.push_back(e.b);
            _children[root].push_back(e.b);
            _parents[e.b] = root;
            Std_utils::pop(g._edges, i);
            i = -1;
        } else {
            if(e.b == root){
                to_pop.push_back(e.a);
                _children[root].push_back(e.a);
                _parents[e.a] = root;
                Std_utils::pop(g._edges, i);
                i = -1;
            }
        }
    }

    for(unsigned i = 0; i < to_pop.size(); i++)
        fill_children(g, to_pop[i]);
}

// -----------------------------------------------------------------------------

void Skeleton::fill_frames(const Graph& g)
{
    for(int i = 0; i < _nb_joints; i++)
    {
        Vec3 org = g.get_vertex( i );
        Vec3 end = Vec3::zero();
        int nb_sons = _children[i].size();
        for(int s = 0; s < nb_sons; s++)
        {
            int sid = _children[i][s];
            end += g.get_vertex( sid );
        }

        Vec3 x, y, z;
        if( nb_sons > 0){
            end /= (float)nb_sons;
            x = (end - org).normalized();
        }else
            x = Vec3::unit_x();

        x.coordinate_system(y, z);

        Transfo tr(Mat3(x, y, z), org);
        _frames    [i] = tr;
        _lcl_frames[i] = tr.fast_invert();
    }
}

// -----------------------------------------------------------------------------

void Skeleton::fill_bones()
{
    for(int bid = 0; bid < _nb_joints; bid++)
    {
        Vec3 org = _frames[bid].get_translation();
        Vec3 end = Vec3::zero();
        int nb_sons = _children[bid].size();
        for(int s = 0; s < nb_sons; s++)
        {
            //int sid = _children[bid][s];
            //end += _frames[sid].get_translation();
			end += _frames[_children[bid][0]].get_translation();
        }
        end /= (float)nb_sons;

        if(nb_sons == 0 ){
            // We set a minimal length for the leaves
            _bones[bid] = Bone_cu(org.to_point3(), _frames[bid].x(), 0.1f, 0.f);
            _anim_bones[bid]->set_length( 0.1f );
        }else{
            _bones[bid] = Bone_cu(org.to_point3(), end.to_point3(), 0.f);
            _anim_bones[bid]->set_length( (org-end).norm() );
        }

    }
}

// -----------------------------------------------------------------------------

static void custom_bone( float l  /*bone_length*/,
                         float js /*joint radius*/,
                         float ps /* parent joint radius*/)

{
    //l = 1.f;
    const float b0 = ps + (l / 10.f); // length of the base
    const float w0 = l / 15.f;        // width of the base
    // First pyramid
    glBegin( GL_TRIANGLE_FAN );
    {
        glVertex3f( ps, 0.f, 0.f );
        glVertex3f( b0, 0.f, -w0 );
        glVertex3f( b0, -w0, 0.f );
        glVertex3f( b0, 0.f,  w0 );
        glVertex3f( b0,  w0, 0.f );
        glVertex3f( b0, 0.f, -w0 );
    }
    glAssert( glEnd() );

    const float w1 = w0 / 3.f; // Width of the base at the opposite
    l = l-js;
    glBegin( GL_QUAD_STRIP );
    {
        glVertex3f( b0, 0.f, -w0 );// a
        glVertex3f(  l, 0.f, -w1 );// 0
        glVertex3f( b0, -w0, 0.f );// b
        glVertex3f(  l, -w1, 0.f );// 1
        glVertex3f( b0, 0.f,  w0 );// c
        glVertex3f(  l, 0.f,  w1 );// 2
        glVertex3f( b0,  w0, 0.f );// d
        glVertex3f(  l,  w1, 0.f );// 3
        glVertex3f( b0, 0.f, -w0 );// a
        glVertex3f(  l, 0.f, -w1 );// 0
    }
    glAssert( glEnd() );

    // The bone's cap is flat
    glBegin( GL_QUADS );
    {
        glVertex3f( l, 0.f, -w1 );
        glVertex3f( l, -w1, 0.f );
        glVertex3f( l, 0.f,  w1 );
        glVertex3f( l,  w1, 0.f );
    }
    glAssert( glEnd() );
}

// -----------------------------------------------------------------------------

static void draw_bone(const Point3& p0,
                      const Point3& p1,
                      float rad_joint,
                      float rad_pjoint)
{
    glPushMatrix();
    Vec3 fx = (p1-p0).normalized(), fy, fz;
    fx.coordinate_system(fy, fz);
    Transfo tr( Mat3(fx, fy, fz), p0.to_vec3() );

    glMultMatrixf( tr.transpose().m );

    custom_bone( (p1-p0).norm(), rad_joint, rad_pjoint);
    glPopMatrix();
}

// -----------------------------------------------------------------------------

static void draw_joint( float fc )
{
    glPushMatrix();
    glScalef(fc, fc, fc);
    g_primitive_printer.draw( g_sphere_lr_vbo );
    glPopMatrix();
}

// -----------------------------------------------------------------------------

static void draw_frame(const Transfo& frame, float size_axis)
{
    Transfo tr = frame.normalized();
    Point3 pos = tr.get_translation().to_point3();

    Point3 dx = pos + tr.x() * size_axis;
    Point3 dy = pos + tr.y() * size_axis;
    Point3 dz = pos + tr.z() * size_axis;

    glBegin(GL_LINES);{
        // Local frame
        glColor4f(1.f, 0.f, 0.f, 1.f);
        glVertex3f(pos.x, pos.y, pos.z);
        glVertex3f(dx.x, dx.y, dx.z);
        glColor4f(0.f, 1.f, 0.f, 1.f);
        glVertex3f(pos.x, pos.y, pos.z);
        glVertex3f(dy.x, dy.y, dy.z);
        glColor4f(0.f, 0.f, 1.f, 1.f);
        glVertex3f(pos.x, pos.y, pos.z);
        glVertex3f(dz.x, dz.y, dz.z);
    }glAssert( glEnd() );

}

// -----------------------------------------------------------------------------

static void set_color_selection(bool is_select,
                                const std::vector<int>& selected_joints,
                                int id)
{
    if( is_select ){
        if(selected_joints[selected_joints.size()-1] == id)
            glAssert( glColor4f(1.f, 1.f, 1.f, 1.f) );
        else
            glAssert( glColor4f(1.f, 1.0f, 0.2f, 1.f) );
    }
}

// -----------------------------------------------------------------------------

void Skeleton::subdraw(const Tbx::Camera& cam,
                       const std::vector<int>& selected_joints,
                       bool rest_pose)
{
    for(int i = 0; i < _nb_joints; i++)
    {

        const bool  is_select = Std_utils::exists(selected_joints, i);
        const Bone* bone      = _anim_bones[i];
        const float len       = _anim_bones[i]->length();
        const float rad       = (len / 30.f);

        const Transfo b_frame = rest_pose ? _frames[i] : _anim_frames[i];
        const Point3 org = bone->org();
        const Point3 end = bone->end();

        // draw joint with the joint's id for selection
        _pick.set_name( i );

        glAssert( glMatrixMode(GL_MODELVIEW) );
        glAssert( glPushMatrix() );
        {
            glAssert( glTranslatef(org.x, org.y, org.z) );

            glAssert( glColor4f(1.f, 0.f, 0.0f, 1.f) );
            set_color_selection(is_select, selected_joints, i);

            draw_joint( rad );
        }
        glAssert( glPopMatrix() );

        const float axis_size = 0.03f;
        // -----------


        // draw bone with name
        GLLineWidthSave save_line_width( 2.0f );
        if( _pick.is_pick_init() ) glAssert( glLineWidth(5.f) );


        if( is_leaf(i) )
        {
            draw_frame(b_frame, axis_size);
            if(_parents[i] >= 0)
            {
                glAssert( glPushMatrix() );
                glAssert( glTranslatef(org.x, org.y, org.z) );
                glAssert( glColor4f(0.f, 0.1f, 0.8f, 1.f) );
                set_color_selection(is_select, selected_joints, i);
                draw_joint(_anim_bones[_parents[i]]->length() / 50.f);
                glAssert( glPopMatrix() );
            }
        }
        else
        {
            // Draw bone wires ---------------
            {
                glAssert( glColor4f(0.f, 0.f, 0.f, 1.f) );
                set_color_selection(is_select, selected_joints, i);

                GLPolygonModeSave save_poly_mode;
                glAssert( glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) );
                glPushMatrix();
                // offset a little the wires so they won't be hidden
                Vec3 p = cam.get_pos();
                glTranslatef(p.x, p.y, p.z);
                const float eps = 1.f - 0.001f;
                glScalef(eps, eps, eps);
                glTranslatef(-p.x, -p.y, -p.z);
                draw_bone(org, end, rad, rad);
                glPopMatrix();
            }
            // -------------------------------

            // Draw bone faces ------------
            Color c = Color::pseudo_rand(i);
            glAssert( glColor4f(c.r, c.g, c.b, 1.f) );
            draw_bone( org, end, rad, rad);
            // ----------------------------
        }
    }
}

// -----------------------------------------------------------------------------

//Skeleton_env::DBone_id Skeleton::get_bone_device_idx(EBone::Id i) const {
//    return Skeleton_env::bone_hidx_to_didx(_skel_id, i);
//}

// -----------------------------------------------------------------------------

//EBone::Id Skeleton::get_bone_idx(Skeleton_env::DBone_id d_idx) const {
//    return Skeleton_env::bone_didx_to_hidx(_skel_id, d_idx);
//}

// -----------------------------------------------------------------------------

int Skeleton::get_nb_bone_of_type(EBone::Bone_t type)
{
    int acc = 0;
    for(int i = 0; i < _nb_joints; i++)
        if(bone_type(i) == type)
            acc++;

    return acc;
}

// -----------------------------------------------------------------------------

const Transfo&  Skeleton::get_transfo(EBone::Id bone_id) const {
    assert(bone_id >= 0);
    assert(bone_id < _nb_joints);
    return _h_transfos[bone_id];
}

// -----------------------------------------------------------------------------

const Dual_quat_cu&  Skeleton::get_dual_quat(EBone::Id bone_id) const {
    assert(bone_id >= 0);
    assert(bone_id < _nb_joints);
    return _h_dual_quat[bone_id];
}

// -----------------------------------------------------------------------------

