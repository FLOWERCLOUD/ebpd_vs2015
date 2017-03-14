#include "animated_mesh_ctrl.hpp"

#include "../animation/animesh.hpp"
#include "../global_datas/toolglobals.hpp"
#include "../global_datas/cuda_globals.hpp"
#include "../control/cuda_ctrl.hpp"
#include "toolbox/std_utils/vector.hpp"
#include "../parsers/loader_skel.hpp"
#include "../animation/skeleton.hpp"
//#include "../meshes/vcg_lib/utils_sampling.hpp"
#include "meshes/mesh_utils_loader.hpp"
#include "meshes/mesh_utils.hpp"
#include "parsers/obj_loader.hpp"

#include <iostream>
using std::cout;
using std::endl;
using namespace Tbx;
// -----------------------------------------------------------------------------

Animated_mesh_ctrl::Animated_mesh_ctrl(Animesh* am) :
    _incremental_deformation( false ),
    _display(true),
    _display_points(false),
    _do_smooth(false),
    _draw_rot_axis(false),
    _auto_precompute(true),
    _factor_bones(false),
    _nb_iter(7),
    _blending_type((int)EAnimesh::DUAL_QUAT_BLENDING),
    _d_selected_points(0),
    _bone_caps(am->get_skel()->nb_joints()),
    _bone_anim_caps(am->get_skel()->nb_joints()),
    _sample_list(am->get_skel()->nb_joints()),
    _sample_anim_list(am->get_skel()->nb_joints()),
    _animesh(am),
    _skel( am->get_skel() )
{
    int n = am->get_skel()->nb_joints();
    for(int i = 0; i < n; i++){
        _bone_caps[i].jcap.enable = false;
        _bone_caps[i].pcap.enable = false;
        _bone_anim_caps[i].jcap.enable = false;
        _bone_anim_caps[i].pcap.enable = false;
    }

    _normals_bo = new GlBuffer_obj<Vec3>(GL_ARRAY_BUFFER);
    _rest_nbo   = new GlBuffer_obj<Vec3>(GL_ARRAY_BUFFER);
    _anim_bo    = new GlBuffer_obj<Vec3>(GL_ARRAY_BUFFER);
    _rest_bo    = new GlBuffer_obj<Vec3>(GL_ARRAY_BUFFER);
    _color_bo   = new GlBuffer_obj<Vec4>(GL_ARRAY_BUFFER);
}

// -----------------------------------------------------------------------------

Animated_mesh_ctrl::~Animated_mesh_ctrl()
{
	_d_selected_points.clear();
    delete _normals_bo;
    delete _anim_bo;
    delete _rest_bo;
    delete _color_bo;
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::draw_rotation_axis()
{
    if( !_draw_rot_axis || _skel == 0 || _animesh == 0)
        return;

    const int nb_joints = _skel->nb_joints();
    std::vector<Vec3> h_rot (nb_joints);
    std::vector<Vec3> h_half(nb_joints);
    _skel->compute_joints_half_angles(h_half, h_rot);

    glAssert( glColor3f(1.f, 0.f, 0.f) );
    glBegin(GL_LINES);
    for(int i = 0; i < _skel->nb_joints(); i++)
    {
        Vec3 v   = _skel->joint_pos(i);
        Vec3 rot = h_rot[i].normalized()*1.f;

        glVertex3f(v.x - rot.x, v.y - rot.y, v.z - rot.z);
        glVertex3f(v.x + rot.x, v.y + rot.y, v.z + rot.z);
    }
    glEnd();
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::set_bone_type(int bone_id, int bone_type)
{
    if(bone_type == _skel->bone_type(bone_id)) return;

    if(bone_type == EBone::HRBF)
    {
        _animesh->set_bone_type(bone_id, bone_type);
        update_bone_samples(bone_id);
    }
    else
        _animesh->set_bone_type(bone_id, bone_type);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::set_do_smoothing(bool state)
{
    if(_animesh != 0){
        _animesh->set_smooth_mesh(state);
        _do_smooth = state;
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::set_draw_rot_axis(bool state)
{
    _draw_rot_axis = state;
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::set_smooth_factor(int i, float fact){
    _animesh->set_smooth_factor(i, fact);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::set_ssd_weight(int id_vertex,
                                        int id_joint,
                                        float weight)
{
    _animesh->set_ssd_weight(id_vertex, id_joint, weight);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::set_ssd_weight(const Loader::Abs_skeleton& skel)
{
    if( skel._weights.size() == 0 ) return;

    std::vector<std::map<int, float> > weights( skel._weights.size() );

    for(unsigned i = 0; i < skel._weights.size(); i++)
    {
        const std::vector< std::pair<int, float> >& bones = skel._weights[i];
        for(unsigned j = 0; j < bones.size(); j++)
            weights[i][ bones[j].first ] = bones[j].second;
    }

    if( _animesh->_mesh->get_nb_vertices() == (int)skel._weights.size() )
    {
        _animesh->set_ssd_weights( weights );
        _animesh->update_host_ssd_weights();
    }
    else
    {
        std::cerr << "ERROR: can't load skinning weights" << std::endl;
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::init_rigid_ssd_weights()
{
    _animesh->init_rigid_ssd_weights();

    const std::vector<int>& set = Cuda_ctrl::_skeleton.get_selection_set();
    if( _animesh->get_color_type() == EAnimesh::SSD_WEIGHTS && set.size() > 0)
        _animesh->set_color_ssd_weight( set[set.size()-1] );
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::switch_do_smoothing(){
    set_do_smoothing(!_do_smooth);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::enable_update_base_potential(bool state)
{
    _animesh->set_enable_update_base_potential(state);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::do_ssd_skinning(){
    _blending_type = EAnimesh::MATRIX_BLENDING;
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::do_dual_quat_skinning(){
    _blending_type = EAnimesh::DUAL_QUAT_BLENDING;
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::set_implicit_skinning(bool s){
    _animesh->set_implicit_skinning(s);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::switch_implicit_skinning(){
    _animesh->switch_implicit_skinning();
}
// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::update_base_potential()
{
    assert(_animesh != 0);
    _animesh->update_base_potential();
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::update_clusters( EAnimesh::Cluster_type type )
{
    _animesh->clusterize(type);

    const std::vector<int>& set = Cuda_ctrl::_skeleton.get_selection_set();
    if( _animesh->get_color_type() == EAnimesh::SSD_WEIGHTS )
    {
        if( set.size() > 0 )
            _animesh->set_color_ssd_weight( set[set.size()-1] );
    }else
        _animesh->set_colors(_animesh->get_color_type());
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::deform_mesh(bool refresh)
{
    if( _incremental_deformation )
        _animesh->transform_vertices_incr( refresh );
    else
        _animesh->transform_vertices((EAnimesh::Blending_type)get_blending_type(), refresh);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::reset_verts_to_rest_pose()
{
    _animesh->reset_vertices();
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::topology_diffuse_ssd_weights(float alpha, int nb_iter)
{
    _animesh->topology_diffuse_ssd_weights(nb_iter, alpha);

    const std::vector<int>& set = Cuda_ctrl::_skeleton.get_selection_set();
    if( _animesh->get_color_type() == EAnimesh::SSD_WEIGHTS && set.size() > 0)
            _animesh->set_color_ssd_weight( set[set.size()-1] );
}

void Animated_mesh_ctrl::geodesic_diffuse_ssd_weights(float alpha, int nb_iter)
{
    _animesh->geodesic_diffuse_ssd_weights(nb_iter, alpha);

    const std::vector<int>& set = Cuda_ctrl::_skeleton.get_selection_set();
    if( _animesh->get_color_type() == EAnimesh::SSD_WEIGHTS && set.size() > 0)
            _animesh->set_color_ssd_weight( set[set.size()-1] );
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::heat_diffuse_ssd_weights(float heat)
{
    _animesh->heat_diffuse_ssd_weights(heat);

    const std::vector<int>& set = Cuda_ctrl::_skeleton.get_selection_set();
    if( _animesh->get_color_type() == EAnimesh::SSD_WEIGHTS && set.size() > 0)
            _animesh->set_color_ssd_weight( set[set.size()-1] );
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::write_bone_types(std::ofstream& file)
{
    file << "[BONE_TYPES]" << std::endl;
    file << "nb_bone "     << _skel->nb_joints() << std::endl;

    for(int i = 0; i < _skel->nb_joints(); i++ )
    {
        file << "bone_id "   << i                         << std::endl;
        file << "bone_type " << _skel->bone_type( i ) << std::endl;
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::write_hrbf_env(std::ofstream& file)
{
    file << "[HRBF_ENV]" << std::endl;
    file << "nb_bone "   << _sample_list.size() << std::endl;

    for(unsigned i = 0; i < _sample_list.size(); i++ )
    {
        file << "bone_id " << i << std::endl;

        write_samples(file, _sample_list[i].nodes, _sample_list[i].n_nodes);
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::write_hrbf_caps_env(std::ofstream& file, bool jcap)
{
    if( jcap ) file << "[HRBF_JCAPS_ENV]" << std::endl;
    else       file << "[HRBF_PCAPS_ENV]" << std::endl;

    file << "nb_bone " << _bone_caps.size() << std::endl;

    for(unsigned i = 0; i < _bone_caps.size(); i++ )
    {
        const Cap&  cap_list = jcap ? _bone_caps[i].jcap : _bone_caps[i].pcap;
        const float rad      = _animesh->get_junction_radius(i);

        file << "bone_id "       << i               << std::endl;
        file << "is_cap_enable " << cap_list.enable << std::endl;
        file << "cap_radius "    << rad             << std::endl;
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::write_samples(std::ofstream& file,
                                       const std::vector<Vec3>& nodes,
                                       const std::vector<Vec3>& n_nodes )
{
    assert(nodes.size() == n_nodes.size());

    file << "nb_points " << nodes.size() << std::endl;

    for(unsigned j = 0; j < nodes.size(); ++j)
    {
        const Vec3 pt = nodes  [j];
        const Vec3 n  = n_nodes[j];
        file << pt.x << " " << pt.y << " " << pt.z << std::endl;
        file << n.x  << " " << n.y  << " " << n.z  << std::endl;
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::write_ssd_is_lerp( std::ofstream& file )
{
    std::vector<float> lerp_facts;
    _animesh->get_ssd_lerp( lerp_facts );
    int size = (int)lerp_facts.size();

    file << "[SSD_IS_LERP]" << std::endl;
    file << "nb_points " << size << std::endl;
    for(int i = 0; i < size; ++i)
        file << lerp_facts[i] << std::endl;
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::write_hrbf_radius( std::ofstream& file )
{
    //file << "[HRBF_RADIUS]" << std::endl;
    //file << "nb_bone "      << _skel->nb_joints() << std::endl;

    //for(int i = 0; i < _skel->nb_joints(); i++ )
    //{
    //    file << "bone_id "     << i                         << std::endl;
    //    file << "hrbf_radius " << _skel->get_hrbf_radius(i) << std::endl;
    //}
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::save_ism(const char* filename)
{
    using namespace std;
    ofstream file(filename, ios_base::out|ios_base::trunc);

    if(!file.is_open()){
        cerr << "Error exporting file " << filename << endl;
        exit(1);
    }

    write_hrbf_env( file );
    write_hrbf_caps_env( file, true  /*write joint caps*/);
    write_hrbf_caps_env( file, false /*write parent caps*/);
    write_bone_types( file );
    write_ssd_is_lerp( file );
    write_hrbf_radius( file );

    file.close();
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::read_bone_types(std::ifstream& file,
                                         std::vector<int>& bones_type)
{
    std::string nil;
    int nb_bones = -1;

    file >> nil /*'nb_bone'*/ >> nb_bones;

    for(int i = 0; i < nb_bones; i++)
    {
        int bone_id = -1;
        file >> nil /*'bone_id'*/    >> bone_id;
        file >> nil /*'bone_type '*/ >> bones_type[bone_id];
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::read_samples(std::ifstream& file,
                                      std::vector<Vec3>& nodes,
                                      std::vector<Vec3>& n_nodes )
{
    assert(nodes.size() == n_nodes.size());

    std::string nil;
    int nb_samples = -1;
    file >> nil /*'nb_points'*/ >> nb_samples;
    nodes.  resize( nb_samples );
    n_nodes.resize( nb_samples );

    for(int j = 0; j < nb_samples; ++j)
    {
        Vec3 pt, n;
        file >> pt.x >> pt.y >> pt.z;
        file >> n.x  >> n.y  >> n.z;

//        Transfo tr = Transfo::rotate(Vec3::unit_x(), M_PI) * Transfo::rotate(Vec3::unit_z(), M_PI);// DEBUG---
        nodes  [j] = /*tr */ pt;
        n_nodes[j] = /*tr */ n;
    }
}

// -----------------------------------------------------------------------------

//void Animated_mesh_ctrl::read_weights(std::ifstream& file,
//                                      std::vector<Vec4>& weights )
//{
//    std::string nil;
//    int nb_samples = -1;
//    file >> nil /*'nb_points'*/ >> nb_samples;
//    weights.resize( nb_samples );
//
//    for(int j = 0; j < nb_samples; ++j)
//    {
//        Vec3 beta;
//        float alpha;
//        file >> beta.x >> beta.y >> beta.z;
//        file >> alpha;
//
//        weights[j] = make_Vec4(beta.x, beta.y, beta.z, alpha);
//    }
//}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::read_hrbf_caps_env(std::ifstream& file, bool jcap)
{
    std::string nil;
    int nb_bones = -1;

    file >> nil /*'nb_bone'*/ >> nb_bones;

    for(int i = 0; i < nb_bones; i++ )
    {
        float rad;
        int bone_id = -1;
        file >> nil /*'bone_id'*/ >> bone_id;

        Cap& cap_list = jcap ? _bone_caps[bone_id].jcap : _bone_caps[bone_id].pcap;
        file >> nil /*'is_cap_enable'*/ >> cap_list.enable;
        file >> nil /*'cap_radius'*/    >> rad;
        _animesh->set_junction_radius(bone_id, rad);
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::read_hrbf_env(std::ifstream& file)
{
    std::string nil;
    int bone_id  = -1;
    int nb_bones_file = 0;

    file >> nil/*'nb_bone'*/ >> nb_bones_file;

    for(int i = 0; i < nb_bones_file; i++ )
    {
        file >> nil/*'bone_id'*/ >> bone_id;
        read_samples(file, _sample_list[bone_id].nodes, _sample_list[bone_id].n_nodes);
        Std_utils::copy( _sample_anim_list[bone_id].  nodes, _sample_list[bone_id].  nodes);
        Std_utils::copy( _sample_anim_list[bone_id].n_nodes, _sample_list[bone_id].n_nodes);
    }
}

// -----------------------------------------------------------------------------

//void Animated_mesh_ctrl::read_hrbf_env_weights(
//        std::ifstream& file,
//        std::vector<std::vector<Vec4> >& bone_weights)
//{
//    std::string nil;
//    int bone_id  = -1;
//    int nb_bones_file = 0;
//
//    file >> nil/*'nb_bone'*/ >> nb_bones_file;
//
//    for(int i = 0; i < nb_bones_file; i++ )
//    {
//        file >> nil/*'bone_id'*/ >> bone_id;
//        read_weights(file, bone_weights[i]);
//    }
//}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::read_ssd_is_lerp(std::ifstream& file)
{
    std::string nil;
    int nb_points = -1;
    file >> nil /* nb_points*/ >> nb_points;

    assert(_animesh->get_mesh()->get_nb_vertices() == nb_points);
    std::vector<float> factors(nb_points);
    for(int i = 0; i<nb_points; i++ )
        file >> factors[i];

    _animesh->set_ssd_is_lerp(factors);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::read_hrbf_radius(std::ifstream& file,
                                          std::vector<float>& radius_hrbf)
{
    std::string nil;
    int nb_bones = -1;

    file >> nil /*'nb_bone'*/ >> nb_bones;

    for(int i = 0; i < nb_bones; ++i)
    {
        int bone_id = -1;
        file >> nil /*'bone_id'*/     >> bone_id;
        file >> nil /*'bone_radius'*/ >> radius_hrbf[bone_id];
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::load_ism(const char* filename)
{
	cout<<"animated mesh ctrl: load ism"<<endl;
	return;
    using namespace std;
    ifstream file(filename);

    if(!file.is_open()) {
        cerr << "Error importing file " << filename << endl;
        return;
    }

    std::vector<int>   bones_type (_skel->nb_joints(), EBone::SSD);
    std::vector<float> radius_hrbf(_skel->nb_joints(), -1.f      );

    std::vector<std::vector<Vec4> > bone_weights(_skel->nb_joints());

    //while( !file.eof() )
    //{
    //    string section;
    //    file >> section;

    //    if(section == "[HRBF_ENV]")              read_hrbf_env( file );
    //    //else if(section == "[HRBF_ENV_WEIGHTS]") read_hrbf_env_weights(file, bone_weights);
    //    else if(section == "[HRBF_JCAPS_ENV]")   read_hrbf_caps_env( file, true  );
    //    else if(section == "[HRBF_PCAPS_ENV]")   read_hrbf_caps_env( file, false );
    //    else if(section == "[BONE_TYPES]")       read_bone_types( file, bones_type );
    //    else if(section == "[SSD_IS_LERP]")      read_ssd_is_lerp( file );
    //    else if(section == "[HRBF_RADIUS]")      read_hrbf_radius( file, radius_hrbf);
    //    else
    //    {
    //        std::cerr << "WARNING ism import: can't read this symbol '";
    //        std::cerr << section << "'" << std::endl;
    //    }
    //}
    //file.close();

    for(int i = 0; i < _skel->nb_joints(); i++)
    {
        const EBone::Bone_t t = (EBone::Bone_t)bones_type[i];

        set_bone_type(i, EBone::HRBF); // -> suppose to make things work and update better...
        if( radius_hrbf[i] > 0.f)
            set_hrbf_radius(i, radius_hrbf[i]);

        update_caps(i, true, true);

        if(t == EBone::HRBF || t == EBone::PRECOMPUTED)
        {
            // Check if weights are here and don't update if not necessary
            //...

            // This call will compute hrbf weights and convert to precomputed
            // if _auto_precompute == true
            update_bone_samples(i);

            if( !_auto_precompute && t == EBone::PRECOMPUTED)
                _animesh->set_bone_type( i, EBone::PRECOMPUTED);
        }
        else
            set_bone_type(i, t);
    }

    update_gl_buffers_size( compute_nb_samples() );
    _animesh->update_base_potential();
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::save_weights(const char* filename){
    _animesh->export_weights(filename);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::save_cluster(const char* filename){
    _animesh->export_cluster(filename);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::load_cluster(const char* filename){
    _animesh->import_cluster(filename);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::load_implicit_proxy(const char* filename,
                                             EBone::Id bone_id,
                                             int minDist,
                                             int nb_samples)
{
    // Load mesh:
    Mesh mesh;
    // Parse file
    Loader::Obj_file loader( filename );
    Loader::Abs_mesh abs_mesh;
    // compute abstract representation
    loader.get_mesh( abs_mesh );
    // load for opengl
    Mesh_utils::load_mesh( mesh, abs_mesh );

    Mesh_utils::translate(mesh, g_mesh->get_offset() );
    Mesh_utils::scale    (mesh, g_mesh->get_scale()  );
    std::cout << "scale proxy: "<< g_mesh->get_scale() << std::endl;

    std::vector<Vec3> in_verts, in_normals;
    mesh.get_vertices( in_verts );
    mesh.get_mean_normals( in_normals );

    std::vector<int> tris( mesh.get_nb_tris() * 3 );
    for(int i = 0; i < mesh.get_nb_tris(); ++i){
        for(int u = 0; u < 3; ++u)
            tris[i*3 + u] = mesh.get_tri( i )[u];
    }

    // Sample mesh:
    _sample_list[bone_id].nodes.  clear();
    _sample_list[bone_id].n_nodes.clear();

    //Utils_sampling::poisson_disk(minDist, nb_samples,
    //                             in_verts, in_normals, tris,
    //                             _sample_list[bone_id].nodes,
    //                             _sample_list[bone_id].n_nodes);
    // Compute implicit surface:
    update_bone_samples(bone_id);
    update_gl_buffers_size( compute_nb_samples() );
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::load_weights(const char* filename)
{
    int len = strlen(filename);
    bool has_commas = (filename[len-4] == '.') &
            (filename[len-3] == 'c') &
            (filename[len-2] == 's') &
            (filename[len-1] == 'v');

    _animesh->read_weights_from_file(filename, has_commas);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::select(int vert_id)
{
    add_to_selection( vert_id );
    update_device_selection();
}

// -----------------------------------------------------------------------------

bool Animated_mesh_ctrl::select(const Camera& cam,
                                int x, int y,
                                Select_type<int>* heuristic,
                                bool rest_pose)
{
    y = cam.height() - y;
    _animesh->select(x, y, heuristic, rest_pose);
    int  nb_selected   = heuristic->nb_selected();
    int* selection_set = heuristic->get_selected_points();
    for(int i=0; i<nb_selected; i++)
        add_to_selection( selection_set[i] );

    update_device_selection();

    return nb_selected > 0;
}

// -----------------------------------------------------------------------------

// TODO: don't pass the camera, x, y correctness is user responsibility
bool Animated_mesh_ctrl::unselect(const Tbx::Camera& cam,
                                  int x, int y,
                                  Select_type<int>* heuristic,
                                  bool rest_pose)
{
    y = cam.height() - y;
    _animesh->select(x, y, heuristic, rest_pose);
    int  nb_selected   = heuristic->nb_selected();
    int* selection_set = heuristic->get_selected_points();
    for(int i=0; i<nb_selected; i++)
        remove_from_selection( selection_set[i] );

    update_device_selection();

    return nb_selected > 0;
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::reset_selection()
{
    Color cl = Cuda_ctrl::_color.get(Color_ctrl::MESH_POINTS);
    for(unsigned int i=0; i<_selected_points.size(); i++)
        g_mesh->set_point_color_bo(_selected_points[i], cl.r, cl.g, cl.b, cl.a);

    _selected_points.clear();
	_d_selected_points.clear();
}

// -----------------------------------------------------------------------------

Vec3 Animated_mesh_ctrl::cog_mesh_selection()
{
    Vec3 cog = Vec3::zero();

    int size = _selected_points.size();
    const Mesh* m = _animesh->get_mesh();
    for(int i = 0; i < size; i++)
    {
        Vec3 v = m->get_vertex(_selected_points[i]);
        cog = v + cog;
    }

    cog = cog / (float)(size == 0 ? 1 : size);
    return cog;
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::add_to_selection(int id)
{
    std::cout << "vertex id" << id << std::endl;
    // Check for doubles
    bool state = false;
    for(unsigned int i=0; i<_selected_points.size(); i++)
        state = state || (_selected_points[i] == id);

    if(!state)
    {
        Color cl = Cuda_ctrl::_color.get(Color_ctrl::MESH_SELECTED_POINTS);
        _selected_points.push_back(id);
        g_mesh->set_point_color_bo(id, cl.r, cl.g, cl.b, cl.a);
        std::cout << "Vertex ID" << id << std::endl;
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::remove_from_selection(int id)
{
    std::vector<int>::iterator it = _selected_points.begin();
    unsigned int i = 0;
    for(; it<_selected_points.end(); ++it, ++i)
        if( (*it) == id )
            break;

    if(i < _selected_points.size())
    {
        Color cl = Cuda_ctrl::_color.get(Color_ctrl::MESH_POINTS);
        g_mesh->set_point_color_bo(*it, cl.r, cl.g, cl.b, cl.a);
        _selected_points.erase(it);
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::update_device_selection()
{
    _d_selected_points.clear();

    int size = _selected_points.size();
    if(size)
    {
		_d_selected_points.resize(size);
        //Cuda_utils::malloc_d(_d_selected_points, size);
		_d_selected_points = _selected_points;
        //Cuda_utils::mem_cpy_htd(_d_selected_points,
        //                        &(_selected_points[0]),
        //                        size );
    }

}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::set_nb_iter_smooting(int nb_iter)
{
    if(_animesh != 0){
        _animesh->set_smoothing_iter(nb_iter);
        _nb_iter = nb_iter;
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::color_vertex(int vert,
                                      float r, float g, float b, float a)
{
    g_mesh->set_point_color_bo(vert, r, g, b, a);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::color_uniform(float r, float g, float b, float a){
    _animesh->set_colors(EAnimesh::USER_DEFINED, r, g, b, a);
}
void Animated_mesh_ctrl::color_ssd_weights(int joint_id){
    _animesh->set_color_ssd_weight(joint_id);
}
void Animated_mesh_ctrl::color_type(EAnimesh::Color_type type){
    _animesh->set_colors(type);
}
//void Animated_mesh_ctrl::smooth_conservative(){
//    _animesh->set_smoothing_type(EAnimesh::CONSERVATIVE);
//}
//void Animated_mesh_ctrl::smooth_laplacian(){
//    _animesh->set_smoothing_type(EAnimesh::LAPLACIAN);
//}
//void Animated_mesh_ctrl::smooth_tangential(){
//    _animesh->set_smoothing_type(EAnimesh::TANGENTIAL);
//}
//void Animated_mesh_ctrl::smooth_humphrey(){
//    _animesh->set_smoothing_type(EAnimesh::HUMPHREY);
//}
void Animated_mesh_ctrl::set_local_smoothing(bool state){
    _animesh->set_local_smoothing(state);
}
void Animated_mesh_ctrl::set_smooth_force_a (float alpha){
    _animesh->set_smooth_force_a(alpha);
}
//void Animated_mesh_ctrl::set_smooth_force_b (float beta){
//    _animesh->set_smooth_force_b(beta);
//}

//void Animated_mesh_ctrl::set_smooth_smear(float val ){
//    _animesh->set_smooth_smear(val);
//}

void Animated_mesh_ctrl::set_smoothing_weights_diffusion_iter(int nb_iter){
    _animesh->set_smoothing_weights_diffusion_iter(nb_iter);
}

// -----------------------------------------------------------------------------

const Mesh* Animated_mesh_ctrl::get_mesh() const { return g_mesh; }

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::invert_propagation()
{
    for(unsigned i = 0; i < _selected_points.size(); i++)
        _animesh->set_flip_propagation(_selected_points[i], true);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::reset_invert_propagation()
{
    _animesh->reset_flip_propagation();
}

// -----------------------------------------------------------------------------

int Animated_mesh_ctrl::get_nearest_bone(int vert_idx){
    return _animesh->get_nearest_bone(vert_idx);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::copy_bone_samples_to_list()
{
    //_sample_list.clear();

    //int nb_joints = _skel->nb_joints();
    //int acc       = 0;
    //_sample_list.resize( nb_joints );
    //for(int i = 0; i < nb_joints; i++)
    //{
    //    if(_skel->bone_type(i) == EBone::HRBF)
    //    {
    //        const Bone_hrbf* b = (const Bone_hrbf*)_skel->get_bone(i);
    //        const HermiteRBF h = b->get_hrbf();

    //        h.get_samples( _sample_list[i].nodes   );
    //        h.get_normals( _sample_list[i].n_nodes );

    //        int nb_samples = _sample_list[i].nodes.size();
    //        resize_samples_anim(i, nb_samples);

    //        acc += nb_samples;
    //        if(_auto_precompute)
    //            set_bone_type( i, EBone::PRECOMPUTED );
    //    }
    //}
    //update_gl_buffers_size(acc);
    //transform_samples();
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::update_bone_samples(int bone_id)
{
    std::vector<Vec3> nodes, n_nodes;

    if( _factor_bones && bone_id != _skel->root() )
    {
        int parent = _skel->parent( bone_id );
        const std::vector<int>& sons = _skel->get_sons( parent );
        assert(sons.size() > 0);
        int son0 = sons[0];

        //if(son0 != bone_id) // In factor bone mode samples are always in the first son
            //return;
        // <- yea but caps are not stored in the first son ...

        int nb_nodes = _sample_list[son0].nodes.size();
        nodes.  resize( nb_nodes );
        n_nodes.resize( nb_nodes );
        resize_samples_anim(son0, nb_nodes);
        for(int i = 0; i < nb_nodes; i++) {
            nodes  [i] = _sample_list[son0].nodes  [i];
            n_nodes[i] = _sample_list[son0].n_nodes[i];
        }

        for( unsigned i = 0; i < sons.size(); i++) {
            int bid = sons[i];
            if(_bone_caps[bid].jcap.enable && !_skel->is_leaf(bid))
            {
                for(unsigned j = 0; j < _bone_caps[bid].jcap.nodes.size(); j++){
                    nodes.  push_back( _bone_caps[bid].jcap.nodes  [j] );
                    n_nodes.push_back( _bone_caps[bid].jcap.n_nodes[j] );
                }
            }
        }

        if(_bone_caps[son0].pcap.enable) {
            for(unsigned i = 0; i < _bone_caps[son0].pcap.nodes.size(); i++){
                nodes.  push_back( _bone_caps[son0].pcap.nodes  [i] );
                n_nodes.push_back( _bone_caps[son0].pcap.n_nodes[i] );
            }
        }

        _animesh->update_bone_samples(son0, nodes, n_nodes);
    }
    else
    {
        int nb_nodes = _sample_list[bone_id].nodes.size();
        nodes.  resize( nb_nodes );
        n_nodes.resize( nb_nodes );
        resize_samples_anim(bone_id, nb_nodes);

        // Concat nodes and normals
        for(int i = 0; i < nb_nodes; i++)
        {
            nodes  [i] = _sample_list[bone_id].nodes  [i];
            n_nodes[i] = _sample_list[bone_id].n_nodes[i];
        }
        // concat joint caps
        if(_bone_caps[bone_id].jcap.enable && !_skel->is_leaf(bone_id))
        {
            for(unsigned j = 0; j < _bone_caps[bone_id].jcap.nodes.size(); j++) {
                nodes.  push_back( _bone_caps[bone_id].jcap.nodes  [j] );
                n_nodes.push_back( _bone_caps[bone_id].jcap.n_nodes[j] );
            }
        }
        // concat parent caps
        if(_bone_caps[bone_id].pcap.enable)
        {
            for(unsigned i = 0; i < _bone_caps[bone_id].pcap.nodes.size(); i++){
                nodes.  push_back( _bone_caps[bone_id].pcap.nodes  [i] );
                n_nodes.push_back( _bone_caps[bone_id].pcap.n_nodes[i] );
            }
        }
        _animesh->update_bone_samples(bone_id, nodes, n_nodes);
    }

    if(_auto_precompute) {
        int type = _skel->bone_type( bone_id );
        if(EBone::PRECOMPUTED != type && EBone::SSD != type)
            _animesh->set_bone_type( bone_id, EBone::PRECOMPUTED);
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::update_caps(int bone_id, bool jcap, bool pcap)
{
    std::vector<Vec3>& jnodes   = _bone_caps[bone_id].jcap.nodes;
    std::vector<Vec3>& jn_nodes = _bone_caps[bone_id].jcap.n_nodes;

    std::vector<Vec3>& pnodes   = _bone_caps[bone_id].pcap.nodes;
    std::vector<Vec3>& pn_nodes = _bone_caps[bone_id].pcap.n_nodes;

    if(jcap && !_skel->is_leaf(bone_id))
    {
        jnodes.  clear();
        jn_nodes.clear();
        _animesh->compute_jcaps(bone_id, jnodes, jn_nodes);
    }

    if(pcap)
    {
        int parent = _skel->parent( bone_id );
        if(_factor_bones && parent != -1)
        {
            const std::vector<int>& sons = _skel->get_sons( parent );
            assert(sons.size() > 0);
            _bone_caps[ sons[0] ].pcap.nodes.  clear();
            _bone_caps[ sons[0] ].pcap.n_nodes.clear();
            _animesh->compute_pcaps(sons[0],
                                    (sons.size() > 1),
                                    _bone_caps[ sons[0] ].pcap.nodes,
                                    _bone_caps[ sons[0] ].pcap.n_nodes);
        }
        else
        {
            pnodes.  clear();
            pn_nodes.clear();
            _animesh->compute_pcaps(bone_id, false, pnodes, pn_nodes);
        }
    }

    _bone_anim_caps[bone_id].jcap.enable = _bone_caps[bone_id].jcap.enable;
    _bone_anim_caps[bone_id].jcap.nodes.  resize( jnodes.  size() );
    _bone_anim_caps[bone_id].jcap.n_nodes.resize( jn_nodes.size() );

    _bone_anim_caps[bone_id].pcap.enable = _bone_caps[bone_id].pcap.enable;
    _bone_anim_caps[bone_id].pcap.nodes.  resize( pnodes.  size() );
    _bone_anim_caps[bone_id].pcap.n_nodes.resize( pn_nodes.size() );
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::choose_hrbf_samples_ad_hoc(int bone_id,
                                                    float jmax,
                                                    float pmax,
                                                    float minDist,
                                                    float fold)
{
    Animesh::Adhoc_sampling heur(_animesh);
    heur._bone_id = bone_id;
    heur._jmax = jmax;
    heur._pmax = pmax;
    heur._mind = minDist;
    heur._fold = fold;
    heur._factor_siblings = _factor_bones;

    update_caps(bone_id, true, true); // FIXME: dunno why it's here

    _sample_list[bone_id].nodes.  clear();
    _sample_list[bone_id].n_nodes.clear();

    heur.sample(_sample_list[bone_id].nodes,
                _sample_list[bone_id].n_nodes);

    update_bone_samples(bone_id);
    update_gl_buffers_size( compute_nb_samples() );
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::choose_hrbf_samples_poisson(int bone_id,
                                                     float jmax,
                                                     float pmax,
                                                     float minDist,
                                                     int nb_samples,
                                                     float fold)
{
    Animesh::Poisson_disk_sampling heur(_animesh);
    heur._bone_id = bone_id;
    heur._jmax = jmax;
    heur._pmax = pmax;
    heur._min_d = minDist;
    heur._nb_samples = nb_samples;
    heur._fold = fold;
    heur._factor_siblings = _factor_bones;

    update_caps(bone_id, true, true); // FIXME: dunno why it's here

    _sample_list[bone_id].nodes.  clear();
    _sample_list[bone_id].n_nodes.clear();

    heur.sample(_sample_list[bone_id].nodes,
                _sample_list[bone_id].n_nodes);

    update_bone_samples(bone_id);
    update_gl_buffers_size( compute_nb_samples() );
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::choose_hrbf_samples_gael(int /*bone_id*/)
{
    std::cerr << "WARNING: Function not implemented";
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::set_hrbf_radius(int bone_id, float rad)
{
    if(_skel->bone_type(bone_id) == EBone::HRBF)
        _skel->set_bone_hrbf_radius( bone_id, rad);
    else if( _skel->bone_type(bone_id) == EBone::PRECOMPUTED )
    {
        bool tmp = _auto_precompute;
        // disable _auto_precompute. otherwise set_bone_type() will convert
        // back to precomputed the bone without letting us the chance to change
        // the radius first
        _auto_precompute = false;
        this->set_bone_type(bone_id, EBone::HRBF);
        _skel->set_bone_hrbf_radius( bone_id, rad);
        this->set_bone_type(bone_id, EBone::PRECOMPUTED);
        _auto_precompute = tmp;
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::delete_sample(int bone_id, int idx)
{
    std::vector<Vec3>::iterator it = _sample_list[bone_id].nodes.begin();
    _sample_list[bone_id].nodes.erase( it+idx );
    it = _sample_list[bone_id].n_nodes.begin();
    _sample_list[bone_id].n_nodes.erase( it+idx );
    update_bone_samples(bone_id);
    update_gl_buffers_size( compute_nb_samples() );
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::empty_samples(int bone_id)
{
    reset_samples_selection();
    _sample_list[bone_id].nodes.  clear();
    _sample_list[bone_id].n_nodes.clear();

    update_bone_samples(bone_id);
    update_gl_buffers_size( compute_nb_samples() );
}

// -----------------------------------------------------------------------------

int Animated_mesh_ctrl::add_sample(int bone_id,
                                   const Vec3& p,
                                   const Vec3& n)
{
    _sample_list[bone_id].nodes.  push_back(p);
    _sample_list[bone_id].n_nodes.push_back(n);

    update_bone_samples(bone_id);
    update_gl_buffers_size( compute_nb_samples() );

    return _sample_list[bone_id].nodes.size()-1;
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::add_samples(int bone_id,
                                     const std::vector<Vec3>& p,
                                     const std::vector<Vec3>& n)
{
    assert(p.size() == n.size());

    _sample_list[bone_id].nodes.  reserve( _sample_list[bone_id].  nodes.size() + p.size() );
    _sample_list[bone_id].n_nodes.reserve( _sample_list[bone_id].n_nodes.size() + n.size() );

    for(unsigned i = 0; i < p.size(); i++)
    {
        _sample_list[bone_id].nodes.  push_back(p[i]);
        _sample_list[bone_id].n_nodes.push_back(n[i]);
    }

    update_bone_samples(bone_id);
    update_gl_buffers_size( compute_nb_samples() );
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::incr_junction_rad(int bone_id, float incr)
{
    float r = _animesh->get_junction_radius(bone_id);
    _animesh->set_junction_radius(bone_id, r+incr);

    // Recompute caps position
    update_caps(bone_id,
                false,
                _bone_caps[bone_id].pcap.enable);

    update_bone_samples(bone_id);

    int pt = _skel->parent( bone_id );
    if( pt > -1)
    {
        update_caps(pt,
                    _bone_caps[pt].jcap.enable,
                    false);

        update_bone_samples(pt);

        /*
        const std::vector<int>& c = _skel->get_sons(pt);
        for(unsigned i = 0; i < c.size(); i++)
        {
            const int cbone = c[i];
            // Recompute caps position
            update_caps(cbone,
                        _bone_caps[cbone].jcap.enable,
                        false);

            update_bone_samples(cbone);
        }
        */
    }

}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::set_jcap(int bone_id, bool state)
{
    _bone_caps     [bone_id].jcap.enable = state;
    _bone_anim_caps[bone_id].jcap.enable = state;

    update_caps(bone_id,
                state,
                _bone_caps[bone_id].pcap.enable);

    update_bone_samples(bone_id);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::set_pcap(int bone_id, bool state)
{
    _bone_caps     [bone_id].pcap.enable = state;
    _bone_anim_caps[bone_id].pcap.enable = state;

    update_caps(bone_id,
                _bone_caps[bone_id].jcap.enable,
                state);

    update_bone_samples(bone_id);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::transform_samples(const std::vector<int>& bone_ids)
{
    Vec3* vbo_ptr = 0;
    Vec3* nbo_ptr = 0;

    if(_anim_bo->size() != 0){
        _anim_bo->   map_to(vbo_ptr, GL_WRITE_ONLY);
        _normals_bo->map_to(nbo_ptr, GL_WRITE_ONLY);
    }

    int acc = 0;
    int nb_bones = bone_ids.size() == 0 ? _sample_list.size() : bone_ids.size();
    for(int i = 0; i < nb_bones; i++)
    {
        // Transform samples
        const int bone_id = bone_ids.size() == 0 ? i : bone_ids[i];
        const Transfo& tr = _skel->get_transfo( bone_id );

//        assert(_sample_anim_list[bone_id].nodes.  size() == _sample_list[bone_id].nodes.  size() );
//        assert(_sample_anim_list[bone_id].n_nodes.size() == _sample_list[bone_id].n_nodes.size() );
        if(_sample_anim_list[bone_id].nodes.size() != _sample_list[bone_id].nodes.size())
            resize_samples_anim(bone_id, _sample_list[bone_id].nodes.size());

        for(unsigned j = 0; j < _sample_list[bone_id].nodes.size(); j++)
        {
            Point3 p ( _sample_list[bone_id].nodes[j] );
            Vec3  n = _sample_list[bone_id].n_nodes[j];

            Vec3 pos = Vec3(tr * p);
            Vec3 nor = tr * n;
            _sample_anim_list[bone_id].nodes  [j] = pos;
            _sample_anim_list[bone_id].n_nodes[j] = nor;

            // update vbo
            vbo_ptr[acc] = pos;

            nbo_ptr[acc*2 + 0] = pos;
            nbo_ptr[acc*2 + 1] = pos + nor;
            acc++;
        }

        transform_caps(bone_id, tr);
    }

    if(_anim_bo->size() != 0){
        _anim_bo->   unmap();
        _normals_bo->unmap();
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::transform_caps(int bone_id, const Transfo& tr)
{
    // Transform jcaps
    if(_bone_caps[bone_id].jcap.enable)
        for(unsigned j = 0; j < _bone_caps[bone_id].jcap.nodes.size(); j++)
        {
            Point3 p ( _bone_caps[bone_id].jcap.nodes[j] );
            Vec3  n = _bone_caps[bone_id].jcap.n_nodes[j];
            _bone_anim_caps[bone_id].jcap.nodes  [j] = Vec3( tr * p );
            _bone_anim_caps[bone_id].jcap.n_nodes[j] = tr * n;
        }

    // Transform pcaps
    if(_bone_caps[bone_id].pcap.enable)
        for(unsigned j = 0; j < _bone_caps[bone_id].pcap.nodes.size(); j++)
        {
            Point3 p ( _bone_caps[bone_id].pcap.nodes[j] );
            Vec3  n = _bone_caps[bone_id].pcap.n_nodes[j];
            _bone_anim_caps[bone_id].pcap.nodes  [j] = Vec3( tr * p );
            _bone_anim_caps[bone_id].pcap.n_nodes[j] = tr * n;
        }
}

// -----------------------------------------------------------------------------

static Vec4 colorToVec4(const Color& cl){
    return Vec4(cl.r, cl.g, cl.b, cl.a);
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::set_sample_color(Samp_id id, const Color& c)
{
    Vec4* ptr_cbo = 0;
    _color_bo->map_to(ptr_cbo, GL_WRITE_ONLY);

    int idx = compute_offset(id.bone_id) + id.samp_id;
    ptr_cbo[idx] = colorToVec4( c );

    _color_bo->unmap();
}

// -----------------------------------------------------------------------------

bool Animated_mesh_ctrl::select_samples(const Camera& c,
                                        int x, int y,
                                        Select_type<Samp_id>* heuristic,
                                        bool rest_pose)
{
    y = c.height() - y;

    samples_selection(c, x, y, heuristic, rest_pose);

    int nb_selected = heuristic->nb_selected();
    Samp_id* selection_set = heuristic->get_selected_points();

    for(int i=0; i<nb_selected; i++)
        add_sample_to_selection( selection_set[i] );

    return nb_selected > 0;
}

// -----------------------------------------------------------------------------

bool Animated_mesh_ctrl::unselect_samples(const Tbx::Camera& c,
                                          int x, int y,
                                          Select_type<Samp_id>* heuristic,
                                          bool rest_pose)
{
    y = c.height() - y;

    samples_selection(c, x, y, heuristic, rest_pose);

    int nb_selected = heuristic->nb_selected();
    Samp_id* selection_set = heuristic->get_selected_points();
    for(int i = 0; i < nb_selected; i++)
        remove_sample_from_selection( selection_set[i] );

    return nb_selected > 0;
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::reset_samples_selection()
{
    for(unsigned int i=0; i<_selected_samples.size(); i++)
    {
        Samp_id id = _selected_samples[i];
        Color cl = Color::pseudo_rand(id.bone_id);
        set_sample_color(id, cl);
    }

    _selected_samples.clear();
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::delete_selected_samples()
{
    std::vector<bool> update_bone(_sample_list.size(), false);

    std::vector<HSample_list> tmp_sample_list( _sample_list.size() );

    for(unsigned i = 0; i < _sample_list.size(); i++)
    {
        tmp_sample_list[i].nodes.  reserve( _sample_list[i].nodes.size() );
        tmp_sample_list[i].n_nodes.reserve( _sample_list[i].nodes.size() );
        for(unsigned j = 0; j < _sample_list[i].nodes.size(); j++)
        {
            bool is_selected = false;

            for(unsigned s = 0; s < _selected_samples.size(); s++)
            {
                Samp_id id = _selected_samples[s];
                if( id.bone_id == (int)i && id.samp_id == (int)j )
                {
                    is_selected = true;
                    break;
                }
            }

            if(!is_selected)
            {
                tmp_sample_list[i].nodes.  push_back( _sample_list[i].  nodes[j] );
                tmp_sample_list[i].n_nodes.push_back( _sample_list[i].n_nodes[j] );
            }else
                update_bone[i] = true;
        }
    }

    _sample_list.swap(tmp_sample_list);

    for(unsigned i = 0; i < update_bone.size(); i++)
        if( update_bone[i] )
            update_bone_samples(i);

    update_gl_buffers_size( compute_nb_samples() );
    _selected_samples.clear();
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::add_sample_to_selection(Samp_id id)
{
    // Check for doubles
    bool state = false;
    for(unsigned int i=0; i<_selected_samples.size(); i++){
        state = state || (_selected_samples[i] == id);
    }

    if(!state)
    {
        Color cl = Cuda_ctrl::_color.get(Color_ctrl::HRBF_SELECTED_POINTS);
        set_sample_color(id, cl);
        _selected_samples.push_back(id);
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::remove_sample_from_selection(Samp_id id)
{
    std::vector<Samp_id>::iterator it = _selected_samples.begin();
    unsigned int i = 0;
    for(; it < _selected_samples.end(); ++it, ++i)
        if( (*it) == id )
            break;

    if(i < _selected_samples.size())
    {
        Color cl = Cuda_ctrl::_color.get(Color_ctrl::HRBF_POINTS);
        set_sample_color(id, cl);
        _selected_samples.erase(it);
    }
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::resize_samples_anim(int bone_id, int size)
{
    _sample_anim_list[bone_id].nodes.  resize( size );
    _sample_anim_list[bone_id].n_nodes.resize( size );
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::samples_selection(const Camera& ,
                                           int x, int y,
                                           Select_type<Samp_id>* selection_set,
                                           bool rest_pose)
{
    GLint viewport[4];
    GLdouble modelview[16];
    GLdouble projection[16];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    GLdouble vx, vy, vz;
    float depth;
    // Read depth buffer at mouse position
    glReadPixels( x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);

    const std::vector<int>& set = Cuda_ctrl::_skeleton.get_selection_set();
    for(unsigned i = 0; i < set.size(); i++)
    {
        int bone = set[i];
        unsigned size = rest_pose ? _sample_list[bone].nodes.size() :
                                    _sample_anim_list[bone].nodes.size();

        for(unsigned j = 0; j < size; j++)
        {
            Vec3 p = rest_pose ? _sample_list[bone].nodes[j] :
                                      _sample_anim_list[bone].nodes[j];

            gluProject(p.x, p.y, p.z,
                       modelview, projection, viewport,
                       &vx, &vy, &vz);

            Samp_id id = {bone, (int)j};
            selection_set->test(id,
                                Vec3((float)vx, (float)vy, (float)vz),
                                Vec3((float)x , (float)y , depth    ) );
        }
    }
}

// -----------------------------------------------------------------------------

Vec3 Animated_mesh_ctrl::get_sample_pos(Samp_id id)
{
    return _sample_list[id.bone_id].nodes[id.samp_id];
}

// -----------------------------------------------------------------------------

Vec3 Animated_mesh_ctrl::get_sample_normal(Samp_id id)
{
    return _sample_list[id.bone_id].n_nodes[id.samp_id];
}

// -----------------------------------------------------------------------------

Vec3 Animated_mesh_ctrl::get_sample_anim_normal(Samp_id id)
{
    const Skeleton* s = _skel;
    const Transfo& tr = s->get_transfo( id.bone_id );
    Point3 node(_sample_list[id.bone_id].n_nodes[id.samp_id]);
    return ( tr * node).to_vec3();
}

// -----------------------------------------------------------------------------

Vec3 Animated_mesh_ctrl::get_sample_anim_pos(Samp_id id)
{
    const Skeleton* s = _skel;
    const Transfo& tr = s->get_transfo( id.bone_id );
    Point3 node(_sample_list[id.bone_id].nodes[id.samp_id]);
    return ( tr * node).to_vec3();
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::transform_selected_samples(const Transfo& t)
{
    std::vector<bool> update_bone(_sample_list.size(), false);

    for(unsigned i = 0; i < _selected_samples.size(); i++)
    {
        Samp_id id = _selected_samples[i];
        Vec3& pos = _sample_list[id.bone_id].nodes[id.samp_id];
        Vec3& nor = _sample_list[id.bone_id].n_nodes[id.samp_id];

        pos = (t * pos.to_point3()).to_vec3();
        nor = (t * nor);

        update_bone[id.bone_id] = true;
    }

    for(unsigned i = 0; i < update_bone.size(); i++)
        if( update_bone[i] ) update_bone_samples(i);

    update_vbo_rest_pose( compute_nb_samples() );
}

// -----------------------------------------------------------------------------

Vec3 Animated_mesh_ctrl::cog_sample_selection()
{
    Vec3 cog = Vec3::zero();

    int size = _selected_samples.size();
    for(int i = 0; i < size; i++)
    {
        Samp_id id = _selected_samples[i];
        const Vec3& v = _sample_list[id.bone_id].nodes[id.samp_id];

        cog = v + cog;
    }

    cog = cog / (float)(size == 0 ? 1 : size);
    return cog;
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::update_vbo_rest_pose(int size)
{
    std::vector<Vec3> buff_nodes  (size    );
    std::vector<Vec3> buff_normals(size * 2);
    int acc = 0;
    for(unsigned i = 0; i < _sample_list.size(); i++)
    {
        for(unsigned j = 0; j < _sample_list[i].nodes.size(); j++)
        {
            Vec3 p = _sample_list[i].nodes  [j];
            Vec3 n = _sample_list[i].n_nodes[j];

            buff_nodes[acc] = p;

            buff_normals[acc*2 + 0] = p;
            buff_normals[acc*2 + 1] = p+n;
            acc++;
        }
    }

    _rest_bo->set_data(size, &(buff_nodes[0]), GL_STATIC_DRAW);
    // two points for a line :
    _rest_nbo->set_data(size * 2, &(buff_normals[0]), GL_STATIC_DRAW);
    _normals_bo->set_data(size * 2, 0, GL_STATIC_DRAW);

}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::update_gl_buffers_size(int size)
{
    if( size == 0 ) return;

    // update vbo
    update_vbo_rest_pose( size );
    _anim_bo->set_data(size, 0, GL_STATIC_DRAW);

    // update cbo
    std::vector<Color> color(size);
    int idx = 0;
    for(unsigned i = 0; i < _sample_list.size(); i++)
    {
        Color cl = Color::pseudo_rand(i);
        for(unsigned j = 0; j < _sample_list[i].nodes.size(); j++)
        {
            assert(idx < size);
            color[idx] = cl;
            idx++;
        }
    }
    _color_bo->set_data(size, (Vec4*)&(color[0]));
}

// -----------------------------------------------------------------------------

int Animated_mesh_ctrl::compute_nb_samples()
{
    int acc = 0;
    for(unsigned i = 0; i < _sample_list.size(); i++)
        acc += _sample_list[i].nodes.size();

    return acc;
}

// -----------------------------------------------------------------------------

int Animated_mesh_ctrl::compute_offset(int bone_id)
{
    assert( bone_id >= 0                  );
    assert( bone_id < (int)_sample_list.size() );
    int acc = 0;
    for(int i = 0; i < bone_id; i++)
        acc += _sample_list[i].nodes.size();

    return acc;
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::precompute_all_bones()
{
    for(int i = 0; i < _skel->nb_joints(); i++)
    {
        int type = _skel->bone_type( i );
        if(EBone::PRECOMPUTED != type && EBone::SSD != type)
            _animesh->set_bone_type( i, EBone::PRECOMPUTED);
    }
}

// -----------------------------------------------------------------------------


void Animated_mesh_ctrl::draw_caps(const Cap& cap)
{
    if(!cap.enable) return;

    GLPointSizeSave save_point_size;

    glColor3f(1.f, 0.f, 0.f);
    glPointSize(15.f);

    glBegin(GL_POINTS);
    for( unsigned i = 0; i < cap.nodes.size(); i++)
    {
        Vec3 v = cap.nodes[i];
        glVertex3f(v.x, v.y, v.z);
    }
    glEnd();

    GLLineWidthSave save_width;
    glLineWidth( 1.5f );
    glBegin(GL_LINES);
    for( unsigned i = 0; i < cap.nodes.size(); i++)
    {
        Vec3 v = cap.nodes  [i];
        Vec3 n = cap.n_nodes[i];
        glVertex3f(v.x, v.y, v.z);
        v = v + n;
        glVertex3f(v.x, v.y, v.z);
    }
    glEnd();
}

// -----------------------------------------------------------------------------

void Animated_mesh_ctrl::draw_hrbf_points(const std::vector<int>& list,
                                          bool draw_normals,
                                          bool rest_pose)
{

    GLEnabledSave save_point(GL_POINT_SMOOTH, true, true );
    GLEnabledSave save_light(GL_LIGHTING,     true, false);
    GLEnabledSave save_tex  (GL_TEXTURE_2D,   true, false);

    glAssert( glPointSize(10.f) );
    glAssert( glPushMatrix() );
    glAssert( glEnableClientState(GL_VERTEX_ARRAY) );

    _color_bo->bind();
    glAssert( glColorPointer(4,GL_FLOAT,0,0) );

    std::vector<int> done(_skel->nb_joints(), false);
    for(unsigned i = 0; i < list.size(); i++)
    {
        int bone_id = list[i];

        int parent = _skel->parent( bone_id );
        if( _factor_bones && parent != -1)
        {
            const std::vector<int>& sons = _skel->get_sons( parent );
            assert(sons.size() > 0);
            if(done[ sons[0] ] )
                continue;
            else
            {
                bone_id = sons[0];
                done[bone_id] = true;
            }
        }

        int dep  = compute_offset(bone_id);
        int size = _sample_list[bone_id].nodes.size();

        // Draw points
        glAssert( glEnableClientState(GL_COLOR_ARRAY) );
        rest_pose ? _rest_bo->bind() : _anim_bo->bind() ;
        glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );
        glAssert( glDrawArrays(GL_POINTS, dep, size) );

        // draw normals
        if(draw_normals)
        {
            glAssert( glDisableClientState(GL_COLOR_ARRAY) );
            rest_pose ? _rest_nbo->bind() : _normals_bo->bind();
            glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );
            Color cl = Color::pseudo_rand(bone_id);
            glColor3f(cl.r, cl.g, cl.b);
            glAssert( glDrawArrays(GL_LINES, dep*2, size*2) );
        }

        // draw caps
        if(rest_pose){
            draw_caps(_bone_caps[ list[i] ].jcap);
            draw_caps(_bone_caps[ bone_id ].pcap);
        }
        else
        {
            draw_caps(_bone_anim_caps[ list[i] ].jcap);
            draw_caps(_bone_anim_caps[ bone_id ].pcap);
        }
    }



#if 0
    if(draw_normals)
    {
        glAssert( glDisableClientState(GL_COLOR_ARRAY) );
        rest_pose ? _rest_nbo->bind() : _normals_bo->bind();
        glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );

        for(unsigned i = 0; i < list.size(); i++)
        {
            int bone_id = list[i];

            int dep  = compute_offset(bone_id);
            int size = _sample_list[bone_id].nodes.size();

            Color cl = Color::pseudo_rand(bone_id);
            glColor3f(cl.r, cl.g, cl.b);
            glAssert( glDrawArrays(GL_LINES, dep*2, size*2) );
        }
    }
#endif

    // clean up
    _color_bo->unbind();
    glAssert( glVertexPointer(3, GL_FLOAT, 0, 0) );
    glAssert( glColorPointer (4, GL_FLOAT, 0, 0) );

    glAssert( glDisableClientState(GL_VERTEX_ARRAY) );
    glAssert( glDisableClientState(GL_COLOR_ARRAY) );
    glAssert( glPopMatrix() );
}

// -----------------------------------------------------------------------------
