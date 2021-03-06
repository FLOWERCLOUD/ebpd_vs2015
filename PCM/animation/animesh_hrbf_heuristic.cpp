#include "animesh.hpp"

// -----------------------------------------------------------------------------

#include "toolbox/timer.hpp"
#include "toolbox/quick_sort.hpp"
#include "toolbox/mesh_utils/gen_mesh.hpp"
#include "utils_sampling.hpp"
#include "skeleton.hpp"

// -----------------------------------------------------------------------------

#include <iostream>
#include <limits>
#include <cuda.h>

using namespace Cuda_utils;
using namespace HRBF_env;

typedef Animesh::HRBF_sampling HRBF_samp;
typedef Animesh::Adhoc_sampling AdHoc_samp;
typedef Animesh::Poisson_disk_sampling Poisond_samp;

// -----------------------------------------------------------------------------

void HRBF_samp::factor_samples(std::vector<int>& vert_ids,
                               std::vector<Vec3>& vertices,
                               std::vector<Vec3>& normals)
{
    vert_ids.clear();
    vertices.clear();
    normals. clear();

    int parent = _am->_skel->parent( _bone_id );
    std::vector<int> dummy(1, _bone_id);
    const std::vector<int>& sons = parent == -1 ? dummy : _am->_skel->get_sons(parent);
    assert(sons.size() > 0);

    if( sons[0] == _bone_id || _factor_siblings == false)
        for( unsigned i = 0; i < sons.size(); i++)
        {
            const int bone_id = _factor_siblings ? sons[i] : _bone_id;
            std::vector<Vec3>& nors  = _am->h_input_normals_per_bone[bone_id];
            std::vector<Vec3>& verts = _am->h_input_verts_per_bone  [bone_id];
            std::vector<int>&     ids   = _am->h_verts_id_per_bone     [bone_id];
            vert_ids.reserve( verts.size() + vert_ids.size() );
            vertices.reserve( verts.size() + vertices.size() );
            normals .reserve( verts.size() + normals .size() );
            for( unsigned j = 0; j < ids.size(); j++)
            {
                vert_ids.push_back( ids  [j] );
                vertices.push_back( verts[j] );
                normals. push_back( nors [j] );
            }
            if( !_factor_siblings ) break;
        }
}

// -----------------------------------------------------------------------------

void HRBF_samp::clamp_samples(std::vector<int>& vert_ids_,
                              std::vector<Vec3>& verts_,
                              std::vector<Vec3>& normals_)
{
    std::cerr << "clamp disable for testing purpose" << std::endl;
#if 0
    std::vector<int> vert_ids;
    std::vector<Vec3> verts;
    std::vector<Vec3> normals;

    vert_ids.reserve( vert_ids_.size() );
    verts.   reserve( verts_.size()    );
    normals. reserve( normals_.size()  );

    for(unsigned id = 0; id < verts_.size(); id++)
    {
        const int nearest_bone = _am->h_vertices_nearest_bones[ vert_ids_[id] ];
        const Bone* b = _am->_skel->get_bone(nearest_bone);
        const float length = b->length();

        const Point3 vert(verts_[id]);
        const float dist_proj = b->dist_proj_to(vert);

        Vec3 dir_proj = vert - (b->org() + b->dir().normalized() * dist_proj);

        float jlength = length * _jmax;
        float plength = length * _pmax;

        const Vec3 normal = normals_[id];

        const std::vector<int>& sons = _am->_skel->get_sons(nearest_bone);
        bool leaf = sons.size() > 0 ? _am->_skel->is_leaf(sons[0]) : true;

        if( (dist_proj >= -plength ) &&
            (dist_proj < (length + jlength) || leaf) &&
            dir_proj.dot(normal) >= _fold )
        {
            verts.   push_back( verts_   [id] );
            vert_ids.push_back( vert_ids_[id] );
            normals. push_back( normal        );
        }
    }

    vert_ids_.swap( vert_ids );
    verts_.   swap( verts    );
    normals_. swap( normals  );
#endif
}

// -----------------------------------------------------------------------------

void AdHoc_samp::sample(std::vector<Vec3>& out_verts,
                        std::vector<Vec3>& out_normals)
{
    _am->_skel->reset();

    std::vector<Vec3> in_verts;
    std::vector<int>     in_vert_ids;
    std::vector<Vec3> in_normals;
    factor_samples(in_vert_ids, in_verts, in_normals);

    std::vector<bool> done;
    done.resize(in_verts.size(), false);
    for(unsigned id = 0; id < in_verts.size(); id++)
    {

        const Bone* b =_am->_skel->get_bone(_bone_id);
        float length = b->length();

        Point3 vert(in_verts[id]);
        float dist_proj = b->dist_proj_to(vert);

        Vec3 dir_proj = vert - (b->org() + b->dir().normalized() * dist_proj);

        float jlength = length * _jmax;
        float plength = length * _pmax;

        Vec3 normal = in_normals[id];
        if(dist_proj >= -jlength && dist_proj < (length + plength) &&
                dir_proj.dot(normal) >= _fold )
        {
            // Check for to close samples
            float dist = std::numeric_limits<float>::infinity();
            for(unsigned j = 0; j < in_verts.size(); j++)
            {
                float norm = (vert.to_vec3() - in_verts[j]).norm();
                if( (unsigned)id != j && !done[j] && norm < dist)
                    dist = norm;
            }

            if(dist > _mind)
            {
                out_verts.  push_back( in_verts[id] );
                out_normals.push_back( normal       );
            }
        }
        done[id] = true;
    }
    _am->_skel->unreset();
}

// -----------------------------------------------------------------------------

void Poisond_samp::sample(std::vector<Vec3>& out_verts,
                          std::vector<Vec3>& out_normals)
{
    _am->_skel->reset();

    // The goal here is to build from the cluster of vertices bound to a single
    // bone of id '_bone_id' its associated sub mesh, and then sample the
    // surface of this sub mesh with the poisson disk strategy
    std::vector<Vec3> in_verts;
    std::vector<int>     in_vert_ids;
    std::vector<Vec3> in_normals;
    factor_samples(in_vert_ids, in_verts, in_normals);
    clamp_samples (in_vert_ids, in_verts, in_normals);

    if( in_verts.size() == 0) return;

    assert( in_vert_ids.size() == in_verts.size());
    const int size_sub_mesh = in_vert_ids.size();

    std::map<int, int> meshToCluster; // map[mesh_idx] = idx_in_verts_ids
    for(int i = 0; i < size_sub_mesh; i++)
        meshToCluster[ in_vert_ids[i] ] = i;

    // The piece of mesh defined by the bone cluster
    std::vector<int> sub_tris;
    sub_tris.reserve( size_sub_mesh * 3 * 3);

    // Building 'sub_verts' and sub_tris arrays
    std::vector<bool> done(size_sub_mesh, false);
    // Look up vertex cluster
    for(int i = 0; i < size_sub_mesh; i++)
    {
        const int idx = in_vert_ids[i];
        // Look up neighboors
        int nb_neigh = _am->_mesh->get_1st_ring_offset(idx*2 + 1);
        int dep      = _am->_mesh->get_1st_ring_offset(idx*2    );
        int end      = dep + nb_neigh;
        for(int n = dep; n < end; n++)
        {
            int neigh0 = _am->_mesh->get_1st_ring( n );
            int neigh1 = _am->_mesh->get_1st_ring((n+1) >= end ? dep : n+1);

            std::map<int, int>::iterator it0 = meshToCluster.find( neigh0 );
            std::map<int, int>::iterator it1 = meshToCluster.find( neigh1 );

            // Must be in the map (otherwise doesn't belong to the cluster)
            if(it0 != meshToCluster.end() && it1 != meshToCluster.end() )
            {
                // Must not be already treated
                if( !done[it0->second] && !done[it1->second] )
                {
                    // Add the triangles
                    sub_tris.push_back( it0->second );
                    sub_tris.push_back( it1->second );
                    sub_tris.push_back( i           );
                }
            }
        }
        // Tag vertex as treated
        done[i] = true;
    }

    // Compute the poisson disk distribution on the sub mesh
    if(sub_tris.size() > 0)
        Utils_sampling::poisson_disk(_min_d, _nb_samples, in_verts, in_normals, sub_tris, out_verts, out_normals);
    std::cout << "Poisson disk sampling done. " << out_verts.size();
    std::cout << "samples created" << std::endl;

    _am->_skel->unreset();
}

// -----------------------------------------------------------------------------

#if 0
/// @param rad circles's radius
/// @param p circle center
/// @param n planes normal the circle lies in
/// @param out_verts vector where circle vertices will be pushed
/// @param out_verts vector where circle normals will be pushed
static void add_circle(float rad,
                       const Vec3& p,
                       const Vec3& n,
                       std::vector<Vec3>& out_verts,
                       std::vector<Vec3>& out_normals)
{
    // Generates the circle in the xy plane
    Gen_mesh::Line_data* circle = Gen_mesh::circle( rad, 10);
    // Compute circle
    Vec3 x, y, z;
    z = n;
    z.coordinate_system(x, y);
    Transform tr(Mat3_cu(x, y, z), p);


    for(int i = 0; i < circle->nb_vert; i++)
    {
        Point3  p = {circle->vertex [i*3], circle->vertex [i*3+1], circle->vertex [i*3+2]};
        Vec3 n = {circle->normals[i*3], circle->normals[i*3+1], circle->normals[i*3+2]};

        out_verts.  push_back( tr * p );
        out_normals.push_back( tr * n );
    }

    delete circle;
}
#endif

// -----------------------------------------------------------------------------

void Animesh::compute_jcaps(int bone_id,
                                 std::vector<Vec3>& out_verts,
                                 std::vector<Vec3>& out_normals)
{
    _skel->reset();

    const Bone* b = _skel->get_bone(bone_id);

    float jrad = 0.f;// joint radius mean radius
    int nb_sons = _skel->get_sons(bone_id).size();
    for (int i = 0; i < nb_sons; ++i)
        jrad += h_junction_radius[ _skel->get_sons(bone_id)[i] ];

    jrad /= nb_sons > 0 ? (float)nb_sons : 1.f;

    Point3 p = b->end() + b->dir().normalized() * jrad;
    Vec3  n = b->dir().normalized();
    out_verts.  push_back( p );
    out_normals.push_back( n );

    //add_circle(jrad, b->get_org(), n, out_verts, out_normals);

    _skel->unreset();
}

// -----------------------------------------------------------------------------

void Animesh::compute_pcaps(int bone_id,
                                 bool use_parent_dir,
                                 std::vector<Vec3>& out_verts,
                                 std::vector<Vec3>& out_normals)
{
    _skel->reset();

    const Bone* b = _skel->get_bone(bone_id);
    int parent = _skel->parent(bone_id);
    float prad = h_junction_radius[bone_id]; // parent joint radius
    Vec3 p;
    Vec3 n;

    if( use_parent_dir)
    {
        const Bone* pb = _skel->get_bone( parent );
        p =  pb->end() - pb->dir().normalized() * prad;
        n = -pb->dir().normalized();
    }
    else
    {
        p =  b->org() - b->dir().normalized() * prad;
        n = -b->dir().normalized();
    }
    out_verts.  push_back(p);
    out_normals.push_back(n);
    //add_circle(prad, b->get_org() + b->get_dir(), n, out_verts, out_normals);

    _skel->unreset();
}

// -----------------------------------------------------------------------------
