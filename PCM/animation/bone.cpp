#include "bone.hpp"
//#include "precomputed_prim_constants.hpp"

using namespace Tbx;
// Bone CLASS ==================================================================

Obbox Bone::get_obbox() const { return Obbox(); }

// -----------------------------------------------------------------------------

Bbox3 Bone::get_bbox() const { return Bbox3(); }


// END Bone CLASS ==============================================================

// Bone_cylinder CLASS =========================================================

//Obbox Bone_cylinder::get_obbox() const
//{
//    Obbox obbox;
//    obbox._tr = get_frame();
//    Transfo bbox_tr_inv = obbox._tr.fast_invert();
//
//    Point3 pmin = bbox_tr_inv *  _org;
//    Point3 pmax = bbox_tr_inv * (_org + _dir);
//
//
//    float off = 2.f * _radius;
//    obbox._bb.add_point( pmin + Vec3( -off ) );
//    obbox._bb.add_point( pmax + Vec3(  off ) );
//
//    return obbox;
//}
//
//// -----------------------------------------------------------------------------
//
//Bbox3 Bone_cylinder::get_bbox() const
//{
//    return get_obbox().to_bbox();
//}

// END Bone_cylinder CLASS =====================================================

// Bone_hrbf CLASS =============================================================

// If defined enable bbox constructions visualitions with opengl
// (white points are dichotomic steps, colored points are newton iterations)
//#define GL_DEBUG_BBOX

#include "toolbox/portable_includes/port_glew.h" // DEBUG

#if 0
#include "hrbf_env_tex.hpp"
#include "hermiteRBF.hpp"
#include "hermiteRBF.inl"

float dichotomic_search(const Ray_cu& r,
                        float t0, float t1,
                        float iso,
                        const HermiteRBF& hrbf,
                        float eps = 0.00001f)
{
    Vec3 grad;
    float t = t0;
    float f0 = hrbf.fngf_global( grad, r(t0) );
    float f1 = hrbf.fngf_global( grad, r(t1) );

    if(f0 > f1){
        t0 = t1;
        t1 = t;
    }

    Point3 p;
    for(unsigned short i = 0 ; i < 25; ++i)
    {
        t = (t0 + t1) * 0.5f;
        p = r(t);

        #ifdef GL_DEBUG_BBOX
        glColor3f(1.f, 1.f, 1.f);
        glVertex3f(p.x, p.y, p.z);
        #endif

        f0 = hrbf.fngf_global( grad, p );

        if(f0 > iso){
            t1 = t;
            if((f0-iso) < eps) break;
        } else {
            t0 = t;
            if((iso-f0) < eps) break;
        }
    }
    return t;
}

// -----------------------------------------------------------------------------

/// Cast a ray and return the farthest point whose potential is null.
/// We use newton iterations
/// @param start : origin of the ray must be inside the primitive
/// @param dir : direction we do the ray marching if custom direction is enabled
/// otherwise we follow the gradient
/// tr is the transformation to world coordinates
/// (the same as 'points' and 'weights')
/// @param points : samples of the hrbf we want to evaluate
/// @param weights : coefficients of the HRBF we want to evaluate
/// @param custom_dir : if true we don't follow the gradient but use 'dir'
/// defined by the user
/// @return the farthest point which potential is null along the ray.
Point3 push_point(const Point3& start,
                    const Vec3& dir,
                    const HermiteRBF& hrbf,
                    bool custom_dir = false)
{
    const float rad = hrbf.get_radius();
    Vec3  grad;
    Vec3  n_dir = dir.normalized();
    Vec3  step;
    Point3 res  = start;
    Point3 prev = res;

    #ifdef GL_DEBUG_BBOX
    Vec3 cl= (dir + Vec3(0.5f, 0.5f, 0.5f)) * 0.5f;
    glBegin(GL_POINTS);
    #endif

    for(int i = 0; i < 25; i++)
    {
        const float pot      = hrbf.fngf_global( grad, res );
        const float pot_diff = fabsf( pot - rad);
        const float norm     = grad.safe_normalize();
        #ifdef GL_DEBUG_BBOX
        glColor3f(cl.x, cl.y, cl.z);
        glVertex3f(res.x, res.y, res.z);
        #endif

        if( norm < 0.0000001f) break;

        float scale = (pot_diff / norm) * 0.4f;

        step = (custom_dir ? n_dir : grad );

        if( pot > rad)
        {
            Ray_cu r( prev, step);
            float t = dichotomic_search(r, 0.f, (res-prev).norm(), rad, hrbf);
            res = r( t );
            break;
        }

        prev = res;
        res = res + step * scale;

        if( pot_diff <= 0.0001f || scale < 0.00001f)
            break;

    }
    #ifdef GL_DEBUG_BBOX
    glEnd();

    if( (res-start).norm( ) > rad * 2.f)
    {
        glBegin(GL_LINES);
        glVertex3f(start.x, start.y, start.z);
        glVertex3f(res.x, res.y, res.z);
        glEnd();
    }
    #endif

    return res;
}

// -----------------------------------------------------------------------------

/// Cast several rays along a square grid and return the farthest point which
/// potential is zero.
/// @param obbox : oriented bounding box which we add casted points to
/// @param org : origin point of the grid (top left corner)
/// @param x : extrimity point of the grid in x direction (top right corner)
/// @param y : extrimity point of the grid in y direction (bottom left corner)
/// @param points samples of the hrbf we want to evaluate
/// @param weights coefficients of the HRBF we want to evaluate
/// @warning org, x and y parameters must lie in the same plane. Also this
/// (org-x) dot (org-y) == 0 must be true at all time. Otherwise the behavior
/// is undefined.
void push_face(Obbox& obbox,
               const Point3& org,
               const Point3& x,
               const Point3& y,
               const HermiteRBF& hrbf)
{
    int res = 8/*GRID_RES*/;

    Vec3 axis_x = x-org;
    Vec3 axis_y = y-org;
    Vec3 udir = (axis_x.cross(axis_y)).normalized();

    float len_x = axis_x.normalize();
    float len_y = axis_y.normalize();

    float step_x = len_x / (float)res;
    float step_y = len_y / (float)res;

    Transfo tr = obbox._tr.fast_invert();
    for(int i = 0; i < res; i++)
    {
        for(int j = 0; j < res; j++)
        {
            Point3 p = org +
                         axis_x * (step_x * (float)i + step_x * 0.5f) +
                         axis_y * (step_y * (float)j + step_y * 0.5f);

            Point3 res = push_point(p, udir, hrbf, true);

            obbox._bb.add_point( tr * res);
        }
    }
}

// -----------------------------------------------------------------------------

Obbox Bone_hrbf::get_obbox() const
{
    const int hrbf_id = _hrbf.get_id();
    std::vector<Point3> samp_list;
    HRBF_env::get_anim_samples(hrbf_id, samp_list);

    Obbox obbox;
    obbox._tr = get_frame();
    Transfo bbox_tr_inv = obbox._tr.fast_invert();

    Point3 pmin = bbox_tr_inv *  _org;
    Point3 pmax = bbox_tr_inv * (_org + _dir);

    obbox._bb = Bbox3(pmin, pmax);

    const HermiteRBF& hrbf = get_hrbf();

    // Seek zero along samples normals of the HRBF
    for(unsigned i = 0; i < samp_list.size(); i++)
    {
        Point3 pt = push_point(samp_list[i], Vec3(), hrbf);
        obbox._bb.add_point(bbox_tr_inv * pt);
    }



#if 1
    // Push obbox faces
    std::vector<Point3> corners;
    /**
        @code

            6 +----+ 7
             /|   /|
          2 +----+3|
            |4+--|-+5
            |/   |/
            +----+
           0      1
        // Vertex 0 is pmin and vertex 7 pmax
        @endcode
    */
    // Get obox in world coordinates
    obbox._bb.get_corners(corners);
    for(int i = 0; i < 8; ++i)
        corners[i] = obbox._tr * corners[i];


    Point3 list[6][3] = {{corners[2], corners[3], corners[0]}, // FRONT
                           {corners[0], corners[1], corners[4]}, // BOTTOM
                           {corners[3], corners[7], corners[1]}, // RIGHT
                           {corners[6], corners[2], corners[4]}, // LEFT
                           {corners[7], corners[6], corners[5]}, // REAR
                           {corners[6], corners[7], corners[2]}, // TOP
                          };

    // Pushing according a grid from the box's faces
    for(int i = 0; i < 6; ++i) {
        push_face(obbox, list[i][0], list[i][1], list[i][2], hrbf);
    }
#endif

    return obbox;
}

// -----------------------------------------------------------------------------

Bbox3 Bone_hrbf::get_bbox() const
{
    return get_obbox().to_bbox();
}

// END Bone_hrbf CLASS =========================================================

// Bone_precomputed CLASS ======================================================

Obbox Bone_precomputed::get_obbox() const
{
    Obbox tmp = _obbox;

    tmp._tr = Precomputed_env::get_user_transform( _primitive.get_id() ) * tmp._tr;

    return  tmp;
}

// -----------------------------------------------------------------------------

Bbox3 Bone_precomputed::get_bbox() const
{
    return get_obbox().to_bbox();
}

// END Bone_precomputed CLASS ==================================================


#endif

// =============================================================================
namespace EBone {
// =============================================================================

// -----------------------------------------------------------------------------

std::string type_to_string(int type)
{
    std::string res = "";
    switch( type ){
    case EBone::CYLINDER:    res = "implicit cylinder";     break;
    case EBone::HRBF:        res = "hermite rbf";           break;
    case EBone::SSD:         res = "ssd";                   break;
    case EBone::PRECOMPUTED: res = "precomputed primitive"; break;
    default: //unknown bone type !
        assert(false);
        break;
    }

    return res;
}

} // END BONE_TYPE NAMESPACE ===================================================
