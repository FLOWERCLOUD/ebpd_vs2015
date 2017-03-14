#ifndef BONE_HPP__
#define BONE_HPP__

#include <cassert>
//#include "toolbox/cuda_utils/cuda_compiler_interop.hpp"
//#include "toolbox/maths/Vec.hpp"
#include "toolbox/maths/bbox3.hpp"
#include "bone_type.h"
#include "QGLViewer/vec.h"
#include "toolbox/maths/mat3.hpp"
#include "toolbox/maths/vec3.hpp"
using qglviewer::Vec;
using EBone::Bone_t;

// TODO : to be moved in Skeleton_env
struct Skeleton;

/**
  @class Bone_cu
  @brief Mother class of various bones types

  A bone defines a segment between two 3D points.
  Attribute '_org' defines the starting point of the segment.

  The second ending point can be computed with the direction of the bone _dir
  whose magnitude equals the length of the bone.

  Bones points toward the skeleton leaves. Usually bones at the leaves are zero
  length

  @code                       ->
       _org           _dir   (_org + _dir)
         +-------------->---------+
      joint                    son_joint (if any)
  @endcode

  Usually bones are associated to an implicit primitive. Thus this class is
  often subclass.
*/
class Bone_cu {
public:
    
    Bone_cu() { }

    
    Bone_cu(const Vec& p1, const Vec& p2, float r):
        _org(p1),
        _radius(r),
        _dir(p2-p1),
        _length((p2-p1).norm())
    { }

    
    Bone_cu(const Vec& org, const Vec& dir, float length, float r):
        _org(org),
        _radius(r),
        
        _length( length )
    {
		_dir = dir;
		_dir.normalize();
		_dir = _dir* length;

	}

    // -------------------------------------------------------------------------
    /// @name Getters
    // -------------------------------------------------------------------------
    Vec org()    const{ return _org;        }
    Vec end()    const{ return _org + _dir; }
    float  length() const{ return _length;     }
    float  radius() const{ return _radius;     }
    Vec   dir()    const{ return _dir;        }

    void set_length (float  l ){ _length = l;    }
    void set_radius (float  r ){ _radius = r;    }
    void incr_radius(float ext){ _radius += ext; }

    // -------------------------------------------------------------------------
    /// @name Setters
    // -------------------------------------------------------------------------

    
    void set_start_end(const Vec& p0, const Vec& p1){
        _org = p0;
        _dir = p1 - p0;
        _length = _dir.norm();
    }

    
    void set_orientation(const Vec& org, const Vec& dir){
        _org = org;
		_dir = dir;
		_dir.normalize();
        _dir = dir * _length;
    }

    // -------------------------------------------------------------------------
    /// @name Utilities
    // -------------------------------------------------------------------------

    /// 'p' is projected on the bone line,
    /// then it returns the distance from the  origine '_org'
    
    float dist_proj_to(const Vec& p) const
    {
        const Vec op = p - _org;
		Vec temp = _dir;
		temp.normalize();
        return op*temp;
    }

    /// Orthogonal distance from the bone line to a point
    
    float dist_ortho_to(const Vec& p) const
    {
        const Vec op = p - _org;
		Vec temp = _dir;
		temp.normalize();
        return (op^temp).norm();
    }

    /// squared distance from a point 'p'
    /// to the bone's segment (_org; _org+_dir).
    
    float dist_sq_to(const Vec& p) const {
        Vec op = p - _org;
        float x = op*(_dir) / (_length * _length);
        x = fminf(1.f, fmaxf(0.f, x));
        Vec proj = _org + _dir * x;
        float d = (proj-p).squaredNorm();
        return d;
    }

    /// euclidean distance from a point 'p'
    /// to the bone's segment (_org; _org+_dir).
    
    float dist_to(const Vec& p) const {
        return sqrtf( dist_sq_to( p ) );
    }

    /// project p on the bone segment if the projection is outside the segment
    /// then returns the origin or the end point of the bone.
    
    Vec project(const Vec& p) const
    {
        const Vec op = p - _org;
		Vec temp = _dir;
		temp.normalize();
        float d = op*temp; // projected dist from origin

        if(d < 0)            return _org;
        else if(d > _length) return _org + _dir;
        else                 return _org + temp * d;
    }

    /// Get the local frame of the bone. This method only guarantes to generate
    /// a frame with an x direction parallel to the bone and centered about '_org'
    
    Tbx::Transfo get_frame() const
    {
        Vec x = _dir;
		x.normalize();
		x^Vec(0.f, 1.f, 0.f);
        Vec ortho = x^(Vec(0.f, 1.f, 0.f));
        Vec z, y;
        if (ortho.squaredNorm() < 1e-06f * 1e-06f)
        {
            ortho = Vec(0.f, 0.f, 1.f)^x;
            y = ortho;
			y.normalize();
            z = x^y;
			z.normalize();
        }
        else
        {
            z = ortho;
			z.normalize();
            y = z^x;
			y.normalize();
        }

        return Tbx::Transfo(  Tbx::Mat3( Tbx::Vec3(x.x,x.y,x.z), Tbx::Vec3(y.x,y.y,y.z), Tbx::Vec3(z.x,z.y,z.z)), 
			Tbx::Vec3(_org.x ,_org.y , _org.z) );
    }

    // -------------------------------------------------------------------------
    /// @name Attributes
    // -------------------------------------------------------------------------
protected:
    Vec _org; ///< Bone origin (first joint position)
    float _radius; ///< Bone radius
    Vec _dir;  ///< Bone direction towards its son if any
    float _length; ///< Bone length (o + v.normalized*length = bone_end_point)
};
// =============================================================================

/** @class Bone_cu
  @brief Subclass of Bone_cu.

  @see Bone_type Bone_cu
*/
class Bone : public Bone_cu {
public:
    friend struct Skeleton;

    Bone() : Bone_cu(), _bone_id(-1) {
        _length = 0.f;
        _radius = 0.f;
        _dir    = Vec(0.f, 0.f, 0.f);
        _org    = Vec(0.f, 0.f, 0.f);
    }

    virtual ~Bone(){}

    EBone::Id get_bone_id() const { return _bone_id; }

    Bone_cu get_bone_cu() const {
        assert(_bone_id != -1  );
        assert(_length  >= 0.f );
        assert(_radius  >= 0.f );
        return Bone_cu(_org, _org+_dir, _radius);
    }

    /// Get the bone type
    /// @see Bone_type
    virtual EBone::Bone_t get_type() const = 0;

    /// Get the oriented bounding box associated to the bone
 //   virtual Obbox get_obbox() const;
    /// Get the axis aligned bounding box associated to the bone
 //   virtual Bbox3 get_bbox() const;

protected:
    EBone::Id _bone_id; ///< Bone identifier in skeleton class
};

// =============================================================================

// =============================================================================

namespace EBone {

/// @param type Bone's type from the enum field of Bone_type namespace
std::string type_to_string(int type);

}

#endif // BONE_HPP__
