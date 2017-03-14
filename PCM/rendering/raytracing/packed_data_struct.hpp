#ifndef PACKED_DATA_STRUCT_HPP__
#define PACKED_DATA_STRUCT_HPP__

/// Various structures used in some CUDA kernels
/// This diminish the number of attributes to write when calling a kernel

#include "toolbox/maths/point3.hpp"
#include "camera.hpp"

// =============================================================================

/// @brief A class that stores the attributes of teh camera
struct Camera_data{

    Point3 _pos;
    Vec3  _dir;
    Vec3  _hor;
    Vec3  _ver;
    float    _near;
    float    _far;
    float    _fov;
    bool     _ortho;

    IF_CUDA_DEVICE_HOST inline
    Camera_data(): _pos(), _dir(), _hor(), _ver(), _near(0.f), _far(0.f), _fov(0.f), _ortho(false)
    {   }

    IF_CUDA_DEVICE_HOST inline
    Camera_data(const Point3& p,	const Vec3& d,	const Vec3& h, const Vec3& v, float n, float f):
    _pos(p), _dir(d), _hor(h), _ver(v), _near(n), _far(f)
    {   }

    Camera_data(const Camera& cam)
    {
        _pos   = cam.get_pos().to_point3();
        _dir   = cam.get_dir();
        _hor   = -cam.get_x(); // FIXME: why is this inverted ? (trace back the use of camera_data to find that)
        _ver   = cam.get_y();
        _near  = cam.get_near();
        _far   = cam.get_far();
        _ortho = cam.is_ortho();
        _fov = _ortho ? cam.get_ortho_zoom() : cam.fovy();
    }
};

// =============================================================================

/// @brief A class that stores the size of the PBOs and the addresses
/// they ared mapped to
struct PBO_data {
    float4* d_rendu;
    float*  d_depth;

    int start_x;
    int start_y;
    int width;
    int height;

    IF_CUDA_DEVICE_HOST inline PBO_data():
        d_rendu(0), d_depth(0), start_x(0), start_y(0), width(0), height(0)
    {   }

    IF_CUDA_DEVICE_HOST inline
    PBO_data(float4* d_r, float* d_d,	int s_x, int s_y, int w, int h) :
        d_rendu(d_r), d_depth(d_d), start_x(s_x), start_y(s_y), width(w), height(h)
    {  }
};

// =============================================================================

#endif // PACKED_DATA_STRUCT_HPP__
