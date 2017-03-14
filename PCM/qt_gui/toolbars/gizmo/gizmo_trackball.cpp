#include "gizmo_trackball.hpp"

#include "toolbox/gl_utils/glsave.hpp"
#include "../../global_datas/g_vbo_primitives.hpp"
#include "toolbox/gl_utils/glu_utils.hpp"
#include "toolbox/maths/trackball.hpp"
#include "../../rendering/camera.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
using namespace Tbx;

// -----------------------------------------------------------------------------

Gizmo_trackball::Gizmo_trackball() : Gizmo(),
    _rad_sphere(6.5f),
    _rad_handles(1.0f),
    _ortho_factor(20.f),
    _selected_axis(NOAXIS)
{
    _frame = Transfo::identity();
}

// -----------------------------------------------------------------------------

void Gizmo_trackball::draw(const Camera& cam)
{
    if(!_show) return;

    GLEnabledSave save_light(GL_LIGHTING  , true, false);
    GLEnabledSave save_textu(GL_TEXTURE_2D, true, false);
    GLEnabledSave save_depth(GL_DEPTH_TEST, true, true);

    Vec3 org = _frame.get_translation();
    const float dist_cam =  !cam.is_ortho() ? (org-cam.get_pos()).norm() : _ortho_factor;
    float s = dist_cam / 50.f;

    GLLineWidthSave save_line_width;

    glLineWidth( _rad_handles );

    glPushMatrix();
    {
        glMultMatrixf( _frame.transpose().m );
        glScalef(s, s, s);
        glScalef(_rad_sphere, _rad_sphere, _rad_sphere);

        glColor4f(1.f, 1.f, 1.f, 1.f);

        // X axis
        glPushMatrix();
        glRotatef(90.f, 0.f, 1.f, 0.f);
        g_primitive_printer.draw( g_circle_vbo );
        glPopMatrix();

        // Y axis
        glPushMatrix();
        glRotatef(90.f, 1.f, 0.f, 0.f);
        g_primitive_printer.draw( g_circle_vbo );
        glPopMatrix();

        // Z Axis
        g_primitive_printer.draw( g_circle_vbo );
    }
    glPopMatrix();
}

// -----------------------------------------------------------------------------

bool Gizmo_trackball::select_constraint(const Camera& cam, int px, int py)
{
    if(!_show){
        _selected_axis = NOAXIS;
        return false;
    }

    _old_frame = _frame;
    py = cam.height() - py; /////////// Invert because of opengl
    _clicked.x = px;
    _clicked.y = py;
    _selected_axis = BALL;
    return true;
}

// -----------------------------------------------------------------------------

TRS Gizmo_trackball::slide(const Camera& cam, int px, int py)
{
    py = cam.height() - py; /////////// Invert because of opengl
    if( _selected_axis == BALL) return trackball_rotation(cam, px, py);
    else                        return TRS();
}

// -----------------------------------------------------------------------------

TRS Gizmo_trackball::trackball_rotation(const Camera& cam, int px, int py)
{
    const Point3 org    = Point3(_frame.get_translation());
    const Vec3  picked = Vec3((float)_clicked.x, (float)_clicked.y, 0.f);
    const Vec3  curr   = Vec3((float)px, (float)py, 0.f);
    const Vec3  mouse_vec = curr - picked;

    if( mouse_vec.norm() < 0.001f ) return TRS();

#if 0
    // Pseudo trackBall
    Vec3 fx = cam.get_x();
    Vec3 fy = cam.get_y();

    Vec3 axis = fx * mouse_vec.y + fy * mouse_vec.x;
    axis.normalize();

    float norm  = mouse_vec.norm() / 60.f;
    float angle = fmodf( norm, 2.f* M_PI);

    // Rotation in local coordinates
    return TRS::rotation(_old_frame.fast_invert() * axis, angle);
#else

    Point3 proj_org = cam.project(org);

    TrackBall ball(cam.width(), cam.height(), proj_org.x, proj_org.y, 0.5f);
    ball.set_picked_point(picked.x, picked.y);

    Vec3 eye_axis;
    float angle;
    ball.roll( (float*)&eye_axis, angle, curr.x, curr.y );
    Vec3 world_axis = cam.get_eye_transfo().fast_invert() * eye_axis;

    // Rotation in local coordinates
    return TRS::rotation(_old_frame.fast_invert() * world_axis, angle);
#endif

}

// -----------------------------------------------------------------------------
