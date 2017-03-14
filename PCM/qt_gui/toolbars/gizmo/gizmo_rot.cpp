#include "gizmo_rot.hpp"

#include "toolbox/gl_utils/glsave.hpp"
#include "toolbox/gl_utils/glu_utils.hpp"
#include "toolbox/maths/trackball.hpp"
#include "../../global_datas/g_vbo_primitives.hpp"
#include "../../rendering/camera.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
using namespace Tbx;

// -----------------------------------------------------------------------------

Gizmo_rot::Gizmo_rot() : Gizmo(),
    _enable_trackball(false),
    _rad_sphere(6.5f),
    _rad_handles(2.0f),
    _ortho_factor(20.f),
    _selected_axis(NOAXIS)
    
{
//	_pick = 2;
    _pick._pick_size = 3.0f;
    _frame = Transfo::identity();
}

// -----------------------------------------------------------------------------

static void draw_thick_prim( Prim_id id )
{
    // HACK: to draw thicker lines and fill gaps. This way selection
    // is easier for the user when clicking

    for (int i = 0; i < 8; ++i) {
        float s = i*0.01f;
        glPushMatrix();
        glTranslatef(0.f, 0.f, s);
        glScalef( 1.f+s, 1.f+s, 1.f+s );
        g_primitive_printer.draw( id );
        glRotatef(-15.f, 0.f, 0.f, 1.f);
        g_primitive_printer.draw( id );
        glRotatef(30.f, 0.f, 0.f, 1.f);
        g_primitive_printer.draw( id );
        glPopMatrix();

        glPushMatrix();
        glTranslatef(0.f, 0.f, -s);
        glScalef( 1.f-s, 1.f-s, 1.f-s );
        g_primitive_printer.draw( id );
        glRotatef(-15.f, 0.f, 0.f, 1.f);
        g_primitive_printer.draw( id );
        glRotatef(30.f, 0.f, 0.f, 1.f);
        g_primitive_printer.draw( id );
        glPopMatrix();
    }

}

// -----------------------------------------------------------------------------

void Gizmo_rot::draw_arc_circle(Axis_t axis,
                                const Vec3& a,
                                const Vec3& cam_dir,
                                float scale,
                                const Color& cl)
{
    glPushMatrix();
    {
        if( _selected_axis == axis ) Color::yellow().set_gl_state();
        else                         cl.set_gl_state();

        _pick.set_name(axis);

        Vec3 x, y, z;
        z = a.normalized();
        x = z.proj_on_plane( -cam_dir );
        if( x.normalize() > 0.0001f )
        {
            y = x.cross( z );
            Transfo tr = Transfo( Mat3(x, y, z) );
            tr.set_translation( _frame );
            glMultMatrixf( tr.transpose().m );

            glScalef(scale, scale, scale);

            glRotatef(-90.f, 0.f, 0.f, 1.f);
            g_primitive_printer.draw( g_arc_circle_vbo );


            if( _pick.is_pick_init() )
                draw_thick_prim(g_arc_circle_lr_vbo);

        }
    }
    glPopMatrix();
}

// -----------------------------------------------------------------------------

void Gizmo_rot::draw_circle(Axis_t axis,
                            const Vec3& cam_dir,
                            float scale,
                            const Color& cl)
{
    glPushMatrix();
    {
        // CAM
        Vec3 fy, fz;
        cam_dir.coordinate_system(fy, fz);
        Transfo tr(Mat3(cam_dir, fy, fz), _frame.get_translation());

        glMultMatrixf( tr.transpose().m );
        glScalef(scale, scale, scale);

        if( _selected_axis == axis && axis != NOAXIS )
            Color::yellow().set_gl_state();
        else
            cl.set_gl_state();

        _pick.set_name(axis);
        glRotatef(90.f, 0.f, 1.f, 0.f);
        g_primitive_printer.draw( g_circle_vbo );

        if( _pick.is_pick_init() && axis != NOAXIS )
            draw_thick_prim( g_circle_lr_vbo );
    }
    glPopMatrix();
}

// -----------------------------------------------------------------------------

void Gizmo_rot::draw(const Camera& cam)
{
    if(!_show) return;

    GLEnabledSave save_light ( GL_LIGHTING  , true, false);
    GLEnabledSave save_textu ( GL_TEXTURE_2D, true, false);
    GLEnabledSave save_depth ( GL_DEPTH_TEST, true, false);
    GLEnabledSave save_smooth(GL_LINE_SMOOTH, false, false);

    Vec3 org = _frame.get_translation();
    const float dist_cam =  !cam.is_ortho() ? (org-cam.get_pos()).norm() : _ortho_factor;
    float s = dist_cam / 50.f;

    GLLineWidthSave save_line_width;

    Vec3 cam_dir = cam.get_dir();

    if(_pick.is_pick_init()){
        glLineWidth( _rad_handles*20.0f ); // So that picking is easier
        glDisable(GL_LINE_SMOOTH);
    }
    else
        glLineWidth( _rad_handles );



    float fac = s * _rad_sphere * 0.75f;

    // black outline of the sphere
    draw_circle(NOAXIS, cam_dir, fac, Color::black());

    draw_arc_circle(X, _frame.x(), cam_dir, fac, Color::red()   );
    draw_arc_circle(Y, _frame.y(), cam_dir, fac, Color::green() );
    draw_arc_circle(Z, _frame.z(), cam_dir, fac, Color::blue()  );

    // Circle parallel to the camera
    draw_circle(CAM, cam_dir, s * _rad_sphere, Color::white());

    if(_selected_axis < NOAXIS) draw_tangent( cam_dir, s );
}

// -----------------------------------------------------------------------------

void Gizmo_rot::draw_tangent(const Vec3& cam_dir, float dist_cam)
{
    GLLineWidthSave save_line_width(_rad_handles);
    glColor4f(1.f, 1.f, 0.f, 1.f);
    Vec3 t  = _picked_tangent.normalized() * dist_cam * _rad_sphere / 2.f;
    Vec3 p0 = _picked_pos - t; // First point on the tangent
    Vec3 p1 = _picked_pos + t; // Second point on the tangent

    Vec3 n  = cam_dir.cross(_picked_tangent).normalized();
    t = t.normalized() * 2.f;
    Vec3 v0 = (  n - t).normalized() * dist_cam;
    Vec3 v1 = (- n - t).normalized() * dist_cam;

    glBegin(GL_LINES);
    // A line for the tangent
    //glVertex3f(p0.x, p0.y, p0.z);
    //glVertex3f(p1.x, p1.y, p1.z);
    //glColor4f(0.f, 0.f, 0.f, 1.f);
    // First tip of the arrow
    glVertex3f(p1.x, p1.y, p1.z);
    Vec3 p = p1 + v0;
    glVertex3f(p.x, p.y, p.z);

    glVertex3f(p1.x, p1.y, p1.z);
    p = p1 + v1;
    glVertex3f(p.x, p.y, p.z);

    // Second tip of the arrow
    glVertex3f(p0.x, p0.y, p0.z);
    p = p0 - v0;
    glVertex3f(p.x, p.y, p.z);

    glVertex3f(p0.x, p0.y, p0.z);
    p = p0 - v1;
    glVertex3f(p.x, p.y, p.z);

    glEnd();
}

// -----------------------------------------------------------------------------

bool Gizmo_rot::select_constraint(const Camera& cam, int px, int py)
{
    if(!_show){
        _selected_axis = NOAXIS;
        return false;
    }

    _old_frame = _frame;
    py = cam.height() - py; /////////// Invert because of opengl

    GLfloat m[16];
    glGetFloatv(GL_PROJECTION_MATRIX, m);
    _pick.begin( m, (GLfloat)px, (GLfloat)py );
    draw( cam );
    int idx = _pick.end();
    _selected_axis =  idx > -1 ? (Axis_t)idx : NOAXIS;

    Vec3 a[4] = { _frame.x(), _frame.y(), _frame.z(), cam.get_dir() };
    if( _selected_axis < NOAXIS ) _axis = a[_selected_axis].normalized();
    _clicked.x = px;
    _clicked.y = py;

    if(idx > -1){
        _pick.world_picked_pos( (GLfloat*)&_picked_pos );
        Vec3 normal  = _picked_pos - _frame.get_translation();
        _picked_tangent = normal.cross( _axis ).normalized();
    }

    return (idx > -1);
}

// -----------------------------------------------------------------------------

TRS Gizmo_rot::slide(const Camera& cam, int px, int py)
{
    py = cam.height() - py; /////////// Invert because of opengl

    if( _selected_axis < NOAXIS ) return axis_rotation(cam, px, py);
    else                          return TRS();
}

// -----------------------------------------------------------------------------

TRS Gizmo_rot::axis_rotation(const Camera& cam, int px, int py)
{
    Vec3 picked = Vec3((float)_clicked.x, (float)_clicked.y, 0.f);
    Vec3 curr   = Vec3((float)px, (float)py, 0.f);
    Vec3 mouse_vec = curr - picked;

    if( mouse_vec.norm() < 0.001f ) return TRS();

    Vec3 t  = _picked_tangent.normalized() * 3.0f;
    Point3  p0( _picked_pos - t );
    Point3  p1( _picked_pos + t );

    //Vec3 proj_tangent = Glu_utils::project( p0 ) - Glu_utils::project( p1 );
    Vec3 proj_tangent = cam.project( p0 ) - cam.project( p1 );
    proj_tangent.normalize();


    float dot = proj_tangent.dot(  mouse_vec ) / 50.f;
    float angle = (dot > 0.f ? 1.f : -1.f) * fmodf( fabsf(dot), 2.f* M_PI);

    // Rotation in local coordinates
    return TRS::rotation(_old_frame.fast_invert() * _axis, angle);
}

// -----------------------------------------------------------------------------
