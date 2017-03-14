#include "gl_skeleton.hpp"

#include "skeleton.hpp"
#include "toolbox/gl_utils/glsave.hpp"
#include "global_datas/g_vbo_primitives.hpp"

#include "toolbox/gl_utils/gldirect_draw.hpp"

using namespace Tbx;
// -----------------------------------------------------------------------------
//static GlDirect_draw* draw;
#include <QTime>

GL_skeleton::GL_skeleton(const Skeleton* skel) :
    _skel(skel)
{
    /*
    draw = new GlDirect_draw();
    float l  = 50.f;
    float js =0.2f;
    float ps =0.2f;

    // Some tests with direct draw
    //l = 1.f;
    const float b0 = ps + (l / 10.f); // length of the base
    const float w0 = l / 15.f;        // width of the base


    draw->normal3f(0.f, 1.f, 0.f);
    draw->set_auto_normals( true );
    draw->color3f(0.5f, 1.f, 0.f);
    // First pyramid
    draw->begin( GL_TRIANGLE_FAN );
    {
        draw->vertex3f( ps, 0.f, 0.f );
        draw->vertex3f( b0, 0.f, -w0 );
        draw->vertex3f( b0, -w0, 0.f );
        draw->vertex3f( b0, 0.f,  w0 );
        draw->vertex3f( b0,  w0, 0.f );
        draw->vertex3f( b0, 0.f, -w0 );
    }
    draw->end();

    const float w1 = w0 / 3.f; // Width of the base at the opposite
    l = l-js;
    draw->begin( GL_QUAD_STRIP );
    {
        draw->vertex3f( b0, 0.f, -w0 );// a
        draw->vertex3f(  l, 0.f, -w1 );// 0
        draw->vertex3f( b0, -w0, 0.f );// b
        draw->vertex3f(  l, -w1, 0.f );// 1
        draw->vertex3f( b0, 0.f,  w0 );// c
        draw->vertex3f(  l, 0.f,  w1 );// 2
        draw->vertex3f( b0,  w0, 0.f );// d
        draw->vertex3f(  l,  w1, 0.f );// 3
        draw->vertex3f( b0, 0.f, -w0 );// a
        draw->vertex3f(  l, 0.f, -w1 );// 0
    }
    draw->end();

    // The bone's cap is flat
    draw->begin( GL_QUADS );
    {
        draw->vertex3f( l, 0.f, -w1 );
        draw->vertex3f( l, -w1, 0.f );
        draw->vertex3f( l, 0.f,  w1 );
        draw->vertex3f( l,  w1, 0.f );
    }
    draw->end();
    */

}

// -----------------------------------------------------------------------------

int GL_skeleton::nb_joints() const { return _skel->nb_joints(); }

// -----------------------------------------------------------------------------
//#include <sys/time.h>

static void custom_bone( float l  /*bone_length*/,
                         float js /*joint radius*/,
                         float ps /* parent joint radius*/)

{
#if 0

    /*
    using namespace std;
    //    QTime  fps_timer;

    timeval t1, t2;
    double elapsedTime;
    // start timer

    gettimeofday(&t1, NULL);

    //        fps_timer.start();

    // Some tests with direct draw
    GlDirect_draw draw;
    //l = 1.f;
    const float b0 = ps + (l / 10.f); // length of the base
    const float w0 = l / 15.f;        // width of the base

    // First pyramid

    draw.begin( GL_TRIANGLE_FAN );
    {
        draw.vertex3f( ps, 0.f, 0.f );
        draw.vertex3f( b0, 0.f, -w0 );
        draw.vertex3f( b0, -w0, 0.f );
        draw.vertex3f( b0, 0.f,  w0 );
        draw.vertex3f( b0,  w0, 0.f );
        draw.vertex3f( b0, 0.f, -w0 );

    }
    draw.end();

    const float w1 = w0 / 3.f; // Width of the base at the opposite
    l = l-js;
    draw.begin( GL_QUAD_STRIP );
    {
        draw.vertex3f( b0, 0.f, -w0 );// a
        draw.vertex3f(  l, 0.f, -w1 );// 0
        draw.vertex3f( b0, -w0, 0.f );// b
        draw.vertex3f(  l, -w1, 0.f );// 1
        draw.vertex3f( b0, 0.f,  w0 );// c
        draw.vertex3f(  l, 0.f,  w1 );// 2
        draw.vertex3f( b0,  w0, 0.f );// d
        draw.vertex3f(  l,  w1, 0.f );// 3
        draw.vertex3f( b0, 0.f, -w0 );// a
        draw.vertex3f(  l, 0.f, -w1 );// 0
    }
    draw.end();

    // The bone's cap is flat
    draw.begin( GL_QUADS );
    {
        draw.vertex3f( l, 0.f, -w1 );
        draw.vertex3f( l, -w1, 0.f );
        draw.vertex3f( l, 0.f,  w1 );
        draw.vertex3f( l,  w1, 0.f );
    }
    draw.end();


    GLfloat proj[16];
    GLfloat mv  [16];
    glGetFloatv(GL_PROJECTION_MATRIX, proj);
    glGetFloatv(GL_MODELVIEW_MATRIX , mv  );
    draw.set_matrix( mv, proj);
    draw.draw();

    draw.clear();

    gettimeofday(&t2, NULL);
    // compute and print the elapsed time in millisec
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    cout << elapsedTime << " ms.\n";

    //    std::cout << fps_timer.elapsed() << " sec" << std::endl;
    // stop timer
*/
//    glEnable(GL_LIGHTING);
//    glEnable(GL_LIGHT0);
    GLfloat proj[16];
    GLfloat mv  [16];
    glGetFloatv(GL_PROJECTION_MATRIX, proj);
    glGetFloatv(GL_MODELVIEW_MATRIX , mv  );
    draw->set_matrix( mv, proj);
    draw->draw();


#else

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
#endif
}

// -----------------------------------------------------------------------------

static void draw_bone_body(const Transfo& b_frame,
                           const Point3& p0,
                           const Point3& p1,
                           float rad_joint,
                           float rad_pjoint)
{
    glPushMatrix();
#if 0
    Vec3 fx = (p1-p0).normalized(), fy, fz;
    fx.coordinate_system(fy, fz);
    Transfo tr( Mat3_cu(fx, fy, fz), p0.to_vec3() );
    glMultMatrixf( tr.transpose().m );
#else
    Transfo tr = b_frame;
    glAssert( glMultMatrixf( tr.transpose().m ) );
#endif
    custom_bone( (p1-p0).norm(), rad_joint, rad_pjoint);
    glPopMatrix();
}

// -----------------------------------------------------------------------------

static void draw_frame(const Transfo& frame, float size_axis, bool color)
{
    GLLineWidthSave save_line_width( 2.0f );

    Transfo tr = frame.normalized();
    Point3 pos = tr.get_translation().to_point3();

    Point3 dx = pos + tr.x() * size_axis;
    Point3 dy = pos + tr.y() * size_axis;
    Point3 dz = pos + tr.z() * size_axis;

    glBegin(GL_LINES);{
        // Local frame
        if(color) glColor4f(1.f, 0.f, 0.f, 1.f);
        glVertex3f(pos.x, pos.y, pos.z);
        glVertex3f(dx.x, dx.y, dx.z);
        if(color) glColor4f(0.f, 1.f, 0.f, 1.f);
        glVertex3f(pos.x, pos.y, pos.z);
        glVertex3f(dy.x, dy.y, dy.z);
        if(color) glColor4f(0.f, 0.f, 1.f, 1.f);
        glVertex3f(pos.x, pos.y, pos.z);
        glVertex3f(dz.x, dz.y, dz.z);
    }glAssert( glEnd() );

}

// -----------------------------------------------------------------------------

static void draw_joint( const Transfo& b_frame, float fc, bool use_circle)
{
    glAssert( glPushMatrix() );
    Transfo tr = b_frame;
    glAssert( glMultMatrixf( tr.transpose().m ) );
    if( !use_circle )
    {
        glScalef(fc, fc, fc);
        g_primitive_printer.draw( g_sphere_lr_vbo );
    }
    else
    {
        const float nfc = fc * 1.1f;
        glScalef(nfc, nfc, nfc);
        g_primitive_printer.draw( g_circle_lr_vbo );
        glRotatef(90.f, 1.f, 0.f, 0.f);
        g_primitive_printer.draw( g_circle_lr_vbo );
        glRotatef(90.f, 0.f, 1.f, 0.f);
        g_primitive_printer.draw( g_circle_lr_vbo );
    }
    glPopMatrix();
}

// -----------------------------------------------------------------------------

void GL_skeleton::draw_bone(int i, const Color& c, bool rest_pose, bool use_material, bool use_circle)
{
    const Bone* bone = _skel->get_bone(i);
    const float len  = bone->length();
    const float rad  = (len / 30.f);

    const Transfo b_frame = rest_pose ? _skel->bone_frame(i) : _skel->bone_anim_frame(i);
    const Point3 org = bone->org();
    const Point3 end = bone->end();

    glAssert( glMatrixMode(GL_MODELVIEW) );

    if(use_material) c.set_gl_state();
    draw_joint(b_frame, rad, use_circle );

    const float axis_size = 0.3f;

    // Draw bone body
    if( _skel->is_leaf(i) )
    {
        draw_frame(b_frame, axis_size, use_material);
        const int pt = _skel->parent( i );
        if(pt >= 0)
        {
            if(use_material) c.set_gl_state();
            draw_joint( b_frame, _skel->get_bone(pt)->length() / 50.f, use_circle);
        }
    }
    else
    {
        if(use_material){
            Color c = Color::pseudo_rand(i);
            glAssert( glColor4f(c.r, c.g, c.b, 1.f) );
        }
        draw_bone_body(b_frame, org, end, rad, rad);
    }
}

// -----------------------------------------------------------------------------
