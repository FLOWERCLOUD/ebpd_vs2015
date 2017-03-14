#include "qt_gui/paint_canvas.h"
#include "toolbox/gl_utils/glsave.hpp"
#include "toolbox/maths/color.hpp"
#include "rendering/rendering.hpp"
#include "QGLViewer/camera.h"
#include "camera.hpp"
#include "toolbox/gl_utils/vbo_primitives.hpp"
#include "../control/cuda_ctrl.hpp"
#include "../global_datas/toolglobals.hpp"
#include "../global_datas/cuda_globals.hpp"
#include "../animation/animesh.hpp"
#include "../animation/skeleton.hpp"
#include "sample_set.h"
#include "tracer.h"
#include "globals.h"
#include "GlobalObject.h"
#include "render_types.h"
#include "tool.h"
extern bool isShowNoraml;
extern RenderMode::WhichColorMode	which_color_mode_;
extern RenderMode::RenderType which_render_mode;

extern Tbx::VBO_primitives g_primitive_printer;
extern Tbx::Prim_id g_sphere_lr_vbo;
extern Tbx::Prim_id g_sphere_vbo;
extern Tbx::Prim_id g_circle_vbo;
extern Tbx::Prim_id g_arc_circle_vbo;
extern Tbx::Prim_id g_circle_lr_vbo;
extern Tbx::Prim_id g_arc_circle_lr_vbo;
extern Tbx::Prim_id g_grid_vbo;
extern Tbx::Prim_id g_cylinder_vbo;
extern Tbx::Prim_id g_cylinder_cage_vbo;
extern Tbx::Prim_id g_cube_vbo;

static void draw_junction_sphere(bool rest_pose)
{
	if(g_skel == 0 || !Cuda_ctrl::_display._junction_spheres)
		return;

	glColor4f(0.4f, 0.3f, 1.f, 0.5f);
	const std::vector<int>& set = Cuda_ctrl::_skeleton.get_selection_set();
	for(unsigned i = 0; i < set.size(); i++)
	{
		float r  = g_animesh->get_junction_radius(set[i]);
		Tbx::Vec3 v = rest_pose ? g_skel->joint_rest_pos(set[i]) : g_skel->joint_pos(set[i]);
		glPushMatrix();
		glTranslatef(v.x, v.y, v.z);
		glScalef(r, r, r);
		g_primitive_printer.draw(g_sphere_vbo);
		glPopMatrix();
	}
}

static void draw_bbox()
{
	if(!Cuda_ctrl::_display._oriented_bbox && !Cuda_ctrl::_display._aa_bbox)
		return;

	using namespace Cuda_ctrl;
	const std::vector<int>& selection_set = _skeleton.get_selection_set();
	_color.get(Color_ctrl::BOUNDING_BOX).set_gl_state();
	for(unsigned int i = 0; i < selection_set.size(); i++)
	{
		int id = selection_set[i];
		const Bone* b = g_skel->get_bone(id);

		if(b->get_type() == EBone::SSD)
			continue;

		Tbx::Obbox obbox = b->get_obbox();

		Tbx::Vec3 lengths = obbox._bb.lengths();
		if(Cuda_ctrl::_display._oriented_bbox)
		{
			glPushMatrix();

			glMultMatrixf(obbox._tr.transpose().m);
			glTranslatef(obbox._bb.pmin.x, obbox._bb.pmin.y, obbox._bb.pmin.z);
			glScalef(lengths.x, lengths.y, lengths.z);

			glColor3f(0.f, 0.f, 0.f);
			g_primitive_printer.draw(g_cube_vbo);
			glPopMatrix();
		}

		if( Cuda_ctrl::_display._aa_bbox )
		{
			Tbx::Bbox3 bbox = b->get_bbox();

			lengths = bbox.lengths();
			glPushMatrix();


			glTranslatef(bbox.pmin.x, bbox.pmin.y, bbox.pmin.z);
			glScalef(lengths.x, lengths.y, lengths.z);

			glColor3f(1.f, 1.f, 0.f);
			g_primitive_printer.draw( g_cube_vbo );
			glPopMatrix();
		}
	}
}

static void draw_cylinder()
{
	using namespace Cuda_ctrl;
	const std::vector<int>& selection_set = _skeleton.get_selection_set();

	for(unsigned int i = 0; i<selection_set.size(); i++)
	{
		int id = selection_set[i];
		if( g_skel->bone_type(id) == EBone::CYLINDER )
		{
			const Bone* b = g_skel->get_bone(id);
			float rad = b->radius();
			glMatrixMode(GL_MODELVIEW);

			glPushMatrix();
			Tbx::Transfo tr_trans = b->get_frame().transpose();
			glMultMatrixf(tr_trans.m);
			glRotatef(90.f, 0.f, 1.f, 0.f);
			glScalef(rad, rad, b->length());
			g_primitive_printer.draw(g_cylinder_cage_vbo);
			glPopMatrix();
		}

	}
}





static void draw_mesh_points(const qglviewer::Camera* cam, bool rest_pose)
{
	if(Cuda_ctrl::_anim_mesh == 0) return;

	if(Cuda_ctrl::_anim_mesh->is_point_displayed())
	{
		// Do a little offset so mesh points won't hide rbf samples
		const float eps = 1.0001f;
		glPushMatrix();

		//Tbx::Vec3 p = cam->get_pos();
		Tbx::Vec3 p( cam->position().x,cam->position().y,cam->position().y);
		glTranslatef(p.x, p.y, p.z);
		glScalef(eps, eps, eps);
		glTranslatef(-p.x, -p.y, -p.z);

		glPointSize(9.f);
		Tbx::GLEnabledSave save_point (GL_POINT_SMOOTH, true, true );
		rest_pose ? g_animesh->draw_points_rest_pose() : g_mesh->draw_points();
		glPopMatrix();
	}
}

void draw_normals(const std::vector<int>& selected_points,
				  const std::vector<Tbx::Vec3>& d_ssd_normals)
{

	Tbx::Vec3* vert = 0;

	//g_mesh->_mesh_gl._vbo.map_to(vert, GL_READ_ONLY);

	glBegin(GL_LINES);
	glColor4f(1.f, 0.f, 0.f, 1.f);
	for(unsigned i = 0; i < selected_points.size(); i++)
	{
		Tbx::Vec3 n;
		n = d_ssd_normals[selected_points[i]];
		//d_ssd_normals.fetch(selected_points[i], n);
		n.normalize();
		n = n * 5.f;

		const EMesh::Packed_data d = g_mesh->get_packed_vert_map()[selected_points[i]];
		Tbx::Vec3 v = vert[ d._idx_data_unpacked ];

		//glColor3f(n.x, n.y, n.z);
		glVertex3f(v.x, v.y, v.z);
		glVertex3f(v.x + n.x, v.y + n.y, v.z + n.z);

	}
	glEnd();

	//if( g_animesh ) g_animesh->hd_verts_rots.update_host_mem();
	for(unsigned i = 0; i < selected_points.size(); i++)
	{
		if( g_animesh )
		{
			Tbx::Mat2 r = g_animesh->hd_verts_rots[selected_points[i]];
			((r+r)*0.5f).print();

			std::cout << "bla: "<< r.m[1] << std::endl;

			r = Tbx::Mat2::rotate( -asinf(r.m[1]) );
			((r+r)*0.5f).print();

			std::cout << std::endl;
		}
	}

	g_mesh->_mesh_gl._vbo.unmap();
}

void draw_animesh(bool use_color_array, bool use_point_color, bool rest_pose)
{
	if(g_animesh == 0) return;

	if(rest_pose) g_animesh->draw_rest_pose(use_color_array, use_point_color);
	else          g_animesh->draw(use_color_array, use_point_color);
}
static void draw_skeleton(const qglviewer::Camera* cam, bool draw_skel, bool rest_pose)
{
	using namespace Cuda_ctrl;
	std::vector<int> joints = _skeleton.get_selection_set();
	int              node   = _graph.get_selected_node();

	if     ( draw_skel && g_skel != 0 ) g_skel ->draw( Tbx::Camera(cam), joints, rest_pose);
	else if( g_graph != 0 )             g_graph->draw( Tbx::Camera(cam), node );
}

/// what is drawn here will be hidden by the transceluscent mesh
/// Hardware Antialiasing will work
static void plain_objects(const qglviewer::Camera* cam, const Render_context* ctx,
						  float r, float g, float b, float bfactor)
{
	using namespace Cuda_ctrl;
	//Cuda_ctrl::_potential_plane.draw();

	if( _display._wire )
	{
		glAssert( glColor4f(r, g, b, bfactor) );
		if( !ctx->_skeleton )
		{
			if ( _anim_mesh != 0 ) draw_animesh(true, true, true /*rest pose*/);
			else if( g_mesh != 0 ) g_mesh->draw(true, true);
		}
		else
		{
			if(ctx->_draw_mesh) draw_animesh(true, true, ctx->_rest_pose);
		}
	}

	if( !ctx->_plain_phong )
	{
		if(g_mesh!= 0 && g_mesh->get_nb_vertices()>0) g_mesh->debug_draw_edges();

		draw_mesh_points(cam, ctx->_rest_pose);
		if( Cuda_ctrl::_display._grid) draw_grid_lines(cam);

		draw_bbox();
		draw_cylinder();

		if(_debug._show_normals)
			draw_normals(_anim_mesh->get_selected_points(),
			g_animesh->get_ssd_normals());

		if(_debug._show_gradient)
			_debug.draw_gradient( _anim_mesh->get_selected_points(),
			g_animesh->get_gradient(),
			g_animesh->d_vert_buffer_2 /*g_animesh->d_grad_transfo.ptr()*/);

	}
	return;
	g_primitive_printer.draw(g_cylinder_vbo);
	glAssert( glColor4f(0.f, 1.f, 0.f, 1.f) );
	g_primitive_printer.draw(g_circle_lr_vbo);
	g_primitive_printer.draw(g_arc_circle_vbo);

	g_primitive_printer.draw(g_grid_vbo);
	glAssert( glColor4f(0.f, 0.f, 1.f, 0.2f) );
	glDepthMask(false);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glTranslatef(-2.0f ,-2.0f ,-2.0f);
	glScalef( 4.0f ,4.0f ,4.0f);
	g_primitive_printer.draw(g_cube_vbo);
	glPopMatrix();
	glDepthMask(true);
}


/// What follows won't benefit from hardware AA, but will be visible in
/// transluscent mode
static void transparent_objects(const qglviewer::Camera* cam , const Render_context* ctx)
{
	using namespace Cuda_ctrl;
//	Cuda_ctrl::_potential_plane.draw();


	if( !ctx->_skeleton )
	{
		if     ( _anim_mesh != 0 ) draw_animesh(true, false, true /*rest pose*/);
		else if( g_mesh     != 0 ) g_mesh->draw();
	}
	else
	{
		if(ctx->_draw_mesh) draw_animesh(true, false, ctx->_rest_pose);
	}


	if( !ctx->_plain_phong  && Cuda_ctrl::_skeleton.is_displayed())
	{
		if(_anim_mesh != 0)
			Cuda_ctrl::_anim_mesh->draw_rotation_axis();

		glAssert( glColor4f(1.f, 0.f, 0.f, 1.f) );
		draw_junction_sphere(ctx->_rest_pose);
		draw_skeleton(cam, ctx->_skeleton, ctx->_rest_pose);
	}
	return;
	glAssert( glColor4f(1.f, 0.f, 0.f, 1.f) );
	g_primitive_printer.draw(g_circle_lr_vbo);
	g_primitive_printer.draw(g_arc_circle_vbo);
	g_primitive_printer.draw(g_cube_vbo);
	g_primitive_printer.draw(g_grid_vbo);
	glAssert( glColor4f(0.f, 0.f, 1.f, 0.5f) );
	glDepthMask(false);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glTranslatef(-2.0f ,-2.0f ,-2.0f);
	glScalef( 4.0f ,4.0f ,4.0f);
	g_primitive_printer.draw(g_cube_vbo);
	glPopMatrix();
	glDepthMask(true);

}

bool raytrace( const qglviewer::Camera* cam)
{




	return false;

}

bool display_loop(const qglviewer:: Camera* _cam ,Render_context* _render_ctx)
{

	using namespace Tbx;
	const int width  = _cam->screenWidth();
	const int height = _cam->screenHeight();


//	Color cl = Color(1.0f ,1.0f ,1.0f ,1.0f);
    Color cl = Cuda_ctrl::_color.get(Color_ctrl::BACKGROUND);

	glClearColor(cl.r, cl.g, cl.b, cl.a) ;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) ;

	bool refresh = false;

//	refresh = raytrace(_cam);
	std::vector<GLuint> depthValue( _render_ctx->pbo_depth()->size(),0xffffffff);
	std::vector<GLint> colorValue( _render_ctx->pbo_color()->size(),0xffffff);
	_render_ctx->pbo_depth()->set_data(depthValue ,GL_STREAM_DRAW);
	_render_ctx->pbo_color()->set_data(colorValue, GL_STREAM_DRAW);

	float _light0_ambient [4] = { 0.2f, 0.2f, 0.2f, 1.0f };
	float _light0_diffuse [4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	float _light0_specular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	float _light0_position[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

	////    glEnable(GL_LIGHTING);
	////    glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0, GL_AMBIENT , _light0_ambient );
	glLightfv(GL_LIGHT0, GL_DIFFUSE , _light0_diffuse );
	glLightfv(GL_LIGHT0, GL_SPECULAR, _light0_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, _light0_position);
	GL_CHECK_ERRORS();


	///////////////////////////////
	// Draw mesh with the raster //
	///////////////////////////////
	if(0)
	{
		glAssert( glEnable(GL_DEPTH_TEST) );
		// Draw depth of the implicit surface (copy is slow as hell though)
		if( 0)
		{
			_render_ctx->pbo_depth()->bind();
			glAssert( glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE) );
			glAssert( glDrawPixels(width, height,GL_DEPTH_COMPONENT,GL_FLOAT,0) );
			glAssert( glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE) );
			_render_ctx->pbo_depth()->unbind();
		}

		glColor3f(1.f, 1.f, 1.f);
		EMesh::Material mat;
		mat.setup_opengl_materials();

		//if(ctx->_rest_pose)
		//	draw_mesh(*g_mesh, *g_animesh->get_vbo_rest_pose(), *g_animesh->get_nbo_rest_pose(), ctx->_textures);
		//else
		//	draw_mesh(*g_mesh, g_mesh->_mesh_gl._vbo, g_mesh->_mesh_gl._normals_bo, ctx->_textures);
		g_primitive_printer.draw(g_arc_circle_vbo);
		g_primitive_printer.draw(g_cube_vbo);
		g_primitive_printer.draw(g_grid_vbo);
		glAssert( glDisable(GL_DEPTH_TEST) );

		//if(_display._ssao) redraw_with_ssao();

	}else
	{
		RenderFuncWireframe rfunc(_cam ,_render_ctx);

		_render_ctx->peeler()->set_render_func(&rfunc);
		_render_ctx->peeler()->set_background(width, height, _render_ctx->pbo_color(), _render_ctx->pbo_depth());

		_render_ctx->peeler()->peel( Cuda_ctrl::_display._transparency);

		glAssert( glEnable(GL_DEPTH_TEST) );
		glAssert( glClear(GL_DEPTH_BUFFER_BIT) );

		// Draw depth of the depth peeling


		_render_ctx->pbo_depth()->bind();
		glAssert( glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE) );
		glAssert( glDrawPixels(width, height,GL_DEPTH_COMPONENT,GL_FLOAT,0) );
		_render_ctx->pbo_depth()->unbind();


		glAssert( glPolygonOffset(1.1f, 4.f) );
		glAssert( glEnable(GL_POLYGON_OFFSET_FILL) );
		rfunc.f();
		glAssert( glDisable(GL_POLYGON_OFFSET_FILL) );



		glAssert( glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE) );
		RenderFuncWireframe::render( _cam,_render_ctx );
		glAssert( glDisable(GL_DEPTH_TEST) );
	}




	return refresh;
}

// Class RenderFuncWireframe ===================================================

void RenderFuncWireframe::draw_transc_objs()
{
	transparent_objects(_cam ,_ctx);
}

// -----------------------------------------------------------------------------

void RenderFuncWireframe::f()
{
	//glPolygonOffset(1.1f, 4.f);
	//glEnable(GL_POLYGON_OFFSET_FILL);
	transparent_objects(_cam ,_ctx);
	//glDisable(GL_POLYGON_OFFSET_FILL);
}

// -----------------------------------------------------------------------------

void RenderFuncWireframe::render(const qglviewer::Camera* cam, const Render_context* ctx)
{
	Tbx::GLEnabledSave save_light(GL_LIGHTING, true, false);
	glAssert( glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) );
	glAssert( glLineWidth(1.f) );
	glAssert( glHint(GL_LINE_SMOOTH_HINT, GL_NICEST) );
	glAssert( glEnable(GL_LINE_SMOOTH) );

	glAssert( glBlendFunc(GL_DST_ALPHA, GL_ONE_MINUS_SRC_ALPHA) );
	plain_objects(cam, ctx, 0.1f, 0.15f, 0.7f, 1.f);
	//plain_objects(cam , ctx, 1.f, 1.f, 1.f, 1.f);
	glAssert( glDisable(GL_LINE_SMOOTH) );
	glAssert( glPolygonMode(GL_FRONT_AND_BACK, GL_FILL) );
}


void drawSampleSet( PaintCanvas* _curCanvas)
{
	using namespace pcm;
	using namespace RenderMode;
	glEnable(GL_MULTISAMPLE);

	//setView();

	if(!_curCanvas->istakeSnapTile()) _curCanvas->drawCornerAxis();

	//tool mode
	if (_curCanvas->single_operate_tool_!=nullptr && _curCanvas->single_operate_tool_->tool_type()!=Tool::EMPTY_TOOL)
	{
		_curCanvas->single_operate_tool_->draw();
		return;
	}

	glPushAttrib(GL_ALL_ATTRIB_BITS);	
	SampleSet& set = (*Global_SampleSet);

	if ( !set.empty() )
	{
		glDisable(GL_MULTISAMPLE);
		for (int i = 0; i <set.size(); i++ )
		{
			if(isShowNoraml)
			{
				glEnable(GL_MULTISAMPLE);
				set[i].drawNormal(Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum));
				glDisable(GL_MULTISAMPLE);
			}
			//if( i == 0 || i > 9)continue;
			LOCK(set[i]);
			switch (which_color_mode_)
			{
			case VERTEX_COLOR:
				glEnable(GL_MULTISAMPLE);
				set[i].draw(ColorMode::VertexColorMode(),Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum));
				glDisable(GL_MULTISAMPLE);
				break;
			case OBJECT_COLOR:
				glEnable(GL_MULTISAMPLE);
				set[i].draw(ColorMode::ObjectColorMode(), 
					Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum));
				glDisable(GL_MULTISAMPLE);
				break;
			case LABEL_COLOR:
				glEnable(GL_MULTISAMPLE);			
				set[i].draw(ColorMode::LabelColorMode(),
					Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum) );


				glDisable(GL_MULTISAMPLE);
				break;
			case WrapBoxColorMode:   //added by huayun
				if(_curCanvas->show_Graph_WrapBox_){
					const pcm::Vec3& bias = Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum);
					auto camera_look_at = _curCanvas->camera()->viewDirection();




					if(!(set[i].is_visible() ) ){
						break;
					}
					//					 std::cout<<"i:"<<i<<"size1:"<<set[i].wrap_box_link_.size()<<std::endl;
					//					 std::cout<<"i:"<<i<<"size:"<<set[i].wrap_box_link_[0].size()<<std::endl;
					std::vector<LinkNode>::iterator bitr = set[i].wrap_box_link_[ 0].begin();
					std::vector<LinkNode>::iterator eitr = set[i].wrap_box_link_[ 0].end();

					glPointSize( Paint_Param::g_point_size);
					glEnable(GL_POINT_SMOOTH);

					Matrix44 adjust_matrix = set[i].matrix_to_scene_coord();
					glEnable(GL_MULTISAMPLE);

					int count = 0;
					pcm::Vec3	sum( 0.0 ,0.0 ,0.0 );

					QFont textFont("Times", (int)Paint_Param::g_point_size, QFont::Bold);

					for( ; bitr!= eitr;++ bitr , count = count+2){
						LinkNode& ln = *bitr;

						//	 renderText(axis_size, 0, 0, "X");

						glBegin(GL_POINTS);

						ColorType color1 ,color2;
						color1 = Color_Utility::span_color_from_hy_table(ln.labelH_);
						color2 = Color_Utility::span_color_from_hy_table(ln.labelL_);

						glNormal3f(-camera_look_at.x,-camera_look_at.y,-camera_look_at.z);
						glColor4f( color1(0)/255.0f,color1(1)/255.0f,color1(2)/255.0f,color1(3) );
						// glNormal3f( normal_(0), normal_(1), normal_(2));
						Vec4	tmp1(ln.pointH_.x(), ln.pointH_.y(), ln.pointH_.z(),1.);
						Vec4	point_to_show1 = adjust_matrix * tmp1;
						Vec4	tmp2(ln.pointL_.x(), ln.pointL_.y(), ln.pointL_.z(),1.);
						Vec4	point_to_show2 = adjust_matrix * tmp2;
						glVertex3f( point_to_show1(0)+bias(0), point_to_show1(1)+bias(1), point_to_show1(2)+bias(2) );
						glColor4f( color2(0)/255.0f,color2(1)/255.0f,color2(2)/255.0f,color2(3) );
						glVertex3f( point_to_show2(0)+bias(0), point_to_show2(1)+bias(1), point_to_show2(2)+bias(2) );
						glEnd();

						glLineWidth( Paint_Param::g_line_size); 
						glBegin(GL_LINES);
						glNormal3f(-camera_look_at.x,-camera_look_at.y,-camera_look_at.z);
						glColor4f( color2(0)/255.0,color2(1)/255.0,color2(2)/255.0,color2(3) );
						glVertex3f( point_to_show1(0)+bias(0), point_to_show1(1)+bias(1), point_to_show1(2)+bias(2) );
						glVertex3f( point_to_show2(0)+bias(0), point_to_show2(1)+bias(1), point_to_show2(2)+bias(2) );
						glEnd();
						glColor4f( 0.0,0.0, 0.0, 1 );  //绘制全黑
						Vec4 textofpoint_to_show1,textofpoint_to_show2;
						textofpoint_to_show1(0) = point_to_show1(0)*1.1;
						textofpoint_to_show1(1) = point_to_show1(1)*1.1;
						textofpoint_to_show1(2) = point_to_show1(2)*1.1;
						textofpoint_to_show2(0) = point_to_show2(0)*1.1;
						textofpoint_to_show2(1) = point_to_show2(1)*1.1;
						textofpoint_to_show2(2) = point_to_show2(2)*1.1;

						_curCanvas->renderText( textofpoint_to_show1(0)+bias(0) , textofpoint_to_show1(1)+bias(1), textofpoint_to_show1(2)+bias(2), QString::number(ln.labelH_), textFont);
						glLineWidth( Paint_Param::g_line_size*0.2); 
						glBegin(GL_LINES);
						glNormal3f(-camera_look_at.x,-camera_look_at.y,-camera_look_at.z);
						glColor4f( color1(0)/255.0,color1(1)/255.0,color1(2)/255.0,color1(3) );
						glVertex3f( point_to_show1(0)+bias(0), point_to_show1(1)+bias(1), point_to_show1(2)+bias(2) );
						glVertex3f( textofpoint_to_show1(0)+bias(0), textofpoint_to_show1(1)+bias(1), textofpoint_to_show1(2)+bias(2) );
						glEnd();

						glColor4f( 0.0,0.0, 0.0, 1 );  //绘制全黑
						_curCanvas->renderText( textofpoint_to_show2(0)+bias(0), textofpoint_to_show2(1)+bias(1), textofpoint_to_show2(2)+bias(2), QString::number(ln.labelL_),textFont );
						glLineWidth( Paint_Param::g_line_size*0.2); 
						glBegin(GL_LINES);
						glNormal3f(-camera_look_at.x,-camera_look_at.y,-camera_look_at.z);
						glColor4f( color2(0)/255.0,color2(1)/255.0,color2(2)/255.0,color2(3) );
						glVertex3f( point_to_show2(0)+bias(0), point_to_show2(1)+bias(1), point_to_show2(2)+bias(2) );
						glVertex3f( textofpoint_to_show2(0)+bias(0), textofpoint_to_show2(1)+bias(1), textofpoint_to_show2(2)+bias(2) );
						glEnd();
						glColor4f( 0.0,0.0, 0.0, 1 );  //绘制全黑

						sum(0) += point_to_show1(0) + bias(0) + point_to_show2(0) + bias(0);
						sum(1) += point_to_show1(1) + bias(1) + point_to_show2(1) + bias(1);
						sum(2) += point_to_show1(0) + bias(2) + point_to_show2(2) + bias(2);

					}
					pcm::Vec3 center = sum /count;
					glColor4f( 0.0,0.0, 0.0, 1 );  //绘制全黑

					_curCanvas->renderText( center.x()*1.1 ,  center.y()*1.1,  center.z()*1.1, QString("frame")+QString::number(i) ,textFont );
					glLineWidth( Paint_Param::g_line_size*0.3); 
					glBegin(GL_LINES);
					glNormal3f(-camera_look_at.x,-camera_look_at.y,-camera_look_at.z);
					glColor4f( 0.0,0.0, 0.0, 1 );  //绘制全黑
					glVertex3f(  center.x(),  center.y(), center.z() );
					glVertex3f(  center.x()*1.1,  center.y()*1.1, center.z()*1.1 );
					glEnd();
					glColor4f( 0.0,0.0, 0.0, 1 );  //绘制全黑

					glDisable(GL_MULTISAMPLE);

					//绘制原的坐标

				}
				break;
			case EdgePointColorMode:   //added by huayun
				glEnable(GL_MULTISAMPLE);
				if(_curCanvas->show_EdgeVertexs_){

					if(!(set[i].is_visible() ) ){
						break;
					}
					set[i].draw(ColorMode::EdgePointColorMode(),
						Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum));
				}
				glDisable(GL_MULTISAMPLE);
				break;
			case SphereMode:
				{   //added by huayun	
					glEnable(GL_MULTISAMPLE);
					//if(show_Graph_WrapBox_){

					if(!(set[i].is_visible() ) ){
						break;
					}
					set[i].draw(ColorMode::SphereMode(),
						Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum));
					//}
					glDisable(GL_MULTISAMPLE);
					break;
				}
			default:
				break;
			}

			switch(which_render_mode)
			{
			case  RenderMode::PointMode:
				{
					glEnable(GL_MULTISAMPLE);
					RenderMode::RenderType rt = RenderMode::PointMode;
					set[i].draw(which_color_mode_,rt ,Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum));
					glDisable(GL_MULTISAMPLE);
					break;
				}
			case  RenderMode::FlatMode:
				{
					glEnable(GL_MULTISAMPLE);
					RenderMode::RenderType rt = RenderMode::FlatMode;
					set[i].draw(which_color_mode_,rt ,Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum));
					glDisable(GL_MULTISAMPLE);
					break;
				}
			case  RenderMode::WireMode:
				{
					glEnable(GL_MULTISAMPLE);
					RenderMode::RenderType rt = RenderMode::WireMode;
					set[i].draw(which_color_mode_, rt ,Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum));
					glDisable(GL_MULTISAMPLE);
					break;
				}
			case RenderMode::FlatWireMode:
				{
					glEnable(GL_MULTISAMPLE);
					RenderMode::RenderType rt = RenderMode::FlatWireMode;
					set[i].draw(which_color_mode_, rt ,Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum));
					glDisable(GL_MULTISAMPLE);
				}
			case  RenderMode::SmoothMode:
				{
					glEnable(GL_MULTISAMPLE);
					RenderMode::RenderType rt = RenderMode::SmoothMode;
					set[i].draw(which_color_mode_,rt ,Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum));
					glDisable(GL_MULTISAMPLE);
					break;
				}
			case  RenderMode::TextureMode:
				{
					glEnable(GL_MULTISAMPLE);
					RenderMode::RenderType rt  = RenderMode::TextureMode;
					set[i].draw(which_color_mode_,rt ,Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum));
					glDisable(GL_MULTISAMPLE);
					break;
				}
			case  RenderMode::SelectMode:
				{
					glEnable(GL_MULTISAMPLE);
					RenderMode::RenderType rt  = RenderMode::SelectMode;
					set[i].draw(which_color_mode_,rt ,Paint_Param::g_step_size * (ScalarType)(i-_curCanvas->centerframeNum));
					glDisable(GL_MULTISAMPLE);
					break;
				}
			default:{}
			}

			UNLOCK(set[i]);
		}
		glEnable(GL_MULTISAMPLE);
	}

	glPopAttrib();
}



// CLASS Render_context_cu =====================================================

Render_context::Render_context(int w, int h):
    _pbo_color(0),
    _pbo_depth(0),
    _plain_phong(false),
    _textures(true),
    _draw_mesh(true),
    _raytrace(false),
    _skeleton(false),
    _rest_pose(false)
{
    _peeler    = new Peeler();

    allocate(w, h);
}

// -----------------------------------------------------------------------------

Render_context::~Render_context()
{

    delete _pbo_color;
    delete _pbo_depth;
    delete _peeler;

}

// -----------------------------------------------------------------------------

void Render_context::reshape(int w, int h)
{
    allocate(w, h);
}

// -----------------------------------------------------------------------------

void Render_context::allocate(int width, int height)
{
	using namespace Tbx;
	int multisampx , multisampy;
	multisampx = multisampy =1;
    _width  = width;
    _height = height;

    if( _pbo_color != 0 ) _pbo_color->set_data(multisampx*width*multisampy*height, 0);
    else                  _pbo_color = new GlBuffer_obj<GLint>(multisampx*width*multisampy*height, GL_PIXEL_UNPACK_BUFFER);

    if(_pbo_depth != 0) _pbo_depth->set_data(width*height, 0);
    else                _pbo_depth = new GlBuffer_obj<GLuint>(width*height, GL_PIXEL_UNPACK_BUFFER);

    _peeler->reinit_depth_peeling(width, height);
}



