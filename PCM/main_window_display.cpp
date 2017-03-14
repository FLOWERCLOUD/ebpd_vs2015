#include "qt_gui/main_window.h"
#include "qt_gui/paint_canvas.h"
#include "control/cuda_ctrl.hpp"
#include <QMessageBox>

////////////////////////////////////////////////////////////
// Implement what's related to display tab of the toolbox //
////////////////////////////////////////////////////////////

//void main_window::on_dSpinB_near_plane_valueChanged(double val)
//{
//	PaintCanvas* wgl = _viewports->active_viewport();
//	wgl->camera()->set_near(val);
//	update_viewports();
//}
//
//void main_window::on_dSpinB_far_plane_valueChanged(double val)
//{
//	PaintCanvas* wgl = _viewports->active_viewport();
//	wgl->camera()->set_far(val);
//	update_viewports();
//}

// COLOR MESH ==================================================================

void main_window::on_gaussian_curvature_toggled(bool checked)
{
	if(checked){
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::GAUSS_CURV);
		update_viewports();
	}
}

void main_window::on_ssd_interpolation_toggled(bool checked)
{
	if(checked){
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::SSD_INTERPOLATION);
		update_viewports();
	}
}

void main_window::on_ssd_weights_toggled(bool checked)
{

	if(checked)
	{
		const std::vector<int>& set = Cuda_ctrl::_skeleton.get_selection_set();
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::SSD_WEIGHTS);
		if(set.size() > 0)
		{
			int id = set[set.size()-1];
			Cuda_ctrl::_anim_mesh->color_ssd_weights(id);
			update_viewports();
		}
	}
}

void main_window::on_color_smoothing_toggled(bool checked)
{
	if(checked){
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::SMOOTHING_WEIGHTS);
		update_viewports();
	}
}

void main_window::on_color_smoothing_conservative_toggled(bool checked)
{
	if(checked){
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::ANIM_SMOOTH_CONSERVATIVE);
		update_viewports();
	}
}

void main_window::on_color_smoothing_laplacian_toggled(bool checked)
{
	if(checked){
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::ANIM_SMOOTH_LAPLACIAN);
		update_viewports();
	}
}
void main_window::on_base_potential_toggled(bool checked)
{
	if(checked){
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::BASE_POTENTIAL);
		update_viewports();
	}
}

void main_window::on_vertices_state_toggled(bool checked)
{
	if( checked )
	{
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::VERTICES_STATE);
		update_viewports();
	}
}

void main_window::on_implicit_gradient_toggled(bool checked)
{
	if(checked){
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::GRAD_POTENTIAL);
		update_viewports();
	}
}
void main_window::on_cluster_toggled(bool checked)
{
	if(checked){
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::CLUSTER);
		update_viewports();
	}
}

void main_window::on_color_nearest_joint_toggled(bool checked)
{
	if(checked){
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::NEAREST_JOINT);
		update_viewports();
	}
}

void main_window::on_color_normals_toggled(bool checked)
{
	if(checked){
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::NORMAL);
		update_viewports();
	}
}

void main_window::on_color_grey_toggled(bool checked)
{
	if(checked){
		//Cuda_ctrl::_anim_mesh->color_uniform(0.8f, 0.8f, 0.8f, 0.99f);
		Cuda_ctrl::_anim_mesh->color_uniform(1.0f, 1.0f, 1.0f, 0.99f);
		update_viewports();
	}
}
void main_window::on_color_free_vertices_toggled(bool checked)
{
	if(checked){
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::FREE_VERTICES);
		update_viewports();
	}
}
void main_window::on_color_edge_stress_toggled(bool checked)
{
	if(checked){
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::EDGE_STRESS);
		update_viewports();
	}
}
void main_window::on_color_area_stress_toggled(bool checked)
{
	if(checked){
		Cuda_ctrl::_anim_mesh->color_type(EAnimesh::AREA_STRESS);
		update_viewports();
	}
}








//void main_window::on_buton_uniform_point_cl_toggled(bool checked)
//{
//	if( checked )
//	{
//		Color cl = Cuda_ctrl::_color.get(Color_ctrl::MESH_POINTS);
//		g_mesh->set_point_color_bo(cl.r, cl.g, cl.b, cl.a);
//		update_viewports();
//	}
//}

//void main_window::on_pButton_do_select_vert_released()
//{
//	if(spinB_vert_id->value() >= 0 &&
//		Cuda_ctrl::is_animesh_loaded() &&
//		spinB_vert_id->value() < Cuda_ctrl::_anim_mesh->get_mesh()->get_nb_vertices() )
//	{
//		Cuda_ctrl::_anim_mesh->select( spinB_vert_id->value() );
//	}
//	else
//	{
//		QMessageBox::information(this,
//			"Error",
//			"wrong vertex identifier\n");
//
//	}
//	update_viewports();
//}

// END COLOR MESH ==============================================================

void main_window::on_display_skeleton_toggled(bool checked)
{
	Cuda_ctrl::_skeleton.switch_display();
	update_viewports();
}

void main_window::on_wireframe_toggled(bool checked)
{
	Cuda_ctrl::_display._wire = checked;
	update_viewports();
}

void main_window::on_display_oriented_bbox_toggled(bool checked)
{
	Cuda_ctrl::_display._oriented_bbox = checked;
	update_viewports();
}

void main_window::on_horizontalSlider_sliderMoved(int position)
{
	Cuda_ctrl::_display.set_transparency_factor( (float)position/100.f );
	update_viewports();
}

void main_window::on_ssd_raio_toggled(bool checked)
{
	if(checked)
	{
		Cuda_ctrl::_anim_mesh->do_ssd_skinning();
		update_viewports();
	}
}

void main_window::on_dual_quaternion_radio_toggled(bool checked)
{
	if(checked)
	{
		Cuda_ctrl::_anim_mesh->do_dual_quat_skinning();
		update_viewports();
	}
}




